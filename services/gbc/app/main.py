import hmac
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .config import get_settings
from .engine import GbcEngine, GbcInputError
from .reference import load_reference
from .schemas import AnalyzeRequest, PresignRequest
from .storage import LocalStorage, create_storage


logger = logging.getLogger(__name__)
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.engine = GbcEngine(
        load_reference(settings.asset_dir),
        batch_size=settings.batch_size,
        max_samples=settings.max_samples,
    )
    app.state.storage = create_storage(settings)
    yield


app = FastAPI(title="CBIT GBC Service", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_origins),
    allow_credentials=False,
    allow_methods=["GET", "POST", "PUT", "OPTIONS"],
    allow_headers=["Content-Type", "X-CBIT-Gateway"],
)


@app.middleware("http")
async def require_gateway(request: Request, call_next):
    if settings.gateway_secret and request.url.path != "/healthz":
        supplied = request.headers.get("X-CBIT-Gateway", "")
        if not hmac.compare_digest(supplied, settings.gateway_secret):
            return JSONResponse(status_code=403, content={"detail": "Forbidden"})
    return await call_next(request)


@app.exception_handler(GbcInputError)
async def handle_input_error(_request: Request, error: GbcInputError):
    return JSONResponse(status_code=422, content={"detail": str(error)})


@app.get("/healthz")
def health() -> dict[str, object]:
    return {
        "status": "ok",
        "referenceSnps": len(app.state.engine.reference.keys),
        "breeds": len(app.state.engine.reference.breeds),
    }


@app.post("/api/uploads/presign")
def presign_upload(payload: PresignRequest, request: Request) -> dict[str, object]:
    if not payload.fileName.lower().endswith(".txt"):
        raise HTTPException(status_code=422, detail="Only .txt files are supported")
    if payload.size > settings.max_upload_bytes:
        raise HTTPException(status_code=413, detail="File exceeds the 500MB limit")
    target = request.app.state.storage.create_upload(
        payload.size, payload.contentType or "text/plain"
    )
    return {
        "code": 1,
        "objectKey": target.object_key,
        "uploadUrl": target.upload_url,
        "headers": target.headers,
        "expiresIn": target.expires_in,
    }


@app.put("/api/uploads/local/{upload_id}")
async def local_upload(upload_id: str, request: Request) -> dict[str, object]:
    storage = request.app.state.storage
    if not isinstance(storage, LocalStorage):
        raise HTTPException(status_code=404, detail="Not found")
    content_length = int(request.headers.get("content-length", -1))
    if content_length < 1 or content_length > settings.max_upload_bytes:
        raise HTTPException(status_code=413, detail="Invalid upload size")
    try:
        await storage.save(
            f"uploads/{upload_id}.txt", request.stream(), content_length
        )
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    return {"code": 1}


@app.post("/api/mongo/gbc-estimator/")
def analyze(payload: AnalyzeRequest, request: Request) -> dict[str, object]:
    storage = request.app.state.storage
    stream = None
    should_delete = False
    try:
        try:
            stream, size = storage.open(payload.objectKey)
        except FileNotFoundError as error:
            raise HTTPException(
                status_code=404, detail="Uploaded file not found"
            ) from error
        should_delete = True
        if size > settings.max_upload_bytes:
            raise HTTPException(status_code=413, detail="File exceeds the 500MB limit")
        result = request.app.state.engine.estimate(stream, payload.threshold)
        return {
            "msg": "Task completed successfully",
            "code": 1,
            "columns": result.columns,
            "data": result.data,
            "csv": result.csv,
            "metrics": {
                "matchedSnps": result.matched_snps,
                "samples": result.samples,
                "elapsedSeconds": round(result.elapsed_seconds, 3),
            },
        }
    finally:
        if stream is not None:
            stream.close()
        if should_delete:
            try:
                storage.delete(payload.objectKey)
            except Exception:
                logger.exception("Failed to delete upload object %s", payload.objectKey)
