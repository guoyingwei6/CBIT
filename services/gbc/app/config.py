import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


def _origins(value: str) -> tuple[str, ...]:
    return tuple(origin.strip() for origin in value.split(",") if origin.strip())


@dataclass(frozen=True)
class Settings:
    asset_dir: Path
    storage_backend: str
    local_upload_dir: Path
    max_upload_bytes: int
    max_samples: int
    batch_size: int
    presign_expiry_seconds: int
    cors_origins: tuple[str, ...]
    gateway_secret: str
    r2_endpoint_url: str
    r2_access_key_id: str
    r2_secret_access_key: str
    r2_bucket: str


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    service_dir = Path(__file__).resolve().parents[1]
    backend = os.getenv("STORAGE_BACKEND", "local").lower()
    if backend not in {"local", "r2"}:
        raise ValueError("STORAGE_BACKEND must be either 'local' or 'r2'")

    return Settings(
        asset_dir=Path(os.getenv("GBC_ASSET_DIR", service_dir / "assets")),
        storage_backend=backend,
        local_upload_dir=Path(
            os.getenv("LOCAL_UPLOAD_DIR", "/tmp/cbit-gbc-uploads")
        ),
        max_upload_bytes=int(os.getenv("MAX_UPLOAD_BYTES", 500 * 1024 * 1024)),
        max_samples=int(os.getenv("MAX_SAMPLES", 1000)),
        batch_size=int(os.getenv("GBC_BATCH_SIZE", 2048)),
        presign_expiry_seconds=int(os.getenv("PRESIGN_EXPIRY_SECONDS", 900)),
        cors_origins=_origins(
            os.getenv(
                "CORS_ORIGINS",
                "http://localhost:8080,http://127.0.0.1:8080",
            )
        ),
        gateway_secret=os.getenv("GATEWAY_SECRET", ""),
        r2_endpoint_url=os.getenv("R2_ENDPOINT_URL", ""),
        r2_access_key_id=os.getenv("R2_ACCESS_KEY_ID", ""),
        r2_secret_access_key=os.getenv("R2_SECRET_ACCESS_KEY", ""),
        r2_bucket=os.getenv("R2_BUCKET", ""),
    )
