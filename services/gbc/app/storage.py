import re
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterable, BinaryIO
from uuid import uuid4

from botocore.client import Config
from botocore.exceptions import ClientError

from .config import Settings


OBJECT_KEY = re.compile(r"^uploads/[0-9a-f]{32}\.txt$")


@dataclass(frozen=True)
class UploadTarget:
    object_key: str
    upload_url: str
    headers: dict[str, str]
    expires_in: int


class LocalStorage:
    def __init__(self, root: Path, max_upload_bytes: int):
        self.root = root
        self.max_upload_bytes = max_upload_bytes
        self.root.mkdir(parents=True, exist_ok=True)

    def create_upload(self, size: int, content_type: str) -> UploadTarget:
        upload_id = uuid4().hex
        return UploadTarget(
            object_key=f"uploads/{upload_id}.txt",
            upload_url=f"/api/uploads/local/{upload_id}",
            headers={"Content-Type": content_type},
            expires_in=900,
        )

    def _path(self, object_key: str) -> Path:
        if not OBJECT_KEY.fullmatch(object_key):
            raise ValueError("Invalid upload object key")
        return self.root / object_key.removeprefix("uploads/")

    async def save(
        self,
        object_key: str,
        chunks: AsyncIterable[bytes],
        declared_size: int,
    ) -> None:
        path = self._path(object_key)
        written = 0
        try:
            with path.open("wb") as output:
                async for chunk in chunks:
                    written += len(chunk)
                    if written > self.max_upload_bytes:
                        raise ValueError("Upload exceeds the configured size limit")
                    output.write(chunk)
            if written != declared_size:
                raise ValueError("Uploaded file size does not match Content-Length")
        except Exception:
            path.unlink(missing_ok=True)
            raise

    def open(self, object_key: str) -> tuple[BinaryIO, int]:
        path = self._path(object_key)
        return path.open("rb"), path.stat().st_size

    def delete(self, object_key: str) -> None:
        self._path(object_key).unlink(missing_ok=True)


class R2Storage:
    def __init__(self, settings: Settings):
        required = {
            "R2_ENDPOINT_URL": settings.r2_endpoint_url,
            "R2_ACCESS_KEY_ID": settings.r2_access_key_id,
            "R2_SECRET_ACCESS_KEY": settings.r2_secret_access_key,
            "R2_BUCKET": settings.r2_bucket,
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            raise ValueError(f"Missing R2 settings: {', '.join(missing)}")

        import boto3

        self.bucket = settings.r2_bucket
        self.expiry = settings.presign_expiry_seconds
        self.client = boto3.client(
            "s3",
            endpoint_url=settings.r2_endpoint_url,
            aws_access_key_id=settings.r2_access_key_id,
            aws_secret_access_key=settings.r2_secret_access_key,
            region_name="auto",
            config=Config(
                signature_version="s3v4",
                s3={"addressing_style": "path"},
                retries={"max_attempts": 3, "mode": "standard"},
            ),
        )

    def create_upload(self, size: int, content_type: str) -> UploadTarget:
        upload_id = uuid4().hex
        object_key = f"uploads/{upload_id}.txt"
        headers = {
            "Content-Type": content_type,
            "x-amz-meta-declared-size": str(size),
        }
        url = self.client.generate_presigned_url(
            "put_object",
            Params={
                "Bucket": self.bucket,
                "Key": object_key,
                "ContentType": content_type,
                "Metadata": {"declared-size": str(size)},
            },
            ExpiresIn=self.expiry,
        )
        return UploadTarget(object_key, url, headers, self.expiry)

    def open(self, object_key: str) -> tuple[BinaryIO, int]:
        if not OBJECT_KEY.fullmatch(object_key):
            raise ValueError("Invalid upload object key")
        try:
            response = self.client.get_object(Bucket=self.bucket, Key=object_key)
        except ClientError as error:
            code = str(error.response.get("Error", {}).get("Code", ""))
            if code in {"404", "NoSuchKey", "NotFound"}:
                raise FileNotFoundError(object_key) from error
            raise
        size = int(response["ContentLength"])
        declared_size = int(response.get("Metadata", {}).get("declared-size", -1))
        if declared_size != size:
            response["Body"].close()
            raise ValueError("Uploaded object size does not match its signed metadata")
        return response["Body"], size

    def delete(self, object_key: str) -> None:
        if OBJECT_KEY.fullmatch(object_key):
            self.client.delete_object(Bucket=self.bucket, Key=object_key)


def create_storage(settings: Settings) -> LocalStorage | R2Storage:
    if settings.storage_backend == "r2":
        return R2Storage(settings)
    return LocalStorage(settings.local_upload_dir, settings.max_upload_bytes)
