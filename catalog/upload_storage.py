"""
Where raw uploads live: local ``uploads/`` by default, or **S3-compatible** storage (MinIO) when configured.

When ``MINIO_*`` points at the **shared** MinIO from ``docker-compose.yml``, app uploads use your bucket;
Milvus uses its own bucket/prefix on the same server for segment storage.
"""

from __future__ import annotations

import re
import uuid
from pathlib import Path

from config import (
    MINIO_ACCESS_KEY,
    MINIO_BUCKET,
    MINIO_TLS_VERIFY,
    MINIO_ENDPOINT,
    MINIO_PREFIX,
    MINIO_REGION,
    MINIO_SECRET_KEY,
    MINIO_SKIP_BUCKET_CREATE,
    MINIO_USE_SSL,
    PROJECT_ROOT,
    UPLOADS_DIR,
)

_S3_RE = re.compile(r"^s3://([^/]+)/(.+)$")


def _reraise_minio_tls_help(exc: BaseException) -> None:
    """If ``exc`` is TLS/cert verification against MinIO, raise a short fix hint."""
    import ssl

    try:
        from botocore.exceptions import SSLError as BotoSSLError
    except ImportError:
        BotoSSLError = ()  # type: ignore[misc, assignment]

    seen: set[int] = set()
    chain: list[BaseException] = []
    cur: BaseException | None = exc
    for _ in range(24):
        if cur is None or id(cur) in seen:
            break
        seen.add(id(cur))
        chain.append(cur)
        cur = cur.__cause__ or cur.__context__

    for ex in chain:
        if isinstance(ex, BotoSSLError) or isinstance(ex, ssl.SSLError):
            raise RuntimeError(
                "MinIO HTTPS certificate verification failed (self-signed or private CA). "
                "Set MINIO_CA_BUNDLE=/path/to/public.crt (or your CA PEM), "
                "or MINIO_CERT_CHECK=false, then restart. "
                "For plain HTTP MinIO use MINIO_USE_SSL=false."
            ) from exc
        low = str(ex).lower()
        if "certificate verify failed" in low or "ssl: certificate_verify_failed" in low:
            raise RuntimeError(
                "MinIO HTTPS certificate verification failed (self-signed or private CA). "
                "Set MINIO_CA_BUNDLE=/path/to/public.crt (or your CA PEM), "
                "or MINIO_CERT_CHECK=false, then restart. "
                "For plain HTTP MinIO use MINIO_USE_SSL=false."
            ) from exc


def use_minio_uploads() -> bool:
    """True when MinIO/S3 uploads are enabled (bucket + endpoint + keys)."""
    return bool(
        MINIO_BUCKET.strip()
        and MINIO_ENDPOINT.strip()
        and MINIO_ACCESS_KEY
        and MINIO_SECRET_KEY
    )


def _safe_segment(name: str) -> str:
    return name.replace("..", "_").replace("/", "_").strip() or "upload.bin"


def _local_path(doc_id: uuid.UUID, original_filename: str) -> Path:
    safe = _safe_segment(original_filename)
    return UPLOADS_DIR / f"{doc_id}_{safe}"


def store_upload(doc_id: uuid.UUID, original_filename: str, content: bytes) -> str:
    """
    Persist ``content`` and return ``storage_path`` stored in PostgreSQL.

    Local: relative path under project root, e.g. ``uploads/<uuid>_<name>``.
    MinIO: ``s3://<bucket>/<key>``.
    """
    if use_minio_uploads():
        return _minio_put(doc_id, original_filename, content)
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    dest = _local_path(doc_id, original_filename)
    dest.write_bytes(content)
    return str(dest.relative_to(PROJECT_ROOT))


def read_upload_bytes(storage_path: str) -> bytes:
    """Load raw file bytes for text extraction."""
    if storage_path.startswith("s3://"):
        return _minio_get(storage_path)
    path = PROJECT_ROOT / storage_path
    return path.read_bytes()


def delete_upload(storage_path: str) -> None:
    if storage_path.startswith("s3://"):
        _minio_delete(storage_path)
        return
    path = PROJECT_ROOT / storage_path
    if path.is_file():
        try:
            path.unlink()
        except OSError:
            pass


def _s3_client():
    try:
        import boto3
        from botocore.config import Config as BotoConfig
    except ImportError as e:
        raise ImportError("MinIO uploads require: pip install boto3") from e

    scheme = "https" if MINIO_USE_SSL else "http"
    endpoint = MINIO_ENDPOINT.strip()
    if not endpoint.startswith("http"):
        endpoint_url = f"{scheme}://{endpoint}"
    else:
        endpoint_url = endpoint
    # Path-style URLs (http://host:9000/bucket/key) — virtual-hosted style often returns 400 on MinIO.
    return boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=MINIO_ACCESS_KEY,
        aws_secret_access_key=MINIO_SECRET_KEY,
        region_name=MINIO_REGION,
        config=BotoConfig(
            signature_version="s3v4",
            s3={"addressing_style": "path"},
        ),
        verify=MINIO_TLS_VERIFY,
    )


def _object_key(doc_id: uuid.UUID, original_filename: str) -> str:
    safe = _safe_segment(original_filename)
    prefix = MINIO_PREFIX.strip().strip("/")
    if prefix:
        return f"{prefix}/{doc_id}/{safe}"
    return f"{doc_id}/{safe}"


def _bucket_exists(client, bucket: str) -> bool:
    """Prefer listing: ``head_bucket`` can 400 on some MinIO/proxy setups while the bucket is usable."""
    from botocore.exceptions import ClientError

    try:
        resp = client.list_buckets()
        names = {b.get("Name") for b in resp.get("Buckets", []) if b.get("Name")}
        return bucket in names
    except ClientError:
        try:
            client.head_bucket(Bucket=bucket)
            return True
        except ClientError:
            return False


def _ensure_bucket(client, bucket: str) -> None:
    """Create bucket if missing. MinIO/AWS differ on ``CreateBucketConfiguration``; try compatible variants."""
    from botocore.exceptions import ClientError

    if _bucket_exists(client, bucket):
        return

    region = (MINIO_REGION or "us-east-1").strip()
    last: Exception | None = None

    # 1) Plain create (correct for default MinIO single-site)
    try:
        client.create_bucket(Bucket=bucket)
        return
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "")
        if code in ("BucketAlreadyOwnedByYou", "BucketAlreadyExists"):
            return
        last = e

    # 2) Non–us-east-1 AWS-style location
    if region != "us-east-1":
        try:
            client.create_bucket(
                Bucket=bucket,
                CreateBucketConfiguration={"LocationConstraint": region},
            )
            return
        except ClientError as e:
            last = e

    # 3) Some MinIO builds expect an explicit location even when region is us-east-1
    try:
        client.create_bucket(
            Bucket=bucket,
            CreateBucketConfiguration={"LocationConstraint": "us-east-1"},
        )
        return
    except ClientError as e:
        last = e

    raise RuntimeError(
        f"Could not create or access MinIO bucket {bucket!r} (last error: {last}). "
        "Create the bucket in the MinIO console (or `mc mb`), or set MINIO_REGION to match the server, "
        "then retry."
    ) from last


def _minio_put(doc_id: uuid.UUID, original_filename: str, content: bytes) -> str:
    try:
        client = _s3_client()
        key = _object_key(doc_id, original_filename)
        bucket = MINIO_BUCKET.strip()
        if not MINIO_SKIP_BUCKET_CREATE:
            _ensure_bucket(client, bucket)
        client.put_object(Bucket=bucket, Key=key, Body=content)
        return f"s3://{bucket}/{key}"
    except Exception as e:
        _reraise_minio_tls_help(e)
        raise


def _minio_get(storage_path: str) -> bytes:
    m = _S3_RE.match(storage_path.strip())
    if not m:
        raise ValueError(f"Invalid s3 storage_path: {storage_path!r}")
    bucket, key = m.group(1), m.group(2)
    try:
        client = _s3_client()
        resp = client.get_object(Bucket=bucket, Key=key)
        return resp["Body"].read()
    except Exception as e:
        _reraise_minio_tls_help(e)
        raise


def _minio_delete(storage_path: str) -> None:
    m = _S3_RE.match(storage_path.strip())
    if not m:
        return
    bucket, key = m.group(1), m.group(2)
    client = _s3_client()
    try:
        client.delete_object(Bucket=bucket, Key=key)
    except Exception:
        pass
