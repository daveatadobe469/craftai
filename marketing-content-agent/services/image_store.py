# [image-based-campaign] Image storage: save bytes to a backend, return a reference.
# Two backends behind one save() contract, chosen by IMAGE_STORAGE_BACKEND:
#   "local"    — disk under data/, served via the API's /media mount (dev default)
#   "firebase" — Firebase Storage bucket, returns a signed URL for the UI
from __future__ import annotations

import os
import uuid
from dataclasses import dataclass

from config import settings

_MIME = {"jpg": "image/jpeg", "png": "image/png", "webp": "image/webp"}


@dataclass
class ImageRef:
    """Pointer to a stored image — never the raw bytes."""
    path: str  # durable locator: filesystem path (local) or gs:// URI (firebase)
    url: str   # HTTP URL the UI can render


def _sniff_ext(data: bytes) -> str:
    # [image-based-campaign] Pick a file extension from magic bytes so the stored
    # name (and served Content-Type) matches the real format. Cloudflare's
    # flux-1-schnell returns JPEG, not PNG.
    if data[:3] == b"\xff\xd8\xff":
        return "jpg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    return "png"


def _filename(name: str | None, data: bytes) -> str:
    # [image-based-campaign] Keep the caller's name (uploads); else uuid + sniffed ext.
    return name or f"{uuid.uuid4().hex}.{_sniff_ext(data)}"


def _content_type(fname: str) -> str:
    return _MIME.get(fname.rsplit(".", 1)[-1].lower(), "application/octet-stream")


class LocalImageStore:
    """Writes images under data/<subdir>/<brief_id>/ and serves them via /media."""

    def save(self, data: bytes, subdir: str, brief_id: str, name: str | None = None) -> ImageRef:
        fname = _filename(name, data)
        rel_dir = os.path.join(subdir, brief_id)
        abs_dir = os.path.join(settings.data_root, rel_dir)
        os.makedirs(abs_dir, exist_ok=True)

        abs_path = os.path.join(abs_dir, fname)
        with open(abs_path, "wb") as f:
            f.write(data)

        url = f"{settings.MEDIA_BASE_URL.rstrip('/')}/media/{subdir}/{brief_id}/{fname}"
        return ImageRef(path=abs_path, url=url)


# [image-based-campaign] ── Firebase Storage backend ──────────────────────────
def _firebase_bucket():
    """Initialise the Firebase app once (idempotent) and return the bucket."""
    if not settings.FIREBASE_STORAGE_BUCKET:
        raise ValueError("Firebase backend needs FIREBASE_STORAGE_BUCKET.")
    try:
        import firebase_admin
        from firebase_admin import credentials, storage
    except ImportError as exc:
        raise ValueError(
            "Firebase backend needs the firebase-admin package: pip install firebase-admin"
        ) from exc

    if not firebase_admin._apps:  # not yet initialised in this process
        if not settings.FIREBASE_CREDENTIALS_PATH:
            raise ValueError("Firebase backend needs FIREBASE_CREDENTIALS_PATH (service-account JSON).")
        cred = credentials.Certificate(settings.FIREBASE_CREDENTIALS_PATH)
        firebase_admin.initialize_app(cred, {"storageBucket": settings.FIREBASE_STORAGE_BUCKET})
    return storage.bucket(settings.FIREBASE_STORAGE_BUCKET)


class FirebaseImageStore:
    """Uploads to a Firebase Storage bucket; returns a signed URL the UI can render."""

    def __init__(self) -> None:
        self._bucket = _firebase_bucket()

    def save(self, data: bytes, subdir: str, brief_id: str, name: str | None = None) -> ImageRef:
        from datetime import timedelta

        fname = _filename(name, data)
        blob_path = f"{subdir}/{brief_id}/{fname}"
        blob = self._bucket.blob(blob_path)
        blob.upload_from_string(data, content_type=_content_type(fname))
        # Signed URL so the bucket stays private; TTL is configurable.
        url = blob.generate_signed_url(
            expiration=timedelta(days=settings.FIREBASE_SIGNED_URL_TTL_DAYS),
            method="GET",
        )
        return ImageRef(path=f"gs://{self._bucket.name}/{blob_path}", url=url)


def get_image_store():
    # [image-based-campaign] Backend switch — one branch per store.
    backend = settings.IMAGE_STORAGE_BACKEND
    if backend == "local":
        return LocalImageStore()
    if backend == "firebase":
        return FirebaseImageStore()
    raise ValueError(f"Unknown IMAGE_STORAGE_BACKEND: {backend!r}. Use 'local' or 'firebase'.")
