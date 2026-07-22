# [image-based-campaign] Image storage: save bytes to a backend, return a reference.
# Backends sit behind one save() contract, chosen by IMAGE_STORAGE_BACKEND:
#   "local"      — disk under data/, served via the API's /media mount
#   "cloudinary" — CDN-backed, permanent public URLs (works from a deployed UI)
# Adding another store is a class with the same save() signature plus a branch in
# get_image_store(); nothing else in the codebase changes.
from __future__ import annotations

import hashlib
import os
import time
import uuid
from dataclasses import dataclass

import httpx

from config import settings

_MIME = {"jpg": "image/jpeg", "png": "image/png", "webp": "image/webp"}


@dataclass
class ImageRef:
    """Pointer to a stored image — never the raw bytes."""
    path: str  # durable locator (filesystem path for the local backend)
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


# [image-based-campaign] ── Cloudinary backend ────────────────────────────────
_CLOUDINARY_UPLOAD = "https://api.cloudinary.com/v1_1/{cloud}/image/upload"


def _cloudinary_signature(params: dict[str, str], secret: str) -> str:
    """Cloudinary signed upload: sha1 of alphabetically-sorted params + api_secret."""
    to_sign = "&".join(f"{k}={params[k]}" for k in sorted(params))
    return hashlib.sha1(f"{to_sign}{secret}".encode()).hexdigest()


class CloudinaryImageStore:
    """Uploads to Cloudinary; returns a permanent public CDN URL.

    Note: images are publicly readable by anyone holding the URL — the tradeoff
    for having no signed-URL expiry to manage.
    """

    def __init__(self) -> None:
        missing = [n for n, v in (
            ("CLOUDINARY_CLOUD_NAME", settings.CLOUDINARY_CLOUD_NAME),
            ("CLOUDINARY_API_KEY", settings.CLOUDINARY_API_KEY),
            ("CLOUDINARY_API_SECRET", settings.CLOUDINARY_API_SECRET),
        ) if not v]
        if missing:
            raise ValueError(f"Cloudinary backend needs: {', '.join(missing)}.")

    def save(self, data: bytes, subdir: str, brief_id: str, name: str | None = None) -> ImageRef:
        fname = _filename(name, data)
        stem = fname.rsplit(".", 1)[0]
        folder = f"{settings.CLOUDINARY_FOLDER}/{subdir}/{brief_id}".strip("/")

        signed = {"folder": folder, "public_id": stem, "timestamp": str(int(time.time()))}
        form = {
            **signed,
            "api_key": settings.CLOUDINARY_API_KEY,
            "signature": _cloudinary_signature(signed, settings.CLOUDINARY_API_SECRET),
        }
        resp = httpx.post(
            _CLOUDINARY_UPLOAD.format(cloud=settings.CLOUDINARY_CLOUD_NAME),
            data=form,
            files={"file": (fname, data, _content_type(fname))},
            timeout=60,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"Cloudinary upload failed ({resp.status_code}): {resp.text[:300]}")

        body = resp.json()
        # public_id is the durable locator (re-fetch / delete later); secure_url renders.
        return ImageRef(path=body["public_id"], url=body["secure_url"])


#: Values that all mean "no cloud storage — keep images on local disk".
_LOCAL_ALIASES = {"", "none", "null", "off", "false", "local", "disk"}


def get_image_store():
    """[image-based-campaign] Backend switch — add one branch per new store.
    A cloud backend only needs to implement save(bytes, ...) -> ImageRef.

    Parsing is forgiving: the value is case-insensitive, and anything meaning
    "no cloud" (NONE, off, empty, ...) falls back to local disk rather than
    erroring — local storage always works, so it is the safe default."""
    backend = (settings.IMAGE_STORAGE_BACKEND or "").strip().lower()
    if backend in _LOCAL_ALIASES:
        return LocalImageStore()
    if backend == "cloudinary":
        return CloudinaryImageStore()
    raise ValueError(
        f"Unknown IMAGE_STORAGE_BACKEND: {settings.IMAGE_STORAGE_BACKEND!r}. "
        "Use 'local' (or 'none') for disk, or 'cloudinary'."
    )
