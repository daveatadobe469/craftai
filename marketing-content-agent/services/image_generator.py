# [image-based-campaign] Image generation (text → image), provider-agnostic.
# Cloudflare Workers AI backend for the capstone; a Gemini/Imagen backend can be
# added later by implementing generate() and switching IMAGE_PROVIDER.
from __future__ import annotations

import base64
import binascii

import httpx

from config import settings

_CF_BASE = "https://api.cloudflare.com/client/v4/accounts"
_GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/interactions"
_TIMEOUT_S = 60.0


def _cf_endpoint() -> str:
    return f"{_CF_BASE}/{settings.CLOUDFLARE_ACCOUNT_ID}/ai/run/{settings.IMAGE_MODEL}"


def _cf_error_detail(resp: httpx.Response) -> str:
    """Pull Cloudflare's own error message/code out of its JSON error envelope."""
    try:
        payload = resp.json()
    except ValueError:
        return resp.text[:300] or f"HTTP {resp.status_code} with an empty body"
    errors = payload.get("errors") or []
    if errors:
        return "; ".join(f"{e.get('message', e)} (code {e.get('code')})" for e in errors)
    return str(payload)[:300]


def _decode_cf_response(resp: httpx.Response) -> bytes:
    """Cloudflare returns either base64-in-JSON (flux) or raw image bytes (SDXL)."""
    ctype = resp.headers.get("content-type", "")
    if ctype.startswith("image/"):
        return resp.content

    try:
        payload = resp.json()
    except ValueError as exc:
        raise RuntimeError(
            f"Cloudflare returned a non-image, non-JSON response: {resp.text[:300]!r}"
        ) from exc

    b64 = (payload.get("result") or {}).get("image", "")
    if not b64:
        raise RuntimeError(f"Cloudflare image response had no image: {_cf_error_detail(resp)}")
    try:
        return base64.b64decode(b64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise RuntimeError(f"Cloudflare returned malformed base64 image data: {exc}") from exc


def _generate_cloudflare(prompt: str) -> bytes:
    if not settings.CLOUDFLARE_ACCOUNT_ID or not settings.CLOUDFLARE_API_TOKEN:
        raise ValueError("Cloudflare image gen needs CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN.")

    headers = {"Authorization": f"Bearer {settings.CLOUDFLARE_API_TOKEN}"}
    # stable-diffusion-xl-lightning's field is "num_steps" (1-20, default 20).
    body = {"prompt": prompt, "num_steps": settings.CLOUDFLARE_IMAGE_STEPS}
    try:
        resp = httpx.post(_cf_endpoint(), headers=headers, json=body, timeout=_TIMEOUT_S)
    except httpx.TimeoutException as exc:
        raise RuntimeError(f"Cloudflare image request timed out after {_TIMEOUT_S:.0f}s.") from exc
    except httpx.RequestError as exc:
        raise RuntimeError(f"Cloudflare image request failed (network error): {exc}") from exc

    if resp.status_code == 401 or resp.status_code == 403:
        raise RuntimeError(
            f"Cloudflare rejected the request as unauthorized ({resp.status_code}) — "
            f"check CLOUDFLARE_ACCOUNT_ID / CLOUDFLARE_API_TOKEN: {_cf_error_detail(resp)}"
        )
    if resp.status_code == 429:
        raise RuntimeError(f"Cloudflare rate-limited the request (429): {_cf_error_detail(resp)}")
    if resp.status_code >= 500:
        raise RuntimeError(
            f"Cloudflare service error ({resp.status_code}) — try again later: {_cf_error_detail(resp)}"
        )
    if resp.status_code != 200:
        raise RuntimeError(f"Cloudflare image generation failed ({resp.status_code}): {_cf_error_detail(resp)}")

    return _decode_cf_response(resp)


# [image-based-campaign] ── Gemini (Interactions API) ──────────────────────────
def _gemini_body(prompt: str) -> dict:
    return {
        "model": settings.GEMINI_IMAGE_MODEL,
        "input": [{"type": "text", "text": prompt}],
        "response_format": {
            "type": "image",
            "mime_type": "image/jpeg",
            "aspect_ratio": settings.IMAGE_ASPECT_RATIO,
            "image_size": settings.IMAGE_SIZE,
        },
    }


def _decode_gemini_response(payload: dict) -> bytes:
    """Prefer output_image; fall back to the first image block in steps."""
    b64 = ((payload.get("output_image") or {}).get("data")) or ""
    if not b64:
        for step in payload.get("steps") or []:
            for block in step.get("content") or []:
                if block.get("type") == "image" and block.get("data"):
                    b64 = block["data"]
                    break
            if b64:
                break
    if not b64:
        raise RuntimeError(f"Gemini image response had no image: {str(payload)[:300]}")
    return base64.b64decode(b64)


def _generate_gemini(prompt: str) -> bytes:
    if not settings.GEMINI_API_KEY:
        raise ValueError("Gemini image gen needs GEMINI_API_KEY.")

    # API key goes in a header, never a URL query param.
    headers = {
        "x-goog-api-key": settings.GEMINI_API_KEY,
        "Content-Type": "application/json",
    }
    resp = httpx.post(_GEMINI_URL, headers=headers, json=_gemini_body(prompt), timeout=_TIMEOUT_S)
    resp.raise_for_status()
    return _decode_gemini_response(resp.json())


def generate_image(prompt: str) -> bytes:
    """Return raw image bytes for a text prompt. Blocking — call via executor."""
    # [image-based-campaign] Provider dispatch — one line per new backend.
    if settings.IMAGE_PROVIDER == "cloudflare":
        return _generate_cloudflare(prompt)
    if settings.IMAGE_PROVIDER == "gemini":
        return _generate_gemini(prompt)
    raise ValueError(f"Unknown IMAGE_PROVIDER: {settings.IMAGE_PROVIDER!r}.")
