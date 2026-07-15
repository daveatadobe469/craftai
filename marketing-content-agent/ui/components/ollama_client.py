from __future__ import annotations

from typing import Any

import httpx


def _parse_ollama_tags(data: dict[str, Any], base_url: str) -> dict[str, Any]:
    models: list[dict[str, Any]] = []
    for m in data.get("models", []):
        name = m.get("name", "")
        size_bytes = m.get("size", 0)
        details = m.get("details", {}) or {}
        families = details.get("families") or []
        family = details.get("family") or (families[0] if families else "unknown")
        models.append(
            {
                "name": name,
                "size_gb": round(size_bytes / 1e9, 2),
                "family": str(family),
            }
        )
    return {
        "available": True,
        "base_url": base_url.rstrip("/"),
        "models": models,
        "error": None,
    }


def _candidate_ollama_urls(base_url: str) -> list[str]:
    base = base_url.rstrip("/")
    urls = [base]
    if "localhost" in base:
        urls.append(base.replace("localhost", "127.0.0.1"))
    elif "127.0.0.1" in base:
        urls.append(base.replace("127.0.0.1", "localhost"))
    return list(dict.fromkeys(urls))


def fetch_ollama_models_direct(ollama_base_url: str) -> dict[str, Any]:
    """Query Ollama /api/tags from the Streamlit host (bypasses CRAFTAI API)."""
    last_error = ""
    for url in _candidate_ollama_urls(ollama_base_url):
        try:
            resp = httpx.get(f"{url}/api/tags", timeout=8.0)
            resp.raise_for_status()
            return _parse_ollama_tags(resp.json(), url)
        except httpx.ConnectError:
            last_error = f"Cannot connect to Ollama at {url}. Is `ollama serve` running?"
        except httpx.HTTPStatusError as exc:
            last_error = f"Ollama API error at {url}: {exc.response.status_code}"
        except Exception as exc:
            last_error = str(exc)

    return {
        "available": False,
        "base_url": ollama_base_url.rstrip("/"),
        "models": [],
        "error": last_error or f"Cannot connect to Ollama at {ollama_base_url}",
    }


def fetch_ollama_models(api_base: str, ollama_base_url: str) -> dict[str, Any]:
    """
    Fetch installed Ollama models via CRAFTAI API, falling back to a direct
    local query when the backend is not running.
    """
    api_base = api_base.rstrip("/")
    try:
        resp = httpx.get(
            f"{api_base}/config/ollama/models",
            params={"base_url": ollama_base_url},
            timeout=10.0,
        )
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, dict):
            data.setdefault("via_api", True)
        return data
    except httpx.ConnectError:
        direct = fetch_ollama_models_direct(ollama_base_url)
        if direct.get("available"):
            direct["via_api"] = False
            direct["api_warning"] = (
                f"CRAFTAI API is not running at `{api_base}`. "
                "Start it with `python main.py` to save settings and run briefs. "
                "Showing Ollama models via direct local connection."
            )
            return direct
        return {
            "available": False,
            "base_url": ollama_base_url.rstrip("/"),
            "models": [],
            "error": (
                f"CRAFTAI API not reachable at `{api_base}` (run `python main.py`). "
                f"Also could not reach Ollama directly: {direct.get('error', '')}"
            ),
        }
    except httpx.HTTPStatusError as exc:
        return {
            "available": False,
            "base_url": ollama_base_url.rstrip("/"),
            "models": [],
            "error": f"CRAFTAI API error {exc.response.status_code} at {api_base}",
        }
    except Exception as exc:
        direct = fetch_ollama_models_direct(ollama_base_url)
        if direct.get("available"):
            direct["via_api"] = False
            direct["api_warning"] = (
                f"CRAFTAI API request failed ({exc}). "
                "Showing Ollama models via direct local connection."
            )
            return direct
        return {
            "available": False,
            "base_url": ollama_base_url.rstrip("/"),
            "models": [],
            "error": str(exc),
        }


def api_is_reachable(api_base: str, timeout: float = 4.0) -> bool:
    api_base = api_base.rstrip("/")
    try:
        resp = httpx.get(f"{api_base.replace('/api/v1', '')}/health", timeout=timeout)
        if resp.status_code == 200:
            return True
    except Exception:
        pass
    try:
        resp = httpx.get(f"{api_base}/config", timeout=timeout)
        return resp.status_code == 200
    except Exception:
        return False
