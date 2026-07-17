"""Portable path helpers for committed artifacts (no private absolute paths)."""

from __future__ import annotations

import re
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

_URL_RE = re.compile(r"^(https?|hf)://", re.IGNORECASE)
_HF_HUB_RE = re.compile(
    r"(?:^|[\\/])(?:\.cache[\\/])?huggingface[\\/](hub[\\/].+)$",
    re.IGNORECASE,
)
_UNIX_HOME_RE = re.compile(r"^(?:/Users/[^/]+|/home/[^/]+|/root)(/.*)?$", re.IGNORECASE)
_WIN_USER_RE = re.compile(r"^[A-Za-z]:[\\/]Users[\\/][^\\/]+([\\/].*)?$", re.IGNORECASE)


def default_repo_root() -> Path:
    """Repository root containing ``pyproject.toml`` / ``src/openarm_pipeline``."""
    return Path(__file__).resolve().parents[2]


def _as_posix_style(value: str) -> str:
    return value.replace("\\", "/")


def sanitize_path_string(
    value: str,
    *,
    repo_root: str | Path | None = None,
) -> str:
    """Rewrite private absolute filesystem paths to portable forms.

    Leaves public URLs, Hugging Face repo IDs, and already-relative paths unchanged.
    """
    if not isinstance(value, str) or not value:
        return value

    raw = value.strip()
    if _URL_RE.match(raw):
        return raw

    # Dataset / model IDs like "lerobot/svla_so100_pickplace" (not filesystem paths).
    if (
        "/" in raw
        and not raw.startswith(("/", ".", "~"))
        and not re.match(r"^[A-Za-z]:[\\/]", raw)
        and "://" not in raw
        and "\\" not in raw
    ):
        return raw

    root = Path(repo_root) if repo_root is not None else default_repo_root()
    root = root.resolve()

    # Hugging Face hub cache → <HF_HOME>/hub/...
    hf_match = _HF_HUB_RE.search(_as_posix_style(raw))
    if hf_match:
        return "<HF_HOME>/" + _as_posix_style(hf_match.group(1))

    # Absolute path under the repository → repo-relative.
    try:
        p = Path(raw)
        if p.is_absolute():
            resolved = p.resolve()
            try:
                rel = resolved.relative_to(root)
                return rel.as_posix()
            except ValueError:
                pass
    except (OSError, RuntimeError, ValueError):
        pass

    posix = _as_posix_style(raw)

    # Strip macOS / Linux home prefixes; keep trailing relative segment when useful.
    m = _UNIX_HOME_RE.match(posix)
    if m:
        rest = (m.group(1) or "").lstrip("/")
        return _portable_after_home(rest)

    m = _WIN_USER_RE.match(raw) or _WIN_USER_RE.match(posix)
    if m:
        rest = (m.group(1) or "").replace("\\", "/").lstrip("/")
        return _portable_after_home(rest)

    # Already relative / logical.
    if not posix.startswith("/") and not re.match(r"^[A-Za-z]:/", posix):
        return PurePosixPath(posix).as_posix() if "\\" in raw else raw

    # Unknown absolute path with no reproducibility value.
    return "<REDACTED_LOCAL_PATH>"


def _portable_after_home(rest: str) -> str:
    if not rest:
        return "<REDACTED_LOCAL_PATH>"
    posix = _as_posix_style(rest)
    hf_match = _HF_HUB_RE.search("/" + posix)
    if hf_match:
        return "<HF_HOME>/" + _as_posix_style(hf_match.group(1))
    for marker in (
        "configs/",
        "artifacts/",
        "data/",
        "tasks/",
        "docs/",
        "scripts/",
        "tests/",
        "src/",
    ):
        idx = posix.find(marker)
        if idx >= 0:
            return posix[idx:]
    # Keep last two segments as a weak portable hint.
    parts = [p for p in PurePosixPath(posix).parts if p not in (".",)]
    if len(parts) >= 2:
        return "/".join(parts[-2:])
    return "<REDACTED_LOCAL_PATH>"


def sanitize_for_artifact(obj: Any, *, repo_root: str | Path | None = None) -> Any:
    """Recursively sanitize path-like strings in JSON-serializable structures."""
    if isinstance(obj, Path):
        return sanitize_path_string(str(obj), repo_root=repo_root)
    if isinstance(obj, PureWindowsPath):
        return sanitize_path_string(str(obj), repo_root=repo_root)
    if isinstance(obj, str):
        # Only touch strings that look like filesystem paths.
        if (
            obj.startswith(("/", "~"))
            or re.match(r"^[A-Za-z]:[\\/]", obj)
            or "\\" in obj
            or "/Users/" in obj
            or "/home/" in obj
            or obj.startswith("file://")
            or ".cache/huggingface" in obj.replace("\\", "/")
        ):
            return sanitize_path_string(obj, repo_root=repo_root)
        return obj
    if isinstance(obj, dict):
        return {k: sanitize_for_artifact(v, repo_root=repo_root) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_for_artifact(v, repo_root=repo_root) for v in obj]
    if isinstance(obj, tuple):
        return [sanitize_for_artifact(v, repo_root=repo_root) for v in obj]
    return obj
