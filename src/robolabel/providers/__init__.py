"""VLM providers. Each concrete provider is one self-registering module.

Importing this package registers the built-in providers (mock, gemini, openai,
and — if its heavy deps import — qwen). Use :func:`build_provider` to construct
one by name.
"""

from __future__ import annotations

# Always-available providers self-register on import.
from . import gemini, mock, openai  # noqa: E402,F401
from .base import (
    MissingCredentialError,
    ProviderResponse,
    TwoStageResult,
    VLMProvider,
    available_providers,
    build_provider,
    extract_json,
    register_provider,
    try_extract_json,
)

# Qwen registers only if the module imports (the heavy deps are deferred to
# construction, so this import is cheap and safe even without the extra).
try:  # pragma: no cover
    from . import qwen  # noqa: F401
except Exception:  # noqa: BLE001
    pass

__all__ = [
    "VLMProvider",
    "ProviderResponse",
    "TwoStageResult",
    "build_provider",
    "register_provider",
    "available_providers",
    "extract_json",
    "try_extract_json",
    "MissingCredentialError",
]
