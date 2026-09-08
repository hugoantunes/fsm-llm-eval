"""Shared test helpers."""

import importlib.util
from pathlib import Path
from types import ModuleType

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Brands, people and companies that must never appear in a fictional domain.
#: The knowledge base and the scenarios are both checked against this list, so
#: it is written once (T-01, T-05).
FORBIDDEN_NAMES = (
    "amazon",
    "shopify",
    "mercado livre",
    "magalu",
    "americanas",
    "nike",
    "adidas",
    "zara",
    "apple",
    "google",
    "microsoft",
    "netflix",
    "correios",
    "fedex",
    "dhl",
    "ups",
    "visa",
    "mastercard",
    "paypal",
    "pix",
)


def load_script(relative_path: str) -> ModuleType:
    """Import a script that lives outside any package, by repo-relative path."""
    file = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(file.stem, file)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
