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


#: A synthetic ``configs/models.yaml``: the shape of the real one with names no
#: machine has to hold. The config loader and the LLM client both build on it.
MINIMAL_MODELS_YAML = """\
ollama:
  version: "0.33.3"
  env:
    OLLAMA_NUM_PARALLEL: 2
num_ctx: 8192
max_prompt_fraction: 0.8
client:
  timeout_s: 300
  retries: 2
  backoff_s: 2
models:
  agent:
    name: "agent-model"
    temperature: 0.7
    seed: 42
  simulator:
    name: "small-model"
  judge:
    name: "judge-model"
    temperature: 0
    seed: 42
"""


def write_models_config(directory: Path, text: str = MINIMAL_MODELS_YAML) -> Path:
    """Write a synthetic model config into ``directory`` and return its path."""
    path = directory / "models.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def load_script(relative_path: str) -> ModuleType:
    """Import a script that lives outside any package, by repo-relative path."""
    file = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(file.stem, file)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
