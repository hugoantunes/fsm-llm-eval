"""Fixtures shared by the test suite.

What a test asks for by name lives here; the scaffolding that does not need
pytest (the fakes, the synthetic files, ``load_script``) lives in ``helpers.py``.

The ``real_*`` fixtures load what the experiment actually runs on, once per
session: a test that asserts on ``data/kb/`` or ``data/fsm/`` is checking the
files the thesis reports on rather than a copy of them. They are read-only, and
the synthetic counterparts a loader is tested against stay in the test modules,
under their own names, so the two can never be confused.
"""

from pathlib import Path

import pytest

from helpers import CONFIG, EXAMPLES_DIR, FSM_DIR, KB_DIR
from sim.config import load_models_config
from sim.fsm import FsmSpec, load_fsm
from sim.kb import KnowledgeBase, load_kb
from sim.llm import LlmClient
from sim.schemas import Scenario, load_scenarios


@pytest.fixture(scope="session")
def real_kb() -> KnowledgeBase:
    """The knowledge base of ``data/kb/`` (T-01)."""
    return load_kb(KB_DIR)


@pytest.fixture(scope="session")
def real_fsm(real_kb: KnowledgeBase) -> FsmSpec:
    """The machine of ``data/fsm/``, cross-checked against the real KB (T-02)."""
    return load_fsm(FSM_DIR, kb=real_kb)


@pytest.fixture(scope="session")
def example_scenarios(real_kb: KnowledgeBase, real_fsm: FsmSpec) -> dict[str, Scenario]:
    """The scenarios of ``data/scenarios/examples/``, by ID (T-05)."""
    scenarios = load_scenarios(EXAMPLES_DIR, kb=real_kb, fsm=real_fsm)
    return {scenario.id: scenario for scenario in scenarios}


@pytest.fixture(scope="session")
def prompt_budget() -> int:
    """The real config's prompt-token ceiling, above which T-07 refuses a call."""
    return load_models_config(CONFIG).max_prompt_tokens


@pytest.fixture
def real_client(tmp_path: Path) -> LlmClient:
    """A client on the real config and the real Ollama server.

    For ``integration``-marked tests only: everything else takes a
    :class:`~helpers.FakeLlm`, or an :class:`~sim.llm.LlmClient` over the fake
    transport of ``test_llm.py``. The cache and the log go under ``tmp_path``, so
    a test never reads an answer another run cached.
    """
    return LlmClient(
        load_models_config(CONFIG),
        cache_dir=tmp_path / "cache",
        log_path=tmp_path / "llm_calls.jsonl",
    )
