"""Fixtures shared by the test suite.

What a test asks for by name lives here; the scaffolding that does not need
pytest (the fakes, the synthetic files, ``load_script``) lives in ``helpers.py``.

The ``real_*`` fixtures load what the experiment actually runs on, once per
session: a test that asserts on ``data/kb/`` or ``data/fsm/`` is checking the
files the thesis reports on rather than a copy of them. They are read-only, and
the synthetic counterparts a loader is tested against stay in the test modules,
under their own names, so the two can never be confused.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from helpers import (
    CONFIG,
    DOCS_DIR,
    EXAMPLES_DIR,
    FSM_DIR,
    KB_DIR,
    PROMPTS_DIR,
    V1_DIR,
    CannedEvalLlm,
    FakeLlm,
    RunCanned,
    make_llm_call_record,
    make_manifest,
    make_turn_record,
    write_llm_calls,
)
from helpers import canned_llm_factory as make_canned_llm
from sim.config import load_models_config
from sim.fsm import FsmSpec, load_fsm
from sim.kb import KnowledgeBase, load_kb
from sim.llm import LlmClient
from sim.prompts import Prompt, load_prompt
from sim.runner import AGENTS, run_experiment
from sim.schemas import Manifest, Scenario, TurnRecord, load_scenarios


@pytest.fixture(scope="session")
def metrics_doc() -> str:
    """The metrics cards of ``docs/metrics.md`` (T-04)."""
    return (DOCS_DIR / "metrics.md").read_text(encoding="utf-8")


@pytest.fixture(scope="session")
def parity_doc() -> str:
    """The parity checklist of ``docs/parity.md`` (T-16)."""
    return (DOCS_DIR / "parity.md").read_text(encoding="utf-8")


@pytest.fixture(scope="session")
def judge_validation_doc() -> str:
    """The T-16 protocol and results in ``docs/judge_validation.md``."""
    return (DOCS_DIR / "judge_validation.md").read_text(encoding="utf-8")


@pytest.fixture(scope="session")
def judge_shared_prompt() -> Prompt:
    """The blindness block both judge calls include (T-04)."""
    return load_prompt("judge_shared", directory=PROMPTS_DIR)


@pytest.fixture(scope="session")
def judge_facts_prompt() -> Prompt:
    """The facts-and-claims rubric (T-04)."""
    return load_prompt("judge_facts", directory=PROMPTS_DIR)


@pytest.fixture(scope="session")
def judge_global_prompt() -> Prompt:
    """The global-judgement rubric (T-04)."""
    return load_prompt("judge_global", directory=PROMPTS_DIR)


@pytest.fixture(scope="session")
def judge_prompt_templates(
    judge_shared_prompt: Prompt,
    judge_facts_prompt: Prompt,
    judge_global_prompt: Prompt,
) -> str:
    """The three judge files concatenated, for scans that cover every call."""
    return "\n".join(
        (
            judge_shared_prompt.template,
            judge_facts_prompt.template,
            judge_global_prompt.template,
        )
    )


@pytest.fixture(scope="session")
def real_kb() -> KnowledgeBase:
    """The knowledge base of ``data/kb/`` (T-01)."""
    return load_kb(KB_DIR)


@pytest.fixture(scope="session")
def real_fsm() -> FsmSpec:
    """The machine of ``data/fsm/`` (T-02)."""
    return load_fsm(FSM_DIR)


@pytest.fixture(scope="session")
def example_scenarios(real_kb: KnowledgeBase, real_fsm: FsmSpec) -> dict[str, Scenario]:
    """The scenarios of ``data/scenarios/examples/``, by ID (T-05)."""
    scenarios = load_scenarios(EXAMPLES_DIR, kb=real_kb, fsm=real_fsm)
    return {scenario.id: scenario for scenario in scenarios}


@pytest.fixture(scope="session")
def v1_scenarios(real_kb: KnowledgeBase, real_fsm: FsmSpec) -> dict[str, Scenario]:
    """The golden dataset of ``data/scenarios/v1/``, by ID (T-06)."""
    scenarios = load_scenarios(V1_DIR, kb=real_kb, fsm=real_fsm)
    return {scenario.id: scenario for scenario in scenarios}


@pytest.fixture(scope="session")
def two_example_scenarios(
    example_scenarios: dict[str, Scenario],
) -> tuple[Scenario, Scenario]:
    """Happy-path and adversarial examples: the 2x2x1 pair of T-14a."""
    return (example_scenarios["happy_path_01"], example_scenarios["adversarial_01"])


@pytest.fixture
def run_dir(tmp_path: Path) -> Path:
    """An experiment directory under pytest's tmp path, never the repo ``runs/``."""
    path = tmp_path / "runs" / "exp"
    path.mkdir(parents=True)
    return path


@pytest.fixture
def two_scenario_dir(
    tmp_path: Path, two_example_scenarios: tuple[Scenario, Scenario]
) -> Path:
    """A directory with only the 2x2x1 pair, so the CLI can load exactly those."""
    directory = tmp_path / "scenarios"
    directory.mkdir()
    (directory / "pair.jsonl").write_text(
        "".join(
            scenario.model_dump_json() + "\n" for scenario in two_example_scenarios
        ),
        encoding="utf-8",
    )
    return directory


@pytest.fixture
def two_turn_fsm_records() -> list[TurnRecord]:
    """Two FSM turns with true ``state_after`` gold for the T-13 labeler."""
    return [
        make_turn_record(
            1,
            user_message="Hi, where is order NL-20260145?",
            agent_reply="Hello. What e-mail was used in the purchase?",
            llm_latency_s=1.0,
            turn_latency_s=1.2,
            state_before="greeting",
            state_after="identification",
            event="request_received",
        ),
        make_turn_record(
            2,
            user_message="customer@example.com. That's all, goodbye.",
            agent_reply="Glad to help. Goodbye.",
            llm_latency_s=1.1,
            turn_latency_s=1.4,
            state_before="identification",
            state_after="closing",
            event="farewell",
        ),
    ]


@pytest.fixture
def canned_llm_factory() -> Callable[..., FakeLlm]:
    """One-turn FakeLlm per job, so a thread pool cannot mix their queues."""
    return make_canned_llm


@pytest.fixture
def run_canned(
    real_kb: KnowledgeBase,
    real_fsm: FsmSpec,
    run_dir: Path,
    two_example_scenarios: tuple[Scenario, Scenario],
) -> RunCanned:
    """Run the experiment on the 2x2 pair with a one-turn fake per job.

    Each brief is cut to its last beat. The script is a mandatory ordered plan
    (T-11), so a one-turn canned dialogue needs a one-beat brief; the runner's
    schedule, files and manifest are what these tests are about, and the last
    beat of both examples is a closing beat that spells nothing out word for
    word.
    """
    config = load_models_config(CONFIG)
    one_beat = [
        scenario.model_copy(update={"script": scenario.script[-1:]})
        for scenario in two_example_scenarios
    ]

    def _run(**kwargs: Any) -> Manifest:
        params: dict[str, Any] = {
            "scenarios": one_beat,
            "agents": AGENTS,
            "reps": 1,
            "parallel": 1,
            "resume": False,
            "run_dir": run_dir,
            "llm_factory": make_canned_llm,
            "kb": real_kb,
            "fsm": real_fsm,
            "config": config,
            "scenarios_dir": EXAMPLES_DIR,
            "exp_id": "exp",
        }
        params.update(kwargs)
        return run_experiment(**params)

    return _run


@pytest.fixture
def canned_eval_llm() -> CannedEvalLlm:
    """Repeating judge and labeler answers, so a second eval does not run dry."""
    return CannedEvalLlm()


@pytest.fixture
def stats_run_dir(tmp_path: Path) -> Path:
    """A run directory with a canned manifest and ``llm_calls.jsonl`` (T-15)."""
    run_dir = tmp_path / "runs" / "exp"
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text(
        make_manifest().model_dump_json() + "\n", encoding="utf-8"
    )
    write_llm_calls(
        run_dir,
        [
            make_llm_call_record(
                caller="baseline", prompt_tokens=100, output_tokens=10, latency_s=1.0
            ),
            make_llm_call_record(
                caller="baseline",
                prompt_tokens=200,
                output_tokens=30,
                latency_s=3.0,
                prompt_hash="bb",
            ),
            make_llm_call_record(
                caller="fsm", prompt_tokens=80, output_tokens=10, latency_s=4.0
            ),
            make_llm_call_record(
                caller="fsm",
                prompt_tokens=80,
                output_tokens=10,
                latency_s=0.5,
                cached=True,
                prompt_hash="cc",
            ),
            make_llm_call_record(
                caller="classifier",
                role="simulator",
                prompt_tokens=40,
                latency_s=0.5,
            ),
            make_llm_call_record(
                caller="simulated_user",
                role="simulator",
                prompt_tokens=60,
                latency_s=1.0,
            ),
            make_llm_call_record(
                caller="judge_facts",
                role="judge",
                prompt_tokens=400,
                latency_s=10.0,
            ),
            make_llm_call_record(
                caller="judge_facts",
                role="judge",
                prompt_tokens=400,
                latency_s=10.0,
                prompt_hash="dd",
            ),
            make_llm_call_record(
                caller="judge_global",
                role="judge",
                prompt_tokens=350,
                latency_s=20.0,
            ),
            make_llm_call_record(
                caller="judge_global",
                role="judge",
                prompt_tokens=350,
                latency_s=20.0,
                prompt_hash="ee",
            ),
            make_llm_call_record(
                caller="stage_labeler",
                role="simulator",
                prompt_tokens=200,
                latency_s=5.0,
            ),
            make_llm_call_record(
                caller="stage_labeler",
                role="simulator",
                prompt_tokens=200,
                latency_s=5.0,
                prompt_hash="ff",
            ),
        ],
    )
    return run_dir


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
