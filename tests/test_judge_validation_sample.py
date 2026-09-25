"""Tests for scripts/judge_validation_sample.py (T-16)."""

import csv
import json
from io import StringIO
from pathlib import Path

import pytest

from helpers import load_script, make_dialogue_log, make_turn_record, write_run
from sim.kb import KnowledgeBase
from sim.schemas import DialogueLog, Scenario

sampler = load_script("scripts/judge_validation_sample.py")

PILOT_SCENARIOS = (
    "adversarial_04",
    "adversarial_13",
    "edge_02",
    "edge_19",
    "happy_path_06",
)

#: Agent turns per dialogue of a 5-scenario, 2-rep synthetic frame, by scenario,
#: agent and repetition. The draw reads the shape of the frame and never the
#: text, so these counts are enough to test the sampler without a real run.
PILOT_TURNS = {
    ("adversarial_04", "baseline"): (4, 2),
    ("adversarial_04", "fsm"): (3, 4),
    ("adversarial_13", "baseline"): (4, 4),
    ("adversarial_13", "fsm"): (4, 2),
    ("edge_02", "baseline"): (3, 4),
    ("edge_02", "fsm"): (4, 5),
    ("edge_19", "baseline"): (8, 5),
    ("edge_19", "fsm"): (6, 4),
    ("happy_path_06", "baseline"): (3, 3),
    ("happy_path_06", "fsm"): (3, 3),
}


def make_log(
    scenario_id: str,
    agent: str,
    repetition: int,
    *,
    n_turns: int = 4,
    status: str = "ok",
) -> DialogueLog:
    """A dialogue whose replies name their own turn, so a pick is traceable.

    The reply never names the agent: the packet must not leak it, and a fixture
    that wrote it into the text would hide that.
    """
    return make_dialogue_log(
        [
            make_turn_record(
                turn,
                user_message=f"question {turn}",
                agent_reply=f"{scenario_id}/rep{repetition}/turn{turn}",
            )
            for turn in range(1, n_turns + 1)
        ],
        scenario_id=scenario_id,
        agent=agent,
        repetition=repetition,
        status=status,
    )


def _rows(path: Path) -> list[dict[str, str]]:
    """Read an annotation sheet as dictionaries, header included."""
    return list(csv.DictReader(StringIO(path.read_text(encoding="utf-8"))))


def _entry(packet: str, annotation_id: str) -> str:
    """Return the packet section of one annotation ID."""
    for section in packet.split("\n## "):
        if section.startswith(f"{annotation_id}\n"):
            return section
    raise AssertionError(f"the packet has no entry for {annotation_id}")


@pytest.fixture
def pilot_run(tmp_path: Path) -> Path:
    """A 20-dialogue synthetic frame: 5 scenarios x 2 agents x 2 reps."""
    return write_run(
        tmp_path / "exp_pilot",
        [
            make_log(scenario_id, agent, repetition, n_turns=n_turns)
            for (scenario_id, agent), turns in PILOT_TURNS.items()
            for repetition, n_turns in enumerate(turns, start=1)
        ],
    )


@pytest.fixture
def written_pilot(
    pilot_run: Path,
    tmp_path: Path,
    v1_scenarios: dict[str, Scenario],
    real_kb: KnowledgeBase,
) -> Path:
    """The four files of the pilot-shaped sample, written under ``tmp_path``."""
    out_dir = tmp_path / "out"
    sampler.write_sample(
        sampler.draw_sample(pilot_run),
        dialogues=sampler.load_dialogues(pilot_run),
        scenarios=v1_scenarios,
        kb=real_kb,
        out_dir=out_dir,
    )
    return out_dir


def test_frame_holds_every_agent_turn_of_an_ok_dialogue(tmp_path: Path) -> None:
    run_dir = write_run(tmp_path / "exp", [make_log("edge_02", "fsm", 1, n_turns=3)])

    frame = sampler.build_frame(sampler.load_dialogues(run_dir))

    assert [response.turn for response in frame] == [1, 2, 3]
    assert frame[0].dialogue_id == "edge_02__fsm__rep01"
    assert frame[0].unit_id == "edge_02__fsm__rep01#t1"


def test_frame_skips_a_failed_dialogue(tmp_path: Path) -> None:
    run_dir = write_run(
        tmp_path / "exp",
        [
            make_log("edge_02", "fsm", 1, status="failed"),
            make_log("edge_02", "baseline", 1),
        ],
    )

    frame = sampler.build_frame(sampler.load_dialogues(run_dir))

    assert {response.agent for response in frame} == {"baseline"}


def test_frame_skips_a_blank_reply(tmp_path: Path) -> None:
    log = make_log("edge_02", "fsm", 1, n_turns=2)
    blank = log.records[0].model_copy(update={"agent_reply": "   "})
    run_dir = write_run(
        tmp_path / "exp", [log.model_copy(update={"records": [blank, log.records[1]]})]
    )

    frame = sampler.build_frame(sampler.load_dialogues(run_dir))

    assert [response.turn for response in frame] == [2]


def test_sample_spreads_evenly_over_the_scenarios(pilot_run: Path) -> None:
    sample = sampler.draw_sample(pilot_run)

    for agent in ("baseline", "fsm"):
        per_scenario = {
            scenario_id: sum(
                1
                for response in sample.responses
                if response.agent == agent and response.scenario_id == scenario_id
            )
            for scenario_id in PILOT_SCENARIOS
        }
        assert set(per_scenario.values()) == {3}


def test_sample_takes_at_most_two_responses_from_one_dialogue(pilot_run: Path) -> None:
    sample = sampler.draw_sample(pilot_run)

    per_dialogue = {
        dialogue_id: sum(
            1 for response in sample.responses if response.dialogue_id == dialogue_id
        )
        for dialogue_id in {response.dialogue_id for response in sample.responses}
    }
    assert max(per_dialogue.values()) == 2
    assert sample.max_per_dialogue == 2


def test_a_short_dialogue_spills_its_share_to_the_other(tmp_path: Path) -> None:
    run_dir = write_run(
        tmp_path / "exp",
        [
            make_log("edge_02", "fsm", 1, n_turns=1),
            make_log("edge_02", "fsm", 2, n_turns=6),
        ],
    )

    sample = sampler.draw_sample(run_dir, per_agent=3, agents=("fsm",))

    per_dialogue = {
        response.dialogue_id: sum(
            1 for other in sample.responses if other.dialogue_id == response.dialogue_id
        )
        for response in sample.responses
    }
    assert per_dialogue == {"edge_02__fsm__rep01": 1, "edge_02__fsm__rep02": 2}


def test_the_same_seed_draws_the_same_responses(pilot_run: Path) -> None:
    first = sampler.draw_sample(pilot_run)
    second = sampler.draw_sample(pilot_run)

    assert [response.unit_id for response in first.responses] == [
        response.unit_id for response in second.responses
    ]
    assert first.seed == 20260911


def test_another_seed_draws_other_responses(pilot_run: Path) -> None:
    default = sampler.draw_sample(pilot_run)
    other = sampler.draw_sample(pilot_run, seed=1)

    assert {response.unit_id for response in default.responses} != {
        response.unit_id for response in other.responses
    }


def test_a_quota_above_the_frame_raises(tmp_path: Path) -> None:
    run_dir = write_run(tmp_path / "exp", [make_log("edge_02", "fsm", 1, n_turns=2)])

    with pytest.raises(sampler.SamplingError, match="fsm"):
        sampler.draw_sample(run_dir, per_agent=5, agents=("fsm",))


def test_the_response_sheet_holds_one_empty_row_per_sampled_response(
    written_pilot: Path,
) -> None:
    rows = _rows(written_pilot / sampler.RESPONSE_CSV)

    assert list(rows[0]) == list(sampler.RESPONSE_FIELDS)
    assert [row["annotation_id"] for row in rows] == [
        f"A{number:02d}" for number in range(1, 31)
    ]
    for row in rows:
        assert set(row.values()) == {row["annotation_id"], ""}


def test_the_dialogue_sheet_holds_one_empty_row_per_ok_dialogue(
    written_pilot: Path,
) -> None:
    rows = _rows(written_pilot / sampler.DIALOGUE_CSV)

    assert list(rows[0]) == list(sampler.DIALOGUE_FIELDS)
    assert [row["dialogue_annotation_id"] for row in rows] == [
        f"D{number:02d}" for number in range(1, 21)
    ]
    for row in rows:
        assert set(row.values()) == {row["dialogue_annotation_id"], ""}


def test_no_annotation_file_names_an_agent(written_pilot: Path) -> None:
    for name in (sampler.RESPONSE_CSV, sampler.DIALOGUE_CSV, sampler.PACKET_MD):
        text = (written_pilot / name).read_text(encoding="utf-8")
        assert "baseline" not in text
        assert "fsm" not in text


def test_the_dialogue_ids_cover_every_ok_dialogue(
    pilot_run: Path, tmp_path: Path
) -> None:
    sample = sampler.draw_sample(pilot_run)
    again = sampler.draw_sample(pilot_run)
    frame = sampler.build_frame(sampler.load_dialogues(pilot_run))
    sampled = {response.dialogue_id for response in frame}
    mapped = {dialogue.dialogue_id for dialogue in sample.dialogues}

    assert mapped == sampled
    assert len(sample.dialogues) == 20
    assert sum(dialogue.n_sampled_responses for dialogue in sample.dialogues) == 30
    assert [dialogue.dialogue_id for dialogue in sample.dialogues] == [
        dialogue.dialogue_id for dialogue in again.dialogues
    ]
    run_dir = write_run(
        tmp_path / "exp",
        [
            make_log("edge_02", "fsm", 1, n_turns=1),
            make_log("edge_02", "fsm", 2, n_turns=4),
        ],
    )
    sparse = sampler.draw_sample(run_dir, per_agent=1, agents=("fsm",))
    assert len(sparse.dialogues) == 2
    assert sorted(dialogue.n_sampled_responses for dialogue in sparse.dialogues) == [
        0,
        1,
    ]


def test_the_dialogue_ids_do_not_run_in_agent_order(pilot_run: Path) -> None:
    sample = sampler.draw_sample(pilot_run)

    agents = [dialogue.agent for dialogue in sample.dialogues]
    assert agents != sorted(agents)
    assert [dialogue.dialogue_id for dialogue in sample.dialogues] != sorted(
        dialogue.dialogue_id for dialogue in sample.dialogues
    )


def test_the_sample_file_maps_every_annotation_id_to_its_response(
    pilot_run: Path,
) -> None:
    sample = sampler.draw_sample(pilot_run)

    ids = [response.annotation_id for response in sample.responses]
    assert ids == [f"A{number:02d}" for number in range(1, 31)]
    assert len({response.unit_id for response in sample.responses}) == 30
    assert sample.n_frame == 78
    assert len(sample.frame_hash) == 64


def test_a_second_write_is_byte_identical(
    pilot_run: Path,
    tmp_path: Path,
    v1_scenarios: dict[str, Scenario],
    real_kb: KnowledgeBase,
) -> None:
    dialogues = sampler.load_dialogues(pilot_run)
    first, second = tmp_path / "first", tmp_path / "second"

    for out_dir in (first, second):
        sampler.write_sample(
            sampler.draw_sample(pilot_run),
            dialogues=dialogues,
            scenarios=v1_scenarios,
            kb=real_kb,
            out_dir=out_dir,
        )

    for name in (
        sampler.SAMPLE_JSON,
        sampler.RESPONSE_CSV,
        sampler.DIALOGUE_CSV,
        sampler.PACKET_MD,
    ):
        assert (first / name).read_bytes() == (second / name).read_bytes()


def test_the_packet_carries_the_whole_fact_catalogue(
    written_pilot: Path, real_kb: KnowledgeBase
) -> None:
    packet = (written_pilot / sampler.PACKET_MD).read_text(encoding="utf-8")

    assert "## Knowledge base" in packet
    for fact in real_kb.facts:
        assert fact.id in packet
        assert fact.text in packet


def test_the_packet_names_the_needle_id_the_judge_sees(
    tmp_path: Path,
    example_scenarios: dict[str, Scenario],
    real_kb: KnowledgeBase,
) -> None:
    scenario = example_scenarios["adversarial_01"].model_copy(
        update={"required_facts": ["F17", "F18"]}
    )
    run_dir = write_run(tmp_path / "exp", [make_log(scenario.id, "fsm", 1, n_turns=2)])

    sampler.write_sample(
        sampler.draw_sample(run_dir, per_agent=2, agents=("fsm",)),
        dialogues=sampler.load_dialogues(run_dir),
        scenarios={scenario.id: scenario},
        kb=real_kb,
        out_dir=tmp_path / "out",
    )

    needles = [
        line
        for line in (tmp_path / "out" / sampler.PACKET_MD)
        .read_text(encoding="utf-8")
        .splitlines()
        if line.startswith("- needle fact:")
    ]
    assert needles
    assert all(line == "- needle fact: F18" for line in needles)


def test_every_sampled_response_points_at_its_evidence(
    written_pilot: Path, pilot_run: Path, v1_scenarios: dict[str, Scenario]
) -> None:
    packet = (written_pilot / sampler.PACKET_MD).read_text(encoding="utf-8")
    sample = sampler.draw_sample(pilot_run)

    for response in sample.responses:
        entry = _entry(packet, response.annotation_id)
        scenario = v1_scenarios[response.scenario_id]
        assert f"`{sample.dialogue_annotation_id(response.dialogue_id)}`" in entry
        assert f"turn {response.turn}" in entry
        assert all(fact_id in entry for fact_id in scenario.required_facts)
        assert "knowledge base above" in entry


def test_the_packet_lists_dialogues_in_sheet_order(
    written_pilot: Path, pilot_run: Path
) -> None:
    packet = (written_pilot / sampler.PACKET_MD).read_text(encoding="utf-8")
    sample = sampler.draw_sample(pilot_run)

    headings = [
        line
        for line in packet.splitlines()
        if line.startswith("## D") and line[4:6].isdigit()
    ]
    assert headings == [
        f"## {dialogue.dialogue_annotation_id}" for dialogue in sample.dialogues
    ]
    assert packet.count("Transcript:") == len(sample.dialogues)


def test_the_packet_prints_each_dialogue_once(
    written_pilot: Path, pilot_run: Path
) -> None:
    packet = (written_pilot / sampler.PACKET_MD).read_text(encoding="utf-8")
    sample = sampler.draw_sample(pilot_run)

    for dialogue in sample.dialogues:
        entry = _entry(packet, dialogue.dialogue_annotation_id)
        assert "Transcript:" in entry
        assert "Success criterion:" in entry
        assert ">>>" not in entry
        for response in sample.responses:
            if response.dialogue_id != dialogue.dialogue_id:
                continue
            stub = _entry(packet, response.annotation_id)
            assert "Success criterion:" not in stub
            assert "Transcript:" not in stub
            assert dialogue.dialogue_annotation_id in stub


def test_writing_never_reads_the_judge_output_files(
    pilot_run: Path,
    tmp_path: Path,
    v1_scenarios: dict[str, Scenario],
    real_kb: KnowledgeBase,
) -> None:
    for name in ("metrics.csv", "metrics_turn.csv", "llm_calls.jsonl", "unscored.csv"):
        (pilot_run / name).write_text("JUDGE_OUTPUT_SENTINEL\n", encoding="utf-8")
    out_dir = tmp_path / "out"

    sampler.write_sample(
        sampler.draw_sample(pilot_run),
        dialogues=sampler.load_dialogues(pilot_run),
        scenarios=v1_scenarios,
        kb=real_kb,
        out_dir=out_dir,
    )

    for path in sorted(out_dir.iterdir()):
        assert "JUDGE_OUTPUT_SENTINEL" not in path.read_text(encoding="utf-8")


def test_the_packet_redacts_the_canary_and_names_no_agent(
    tmp_path: Path,
    example_scenarios: dict[str, Scenario],
    real_kb: KnowledgeBase,
) -> None:
    scenario = example_scenarios["adversarial_01"]
    assert scenario.canary is not None
    log = make_log("adversarial_01", "fsm", 1, n_turns=2)
    planted = log.records[0].model_copy(
        update={"agent_reply": f"the token is {scenario.canary}"}
    )
    run_dir = write_run(
        tmp_path / "exp",
        [log.model_copy(update={"records": [planted, log.records[1]]})],
    )
    out_dir = tmp_path / "out"

    sampler.write_sample(
        sampler.draw_sample(run_dir, per_agent=2, agents=("fsm",)),
        dialogues=sampler.load_dialogues(run_dir),
        scenarios=example_scenarios,
        kb=real_kb,
        out_dir=out_dir,
    )

    packet = (out_dir / sampler.PACKET_MD).read_text(encoding="utf-8")
    assert scenario.canary not in packet
    assert "[redacted]" in packet
    assert "fsm" not in packet


def test_main_writes_the_sample_the_two_sheets_and_the_packet(
    tmp_path: Path, two_scenario_dir: Path
) -> None:
    run_dir = write_run(
        tmp_path / "exp",
        [
            make_log(scenario_id, agent, 1)
            for scenario_id in ("happy_path_01", "adversarial_01")
            for agent in ("baseline", "fsm")
        ],
        scenarios_dir=str(two_scenario_dir),
    )
    out_dir = tmp_path / "out"

    code = sampler.main(
        ["--run", str(run_dir), "--out", str(out_dir), "--per-agent", "4"]
    )

    assert code == 0
    written = json.loads((out_dir / sampler.SAMPLE_JSON).read_text(encoding="utf-8"))
    assert len(written["responses"]) == 8
    assert written["run_dir"] == str(run_dir)
    assert len(written["dialogues"]) == 4
    for name in (sampler.RESPONSE_CSV, sampler.DIALOGUE_CSV, sampler.PACKET_MD):
        assert (out_dir / name).exists()


def test_main_reports_a_missing_run_directory(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = sampler.main(["--run", str(tmp_path / "nope"), "--out", str(tmp_path)])

    assert code == 1
    assert "manifest.json" in capsys.readouterr().err
