"""Tests for docs/parity.md (T-16)."""

from helpers import FROZEN_V1_HASH


def test_parity_doc_lists_equals_diffs_and_both_token_accounts(parity_doc: str) -> None:
    assert "Held equal" in parity_doc
    assert "Allowed to differ" in parity_doc
    assert "census of all 20 ok dialogues" in parity_doc
    assert "30/76 eligible agent responses" in parity_doc
    assert FROZEN_V1_HASH in parity_doc
    assert "agent_shared.md" in parity_doc
    assert "gemma4:12b" in parity_doc
    assert "Prompt tokens per agent turn" in parity_doc
    assert "classifier" in parity_doc
