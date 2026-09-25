"""Tests for docs/parity.md (T-16)."""

from helpers import FROZEN_V1_HASH


def test_parity_doc_quotes_the_frozen_v1_hash(parity_doc: str) -> None:
    assert FROZEN_V1_HASH in parity_doc
