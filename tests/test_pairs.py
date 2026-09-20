from __future__ import annotations

import pytest

from dafx26_demo.pairs import DEFAULT_PAIR, ObservedPairError, get_observed_pairs, validate_pair


def test_observed_pairs_match_paper_table() -> None:
    pairs = get_observed_pairs()
    assert ("Chopin", "etude") in pairs
    assert ("Chopin", "nocturne") not in pairs
    assert ("Bach", "fugue") in pairs
    assert len(pairs) == 19


def test_default_pair_is_observed() -> None:
    assert DEFAULT_PAIR in get_observed_pairs()
    assert DEFAULT_PAIR != ("Chopin", "nocturne")


def test_validate_pair_rejects_unseen_combinations() -> None:
    validate_pair("Chopin", "etude")
    with pytest.raises(ObservedPairError, match="nocturne"):
        validate_pair("Chopin", "nocturne")
    with pytest.raises(ObservedPairError):
        validate_pair("Unknown", "sonata")
