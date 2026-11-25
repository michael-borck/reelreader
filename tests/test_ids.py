from __future__ import annotations

from reelreader.ids import LENGTH, is_id, new_id


def test_shape_and_alphabet():
    identifier = new_id()
    assert len(identifier) == LENGTH
    assert is_id(identifier)
    assert identifier.isupper() or identifier.isdigit()


def test_sorts_by_time():
    early = new_id(at=1_700_000_000.0)
    late = new_id(at=1_700_000_001.0)
    assert early < late


def test_unique():
    identifiers = {new_id() for _ in range(500)}
    assert len(identifiers) == 500


def test_rejects_bad_ids():
    assert not is_id("too-short")
    assert not is_id("U" * LENGTH)
