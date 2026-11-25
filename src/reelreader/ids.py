"""ULID-style identifiers: lexicographically sortable, 26 characters."""

from __future__ import annotations

import os
import time

ALPHABET = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
LENGTH = 26


def new_id(at: float | None = None) -> str:
    timestamp_ms = int((time.time() if at is None else at) * 1000)
    value = (timestamp_ms << 80) | int.from_bytes(os.urandom(10), "big")
    return _encode(value)


def _encode(value: int) -> str:
    chars: list[str] = []
    for _ in range(LENGTH):
        chars.append(ALPHABET[value & 31])
        value >>= 5
    return "".join(reversed(chars))


def is_id(value: str) -> bool:
    return len(value) == LENGTH and all(char in ALPHABET for char in value)
