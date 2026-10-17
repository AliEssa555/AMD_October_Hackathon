"""Shared fixtures. ``REPO`` is the repository root, so tests never depend on the CWD."""

from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def repo() -> Path:
    return REPO
