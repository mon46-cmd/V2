"""Pytest configuration for V8 tests.

Adds the src/ directory to sys.path so that ``import core`` and
``import downloader`` work without an editable install.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Repo root is two levels up from this file (tests/conftest.py).
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC       = REPO_ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
