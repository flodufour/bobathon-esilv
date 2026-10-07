"""Package root — Parkinson's disease motor-score prediction challenge.

Exposes `PROJECT_ROOT`: the absolute path to the project root,
derived from this file's location. Any module that needs a
project-relative path imports this constant instead of hard-coding
a CWD-relative string.
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
