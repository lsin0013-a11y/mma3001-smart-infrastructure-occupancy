"""Shared pytest fixtures for the smart_infra test suite."""

import sys
from pathlib import Path

# Allow "import smart_infra" during testing without an editable install.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
