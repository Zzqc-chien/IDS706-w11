import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "dags"), str(ROOT / "scripts")]

MONTHS = ["2026-04", "2026-05"]


@pytest.fixture(scope="session")
def raw_dir(tmp_path_factory):
    from make_sample_data import make_month

    out = tmp_path_factory.mktemp("raw")
    for m in MONTHS:
        make_month(m, out)
    return out
