import os

import pytest

os.environ.setdefault("NUMBA_NUM_THREADS", "1")

from busopt.config import load_config  # noqa: E402
from busopt.opt.problem import build_instance  # noqa: E402
from busopt.synthetic import toy_tables  # noqa: E402


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture(scope="session")
def tables():
    return toy_tables()


@pytest.fixture()
def inst(tables, cfg):
    return build_instance(tables, cfg, "weekday")
