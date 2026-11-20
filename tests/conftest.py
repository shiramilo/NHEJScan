import os
import shutil
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))


@pytest.fixture(scope="session")
def trace_ext():
    from nhejscan.build import ensure_trace_ext
    return ensure_trace_ext()


def have(*tools):
    return all(shutil.which(t) for t in tools)
