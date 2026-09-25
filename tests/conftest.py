import importlib.util

import pytest


def pytest_collection_modifyitems(config, items):
    if importlib.util.find_spec("cudf") is None:
        skip = pytest.mark.skip(reason="cuDF not installed")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip)
