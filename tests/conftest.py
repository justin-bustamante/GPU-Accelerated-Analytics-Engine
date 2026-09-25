import importlib.util

import pytest

from engine.datagen.generator import GeneratorConfig, generate

# Small but structurally complete: multiple months, multiple files per month,
# multiple row groups per file, and a small product/vendor space.
SMALL = GeneratorConfig(
    rows=12_345,
    months=3,
    rows_per_file=2_000,
    row_group_size=700,
    n_vendors=50,
    n_products=400,
    n_users=1_000,
    latency_null_fraction=0.02,
)


def pytest_collection_modifyitems(config, items):
    if importlib.util.find_spec("cudf") is None:
        skip = pytest.mark.skip(reason="cuDF not installed")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip)


@pytest.fixture(scope="session")
def small_dataset(tmp_path_factory):
    root = tmp_path_factory.mktemp("small")
    manifest = generate(SMALL, root, log=lambda _: None)
    return root, manifest
