import dataclasses
import json

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import pytest
from conftest import SMALL

from engine.datagen import schema as S
from engine.datagen.generator import (
    MANIFEST_NAME,
    GeneratorConfig,
    allocate,
    generate,
    plan_files,
)


def read_events(root) -> pa.Table:
    return ds.dataset(root / "events", format="parquet", partitioning="hive").to_table()


# ---------------------------------------------------------------------------- planning


@pytest.mark.parametrize("total", [0, 1, 7, 1000, 123_456_789])
def test_allocate_is_exact(total):
    parts = allocate(total, [0.9, 0.85, 1.3, 1.5, 0.01])
    assert sum(parts) == total
    assert all(p >= 0 for p in parts)


def test_plan_is_contiguous_and_bounded():
    cfg = GeneratorConfig(rows=10_000_001, rows_per_file=1_000_000)
    tasks = plan_files(cfg)
    assert sum(t.rows for t in tasks) == cfg.rows
    assert all(0 < t.rows <= cfg.rows_per_file for t in tasks)
    next_id = 0
    for t in tasks:
        assert t.first_event_id == next_id
        next_id += t.rows
    assert len({t.relpath for t in tasks}) == len(tasks)


def test_plan_scales_without_materializing_rows():
    # Planning 1B rows must be instant: it only produces file descriptors.
    tasks = plan_files(GeneratorConfig(rows=1_000_000_000))
    assert sum(t.rows for t in tasks) == 1_000_000_000


def test_fewer_rows_than_months_skips_empty_months(tmp_path):
    manifest = generate(dataclasses.replace(SMALL, rows=2, months=24), tmp_path, log=lambda _: None)
    assert manifest["total_rows"] == 2
    assert read_events(tmp_path).num_rows == 2


def test_zero_rows(tmp_path):
    manifest = generate(dataclasses.replace(SMALL, rows=0), tmp_path, log=lambda _: None)
    assert manifest["total_files"] == 0


# ------------------------------------------------------------------------------ output


def test_layout_and_manifest(small_dataset):
    root, manifest = small_dataset
    assert json.loads((root / MANIFEST_NAME).read_text()) == manifest
    assert manifest["total_rows"] == SMALL.rows
    files = sorted((root / "events").rglob("*.parquet"))
    assert len(files) == manifest["total_files"]
    assert {str(f.relative_to(root)) for f in files} == {f["path"] for f in manifest["files"]}
    for entry in manifest["files"]:
        assert pq.read_metadata(root / entry["path"]).num_rows == entry["rows"]
    assert not list(root.rglob("*.tmp"))


def test_schema_is_exact(small_dataset):
    root, _ = small_dataset
    for f in (root / "events").rglob("*.parquet"):
        assert pq.read_schema(f).remove_metadata().equals(S.EVENT_SCHEMA)
    assert (
        pq.read_schema(root / "products/products.parquet")
        .remove_metadata()
        .equals(S.PRODUCT_SCHEMA)
    )


def test_row_groups_respect_configured_size(small_dataset):
    root, _ = small_dataset
    for f in (root / "events").rglob("*.parquet"):
        md = pq.read_metadata(f)
        assert md.num_row_groups >= 2
        assert all(
            md.row_group(i).num_rows <= SMALL.row_group_size for i in range(md.num_row_groups)
        )


def test_event_ids_are_dense_and_unique(small_dataset):
    root, _ = small_dataset
    ids = np.sort(read_events(root)["event_id"].to_numpy())
    assert np.array_equal(ids, np.arange(SMALL.rows, dtype=np.uint64))


def test_timestamps_match_partition_and_are_sorted_per_file(small_dataset):
    root, _ = small_dataset
    t = read_events(root)
    assert pc.all(pc.equal(pc.year(t["timestamp"]), t["year"].cast(pa.int64()))).as_py()
    assert pc.all(pc.equal(pc.month(t["timestamp"]), t["month"].cast(pa.int64()))).as_py()
    for f in (root / "events").rglob("*.parquet"):
        ts = pq.read_table(f, columns=["timestamp"])["timestamp"].to_numpy()
        assert np.all(ts[:-1] <= ts[1:])


def test_values_and_domains(small_dataset):
    root, _ = small_dataset
    t = read_events(root)
    assert set(pc.unique(t["region"]).to_pylist()) <= set(S.names(S.REGIONS))
    assert set(pc.unique(t["event_type"]).to_pylist()) <= set(S.names(S.EVENT_TYPES))
    assert pc.min(t["quantity"]).as_py() >= 1
    assert pc.max(t["user_id"]).as_py() <= SMALL.n_users
    refunds = pc.equal(t["event_type"], S.REFUND)
    assert pc.all(pc.less(pc.filter(t["value"], refunds), 0)).as_py()
    assert pc.all(pc.greater(pc.filter(t["value"], pc.invert(refunds)), 0)).as_py()
    # Only latency_ms is nullable, at roughly the configured rate.
    nulls = {name: t[name].null_count for name in S.EVENT_SCHEMA.names}
    assert all(v == 0 for k, v in nulls.items() if k != "latency_ms")
    assert 0.01 < nulls["latency_ms"] / t.num_rows < 0.03


def test_events_are_consistent_with_products(small_dataset):
    root, _ = small_dataset
    events = read_events(root).select(["product_id", "vendor_id", "category"])
    products = pq.read_table(root / "products/products.parquet")
    joined = events.join(products, "product_id", right_suffix="_p")
    assert joined.num_rows == events.num_rows  # every event has a product
    assert pc.all(pc.equal(joined["vendor_id"], joined["vendor_id_p"])).as_py()
    assert pc.all(pc.equal(joined["category"], joined["category_p"])).as_py()


# ------------------------------------------------------------------------- determinism


def test_output_independent_of_worker_count(tmp_path, small_dataset):
    root, manifest = small_dataset
    other = tmp_path / "parallel"
    parallel = generate(SMALL, other, workers=2, log=lambda _: None)
    assert parallel["fingerprint"] == manifest["fingerprint"]
    for entry in manifest["files"]:
        assert pq.read_table(root / entry["path"]).equals(pq.read_table(other / entry["path"]))


def test_seed_changes_data(tmp_path, small_dataset):
    root, manifest = small_dataset
    other = generate(dataclasses.replace(SMALL, seed=7), tmp_path, log=lambda _: None)
    assert other["fingerprint"] != manifest["fingerprint"]
    path = manifest["files"][0]["path"]
    assert not pq.read_table(root / path).equals(pq.read_table(tmp_path / path))


# ---------------------------------------------------------------------- safety/errors


def test_refuses_nonempty_output(tmp_path):
    (tmp_path / "keep.txt").write_text("mine")
    with pytest.raises(FileExistsError):
        generate(SMALL, tmp_path, log=lambda _: None)
    with pytest.raises(FileExistsError, match="no _manifest.json"):
        generate(SMALL, tmp_path, overwrite=True, log=lambda _: None)
    assert (tmp_path / "keep.txt").exists()


def test_overwrite_replaces_generated_dataset(tmp_path):
    generate(SMALL, tmp_path, log=lambda _: None)
    m = generate(dataclasses.replace(SMALL, rows=10), tmp_path, overwrite=True, log=lambda _: None)
    assert read_events(tmp_path).num_rows == m["total_rows"] == 10


@pytest.mark.parametrize(
    "field,value",
    [
        ("rows", -1),
        ("months", 0),
        ("rows_per_file", 0),
        ("start_month", 13),
        ("latency_null_fraction", 1.0),
    ],
)
def test_invalid_config(tmp_path, field, value):
    with pytest.raises(ValueError):
        generate(dataclasses.replace(SMALL, **{field: value}), tmp_path, log=lambda _: None)
