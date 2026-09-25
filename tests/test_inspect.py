import shutil

import pyarrow as pa
import pyarrow.parquet as pq
from conftest import SMALL

from engine.cli import main
from engine.io.metadata import scan_metadata


def test_footer_scan_matches_manifest(small_dataset):
    root, manifest = small_dataset
    stats = scan_metadata(root)
    assert stats.problems == []
    assert stats.rows == SMALL.rows
    assert len(stats.files) == manifest["total_files"]
    assert stats.bytes_on_disk == manifest["events_bytes"]
    assert {dict(p)["month"] for p in stats.partitions} == {"01", "02", "03"}
    assert list(stats.columns) == [f.name for f in pa.schema(stats.schema)]
    assert stats.timestamp_min.year == 2025 and stats.timestamp_max.month == 3


def test_decoded_width_is_measured(small_dataset):
    root, _ = small_dataset
    stats = scan_metadata(root)
    # Fixed-width columns decode to exactly their type width.
    assert stats.columns["event_id"].decoded_bytes_per_row == 8
    assert stats.columns["vendor_id"].decoded_bytes_per_row == 4
    assert stats.decoded_bytes_per_row > 48  # 48 B of fixed-width columns + strings


def test_reports_corrupt_and_mismatched_files(tmp_path, small_dataset):
    root, _ = small_dataset
    copy = tmp_path / "copy"
    shutil.copytree(root, copy)
    part = copy / "events/year=2025/month=01"
    (part / "part-99998.parquet").write_bytes(b"not a parquet file")
    pq.write_table(pa.table({"event_id": pa.array([1], pa.int64())}), part / "part-99999.parquet")

    stats = scan_metadata(copy)
    bad = {p.name: msg for p, msg in stats.problems}
    assert "unreadable" in bad["part-99998.parquet"]
    assert "schema differs" in bad["part-99999.parquet"]
    assert stats.rows == SMALL.rows  # the good files are still counted
    assert main(["inspect", "--dataset", str(copy)]) == 1


def test_cli_inspect_ok(small_dataset, capsys):
    root, _ = small_dataset
    assert main(["inspect", "--dataset", str(root)]) == 0
    out = capsys.readouterr().out
    assert f"{SMALL.rows:,}" in out and "COLUMNS" in out


def test_cli_inspect_missing_dataset(tmp_path):
    assert main(["inspect", "--dataset", str(tmp_path / "nope")]) == 2
