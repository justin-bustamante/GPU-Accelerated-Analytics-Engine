"""Dataset inspection from Parquet footers only.

Nothing here reads column data except one sampled row group, which is used to
estimate the *decoded* (in-memory, Arrow/cuDF-like) size per row. That number,
not the compressed on-disk size, is what has to fit in GPU memory, so it is
reported separately and labelled as an estimate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from engine.datagen.generator import EVENTS_DIR, load_manifest


@dataclass
class ColumnStats:
    name: str
    type: str
    compressed_bytes: int = 0
    uncompressed_bytes: int = 0
    decoded_bytes_per_row: float | None = None  # from the sampled row group


@dataclass
class FileStats:
    path: Path
    partition: dict[str, str]
    rows: int
    row_groups: int
    bytes_on_disk: int


@dataclass
class DatasetStats:
    root: Path
    files: list[FileStats] = field(default_factory=list)
    columns: dict[str, ColumnStats] = field(default_factory=dict)
    schema: pa.Schema | None = None
    problems: list[tuple[Path, str]] = field(default_factory=list)
    timestamp_min: object = None
    timestamp_max: object = None
    compression: set[str] = field(default_factory=set)
    decoded_bytes_per_row: float | None = None
    manifest: dict | None = None

    @property
    def rows(self) -> int:
        return sum(f.rows for f in self.files)

    @property
    def bytes_on_disk(self) -> int:
        return sum(f.bytes_on_disk for f in self.files)

    @property
    def row_groups(self) -> int:
        return sum(f.row_groups for f in self.files)

    @property
    def partitions(self) -> set[tuple[tuple[str, str], ...]]:
        return {tuple(sorted(f.partition.items())) for f in self.files}

    @property
    def estimated_decoded_bytes(self) -> int | None:
        if self.decoded_bytes_per_row is None:
            return None
        return int(self.decoded_bytes_per_row * self.rows)


def events_root(root: str | Path) -> Path:
    """Accept either a generated dataset root or a bare directory of Parquet files."""
    root = Path(root)
    return root / EVENTS_DIR if (root / EVENTS_DIR).is_dir() else root


def list_parquet_files(root: str | Path) -> list[Path]:
    return sorted(p for p in events_root(root).rglob("*.parquet") if p.is_file())


def hive_partition(path: Path, base: Path) -> dict[str, str]:
    parts = path.relative_to(base).parts[:-1]
    return dict(p.split("=", 1) for p in parts if "=" in p)


def scan_metadata(root: str | Path, sample_decoded: bool = True) -> DatasetStats:
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"dataset not found: {root}")
    base = events_root(root)
    stats = DatasetStats(root=root, manifest=load_manifest(root))

    for path in list_parquet_files(root):
        try:
            md = pq.read_metadata(path)
            schema = md.schema.to_arrow_schema()
        except Exception as exc:  # corrupt/truncated file: report it, keep going
            stats.problems.append((path, f"unreadable Parquet footer: {exc}"))
            continue

        if stats.schema is None:
            stats.schema = schema
            stats.columns = {f.name: ColumnStats(f.name, str(f.type)) for f in schema}
        elif not schema.equals(stats.schema):
            stats.problems.append((path, f"schema differs from first file: {schema.names}"))
            continue

        for rg in range(md.num_row_groups):
            group = md.row_group(rg)
            for c in range(group.num_columns):
                col = group.column(c)
                s = stats.columns.get(col.path_in_schema)
                if s is not None:
                    s.compressed_bytes += col.total_compressed_size
                    s.uncompressed_bytes += col.total_uncompressed_size
                stats.compression.add(col.compression)
                if col.path_in_schema == "timestamp" and col.is_stats_set:
                    lo, hi = col.statistics.min, col.statistics.max
                    if stats.timestamp_min is None or lo < stats.timestamp_min:
                        stats.timestamp_min = lo
                    if stats.timestamp_max is None or hi > stats.timestamp_max:
                        stats.timestamp_max = hi

        stats.files.append(
            FileStats(
                path,
                hive_partition(path, base),
                md.num_rows,
                md.num_row_groups,
                path.stat().st_size,
            )
        )

    if sample_decoded:
        _sample_decoded_width(stats)
    return stats


def _sample_decoded_width(stats: DatasetStats) -> None:
    """Decode one row group and measure Arrow bytes/row, per column and in total."""
    for f in stats.files:
        if f.rows == 0:
            continue
        table = pq.ParquetFile(f.path).read_row_group(0)
        if table.num_rows == 0:
            continue
        stats.decoded_bytes_per_row = table.nbytes / table.num_rows
        for name in table.column_names:
            if name in stats.columns:
                stats.columns[name].decoded_bytes_per_row = (
                    table.column(name).nbytes / table.num_rows
                )
        return
