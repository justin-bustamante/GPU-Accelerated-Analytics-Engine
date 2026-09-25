"""Streaming, deterministic synthetic Parquet generator.

Design:

1. Plan first. The total row count is split across months (seasonality + growth),
   then each month is split into files of at most `rows_per_file` rows. Every file
   gets a global index and a contiguous `event_id` range up front.
2. Generate each file independently. A file's random stream is seeded from
   (seed, file index) via SeedSequence, so any file can be produced in any order,
   by any worker, and the output does not depend on the number of workers.
3. Peak memory is bounded by one file's worth of rows per worker, never by the
   dataset size. 500M rows costs the same RAM as 5M, just more time.
4. Files are written to a temp name and atomically renamed. `_manifest.json` is
   written last, so its presence means the dataset is complete.
"""

from __future__ import annotations

import calendar
import hashlib
import json
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from engine.datagen import schema as S

GENERATOR_VERSION = 1  # bump when the output for a given config changes
MANIFEST_NAME = "_manifest.json"
EVENTS_DIR = "events"
PRODUCTS_PATH = "products/products.parquet"

_US_PER_HOUR = 3_600_000_000
_US_PER_DAY = 24 * _US_PER_HOUR

# SeedSequence spawn keys: one stream for the product table, one per events file.
_PRODUCTS_STREAM = 0
_EVENTS_STREAM = 1


@dataclass(frozen=True)
class GeneratorConfig:
    rows: int
    seed: int = 42
    start_year: int = 2025
    start_month: int = 1
    months: int = 24
    rows_per_file: int = 2_000_000
    row_group_size: int = 1_000_000
    compression: str = "snappy"
    n_vendors: int = 20_000
    n_products: int = 200_000
    n_users: int = 50_000_000
    latency_null_fraction: float = 0.005

    def validate(self) -> None:
        if self.rows < 0:
            raise ValueError("rows must be >= 0")
        if not 1 <= self.start_month <= 12:
            raise ValueError("start_month must be in 1..12")
        for name in (
            "months",
            "rows_per_file",
            "row_group_size",
            "n_vendors",
            "n_products",
            "n_users",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1")
        if not 0.0 <= self.latency_null_fraction < 1.0:
            raise ValueError("latency_null_fraction must be in [0, 1)")

    def fingerprint(self) -> str:
        """Identifies the logical content of a dataset: same fingerprint, same rows."""
        payload = json.dumps({"v": GENERATOR_VERSION, **asdict(self)}, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


@dataclass(frozen=True)
class FileTask:
    index: int
    year: int
    month: int
    part: int
    rows: int
    first_event_id: int

    @property
    def relpath(self) -> str:
        return (
            f"{EVENTS_DIR}/year={self.year:04d}/month={self.month:02d}/part-{self.part:05d}.parquet"
        )


@dataclass
class Products:
    """Product dimension, held in memory by every worker (small: ~n_products rows)."""

    vendor_id: np.ndarray  # int32, indexed by product_id - 1
    category_idx: np.ndarray  # int32
    base_price: np.ndarray  # float64
    brand_idx: np.ndarray  # int32
    popularity_cdf: np.ndarray  # float64, for sampling product ids in events


# --------------------------------------------------------------------------- planning


def allocate(total: int, weights) -> list[int]:
    """Split `total` into integer parts proportional to `weights` (largest remainder).

    Always sums exactly to `total`, which keeps row counts exact at any scale.
    """
    w = np.asarray(weights, dtype=np.float64)
    ideal = total * w / w.sum()
    parts = np.floor(ideal).astype(np.int64)
    remainder = total - int(parts.sum())
    if remainder:
        order = np.argsort(-(ideal - parts), kind="stable")
        parts[order[:remainder]] += 1
    return [int(p) for p in parts]


def month_sequence(cfg: GeneratorConfig) -> list[tuple[int, int]]:
    out = []
    y, m = cfg.start_year, cfg.start_month
    for _ in range(cfg.months):
        out.append((y, m))
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def plan_files(cfg: GeneratorConfig) -> list[FileTask]:
    months = month_sequence(cfg)
    weights = [
        S.SEASONALITY[m - 1] * S.ANNUAL_GROWTH ** (i / 12) for i, (_, m) in enumerate(months)
    ]
    tasks: list[FileTask] = []
    next_event_id = 0
    for (year, month), month_rows in zip(months, allocate(cfg.rows, weights), strict=True):
        if month_rows == 0:
            continue
        n_files = -(-month_rows // cfg.rows_per_file)  # ceil
        for part, rows in enumerate(allocate(month_rows, [1.0] * n_files)):
            tasks.append(FileTask(len(tasks), year, month, part, rows, next_event_id))
            next_event_id += rows
    return tasks


# ------------------------------------------------------------------------- generation


def _rng(cfg: GeneratorConfig, *key: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence(cfg.seed, spawn_key=key))


def build_products(cfg: GeneratorConfig) -> Products:
    rng = _rng(cfg, _PRODUCTS_STREAM)
    n = cfg.n_products
    vendor_cdf = S.zipf_cdf(cfg.n_vendors, S.VENDOR_SIZE_EXPONENT, rng)
    vendor_id = (np.searchsorted(vendor_cdf, rng.random(n), side="right") + 1).astype(np.int32)
    category_idx = np.searchsorted(
        S.cdf([w for _, w in S.CATEGORIES]), rng.random(n), side="right"
    ).astype(np.int32)
    base_price = np.clip(np.round(rng.lognormal(np.log(25.0), 1.0, n), 2), 0.5, 5000.0)
    n_brands = max(1, cfg.n_vendors // 2)
    brand_idx = rng.integers(0, n_brands, n, dtype=np.int32)
    popularity_cdf = S.zipf_cdf(n, S.PRODUCT_POPULARITY_EXPONENT, rng)
    return Products(vendor_id, category_idx, base_price, brand_idx, popularity_cdf)


def products_table(p: Products) -> pa.Table:
    n = len(p.vendor_id)
    return pa.Table.from_arrays(
        [
            pa.array(np.arange(1, n + 1, dtype=np.int32)),
            pc.take(pa.array(S.names(S.CATEGORIES)), pa.array(p.category_idx)),
            pa.array(p.vendor_id),
            pa.array(p.base_price),
            pc.take(
                pa.array([f"brand-{i:05d}" for i in range(int(p.brand_idx.max()) + 1)]),
                pa.array(p.brand_idx),
            ),
        ],
        schema=S.PRODUCT_SCHEMA,
    )


def generate_events(cfg: GeneratorConfig, task: FileTask, products: Products) -> pa.Table:
    rng = _rng(cfg, _EVENTS_STREAM, task.index)
    n = task.rows

    # Timestamps: uniform day within the month, diurnal hour-of-day, uniform within
    # the hour. Sorted within the file so row-group min/max statistics are tight,
    # which later makes row-group-level pruning possible.
    month_start = int(np.datetime64(f"{task.year:04d}-{task.month:02d}-01", "us").astype(np.int64))
    days = calendar.monthrange(task.year, task.month)[1]
    ts = (
        month_start
        + rng.integers(0, days, n, dtype=np.int64) * _US_PER_DAY
        + np.searchsorted(S.cdf(S.DIURNAL), rng.random(n), side="right") * _US_PER_HOUR
        + rng.integers(0, _US_PER_HOUR, n, dtype=np.int64)
    )
    ts.sort()

    # Products are Zipf-popular; vendor and category follow from the product so
    # events are consistent with the products table (matters for the join workload).
    pidx = np.searchsorted(products.popularity_cdf, rng.random(n), side="right")
    region_idx = np.searchsorted(S.cdf([w for _, w in S.REGIONS]), rng.random(n), side="right")
    etype_idx = np.searchsorted(S.cdf([w for _, w in S.EVENT_TYPES]), rng.random(n), side="right")

    quantity = np.minimum(rng.geometric(0.6, n), 20).astype(np.int32)
    discount = np.where(rng.random(n) < 0.2, rng.choice([0.05, 0.1, 0.15, 0.2, 0.3], n), 0.0)
    sign = np.where(etype_idx == S.names(S.EVENT_TYPES).index(S.REFUND), -1.0, 1.0)
    value = np.round(sign * products.base_price[pidx] * quantity * (1.0 - discount), 2)

    latency = rng.lognormal(np.log(80.0), 0.6, n).astype(np.float32)
    latency_null = rng.random(n) < cfg.latency_null_fraction

    columns = [
        pa.array(np.arange(task.first_event_id, task.first_event_id + n, dtype=np.uint64)),
        pa.array(ts, type=pa.timestamp("us")),
        pa.array(rng.integers(1, cfg.n_users + 1, n, dtype=np.uint64)),
        pa.array(products.vendor_id[pidx]),
        pa.array((pidx + 1).astype(np.int32)),
        pc.take(pa.array(S.names(S.REGIONS)), pa.array(region_idx.astype(np.int32))),
        pc.take(pa.array(S.names(S.CATEGORIES)), pa.array(products.category_idx[pidx])),
        pc.take(pa.array(S.names(S.EVENT_TYPES)), pa.array(etype_idx.astype(np.int32))),
        pa.array(quantity),
        pa.array(value),
        pa.array(latency, mask=latency_null),
    ]
    return pa.Table.from_arrays(columns, schema=S.EVENT_SCHEMA)


def _write_atomic(table: pa.Table, path: Path, cfg: GeneratorConfig) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    pq.write_table(table, tmp, row_group_size=cfg.row_group_size, compression=cfg.compression)
    os.replace(tmp, path)
    return path.stat().st_size


# Per-process cache so each worker builds the product table once, not once per file.
_worker_products: tuple[GeneratorConfig, Products] | None = None


def _products_for(cfg: GeneratorConfig) -> Products:
    global _worker_products
    if _worker_products is None or _worker_products[0] != cfg:
        _worker_products = (cfg, build_products(cfg))
    return _worker_products[1]


def _run_task(cfg: GeneratorConfig, root: str, task: FileTask) -> tuple[int, int]:
    table = generate_events(cfg, task, _products_for(cfg))
    return task.index, _write_atomic(table, Path(root) / task.relpath, cfg)


# --------------------------------------------------------------------------- driver


def _prepare_output(root: Path, overwrite: bool) -> None:
    if root.exists() and any(root.iterdir()):
        if not overwrite:
            raise FileExistsError(f"{root} is not empty (pass --overwrite to replace it)")
        if not (root / MANIFEST_NAME).exists():
            # Refuse to wipe a directory we cannot prove we created.
            raise FileExistsError(f"refusing to overwrite {root}: it has no {MANIFEST_NAME}")
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)


def generate(
    cfg: GeneratorConfig,
    output: str | Path,
    workers: int = 1,
    overwrite: bool = False,
    log=None,
) -> dict:
    """Generate a dataset under `output` and return its manifest."""
    cfg.validate()
    log = log or (lambda msg: print(msg, file=sys.stderr, flush=True))
    root = Path(output)
    _prepare_output(root, overwrite)
    tasks = plan_files(cfg)
    started = time.perf_counter()

    products = _products_for(cfg)
    products_bytes = _write_atomic(products_table(products), root / PRODUCTS_PATH, cfg)

    sizes: dict[int, int] = {}
    done_rows = 0
    last_log = started

    def record(index: int, size: int) -> None:
        nonlocal done_rows, last_log
        sizes[index] = size
        done_rows += tasks[index].rows
        now = time.perf_counter()
        if len(sizes) == len(tasks) or now - last_log >= 5.0:
            last_log = now
            elapsed = now - started
            log(
                f"  {len(sizes):>5}/{len(tasks)} files  {done_rows:>14,} rows  "
                f"{done_rows / max(elapsed, 1e-9):>12,.0f} rows/s"
            )

    if workers <= 1:
        for t in tasks:
            record(*_run_task(cfg, str(root), t))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for index, size in pool.map(
                _run_task, [cfg] * len(tasks), [str(root)] * len(tasks), tasks, chunksize=1
            ):
                record(index, size)

    elapsed = time.perf_counter() - started
    manifest = {
        "format_version": 1,
        "generator_version": GENERATOR_VERSION,
        "fingerprint": cfg.fingerprint(),
        "config": asdict(cfg),
        "created_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "pyarrow_version": pa.__version__,
        "generation_seconds": round(elapsed, 3),
        "total_rows": cfg.rows,
        "total_files": len(tasks),
        "events_bytes": sum(sizes.values()),
        "products": {"path": PRODUCTS_PATH, "rows": cfg.n_products, "bytes": products_bytes},
        "event_schema": {f.name: str(f.type) for f in S.EVENT_SCHEMA},
        "files": [
            {
                "path": t.relpath,
                "rows": t.rows,
                "first_event_id": t.first_event_id,
                "bytes": sizes[t.index],
            }
            for t in tasks
        ],
    }
    tmp = root / (MANIFEST_NAME + ".tmp")
    tmp.write_text(json.dumps(manifest, indent=2))
    os.replace(tmp, root / MANIFEST_NAME)
    return manifest


def load_manifest(root: str | Path) -> dict | None:
    path = Path(root) / MANIFEST_NAME
    return json.loads(path.read_text()) if path.exists() else None
