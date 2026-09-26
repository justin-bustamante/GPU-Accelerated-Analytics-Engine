# Design plan

This document fixes the design decisions for Milestone 1 and sketches the
shape of the milestones after it. Every number in it is one of:

- **measured**: by running code in this repo, with the machine named;
- **spec sheet**: a vendor figure, not yet verified;
- **extrapolated**: arithmetic from measured values, labelled as such.

Nothing here is a GPU benchmark result. There are none yet.

---

## 1. Technical thesis

A consumer GPU can beat a strong multi-core CPU engine (DuckDB) on analytics
only when the data a query needs reaches the GPU faster than the CPU could have
processed it. On an RTX 4070 SUPER that means getting past a ~32 GB/s PCIe link,
Parquet decode, and 12 GB of VRAM that is shared with the desktop.

For datasets larger than VRAM, the engine's job is threefold:

- shrink the data to the *pruned working set*;
- stream that set through the GPU in memory-safe batches, overlapping I/O and
  decode with compute;
- know when the CPU is simply the better choice.

The project measures where those lines fall.

## 2. Smallest useful MVP

One loop, end to end, for two workloads (`vendor_groupby` and `filter_projection`):

```
generate Parquet ─► DuckDB runs workload ─┐
                 └► cuDF runs workload ───┴─► validate ─► append JSONL result (with phase timings)
```

Then the first engine feature: **fixed-size batching** of `vendor_groupby` over a
dataset whose projected working set exceeds free VRAM, with partial aggregates
merged on the GPU and validated against DuckDB.

That is Milestones 1–4, restricted to two workloads. Everything after it is an
optimization of this loop, measured against it:

- adaptive batching
- pruning
- prefetch
- S3
- routing

If the MVP exists and nothing else does, the project already answers two
questions: "where does the GPU win at in-memory scale?" and "can I process more
data than fits in VRAM, and at what cost?"

## 3. Repository architecture

What exists now (Milestone 1):

```
src/engine/
  cli.py                 # python -m engine {generate, inspect, inspect-gpu}
  datagen/
    schema.py            # event/product schemas, distributions
    generator.py         # plan → per-file deterministic generation → atomic writes
  io/
    metadata.py          # footer-only dataset scan (files, row groups, per-column sizes)
  telemetry/
    hardware.py          # host info, NVML GPU info, cuDF smoke test
tests/                   # CPU tests run everywhere; @pytest.mark.gpu tests skip without cuDF
docs/                    # plan.md (this), environment.md
```

Where later milestones land. Modules are created when their milestone starts,
never as empty placeholders.

```
src/engine/
  workloads/     M2  one logical definition per workload: required columns, predicate,
                     DuckDB SQL, cuDF implementation, partial-merge function
  backends/      M2  duckdb.py (CPU), M3 cudf.py (GPU); same interface, same inputs
  bench/         M2  runner, result record, correctness comparison, JSONL writer
  execution/     M4  batch iterator (fixed), M5 adaptive sizing + OOM retry, M7 prefetch
  planner/       M6  Plan object: pruning, projection, backend, batch strategy.
                     `explain` prints it, `benchmark` executes the *same* object
  io/            M6  file/row-group selection, M8 filesystem abstraction (local / S3)
benchmarks/
  configs/       M2  benchmark matrices as TOML
  results/       M2  append-only JSONL, committed
  analysis/      M9  scripts that turn results into tables/charts for docs/findings.md
```

Deviations from the sketch in the brief, and why:

- **`workloads/` is first-class.** A workload's logical definition is shared by
  every backend, by the planner (which columns and partitions it needs) and by
  the merge step (how partials combine). Keeping it in one place is what
  prevents the CPU and GPU from quietly computing different things.
- **`batching/` and pipelining are merged into `execution/`.** They are the same
  loop at different levels of sophistication.
- **No `scripts/generate_data.py`.** Generation is `python -m engine generate`,
  one code path.

## 4. Milestone 1 dependencies

| Package | Constraint | Why |
|---|---|---|
| Python | ≥ 3.11 (3.12 recommended) | cuDF 26.08 requires ≥ 3.11 |
| numpy | `>=2.0,<3` | generator; matches cuDF's range |
| pyarrow | `>=19,<24` | Parquet read/write. Upper bound matches cuDF 26.08, so CPU-only and GPU installs get the same pyarrow |
| nvidia-ml-py | `>=12.535` | NVML: driver, CUDA version, VRAM, PCIe *without* needing cuDF |
| psutil | `>=5.9` | host RAM/CPU facts (and CPU utilization sampling later) |
| pytest, ruff | dev only | tests, lint/format |
| **cudf-cu12 / cudf-cu13** | `==26.8.*`, optional extra | GPU. Pulls in rmm, pylibcudf, cupy, numba-cuda, pandas 3, CUDA user-space wheels |

Deliberately *not* in Milestone 1:

- `duckdb` (Milestone 2)
- `polars` (optional secondary baseline, later)
- `boto3` (probably never: `pyarrow.fs.S3FileSystem` covers Milestone 8)

Checked on PyPI 2026-09-25:

- RAPIDS 26.08.1 is current.
- pyarrow 25.0.1 is the newest, but cuDF rejects it.

Tested in the build container:

- Python 3.11.15, numpy 2.4.6, pyarrow 23.0.1.

## 5. Development environment

The details are in [environment.md](environment.md). Summary:

- **OS.** Ubuntu 24.04, either natively or under WSL2 on Windows 11.
  - WSL2 is fine for development and most benchmarks.
  - Keep data on the Linux filesystem, not `/mnt/c`.
- **Driver.** Install the NVIDIA driver on Windows only (for WSL2) or on the
  host (for native Linux).
  - No system CUDA toolkit is needed, because the pip wheels bring CUDA
    user-space.
- **Wheels.** The driver's reported CUDA version picks `cudf-cu13` (driver
  reports 13.x) or `cudf-cu12`.
- **Python.** A plain venv (`uv venv`).
  - Docker is postponed to Milestone 9, for one-command reproducible benchmark
    runs.
- **VRAM.** Remember the GPU is probably also driving the desktop. Free VRAM is
  below 12 GB and it varies, so it is recorded with every result.

## 6. Verifying CUDA and cuDF

The layered checks are in [environment.md](environment.md#verifying-that-cuda-and-cudf-can-use-the-gpu):

1. `nvidia-smi` checks the driver.
2. `python -m engine inspect-gpu` checks NVML from Python. It needs no cuDF.
3. `import cudf` checks that the wheels match the driver.
4. `python -m engine inspect-gpu --smoke --require-gpu` runs a 10M-row GPU
   group-by checked against NumPy, and exits non-zero on failure.
5. `pytest -m gpu` runs the same smoke test under the test suite.

## 7. Synthetic event schema

Fact table `events` (Hive-partitioned Parquet). Decoded widths are **measured**:
`inspect` decodes one row group and reports Arrow bytes/row per column.

| Column | Arrow type | Null | Cardinality / distribution | Decoded B/row |
|---|---|---|---|---|
| `event_id` | `uint64` | no | dense 0..N-1, unique, file-contiguous | 8.0 |
| `timestamp` | `timestamp[us]` (naive, UTC) | no | 2025-01 .. 2026-12; seasonal months (Nov/Dec peak), +15 %/yr growth, diurnal hour-of-day curve; sorted within each file | 8.0 |
| `user_id` | `uint64` | no | uniform over 1..50M | 8.0 |
| `vendor_id` | `int32` | no | ≈20K vendors; derived from the product (skewed: vendors own Zipf-sized catalogs) | 4.0 |
| `product_id` | `int32` | no | 1..200K; Zipf popularity (s=0.8), popular ids scattered across the id range | 4.0 |
| `region` | `string` | no | 9 values, weighted | 11.8 |
| `category` | `string` | no | 20 values; derived from the product | 10.7 |
| `event_type` | `string` | no | view 55 %, search 15 %, add_to_cart 14 %, purchase 13 %, refund 3 % | 9.9 |
| `quantity` | `int32` | no | 1 + geometric, capped at 20 | 4.0 |
| `value` | `float64` | no | `base_price × quantity × (1 − discount)`, rounded to cents; **negative for refunds** | 8.0 |
| `latency_ms` | `float32` | **yes (0.5 %)** | log-normal, median ≈ 80 ms | 4.1 |
| **total** | | | | **80.5** |

Dimension table `products` (`products/products.parquet`, 200K rows):

- `product_id int32`
- `category string`
- `vendor_id int32`
- `base_price float64`
- `brand string`

Events are consistent with it: every event's `vendor_id` and `category` equal
its product's (tested). That makes the Milestone 5 join meaningful.

Choices worth being able to defend:

- **Strings are logical `string`, not dictionary/categorical.** The Parquet
  writer dictionary-encodes them on disk anyway. The region column is 1.3 % of
  on-disk bytes, but 11.8 B/row once decoded. Keeping the logical type plain
  avoids DuckDB, cuDF and pyarrow each doing something different with
  categoricals.
- **One nullable measure (`latency_ms`).** `AVG` must skip nulls identically on
  both engines, and batched merges must carry non-null counts rather than row
  counts. That is a real correctness hazard, planted on purpose.
- **Negative values (refunds)** keep sums from being trivially monotone and
  exercise sign handling in top-k.
- **Partition columns (`year`, `month`) live in directory names, not in files.**
  The same information is in `timestamp`. Mapping timestamp predicates to
  partitions is the engine's job in Milestone 6.

Layout:

```
<dataset>/
  _manifest.json                               # written last: presence == complete
  products/products.parquet
  events/year=2025/month=01/part-00000.parquet
                           /part-00001.parquet
         ...
         year=2026/month=12/part-00000.parquet
```

## 8. Data generator

Implementation: `src/engine/datagen/generator.py`.

1. **Plan, then execute.**
   - Split the total rows across months (seasonality × growth) with a
     largest-remainder allocation, so totals are exact at any scale.
   - Split each month into files of at most `rows_per_file` (default 2M).
   - Assign each file a global index and a contiguous `event_id` range.
   - Planning 1B rows produces a list of 512 file descriptors and no data.
2. **Each file is independent and deterministic.**
   - Its RNG is `SeedSequence(seed, spawn_key=(1, file_index))`.
   - Files can therefore be generated in any order by any number of processes,
     and the output does not depend on `--workers` (tested).
   - `_manifest.json` records a `fingerprint` of (generator version, config) that
     benchmark results can reference.
3. **Bounded memory.**
   - A worker holds one file's rows at a time plus the small product table.
   - Peak RSS follows `rows_per_file`, not dataset size.
4. **Crash safety.**
   - Each file is written as `*.tmp` and atomically renamed.
   - The manifest is written last.
   - `--overwrite` only deletes a directory that contains a manifest, i.e. one
     the tool itself created.
5. **Pruning-friendly on purpose.**
   - Timestamps are sorted within each file, so row-group min/max statistics
     are tight.
   - Row-group size is configurable (default 1M rows) because row groups are
     the natural batch unit later.

**Measured** in the build container, with a single worker unless noted
(4 vCPU Xeon @ 2.1 GHz, 16 GB RAM; *not* the target desktop, so treat speeds as
illustrative):

| Rows | Files | On disk | On-disk B/row | Peak RSS | Wall |
|---|---|---|---|---|---|
| 1M | 24 | 40.6 MB | 40.6 | – | 0.8 s (4 workers) |
| 10M | 24 | 328 MB | 32.8 | 249 MB | 10.7 s |
| 30M | 25 | 924 MB | 30.8 | 400 MB | 29.1 s |
| 60M | 47 | 1.85 GB | 30.8 | 425 MB | 55.1 s |

What the measurements say:

- **Memory is bounded.** From 30M to 60M rows the largest file stayed around
  1.9M rows, and peak RSS stayed around 400 MB.
- **On-disk bytes/row is not constant.** Small files pay per-file and
  per-dictionary overhead. Size estimates should use the at-scale value
  (~31 B/row), never the 1M-row value.
- **Decoded size is 2.6× the on-disk size** (80.5 vs 30.8 B/row). What has to
  fit in VRAM is the decoded size. Every size claim in this project will
  therefore name which size it means.

### Sizing consequences (extrapolated from the measurements above)

| Rows | On disk (~30.8 B/row) | Decoded, all columns (80.5 B/row) | Decoded ÷ 12 GB |
|---|---|---|---|
| 100M | ~3.1 GB | ~8 GB | 0.7× |
| 250M | ~7.7 GB | ~20 GB | 1.7× |
| 500M | ~15 GB | ~40 GB | 3.4× |
| 1B | ~31 GB | ~81 GB | 6.7× |

Two consequences shape the benchmark plan:

- **The row counts in the brief do not produce 100 GB on disk.** 500M rows is
  ~15 GB of Parquet with this schema. Reaching 100 GB on disk would take ~3B
  rows, or deliberately wider rows.
- **Column pruning can make the out-of-core problem disappear.**
  `vendor_groupby` reads only ~16 B/row, so 500M rows is ~8 GB decoded, which is
  near the free-VRAM limit rather than far beyond it. "Larger than VRAM" has to
  be defined against the *projected working set of the query*, not the dataset.
  Section 9 lists that working set for each workload.

## 9. The first four workloads

Each workload has one logical definition. DuckDB runs the SQL, and cuDF
implements the same semantics. Working-set widths come from the measured
per-column decoded sizes.

**W1 `filter_projection`**: selective scan; I/O and transfer should dominate.

```sql
SELECT vendor_id, value, timestamp
FROM events
WHERE region = 'eu-west' AND event_type = 'purchase'
  AND timestamp >= TIMESTAMP '2025-06-01' AND timestamp < TIMESTAMP '2025-09-01'
```

- Reads: 5 columns, ~41.7 B/row.
- Output: roughly 0.2 % of rows (estimate from the generator weights, to be
  measured).
- Batched execution: concatenate the filtered batches.

**W2 `vendor_groupby`**: wide, moderately high-cardinality aggregation (~20K groups).

```sql
SELECT vendor_id, SUM(value) AS total_value, COUNT(*) AS events,
       AVG(latency_ms) AS avg_latency_ms
FROM events GROUP BY vendor_id
```

- Reads: 3 columns, ~16.1 B/row.
- Batched execution: each batch emits `(vendor_id, sum_value, count_rows,
  sum_latency, count_latency_non_null)`. The merge re-aggregates by `vendor_id`,
  and `AVG = sum_latency / count_latency_non_null`. You cannot average averages,
  and the null count differs from the row count.

**W3 `multi_groupby`**: composite string+int keys and far more groups (up to
vendor × category × region), so intermediate state pressure grows.

```sql
SELECT region, category, vendor_id, SUM(value) AS total_value, COUNT(*) AS events,
       AVG(quantity) AS avg_quantity
FROM events GROUP BY region, category, vendor_id
```

- Reads: 5 columns, ~38.5 B/row.
- Batched execution: partial `(sum, count)` per key, then re-aggregate.
- Group count: DuckDB measures it in Milestone 2. It drives how large the
  partial results get.

**W4 `top_vendors`**: aggregation followed by ranking.

```sql
SELECT vendor_id, SUM(value) AS revenue
FROM events WHERE event_type = 'purchase'
GROUP BY vendor_id
ORDER BY revenue DESC, vendor_id ASC
LIMIT 100
```

- Reads: 3 columns, ~21.9 B/row.
- Batched execution: top-k per batch is **wrong**, because a vendor ranked 101st
  in every batch can be 1st overall. Batches must emit partial sums for *all*
  vendors, merge, and only then rank. `vendor_id` breaks ties, so the output is
  deterministic.

Join (W5) and time-window (W6) come after the batching engine exists. The
products table they need is already generated.

## 10. Benchmark result schema

Results are one JSON object per line, appended to `benchmarks/results/<date>-<host>.jsonl`.

Why JSON Lines:

- Appending is crash-safe.
- Files diff in git, and the schema can grow.
- DuckDB can query the files directly (`read_json('benchmarks/results/*.jsonl')`)
  for analysis.

`schema_version` guards against schema changes.

| Group | Fields |
|---|---|
| identity | `schema_version`, `run_id` (uuid), `timestamp_utc`, `git_commit`, `git_dirty`, `engine_version` |
| host | `cpu_model`, `cpu_cores_physical`, `cpu_cores_logical`, `ram_bytes`, `os`, `wsl` |
| gpu | `gpu_name`, `vram_total_bytes`, `vram_free_bytes_at_start`, `driver_version`, `cuda_driver_version`, `pcie_gen_max`, `pcie_width_max` |
| software | versions of `python`, `pyarrow`, `duckdb`, `cudf`, `rmm` |
| dataset | `dataset_path`, `storage` (`local`/`s3`), `fingerprint`, `rows_total`, `files_total`, `bytes_on_disk_total`, `decoded_bytes_per_row_est` |
| workload | `workload`, `params`, `backend` (`duckdb`/`polars`/`cudf`), `threads`, `batch_mode` (`none`/`fixed`/`adaptive`), `batch_target_bytes`, `prefetch` |
| scan | `files_scanned`, `row_groups_scanned`, `columns_scanned` (list), `rows_scanned`, `bytes_scanned_on_disk` |
| timing | `cache_state` (`warm`/`cold`), `warmup_runs`, `wall_seconds` (list, one per timed run), `wall_seconds_median`, `phases` (`read`, `decode`, `h2d`, `compute`, `merge`, `d2h`: seconds or `null` when not separable), `cuda_init_seconds` |
| resources | `peak_vram_bytes` (RMM statistics), `peak_host_rss_bytes`, `gpu_util_mean_pct`, `cpu_util_mean_pct` (sampled; `null` if unavailable) |
| batching | `batches`, `batch_rows` (list), `oom_retries` (list of `{from_bytes, to_bytes}`), `cpu_fallback` |
| result | `result_rows`, `result_checksum`, `validated` (`true`/`false`/`null`), `reference_backend`, `max_rel_error`, `tolerance` |
| status | `success`, `error_type`, `error_message` |

Derived values (throughput in rows/s, on-disk bytes/s and decoded bytes/s;
speedups) are computed at analysis time from the raw fields, never stored as
the only copy.

Failures are recorded too, with `success: false`. An OOM that happened is data.

## 11. Correctness: CPU vs GPU

1. **Normalize both results to Arrow.**
   - Cast to canonical types: keys as-is, counts to `int64`, sums and averages
     to `float64`.
   - Sort by the key columns (group-bys) or by all columns (W1).
   - W4 is already ordered.
2. **Compare column by column.**
   - Keys, counts and strings: exact.
   - Null masks: exact.
   - Floats: `|a − b| ≤ rtol · |b|` with `rtol = 1e-9`.
   - The observed `max_rel_error` is recorded with every result.
   - Floating-point sums legitimately differ with summation order (GPU
     reductions and batch merges reorder them). If 1e-9 is ever exceeded, the
     tolerance is not loosened. The failure gets investigated.
3. **Top-k near-ties.** The two result sets must agree, except where a boundary
   vendor's revenue is within tolerance of the k-th value. Each vendor's revenue
   must match within tolerance.
4. **Invariants that cost nothing.**
   - Σ `events` over groups == rows scanned.
   - Σ `total_value` over groups == a global `SUM(value)`.
   - Batched results == single-pass results on data that fits.
5. **A third opinion on small data.** On test-sized datasets, a pyarrow-compute
   implementation acts as an independent oracle. That catches the case where
   DuckDB and cuDF agree with each other and are both wrong about the intended
   semantics (e.g. nulls in `AVG`).
6. **`result_checksum`** is a SHA-256 of the normalized table, with floats
   rounded to 9 significant digits. It tracks drift within one backend across
   runs and commits. It is *not* the CPU/GPU pass/fail test, because rounding
   can straddle a boundary.

## 12. The first meaningful benchmark

**Setup:** `vendor_groupby`, single pass (no batching), DuckDB vs cuDF, on 1M,
10M, 50M and 100M rows of the same generated data.

**Why this benchmark:**

- Even 100M rows is ~1.6 GB decoded for its 3 columns, so everything fits in
  VRAM.
- The only question asked is the cleanest one: *at in-memory scale, where does
  the GPU start winning, and where does its time go?*

**Method:**

- Warm page cache.
- 1 warm-up plus 5 timed runs; report the median and the spread.
- DuckDB uses all cores.
- CUDA initialization is reported separately.
- Every run is validated.

**Timings recorded for cuDF:**

- `read_parquet(columns=[...])` (host read + transfer + GPU decode)
- the group-by itself
- the device-to-host copy of the result

**Timings recorded for DuckDB:**

- the full query
- a scan-only query over the same columns, which estimates how much of the time
  is reading

**Hypotheses** (not results):

- Fixed GPU overheads let DuckDB win at 1M rows.
- For cuDF, most time goes to the read path rather than the group-by kernel.
- cuDF's Parquet reader decompresses and decodes on the GPU. It should
  therefore move roughly the compressed size (~31 B/row across all columns)
  over PCIe, not the decoded size. Milestone 3 checks this.

## 13. Deliberately not built yet

- Any query language or parser. Workloads are named, parameterized definitions.
  A SQL front-end would be a separate project.
- Batching, adaptive sizing, OOM retry (Milestones 4–5). The GPU path must be
  correct and profiled first.
- Pruning, the `explain` command (Milestone 6). `explain` should print the plan
  object that actually executes, so it waits for the planner.
- Prefetch, pinned memory, double buffering (Milestone 7). They need a
  single-threaded baseline to compare against.
- S3 (Milestone 8), Polars, Spark, Docker, custom kernels, automatic routing.
- Charts, dashboards, telemetry daemons. The JSONL results plus a script are
  enough until Milestone 9.
- Wider "padding" columns to inflate on-disk size. This is only worth doing if
  we decide storage-size targets matter (see the open questions).

## 14. Milestone 1 tasks

| # | Task | Status |
|---|---|---|
| 1.1 | Repo skeleton: `pyproject.toml`, src layout, ruff, pytest, Makefile, `.gitignore` | done |
| 1.2 | `inspect-gpu`: host facts, NVML driver/CUDA/VRAM/PCIe, suggested RAPIDS wheel; works with no GPU and no cuDF | done |
| 1.3 | GPU smoke test (`--smoke`, `--require-gpu`) and `@pytest.mark.gpu` test | done (unrun: no GPU in build container) |
| 1.4 | Schemas and distributions (`datagen/schema.py`) | done |
| 1.5 | Planner: exact row allocation, file plan, event-id ranges | done |
| 1.6 | Per-file deterministic generation, parallel workers, atomic writes, manifest | done |
| 1.7 | `inspect`: footer-only scan, per-column sizes, sampled decoded width, corrupt/mismatched-file reporting | done |
| 1.8 | Tests: 48 CPU tests (exact counts, dense ids, schema, partition/timestamp agreement, product consistency, worker-count independence, seed sensitivity, overwrite safety, corrupt files) | done |
| 1.9 | CI: ruff + pytest on CPU (GPU tests skip) | done |
| 1.10 | **On the 4070 SUPER:** run the environment.md checks 1–5; record the `inspect-gpu --json` output | **you** |
| 1.11 | **On the 4070 SUPER:** `make dev-data`, then generate 100M rows on the NVMe disk; record generation time and `inspect` output | **you** |

Milestone 1 is complete when tasks 1.10 and 1.11 have been done on the real machine.

## Open questions before Milestone 2

1. **Windows + WSL2, or native Linux?** This changes cache-control methodology
   and some memory behaviour.
2. **How much host RAM and free NVMe space does the desktop have?** This decides
   whether large datasets run warm (page cache) or cold (disk-bound). Those are
   different experiments.
3. **What should "larger than VRAM" mean in headline claims?** Recommendation:
   the query's *projected decoded working set* versus *free* VRAM. Keep the
   11-column schema, and reach >VRAM through row count (≥ 500M rows puts W1 and
   W3 near 20 GB decoded). Widen the schema only if on-disk size targets
   (e.g. "100 GB") turn out to matter.
