# GPU-Accelerated Analytics Engine

A small, local-first analytics engine for streaming large Parquet datasets
through a single consumer GPU (NVIDIA RTX 4070 SUPER, 12 GB), and for measuring
honestly when that beats a strong CPU engine and when it doesn't.

## Why

While working with large-scale analytics pipelines, I became interested in how
storage layout and compute choices affect performance:

- How much data really needs to be scanned?
- How much time goes to moving data rather than computing on it?
- What happens when a workload outgrows one memory tier?

NVIDIA Summer Bridge later introduced me to GPU computing, and I wanted to
experiment with accelerated computing on the RTX 4070 SUPER already in my
desktop, outside of the usual ML workloads.

The plan is to start by benchmarking ordinary analytics operations (filters,
group-bys, top-k, joins) on the GPU against DuckDB. The more interesting problem
starts when the data no longer fits in 12 GB of VRAM. At that point the engine
has to decide:

- what to read;
- how much to put on the GPU at once;
- how to keep the GPU busy while the CPU prepares the next batch;
- whether to use the GPU at all.

## Status

| Milestone | Scope | State |
|---|---|---|
| 1 | Environment validation, GPU detection, synthetic Parquet generator | code done; awaiting a run on the 4070 SUPER |
| 2 | DuckDB baseline, benchmark runner, correctness framework | not started |
| 3 | cuDF implementations, CPU/GPU validation, first comparison | not started |
| 4 | Out-of-core fixed batching | not started |
| 5 | Adaptive, memory-aware batching, OOM recovery | not started |
| 6 | Partition and column pruning, `explain` | not started |
| 7 | Pipelined (prefetching) execution | not started |
| 8 | S3-backed Parquet | not started |
| 9 | Benchmark matrix and written findings | not started |
| 10 | Benchmark-informed CPU/GPU routing | maybe |

**There are no GPU benchmark results yet.** Numbers appear in
`docs/findings.md` only after they are measured, validated against the CPU, and
reproducible. That includes the cases where the CPU wins.

## Quick start

```bash
uv venv -p 3.12 .venv && source .venv/bin/activate
uv pip install -e ".[dev]"
python -m engine inspect-gpu              # driver, CUDA version, VRAM, which cuDF wheel to use
uv pip install -e ".[dev,gpu-cu13]"       # or gpu-cu12; see docs/environment.md
python -m engine inspect-gpu --smoke --require-gpu

python -m engine generate --rows 10M --output ./data/events-10m --workers 4
python -m engine inspect --dataset ./data/events-10m
pytest
```

`inspect` reads only Parquet footers, plus one sampled row group, and reports
two sizes for the dataset:

- the size on disk;
- the estimated *decoded* size, which is what would have to fit in GPU memory.

For this schema the decoded size is about 2.6× the on-disk size (measured:
~31 B/row on disk at scale, 80.5 B/row decoded).

## Synthetic data

All data is synthetic. It has the general shape of event/reporting data:

- skewed vendor and product keys;
- low-cardinality string dimensions;
- a nullable measure;
- negative values for refunds;
- a product dimension table that events join against consistently.

Data is Hive-partitioned by `year`/`month` and generated file by file from a
seed. Memory use is bounded by file size, not dataset size. The output is
identical regardless of how many worker processes produce it. The schema is in
[docs/plan.md](docs/plan.md#7-synthetic-event-schema).

## Documentation

- [docs/plan.md](docs/plan.md): thesis, architecture, schema, workloads, benchmark
  and correctness methodology, and what is deliberately not built yet
- [docs/environment.md](docs/environment.md): WSL2 vs native Linux, choosing
  RAPIDS wheels, verifying the GPU path
