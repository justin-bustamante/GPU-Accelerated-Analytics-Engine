# Environment

Target machine: one desktop with an NVIDIA RTX 4070 SUPER (12 GB). The project
must always stay runnable there. Software versions below were checked against
PyPI on 2026-09-25. Re-check them before upgrading instead of trusting this page.

## What the hardware should look like (spec sheet, to be measured)

| Property | Expected | Why it matters here |
|---|---|---|
| Architecture / compute capability | Ada Lovelace / 8.9 | RAPIDS needs ≥ 7.0 |
| VRAM | 12 GB GDDR6X | Upper bound on a batch plus its working memory |
| Device memory bandwidth | ~504 GB/s | Ceiling for GPU scans and aggregations |
| Host link | PCIe 4.0 x16, ~32 GB/s theoretical each way | Ceiling for moving data onto the GPU |
| FP64 throughput | a small fraction of FP32 (consumer part) | `value` is float64; sums should be memory-bound, not FP64-bound, but that is a hypothesis |

The roughly 20× gap between device memory bandwidth and the PCIe link is the
central tension in this project. Data that crosses PCIe once and is touched once
is limited by the link and by decode, not by the GPU's compute. `inspect-gpu`
reports the negotiated PCIe link. A PCIe microbenchmark comes in Milestone 3/4.

Two practical facts about a desktop GPU:

- **It is probably also driving your monitors.** The desktop compositor, browser
  and anything else with a window hold VRAM, so *free* VRAM is less than 12 GB and
  it moves around. The engine must plan batches from measured free memory, never
  from the nameplate. Close GPU-heavy apps while benchmarking and record free VRAM
  with every result.
- **The PCIe link downshifts at idle.** NVML can report "gen1" until the card is
  under load. `inspect-gpu` shows both the current and the max link.

## Recommended OS setup

| | Native Linux (Ubuntu 24.04) | Windows 11 + WSL2 (Ubuntu 24.04) |
|---|---|---|
| RAPIDS support | Yes | Yes (single GPU, which is all we need) |
| Benchmark cleanliness | Best: direct page-cache control, no VM layer | Good, with caveats below |
| Setup effort | Dual boot / dedicated disk | Minimal if Windows is already the daily OS |

**Recommendation:** If the desktop already runs Windows, use WSL2 for
development and for most benchmarks. Treat native Linux as an optional
confirmation step for the headline numbers, if you ever want one. Native Windows
Python is not an option, because cuDF is Linux-only.

WSL2 specifics that affect this project:

1. **Keep datasets on the Linux filesystem** (for example `~/data/events`), never
   under `/mnt/c/...`. Windows drives are reached through a 9P bridge, and I/O
   benchmarks there would mostly measure that bridge. `generate` warns about this.
2. **Install the NVIDIA driver on Windows only.** Do not install a Linux display
   driver inside WSL2. The Windows driver exposes the GPU to WSL, and
   `nvidia-smi` is available inside WSL as `/usr/lib/wsl/lib/nvidia-smi`.
3. **Memory cap.** By default WSL2 gets a share of host RAM, not all of it. Set
   `memory=` in `%UserProfile%\.wslconfig` deliberately, because it decides how
   much of a dataset can sit in the page cache ("warm" runs).
4. **Unified memory oversubscription is not available on WSL2.** That is fine,
   because this project deliberately manages GPU memory explicitly (batching)
   instead of relying on managed-memory paging. Pinned host memory also has
   tighter limits under WSL2, which matters in Milestone 7.
5. **Cold-cache runs are fuzzy.** Dropping Linux caches inside WSL2 does not drop
   the Windows host's cache of the virtual disk. Report warm-cache numbers as the
   default and label any "cold" numbers carefully.

## Choosing RAPIDS wheels

No system-wide CUDA toolkit is needed. The cuDF pip wheels pull in the CUDA
user-space libraries they need (`cuda-toolkit`, `cuda-bindings`, and so on). The
only system requirement is the NVIDIA driver, and the driver decides the CUDA
major version:

1. Find the driver's CUDA version. Either run `nvidia-smi` and read the
   "CUDA Version" in its header, or run `python -m engine inspect-gpu` (see below).
   That number is the *highest* CUDA version the driver supports.
2. If it reports **13.x** (R580-series driver or newer), use `cudf-cu13`.
   `cudf-cu12` also works on such a driver.
3. If it reports **12.x**, use `cudf-cu12`, or update the driver.
4. Check the minimum driver requirements on the RAPIDS install page
   (docs.rapids.ai/install) for the release you are installing.

State of PyPI when this was written:

- `cudf-cu12` and `cudf-cu13` **26.8.1** are the latest.
- cuDF 26.08 requires Python ≥ 3.11, `numpy>=2,<3`, `pandas>=3`, and **`pyarrow>=19,<24`**.
- The newest pyarrow on PyPI is 25.x, so an unconstrained `pip install pyarrow` on
  its own would fight with cuDF.

This project caps pyarrow at `<24` in its core dependencies. That way the
CPU-only and GPU environments resolve to the same pyarrow, which keeps CPU and
GPU runs comparable.

## Setup

```bash
# inside Ubuntu 24.04 (native or WSL2)
curl -LsSf https://astral.sh/uv/install.sh | sh        # or use python -m venv + pip
git clone <this repo> && cd gpu-analytics-engine
uv venv -p 3.12 .venv && source .venv/bin/activate

uv pip install -e ".[dev]"                 # CPU-only first: no GPU packages yet
python -m engine inspect-gpu               # read "Max CUDA (driver)" and "RAPIDS wheel"

uv pip install -e ".[dev,gpu-cu13]"        # or gpu-cu12, per the step above
```

## Verifying that CUDA and cuDF can use the GPU

Run these in order. Each step isolates one layer, so a failure points at exactly one thing.

| Step | Command | Proves |
|---|---|---|
| 1 | `nvidia-smi` | The driver is loaded and sees the 4070 SUPER. The header shows the max CUDA version. |
| 2 | `python -m engine inspect-gpu` | NVML is reachable from Python: device name, compute capability, VRAM total/free, PCIe link. Needs no cuDF. |
| 3 | `python -c "import cudf; print(cudf.__version__, cudf.Series([1, 2, 3]).sum())"` | The cuDF wheels load and their CUDA runtime matches the driver. |
| 4 | `python -m engine inspect-gpu --smoke --require-gpu` | A real GPU group-by (10M rows) matches a NumPy reference. Reports first-run versus second-run time and free VRAM as CUDA sees it. Exits 1 on any failure. |
| 5 | `pytest -m gpu` | The same smoke test runs under the test suite. |

Optional cross-check: the `rapids-cli` package on PyPI provides a `rapids doctor` command.

The first-run versus second-run gap in step 4 is worth noting. The first call
pays for CUDA context creation and kernel loading. This is why every benchmark
in this project does warm-up iterations and reports CUDA initialization
separately instead of folding it into query time.

## Containers (not yet)

A container adds a layer between the profiler and the hardware and buys little
right now, because the pip wheels already pin the CUDA user-space. The Docker
route (NVIDIA Container Toolkit + a `rapidsai/base` image) is worth revisiting
for Milestone 9, when a benchmark run should be reproducible by someone else
from one command.
