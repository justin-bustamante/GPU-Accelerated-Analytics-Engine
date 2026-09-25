"""Host and GPU discovery.

GPU facts come from NVML (the library behind nvidia-smi), which only needs the
NVIDIA driver, not CUDA or cuDF. That ordering matters: the driver's supported
CUDA version decides which RAPIDS wheels (cu12 or cu13) to install, so we must be
able to read it before any RAPIDS package exists.
"""

from __future__ import annotations

import contextlib
import importlib.metadata as md
import platform
import time
from pathlib import Path

import psutil

# RAPIDS supports Volta (7.0) and newer. The RTX 4070 SUPER is Ada, 8.9.
MIN_COMPUTE_CAPABILITY = (7, 0)

TRACKED_PACKAGES = [
    "numpy",
    "pyarrow",
    "pandas",
    "duckdb",
    "polars",
    "cudf-cu12",
    "cudf-cu13",
    "rmm-cu12",
    "rmm-cu13",
    "cupy-cuda12x",
    "cupy-cuda13x",
    "nvidia-ml-py",
]


def is_wsl() -> bool:
    try:
        return "microsoft" in Path("/proc/version").read_text().lower()
    except OSError:
        return False


def cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def host_info() -> dict:
    return {
        "os": platform.platform(),
        "wsl": is_wsl(),
        "python": platform.python_version(),
        "cpu_model": cpu_model(),
        "cpu_cores_physical": psutil.cpu_count(logical=False),
        "cpu_cores_logical": psutil.cpu_count(logical=True),
        "ram_total_bytes": psutil.virtual_memory().total,
        "ram_available_bytes": psutil.virtual_memory().available,
    }


def package_versions() -> dict[str, str]:
    out = {}
    for name in TRACKED_PACKAGES:
        with contextlib.suppress(md.PackageNotFoundError):
            out[name] = md.version(name)
    return out


def _s(value) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


def _optional(fn, *args):
    """Some NVML queries are unsupported on some platforms (notably WSL2)."""
    try:
        return fn(*args)
    except Exception:
        return None


def gpu_info() -> dict:
    """Driver and per-device facts via NVML. Never raises; failures go in 'error'."""
    try:
        import pynvml as nv
    except ImportError:
        return {"available": False, "error": "nvidia-ml-py is not installed", "devices": []}
    try:
        nv.nvmlInit()
    except Exception as exc:
        return {
            "available": False,
            "error": f"NVML init failed (no NVIDIA driver?): {exc}",
            "devices": [],
        }
    try:
        cuda = _optional(nv.nvmlSystemGetCudaDriverVersion)
        info = {
            "available": True,
            "error": None,
            "driver_version": _s(nv.nvmlSystemGetDriverVersion()),
            "cuda_driver_version": f"{cuda // 1000}.{cuda % 1000 // 10}" if cuda else None,
            "devices": [],
        }
        for i in range(nv.nvmlDeviceGetCount()):
            h = nv.nvmlDeviceGetHandleByIndex(i)
            mem = nv.nvmlDeviceGetMemoryInfo(h)
            cc = _optional(nv.nvmlDeviceGetCudaComputeCapability, h)
            util = _optional(nv.nvmlDeviceGetUtilizationRates, h)
            info["devices"].append(
                {
                    "index": i,
                    "name": _s(nv.nvmlDeviceGetName(h)),
                    "memory_total_bytes": int(mem.total),
                    "memory_free_bytes": int(mem.free),
                    "memory_used_bytes": int(mem.used),
                    "compute_capability": f"{cc[0]}.{cc[1]}" if cc else None,
                    # Idle GPUs downshift the link, so "current" can read gen 1;
                    # "max" is what transfers will actually run at under load.
                    "pcie_gen_current": _optional(nv.nvmlDeviceGetCurrPcieLinkGeneration, h),
                    "pcie_gen_max": _optional(nv.nvmlDeviceGetMaxPcieLinkGeneration, h),
                    "pcie_width_current": _optional(nv.nvmlDeviceGetCurrPcieLinkWidth, h),
                    "pcie_width_max": _optional(nv.nvmlDeviceGetMaxPcieLinkWidth, h),
                    "utilization_gpu_pct": util.gpu if util else None,
                    "display_active": _optional(nv.nvmlDeviceGetDisplayActive, h),
                }
            )
        return info
    except Exception as exc:
        return {"available": False, "error": f"NVML query failed: {exc}", "devices": []}
    finally:
        _optional(nv.nvmlShutdown)


def suggested_rapids_wheel(cuda_driver_version: str | None) -> str | None:
    """cu13 wheels need a driver that supports CUDA 13; otherwise use cu12."""
    if not cuda_driver_version:
        return None
    major = int(cuda_driver_version.split(".")[0])
    if major >= 13:
        return "cudf-cu13  (cudf-cu12 also works on this driver)"
    if major == 12:
        return "cudf-cu12"
    return None


def rapids_smoke_test(rows: int = 10_000_000, groups: int = 1_000) -> dict:
    """Run a small group-by on the GPU and check it against NumPy.

    Runs twice on purpose: the first call pays CUDA context creation and kernel
    loading, the second shows steady-state cost. The gap is the reason every
    benchmark in this project does warm-up runs before timing.
    """
    try:
        import cudf
        import cupy
    except Exception as exc:
        return {"ok": False, "error": f"cuDF import failed: {exc}"}

    import numpy as np

    rng = np.random.default_rng(0)
    keys = rng.integers(0, groups, rows)
    vals = rng.random(rows)
    expected_sum = np.bincount(keys, weights=vals, minlength=groups)
    expected_count = np.bincount(keys, minlength=groups)

    def run():
        t0 = time.perf_counter()
        gdf = cudf.DataFrame({"k": keys, "v": vals})  # host -> device copy
        res = gdf.groupby("k")["v"].agg(["sum", "count"]).sort_index().to_pandas()
        return time.perf_counter() - t0, res

    try:
        free_before, total = cupy.cuda.runtime.memGetInfo()
        first_s, _ = run()
        second_s, res = run()
        free_after, _ = cupy.cuda.runtime.memGetInfo()
    except Exception as exc:
        return {"ok": False, "error": f"GPU execution failed: {exc}"}

    ok = (
        res.index.to_numpy().tolist() == list(range(groups))
        and np.array_equal(res["count"].to_numpy(), expected_count)
        and np.allclose(res["sum"].to_numpy(), expected_sum, rtol=1e-9, atol=0)
    )
    return {
        "ok": bool(ok),
        "error": None if ok else "GPU result does not match NumPy reference",
        "cudf_version": cudf.__version__,
        "rows": rows,
        "first_run_seconds": first_s,
        "second_run_seconds": second_s,
        "device_memory_total_bytes": int(total),
        "device_memory_free_before_bytes": int(free_before),
        "device_memory_free_after_bytes": int(free_after),
    }
