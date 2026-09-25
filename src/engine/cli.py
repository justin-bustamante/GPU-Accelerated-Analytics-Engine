"""Command-line entry point: `python -m engine <command>`."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

_SUFFIXES = {"": 1, "K": 10**3, "M": 10**6, "B": 10**9, "G": 10**9}


def parse_count(text: str) -> int:
    """'100000', '100K', '10M', '1.5B' -> int."""
    m = re.fullmatch(r"\s*([0-9]*\.?[0-9]+)\s*([KMBG]?)\s*", text.upper())
    if not m:
        raise argparse.ArgumentTypeError(f"not a row count: {text!r}")
    value = float(m.group(1)) * _SUFFIXES[m.group(2)]
    if value != int(value):
        raise argparse.ArgumentTypeError(f"not a whole number of rows: {text!r}")
    return int(value)


def fmt_bytes(n: float | None) -> str:
    if n is None:
        return "n/a"
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1000 or unit == "TB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.2f} {unit}"
        n /= 1000
    raise AssertionError


def _section(title: str) -> None:
    print(f"\n{title}\n{'-' * 48}")


def _row(label: str, value) -> None:
    print(f"{label:<22}{value}")


# ---------------------------------------------------------------------------- generate


def cmd_generate(args: argparse.Namespace) -> int:
    from engine.datagen.generator import GeneratorConfig, generate
    from engine.telemetry.hardware import is_wsl

    out = Path(args.output).resolve()
    if is_wsl() and re.match(r"^/mnt/[a-z]/", str(out)):
        print(
            f"warning: {out} is on the Windows filesystem. Access from WSL2 goes through a "
            "9P bridge and is far slower than the Linux filesystem; I/O benchmarks on it "
            "will mostly measure that bridge.",
            file=sys.stderr,
        )

    cfg = GeneratorConfig(
        rows=args.rows,
        seed=args.seed,
        months=args.months,
        rows_per_file=args.rows_per_file,
        row_group_size=args.row_group_size,
        compression=args.compression,
    )
    print(
        f"generating {cfg.rows:,} rows -> {out}  (seed={cfg.seed}, workers={args.workers})",
        file=sys.stderr,
    )
    manifest = generate(cfg, out, workers=args.workers, overwrite=args.overwrite)
    secs = manifest["generation_seconds"]
    print(
        f"done: {manifest['total_rows']:,} rows, {manifest['total_files']} files, "
        f"{fmt_bytes(manifest['events_bytes'])} on disk in {secs:.1f}s  "
        f"(fingerprint {manifest['fingerprint']})"
    )
    return 0


# ------------------------------------------------------------------------- inspect-gpu


def cmd_inspect_gpu(args: argparse.Namespace) -> int:
    from engine.telemetry import hardware as hw

    report = {"host": hw.host_info(), "gpu": hw.gpu_info(), "packages": hw.package_versions()}
    if args.smoke:
        report["smoke_test"] = hw.rapids_smoke_test()

    gpu = report["gpu"]
    has_cudf = any(k.startswith("cudf-") for k in report["packages"])
    smoke = report.get("smoke_test")
    healthy = (
        gpu["available"] and bool(gpu["devices"]) and has_cudf and (smoke is None or smoke["ok"])
    )

    if args.json:
        print(json.dumps(report, indent=2))
        return 0 if healthy or not args.require_gpu else 1

    h = report["host"]
    _section("HOST")
    _row("OS", h["os"] + ("  [WSL2]" if h["wsl"] else ""))
    _row("CPU", f"{h['cpu_model']}  ({h['cpu_cores_physical']}C/{h['cpu_cores_logical']}T)")
    _row(
        "RAM",
        f"{fmt_bytes(h['ram_total_bytes'])}  ({fmt_bytes(h['ram_available_bytes'])} available)",
    )
    _row("Python", h["python"])

    _section("GPU DRIVER")
    if not gpu["available"]:
        _row("Status", f"unavailable: {gpu['error']}")
    else:
        _row("Driver", gpu["driver_version"])
        _row("Max CUDA (driver)", gpu["cuda_driver_version"] or "unknown")
        _row("RAPIDS wheel", hw.suggested_rapids_wheel(gpu["cuda_driver_version"]) or "unknown")

    for d in gpu["devices"]:
        _section(f"GPU DEVICE {d['index']}")
        _row("Name", d["name"])
        _row("Compute capability", d["compute_capability"] or "unknown")
        _row("VRAM total", fmt_bytes(d["memory_total_bytes"]))
        _row("VRAM free", fmt_bytes(d["memory_free_bytes"]))
        _row(
            "VRAM used",
            fmt_bytes(d["memory_used_bytes"])
            + ("  (this GPU is driving a display)" if d["display_active"] else ""),
        )
        if d["pcie_gen_max"]:
            _row(
                "PCIe link",
                f"gen{d['pcie_gen_current']} x{d['pcie_width_current']} now, "
                f"gen{d['pcie_gen_max']} x{d['pcie_width_max']} max",
            )
        if d["utilization_gpu_pct"] is not None:
            _row("Utilization", f"{d['utilization_gpu_pct']}%")

    _section("PACKAGES")
    for name, version in report["packages"].items():
        _row(name, version)
    if not has_cudf:
        _row("cuDF", "not installed (CPU-only environment)")

    if smoke:
        _section("GPU SMOKE TEST (cuDF group-by vs NumPy)")
        if smoke.get("error"):
            _row("Result", f"FAILED: {smoke['error']}")
        if "first_run_seconds" in smoke:
            _row("Result", "PASS" if smoke["ok"] else "FAIL")
            _row("Rows", f"{smoke['rows']:,}")
            _row("First run", f"{smoke['first_run_seconds']:.3f} s  (includes CUDA init)")
            _row("Second run", f"{smoke['second_run_seconds']:.3f} s")
            _row(
                "Free VRAM",
                f"{fmt_bytes(smoke['device_memory_free_before_bytes'])} of "
                f"{fmt_bytes(smoke['device_memory_total_bytes'])} (as seen by CUDA)",
            )

    return 0 if healthy or not args.require_gpu else 1


# -------------------------------------------------------------------------------- main


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m engine", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    g = sub.add_parser("generate", help="generate a synthetic partitioned Parquet dataset")
    g.add_argument("--rows", type=parse_count, required=True, help="e.g. 100K, 10M, 250M")
    g.add_argument("--output", required=True, help="dataset root directory")
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--months", type=int, default=24, help="months starting 2025-01 (default 24)")
    g.add_argument("--rows-per-file", type=parse_count, default=2_000_000)
    g.add_argument("--row-group-size", type=parse_count, default=1_000_000)
    g.add_argument(
        "--compression", default="snappy", choices=["snappy", "zstd", "lz4", "gzip", "none"]
    )
    g.add_argument("--workers", type=int, default=1, help="parallel generator processes")
    g.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing dataset (only one this tool generated)",
    )
    g.set_defaults(func=cmd_generate)

    d = sub.add_parser("inspect-gpu", help="report host, driver, GPU and RAPIDS status")
    d.add_argument("--smoke", action="store_true", help="also run a small cuDF group-by")
    d.add_argument(
        "--require-gpu", action="store_true", help="exit 1 unless a GPU and cuDF are usable"
    )
    d.add_argument("--json", action="store_true")
    d.set_defaults(func=cmd_inspect_gpu)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (FileExistsError, FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
