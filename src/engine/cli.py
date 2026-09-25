"""Command-line entry point: `python -m engine <command>`."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
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


# ----------------------------------------------------------------------------- inspect


def cmd_inspect(args: argparse.Namespace) -> int:
    from engine.io.metadata import scan_metadata

    t0 = time.perf_counter()
    stats = scan_metadata(args.dataset)
    scan_s = time.perf_counter() - t0

    if args.json:
        print(json.dumps(_inspect_json(stats, scan_s), indent=2, default=str))
        return 1 if stats.problems else 0

    _section(f"DATASET  {stats.root}")
    _row("Format", f"Parquet ({', '.join(sorted(stats.compression)) or 'n/a'})")
    keys = list(dict.fromkeys(k for f in stats.files for k in f.partition))
    _row("Partitions", f"{len(stats.partitions)}" + (f"  ({', '.join(keys)})" if keys else ""))
    _row("Files", f"{len(stats.files):,}")
    _row("Row groups", f"{stats.row_groups:,}")
    _row("Rows", f"{stats.rows:,}")
    _row("Size on disk", fmt_bytes(stats.bytes_on_disk))
    if stats.rows:
        _row("  per row", f"{stats.bytes_on_disk / stats.rows:.1f} B")
    if stats.decoded_bytes_per_row is not None:
        _row(
            "Decoded size (est.)",
            f"{fmt_bytes(stats.estimated_decoded_bytes)}  "
            f"({stats.decoded_bytes_per_row:.1f} B/row, sampled from one row group)",
        )
    if stats.timestamp_min is not None:
        _row("Time range", f"{stats.timestamp_min} .. {stats.timestamp_max}")
    if stats.manifest:
        m = stats.manifest
        _row(
            "Manifest",
            f"fingerprint {m['fingerprint']}, seed {m['config']['seed']}, "
            f"generator v{m['generator_version']}",
        )
        if m["total_rows"] != stats.rows:
            stats.problems.append(
                (stats.root, f"manifest says {m['total_rows']:,} rows, footers say {stats.rows:,}")
            )
    _row("Footer scan", f"{scan_s * 1000:.0f} ms")

    if stats.columns:
        _section("COLUMNS")
        total = sum(c.compressed_bytes for c in stats.columns.values()) or 1
        print(f"{'name':<14}{'type':<16}{'on disk':>12}{'share':>8}{'decoded B/row':>15}")
        for c in stats.columns.values():
            dec = f"{c.decoded_bytes_per_row:.1f}" if c.decoded_bytes_per_row is not None else "-"
            print(
                f"{c.name:<14}{c.type:<16}{fmt_bytes(c.compressed_bytes):>12}"
                f"{100 * c.compressed_bytes / total:>7.1f}%{dec:>15}"
            )

    if stats.problems:
        _section(f"PROBLEMS ({len(stats.problems)})")
        for path, msg in stats.problems:
            print(f"{path}: {msg}")
        return 1
    return 0


def _inspect_json(stats, scan_s: float) -> dict:
    return {
        "root": str(stats.root),
        "files": len(stats.files),
        "partitions": len(stats.partitions),
        "row_groups": stats.row_groups,
        "rows": stats.rows,
        "bytes_on_disk": stats.bytes_on_disk,
        "decoded_bytes_per_row_est": stats.decoded_bytes_per_row,
        "decoded_bytes_est": stats.estimated_decoded_bytes,
        "timestamp_min": stats.timestamp_min,
        "timestamp_max": stats.timestamp_max,
        "compression": sorted(stats.compression),
        "columns": [vars(c) for c in stats.columns.values()],
        "fingerprint": stats.manifest["fingerprint"] if stats.manifest else None,
        "footer_scan_seconds": scan_s,
        "problems": [{"path": str(p), "error": e} for p, e in stats.problems],
    }


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

    i = sub.add_parser("inspect", help="summarize a Parquet dataset from its footers")
    i.add_argument("--dataset", required=True)
    i.add_argument("--json", action="store_true")
    i.set_defaults(func=cmd_inspect)

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
