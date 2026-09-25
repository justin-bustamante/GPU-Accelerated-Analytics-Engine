import json

import pytest

from engine.cli import main
from engine.telemetry import hardware as hw


def test_gpu_info_never_raises():
    info = hw.gpu_info()
    assert set(info) >= {"available", "devices"}
    if not info["available"]:
        assert info["error"] and info["devices"] == []


def test_inspect_gpu_reports_without_gpu(capsys):
    assert main(["inspect-gpu", "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["host"]["ram_total_bytes"] > 0
    assert "pyarrow" in report["packages"]


def test_require_gpu_fails_cleanly_without_cudf(capsys):
    if any(k.startswith("cudf-") for k in hw.package_versions()) and hw.gpu_info()["available"]:
        pytest.skip("GPU environment")
    assert main(["inspect-gpu", "--require-gpu"]) == 1


@pytest.mark.parametrize(
    "cuda,expected", [("13.0", "cudf-cu13"), ("12.8", "cudf-cu12"), ("11.8", None), (None, None)]
)
def test_suggested_rapids_wheel(cuda, expected):
    got = hw.suggested_rapids_wheel(cuda)
    assert got == expected if expected is None else got.startswith(expected)


@pytest.mark.gpu
def test_gpu_smoke():
    result = hw.rapids_smoke_test(rows=1_000_000)
    assert result["ok"], result["error"]
