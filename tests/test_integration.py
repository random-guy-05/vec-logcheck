import json
import subprocess
import sys

import anndata as ad
import numpy as np
import pandas as pd


def _write(path, matrix):
    ad.AnnData(
        X=np.asarray(matrix, dtype=np.float32),
        var=pd.DataFrame(index=[f"g{i}" for i in range(matrix.shape[1])]),
    ).write_h5ad(path)


def test_cli_flags_raw_counts_and_accepts_continuous_log_range(tmp_path):
    raw = tmp_path / "raw.h5ad"
    loglike = tmp_path / "loglike.h5ad"
    report_path = tmp_path / "report.json"

    rng = np.random.default_rng(7)
    _write(raw, rng.poisson(5, size=(160, 40)))

    continuous = rng.uniform(0, 6, size=(160, 40))
    continuous[rng.random(continuous.shape) < 0.35] = 0
    _write(loglike, continuous)

    raw_run = subprocess.run(
        [
            sys.executable,
            "-m",
            "vec_logcheck.cli",
            str(raw),
            "--json",
            str(report_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert raw_run.returncode == 4
    assert "LIKELY_RAW_COUNTS" in raw_run.stdout
    assert json.loads(report_path.read_text())["classification"] == "LIKELY_RAW_COUNTS"

    log_run = subprocess.run(
        [sys.executable, "-m", "vec_logcheck.cli", str(loglike)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert log_run.returncode == 0, log_run.stdout + log_run.stderr
    assert "NOT_OBVIOUSLY_RAW" in log_run.stdout
