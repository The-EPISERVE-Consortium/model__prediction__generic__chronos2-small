import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

CHRONOS_STUB = """
import numpy as np
import pandas as pd


class _Pipeline:
    \"\"\"Stub Chronos-2 pipeline.\"\"\"

    def predict_df(
        self,
        context_df,
        prediction_length,
        quantile_levels,
        id_column,
        timestamp_column,
        target,
    ):
        \"\"\"Return deterministic Chronos-style predictions.

        Args:
            context_df: Historical context frame.
            prediction_length: Number of forecast steps.
            quantile_levels: Requested quantile levels.
            id_column: Series ID column name.
            timestamp_column: Timestamp column name.
            target: Target column name.

        Returns:
            Data frame with point and quantile predictions.
        \"\"\"
        del context_df, quantile_levels, id_column, timestamp_column, target
        return pd.DataFrame({
            "predictions": np.full(prediction_length, 12.0),
            "0.1": np.full(prediction_length, 9.0),
            "0.5": np.full(prediction_length, 12.0),
            "0.9": np.full(prediction_length, 15.0),
        })


class Chronos2Pipeline:
    \"\"\"Stub Chronos-2 loader.\"\"\"

    @classmethod
    def from_pretrained(cls, model_id, device_map="cpu"):
        \"\"\"Return a stub Chronos pipeline.

        Args:
            model_id: Hugging Face model ID.
            device_map: Device mapping.

        Returns:
            Stub pipeline instance.
        \"\"\"
        del model_id, device_map
        return _Pipeline()
"""


def _write_work_input(tmp_path, config, df=None):
    """Write a temporary work/input tree for CLI tests.

    Args:
        tmp_path: Temporary directory provided by pytest.
        config: Config dictionary to write.
        df: Optional input data frame to write as parquet.

    Returns:
        Work directory path.
    """
    work_dir = tmp_path / "work"
    input_dir = work_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    (input_dir / "config.json").write_text(json.dumps(config))
    if df is not None:
        df.to_parquet(input_dir / "input.parquet")
    (tmp_path / "chronos.py").write_text(CHRONOS_STUB)
    return work_dir


def _run_cli(tmp_path, work_dir, args=()):
    """Run src/run.py in a subprocess with a stub chronos module.

    Args:
        tmp_path: Temporary directory containing the stub chronos module.
        work_dir: Work directory to expose through WORK_DIR.
        args: Extra command-line arguments for run.py.

    Returns:
        Completed subprocess result.
    """
    env = os.environ.copy()
    env["PYTHONPATH"] = str(tmp_path)
    env["WORK_DIR"] = str(work_dir)
    return subprocess.run(
        [sys.executable, str(REPO_ROOT / "src" / "run.py"), *args],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def _make_df(n=20):
    """Create a sample multivariate CLI input data frame.

    Args:
        n: Number of rows to create.

    Returns:
        Data frame with date, incidence, temperature, and mobility columns.
    """
    return pd.DataFrame({
        "date": pd.date_range("2022-01-01", periods=n, freq="D"),
        "incidence": np.linspace(10.0, 20.0, n),
        "temperature": np.linspace(4.0, 14.0, n),
        "mobility": np.linspace(80.0, 120.0, n),
    })


def test_check_config_accepts_valid_config_without_data(tmp_path):
    """--check-config validates config.json without reading input.parquet."""
    work_dir = _write_work_input(tmp_path, {
        "target_column": "temperature",
        "history_length": 10,
        "prediction_length": 5,
    })
    result = _run_cli(tmp_path, work_dir, args=("--check-config",))
    assert result.returncode == 0, result.stderr
    assert "config.json OK" in result.stdout
    assert "input.parquet" not in result.stderr


def test_missing_required_config_fails_before_data(tmp_path):
    """Missing required config keys produce clean errors before data loading."""
    work_dir = _write_work_input(tmp_path, {
        "history_length": 10,
        "prediction_length": 5,
    })
    result = _run_cli(tmp_path, work_dir)
    assert result.returncode == 1
    assert "missing required key: 'target_column'" in result.stderr
    assert "input.parquet" not in result.stderr


def test_missing_input_reports_resolved_path(tmp_path):
    """Missing parquet errors include the resolved input path."""
    work_dir = _write_work_input(tmp_path, {
        "target_column": "temperature",
        "history_length": 10,
        "prediction_length": 5,
    })
    result = _run_cli(tmp_path, work_dir)
    assert result.returncode == 1
    assert f"{work_dir.resolve()}/input/input.parquet not found" in result.stderr


def test_cli_writes_target_only_predictions(tmp_path):
    """CLI writes predictions for the configured target column only."""
    work_dir = _write_work_input(tmp_path, {
        "target_column": "temperature",
        "history_length": 10,
        "prediction_length": 3,
    }, df=_make_df())
    result = _run_cli(tmp_path, work_dir)
    assert result.returncode == 0, result.stderr

    output = pd.read_csv(work_dir / "output" / "predictions.tsv", sep="\t")
    assert list(output.columns) == [
        "date",
        "temperature",
        "temperature_q10",
        "temperature_q90",
    ]
    assert len(output) == 3


def test_prediction_offset_selects_history_window(tmp_path):
    """prediction_offset skips rows at the end before choosing history."""
    work_dir = _write_work_input(tmp_path, {
        "target_column": "temperature",
        "history_length": 5,
        "prediction_length": 2,
        "prediction_offset": 3,
    }, df=_make_df(n=20))
    result = _run_cli(tmp_path, work_dir)
    assert result.returncode == 0, result.stderr
    assert "Using rows 12-16" in result.stdout


def test_rejects_too_few_columns(tmp_path):
    """input.parquet must contain x plus at least one y column."""
    work_dir = _write_work_input(tmp_path, {
        "target_column": "temperature",
        "history_length": 3,
        "prediction_length": 2,
    }, df=pd.DataFrame({"date": pd.date_range("2022-01-01", periods=5, freq="D")}))
    result = _run_cli(tmp_path, work_dir)
    assert result.returncode == 1
    assert "at least 2 columns" in result.stderr


def test_rejects_not_enough_rows(tmp_path):
    """history_length plus prediction_offset cannot exceed input rows."""
    work_dir = _write_work_input(tmp_path, {
        "target_column": "temperature",
        "history_length": 20,
        "prediction_length": 2,
        "prediction_offset": 1,
    }, df=_make_df(n=20))
    result = _run_cli(tmp_path, work_dir)
    assert result.returncode == 1
    assert "not enough input data" in result.stderr


def test_model_runtime_error_exits_cleanly_without_traceback(tmp_path):
    """Non-ValueError model failures are rendered as clean ERROR lines."""
    work_dir = _write_work_input(tmp_path, {
        "target_column": "temperature",
        "history_length": 10,
        "prediction_length": 2,
    }, df=_make_df())
    (tmp_path / "chronos.py").write_text(
        CHRONOS_STUB.replace(
            "return pd.DataFrame({",
            "raise RuntimeError('simulated chronos failure')\n        return pd.DataFrame({",
        )
    )
    result = _run_cli(tmp_path, work_dir)
    assert result.returncode == 1
    assert "ERROR: model failed" in result.stderr
    assert "Traceback" not in result.stderr


@pytest.mark.parametrize(
    ("config", "message"),
    [
        ({"target_column": "temperature", "history_length": 2, "prediction_length": 1}, "'history_length' must be >= 3"),
        ({"target_column": "temperature", "history_length": 3, "prediction_length": 0}, "'prediction_length' must be >= 1"),
        ({"target_column": "temperature", "history_length": 3, "prediction_length": 1, "prediction_offset": -1}, "'prediction_offset' must be >= 0"),
    ],
)
def test_invalid_lengths_exit_before_loading_data(tmp_path, config, message):
    """Invalid length config values fail before input data is read."""
    work_dir = _write_work_input(tmp_path, config)
    result = _run_cli(tmp_path, work_dir)
    assert result.returncode == 1
    assert message in result.stderr
    assert "input.parquet not found" not in result.stderr
