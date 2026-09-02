import sys
import types
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

if "chronos" not in sys.modules:
    chronos_stub = types.ModuleType("chronos")
    chronos_stub.Chronos2Pipeline = MagicMock()
    sys.modules["chronos"] = chronos_stub

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
import model as model_module
from model import _MAX_PREDICTION_STEPS, predict


def _make_df(n=100):
    """Create a sample multivariate input data frame.

    Args:
        n: Number of rows to create.

    Returns:
        Data frame with a date column and three numeric y columns.
    """
    return pd.DataFrame({
        "date": pd.date_range("2022-01-01", periods=n, freq="D"),
        "incidence": np.linspace(10.0, 20.0, n),
        "temperature": np.linspace(4.0, 14.0, n),
        "mobility": np.linspace(80.0, 120.0, n),
    })


def _mock_predict_df(
    context_df,
    prediction_length,
    quantile_levels,
    id_column,
    timestamp_column,
    target,
):
    """Return deterministic Chronos-style forecast rows.

    Args:
        context_df: Historical context frame passed to Chronos.
        prediction_length: Number of forecast steps.
        quantile_levels: Requested quantile levels.
        id_column: Column identifying the series.
        timestamp_column: Timestamp column name.
        target: Target column name.

    Returns:
        Data frame shaped like Chronos forecast output.
    """
    del context_df, quantile_levels, id_column, timestamp_column, target
    return pd.DataFrame({
        "predictions": np.full(prediction_length, 12.0),
        "0.1": np.full(prediction_length, 9.0),
        "0.5": np.full(prediction_length, 12.0),
        "0.9": np.full(prediction_length, 15.0),
    })


@pytest.fixture(autouse=True)
def mock_chronos():
    """Mock the Chronos-2 pipeline used by the model.

    Yields:
        MagicMock: Mocked Chronos2Pipeline class.
    """
    mock_instance = MagicMock()
    mock_instance.predict_df.side_effect = _mock_predict_df
    model_module._pipeline = None
    with patch.object(model_module, "Chronos2Pipeline", create=True) as mock_cls:
        mock_cls.from_pretrained.return_value = mock_instance
        yield mock_cls
    model_module._pipeline = None


def test_output_row_count():
    """Forecast output has one row per requested step."""
    result = predict(_make_df(), "temperature", prediction_length=10)
    assert len(result) == 10


def test_output_columns():
    """Forecast output exposes x, point, q10, and q90 columns for the target."""
    result = predict(_make_df(), "temperature", prediction_length=3)
    assert list(result.columns) == ["date", "temperature", "temperature_q10", "temperature_q90"]


def test_x_extrapolation_daily():
    """Forecast timestamps continue the observed daily cadence."""
    df = _make_df(n=20)
    result = predict(df, "temperature", prediction_length=3)
    assert list(result["date"]) == list(pd.date_range("2022-01-21", periods=3, freq="D"))


def test_object_datetime_x_extrapolates_as_datetime():
    """Object columns containing datetimes are treated as timestamp columns."""
    df = pd.DataFrame({
        "date": [
            datetime(2022, 1, 1),
            datetime(2022, 1, 2),
            datetime(2022, 1, 3),
        ],
        "temperature": [1.0, 2.0, 3.0],
        "mobility": [10.0, 11.0, 12.0],
    })
    result = predict(df, "temperature", prediction_length=2)
    assert list(result["date"]) == [pd.Timestamp("2022-01-04"), pd.Timestamp("2022-01-05")]


def test_string_date_x_is_parsed_as_datetime():
    """String date columns are parsed when every value is datetime-like."""
    df = pd.DataFrame({
        "date": ["2022-01-01", "2022-01-02", "2022-01-03"],
        "temperature": [1.0, 2.0, 3.0],
        "mobility": [10.0, 11.0, 12.0],
    })
    result = predict(df, "temperature", prediction_length=2)
    assert list(result["date"]) == [pd.Timestamp("2022-01-04"), pd.Timestamp("2022-01-05")]


def test_chronos_called_with_multivariate_context(mock_chronos):
    """Chronos receives all numeric y columns, not just the target."""
    df = _make_df(n=20)
    predict(df, "temperature", prediction_length=3)

    call = mock_chronos.from_pretrained.return_value.predict_df.call_args
    context_df = call.args[0]
    assert {"incidence", "temperature", "mobility"}.issubset(context_df.columns)
    assert call.kwargs["target"] == "temperature"
    assert call.kwargs["id_column"] == "item_id"
    assert call.kwargs["timestamp_column"] == "date"


def test_model_loaded_once(mock_chronos):
    """Chronos pipeline is cached across predictions."""
    predict(_make_df(), "temperature", prediction_length=2)
    predict(_make_df(), "temperature", prediction_length=2)
    assert mock_chronos.from_pretrained.call_count == 1


def test_model_uses_chronos2_small_on_cpu(mock_chronos):
    """Chronos-2 small is loaded with CPU device mapping."""
    predict(_make_df(), "temperature", prediction_length=2)
    mock_chronos.from_pretrained.assert_called_once_with(
        "autogluon/chronos-2-small",
        device_map="cpu",
    )


def test_missing_target_raises_clear_error():
    """Missing target_column values are rejected clearly."""
    with pytest.raises(ValueError, match="target_column 'missing' is not present"):
        predict(_make_df(), "missing", prediction_length=2)


def test_non_numeric_feature_raises_clear_error():
    """All y columns must be numeric because Chronos uses them as time-series features."""
    df = _make_df(n=10)
    df["mobility"] = ["high"] * len(df)
    with pytest.raises(ValueError, match="column 'mobility' must be numeric"):
        predict(df, "temperature", prediction_length=2)


def test_all_nan_column_raises_clear_error():
    """All-NaN y columns are rejected before calling Chronos."""
    df = _make_df(n=10)
    df["mobility"] = np.nan
    with pytest.raises(ValueError, match="'mobility' must contain at least one non-NaN"):
        predict(df, "temperature", prediction_length=2)


def test_long_nan_gap_raises_clear_error():
    """Long interior NaN gaps are rejected before interpolation."""
    df = _make_df(n=20)
    df.loc[2:10, "mobility"] = np.nan
    with pytest.raises(ValueError, match="NaN gap longer than 8"):
        predict(df, "temperature", prediction_length=2)


def test_short_nan_gap_is_interpolated(mock_chronos):
    """Short interior NaN gaps are interpolated in the Chronos context frame."""
    df = _make_df(n=20)
    df.loc[5:7, "mobility"] = np.nan
    predict(df, "temperature", prediction_length=2)

    context_df = mock_chronos.from_pretrained.return_value.predict_df.call_args.args[0]
    assert context_df["mobility"].isna().sum() == 0


def test_duplicate_x_raises_clear_error():
    """Duplicate timestamps are rejected."""
    df = _make_df(n=5)
    df.loc[2, "date"] = df.loc[1, "date"]
    with pytest.raises(ValueError, match="unique"):
        predict(df, "temperature", prediction_length=2)


def test_nonmonotonic_x_raises_clear_error():
    """Non-monotonic timestamps are rejected."""
    df = _make_df(n=5)
    df.loc[2, "date"] = pd.Timestamp("2021-12-01")
    with pytest.raises(ValueError, match="monotonic"):
        predict(df, "temperature", prediction_length=2)


def test_irregular_x_raises_clear_error():
    """Irregular timestamp spacing is rejected."""
    df = _make_df(n=5)
    df["date"] = pd.to_datetime([
        "2022-01-01",
        "2022-01-02",
        "2022-01-04",
        "2022-01-05",
        "2022-01-06",
    ])
    with pytest.raises(ValueError, match="uniform step"):
        predict(df, "temperature", prediction_length=2)


def test_prediction_length_must_be_positive():
    """Zero or negative horizons are rejected."""
    with pytest.raises(ValueError, match="prediction_length must be > 0"):
        predict(_make_df(), "temperature", prediction_length=0)


def test_prediction_length_max_is_enforced():
    """Horizons above the Chronos-2 maximum are rejected."""
    with pytest.raises(ValueError, match="exceeds max"):
        predict(_make_df(), "temperature", prediction_length=_MAX_PREDICTION_STEPS + 1)


def test_unexpected_chronos_output_length_raises(mock_chronos):
    """Chronos output row count must match the requested horizon."""
    mock_chronos.from_pretrained.return_value.predict_df.return_value = pd.DataFrame({
        "predictions": [1.0],
        "0.1": [0.5],
        "0.9": [1.5],
    })
    mock_chronos.from_pretrained.return_value.predict_df.side_effect = None

    with pytest.raises(ValueError, match="returned 1 rows; expected 2"):
        predict(_make_df(), "temperature", prediction_length=2)


def test_missing_chronos_quantile_columns_raises(mock_chronos):
    """Chronos output must contain point and interval forecast columns."""
    mock_chronos.from_pretrained.return_value.predict_df.return_value = pd.DataFrame({
        "predictions": [1.0, 2.0],
    })
    mock_chronos.from_pretrained.return_value.predict_df.side_effect = None

    with pytest.raises(ValueError, match="missing expected"):
        predict(_make_df(), "temperature", prediction_length=2)
