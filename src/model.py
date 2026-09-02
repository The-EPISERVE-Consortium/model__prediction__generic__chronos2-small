from datetime import date

import numpy as np
import pandas as pd
from chronos import Chronos2Pipeline

_pipeline = None

_MODEL_ID = "autogluon/chronos-2-small"
_SERIES_ID = "series_0"
_ID_COLUMN = "item_id"
_MAX_CONTEXT = 8192
_MIN_CONTEXT = 3
_MAX_PREDICTION_STEPS = 1024
_MAX_INTERPOLATION_GAP = 8
_QUANTILE_LEVELS = [0.1, 0.5, 0.9]


def _validate_ordered_x(x_series: pd.Series) -> None:
    """Validate timestamp values that must represent an ordered series.

    Args:
        x_series: Observed timestamp values.

    Raises:
        ValueError: If x values contain missing, duplicate, or non-monotonic
            entries.
    """
    if x_series.isna().any():
        raise ValueError("x values must not contain NaN")
    if not x_series.is_unique:
        raise ValueError("x values must be unique")
    if not x_series.is_monotonic_increasing:
        raise ValueError("x values must be monotonic increasing")


def _validate_uniform_step(diffs: pd.Series) -> None:
    """Validate that timestamp diffs have a consistent positive step.

    Args:
        diffs: Consecutive differences between timestamp values.

    Raises:
        ValueError: If diffs are not positive and uniform.
    """
    first_diff = diffs.iloc[0]
    diff_ns = diffs.map(lambda value: pd.Timedelta(value).value).to_numpy(dtype=np.float64)
    first_diff_ns = float(pd.Timedelta(first_diff).value)
    if first_diff_ns <= 0:
        raise ValueError("x values must have a positive step")
    if not np.allclose(diff_ns, first_diff_ns, rtol=0, atol=1):
        raise ValueError("x values must have a uniform step")


def _coerce_datetime_x(x_series: pd.Series) -> pd.Series:
    """Convert x values to pandas datetimes.

    Args:
        x_series: Observed x-axis values.

    Returns:
        Datetime-converted x series.

    Raises:
        ValueError: If x values are not datetime-like.
    """
    if pd.api.types.is_datetime64_any_dtype(x_series):
        return x_series

    if pd.api.types.is_object_dtype(x_series) or pd.api.types.is_string_dtype(x_series):
        values = x_series.dropna()
        if values.empty:
            raise ValueError("x values must contain at least one timestamp")
        datetime_types = (date, np.datetime64, pd.Timestamp)
        if values.map(lambda value: isinstance(value, datetime_types)).all():
            return pd.to_datetime(x_series)
        converted = pd.to_datetime(x_series, errors="coerce")
        if converted.notna().all():
            return converted

    raise ValueError("x column must contain datetime values")


def _datetime_step(x_series: pd.Series):
    """Infer the datetime step from a validated datetime x series.

    Args:
        x_series: Datetime x-axis values.

    Returns:
        Date offset or timedelta used to extrapolate future timestamps.

    Raises:
        ValueError: If the datetime values do not have an inferable calendar
            frequency or uniform timedelta step.
    """
    freq = pd.infer_freq(pd.DatetimeIndex(x_series))
    if freq is not None:
        return pd.tseries.frequencies.to_offset(freq)

    diffs = x_series.diff().dropna()
    _validate_uniform_step(diffs)
    return diffs.iloc[0]


def _extrapolate_x(x_series: pd.Series, n: int) -> list:
    """Return future timestamps extrapolated from the observed x series.

    Args:
        x_series: Observed datetime x-axis values.
        n: Number of future values to generate.

    Returns:
        Future timestamp values.

    Raises:
        ValueError: If fewer than two x values are provided or the step cannot
            be inferred.
    """
    if len(x_series) < 2:
        raise ValueError("x column must contain at least 2 values to infer step size")
    _validate_ordered_x(x_series)
    step = _datetime_step(x_series)
    last = x_series.iloc[-1]
    return [last + step * (i + 1) for i in range(n)]


def _validate_numeric_columns(df: pd.DataFrame, columns: list[str]) -> None:
    """Validate that selected data columns are numeric.

    Args:
        df: Input data frame.
        columns: Columns that must contain numeric values.

    Raises:
        ValueError: If any selected column is not numeric.
    """
    for column in columns:
        if not pd.api.types.is_numeric_dtype(df[column]):
            raise ValueError(f"column '{column}' must be numeric")


def _longest_interior_nan_run(series: pd.Series) -> int:
    """Return the longest consecutive interior NaN run.

    Leading and trailing NaN runs are excluded because the nearest observed
    value can be carried over them without interpolation.

    Args:
        series: Series to inspect for missing values.

    Returns:
        Length of the longest interior run of missing values.
    """
    values = series.to_numpy()
    valid_positions = np.flatnonzero(~pd.isna(values))
    if len(valid_positions) < 2:
        return 0
    first_valid, last_valid = int(valid_positions[0]), int(valid_positions[-1])
    longest = 0
    current = 0
    for i in range(first_valid + 1, last_valid):
        if pd.isna(values[i]):
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def _fill_numeric_series(series: pd.Series) -> pd.Series:
    """Fill supported missing values in one numeric input series.

    Args:
        series: Numeric input series.

    Returns:
        Float64 series with missing values filled.

    Raises:
        ValueError: If the series is all missing or has a long interior gap.
    """
    name = series.name if series.name is not None else "value"
    numeric = series.astype(np.float64)
    if numeric.isna().all():
        raise ValueError(f"column '{name}' must contain at least one non-NaN value")
    longest_run = _longest_interior_nan_run(numeric)
    if longest_run > _MAX_INTERPOLATION_GAP:
        raise ValueError(
            f"column '{name}' has a NaN gap longer than "
            f"{_MAX_INTERPOLATION_GAP} consecutive values ({longest_run})"
        )
    return (
        numeric
        .interpolate(method="linear", limit=_MAX_INTERPOLATION_GAP, limit_direction="both")
        .bfill()
        .ffill()
    )


def _model():
    """Return the cached Chronos-2 pipeline instance.

    Returns:
        Chronos2Pipeline: Loaded Chronos-2 small pipeline running on CPU.
    """
    global _pipeline
    if _pipeline is None:
        _pipeline = Chronos2Pipeline.from_pretrained(_MODEL_ID, device_map="cpu")
    return _pipeline


def _prepare_context_df(
    df: pd.DataFrame,
    x_col: str,
    target_column: str,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Prepare Chronos context data from the model input frame.

    Args:
        df: Historical input data frame.
        x_col: Name of the timestamp column.
        target_column: Name of the target column to forecast.

    Returns:
        Tuple of Chronos-ready context data, coerced timestamp series, and
        numeric feature column names.

    Raises:
        ValueError: If target, timestamp, or feature columns are invalid.
    """
    if target_column not in df.columns:
        raise ValueError(f"target_column '{target_column}' is not present in input.parquet")

    feature_columns = [column for column in df.columns if column != x_col]
    if not feature_columns:
        raise ValueError("input.parquet must contain at least one y column")
    _validate_numeric_columns(df, feature_columns)

    x_series = _coerce_datetime_x(df[x_col])
    _validate_ordered_x(x_series)
    _datetime_step(x_series)

    context_df = pd.DataFrame({
        _ID_COLUMN: _SERIES_ID,
        x_col: x_series,
    })
    for column in feature_columns:
        context_df[column] = _fill_numeric_series(df[column])

    return context_df, x_series, feature_columns


def _extract_forecast_columns(pred_df: pd.DataFrame) -> tuple[str, str, str]:
    """Find Chronos output columns for median and interval forecasts.

    Args:
        pred_df: Data frame returned by Chronos.

    Returns:
        Tuple of point, q10, and q90 column names.

    Raises:
        ValueError: If expected forecast columns are missing.
    """
    point_candidates = ("predictions", "0.5", 0.5, "mean")
    q10_candidates = ("0.1", 0.1, "q0.1", "q10")
    q90_candidates = ("0.9", 0.9, "q0.9", "q90")

    def first_present(candidates):
        """Return the first candidate column present in the forecast frame.

        Args:
            candidates: Candidate column names.

        Returns:
            Matching column name or ``None``.
        """
        for candidate in candidates:
            if candidate in pred_df.columns:
                return candidate
        return None

    point_col = first_present(point_candidates)
    q10_col = first_present(q10_candidates)
    q90_col = first_present(q90_candidates)
    if point_col is None or q10_col is None or q90_col is None:
        raise ValueError(
            "Chronos forecast output is missing expected prediction, 0.1, or 0.9 columns: "
            f"{list(pred_df.columns)}"
        )
    return point_col, q10_col, q90_col


def predict(
    df: pd.DataFrame,
    target_column: str,
    prediction_length: int,
) -> pd.DataFrame:
    """Forecast one target column from multivariate historical context.

    Args:
        df: Historical data with x in the first column and one or more numeric
            y columns after it.
        target_column: Name of the y column to forecast.
        prediction_length: Number of steps ahead to predict.

    Returns:
        DataFrame with future x values, point forecasts, and q10/q90 bounds for
        the selected target column.

    Raises:
        ValueError: If the input data, target column, or prediction length is
            invalid, or if Chronos returns an unexpected output shape.
    """
    if prediction_length <= 0:
        raise ValueError("prediction_length must be > 0")
    if prediction_length > _MAX_PREDICTION_STEPS:
        raise ValueError(
            f"prediction_length={prediction_length} exceeds max {_MAX_PREDICTION_STEPS} steps"
        )
    if len(df) < _MIN_CONTEXT:
        raise ValueError(
            f"input history must contain at least {_MIN_CONTEXT} points "
            f"(got {len(df)}) to forecast"
        )
    if len(df.columns) < 2:
        raise ValueError("input.parquet must contain x plus at least one y column")

    x_col = df.columns[0]
    context_df, x_series, _ = _prepare_context_df(df, x_col, target_column)
    future_x = _extrapolate_x(x_series, prediction_length)

    pred_df = _model().predict_df(
        context_df,
        prediction_length=prediction_length,
        quantile_levels=_QUANTILE_LEVELS,
        id_column=_ID_COLUMN,
        timestamp_column=x_col,
        target=target_column,
    )
    if len(pred_df) != prediction_length:
        raise ValueError(
            f"Chronos forecast returned {len(pred_df)} rows; expected {prediction_length}"
        )

    point_col, q10_col, q90_col = _extract_forecast_columns(pred_df)
    return pd.DataFrame({
        x_col: future_x,
        target_column: pred_df[point_col].to_numpy(dtype=np.float64),
        f"{target_column}_q10": pred_df[q10_col].to_numpy(dtype=np.float64),
        f"{target_column}_q90": pred_df[q90_col].to_numpy(dtype=np.float64),
    })
