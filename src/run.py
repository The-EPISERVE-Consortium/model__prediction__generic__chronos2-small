import json
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config_schema import SchemaError, validate_config
from model import _MAX_CONTEXT, predict

CHECK_ONLY = "--check-config" in sys.argv[1:]

_work = Path(os.environ["WORK_DIR"]) if "WORK_DIR" in os.environ else Path("./work")
if "WORK_DIR" not in os.environ and not _work.exists():
    _work = Path("/work")
_work = _work.resolve()
INPUT = _work / "input"
OUTPUT = _work / "output"

config_path = INPUT / "config.json"
if not config_path.exists():
    print(f"ERROR: {config_path} not found", file=sys.stderr)
    sys.exit(1)

try:
    config = json.loads(config_path.read_text())
except json.JSONDecodeError as exc:
    print(f"ERROR: invalid config.json: {exc}", file=sys.stderr)
    sys.exit(1)

# Validate config.json against the model card's episerve:parameters in
# fdo.json: required keys, types, bounds, no unknown keys. Defaults for
# omitted optional keys come from the card too.
try:
    resolved, config_errors = validate_config(config)
except SchemaError as exc:
    print(f"ERROR: model card (fdo.json) parameters are invalid: {exc}", file=sys.stderr)
    sys.exit(1)

if config_errors:
    for error in config_errors:
        print(f"ERROR: {error}", file=sys.stderr)
    sys.exit(1)

history_length = resolved["history_length"]
prediction_length = resolved["prediction_length"]
prediction_offset = resolved["prediction_offset"]
print(
    "Config: "
    f"history_length={history_length}, "
    f"prediction_length={prediction_length}, "
    f"prediction_offset={prediction_offset}"
)

if history_length > _MAX_CONTEXT:
    print(
        f"WARNING: history_length={history_length} exceeds Chronos-2 max context "
        f"{_MAX_CONTEXT}; only the most recent {_MAX_CONTEXT} points will be used.",
        file=sys.stderr,
    )

if CHECK_ONLY:
    print("config.json OK")
    sys.exit(0)

data_path = INPUT / "input.parquet"
if not data_path.exists():
    print(f"ERROR: {data_path} not found", file=sys.stderr)
    sys.exit(1)

df_full = pd.read_parquet(data_path)
print(f"Loaded {len(df_full)} rows, columns: {list(df_full.columns)}")

if len(df_full.columns) < 2:
    print(
        "ERROR: input.parquet must have at least 2 columns: x (first) and "
        "the target y (second); any further columns are multivariate context",
        file=sys.stderr,
    )
    sys.exit(1)

if history_length + prediction_offset > len(df_full):
    print(
        f"ERROR: not enough input data: "
        f"history_length ({history_length}) + prediction_offset ({prediction_offset}) "
        f"= {history_length + prediction_offset} rows required, "
        f"but input.parquet only has {len(df_full)} rows. "
        f"Reduce history_length/prediction_offset or provide more input data.",
        file=sys.stderr,
    )
    sys.exit(1)

x_col = df_full.columns[0]
target_column = df_full.columns[1]
end_idx = len(df_full) - prediction_offset
start_idx = end_idx - history_length
if history_length > _MAX_CONTEXT:
    start_idx = end_idx - _MAX_CONTEXT
df = df_full.iloc[start_idx:end_idx].reset_index(drop=True)

print(f"x column: {x_col!r}, target column: {target_column!r}")
print(f"Using rows {start_idx}-{end_idx - 1} ({df[x_col].iloc[0]} - {df[x_col].iloc[-1]})")

try:
    predictions = predict(df, target_column, prediction_length)
except ValueError as exc:
    print(f"ERROR: {exc}", file=sys.stderr)
    sys.exit(1)
except Exception as exc:
    print(f"ERROR: model failed: {exc}", file=sys.stderr)
    sys.exit(1)

OUTPUT.mkdir(parents=True, exist_ok=True)
out_path = OUTPUT / "predictions.tsv"
predictions.to_csv(out_path, sep="\t", index=False)
print(f"Written {len(predictions)} rows to {out_path}")
print("Done.")
