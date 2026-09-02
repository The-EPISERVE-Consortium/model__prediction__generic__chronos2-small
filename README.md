# model__prediction__generic__chronos2-small

Zero-shot time-series forecasting with
[Chronos-2 small](https://huggingface.co/autogluon/chronos-2-small). Takes a
parquet with a leading datetime x column and one or more numeric y columns,
and forecasts one configured `target_column` — using the other y columns as
historical multivariate context — with 10th/90th-percentile bounds.

**In:** `input.parquet` + `config.json` under `/work/input/`.
**Out:** `predictions.tsv` under `/work/output/`.

## Input

| Path | Description |
|---|---|
| `/work/input/input.parquet` | ≥ 2 columns: x (first, datetime-like) and one or more numeric y columns. |
| `/work/input/config.json` | Run parameters — see [Config](#config). |

**x column** — the first column. Must be unique, monotonic increasing, and
evenly spaced. String date columns are parsed as datetimes; future timestamps
continue the inferred frequency / step.

**y columns** — every column after x, all numeric. `config.target_column` is
the one that gets forecast; the rest are passed to Chronos-2 as historical
context. Interior NaN gaps of ≤ 8 rows are interpolated per column; longer
interior gaps are rejected. Leading/trailing NaN runs are filled from the
nearest value. Needs ≥ `history_length + prediction_offset` rows.

## Output

`/work/output/predictions.tsv` — one row per forecast step:

| Column | Description |
|---|---|
| `<x_col>` | Extrapolated future timestamp. |
| `<target_column>` | Point forecast. |
| `<target_column>_q10` / `<target_column>_q90` | 10th / 90th percentile. |

## Config

Parameters are declared in [`fdo.json`](fdo.json) (`additionalProperty`) — the
source of truth for names, whether each is required, defaults, and
`minValue`/`maxValue`. `src/run.py` validates `config.json` against it at
startup (unknown keys warn and are ignored); cross-field rules it can't
express (e.g. `history_length + prediction_offset` ≤ input rows) are checked
once the data loads.

`target_column` names the y column to forecast. `prediction_offset` skips rows
at the end before the history window — set it to `prediction_length` to
forecast over already-known rows for back-testing.

### Checking a config without running the model

```bash
docker run --rm -v $(pwd)/work/input:/work/input <image> --check-config
```

Validates `/work/input/config.json` against `fdo.json` and exits `0`
(printing `config.json OK`) or `1` with `ERROR:` lines. It reads no input
data and does not load the Chronos-2 weights. Locally:
`WORK_DIR=./work python src/run.py --check-config`.

## Develop

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt -r requirements-test.txt
pytest tests/ -v          # Chronos-2 is mocked — no model download

docker build -t episerve/generic-chronos2-small:dev .
docker run --rm -v $(pwd)/work/input:/work/input -v $(pwd)/work/output:/work/output \
  episerve/generic-chronos2-small:dev
```

Chronos-2 small weights (~120 MB) download from Hugging Face on the first real run.

## Release

Push to `main` → CI builds, tests inside the image, and pushes `:latest` to
GHCR. `git tag v0.1.0 && git push origin v0.1.0` also publishes the pinned
`:v0.1.0` image.

## Limitations

- **Chronos output-column names are an assumed contract.** `predict()` finds
  the point / q10 / q90 columns by matching known names (`predictions`,
  `0.1`, `0.9`, …); a Chronos release that renames them makes the run fail
  with a clear `ERROR:` (not silently). The tests mock Chronos, so this
  mapping is not verified against the real model.
- **Build is not pinned.** Dependencies and the Docker base image are
  unpinned.
