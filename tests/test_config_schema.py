"""Tests for config schema parsing and validation."""

import json
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

if "chronos" not in sys.modules:
    chronos_stub = types.ModuleType("chronos")
    chronos_stub.Chronos2Pipeline = MagicMock()
    sys.modules["chronos"] = chronos_stub

sys.path.insert(0, str(REPO_ROOT / "src"))

import model  # noqa: E402
from config_schema import ParamSpec, SchemaError, load_specs, validate_config  # noqa: E402


def test_fdo_json_declares_expected_params():
    """fdo.json declares the supported config parameters in order."""
    names = [spec.name for spec in load_specs()]
    assert names == [
        "target_column",
        "history_length",
        "prediction_length",
        "prediction_offset",
    ]


def test_fdo_bounds_match_model_constants():
    """fdo.json bounds agree with the model constants."""
    specs = {spec.name: spec for spec in load_specs()}
    assert specs["target_column"].value_type == "string"
    assert specs["history_length"].min_value == model._MIN_CONTEXT
    assert specs["history_length"].max_value is None
    assert specs["prediction_length"].min_value == 1
    assert specs["prediction_length"].max_value == model._MAX_PREDICTION_STEPS
    assert specs["prediction_offset"].min_value == 0
    assert specs["prediction_offset"].default == 0


def test_valid_config_fills_default_offset():
    """A valid config receives the default prediction_offset."""
    resolved, errors, warnings = validate_config({
        "target_column": "temperature",
        "history_length": 20,
        "prediction_length": 10,
    })
    assert errors == []
    assert warnings == []
    assert resolved["prediction_offset"] == 0


def test_explicit_offset_is_kept():
    """A valid explicit prediction_offset is preserved."""
    resolved, errors, _ = validate_config({
        "target_column": "temperature",
        "history_length": 20,
        "prediction_length": 10,
        "prediction_offset": 4,
    })
    assert errors == []
    assert resolved["prediction_offset"] == 4


@pytest.mark.parametrize(
    ("config", "needle"),
    [
        ({"history_length": 10, "prediction_length": 5}, "missing required key: 'target_column'"),
        ({"target_column": "temperature", "prediction_length": 5}, "missing required key: 'history_length'"),
        ({"target_column": "temperature", "history_length": 10}, "missing required key: 'prediction_length'"),
        ({"target_column": 1, "history_length": 10, "prediction_length": 5}, "'target_column' must be a string"),
        ({"target_column": "temperature", "history_length": 2, "prediction_length": 5}, "'history_length' must be >= 3"),
        ({"target_column": "temperature", "history_length": 10, "prediction_length": 0}, "'prediction_length' must be >= 1"),
        ({"target_column": "temperature", "history_length": 10, "prediction_length": 1025}, "'prediction_length' must be <= 1024"),
        ({"target_column": "temperature", "history_length": 10, "prediction_length": 5, "prediction_offset": -1}, "'prediction_offset' must be >= 0"),
    ],
)
def test_invalid_config_reports_clear_error(config, needle):
    """Invalid config values return clear validation errors."""
    _, errors, _ = validate_config(config)
    assert any(needle in error for error in errors), errors


def test_unknown_key_is_a_warning_not_an_error():
    """Unknown config keys are warnings and are not resolved."""
    resolved, errors, warnings = validate_config({
        "target_column": "temperature",
        "history_length": 10,
        "prediction_length": 5,
        "extra": 3,
    })
    assert errors == []
    assert any("extra" in warning for warning in warnings)
    assert "extra" not in resolved


def test_multiple_errors_are_all_reported():
    """Validation returns all independent per-key errors."""
    _, errors, _ = validate_config({
        "target_column": "temperature",
        "history_length": 1,
        "prediction_length": 99999,
    })
    assert len(errors) == 2


def test_load_specs_rejects_non_list_additional_property(tmp_path):
    """load_specs rejects non-list additionalProperty values."""
    bad = tmp_path / "fdo.json"
    bad.write_text(json.dumps({"additionalProperty": "nope"}))
    with pytest.raises(SchemaError):
        load_specs(bad)


def test_load_specs_rejects_missing_file(tmp_path):
    """load_specs rejects a missing fdo.json path."""
    with pytest.raises(SchemaError):
        load_specs(tmp_path / "does-not-exist.json")


def test_load_specs_rejects_invalid_json(tmp_path):
    """load_specs rejects invalid JSON."""
    bad = tmp_path / "fdo.json"
    bad.write_text("{ not json")
    with pytest.raises(SchemaError):
        load_specs(bad)


def test_load_specs_rejects_duplicate_names(tmp_path):
    """load_specs rejects duplicate parameter names."""
    bad = tmp_path / "fdo.json"
    bad.write_text(json.dumps({"additionalProperty": [
        {"name": "x", "valueRequired": True},
        {"name": "x", "valueRequired": False},
    ]}))
    with pytest.raises(SchemaError):
        load_specs(bad)


@pytest.mark.parametrize("entry", [
    {"valueRequired": True},
    {"name": "x", "valueRequired": "yes"},
    {"name": "x", "valueType": "date"},
    {"name": "x", "minValue": 1.5},
    {"name": "x", "minValue": 10, "maxValue": 1},
    {"name": "x", "valueRequired": True, "value": 0},
    {"name": "x", "valueType": "string", "value": 0},
])
def test_param_spec_rejects_malformed_entry(entry):
    """ParamSpec rejects malformed parameter declarations."""
    with pytest.raises(SchemaError):
        ParamSpec(entry)
