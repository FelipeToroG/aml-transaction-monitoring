"""Tests for CostMatrix.from_yaml consistency enforcement.

from_yaml recomputes the derived FN/FP costs from their documented source
inputs and hard-fails when a stored derived value has drifted. In a
compliance system an internally inconsistent cost config must not load
silently, so the drift is surfaced as a ValueError at load time.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.evaluation.metrics import CostMatrix

CONFIG_PATH = "configs/cost_matrix.yaml"


def _load_config_dict() -> dict:
    return yaml.safe_load(Path(CONFIG_PATH).read_text())


def test_shipped_config_loads_consistent():
    """The committed cost matrix is internally consistent and loads.

    Also exercises the one-cent tolerance: the stored FP (22.17) is the
    rounded form of 95 * 14 / 60 = 22.1666..., so a strict equality check
    would false-fail here.
    """
    cm = CostMatrix.from_yaml(CONFIG_PATH)
    assert cm.false_negative_cost_usd == 16000.0
    assert cm.false_positive_cost_usd == 22.17


def test_tampered_fn_cost_raises(tmp_path):
    """A stored FN cost that no longer matches its source inputs hard-fails."""
    cfg = _load_config_dict()
    # Source inputs still imply 8500 + 25000 * 0.30 = 16000; this stored
    # value contradicts them by far more than a cent.
    cfg["derived"]["false_negative_cost_usd"] = 99999.0
    tampered = tmp_path / "cost_matrix.yaml"
    tampered.write_text(yaml.safe_dump(cfg))

    with pytest.raises(ValueError, match="false_negative_cost_usd"):
        CostMatrix.from_yaml(str(tampered))


def test_tampered_fp_cost_raises(tmp_path):
    """A stored FP cost beyond the one-cent tolerance hard-fails."""
    cfg = _load_config_dict()
    cfg["derived"]["false_positive_cost_usd"] = 50.0
    tampered = tmp_path / "cost_matrix.yaml"
    tampered.write_text(yaml.safe_dump(cfg))

    with pytest.raises(ValueError, match="false_positive_cost_usd"):
        CostMatrix.from_yaml(str(tampered))


def test_drifted_source_input_raises(tmp_path):
    """Changing a source input without updating the derived value hard-fails.

    The drift can originate on either side; here the detection probability
    is bumped but the stored FN cost is left at 16000, which no longer
    reflects the inputs.
    """
    cfg = _load_config_dict()
    cfg["regulatory_cost"]["detection_probability_per_missed_case"] = 0.50
    tampered = tmp_path / "cost_matrix.yaml"
    tampered.write_text(yaml.safe_dump(cfg))

    with pytest.raises(ValueError, match="false_negative_cost_usd"):
        CostMatrix.from_yaml(str(tampered))
