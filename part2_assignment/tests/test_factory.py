# tests/test_factory.py

import json
import subprocess
import sys
from pathlib import Path

import pytest


BASE_DIR = Path(__file__).resolve().parents[1]
SCRIPT_PATH = BASE_DIR / "factory" / "main.py"
TEST_DATA_FOLDER = Path(__file__).resolve().parent / "data"


def execute_production(test_case: str) -> dict:
    input_file = TEST_DATA_FOLDER / test_case
    process = subprocess.run(
        [sys.executable, str(SCRIPT_PATH)],
        input=input_file.read_text(),
        text=True,
        capture_output=True,
        check=True,
    )
    return json.loads(process.stdout)


def parse_config(test_case: str) -> dict:
    return json.loads((TEST_DATA_FOLDER / test_case).read_text())


def test_production_optimal():
    config = parse_config("factory_sample.json")
    outcome = execute_production("factory_sample.json")
    assert outcome["status"] == "ok"
    method_rates = outcome["per_recipe_crafts_per_min"]
    enhancements = config.get("modules", {})
    goal_recipe = config["recipes"]["green_circuit"]
    device = goal_recipe["machine"]
    yield_multiplier = 1.0 + float(enhancements.get(device, {}).get("prod", 0.0))
    computed_yield = method_rates["green_circuit"] * yield_multiplier
    assert pytest.approx(computed_yield, rel=1e-6) == config["target"]["rate_per_min"]

    resource_usage = outcome["raw_consumption_per_min"]
    for item, cap in config["limits"]["raw_supply_per_min"].items():
        assert resource_usage[item] <= cap + 1e-6


def test_production_unachievable_indicates_limits():
    input_data = parse_config("factory_infeasible.json")
    result = execute_production("factory_infeasible.json")
    assert result["status"] == "infeasible"
    assert result["max_feasible_target_per_min"] < input_data["target"]["rate_per_min"]
    clues = set(result["bottleneck_hint"])
    assert "copper_ore supply" in clues