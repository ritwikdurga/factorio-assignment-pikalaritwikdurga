# tests/test_belts.py

import json
import subprocess
import sys
from pathlib import Path

import pytest


BASE_DIR = Path(__file__).resolve().parents[1]
SCRIPT_PATH = BASE_DIR / "belts" / "main.py"
TEST_DATA_FOLDER = Path(__file__).resolve().parent / "data"


def execute_network(test_case: str) -> dict:
    input_file = TEST_DATA_FOLDER / test_case
    process = subprocess.run(
        [sys.executable, str(SCRIPT_PATH)],
        input=input_file.read_text(),
        text=True,
        capture_output=True,
        check=True,
    )
    return json.loads(process.stdout)


def test_network_optimal_capacity():
    outcome = execute_network("belts_sample.json")
    assert outcome["status"] == "ok"
    assert pytest.approx(outcome["max_flow_per_min"], rel=1e-6) == 1500.0
    transfers = {(link["from"], link["to"]): link["flow"] for link in outcome["flows"]}
    assert pytest.approx(transfers[("s1", "a")], rel=1e-6) == 900.0
    assert pytest.approx(transfers[("s2", "a")], rel=1e-6) == 600.0
    assert pytest.approx(transfers[("b", "sink")], rel=1e-6) == 900.0
    assert pytest.approx(transfers[("c", "sink")], rel=1e-6) == 600.0


def test_network_unachievable_indicates_partition():
    outcome = execute_network("belts_infeasible.json")
    assert outcome["status"] == "infeasible"
    assert outcome["max_flow_per_min"] < 1500.0
    assert outcome["deficit"]["demand_balance"] > 0
    assert outcome["deficit"]["tight_edges"], "expected tight edges in certificate"