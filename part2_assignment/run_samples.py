#!/usr/bin/env python3
"""
Helper – run both tools on the bundled sample and show side-by-side.
Usage:
    python run_samples.py "python factory/main.py" "python belts/main.py"
"""
import json, subprocess, sys, textwrap, pathlib

def execute_tool(command: str, input_data: str, timeout: float = 30.0):
    process = subprocess.run(
        command.split(),
        input=input_data.encode(),
        capture_output=True,
        timeout=timeout,
    )
    if process.returncode:
        print(process.stderr.decode())
        sys.exit(process.returncode)
    return json.loads(process.stdout)

production_runner, transport_runner = sys.argv[1:3]

# Read sample inputs from the pytest fixtures
data_dir = pathlib.Path(__file__).with_name("tests") / "data"
transport_input = data_dir.joinpath("belts_sample.json").read_text()
production_input = data_dir.joinpath("factory_sample.json").read_text()

print("Production sample →", json.dumps(execute_tool(production_runner, production_input), indent=2))
print("Transport   sample →", json.dumps(execute_tool(transport_runner, transport_input), indent=2))