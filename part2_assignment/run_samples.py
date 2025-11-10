#!/usr/bin/env python3
"""
Helper – run both tools on the bundled sample and show side-by-side.
Usage:
    python run_samples.py "python factory/main.py" "python belts/main.py"
"""
import json, subprocess, sys, textwrap, pathlib

def execute_tool(command: str, input_data: str):
    process = subprocess.run(command.split(), input=input_data.encode(), capture_output=True, timeout=5)
    if process.returncode:
        print(process.stderr.decode())
        sys.exit(process.returncode)
    return json.loads(process.stdout)

production_runner, transport_runner = sys.argv[1:3]

# Read belts sample from test file (it has embedded SAMPLE string)
transport_input = pathlib.Path(__file__).with_name("tests").joinpath("test_belts.py").read_text().split('SAMPLE = """')[1].split('"""')[0]
# Read factory sample from JSON file
production_input = pathlib.Path(__file__).with_name("tests").joinpath("factory_sample.json").read_text()

print("Production sample →", json.dumps(execute_tool(production_runner, production_input), indent=2))
print("Transport   sample →", json.dumps(execute_tool(transport_runner, transport_input), indent=2))