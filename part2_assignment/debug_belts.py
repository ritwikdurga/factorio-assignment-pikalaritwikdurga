import json
import sys
import numpy as np
from scipy.optimize import linprog

# Test with the infeasible case
infeasible_input = {
    "sources": {
      "s1": 900,
      "s2": 600
    },
    "sink": "sink",
    "node_caps": {
      "a": {"throughput": 1200},
      "b": {"out": 600}
    },
    "edges": [
      {"from": "s1", "to": "a", "lower": 0, "upper": 900},
      {"from": "s2", "to": "a", "lower": 0, "upper": 600},
      {"from": "a", "to": "b", "lower": 0, "upper": 600},
      {"from": "b", "to": "sink", "lower": 0, "upper": 600},
      {"from": "a", "to": "c", "lower": 0, "upper": 600},
      {"from": "c", "to": "sink", "lower": 0, "upper": 600}
    ]
}

print("=== Manual LP Test ===")
print("\nEdges:")
edges = infeasible_input["edges"]
for i, e in enumerate(edges):
    print(f"  {i}: {e['from']} -> {e['to']} [0, {e['upper']}]")

print("\nNodes: s1, s2, a, b, c, sink")
print("Sources: s1=900, s2=600")
print("Sink: sink")

# Build constraints manually
num_edges = len(edges)

# Objective: maximize flow to sink
c = []
for i, e in enumerate(edges):
    if e["to"] == "sink":
        c.append(-1.0)
    else:
        c.append(1e-6 * (i + 1))

print(f"\nObjective c: {c}")

# Conservation at non-sink nodes
nodes = ["s1", "s2", "a", "b", "c"]  # exclude sink
A_eq = []
b_eq = []

for node in nodes:
    row = [0.0] * num_edges
    for i, e in enumerate(edges):
        if e["from"] == node:
            row[i] = 1.0  # outflow
        if e["to"] == node:
            row[i] = -1.0  # inflow
    
    supply = 900 if node == "s1" else (600 if node == "s2" else 0)
    
    A_eq.append(row)
    b_eq.append(supply)
    print(f"\nNode {node}: {row} = {supply}")

# Node capacity constraints
A_ub = []
b_ub = []

# a: throughput 1200
# inflow to a
row = [0.0] * num_edges
for i, e in enumerate(edges):
    if e["to"] == "a":
        row[i] = 1.0
A_ub.append(row)
b_ub.append(1200)
print(f"\nNode a inflow: {row} <= 1200")

# outflow from a
row = [0.0] * num_edges
for i, e in enumerate(edges):
    if e["from"] == "a":
        row[i] = 1.0
A_ub.append(row)
b_ub.append(1200)
print(f"Node a outflow: {row} <= 1200")

# b: out 600
row = [0.0] * num_edges
for i, e in enumerate(edges):
    if e["from"] == "b":
        row[i] = 1.0
A_ub.append(row)
b_ub.append(600)
print(f"Node b outflow: {row} <= 600")

# Bounds
bounds = [(0, e["upper"]) for e in edges]
print(f"\nBounds: {bounds}")

# Solve
print("\n=== Solving LP ===")
result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")

print(f"Success: {result.success}")
print(f"Message: {result.message}")
if result.success:
    print(f"Flows: {result.x}")
    print(f"\nFlow breakdown:")
    for i, (e, flow) in enumerate(zip(edges, result.x)):
        print(f"  {e['from']} -> {e['to']}: {flow}")
    
    total_to_sink = sum(result.x[i] for i, e in enumerate(edges) if e["to"] == "sink")
    print(f"\nTotal to sink: {total_to_sink}")
else:
    print(f"Optimization failed!")

print("\n=== Running actual script ===")
import subprocess
process = subprocess.run(
    [sys.executable, "belts/main.py"],
    input=json.dumps(infeasible_input),
    text=True,
    capture_output=True
)

print("Result:", process.stdout)