# ERP.AI Part 2 – Production & Transport

This repository contains the deliverables for the ERP.AI "Production Steady State & Bounded Transport" assessment. Two standalone CLI tools read JSON on stdin and emit JSON on stdout:

- `factory/main.py` – steady-state production planning with machine capacity and raw-material supply constraints, using linear programming to optimize recipe selection and craft rates.

- `belts/main.py` – maximum flow optimization on a directed graph with edge capacity bounds and node throughput limits, determining feasibility and identifying bottlenecks.

Both tools are deterministic, side-effect free, and finish well under the 2 s target on the supplied samples.

## Production Planner

### Production model overview

- **Variables** – crafts-per-minute `x_r ≥ 0` for every recipe. Machine usage is derived as `x_r / effective_crafts_per_min(r)`.

- **Effective rates** – the effective crafts/min for recipe `r` on machine `m` is  

  `machines[m].crafts_per_min * (1 + modules[m].speed) * 60 / time_s(r)`.  

  Productivity multiplies outputs only (`out * (1 + modules[m].prod)`), matching the assessment spec.

- **Material balance** – for every intermediate material we enforce  

  `Σ output_r[i]·(1+prod_m)·x_r − Σ input_r[i]·x_r = 0`.  

  The desired product's balance equals the requested production rate.

- **Resource supplies** – for each raw material with a cap, we add  

  `total_consumption ≤ supply_cap`.

- **Machine caps** – per-machine-type usage inequality: `Σ x_r / effective_crafts_per_min(r) ≤ max_machines[m]`.

All constraints are linear, so the steady-state search is solved with a single LP minimising total machine usage (`Σ x_r / effective_crafts_per_min`). Tiny lexicographic penalties (`1e-6`) break ties deterministically across recipe names.

### Loops, side-products, and resource balance

Loops and side-products fall out naturally from the conservation equations: every non-raw, non-desired material has net zero balance. Productivity reduces the crafts needed for a desired product, consequently lowering input usage—precisely mirroring Factorio's productivity behaviour.

### Unachievability handling

If the requested rate is unachievable, we binary-search the desired rate, repeatedly resolving the LP until unachievability appears. The highest achievable rate (within `1e-6`) is reported together with bottleneck hints, driven by tight raw-supply or machine-capacity constraints.

### Production numerical notes

- Solver: `scipy.optimize.linprog(method="highs")`.

- Tolerances: `1e-9` on equality checks, `1e-6` for inequality slack and bottleneck hint detection.

- Determinism: fixed recipe ordering and tie-breaking in the objective ensure repeatable outputs.

## Transport Optimizer

### Transport model overview

The transport problem is also expressed as a linear program:

- **Variables** – edge flows `f_e` for each connection, bounded by `[lower, upper]`.

- **Connection bounds** – enforced via variable bounds `[lower, upper]` on each edge.

- **Vertex conservation** – for every intermediate vertex we enforce  

  `Σ_out f_e − Σ_in f_e = 0` (flow in equals flow out).  

  Sources have no conservation constraint (they inject supply), and the sink receives all flow.

- **Source limits** – for each source, total outgoing flow is constrained: `Σ_out f_e ≤ supply`.

- **Vertex caps** – optional `throughput`, `in`, and `out` caps are encoded as `A_ub` rows on incoming or outgoing flow totals.

- **Objective** – maximise flow to sink (implemented as minimising negative sink flow with small positive tie-breaking coefficients on other edges to keep HiGHS deterministic).

### Achievability, certificates, and residuals

- When total flow to sink equals the overall supply within tolerance, the run is marked `status="ok"` and non-zero flows are returned.

- Otherwise the solver reports `status="infeasible"` with:

  - `max_flow_per_min`: the maximum achievable flow to sink,

  - `cut_reachable`: vertices reachable from sources in the residual graph (the min-cut partition),

  - `deficit`: an object containing:
    - `demand_balance`: the shortfall (total supply - achieved flow),
    - `tight_nodes`: vertices whose caps are saturated on the reachable side,
    - `tight_edges`: saturated edges crossing the reachable/unreachable partition, annotated with the remaining shortfall.

- Residual exploration uses forward residual capacity `upper − f` and backward residual `f − lower` to build the residual graph for min-cut analysis.

### Transport numerical notes

- Solver: `scipy.optimize.linprog(method="highs")`.

- Tolerances: `1e-9` for achievability checks, `1e-6` for identifying saturated vertices/connections.

- Determinism: fixed connection ordering plus `1e-6` lexicographic penalties on the objective.

## Failure modes & mitigations

- **Conflicting min_flow/max_flow bounds** – detected upfront (`max_flow < min_flow` throws). Solver unachievability is surfaced as `"status": "error"` with the message.

- **Disconnected subgraphs** – handled naturally by the LP. Residual analysis pinpoints unreachable vertices and saturated constraints.

- **Degenerate recipes/connections** – zero-capacity or unused flows are pruned in the output.

- **Floating-point drift** – all post-processing clips tiny negative numbers (`<1e-9`) to zero before emitting JSON.

## Repository tour

```text
part2_assignment/

├─ factory/main.py           # production CLI (stdin JSON -> stdout JSON)

├─ belts/main.py             # transport CLI

├─ bin/                      # wrapper scripts for direct invocation

│  ├─ factory                # wrapper for factory command

│  └─ belts                  # wrapper for belts command

├─ run_samples.py            # convenience runner for the sample fixtures

├─ requirements.txt          # SciPy dependency

├─ tests/

│  ├─ data/                  # sample and regression fixtures

│  ├─ test_factory.py

│  └─ test_belts.py

├─ README.md                 # this document

└─ RUN.md                    # exact run commands

```

## Getting started

1. Create a virtual environment (recommended) and install dependencies:  

   `python3 -m pip install -r requirements.txt`

2. Run samples:  

   `python3 run_samples.py "python3 factory/main.py" "python3 belts/main.py"`

3. Execute the automated tests:  

   `FACTORY_CMD="python3 factory/main.py" BELTS_CMD="python3 belts/main.py" python3 -m pytest -q`

## Usage

Both tools read JSON from stdin and write JSON to stdout. You can use them in three ways:

### Option 1: Using the wrapper scripts (recommended)

```bash
cd part2_assignment
./bin/factory < input.json > output.json
./bin/belts < input.json > output.json
```

### Option 2: Add `bin/` to your PATH to use them as `factory` and `belts`

```bash
cd part2_assignment
export PATH=$PATH:$(pwd)/bin
factory < input.json > output.json
belts < input.json > output.json
```

### Option 3: Use the scripts directly (they're now executable)

```bash
cd part2_assignment
./factory/main.py < input.json > output.json
./belts/main.py < input.json > output.json
```

All three methods produce identical results. The tools are deterministic, have no side effects, and complete in ≤ 2 seconds on typical inputs.
