# ERP.AI Part 2 – Production & Transport

This repository contains the deliverables for the ERP.AI "Production Steady State & Bounded Transport" assessment. Two standalone CLI tools read JSON on stdin and emit JSON on stdout:

- `production/main.py` – steady-state production planning with device and raw-material constraints.

- `transport/main.py` – bounded-flow feasibility on a directed conveyor network.

Both tools are deterministic, side-effect free, and finish well under the 2 s target on the supplied samples.

## Production Planner

### Production model overview

- **Variables** – crafts-per-minute `x_r ≥ 0` for every method. Device usage is derived as `x_r / productivity(device)`.

- **Effective rates** – the effective crafts/min for method `r` on device `m` is  

  `devices[m].crafts_per_min * (1 + enhancements[m].speed) * 60 / time_s(r)`.  

  Yield multiplies outputs only (`out * (1 + enhancements[m].prod)`), matching the assessment spec.

- **Material balance** – for every intermediate material we enforce  

  `Σ yield_r[i]·(1+prod_m)·x_r − Σ req_r[i]·x_r = 0`.  

  The desired product's balance equals the requested production rate.

- **Resource supplies** – for each raw material with a cap, we add  

  `total_gen − total_req ≤ 0` and `total_req − total_gen ≤ cap`.

- **Device caps** – per-device-type usage inequality: `Σ x_r / productivity(r) ≤ max_devices[m]`.

All constraints are linear, so the steady-state search is solved with a single LP minimising total device usage (`Σ x_r / productivity`). Tiny lexicographic penalties (`1e-6`) break ties deterministically across method names.

### Loops, side-products, and resource balance

Loops and side-products fall out naturally from the conservation equations: every non-raw, non-desired material has net zero balance. Yield reduces the crafts needed for a desired product, consequently lowering input usage—precisely mirroring Factorio's yield behaviour.

### Unachievability handling

If the requested rate is unachievable, we binary-search the desired rate, repeatedly resolving the LP until unachievability appears. The highest achievable rate (within `1e-6`) is reported together with clues, driven by tight raw-supply or device-capacity constraints. The JSON response includes the candidate craft plan even in the unachievable case to aid diagnostics.

### Production numerical notes

- Solver: `scipy.optimize.linprog(method="highs")`.

- Tolerances: `1e-9` on equality checks, `1e-6` for inequality slack and clue detection.

- Determinism: fixed method ordering and tie-breaking in the objective ensure repeatable outputs.

## Transport Optimizer

### Transport model overview

The transport problem is also expressed as a linear program:

- **Variables** – connection flows `f_e`, per-supply slack `leftover_s ≥ 0`, and an aggregate achieved flow `z`.

- **Connection bounds** – enforced via variable bounds `[min_flow, max_flow]`.

- **Vertex conservation** – for every vertex we enforce  

  `Σ_out f_e − Σ_in f_e + leftover_s = supply` (supply),  

  `Σ_out f_e − Σ_in f_e + z = 0` (demand), or  

  `Σ_out f_e − Σ_in f_e = 0` (intermediate).  

  Slack variables let the LP reduce supply usage when the network cannot absorb the entire supply while still maximising demand throughput.

- **Vertex caps** – optional `throughput`, `in`, and `out` caps are encoded as `A_ub` rows on incoming or outgoing flow totals.

- **Objective** – maximise achieved flow `z` (implemented as minimising `-z` with small positive tie-breaking coefficients to keep HiGHS deterministic).

### Achievability, certificates, and residuals

- When `z` equals the overall supply within tolerance, the run is marked `status="ok"` and non-zero flows are returned.

- Otherwise the solver reports `status="infeasible"` with:

  - `max_flow_per_min = z`,

  - `reachable_set`: vertices reachable from a synthetic master-supply in the residual graph,

  - `saturated_vertices`: vertices whose caps are saturated on the reachable side,

  - `saturated_connections`: saturated connections crossing the reachable/unreachable partition, annotated with the remaining shortfall.

- Residual exploration uses forward residual capacity `max_flow − f` and backward residual `f − min_flow`, plus synthetic supply connections capturing leftover supply capacity.

### Transport numerical notes

- Solver: `scipy.optimize.linprog(method="highs")`.

- Tolerances: `1e-9` for achievability checks, `1e-6` for identifying saturated vertices/connections.

- Determinism: fixed connection ordering plus `1e-6` lexicographic penalties on the objective.

## Failure modes & mitigations

- **Conflicting min_flow/max_flow bounds** – detected upfront (`max_flow < min_flow` throws). Solver unachievability is surfaced as `"status": "error"` with the message.

- **Disconnected subgraphs** – handled naturally by the LP. Residual analysis pinpoints unreachable vertices and saturated constraints.

- **Degenerate methods/connections** – zero-capacity or unused flows are pruned in the output.

- **Floating-point drift** – all post-processing clips tiny negative numbers (`<1e-9`) to zero before emitting JSON.

## Repository tour

```text
part2_assignment/

├─ production/main.py        # production CLI (stdin JSON -> stdout JSON)

├─ transport/main.py         # transport CLI

├─ run_samples.py            # convenience runner for the sample fixtures

├─ requirements.txt          # SciPy dependency

├─ tests/

│  ├─ data/                  # sample and regression fixtures

│  ├─ test_production.py

│  └─ test_transport.py

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
