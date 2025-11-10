import json
import sys
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, List, Tuple
from scipy.optimize import linprog

TOL = 1e-9
TIE_EPS = 1e-6

@dataclass
class Connection:
    start: str
    end: str
    min_flow: float
    max_flow: float

class NetworkOptimizer:
    def __init__(self, config: Dict):
        self.config = config
        self.supplies: Dict[str, float] = {
            key: float(val) for key, val in config.get("sources", {}).items()
        }
        if not self.supplies:
            raise ValueError("at least one source required")
        self.supply_keys = sorted(self.supplies.keys())
        self.demand = config["sink"]
        self.capacity_limits = {
            key: {
                subkey: float(subval)
                for subkey, subval in details.items()
            }
            for key, details in config.get("node_caps", {}).items()
        }
        connections_raw = config.get("edges", [])
        if not connections_raw:
            raise ValueError("at least one edge required")
        self.connections: List[Connection] = []
        self.vertices = set(self.supplies.keys()) | {self.demand}
        for details in connections_raw:
            start = details["from"]
            end = details["to"]
            min_flow = float(details.get("lower", 0.0))
            max_flow = float(details["upper"])
            if max_flow < min_flow - 1e-9:
                raise ValueError(f"edge {start}->{end} has upper < lower")
            self.connections.append(Connection(start, end, min_flow, max_flow))
            self.vertices.add(start)
            self.vertices.add(end)
        self.vertices = sorted(self.vertices)
        self.overall_supply = sum(self.supplies.values())
        if self.overall_supply < 0:
            raise ValueError("total supply must be non-negative")
        self.connection_count = len(self.connections)
        self.supply_map = {key: idx for idx, key in enumerate(self.supply_keys)}
        self.unused_var_start = self.connection_count + len(self.supply_keys)
        self.total_vars = self.unused_var_start + 1

    def optimize(self) -> Dict:
        objective = [TIE_EPS * (idx + 1) for idx in range(self.connection_count)]
        for idx in range(len(self.supply_keys)):
            objective.append(TIE_EPS * (self.connection_count + idx + 1))
        objective.append(-1.0)
        var_bounds = []
        for conn in self.connections:
            var_bounds.append((conn.min_flow, conn.max_flow))
        for key in self.supply_keys:
            amount = self.supplies[key]
            var_bounds.append((0.0, amount))
        var_bounds.append((0.0, self.overall_supply))
        eq_matrix, eq_rhs = self._create_eq_constraints()
        ineq_matrix, ineq_rhs = self._create_ineq_constraints()
        opt_result = linprog(
            objective,
            A_ub=ineq_matrix if ineq_matrix else None,
            b_ub=ineq_rhs if ineq_rhs else None,
            A_eq=eq_matrix if eq_matrix else None,
            b_eq=eq_rhs if eq_rhs else None,
            bounds=var_bounds,
            method="highs",
        )
        if not opt_result.success:
            return {
                "status": "infeasible",
                "reachable_set": [],
                "shortfall": {
                    "balance_needed": self.overall_supply,
                    "saturated_vertices": [],
                    "saturated_connections": [],
                },
            }
        conn_flows = opt_result.x[: self.connection_count]
        leftovers = opt_result.x[self.connection_count : self.unused_var_start]
        achieved = float(opt_result.x[self.unused_var_start])
        shortfall = max(0.0, self.overall_supply - achieved)
        rounded_flows = [0.0 if abs(val) < 1e-9 else float(val) for val in conn_flows]
        rounded_leftovers = [max(0.0, float(l)) for l in leftovers]
        if shortfall <= 1e-6:
            return self._format_optimal(rounded_flows, achieved)
        return self._format_suboptimal(rounded_flows, rounded_leftovers, achieved, shortfall)

    def _create_eq_constraints(self) -> Tuple[List[List[float]], List[float]]:
        constraint_rows: List[List[float]] = []
        right_sides: List[float] = []
        for vert in self.vertices:
            row = [0.0] * self.total_vars
            for idx, conn in enumerate(self.connections):
                if conn.start == vert:
                    row[idx] += 1.0
                if conn.end == vert:
                    row[idx] -= 1.0
            if vert in self.supplies:
                leftover_idx = self.connection_count + self.supply_map[vert]
                row[leftover_idx] += 1.0
                right_sides.append(self.supplies[vert])
            elif vert == self.demand:
                row[self.unused_var_start] += 1.0
                right_sides.append(0.0)
            else:
                right_sides.append(0.0)
            constraint_rows.append(row)
        return constraint_rows, right_sides

    def _create_ineq_constraints(self) -> Tuple[List[List[float]], List[float]]:
        ineq_rows: List[List[float]] = []
        bounds_rhs: List[float] = []
        for vert, limits in self.capacity_limits.items():
            if "throughput" in limits:
                limit_val = limits["throughput"]
                ineq_rows.append(self._inflow_row(vert))
                bounds_rhs.append(limit_val)
                ineq_rows.append(self._outflow_row(vert))
                bounds_rhs.append(limit_val)
            if "in" in limits:
                ineq_rows.append(self._inflow_row(vert))
                bounds_rhs.append(limits["in"])
            if "out" in limits:
                ineq_rows.append(self._outflow_row(vert))
                bounds_rhs.append(limits["out"])
        return ineq_rows, bounds_rhs

    def _inflow_row(self, vert: str) -> List[float]:
        row = [0.0] * self.total_vars
        for idx, conn in enumerate(self.connections):
            if conn.end == vert:
                row[idx] += 1.0
        return row

    def _outflow_row(self, vert: str) -> List[float]:
        row = [0.0] * self.total_vars
        for idx, conn in enumerate(self.connections):
            if conn.start == vert:
                row[idx] += 1.0
        return row

    def _format_optimal(self, flows: List[float], achieved: float) -> Dict:
        output_flows = []
        for conn, amt in zip(self.connections, flows):
            if abs(amt) < 1e-9:
                continue
            output_flows.append(
                {"from": conn.start, "to": conn.end, "flow": float(amt)}
            )
        output_flows.sort(key=lambda x: (x["from"], x["to"]))
        return {
            "status": "ok",
            "max_flow_per_min": float(achieved),
            "flows": output_flows,
        }

    def _format_suboptimal(
        self,
        flows: List[float],
        leftovers: List[float],
        achieved: float,
        shortfall: float,
    ) -> Dict:
        inflow_totals, outflow_totals = self._vertex_totals(flows)
        reachable = self._find_reachable(flows, leftovers, inflow_totals, outflow_totals)
        cut_vertices = sorted(vert for vert in reachable if vert in self.vertices)
        saturated_vertices = self._saturated_vertices(inflow_totals, outflow_totals, reachable)
        saturated_connections = self._saturated_connections(flows, reachable, shortfall)
        return {
            "status": "infeasible",
            "max_flow_per_min": float(achieved),
            "cut_reachable": cut_vertices,
            "deficit": {
                "demand_balance": float(shortfall),
                "tight_nodes": saturated_vertices,
                "tight_edges": saturated_connections,
            },
        }

    def _vertex_totals(self, flows: List[float]) -> Tuple[Dict[str, float], Dict[str, float]]:
        inflows = defaultdict(float)
        outflows = defaultdict(float)
        for conn, amt in zip(self.connections, flows):
            outflows[conn.start] += amt
            inflows[conn.end] += amt
        return inflows, outflows

    def _find_reachable(
        self,
        flows: List[float],
        leftovers: List[float],
        inflows: Dict[str, float],
        outflows: Dict[str, float],
    ) -> set:
        residual_graph = defaultdict(list)
        for conn, amt in zip(self.connections, flows):
            fwd_cap = conn.max_flow - amt
            bwd_cap = amt - conn.min_flow
            if fwd_cap > TOL:
                residual_graph[conn.start].append((conn.end, fwd_cap))
            if bwd_cap > TOL:
                residual_graph[conn.end].append((conn.start, bwd_cap))
        master_supply = "__master_supply__"
        residual_graph[master_supply] = []
        for key in self.supply_keys:
            idx = self.supply_map[key]
            avail = self.supplies[key] - (outflows[key] - inflows[key])
            avail = max(0.0, avail)
            if avail > TOL:
                residual_graph[master_supply].append((key, avail))
            consumed = outflows[key] - inflows[key]
            if consumed > TOL:
                residual_graph[key].append((master_supply, consumed))
        explored = {master_supply}
        bfs_queue = deque([master_supply])
        while bfs_queue:
            curr = bfs_queue.popleft()
            for neigh, cap in residual_graph.get(curr, []):
                if cap <= TOL or neigh in explored:
                    continue
                explored.add(neigh)
                bfs_queue.append(neigh)
        return explored

    def _saturated_vertices(
        self,
        inflows: Dict[str, float],
        outflows: Dict[str, float],
        reachable: set,
    ) -> List[str]:
        saturated = []
        for vert, limits in self.capacity_limits.items():
            if vert not in reachable:
                continue
            in_total = inflows.get(vert, 0.0)
            out_total = outflows.get(vert, 0.0)
            if "throughput" in limits:
                limit_val = limits["throughput"]
                if limit_val - max(in_total, out_total) <= 1e-6:
                    saturated.append(vert)
                    continue
            if "in" in limits and limits["in"] - in_total <= 1e-6:
                saturated.append(vert)
                continue
            if "out" in limits and limits["out"] - out_total <= 1e-6:
                saturated.append(vert)
        return sorted(set(saturated))

    def _saturated_connections(
        self,
        flows: List[float],
        reachable: set,
        shortfall: float,
    ) -> List[Dict[str, float]]:
        saturated = []
        for conn, amt in zip(self.connections, flows):
            if conn.start in reachable and conn.end not in reachable:
                remaining = conn.max_flow - amt
                if remaining <= 1e-6:
                    saturated.append(
                        {
                            "from": conn.start,
                            "to": conn.end,
                            "flow_needed": float(shortfall),
                        }
                    )
        saturated.sort(key=lambda x: (x["from"], x["to"]))
        return saturated

def entry_point() -> None:
    try:
        input_data = json.load(sys.stdin)
        optimizer = NetworkOptimizer(input_data)
        outcome = optimizer.optimize()
    except Exception as err:  # noqa: BLE001
        outcome = {"status": "error", "message": str(err)}
    json.dump(outcome, sys.stdout, separators=(",", ":"))

if __name__ == "__main__":
    entry_point()