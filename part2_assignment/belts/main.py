#!/usr/bin/env python3
import json
import sys
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
from scipy.optimize import linprog

TOL = 1e-9
TIE_EPS = 1e-6


@dataclass
class Edge:
    src: str
    dst: str
    lower: float
    upper: float


class BeltOptimizer:
    def __init__(self, config: Dict):
        self.config = config
        self.sources: Dict[str, float] = {
            k: float(v) for k, v in config.get("sources", {}).items()
        }
        if not self.sources:
            raise ValueError("At least one source required")
        
        self.sink = config["sink"]
        self.node_caps = {
            node: {k: float(v) for k, v in caps.items()}
            for node, caps in config.get("node_caps", {}).items()
        }
        
        edges_raw = config.get("edges", [])
        if not edges_raw:
            raise ValueError("At least one edge required")
        
        self.edges: List[Edge] = []
        self.nodes = set(self.sources.keys()) | {self.sink}
        
        for e in edges_raw:
            src = e["from"]
            dst = e["to"]
            lower = float(e.get("lower", 0.0))
            upper = float(e["upper"])
            if upper < lower - TOL:
                raise ValueError(f"Edge {src}->{dst} has upper < lower")
            self.edges.append(Edge(src, dst, lower, upper))
            self.nodes.add(src)
            self.nodes.add(dst)
        
        self.nodes = sorted(self.nodes)
        self.total_supply = sum(self.sources.values())
        self.num_edges = len(self.edges)

    def optimize(self) -> Dict:
        """Main optimization using linear programming"""
        # Objective: maximize flow to sink (negative for minimization)
        c = []
        for i, edge in enumerate(self.edges):
            if edge.dst == self.sink:
                c.append(-1.0)  # Maximize sink flow
            else:
                c.append(TIE_EPS * (i + 1))  # Tie-breaking
        
        # Bounds: lower <= flow <= upper
        bounds = [(e.lower, e.upper) for e in self.edges]
        
        # Equality constraints: flow conservation at intermediate nodes
        A_eq, b_eq = self._build_conservation()
        
        # Inequality constraints: source limits and node capacity limits
        A_ub, b_ub = self._build_capacity_constraints()
        
        result = linprog(
            c,
            A_ub=A_ub if A_ub else None,
            b_ub=b_ub if b_ub else None,
            A_eq=A_eq if A_eq else None,
            b_eq=b_eq if b_eq else None,
            bounds=bounds,
            method="highs",
        )
        
        if not result.success:
            # LP failed - shouldn't happen with proper formulation
            flows = np.array([e.lower for e in self.edges])
            total_flow = self._compute_total_flow(flows)
            return self._format_infeasible(flows, total_flow)
        
        flows = result.x
        total_flow = self._compute_total_flow(flows)
        
        # Check if we achieved full supply routing
        if abs(total_flow - self.total_supply) < TOL:
            return self._format_success(flows, total_flow)
        else:
            return self._format_infeasible(flows, total_flow)

    def _build_conservation(self) -> Tuple[List[List[float]], List[float]]:
        """Build flow conservation constraints
        
        Only for intermediate nodes (not sources or sink):
        inflow = outflow
        """
        A_eq = []
        b_eq = []
        
        for node in self.nodes:
            # Skip sources and sink
            if node in self.sources or node == self.sink:
                continue
                
            row = [0.0] * self.num_edges
            for i, edge in enumerate(self.edges):
                if edge.src == node:
                    row[i] += 1.0  # Outflow
                if edge.dst == node:
                    row[i] -= 1.0  # Inflow
            
            # Intermediate nodes: inflow = outflow (balance = 0)
            A_eq.append(row)
            b_eq.append(0.0)
        
        return A_eq, b_eq

    def _build_capacity_constraints(self) -> Tuple[List[List[float]], List[float]]:
        """Build node capacity constraints and source limits"""
        A_ub = []
        b_ub = []
        
        # Source output limits: outflow from source <= supply
        for source, supply in self.sources.items():
            row = [0.0] * self.num_edges
            for i, edge in enumerate(self.edges):
                if edge.src == source:
                    row[i] = 1.0
            A_ub.append(row)
            b_ub.append(supply)
        
        # Node capacity constraints
        for node, caps in self.node_caps.items():
            if "throughput" in caps:
                limit = caps["throughput"]
                # Inflow <= limit
                row_in = [0.0] * self.num_edges
                for i, edge in enumerate(self.edges):
                    if edge.dst == node:
                        row_in[i] = 1.0
                A_ub.append(row_in)
                b_ub.append(limit)
                
                # Outflow <= limit
                row_out = [0.0] * self.num_edges
                for i, edge in enumerate(self.edges):
                    if edge.src == node:
                        row_out[i] = 1.0
                A_ub.append(row_out)
                b_ub.append(limit)
            
            if "in" in caps:
                row = [0.0] * self.num_edges
                for i, edge in enumerate(self.edges):
                    if edge.dst == node:
                        row[i] = 1.0
                A_ub.append(row)
                b_ub.append(caps["in"])
            
            if "out" in caps:
                row = [0.0] * self.num_edges
                for i, edge in enumerate(self.edges):
                    if edge.src == node:
                        row[i] = 1.0
                A_ub.append(row)
                b_ub.append(caps["out"])
        
        return A_ub, b_ub

    def _compute_total_flow(self, flows: List[float]) -> float:
        """Compute total flow reaching sink"""
        total = 0.0
        for i, edge in enumerate(self.edges):
            if edge.dst == self.sink:
                total += flows[i]
        return total

    def _format_success(self, flows: List[float], total_flow: float) -> Dict:
        """Format successful solution"""
        flow_list = []
        for edge, flow in zip(self.edges, flows):
            if abs(flow) > TOL:
                flow_list.append({
                    "from": edge.src,
                    "to": edge.dst,
                    "flow": round(float(flow), 9),
                })
        flow_list.sort(key=lambda x: (x["from"], x["to"]))
        
        return {
            "status": "ok",
            "max_flow_per_min": round(float(total_flow), 9),
            "flows": flow_list,
        }

    def _format_infeasible(self, flows: List[float], total_flow: float) -> Dict:
        """Format infeasible solution with cut certificate"""
        # Build residual graph
        residual = self._build_residual_graph(flows)
        
        # Find reachable nodes from sources via BFS in residual graph
        reachable = set()
        queue = deque(list(self.sources.keys()))
        reachable.update(queue)
        
        while queue:
            node = queue.popleft()
            for neighbor, cap in residual.get(node, []):
                if cap > TOL and neighbor not in reachable:
                    reachable.add(neighbor)
                    queue.append(neighbor)
        
        cut_nodes = sorted([n for n in reachable if n in self.nodes])
        
        # Compute deficit
        deficit = self.total_supply - total_flow
        
        # Find tight nodes and edges
        tight_nodes = self._find_tight_nodes(flows, reachable)
        tight_edges = self._find_tight_edges(flows, reachable, deficit)
        
        return {
            "status": "infeasible",
            "max_flow_per_min": round(float(total_flow), 9),
            "cut_reachable": cut_nodes,
            "deficit": {
                "demand_balance": round(float(deficit), 9),
                "tight_nodes": tight_nodes,
                "tight_edges": tight_edges,
            },
        }

    def _build_residual_graph(self, flows: List[float]) -> Dict[str, List[Tuple[str, float]]]:
        """Build residual graph for min-cut computation"""
        residual = defaultdict(list)
        
        for i, edge in enumerate(self.edges):
            flow = flows[i]
            # Forward capacity
            fwd_cap = edge.upper - flow
            if fwd_cap > TOL:
                residual[edge.src].append((edge.dst, fwd_cap))
            # Backward capacity
            bwd_cap = flow - edge.lower
            if bwd_cap > TOL:
                residual[edge.dst].append((edge.src, bwd_cap))
        
        return residual

    def _find_tight_nodes(self, flows: List[float], reachable: set) -> List[str]:
        """Find nodes at capacity in the reachable set"""
        tight = []
        
        for node, caps in self.node_caps.items():
            if node not in reachable:
                continue
            
            inflow = sum(flows[i] for i, e in enumerate(self.edges) if e.dst == node)
            outflow = sum(flows[i] for i, e in enumerate(self.edges) if e.src == node)
            
            if "throughput" in caps:
                limit = caps["throughput"]
                if max(inflow, outflow) >= limit - TOL:
                    tight.append(node)
                    continue
            
            if "in" in caps and inflow >= caps["in"] - TOL:
                tight.append(node)
                continue
            
            if "out" in caps and outflow >= caps["out"] - TOL:
                tight.append(node)
        
        return sorted(set(tight))

    def _find_tight_edges(self, flows: List[float], reachable: set, deficit: float) -> List[Dict]:
        """Find edges at capacity that are bottlenecks"""
        tight = []
        
        # First, find edges crossing the min-cut (reachable -> unreachable) at capacity
        for i, edge in enumerate(self.edges):
            if edge.src in reachable and edge.dst not in reachable:
                if flows[i] >= edge.upper - TOL:
                    tight.append({
                        "from": edge.src,
                        "to": edge.dst,
                        "flow_needed": round(float(deficit), 9),
                    })
        
        # If no crossing edges, look for internal bottlenecks
        if not tight:
            tight_nodes = self._find_tight_nodes(flows, reachable)
            
            # For each tight node, find edges limited by its capacity
            for node in tight_nodes:
                node_caps = self.node_caps.get(node, {})
                
                # If it has an output cap, find outgoing edges
                if "out" in node_caps:
                    for i, edge in enumerate(self.edges):
                        if edge.src == node and flows[i] > TOL:
                            tight.append({
                                "from": edge.src,
                                "to": edge.dst,
                                "flow_needed": round(float(deficit), 9),
                            })
                # If it has throughput cap, find any edges from it
                elif "throughput" in node_caps:
                    for i, edge in enumerate(self.edges):
                        if edge.src == node and flows[i] > TOL:
                            tight.append({
                                "from": edge.src,
                                "to": edge.dst,
                                "flow_needed": round(float(deficit), 9),
                            })
                            break
        
        # Final fallback: any saturated edge from reachable set
        if not tight:
            for i, edge in enumerate(self.edges):
                if edge.src in reachable and flows[i] >= edge.upper - TOL:
                    tight.append({
                        "from": edge.src,
                        "to": edge.dst,
                        "flow_needed": round(float(deficit), 9),
                    })
        
        tight.sort(key=lambda x: (x["from"], x["to"]))
        return tight


def main():
    try:
        config = json.load(sys.stdin)
        optimizer = BeltOptimizer(config)
        result = optimizer.optimize()
    except Exception as e:
        result = {"status": "error", "message": str(e)}
    json.dump(result, sys.stdout, separators=(",", ":"))


if __name__ == "__main__":
    main()