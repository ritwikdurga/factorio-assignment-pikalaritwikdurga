# factory/main.py

import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from scipy.optimize import linprog


TOL = 1e-9
TIE_EPS = 1e-6


@dataclass(frozen=True)
class RecipeInfo:
    name: str
    machine: str
    inputs: Dict[str, float]
    outputs: Dict[str, float]
    effective_crafts_per_min: float
    prod_multiplier: float


class FactoryOptimizer:
    def __init__(self, config: Dict):
        self.config = config
        self.machines = config["machines"]
        self.recipes_raw = config["recipes"]
        self.modules = config.get("modules", {})
        limits = config.get("limits", {})
        self.raw_supply = limits.get("raw_supply_per_min", {})
        self.max_machines = limits.get("max_machines", {})
        target = config["target"]
        self.target_item = target["item"]
        self.target_rate = float(target["rate_per_min"])

        # Process recipes
        self.recipe_names = sorted(self.recipes_raw.keys())
        self.recipes: List[RecipeInfo] = []
        self.machine_to_recipes: Dict[str, List[str]] = defaultdict(list)
        self.all_items = set()

        for name in self.recipe_names:
            spec = self.recipes_raw[name]
            machine = spec["machine"]
            
            inputs = {k: float(v) for k, v in spec.get("in", {}).items()}
            outputs = {k: float(v) for k, v in spec.get("out", {}).items()}
            
            # Calculate effective crafts per minute
            base_speed = float(self.machines[machine]["crafts_per_min"])
            module = self.modules.get(machine, {})
            speed_bonus = float(module.get("speed", 0.0))
            prod_bonus = float(module.get("prod", 0.0))
            time_s = float(spec["time_s"])
            
            effective_crafts_per_min = base_speed * (1 + speed_bonus) * 60.0 / time_s
            prod_multiplier = 1.0 + prod_bonus
            
            recipe = RecipeInfo(
                name=name,
                machine=machine,
                inputs=inputs,
                outputs=outputs,
                effective_crafts_per_min=effective_crafts_per_min,
                prod_multiplier=prod_multiplier,
            )
            self.recipes.append(recipe)
            self.machine_to_recipes[machine].append(name)
            self.all_items.update(inputs.keys())
            self.all_items.update(outputs.keys())

        # Find target recipe
        self.target_recipe_idx = None
        for i, recipe in enumerate(self.recipes):
            if self.target_item in recipe.outputs:
                self.target_recipe_idx = i
                break

        # Build LP
        self.bounds = [(0.0, None)] * len(self.recipes)
        self.objective = self._build_objective()
        self.A_eq, self.b_eq_items = self._build_conservation()
        self.A_ub, self.b_ub = self._build_constraints()

    def _build_objective(self) -> List[float]:
        """Minimize total machines used, with tie-breaking"""
        obj = []
        for i, recipe in enumerate(self.recipes, start=1):
            machines_per_craft = 1.0 / recipe.effective_crafts_per_min
            obj.append(machines_per_craft + TIE_EPS * i)
        return obj

    def _build_conservation(self) -> Tuple[List[List[float]], List[str]]:
        """Build conservation equations WITH productivity bonuses"""
        items_to_balance = []
        A_eq = []

        # Target item equation
        items_to_balance.append(self.target_item)
        row = []
        for recipe in self.recipes:
            out_amt = recipe.outputs.get(self.target_item, 0.0) * recipe.prod_multiplier
            in_amt = recipe.inputs.get(self.target_item, 0.0)
            row.append(out_amt - in_amt)
        A_eq.append(row)

        # Intermediate items (not raw, not target)
        intermediates = (
            self.all_items
            - set(self.raw_supply.keys())
            - {self.target_item}
        )
        for item in sorted(intermediates):
            items_to_balance.append(item)
            row = []
            for recipe in self.recipes:
                out_amt = recipe.outputs.get(item, 0.0) * recipe.prod_multiplier
                in_amt = recipe.inputs.get(item, 0.0)
                row.append(out_amt - in_amt)
            A_eq.append(row)

        return A_eq, items_to_balance

    def _build_constraints(self) -> Tuple[List[List[float]], List[float]]:
        """Build inequality constraints for raw materials and machines"""
        A_ub = []
        b_ub = []

        # Raw material constraints (net consumption <= supply)
        for item, supply in self.raw_supply.items():
            row = []
            for recipe in self.recipes:
                out_amt = recipe.outputs.get(item, 0.0) * recipe.prod_multiplier
                in_amt = recipe.inputs.get(item, 0.0)
                row.append(out_amt - in_amt)
            A_ub.append([-x for x in row])
            b_ub.append(float(supply))

        # Machine constraints
        for machine, limit in self.max_machines.items():
            row = []
            for recipe in self.recipes:
                if recipe.machine == machine:
                    row.append(1.0 / recipe.effective_crafts_per_min)
                else:
                    row.append(0.0)
            A_ub.append(row)
            b_ub.append(float(limit))

        return A_ub, b_ub

    def _solve_for_rate(self, target_rate: float):
        """Solve LP for a specific target rate (in items/min with productivity)"""
        b_eq = []
        for item in self.b_eq_items:
            if item == self.target_item:
                b_eq.append(target_rate)
            else:
                b_eq.append(0.0)

        result = linprog(
            self.objective,
            A_ub=self.A_ub if self.A_ub else None,
            b_ub=self.b_ub if self.b_ub else None,
            A_eq=self.A_eq if self.A_eq else None,
            b_eq=b_eq if b_eq else None,
            bounds=self.bounds,
            method="highs",
        )
        return result

    def optimize(self) -> Dict:
        """Main optimization routine"""
        # Try exact target first
        result = self._solve_for_rate(self.target_rate)
        
        if result.success and self._validate_solution(result.x, self.target_rate):
            return self._format_success(result.x)

        # Binary search for maximum feasible rate
        lo, hi = 0.0, self.target_rate
        best_x = None
        best_rate = 0.0

        for _ in range(60):
            mid = (lo + hi) / 2.0
            result = self._solve_for_rate(mid)
            
            if result.success and self._validate_solution(result.x, mid):
                best_rate = mid
                best_x = result.x
                lo = mid
            else:
                hi = mid
            
            if hi - lo < 1e-6:
                break

        if best_x is None:
            best_x = np.zeros(len(self.recipes))
            best_rate = 0.0

        return self._format_infeasible(best_rate, best_x)

    def _validate_solution(self, x: np.ndarray, target_rate: float) -> bool:
        """Validate solution satisfies all constraints"""
        if np.any(x < -TOL):
            return False

        # Check conservation equations
        for item, row in zip(self.b_eq_items, self.A_eq):
            lhs = float(np.dot(row, x))
            rhs = target_rate if item == self.target_item else 0.0
            if abs(lhs - rhs) > TOL:
                return False

        # Check inequality constraints
        if self.A_ub:
            lhs_vals = np.dot(self.A_ub, x)
            for lhs, rhs in zip(lhs_vals, self.b_ub):
                if lhs > rhs + TOL:
                    return False

        return True

    def _format_success(self, x: np.ndarray) -> Dict:
        """Format successful solution"""
        crafts = {}
        for recipe, val in zip(self.recipes, x):
            crafts[recipe.name] = 0.0 if abs(val) < TOL else float(val)

        machines = defaultdict(float)
        for recipe, val in zip(self.recipes, x):
            if val > TOL:
                machines[recipe.machine] += val / recipe.effective_crafts_per_min

        raw_consumption = {}
        for item in sorted(self.raw_supply.keys()):
            net = 0.0
            for recipe, val in zip(self.recipes, x):
                out_amt = recipe.outputs.get(item, 0.0) * recipe.prod_multiplier
                in_amt = recipe.inputs.get(item, 0.0)
                net += (in_amt - out_amt) * val
            raw_consumption[item] = 0.0 if abs(net) < TOL else float(net)

        return {
            "status": "ok",
            "per_recipe_crafts_per_min": crafts,
            "per_machine_counts": {m: float(c) for m, c in sorted(machines.items())},
            "raw_consumption_per_min": raw_consumption,
        }

    def _format_infeasible(self, max_rate: float, x: np.ndarray) -> Dict:
        """Format infeasible solution with bottleneck hints"""
        crafts = {}
        for recipe, val in zip(self.recipes, x):
            crafts[recipe.name] = 0.0 if abs(val) < TOL else float(val)

        machines = defaultdict(float)
        for recipe, val in zip(self.recipes, x):
            if val > TOL:
                machines[recipe.machine] += val / recipe.effective_crafts_per_min

        raw_consumption = {}
        for item in sorted(self.raw_supply.keys()):
            net = 0.0
            for recipe, val in zip(self.recipes, x):
                out_amt = recipe.outputs.get(item, 0.0) * recipe.prod_multiplier
                in_amt = recipe.inputs.get(item, 0.0)
                net += (in_amt - out_amt) * val
            raw_consumption[item] = 0.0 if abs(net) < TOL else float(net)

        # Find bottlenecks - check which constraints are tight when trying higher rate
        hints = []
        
        # Try to solve at target rate to see what's blocking us
        target_result = self._solve_for_rate(self.target_rate)
        
        if target_result.success:
            # Check constraints with the target solution
            test_x = target_result.x
            
            # Check machine constraints
            for machine, limit in self.max_machines.items():
                machine_usage = 0.0
                for recipe, val in zip(self.recipes, test_x):
                    if recipe.machine == machine:
                        machine_usage += val / recipe.effective_crafts_per_min
                if machine_usage > limit - TOL:
                    hints.append(f"{machine} cap")
            
            # Check raw material constraints
            for item, supply in self.raw_supply.items():
                net = 0.0
                for recipe, val in zip(self.recipes, test_x):
                    out_amt = recipe.outputs.get(item, 0.0) * recipe.prod_multiplier
                    in_amt = recipe.inputs.get(item, 0.0)
                    net += (in_amt - out_amt) * val
                if net > supply - TOL:
                    hints.append(f"{item} supply")
        else:
            # Solver failed - check which constraints would be violated
            # by checking the dual values or by examining current solution
            for machine, count in machines.items():
                limit = self.max_machines.get(machine)
                if limit:
                    # Check if we're close to limit (within 10%)
                    if count >= limit * 0.9:
                        hints.append(f"{machine} cap")

            for item, consumption in raw_consumption.items():
                supply = self.raw_supply.get(item, 0.0)
                if consumption >= supply * 0.9:
                    hints.append(f"{item} supply")

        return {
            "status": "infeasible",
            "max_feasible_target_per_min": float(max_rate),
            "bottleneck_hint": sorted(set(hints)),
        }


def main():
    try:
        config = json.load(sys.stdin)
        optimizer = FactoryOptimizer(config)
        result = optimizer.optimize()
    except Exception as e:
        result = {"status": "error", "message": str(e)}
    json.dump(result, sys.stdout, separators=(",", ":"))


if __name__ == "__main__":
    main()