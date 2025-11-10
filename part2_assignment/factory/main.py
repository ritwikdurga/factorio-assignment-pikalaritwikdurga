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
class ProductionInfo:
    title: str
    device: str
    requirements: Dict[str, float]
    yields: Dict[str, float]
    productivity: float


class ProductionOptimizer:
    def __init__(self, config: Dict):
        self.config = config
        self.available_devices = config["machines"]
        self.recipe_specs = config["recipes"]
        self.enhancements = config.get("modules", {})
        constraints = config.get("limits", {})
        self.resource_limits = constraints.get("raw_supply_per_min", {})
        self.device_limits = constraints.get("max_machines", {})
        goal = config["target"]
        self.desired_product = goal["item"]
        self.required_output = float(goal["rate_per_min"])

        self.production_methods = sorted(self.recipe_specs.keys())
        self.method_to_index = {title: idx for idx, title in enumerate(self.production_methods)}

        self.processed_recipes: List[ProductionInfo] = []
        self.devices_to_methods: Dict[str, List[str]] = defaultdict(list)
        self.materials = set()

        for title in self.production_methods:
            details = self.recipe_specs[title]
            device = details["machine"]
            if device not in self.available_devices:
                raise ValueError(f"device '{device}' missing for recipe '{title}'")

            requirements = {k: float(v) for k, v in details.get("in", {}).items()}
            yields = {k: float(v) for k, v in details.get("out", {}).items()}
            if not yields:
                raise ValueError(f"recipe '{title}' must produce at least one item")

            enhancement = self.enhancements.get(device, {})
            yield_multiplier = 1.0 + float(enhancement.get("prod", 0.0))
            adjusted_yields = {material: qty * yield_multiplier for material, qty in yields.items()}

            base_efficiency = float(self.available_devices[device]["crafts_per_min"])
            efficiency_multiplier = 1.0 + float(enhancement.get("speed", 0.0))
            duration = float(details["time_s"])
            if duration <= 0:
                raise ValueError(f"recipe '{title}' has non-positive time_s")
            productivity = base_efficiency * efficiency_multiplier * 60.0 / duration
            if productivity <= 0:
                raise ValueError(f"recipe '{title}' has non-positive effective rate")

            info = ProductionInfo(
                title=title,
                device=device,
                requirements=requirements,
                yields=adjusted_yields,
                productivity=productivity,
            )
            self.processed_recipes.append(info)
            self.devices_to_methods[device].append(title)
            self.materials.update(requirements.keys())
            self.materials.update(yields.keys())

        self.variable_bounds = [(0.0, None)] * len(self.processed_recipes)
        self.objective_coeffs = self._construct_objective()
        self.eq_matrix, self.balanced_materials = self._construct_balance_eqs()
        self.ineq_matrix, self.ineq_bounds = self._construct_resource_ineqs()

    def _construct_objective(self) -> List[float]:
        factors = []
        for idx, info in enumerate(self.processed_recipes, start=1):
            device_usage = 1.0 / info.productivity
            factors.append(device_usage + TIE_EPS * idx)
        return factors

    def _construct_balance_eqs(self) -> Tuple[List[List[float]], List[str]]:
        balanced_materials: List[str] = []
        eq_rows: List[List[float]] = []

        def material_balance(material: str) -> List[float]:
            row = []
            for info in self.processed_recipes:
                generated = info.yields.get(material, 0.0)
                required = info.requirements.get(material, 0.0)
                row.append(generated - required)
            return row

        if self.desired_product not in balanced_materials:
            balanced_materials.append(self.desired_product)
            eq_rows.append(material_balance(self.desired_product))

        intermediates = (
            self.materials
            - set(self.resource_limits.keys())
            - {self.desired_product}
        )
        for material in sorted(intermediates):
            balanced_materials.append(material)
            eq_rows.append(material_balance(material))

        return eq_rows, balanced_materials

    def _construct_resource_ineqs(self) -> Tuple[List[List[float]], List[float]]:
        ineq_rows: List[List[float]] = []
        bound_values: List[float] = []

        def material_balance(material: str) -> List[float]:
            row = []
            for info in self.processed_recipes:
                generated = info.yields.get(material, 0.0)
                required = info.requirements.get(material, 0.0)
                row.append(generated - required)
            return row

        for material, limit in self.resource_limits.items():
            row = material_balance(material)
            ineq_rows.append(row)
            bound_values.append(0.0)
            ineq_rows.append([-factor for factor in row])
            bound_values.append(float(limit))

        for device, limit in self.device_limits.items():
            limit_val = float(limit)
            if limit_val < 0:
                raise ValueError(f"negative machine cap for '{device}'")
            row = []
            methods = {title for title in self.devices_to_methods.get(device, [])}
            for info in self.processed_recipes:
                if info.title in methods:
                    row.append(1.0 / info.productivity)
                else:
                    row.append(0.0)
            ineq_rows.append(row)
            bound_values.append(limit_val)

        return ineq_rows, bound_values

    def _optimize_for_output(self, output: float):
        rhs_eq = []
        for material in self.balanced_materials:
            if material == self.desired_product:
                rhs_eq.append(output)
            else:
                rhs_eq.append(0.0)

        return linprog(
            self.objective_coeffs,
            A_ub=self.ineq_matrix if self.ineq_matrix else None,
            b_ub=self.ineq_bounds if self.ineq_bounds else None,
            A_eq=self.eq_matrix if self.eq_matrix else None,
            b_eq=rhs_eq if rhs_eq else None,
            bounds=self.variable_bounds,
            method="highs",
        )

    def optimize(self) -> Dict:
        exact = self._optimize_for_output(self.required_output)
        if exact.success and self._validate_result(exact.x, self.required_output):
            return self._format_optimal(exact.x)

        max_output = 0.0
        optimal_result = None
        min_bound, max_bound = 0.0, self.required_output
        for _ in range(50):
            midpoint = (min_bound + max_bound) / 2.0
            outcome = self._optimize_for_output(midpoint)
            if outcome.success and self._validate_result(outcome.x, midpoint):
                max_output = midpoint
                optimal_result = outcome.x
                min_bound = midpoint
            else:
                max_bound = midpoint

        if optimal_result is None:
            zero_outcome = self._optimize_for_output(0.0)
            optimal_result = zero_outcome.x if zero_outcome.success else np.zeros(len(self.processed_recipes))
            max_output = 0.0

        return self._format_suboptimal(max_output, optimal_result)

    def _validate_result(self, solution: np.ndarray, desired_output: float) -> bool:
        solution = np.array(solution)
        if np.any(solution < -1e-7):
            return False

        for material, row in zip(self.balanced_materials, self.eq_matrix):
            left_side = float(np.dot(row, solution))
            right_side = desired_output if material == self.desired_product else 0.0
            if abs(left_side - right_side) > 5e-7:
                return False

        if self.ineq_matrix:
            left_sides = np.dot(self.ineq_matrix, solution)
            for val, bound in zip(left_sides, self.ineq_bounds):
                if val - bound > 1e-6:
                    return False

        return True

    def _format_optimal(self, solution: np.ndarray) -> Dict:
        method_rates = self._method_outputs(solution)
        device_counts = self._device_requirements(solution)
        resource_usage = self._resource_demand(solution)
        return {
            "status": "ok",
            "per_recipe_crafts_per_min": method_rates,
            "per_machine_counts": device_counts,
            "raw_consumption_per_min": resource_usage,
        }

    def _format_suboptimal(self, output: float, solution: np.ndarray) -> Dict:
        method_rates = self._method_outputs(solution)
        device_counts = self._device_requirements(solution)
        resource_usage = self._resource_demand(solution)

        clues: List[str] = []
        for device, count in device_counts.items():
            limit = self.device_limits.get(device)
            if limit is None:
                continue
            if count >= float(limit) - 1e-6:
                clues.append(f"{device} cap")

        for material, limit in self.resource_limits.items():
            used = resource_usage.get(material, 0.0)
            if used >= float(limit) - 1e-6:
                clues.append(f"{material} supply")

        clues = sorted(set(clues))

        return {
            "status": "infeasible",
            "max_feasible_target_per_min": float(output),
            "bottleneck_hint": clues,
            "per_recipe_crafts_per_min": method_rates,
            "per_machine_counts": device_counts,
            "raw_consumption_per_min": resource_usage,
        }

    def _method_outputs(self, solution: np.ndarray) -> Dict[str, float]:
        output_dict = {}
        for info, value in zip(self.processed_recipes, solution):
            val = 0.0 if abs(value) < 1e-9 else float(value)
            output_dict[info.title] = val
        return {k: output_dict[k] for k in sorted(output_dict.keys())}

    def _device_requirements(self, solution: np.ndarray) -> Dict[str, float]:
        requirements = defaultdict(float)
        for info, output in zip(self.processed_recipes, solution):
            devices_required = output / info.productivity
            if devices_required > 1e-12:
                requirements[info.device] += devices_required
        return {device: float(val) for device, val in sorted(requirements.items())}

    def _resource_demand(self, solution: np.ndarray) -> Dict[str, float]:
        demand = {}
        for material in self.resource_limits.keys():
            total = 0.0
            for info, output in zip(self.processed_recipes, solution):
                generated = info.yields.get(material, 0.0) * output
                required = info.requirements.get(material, 0.0) * output
                total += required - generated
            total = 0.0 if abs(total) < 1e-9 else float(total)
            demand[material] = total
        return {k: demand[k] for k in sorted(demand.keys())}


def entry() -> None:
    try:
        config = json.load(sys.stdin)
        optimizer = ProductionOptimizer(config)
        outcome = optimizer.optimize()
    except Exception as error:  # noqa: BLE001
        outcome = {"status": "error", "message": str(error)}
    json.dump(outcome, sys.stdout, separators=(",", ":"))


if __name__ == "__main__":
    entry()