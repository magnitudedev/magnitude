#!/usr/bin/env python3
"""Independent exact CP-SAT encoding of the public finite solver vocabulary.

This optional experiment dependency never participates in production selection.
The Rust caller revalidates a returned witness against the original model.
"""
import argparse
import importlib.metadata
import json
import time
from ortools.sat.python import cp_model

PIN = "9.14.6206"
if importlib.metadata.version("ortools") != PIN:
    raise RuntimeError(f"reference requires ortools=={PIN}")


class Encoder:
    def __init__(self, source):
        self.source = source
        self.model = cp_model.CpModel()
        self.variables = []
        self.ranges = []
        self.costs = []
        self.unresolved = False
        for variable in source["variables"]:
            runs = variable["domain"]["runs"]
            intervals = []
            for run in runs:
                if run["step"] == 1:
                    intervals.append([run["first"], run["last"]])
                else:
                    count = (run["last"] - run["first"]) // run["step"] + 1
                    if count > 1_000_000:
                        raise ValueError("CP-SAT reference progression exceeds explicit conversion budget")
                    intervals.extend([value, value] for value in range(run["first"], run["last"] + 1, run["step"]))
            self.variables.append(self.model.new_int_var_from_domain(cp_model.Domain.from_intervals(intervals), variable["name"]))
            self.ranges.append((runs[0]["first"], runs[-1]["last"]))

    def conjunction(self, literals):
        if not literals:
            return self.model.new_constant(1)
        if len(literals) == 1:
            return literals[0]
        result = self.model.new_bool_var("conjunction")
        self.model.add_bool_and(literals).only_enforce_if(result)
        self.model.add_bool_or([literal.Not() for literal in literals]).only_enforce_if(result.Not())
        return result

    def literal(self, literal):
        result = self.model.new_bool_var("equality")
        variable = self.variables[literal["variable"]]
        self.model.add(variable == literal["value"]).only_enforce_if(result)
        self.model.add(variable != literal["value"]).only_enforce_if(result.Not())
        return result

    def affine(self, terms):
        return sum(term["coefficient"] * self.variables[term["variable"]] for term in terms)

    def interval(self, interval, guards):
        active = self.conjunction(guards + [self.literal({"variable": p, "value": 1}) for p in interval["presence"]])
        start, end = self.variables[interval["start"]], self.variables[interval["end"]]
        self.model.add(start >= 0).only_enforce_if(active)
        self.model.add(end >= 0).only_enforce_if(active)
        duration = self.model.new_int_var(0, max(0, self.ranges[interval["end"]][1] - self.ranges[interval["start"]][0]), "interval-duration")
        return self.model.new_optional_interval_var(start, duration, end, active, "reservation"), active

    def schedule(self, kind, value, guards):
        if kind == "Activity":
            active = list(guards)
            if value["presence"] is not None:
                active.append(self.literal({"variable": value["presence"], "value": 1}))
            for variable in (value["start"], value["duration"], value["end"]):
                self.model.add(self.variables[variable] >= 0).only_enforce_if(active)
            self.model.add(self.variables[value["end"]] == self.variables[value["start"]] + self.variables[value["duration"]]).only_enforce_if(active)
        elif kind == "Precedence":
            active = list(guards)
            for event in [value["before"], value["after"]]:
                if event["presence"] is not None:
                    active.append(self.literal({"variable": event["presence"], "value": 1}))
            for event in [value["before"], value["after"]]:
                self.model.add(self.variables[event["time"]] >= 0).only_enforce_if(active)
            self.model.add(self.variables[value["before"]["time"]] + value["lag"] <= self.variables[value["after"]["time"]]).only_enforce_if(active)
        elif kind == "NoOverlap":
            # Cumulative capacity one ignores empty intervals, as the source
            # relation requires. CP-SAT no_overlap also orders zero-size events.
            self.schedule("Cumulative", {
                "capacity": 1,
                "reservations": [{"interval": interval, "demand": {"Constant": 1}}
                                 for interval in value["intervals"]],
            }, guards)
        elif kind == "Cumulative":
            intervals, demands = [], []
            for reservation in value["reservations"]:
                interval, active = self.interval(reservation["interval"], guards)
                intervals.append(interval)
                demand_kind, demand = next(iter(reservation["demand"].items()))
                if demand_kind == "Constant":
                    demands.append(demand)
                else:
                    # CP-SAT requires globally nonnegative demand domains. An
                    # absent source reservation leaves its demand unconstrained.
                    masked = self.model.new_int_var(0, max(0, self.ranges[demand][1]), "active-demand")
                    self.model.add(masked == self.variables[demand]).only_enforce_if(active)
                    self.model.add(masked == 0).only_enforce_if(active.Not())
                    demands.append(masked)
            self.model.add_cumulative(intervals, demands, value["capacity"])
        elif kind == "ActivityCumulative":
            self.model.add(self.variables[value["completion"]] >= 0).only_enforce_if(guards)
            reservations = []
            for entry in value["activities"]:
                activity = entry["activity"]
                self.schedule("Activity", activity, guards)
                active = list(guards)
                if activity["presence"] is not None:
                    active.append(self.literal({"variable": activity["presence"], "value": 1}))
                self.model.add(self.variables[activity["end"]] <= self.variables[value["completion"]]).only_enforce_if(active)
                reservations.append({"interval": {"start": activity["start"], "end": activity["end"], "presence": [] if activity["presence"] is None else [activity["presence"]]}, "demand": entry["demand"]})
            self.schedule("Cumulative", {"capacity": value["capacity"], "reservations": reservations}, guards)
        else:
            raise ValueError(f"unsupported scheduling constraint {kind}")

    def constraint(self, kind, value, guards):
        if kind == "Equal":
            self.model.add(self.variables[value["left"]] == self.variables[value["right"]]).only_enforce_if(guards)
        elif kind == "NotEqual":
            self.model.add(self.variables[value["left"]] != self.variables[value["right"]]).only_enforce_if(guards)
        elif kind == "LinearLe":
            self.model.add(self.affine(value["terms"]) <= value["rhs"]).only_enforce_if(guards)
        elif kind == "ExactlyOne":
            self.model.add(sum(self.variables[v] for v in value["variables"]) == 1).only_enforce_if(guards)
        elif kind == "BoolAnd":
            inputs = [self.variables[v] for v in value["inputs"]]
            output = self.variables[value["output"]]
            for v in inputs:
                self.model.add(output <= v).only_enforce_if(guards)
            self.model.add(output >= sum(inputs) - len(inputs) + 1).only_enforce_if(guards)
        elif kind == "Implies":
            self.model.add_bool_or([self.literal(value["consequence"])]).only_enforce_if(guards + [self.literal(value["premise"])])
        elif kind == "Table":
            self.model.add_allowed_assignments([self.variables[v] for v in value["variables"]], value["tuples"]).only_enforce_if(guards)
        elif kind == "InDomain":
            domain = value["domain"]["runs"]
            allowed = []
            for run in domain:
                count = (run["last"] - run["first"]) // run["step"] + 1
                if count > 1_000_000:
                    raise ValueError("reference InDomain conversion exceeds explicit budget")
                allowed.extend([[n] for n in range(run["first"], run["last"] + 1, run["step"])])
            self.model.add_allowed_assignments([self.variables[value["variable"]]], allowed).only_enforce_if(guards)
        elif kind == "InactiveValue":
            self.model.add(self.variables[value["variable"]] == value["inactive"]).only_enforce_if(guards + [self.literal(value["active"]).Not()])
        elif kind == "Schedule":
            self.schedule(*next(iter(value.items())), guards)
        else:
            raise ValueError(f"unsupported constraint {kind}")

    def cost(self, kind, value, guards):
        active = self.conjunction(guards)
        if kind == "Constant":
            self.costs.append(value * active)
            return
        if kind == "Linear":
            upper = value["constant"] + sum(t["coefficient"] * self.ranges[t["variable"]][1 if t["coefficient"] >= 0 else 0] for t in value["terms"])
        elif kind == "Table":
            upper = max(cost for _, cost in value["entries"])
        else:
            raise ValueError(f"unsupported cost {kind}")
        if not 0 <= upper < 2**62:
            raise ValueError("objective outside CP-SAT reference integer envelope")
        cost = self.model.new_int_var(0, upper, "cost")
        self.model.add(cost == 0).only_enforce_if(active.Not())
        if kind == "Linear":
            self.model.add(cost == value["constant"] + self.affine(value["terms"])).only_enforce_if(active)
        else:
            tuples = [assignment + [objective] for assignment, objective in value["entries"]]
            self.model.add_allowed_assignments([self.variables[v] for v in value["variables"]] + [cost], tuples).only_enforce_if(active)
        self.costs.append(cost)

    def factors(self, factors, outer_guards):
        for factor in factors:
            guards = outer_guards + [self.literal(literal) for literal in factor["guards"]]
            category, payload = next(iter(factor["kind"].items()))
            if category == "Unresolved":
                self.unresolved = True
                continue
            if category == "Fragment":
                self.factors(payload["factors"], guards)
                continue
            kind, value = next(iter(payload.items()))
            if category == "Constraint":
                self.constraint(kind, value, guards)
            else:
                self.cost(kind, value, guards)
    def encode(self):
        self.factors(self.source["factors"], [])
        self.model.minimize(sum(self.costs))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("--seconds", type=float, default=60)
    args = parser.parse_args()
    start = time.perf_counter()
    with open(args.input, encoding="utf8") as file:
        instance = json.load(file)
    encoder = Encoder(instance["model"])
    encoder.encode()
    result = {"method": f"CP-SAT {PIN}", "assignments": 0, "reason": None}
    if encoder.unresolved:
        result.update(outcome="Incomplete", reason="unresolved domain coverage")
    else:
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = args.seconds
        solver.parameters.num_search_workers = 1
        solver.parameters.random_seed = 0
        solver.parameters.relative_gap_limit = 0
        solver.parameters.absolute_gap_limit = 0
        status = solver.solve(encoder.model)
        if status == cp_model.OPTIMAL:
            # Read integer expressions, never round a floating objective value.
            result.update(outcome={"Optimal": sum(solver.value(term) for term in encoder.costs)}, witness=[solver.value(v) for v in encoder.variables])
        elif status == cp_model.INFEASIBLE:
            result.update(outcome="Infeasible")
        elif status == cp_model.MODEL_INVALID:
            raise ValueError(solver.solution_info())
        else:
            result.update(outcome="Incomplete", reason=solver.status_name(status))
        result["assignments"] = solver.num_branches
    result["elapsed_ms"] = (time.perf_counter() - start) * 1000
    print(json.dumps(result))


if __name__ == "__main__":
    main()
