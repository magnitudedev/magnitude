"""Adversarial checks for the independently encoded scheduling semantics."""
import unittest

from cp_sat import Encoder, cp_model


def status(schedule, values, guards=()):
    source = {
        "variables": [
            {"name": str(i), "domain": {"runs": [{"first": v, "last": v, "step": 1}]}}
            for i, v in enumerate(values)
        ],
        "factors": [{"guards": list(guards), "kind": {"Constraint": {"Schedule": schedule}}}],
    }
    encoder = Encoder(source)
    encoder.encode()
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    solver.parameters.random_seed = 0
    solver.parameters.relative_gap_limit = 0
    solver.parameters.absolute_gap_limit = 0
    result = solver.solve(encoder.model)
    if result == cp_model.MODEL_INVALID:
        raise AssertionError(solver.solution_info())
    return result


def interval(start, end, presence=()):
    return {"start": start, "end": end, "presence": list(presence)}


class SchedulingSemantics(unittest.TestCase):
    def test_active_times_are_nonnegative_but_absent_times_are_private(self):
        activity = {"start": 0, "duration": 1, "end": 2, "presence": 3}
        self.assertEqual(status({"Activity": activity}, [-1, 1, 0, 1]), cp_model.INFEASIBLE)
        self.assertEqual(status({"Activity": activity}, [-1, -2, -3, 0]), cp_model.OPTIMAL)

    def test_precedence_activates_only_when_both_events_are_present(self):
        relation = {"Precedence": {
            "before": {"time": 0, "presence": 2},
            "after": {"time": 1, "presence": 3}, "lag": 0,
        }}
        self.assertEqual(status(relation, [-1, 0, 1, 0]), cp_model.OPTIMAL)
        self.assertEqual(status(relation, [-1, 0, 1, 1]), cp_model.INFEASIBLE)

    def test_no_overlap_ignores_empty_intervals_and_releases_at_end(self):
        relation = {"NoOverlap": {"intervals": [interval(0, 1), interval(2, 3)]}}
        self.assertEqual(status(relation, [0, 10, 5, 5]), cp_model.OPTIMAL)
        self.assertEqual(status(relation, [0, 10, 5, 6]), cp_model.INFEASIBLE)
        self.assertEqual(status(relation, [0, 10, 10, 11]), cp_model.OPTIMAL)

    def test_absent_capacity_reservations_do_not_constrain_demands(self):
        relation = {"Cumulative": {"capacity": 1, "reservations": [
            {"interval": interval(0, 1, [3]), "demand": {"Variable": 2}},
        ]}}
        self.assertEqual(status(relation, [-2, -1, -3, 0]), cp_model.OPTIMAL)
        self.assertEqual(status(relation, [0, 0, -3, 1]), cp_model.INFEASIBLE)
        self.assertEqual(status(relation, [-2, -1, 1, 1]), cp_model.INFEASIBLE)
        guard = [{"variable": 4, "value": 1}]
        self.assertEqual(status(relation, [-2, -1, -3, 1, 0], guard), cp_model.OPTIMAL)
        self.assertEqual(status(relation, [0, 0, -3, 1, 1], guard), cp_model.INFEASIBLE)

    def test_empty_intervals_consume_no_capacity(self):
        relation = {"Cumulative": {"capacity": 0, "reservations": [
            {"interval": interval(0, 1), "demand": {"Constant": 99}},
        ]}}
        self.assertEqual(status(relation, [3, 3]), cp_model.OPTIMAL)
        self.assertEqual(status(relation, [3, 4]), cp_model.INFEASIBLE)

    def test_completion_remains_nonnegative_with_no_activities(self):
        relation = {"ActivityCumulative": {"completion": 0, "capacity": 0, "activities": []}}
        self.assertEqual(status(relation, [-1]), cp_model.INFEASIBLE)
        self.assertEqual(status(relation, [0]), cp_model.OPTIMAL)
        self.assertEqual(status(relation, [-1, 0], [{"variable": 1, "value": 1}]), cp_model.OPTIMAL)


if __name__ == "__main__":
    unittest.main()
