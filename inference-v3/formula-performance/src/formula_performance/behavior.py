"""Hardware-normalized behavior hypotheses with held-out subsequent checks."""


def characterize(relation):
    predictions = []
    points = relation["points"]
    for baseline in points:
        efficiency = baseline["attainment"]
        if efficiency is None or efficiency > 1:
            continue
        # The equation is parameterized in H. Binding a later hardware point is
        # a check of the earlier hypothesis, not a retrospective fitted average.
        prediction = {
            "baseline": baseline["observation"],
            "created": baseline["created"],
            "kind": "conditional",
            "efficiency": efficiency,
            "expression": "rate(x,H) = baseline_efficiency * ceiling(x,H)",
            "assumptions": (
                "same numerical instance, implementation and timing boundary",
                "hardware-normalized efficiency transfers between these resource regimes",
            ),
            "validation": [],
        }
        for actual in points:
            if (
                actual["created"] <= baseline["created"]
                or not actual["qualified"]
                or actual["hardware_binding"] == baseline["hardware_binding"]
                or actual["implementation"] != baseline["implementation"]
                or actual["semantics"] != baseline["semantics"]
                or actual["coordinates"] != baseline["coordinates"]
                or actual["boundary"] != baseline["boundary"]
                or actual["roofline"]["ceiling"] is None
            ):
                continue
            predicted = efficiency * actual["roofline"]["ceiling"]
            prediction["validation"].append(
                {
                    "observation": actual["observation"],
                    "predicted_rate": predicted,
                    "observed_rate": actual["rate"],
                    "error_rate": actual["rate"] - predicted,
                }
            )
        predictions.append(prediction)
    return predictions
