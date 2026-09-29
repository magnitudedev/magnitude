"""Compose actual sequential invocations without adding workload nodes to the model."""

from collections import defaultdict

from .records import Expression, Manifest, SerialStages, identity


def sequential(manifests):
    """One realized model relation over an explicitly serial sequence of forwards.

    Work and useful quantities add across invocations. Repeated constants remain
    distinct boundary demands under their declared memory-path assumptions. This
    operation never assumes that sibling formulas inside a forward are serial.
    """
    manifests = tuple(manifests)
    if not manifests:
        raise ValueError("a sequential realization needs at least one invocation")
    parameters, unresolved, obligations, instances = {}, {}, [], defaultdict(list)
    stages = defaultdict(list)
    certificates = []

    def rename(expression, prefix):
        return expression.model_copy(
            update={
                "name": prefix + expression.name if expression.name is not None else None,
                "arguments": tuple(rename(t, prefix) for t in expression.arguments),
            }
        )

    def add(expressions):
        expressions = tuple(expressions)
        return (
            expressions[0] if len(expressions) == 1 else Expression(op="add", arguments=expressions)
        )

    for index, manifest in enumerate(manifests):
        prefix = f"invocation:{index}:"
        parameters.update((prefix + k, v) for k, v in manifest.parameters.items())
        unresolved.update((prefix + k, v) for k, v in manifest.unresolved.items())
        for obligation in manifest.obligations:
            obligations.append(
                obligation.model_copy(
                    update={
                        "identity": prefix + obligation.identity,
                        "origins": tuple(prefix + n for n in obligation.origins),
                        "amount": rename(obligation.amount, prefix),
                    }
                )
            )
        for formula in manifest.formulas:
            instances[formula.component].append((prefix, formula))
            nodes = set(formula.nodes)
            stage = tuple(
                prefix + o.identity
                for o in manifest.obligations
                if set(o.origins) <= nodes
                and (o.boundary is None or o.boundary == formula.component)
            )
            if stage:
                stages[formula.component].append(stage)
        certificates.extend(
            s.model_copy(
                update={
                    "identity": prefix + s.identity,
                    "stages": tuple(tuple(prefix + item for item in stage) for stage in s.stages),
                }
            )
            for s in manifest.serial_stages
        )
    formulas = []
    for component, occurrences in instances.items():
        original = occurrences[0][1]
        if any(
            (f.definition, f.version, f.parent, f.primary.name, f.primary.unit)
            != (
                original.definition,
                original.version,
                original.parent,
                original.primary.name,
                original.primary.unit,
            )
            for _, f in occurrences
        ):
            raise ValueError("sequential invocations disagree on a component contract")
        quantities = defaultdict(list)
        for prefix, formula in occurrences:
            for q in formula.quantities:
                quantities[(q.name, q.unit.name, q.unit.dimension)].append((prefix, q))
        useful = tuple(
            items[0][1].model_copy(
                update={"expression": add(rename(q.expression, prefix) for prefix, q in items)}
            )
            for items in quantities.values()
        )
        formulas.append(
            original.model_copy(
                update={
                    "occurrence": None,
                    "semantics": identity(
                        tuple((f.semantics, prefix) for prefix, f in occurrences)
                    ),
                    "primary": original.primary.model_copy(
                        update={
                            "expression": add(
                                rename(f.primary.expression, prefix) for prefix, f in occurrences
                            )
                        }
                    ),
                    "quantities": useful,
                    "nodes": tuple(prefix + n for prefix, f in occurrences for n in f.nodes),
                    "inputs": tuple(prefix + n for prefix, f in occurrences for n in f.inputs),
                    "outputs": tuple(prefix + n for prefix, f in occurrences for n in f.outputs),
                    "dependencies": tuple(
                        sorted({d for _, f in occurrences for d in f.dependencies})
                    ),
                }
            )
        )
        if len(stages[component]) > 1:
            certificates.append(
                SerialStages(
                    identity="forward-sequence:" + component,
                    component=component,
                    stages=tuple(stages[component]),
                    rule="observed sequential forward completion",
                )
            )
    return Manifest(
        graph=identity(tuple(m.graph for m in manifests)),
        formulas=tuple(formulas),
        obligations=tuple(obligations),
        parameters=parameters,
        unresolved=unresolved,
        conditions=tuple(sorted({c for m in manifests for c in m.conditions})),
        serial_stages=tuple(certificates),
        numerical_graph={
            "composition": "sequential-forward-invocations",
            "invocations": [identity(m) for m in manifests],
        },
    )
