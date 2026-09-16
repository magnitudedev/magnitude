"""One model relation shared by the command API and Textual browser."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Component:
    key: str
    parent: str | None
    label: str
    relation: dict


@dataclass(frozen=True)
class ModelTree:
    model: str
    nodes: dict[str, Component]
    analysis: dict
    notice: str


def performance_model(store, model):
    from formula_performance.revision import revision

    try:
        cached = store.get("performance-model", model)
    except KeyError:
        cached = None
    if cached is not None and cached.get("analysis_revision") == revision():
        return cached
    return store.derive(model)


def build_tree(store, model, configured=None):
    report = performance_model(store, model)
    nodes = {
        key: Component(key, r["parent"], model.split(":")[0] if key == "" else r["label"], r)
        for key, r in report["components"].items()
    }
    notice = ""
    if not nodes:
        notice = "No published formula model. Discover model scopes or make a measurement."
    if report.get("historical_without_contract"):
        notice += (
            f" {report['historical_without_contract']} historical measurements "
            "lack analytical contracts."
        )
    return ModelTree(model, nodes, report, notice.strip())
