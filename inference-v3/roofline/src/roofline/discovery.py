"""Isolated metadata discovery; never constructs a device runtime."""

import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .contracts import Experiment, Model, Source, Target, digest, encoded


def preflight(root, store, message):
    from .environment import available_environment

    payload = message["payload"]
    model = Model.model_validate(payload["model"])
    target = Target.model_validate(payload["target"])
    experiment = Experiment.model_validate(payload["experiment"])
    path = Path(model.locations[experiment.targets[0]])
    missing = []
    if not path.is_file():
        missing.append(f"model file is missing: {path}")
    elif not os.access(path, os.R_OK):
        missing.append(f"model file is unreadable: {path}")
    if target.device.backend == "metal" and sys.platform != "darwin":
        missing.append("Metal requires a macOS host")
    if target.device.backend == "cuda" and sys.platform != "linux":
        missing.append("this CUDA integration requires a Linux host")
    prepared = available_environment(root, message["dependency_id"], store)
    return {
        "status": "unavailable" if missing else "ready" if prepared else "preparation-required",
        "unresolved": missing,
        "environment_ready": prepared is not None,
        "artifact_verified": False,
        "scopes": [],
        "preparation": [
            *([] if prepared else ["install locked execution dependencies and build compiler"]),
            "verify complete artifact checksum and device capability",
            "prepare workload inputs and compile selected execution",
        ],
    }


def discover(config, experiment):
    from .environment import dependency_identity
    from .service import transfer_source
    from .sources import capture
    from .store import Store
    from .transport import WorkerClient

    with Store(config.workspace) as store:
        source = (
            store.source(experiment.source) if experiment.source else capture(config.root, store)
        )
    metadata = Source(
        files=tuple(f for f in source.files if f.path.startswith(("src/", "roofline/src/")))
    )

    def resolve(name):
        target = config.targets[name]
        payload = {
            "model": config.models[experiment.model].model_dump(),
            "target": target.model_dump(),
            "experiment": experiment.model_copy(update={"targets": (name,)}).model_dump(),
        }
        try:
            with Store(config.workspace) as store, WorkerClient(target, config.root) as client:
                message = {"payload": payload, "dependency_id": dependency_identity(source)}
                result = client.call({"op": "preflight", **message})
                if result["status"] == "unavailable":
                    return result
                if not result["environment_ready"]:
                    result["unresolved"].append(
                        "scope tracing needs a matching execution environment; "
                        "measurement prepares it automatically"
                    )
                    return result
                transfer_source(client, store, metadata)
                result.update(
                    client.call(
                        {
                            "op": "discover",
                            "source_id": metadata.source_id,
                            **message,
                        }
                    )
                )
                if "/" in experiment.scope and experiment.scope not in {
                    s["selector"] for s in result["scopes"]
                }:
                    result["status"] = "unavailable"
                    result["unresolved"].append(
                        "selected scope does not exist in the production trace"
                    )
                from .query import PAGE

                result["artifact_id"] = store.put_blob(
                    encoded(
                        {
                            "model": experiment.model,
                            "source_id": source.source_id,
                            "target": name,
                            **result,
                        }
                    )
                )
                if result["scopes"]:
                    structure = {
                        "model": experiment.model,
                        "artifact": config.models[experiment.model].sha256,
                        "source_id": source.source_id,
                        "scopes": result["scopes"],
                        "performance_manifest": result["performance_manifest"],
                    }
                    store.put("model-structure", digest(structure), structure)
                result["scope_count"] = len(result["scopes"])
                result["selected_scope"] = next(
                    (s for s in result["scopes"] if s["selector"] == experiment.scope), None
                )
                result["scopes"] = result["scopes"][:PAGE]
                return result
        except (OSError, RuntimeError, ValueError, KeyError) as exc:
            return {"status": "unavailable", "scopes": [], "unresolved": [str(exc)]}

    with ThreadPoolExecutor(max_workers=min(8, len(experiment.targets))) as pool:
        targets = dict(zip(experiment.targets, pool.map(resolve, experiment.targets), strict=True))
    result = {
        "model": experiment.model,
        "source_id": source.source_id,
        "targets": targets,
    }
    if experiment.input_source:
        result["input_producer"] = discover(
            config,
            experiment.model_copy(
                update={
                    "source": experiment.input_source,
                    "targets": (experiment.input_target,),
                    "input_source": None,
                    "input_target": None,
                }
            ),
        )
    return result


def main():
    payload = json.load(sys.stdin)
    model, target = Model.model_validate(payload["model"]), Target.model_validate(payload["target"])
    experiment = Experiment.model_validate(payload["experiment"])
    # Keep dependency/compiler diagnostics away from the JSON result.
    output = os.fdopen(os.dup(1), "w")
    os.dup2(2, 1)
    from .integrations.magnitude import discover as production_scopes

    if experiment.engine != "magnitude":
        result = {
            "scopes": [],
            "unresolved": ["reference integration exposes enclosing scopes only"],
        }
    else:
        result = production_scopes(model, target, experiment)
    output.write(json.dumps(result))
    output.flush()


if __name__ == "__main__":
    main()
