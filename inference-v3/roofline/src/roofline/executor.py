"""A persistent numerical owner; protocol output uses a dedicated inherited descriptor."""

import argparse
import os
import sys
import traceback

from .contracts import Attempt, Request
from .store import Store
from .transport import receive, send


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    # Large composed native entrypoints need room for their host argument frames.
    # Set this on the numerical process's main thread before loading an engine.
    import resource

    soft, hard = resource.getrlimit(resource.RLIMIT_STACK)
    wanted = 64 << 20
    if hard != resource.RLIM_INFINITY:
        wanted = min(wanted, hard)
    if soft != resource.RLIM_INFINITY and soft < wanted:
        resource.setrlimit(resource.RLIMIT_STACK, (wanted, hard))
    # Native compiler and Python logs stay on stderr, away from control frames.
    control = os.fdopen(os.dup(sys.stdout.fileno()), "wb", buffering=0)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    from ops.lab.ownership import exclusive_measurement

    from .integrations import integration

    owner, owner_key = None, None
    with Store(args.root) as store:
        try:
            while True:
                try:
                    message = receive(sys.stdin.buffer)
                except EOFError:
                    break
                try:
                    request = Request.model_validate(message["request"])
                    attempt = Attempt.model_validate(message["attempt"])
                    if request.operation == "characterize":
                        if owner is not None:
                            owner.close()
                            owner, owner_key = None, None
                        from .characterization import execute

                        with exclusive_measurement():
                            artifact = execute(attempt, store)
                        send(control, {"result": [], "artifacts": [artifact]})
                        continue
                    assert request.model is not None
                    key = request.experiment.model_dump(
                        exclude={"source", "against_source", "targets"}
                    )
                    key["model_artifact"] = request.model.sha256
                    key["model_location"] = request.model.locations[attempt.target_name]
                    key["device"] = attempt.target.device.model_dump()
                    if request.experiment.against_source:
                        if owner is not None and owner_key != key:
                            owner.close()
                            owner, owner_key = None, None
                        from .pairing import paired

                        with exclusive_measurement():
                            results, owner = paired(request, attempt, store, owner=owner)
                            owner_key = key if owner is not None else None
                        send(control, {"result": [m.model_dump(mode="json") for m in results]})
                        continue
                    with exclusive_measurement():
                        if owner is None or owner_key != key:
                            if owner is not None:
                                owner.close()
                            owner = integration(request.experiment.engine)(request, attempt, store)
                            owner_key = key
                        else:
                            if owner.request.inputs.get("component") != request.inputs.get(
                                "component"
                            ):
                                for runner in owner.runners.values():
                                    runner.close()
                                owner.runners.clear()
                            owner.request, owner.attempt = request, attempt
                            owner.experiment = request.experiment
                        if message.get("refresh"):
                            owner.refresh()
                        if request.operation == "prepare-inputs":
                            artifact = owner.capture_inputs()
                            send(control, {"result": [], "artifacts": [artifact]})
                            continue
                        results = owner.measure()
                        if any(m.status != "complete" for m in results):
                            owner.close()
                            owner, owner_key = None, None
                    send(control, {"result": [m.model_dump(mode="json") for m in results]})
                except BaseException as exc:
                    traceback.print_exc()
                    if owner is not None:
                        try:
                            owner.close()
                        except Exception:
                            traceback.print_exc()
                        owner, owner_key = None, None
                    send(
                        control,
                        {"error": f"{type(exc).__name__}: {exc}", "code": type(exc).__name__},
                    )
        finally:
            if owner is not None:
                owner.close()


if __name__ == "__main__":
    main()
