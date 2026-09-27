"""Explicit commands for measurement, evidence and execution hosts."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from .config import Configuration, project_root
from .contracts import Attempt, Experiment, Protocol, Request, identity, now
from .query import Queries, compare
from .service import TERMINAL
from .sources import capture, fixture_inputs
from .store import Store
from .transport import DEFAULT_WORKER_ROOT, WorkerClient, ensure_service, local_call


def count(value):
    value = value.lower()
    return int(value[:-1]) * 1024 if value.endswith("k") else int(value)


def experiment_options(parser):
    parser.add_argument("--model", required=True)
    parser.add_argument("--workload", choices=("prose",), default="prose")
    parser.add_argument("--context", type=count, default=2048)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--scope", default="decode")
    parser.add_argument("--step", type=int)
    parser.add_argument("--targets", default="local")
    parser.add_argument("--engine", choices=("magnitude", "llama.cpp"), default="magnitude")
    parser.add_argument("--source")


def parser():
    root = argparse.ArgumentParser(prog="roofline", description=__doc__)
    commands = root.add_subparsers(dest="command")
    measure = commands.add_parser("measure", help="execute and store a performance measurement")
    experiment_options(measure)
    measure.add_argument("--against-source")
    measure.add_argument("--input-source")
    measure.add_argument("--input-target")
    measure.add_argument("--samples", type=int, default=3)
    measure.add_argument("--warmups", type=int, default=1)
    measure.add_argument("--deadline", type=int, default=3600)
    measure.add_argument("--dry-run", action="store_true")
    query = commands.add_parser("query", help="read performance or exact recorded evidence")
    select = query.add_mutually_exclusive_group(required=True)
    for name in ("model", "measurement", "artifact", "source", "request"):
        select.add_argument("--" + name)
    for name in ("workload", "scope", "engine", "targets"):
        query.add_argument("--" + name)
    query.add_argument("--context", type=count)
    query.add_argument("--cursor")
    comparison = commands.add_parser("compare", help="compare two existing measurements")
    comparison.add_argument("--baseline", required=True)
    comparison.add_argument("--candidate", required=True)
    scopes = commands.add_parser("scopes", help="discover production formula selectors")
    experiment_options(scopes)
    commands.add_parser("models")
    commands.add_parser("workloads")
    workers = commands.add_parser("workers")
    actions = workers.add_subparsers(dest="action", required=True)
    actions.add_parser("list")
    for name in ("setup", "pause", "resume"):
        action = actions.add_parser(name)
        action.add_argument("target")
    cancel = commands.add_parser("cancel")
    cancel.add_argument("--request", required=True)
    characterize = commands.add_parser("characterize")
    characterize.add_argument("target")
    export = commands.add_parser("export")
    export.add_argument("--measurement", action="append", required=True)
    export.add_argument("--output", type=Path, required=True)
    imports = commands.add_parser("import")
    imports.add_argument("bundle", type=Path)
    sessions = commands.add_parser("import-session")
    sessions.add_argument("directory", type=Path)
    return root


def experiment(args):
    fields = {
        key: getattr(args, key)
        for key in ("model", "workload", "context", "steps", "scope", "step", "engine", "source")
    }
    fields["targets"] = tuple(args.targets.split(","))
    if args.command == "measure":
        fields["against_source"] = args.against_source
        fields["input_source"] = args.input_source
        fields["input_target"] = args.input_target
        fields["protocol"] = Protocol(
            samples=args.samples, warmups=args.warmups, deadline_seconds=args.deadline
        )
    return Experiment(**fields)


def run(args, config):
    command = args.command
    if command is None:
        from .tui import RooflineApp

        RooflineApp(config).run()
        return None
    if command == "models":
        return {name: model.model_dump() for name, model in config.models.items()}
    if command == "workloads":
        return {
            "prose": {
                "recipe": "prose.moby-dick",
                "continuation": "fixture tokens",
                "context": "positive token count",
                "steps": "1..4096",
            }
        }
    if command in ("query", "compare"):
        with Store(config.workspace, readonly=True) as store:
            queries = Queries(store, config.models, config.targets)
            if command == "compare":
                return compare(store.measurement(args.baseline), store.measurement(args.candidate))
            filters = {
                key: getattr(args, key)
                for key in ("workload", "context", "scope", "engine", "targets")
                if getattr(args, key) is not None
            }
            if "targets" in filters:
                filters["targets"] = filters["targets"].split(",")
            if args.model:
                return queries.model(args.model, filters=filters, cursor=args.cursor)
            if filters:
                raise ValueError("condition filters apply only to query --model")
            if args.measurement:
                return queries.measurement(args.measurement, cursor=args.cursor)
            if args.artifact:
                return queries.artifact(args.artifact, cursor=args.cursor)
            if args.source:
                return queries.source(args.source, cursor=args.cursor)
            if args.cursor:
                raise ValueError("request progress is not paginated")
            return store.request(args.request).model_dump()
    if command in ("measure", "scopes"):
        selected = experiment(args)
        config.validate(selected)
        if command == "scopes" or args.dry_run:
            from .discovery import discover

            result = discover(config, selected)
            if command == "measure":
                result["preparation"] = [
                    "verify artifact checksums",
                    "prepare workload tokens",
                    "load model and compile affected operations",
                    "check and sample",
                ]
            return result
        with Store(config.workspace) as store:
            source = (
                store.source(selected.source) if selected.source else capture(config.root, store)
            )
            if selected.against_source:
                store.source(selected.against_source)
            if selected.input_source:
                producer = store.source(selected.input_source)
                if fixture_inputs(producer, store) != fixture_inputs(source, store):
                    raise ValueError(
                        "shared input producer and consumer require the same fixture recipe"
                    )
            selected = selected.model_copy(update={"source": source.source_id})
            request = Request(
                request_id=identity(),
                created=now(),
                experiment=selected,
                model=config.models[selected.model],
                inputs=fixture_inputs(source, store),
                attempts=(
                    (
                        Attempt(
                            attempt_id=identity(),
                            target_name=selected.input_target,
                            target=config.targets[selected.input_target],
                            role="inputs",
                        ),
                    )
                    if selected.input_target
                    else ()
                )
                + tuple(
                    Attempt(attempt_id=identity(), target_name=name, target=config.targets[name])
                    for name in selected.targets
                ),
            )
            store.put("request", request.request_id, request)
        ensure_service(config.workspace, config.root, "coordinator")
        print(
            json.dumps({"request_id": request.request_id, "status": "submitted"}),
            file=sys.stderr,
            flush=True,
        )
        try:
            while True:
                with Store(config.workspace, readonly=True) as store:
                    request = store.request(request.request_id)
                    if all(a.status in TERMINAL for a in request.attempts):
                        queries = Queries(store, config.models, config.targets)
                        return {
                            "request": request.model_dump(),
                            "measurements": [
                                queries.measurement(m)
                                for a in request.attempts
                                for m in a.measurement_ids
                            ],
                        }
                time.sleep(0.3)
        except KeyboardInterrupt:
            return {
                "request_id": request.request_id,
                "status": "continues in background",
                "query": f"roofline query --request {request.request_id}",
            }
    if command == "cancel":
        ensure_service(config.workspace, config.root, "coordinator")
        return {
            "cancelled": local_call(config.workspace, {"op": "cancel", "request_id": args.request})
        }
    if command == "workers":
        if args.action == "list":
            return {
                name: {
                    "connection": t.connection.model_dump(),
                    "worker_root": t.worker_root or DEFAULT_WORKER_ROOT,
                    "device": t.device.model_dump(),
                    "models": [
                        key for key, model in config.models.items() if name in model.locations
                    ],
                }
                for name, t in config.targets.items()
            }
        target = config.targets[args.target]
        if args.action == "setup":
            from .setup import setup_worker

            return setup_worker(config, args.target)
        with WorkerClient(target, config.root) as client:
            return client.call({"op": args.action})
    if command in ("export", "import", "import-session"):
        from .bundles import export_bundle, import_bundle, import_session

        with Store(config.workspace) as store:
            if command == "export":
                return export_bundle(store, args.measurement, args.output)
            if command == "import":
                return import_bundle(store, args.bundle)
            return import_session(store, config, args.directory)
    if command == "characterize":
        from .characterization import submit

        return submit(config, args.target)
    raise ValueError(f"unsupported command: {command}")


def main():
    args = parser().parse_args()
    try:
        result = run(args, Configuration(project_root()))
        if result is not None:
            print(json.dumps(result, sort_keys=True, default=str))
    except (ValueError, KeyError, OSError, RuntimeError) as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
