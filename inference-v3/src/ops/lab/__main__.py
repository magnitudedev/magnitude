"""Run targeted measurements or browse/import portable model performance evidence."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    browse = commands.add_parser("browse", help="read-only model overview")
    browse.add_argument("store", type=Path)
    export = commands.add_parser("export", help="export evidence for remote transfer")
    export.add_argument("store", type=Path)
    export.add_argument("bundle", type=Path)
    ingest = commands.add_parser("import", help="validate and merge evidence without execution")
    ingest.add_argument("store", type=Path)
    ingest.add_argument("bundle", type=Path)
    run = commands.add_parser("run", help="run a request through a production configuration factory")
    run.add_argument("request", type=Path)
    run.add_argument("--factory", required=True, help="Python module:callable returning Configuration")
    run.add_argument("--store", type=Path, required=True)
    run.add_argument("--bundle", type=Path)
    describe = commands.add_parser("describe", help="list production formula scopes without execution")
    describe.add_argument("--factory", required=True)
    describe.add_argument("--arguments", type=Path, required=True, help="JSON factory arguments")
    args = parser.parse_args()
    from .store import ObservationStore

    if args.command == "run":
        from .execution import MeasurementRequest, execute_request
        request = MeasurementRequest.model_validate_json(args.request.read_text())
        try:
            execute_request(args.factory, request, args.store)
        finally:
            if args.bundle and args.store.exists():
                from .bundles import export_bundle
                with ObservationStore(args.store, read_only=True) as store:
                    export_bundle(store, args.bundle)
    elif args.command == "describe":
        from .execution import describe_configuration
        print(json.dumps(describe_configuration(args.factory, json.loads(args.arguments.read_text())), indent=2))
    elif args.command == "browse":
        from .model_view import ModelApp
        with ObservationStore(args.store, read_only=True) as store:
            ModelApp(store).run()
    elif args.command == "import":
        from .bundles import import_bundle
        with ObservationStore(args.store) as store:
            print(f"Imported {import_bundle(store, args.bundle)} model executions")
    if args.command == "export":
        from .bundles import export_bundle
        with ObservationStore(args.store, read_only=True) as store:
            export_bundle(store, args.bundle)


if __name__ == "__main__":
    main()
