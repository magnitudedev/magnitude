"""Tooling consumes recorded data; Python cases remain ordinary callable functions."""

import argparse
import json
from pathlib import Path

from performance.presentation import export_document, render_tree
from performance.store import DEFAULT_STORE, Store


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--store", type=Path, default=DEFAULT_STORE)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("tui")
    sub.add_parser("rebuild")
    sub.add_parser("list")
    sub.add_parser("check")
    sub.add_parser("incomplete")
    recover = sub.add_parser("recover")
    recover.add_argument("run_id")
    ingest = sub.add_parser("import")
    ingest.add_argument("source", type=Path)
    pull = sub.add_parser("pull")
    pull.add_argument("host")
    pull.add_argument("directory")
    render = sub.add_parser("render")
    render.add_argument("view")
    render.add_argument("--root")
    render.add_argument("--depth", type=int)
    render.add_argument("--document", type=Path)
    args = parser.parse_args()
    store = Store(args.store)
    if args.command == "tui":
        from performance.tui.app import PerformanceApp

        store.refresh()
        PerformanceApp(store).run()
    elif args.command == "import":
        if (
            args.source.is_dir()
            and (args.source / "summary.json").exists()
            and (args.source / "requests.jsonl").exists()
        ):
            from performance.session import ingest as ingest_session

            print(json.dumps(ingest_session(args.source, store)))
        else:
            print(json.dumps(store.import_directory(args.source)))
    elif args.command == "pull":
        print(json.dumps(store.pull(args.host, args.directory)))
    elif args.command == "incomplete":
        print(json.dumps(store.incomplete(), indent=2))
    elif args.command == "recover":
        print(json.dumps(store.recover(args.run_id)))
    elif args.command == "rebuild":
        print(store.refresh()["generation"])
    elif args.command == "render":
        if args.document:
            print(
                export_document(
                    store.state(),
                    args.view,
                    args.document,
                    root=args.root,
                    depth=args.depth,
                    store=store,
                )
            )
        else:
            print(render_tree(store.state(), args.view, root=args.root, depth=args.depth), end="")
    elif args.command == "list":
        print(json.dumps(store.state()["compositions"], indent=2))
    elif args.command == "check":
        state = store.refresh()
        issues = {
            k: {d: a for d, a in v["dimensions"].items() if a["issue"] or a["bound"]["missing"]}
            for k, v in state["components"].items()
        }
        print(json.dumps({k: v for k, v in issues.items() if v}, indent=2))
        return int(any(issues.values()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
