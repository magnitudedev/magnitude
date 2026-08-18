from __future__ import annotations

import argparse

import uvicorn

from .app import create_app


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Instrumented MLX-LM benchmark server")
    result.add_argument(
        "--model", required=True, help="Prepared local MLX model snapshot"
    )
    result.add_argument("--served-model", required=True)
    result.add_argument("--model-revision", required=True)
    result.add_argument("--artifact-manifest", required=True)
    result.add_argument("--host", default="127.0.0.1")
    result.add_argument("--port", type=int, default=8091)
    result.add_argument("--prefill-step-size", type=int, default=2048)
    result.add_argument("--prompt-cache-entries", type=int, default=32)
    return result


def main() -> None:
    args = parser().parse_args()
    app = create_app(
        model_path=args.model,
        served_model=args.served_model,
        model_revision=args.model_revision,
        artifact_manifest=args.artifact_manifest,
        prefill_step_size=args.prefill_step_size,
        prompt_cache_entries=args.prompt_cache_entries,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info", access_log=False)


if __name__ == "__main__":
    main()
