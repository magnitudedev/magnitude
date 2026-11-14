"""magnitude-references: independent float32 references of the model families.

  run            reference logits + sanity gate for a rendered fixture set (exit 1 when the gate fails)
  kl             a build's logits against the cached reference (and against a base build's, and a
                 yardstick's or the storage-resolution bar's, `--yardstick-perturbed SEED...`); a layer
                 slice's reference (a target with a slice's provenance record) is judged by its admission alone,
                 without the sanity gate, which a slice always fails
  prompts        seeded random token prompts [count, length] over a GGUF's vocabulary (.npy), as `check_hf` draws them
  compare        the reference's logits on prompts against another producer's on the same tokens (upstream llama.cpp's,
                 `magnitude-conversion llama-logits`): KL(reference || producer) per position, summarized and judged
                 by the storage-resolution bar (`--yardstick-perturbed SEED...`, as `kl`'s); with `--same-values`, first
                 that the producer's file holds the values the reference computes with
  features       a projector's image features by its vision reference (`families.PROJECTORS`), in llama.cpp's
                 `MTMD_DEBUG_EMBEDDINGS` dump format

`mrig kl` runs `kl`, which builds a missing reference as `run --device cpu` does and admits it
(`cache`); the references' own checks against released code and checkpoints are
`tests/test_released.py`. A GGUF's header and a truncated real-weight GGUF are `catalog-tool
header` and `catalog-tool slice`. `run` takes `--device cuda|mps|cpu` (default auto). Matmuls
run in full float32 (TF32 disabled).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from magnitude_references import profile
from magnitude_references.checks import device_of
from magnitude_rig.host.identity import sha256
from magnitude_references.performance import BATCH_TOKENS
from magnitude_references.run import Options, build_reference, default_output
from magnitude_references.sanity import PERPLEXITY_MAX, TOP1_MIN, Bounds, Gate


def emit(report: dict, path: Path | None) -> None:
    text = json.dumps(report, indent=1)
    print(text)
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n")


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="magnitude-references", description=__doc__,
                                   formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = root.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="reference logits and sanity gate for rendered fixtures")
    run.add_argument("--rendered", type=Path, required=True, help="harness `render` output (JSON Lines)")
    run.add_argument("--artifacts", type=Path, required=True, help="catalog acquisition object (JSON)")
    run.add_argument("--output", type=Path, help="run directory (default: <stamp>-reference-<id> in the repository's results store, "
                     "the main checkout's inference/evaluation/results)")
    run.add_argument("--perplexity-max", type=float, default=PERPLEXITY_MAX)
    run.add_argument("--top1-min", type=float, default=TOP1_MIN)
    run.add_argument("--batch-tokens", type=int, default=BATCH_TOKENS, help="padded tokens per streamed forward")
    run.add_argument("--cache-weights", action="store_true", help="keep dequantized weights resident")
    run.add_argument("--build-identity", type=Path,
                     help="contract build object {commit, dirty_digest} (JSON), recorded verbatim in result.json")
    run.add_argument("--device", default="auto")
    run.add_argument("--variant", help=VARIANT)
    run.add_argument("--json", type=Path, help="also write the report here")

    kl = commands.add_parser("kl", help="a build's logits against the cached reference")
    kl.add_argument("--logits", type=Path, required=True, help="the build's harness `logits` run directory")
    kl.add_argument("--base", type=Path, help="the base build's `logits` run directory, on the same tokens")
    bar = kl.add_mutually_exclusive_group()
    bar.add_argument("--yardstick", type=Path,
                     help="an independent producer's logits on the same tokens: the build's bar, judged as against a base")
    bar.add_argument("--yardstick-perturbed", type=int, nargs="+", default=(), metavar="SEED",
                     help="the storage-resolution bar: the reference over its stored values moved by one step of "
                     "their formats, one draw per seed (the `perturbed:SEED` variant, built and cached when missing); "
                     "the build is within it when every judgeable row's KL is within 64 times the draws' largest there")
    kl.add_argument("--own-rendering", type=Path, help="the build's own `render` of the fixtures, to compare")
    kl.add_argument("--artifacts", type=Path, required=True, help="catalog acquisition object (JSON)")
    kl.add_argument("--build-identity", type=Path, required=True, help="identity of the reference tools' tree")
    kl.add_argument("--cache", type=Path, required=True, help="the host's reference cache")
    kl.add_argument("--output", type=Path, required=True, help="the summary (JSON)")
    kl.add_argument("--variant", help=VARIANT)

    obtain = commands.add_parser("obtain", help="the cached reference of rendered tokens, and the storage-resolution "
                                 "bar's draws, built when missing; no build is judged")
    obtain.add_argument("--rendered", type=Path, required=True, help="harness `render` output (JSON Lines)")
    obtain.add_argument("--yardstick-perturbed", type=int, nargs="+", default=(), metavar="SEED",
                        help="the draws to obtain beside the reference (the `perturbed:SEED` variants)")
    obtain.add_argument("--artifacts", type=Path, required=True, help="catalog acquisition object (JSON)")
    obtain.add_argument("--build-identity", type=Path, required=True, help="identity of the reference tools' tree")
    obtain.add_argument("--cache", type=Path, required=True, help="the host's reference cache")
    obtain.add_argument("--logs", type=Path, required=True, help="where the references' build logs go")
    obtain.add_argument("--output", type=Path, required=True, help="the references obtained (JSON)")
    obtain.add_argument("--profile", action="store_true",
                        help="time each batch of the builds it starts: wall, read and dequantize seconds (`profile`)")

    prompts = commands.add_parser("prompts", help="seeded random token prompts over a GGUF's vocabulary")
    prompts.add_argument("--model", type=Path, required=True, help="the GGUF whose vocabulary they are drawn from")
    prompts.add_argument("--count", type=int, required=True)
    prompts.add_argument("--length", type=int, required=True)
    prompts.add_argument("--seed", type=int, required=True)
    prompts.add_argument("--output", type=Path, required=True, help="the tokens [count, length] (.npy)")

    compare = commands.add_parser("compare", help="the reference's logits against another producer's")
    compare.add_argument("--model", type=Path, required=True, help="the GGUF the reference computes")
    compare.add_argument("--tokens", type=Path, required=True, help="`prompts` output")
    compare.add_argument("--logits", type=Path, required=True,
                         help="the producer's logits [count, length, vocabulary] (.npy)")
    compare.add_argument("--same-values", type=Path, help="the producer's GGUF, which must hold the reference's values")
    compare.add_argument("--yardstick-perturbed", type=int, nargs="+", required=True, metavar="SEED",
                         help="the storage-resolution bar the producer is judged by, one draw per seed, as `kl`'s")
    compare.add_argument("--json", type=Path, help="also write the report here")

    features = commands.add_parser("features", help="a projector's image features, as llama.cpp's mtmd dumps them")
    features.add_argument("--projector", type=Path, required=True, help="the projector GGUF (`mmproj-*.gguf`)")
    features.add_argument("--image", type=Path, required=True, help="an image whose sides are whole cells")
    features.add_argument("--output", type=Path, required=True)
    return root


VARIANT = ("a measurement variant of the family's reference (`Reference.variants`, e.g. deepseek41's `activations`, or "
           "`perturbed:<seed>`, its stored values moved by one step of their formats), cached under its own identity; "
           "default: the model's numerics")


def main(argv: list[str] | None = None) -> int:
    root = parser()
    options = root.parse_args(argv)
    if options.command == "prompts":
        return write_prompts(options)
    if options.command == "compare":
        return compare(options)
    if options.command == "features":
        return write_features(options)
    if options.command == "obtain":
        from magnitude_references import kl
        if options.profile:
            os.environ[profile.ENV] = "1"  # the builds it starts inherit it
        options.logs.mkdir(parents=True, exist_ok=True)
        reference, draws = kl.obtain_all(options.rendered, options.artifacts, options.build_identity, options.cache,
                                         options.logs, None, tuple(options.yardstick_perturbed))
        options.output.write_text(json.dumps(
            {"reference": reference.to_json(), "draws": [draw.to_json() for draw in draws]}, indent=1))
        return 0
    if options.command == "kl":
        from magnitude_references import kl
        summary = kl.run(options.logits, options.artifacts, options.build_identity, options.cache,
                         options.output, options.base, options.own_rendering, options.yardstick, options.variant,
                         tuple(options.yardstick_perturbed))
        print(json.dumps({key: value for key, value in summary.items() if key != "reference"}, indent=1))
        return 0
    if os.environ.get(profile.ENV):
        profile.install()
    output = options.output or default_output(sha256(options.rendered))
    build = None
    if options.build_identity:
        build = json.loads(options.build_identity.read_text())
        if set(build) != {"commit", "dirty_digest"}:
            root.error(f"{options.build_identity}: a build identity has exactly commit and dirty_digest")
    reference = build_reference(options.rendered, options.artifacts, output,
                                Options(None if options.device == "auto" else device_of(options.device), Bounds(options.perplexity_max, options.top1_min),
                                        options.batch_tokens, options.cache_weights, build, options.variant))
    emit({"output": str(output), **reference}, options.json)
    if not reference["sanity"]["passed"]:
        print(f"sanity gate failed on {Gate.failing(reference['sanity'])}: this reference must not be used "
              "for comparison", file=sys.stderr)
        return 1
    return 0


def write_prompts(options: argparse.Namespace) -> int:
    import numpy as np

    from magnitude_references.checks import prompts
    from magnitude_references.gguf import Package
    vocabulary = Package.local(options.model).info("token_embd.weight").array_shape[0]
    np.save(options.output, prompts(options.count, options.length, vocabulary, options.seed).numpy())
    return 0


def write_features(options: argparse.Namespace) -> int:
    """The image's features by the projector's vision reference (`families.open_projector`), in llama.cpp's
    `MTMD_DEBUG_EMBEDDINGS` dump format (`[i32 rows][i32 width]`, then the F32 rows), which the qualification
    `engine/tests/image_features.rs` reads."""
    import numpy as np
    from PIL import Image

    from magnitude_references.families import open_projector
    features = open_projector(options.projector).features(np.asarray(Image.open(options.image).convert("RGB")))
    with options.output.open("wb") as stream:
        stream.write(np.array(features.shape, dtype="<i4").tobytes())
        stream.write(features.numpy().astype("<f4").tobytes())
    return 0


def compare(options: argparse.Namespace) -> int:
    """The reference on the CPU against the producer's logits, judged by the storage-resolution bar of the seeds'
    draws (`kl.within_resolution`), after `checks.same_values`."""
    import numpy as np
    import torch

    from magnitude_references import quality
    from magnitude_references.checks import compare_logits, logits_of, same_values
    from magnitude_references.families import open_reference
    from magnitude_references.gguf import Package
    from magnitude_references.kl import within_resolution
    from magnitude_references.model import Weights
    cpu = device_of("cpu")
    report: dict = {"model": str(options.model), "logits": str(options.logits)}
    if options.same_values:
        model = Weights(Package.local(options.model), cpu, False)
        different = same_values(model, Weights(Package.local(options.same_values), cpu, False))
        report["same_values"] = {"model": str(options.same_values), "different": different}
        if different:
            emit(report, options.json)
            print(f"{options.same_values} holds other values than {options.model}: {different[:8]}", file=sys.stderr)
            return 1
    tokens = torch.from_numpy(np.load(options.tokens))
    reference = logits_of(open_reference(options.model, cpu, False), tokens)
    build, difference = compare_logits(reference, np.load(options.logits))
    draws = [compare_logits(reference, logits_of(open_reference(options.model, cpu, False, variant=f"perturbed:{seed}"),
                                                 tokens))[0] for seed in options.yardstick_perturbed]
    vocabulary = reference.shape[-1]
    entropy = quality.entropy(reference.reshape(-1, vocabulary))
    report |= {"prompts": tokens.shape[0], "length": tokens.shape[1],
               "build": {**quality.summarize(build.kl, build.agree), "relative_max_difference": difference},
               "resolution": {"draws": [{"seed": seed, **quality.summarize(draw.kl, draw.agree),
                                         "worst": quality.worst(draw)}
                                        for seed, draw in zip(options.yardstick_perturbed, draws)],
                              **within_resolution(build.rows, entropy, vocabulary,
                                                  np.stack([draw.kl for draw in draws]), build.kl, build.agree)}}
    emit(report, options.json)
    return 0


def entry() -> None:
    raise SystemExit(main())
