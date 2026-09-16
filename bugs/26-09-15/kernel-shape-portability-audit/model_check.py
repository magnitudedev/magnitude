"""Ordinary production forwards, with compilation warmups excluded from samples."""
import argparse
import json
from pathlib import Path
import time
import statistics

import numpy as np
import ops
from engine import DevicePlan
from engine.data import TokenId
from engine.models.qwen35.formats.gguf import describe
from engine.models.qwen35.inputs import InputPlan
from engine.models.qwen35.runtime import DenseRuntime
from engine.models.sequence import LogitsSelection, ModelRequest
from engine.weights.formats.gguf import GGUFFormat
from engine.weights.tensor_residency import TensorWeights
from ops.lab.ownership import exclusive_measurement

parser = argparse.ArgumentParser()
parser.add_argument("artifact", type=Path)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--samples", type=int, default=7)
parser.add_argument("--context", type=int, default=32)
args = parser.parse_args()
args.output.parent.mkdir(parents=True, exist_ok=True)
records = []
with exclusive_measurement():
    source = GGUFFormat(str(args.artifact.expanduser()))
    device = ops.DeviceRuntime.open(DevicePlan.discover(backend="metal", maximum_bytes=48 << 30))
    weights = TensorWeights(source, device)
    model = sequence = None
    try:
        model = DenseRuntime(describe(source), device, weights, max_sequences=1,
                             prefill_rows=args.context, context_capacity=args.context + 16)
        prompt = tuple(TokenId(i) for i in range(100, 100 + args.context))
        for iteration in range(args.samples + 2):
            sequence = model.create(InputPlan.text(prompt))
            try:
                for stage, tokens in (("prefill", prompt), ("decode-1", (TokenId(100 + args.context),)),
                                      ("decode-2", (TokenId(101 + args.context),))):
                    start = time.perf_counter_ns()
                    batch = model.prepare((ModelRequest(sequence, tokens, LogitsSelection.LAST, (0, 0, 0, 0, 0, 0)),))
                    try:
                        batch.completion.wait()
                        elapsed = time.perf_counter_ns() - start
                        logits = batch.logits.native.float().cpu().numpy().copy()
                        assert np.isfinite(logits).all(), stage
                        batch.advances[0].commit()
                        record = dict(iteration=iteration, stage=stage, elapsed_ns=elapsed,
                                      warmup=iteration < 2, top_token=int(logits.argmax()))
                        records.append(record)
                        print(json.dumps(record), flush=True)
                        if iteration == args.samples + 1:
                            np.save(args.output.with_name(args.output.stem + '-' + stage + '.npy'), logits)
                        args.output.write_text(json.dumps(records, indent=2))
                    finally:
                        batch.close()
            finally:
                sequence.close()
                sequence = None
        for index, compiled in enumerate(model.program._compiled.values()):
            for key, source_text in compiled.evidence().items():
                if key.endswith("/device-source"):
                    name = f"{args.output.stem}-program-{index}-{key.replace('/', '-')}.txt"
                    args.output.with_name(name).write_text(source_text)
        for stage in ("prefill", "decode-1", "decode-2"):
            samples = [r['elapsed_ns'] for r in records if r['stage'] == stage and not r['warmup']]
            print(stage, statistics.median(samples) / 1e6, "ms median", flush=True)
    finally:
        if sequence is not None:
            sequence.close()
        if model is not None:
            model.close()
        weights.close()
        device.close()
        source.close()
