#!/usr/bin/env python3
"""V3 Qwen3.5 forward bench on a GGUF artifact: decode, prefill and concurrent-decode cells.

Forced-token model forwards below the service: no tokenizer, no sampling. Each step is timed
from `prepare` through completion, logit readback and commit. Cells:

- decode: one sequence with `--context` tokens of history (built in `--history-chunk` prefill
  chunks), then `--decode-warmup` warm steps and `--decode-steps` measured steps.
- prefill: for each `--prefill` length, a fresh sequence per forward: one cold, then
  `--prefill-warm` warm forwards.
- concurrent: for each `--sequences` count S, S sequences with `--context` history each, then
  batched decode steps (S requests in one `prepare`). A throwaway pass over forks of the same
  histories first compiles every specialization the measured pass meets, because interleaved
  decode rows fragment each sequence's history and the segment count grows with each step.

Every cell also records one `inspect_forwards` per-kernel profile of the same shape.
"""
import argparse
import dataclasses
import datetime
import hashlib
import json
import pathlib
import platform
import statistics
import sys
import time
import traceback

CELLS = ('decode', 'prefill', 'concurrent')
TOKEN_LIMIT = 100_000  # forced tokens are drawn uniformly from [1, TOKEN_LIMIT)
CAPACITY_QUANTUM = 256  # context capacity rounding; keeps compiled arena shapes stable across runs


def integers(text):
    return tuple(int(value) for value in text.split(',') if value)


def cells(text):
    selected = tuple(value for value in text.split(',') if value)
    unknown = set(selected) - set(CELLS)
    if unknown or not selected:
        raise argparse.ArgumentTypeError(f'cells must be a nonempty subset of {",".join(CELLS)}')
    return selected


parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('--source', type=pathlib.Path, required=True, help='V3 checkout (contains src/)')
parser.add_argument('--artifact', type=pathlib.Path, required=True, help='Qwen3.5 GGUF file')
parser.add_argument('--output', type=pathlib.Path, required=True, help='result file (*.json) or directory (writes result.json)')
parser.add_argument('--backend', choices=('metal', 'cuda'), default='metal' if sys.platform == 'darwin' else 'cuda')
parser.add_argument('--memory-gib', type=int, default=24)
parser.add_argument('--kv', choices=('dense-bf16', 'v3-default'), default='dense-bf16')
parser.add_argument('--cells', type=cells, default=('decode', 'prefill'))
parser.add_argument('--context', type=int, default=256, help='history tokens before decode (decode and concurrent cells)')
parser.add_argument('--prefill', type=integers, default=(32, 128, 512), help='prefill chunk lengths')
parser.add_argument('--prefill-history', type=int, default=0,
                    help='history tokens before the prefill chunks (0 = a fresh sequence per chunk)')
parser.add_argument('--sequences', type=integers, default=(1, 2, 4, 8), help='concurrent sequence counts')
parser.add_argument('--history-chunk', type=int, default=512, help='prefill width used to build history')
parser.add_argument('--decode-warmup', type=int, default=16)
parser.add_argument('--decode-steps', type=int, default=32)
parser.add_argument('--prefill-warm', type=int, default=3)
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()
if args.context < 1 or args.history_chunk < 2 or args.memory_gib < 1 or args.prefill_history < 0:
    parser.error('context, history chunk and memory must be positive (history chunk > 1)')
if min(args.prefill, default=2) < 2 or min(args.sequences, default=1) < 1:
    parser.error('prefill lengths must exceed one and concurrent sequence counts must be positive')
if args.decode_warmup < 0 or args.decode_steps < 0 or args.prefill_warm < 0:
    parser.error('step counts must be nonnegative')
if not args.artifact.is_file():
    parser.error('--artifact must be a GGUF file')

source = args.source.resolve(strict=True)
sys.path.insert(0, str(source / 'src'))
sys.path.insert(0, str(source))
import numpy as np
import ops
from engine import DevicePlan
from engine.data import TokenId
from engine.models.qwen35.formats.gguf import describe
from engine.models.qwen35.inputs import InputPlan
from engine.models.qwen35.inspection import inspect_forwards
from engine.models.qwen35.runtime import DenseRuntime
from engine.models.sequence import LogitsSelection, ModelRequest
from engine.weights.formats.gguf import GGUFFormat
from engine.weights.tensor_residency import TensorWeights

if args.kv == 'dense-bf16':
    ops.default_kv_representation = lambda key, value: ops.dense_kv(key, value, ops.DType.BF16)

output = args.output if args.output.suffix == '.json' else args.output / 'result.json'
output.parent.mkdir(parents=True, exist_ok=True)
decode_tokens = args.decode_warmup + args.decode_steps + 1  # the final token is the profiled step
decoding = {'decode', 'concurrent'} & set(args.cells)
prefilling = 'prefill' in args.cells
context_capacity = max(
    -(-(args.context + decode_tokens) // CAPACITY_QUANTUM) * CAPACITY_QUANTUM if decoding else 0,
    # after history, the cold, warm and profiled chunks run back to back on one sequence
    (args.prefill_history + (args.prefill_warm + 2) * max(args.prefill) if args.prefill_history
     else max(args.prefill)) if prefilling else 0,
)
max_sequences = max(args.sequences) if 'concurrent' in args.cells else 1

report = {
    'schema': 'v3-qwen-forward-bench/2',
    'protocol': (
        'forced-token model forward; no tokenizer or sampling; step time = prepare through '
        'completion, packed logit readback and commit; decode = median of measured steps after '
        'warm steps; prefill = fresh sequence at position 0, median of warm forwards after one '
        'cold; concurrent = S requests per prepare, aggregate tok/s = S / median step; '
        'profiles are separate forwards'
    ),
    'meta': {
        'created': datetime.datetime.now(datetime.UTC).isoformat(),
        'host': platform.node(),
        'argv': sys.argv,
        'script_sha256': hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
        'source': str(source),
        'artifact': {'path': str(args.artifact.resolve()), 'bytes': args.artifact.stat().st_size},
        'backend': args.backend,
        'memory_gib': args.memory_gib,
        'kv': args.kv,
        'cells': list(args.cells),
        'context': args.context,
        'context_capacity': context_capacity,
        'max_sequences': max_sequences,
        'history_chunk': args.history_chunk,
        'prefill_history': args.prefill_history,
        'decode_warmup': args.decode_warmup,
        'decode_steps': args.decode_steps,
        'prefill_warm': args.prefill_warm,
        'tokens': {'seed': args.seed, 'low': 1, 'high': TOKEN_LIMIT},
    },
    'records': [],
    'profiles': [],
    'errors': [],
}


def save():
    output.write_text(json.dumps(report, indent=2, default=str))


def median(values):
    return statistics.median(values) if values else None


def stream(key, count):
    """Deterministic forced tokens; `key` separates sequences so routed experts vary."""
    values = np.random.default_rng((args.seed, key)).integers(1, TOKEN_LIMIT, count)
    return tuple(TokenId(int(value)) for value in values)


def profiler(cell):
    def observe(invocation, observation):
        graph = invocation.compiled.graph
        report['profiles'].append({
            'cell': cell,
            'graph_fingerprint': graph.fingerprint,
            'mode': invocation.mode,
            'positions': invocation.positions,
            'lengths': invocation.lengths,
            'physical_rows': invocation.physical_rows,
            'observation': dataclasses.asdict(observation),
            'formulas': {
                str(call.occurrence): {'id': call.formula.id, 'parent': call.parent, 'nodes': call.nodes}
                for call in graph.formulas
            },
            'nodes': {
                str(node.id): str(getattr(node, 'op', getattr(node, 'operation', type(node).__name__)))
                for node in graph.nodes
            },
        })
        save()
    return observe


artifact = GGUFFormat(str(args.artifact))
description = describe(artifact)
report['meta']['artifact']['identity'] = str(artifact.identity)
report['meta']['geometry'] = str(description.geometry)
save()
plan = DevicePlan.discover(backend=args.backend, maximum_bytes=args.memory_gib << 30)
report['meta']['endpoints'] = [endpoint.model_dump(mode='json') for endpoint in plan.selected_endpoints]
save()

with ops.DeviceRuntime.open(plan) as device:
    weights = TensorWeights(artifact, device)
    model = DenseRuntime(description, device, weights, max_sequences=max_sequences, context_capacity=context_capacity)

    def step(sequences, tokens):
        """One forward over len(sequences) requests; returns (seconds, logits, specialized)."""
        compiled = len(model.program._compiled)
        requests = tuple(
            ModelRequest(sequence, chunk, LogitsSelection.LAST) for sequence, chunk in zip(sequences, tokens, strict=True)
        )
        start = time.perf_counter()
        batch = model.prepare(requests)
        try:
            batch.completion.wait()
            data = device.read(batch.logits, after=batch.completion)
            for advance in batch.advances:
                advance.commit()
            elapsed = time.perf_counter() - start
        finally:
            batch.close()
        logits = np.frombuffer(data, np.float32).reshape(len(requests), -1)
        return elapsed, logits, len(model.program._compiled) != compiled

    def history(sequence, tokens):
        seconds = []
        for offset in range(0, len(tokens), args.history_chunk):
            elapsed, _, _ = step((sequence,), (tokens[offset:offset + args.history_chunk],))
            seconds.append(elapsed)
        return seconds

    def decode_steps(sequences, streams, first, count, record):
        for index in range(first, first + count):
            record['segments'].append(max(len(sequence.state.history_ranges) for sequence in sequences))
            elapsed, logits, specialized = step(sequences, tuple((tokens[index],) for tokens in streams))
            record['finite'] = record['finite'] and bool(np.isfinite(logits).all())
            record['specialized'].append(specialized)
            record['top1'].append([int(value) for value in np.argmax(logits, axis=1)])
            yield elapsed

    def measure_decode(sequences, streams, record):
        base = args.context
        record['warmup_seconds'] = list(decode_steps(sequences, streams, base, args.decode_warmup, record))
        save()
        record['seconds'] = list(decode_steps(sequences, streams, base + args.decode_warmup, args.decode_steps, record))
        record['median_seconds'] = median(record['seconds'])
        if record['median_seconds'] is not None:
            record['tokens_per_second'] = len(sequences) / record['median_seconds']
        save()
        with inspect_forwards(model, observed=profiler(record['label']), kernel_limit=2048):
            list(decode_steps(sequences, streams, base + decode_tokens - 1, 1, {**record, 'segments': [], 'specialized': [], 'top1': []}))

    def decode_record(family, label, count):
        return {
            'family': family, 'label': label, 'context': args.context, 'sequences': count,
            'tokens_per_step': count, 'history_seconds': [], 'warmup_seconds': [], 'seconds': [],
            'median_seconds': None, 'tokens_per_second': None, 'segments': [], 'specialized': [],
            'finite': True, 'top1': [],
        }

    def decode_cell():
        record = decode_record('decode', f'decode-ctx{args.context}', 1)
        report['records'].append(record)
        tokens = stream(0, args.context + decode_tokens)
        sequence = model.create(InputPlan.text(tokens))
        try:
            record['history_seconds'] = history(sequence, tokens[:args.context])
            save()
            measure_decode((sequence,), (tokens,), record)
        finally:
            sequence.close()

    def prefill_cell(length):
        """Without history, each chunk is a fresh sequence at position 0; with --prefill-history H,
        one sequence accepts H tokens of history and the chunks follow each other on it."""
        history_tokens = args.prefill_history
        suffix = f'-after-{history_tokens}' if history_tokens else ''
        record = {
            'family': 'prefill', 'label': f'prefill-{length}{suffix}', 'tokens': length,
            'history': history_tokens, 'history_seconds': [], 'cold_seconds': None,
            'seconds': [], 'median_seconds': None, 'tokens_per_second': None, 'finite': True, 'top1': [],
        }
        report['records'].append(record)
        chunks = args.prefill_warm + 2  # cold, warm, profiled
        tokens = stream(1_000_000 + length, history_tokens + chunks * length)
        shared = None
        if history_tokens:
            shared = model.create(InputPlan.text(tokens))
            record['history_seconds'] = history(shared, tokens[:history_tokens])
            save()
        offsets = iter(range(history_tokens, len(tokens), length))

        def forward():
            if shared is None:
                sequence = model.create(InputPlan.text(tokens[:length]))
                try:
                    elapsed, logits, _ = step((sequence,), (tokens[:length],))
                finally:
                    sequence.close()
            else:
                offset = next(offsets)
                elapsed, logits, _ = step((shared,), (tokens[offset:offset + length],))
            record['finite'] = record['finite'] and bool(np.isfinite(logits).all())
            record['top1'].append(int(np.argmax(logits[0])))
            return elapsed

        try:
            record['cold_seconds'] = forward()
            save()
            for _ in range(args.prefill_warm):
                record['seconds'].append(forward())
                save()
            record['median_seconds'] = median(record['seconds'])
            if record['median_seconds'] is not None:
                record['tokens_per_second'] = length / record['median_seconds']
            with inspect_forwards(model, observed=profiler(record['label']), kernel_limit=2048):
                forward()
            save()
        finally:
            if shared is not None:
                shared.close()

    def concurrent_cell(count):
        record = decode_record('concurrent', f'concurrent-{count}x-ctx{args.context}', count)
        report['records'].append(record)
        streams = tuple(stream(1 + index, args.context + decode_tokens) for index in range(count))
        checkpoints = []
        try:
            for tokens in streams:
                sequence = model.create(InputPlan.text(tokens))
                try:
                    record['history_seconds'].append(history(sequence, tokens[:args.context]))
                    checkpoints.append(sequence.checkpoint())
                finally:
                    sequence.close()
                save()
            for measured in (False, True):
                forks = []
                try:
                    for checkpoint in checkpoints:
                        forks.append(checkpoint.fork())
                    if measured:
                        measure_decode(tuple(forks), streams, record)
                    else:
                        scratch = {**record, 'segments': [], 'specialized': [], 'top1': []}
                        record['compile_pass_seconds'] = list(
                            decode_steps(tuple(forks), streams, args.context, decode_tokens, scratch)
                        )
                        save()
                finally:
                    for fork in forks:
                        fork.close()
        finally:
            for checkpoint in checkpoints:
                checkpoint.close()

    work = {
        'decode': [('decode', decode_cell)],
        'prefill': [(f'prefill-{length}', lambda length=length: prefill_cell(length)) for length in args.prefill],
        'concurrent': [(f'concurrent-{count}', lambda count=count: concurrent_cell(count)) for count in args.sequences],
    }
    try:
        for family in args.cells:
            for label, run in work[family]:
                try:
                    run()
                    print(label, 'done', flush=True)
                except Exception:
                    report['errors'].append({'cell': label, 'traceback': traceback.format_exc()})
                    save()
                    print(traceback.format_exc(), flush=True)
    finally:
        model.close()
        weights.close()
        artifact.close()
        save()

for record in report['records']:
    print(json.dumps({key: record.get(key) for key in ('label', 'median_seconds', 'tokens_per_second')}), flush=True)
sys.exit(1 if report['errors'] else 0)
