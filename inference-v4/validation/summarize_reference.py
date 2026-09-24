#!/usr/bin/env python3
"""Summarize a reference-matrix directory (llama.cpp, V3 R3, V4 R4, probes) as
Markdown rows: engine, host, model, cell, value, unit."""
import json, pathlib, sys


KINDS = ('decode', 'prefill', 'concurrent', 'long')


def after(history):
    return f' @{history}' if history else ''


def rows_llama(path, host, model):
    kind = next(part for part in reversed(path.stem.split('-')) if not part.isdigit())
    if kind in ('decode', 'prefill'):
        for r in json.loads(path.read_text()):
            if r['n_gen']:
                yield ('llama.cpp', host, model, f"decode @{r['n_depth']}", 1000 / r['avg_ts'], 'ms/token')
            else:
                yield ('llama.cpp', host, model, f"prefill-{r['n_prompt']}{after(r['n_depth'])}",
                       r['avg_ns'] / 1e6, 'ms/chunk')
    elif kind == 'concurrent':
        for line in filter(str.strip, path.read_text().splitlines()):
            r = json.loads(line)
            yield ('llama.cpp', host, model, f"concurrent {r['pl']} @{r['pp']}", r['speed_tg'], 'tok/s aggregate')


def rows_v4(path, host, model):
    d = json.loads(path.read_text())
    for c in d.get('decode', []):
        yield ('V4', host, model, f"decode @{c['context']}", c['median_ms'], 'ms/token')
    for c in d.get('prefill', []):
        yield ('V4', host, model, f"prefill-{c['rows']}{after(c.get('history', 0))}", c['median_ms'], 'ms/chunk')
    for c in d.get('concurrent', []):
        yield ('V4', host, model, f"concurrent {c['sequences']} @{c['context']}",
               c['aggregate_tokens_per_second'], 'tok/s aggregate')


def rows_v3(path, host, model):
    # schema v3-qwen-forward-bench/2; records without a median (failed cells) are skipped.
    d = json.loads(path.read_text())
    for c in d['records']:
        if c['median_seconds'] is None:
            continue
        family = c['family']
        if family == 'decode':
            yield ('V3', host, model, f"decode @{c['context']}", c['median_seconds'] * 1e3, 'ms/token')
        elif family == 'prefill':
            yield ('V3', host, model, f"prefill-{c['tokens']}{after(c.get('history', 0))}",
                   c['median_seconds'] * 1e3, 'ms/chunk')
        elif family == 'concurrent':
            yield ('V3', host, model, f"concurrent {c['sequences']} @{c['context']}",
                   c['tokens_per_second'], 'tok/s aggregate')


def rows_bandwidth(path, host):
    d = json.loads(path.read_text())
    best = d['best']
    best = best.get('configuration', best)
    yield ('probe', host, '-', 'stream read (best)', best['gbPerSecond'], 'GB/s')


def main(directory):
    out = []
    for path in sorted(pathlib.Path(directory).iterdir()):
        name = path.name
        stem = path.stem
        # <engine>-<host>-<model>[-<kind>][-<n>]; host names contain dashes, model tags do not.
        tokens = stem.split('-')
        while len(tokens) > 3 and (tokens[-1].isdigit() or tokens[-1] in KINDS):
            tokens.pop()
        parts = [tokens[0], '-'.join(tokens[1:-1]), tokens[-1]]
        try:
            if name.startswith('llama-') and path.suffix in ('.json', '.jsonl'):
                out += rows_llama(path, parts[1], parts[2])
            elif name.startswith('v4-') and path.suffix == '.json':
                out += rows_v4(path, parts[1], parts[2])
            elif name.startswith('v3-') and path.suffix == '.json':
                out += rows_v3(path, parts[1], parts[2])
            elif name.startswith('bandwidth-'):
                out += rows_bandwidth(path, stem.split('-', 1)[1])
        except (KeyError, json.JSONDecodeError, IndexError) as error:
            print(f'skipping {name}: {error!r}', file=sys.stderr)
    print('| Engine | Host | Model | Cell | Value | Unit |')
    print('|---|---|---|---|---:|---|')
    for engine, host, model, cell, value, unit in out:
        print(f'| {engine} | {host} | {model} | {cell} | {value:.2f} | {unit} |')


if __name__ == '__main__':
    main(sys.argv[1])
