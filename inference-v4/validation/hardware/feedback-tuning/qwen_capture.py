"""Capture actual Qwen activations and weights; build independent NumPy references.

Run with an existing MLX/MLX-LM environment. Model acquisition and reference
generation are offline experiment setup, separate from the tuning clock.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm import load


def bf16(x):
    x = np.asarray(x, dtype=np.float32)
    bits = x.view(np.uint32)
    return ((bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)) & np.uint32(0xFFFF0000)).view(np.float32)


def decode(weight, scales, biases):
    codes = np.asarray(weight)
    shifts = np.arange(8, dtype=np.uint32) * 4
    values = ((codes[..., None] >> shifts) & 15).reshape(codes.shape[0], -1).astype(np.float32)
    return (values.reshape(values.shape[0], -1, 64)
            * np.asarray(scales.astype(mx.float32))[..., None]
            + np.asarray(biases.astype(mx.float32))[..., None]).reshape(values.shape)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    model, tokenizer = load(str(args.model))
    layers = model.layers
    captures = {}
    tracked = {id(layer.post_attention_layernorm): i for i, layer in enumerate(layers)}
    original = nn.RMSNorm.__call__

    def capture(self, x):
        if id(self) in tracked:
            captures[tracked[id(self)]] = x
        return original(self, x)

    prompt = 'Explain why a compiler might fuse adjacent operations, and when keeping them separate is faster. Give a concrete example.'
    tokens = tokenizer.encode(prompt)
    tokens = (tokens * 4)[:32]
    cache = model.make_cache()
    regimes = {}
    nn.RMSNorm.__call__ = capture
    try:
        logits = model(mx.array([tokens]), cache=cache)
        mx.eval(logits, list(captures.values()))
        regimes['prefill32'] = dict(captures)
        next_token = int(mx.argmax(logits[0, -1]).item())
        captures.clear()
        logits = model(mx.array([[next_token]]), cache=cache)
        mx.eval(logits, list(captures.values()))
        regimes['decode'] = dict(captures)
    finally:
        nn.RMSNorm.__call__ = original
    for regime, inputs in regimes.items():
        if len(inputs) != len(layers):
            raise RuntimeError(f'{regime}: incomplete activation capture')
    print(json.dumps({'captured_layers': len(layers), 'seconds': time.monotonic()-started}), flush=True)
    tensors = {}
    references = {regime: {} for regime in regimes}
    for i, layer in enumerate(layers):
        prefix = f'l{i}'
        norm = layer.post_attention_layernorm.weight
        tensors[f'{prefix}.norm'] = norm
        states = {}
        for regime, inputs in regimes.items():
            residual = np.asarray(inputs[i].astype(mx.float32)).reshape(-1, norm.size)
            tensors[f'{prefix}.{regime}.residual'] = mx.array(residual)
            normalized = bf16(residual / np.sqrt(np.mean(residual * residual, axis=-1, keepdims=True) + 1e-6)
                              * np.asarray(norm.astype(mx.float32)))
            states[regime] = dict(residual=residual, normalized=normalized)
        for role in ['gate', 'up', 'down']:
            projection = getattr(layer.mlp, role + '_proj')
            matrix = decode(projection.weight, projection.scales, projection.biases)
            for regime, state in states.items():
                source = state['normalized'] if role != 'down' else state['product']
                state[role] = bf16(source @ matrix.T)
                if role == 'up':
                    gate = state['gate']
                    # Stable sigmoid, with the activation and product each rounded.
                    sigmoid = np.exp(-np.logaddexp(np.float32(0), -gate))
                    state['product'] = bf16(bf16(gate * sigmoid) * state['up'])
            del matrix
        for regime, state in states.items():
            state['result'] = state['residual'] + state['down']
            for name, value in state.items():
                if name != 'residual':
                    references[regime][f'{prefix}.{name}'] = value
        print(json.dumps({'reference_layer': i, 'seconds': time.monotonic()-started}), flush=True)
    mx.save_safetensors(str(args.output/'inputs.safetensors'), tensors)
    for regime, ref in references.items():
        np.savez(args.output/f'{regime}-reference.npz', **ref)
    manifest = dict(model=str(args.model.resolve()), revision=args.model.name, layers=len(layers),
                    prompt=prompt, prompt_ids=tokens, decode_token=next_token, hidden=2560,
                    intermediate=9216, quantization=dict(bits=4, group_size=64, mode='affine'),
                    activation='BF16', residual='F32 captured from actual BF16 model residual',
                    reference='NumPy F32 decoded-weight BLAS; BF16 rounding at Seismic dense_suffix boundaries',
                    seconds=time.monotonic()-started,
                    config_sha256=hashlib.sha256((args.model/'config.json').read_bytes()).hexdigest())
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(manifest), flush=True)


if __name__ == '__main__':
    main()
