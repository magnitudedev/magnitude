"""Independent dense Qwen3.5 GGUF mixed-arithmetic oracle.

Uses gguf's decoder, NumPy contractions and explicit storage rounding. No engine
artifact parser, model equations, numerical helpers, state, or operation plans.
Full-model logits are FP32; this is reference generation, never production compute.
"""

import argparse
import hashlib
import json
from pathlib import Path

import gguf
import numpy as np


def bf16(x):
    a = np.asarray(x, np.float32)
    bits = a.view(np.uint32)
    return (((bits + 0x7FFF + ((bits >> 16) & 1)) & np.uint32(0xFFFF0000)).view(np.float32)).astype(
        np.float64
    )


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


class Oracle:
    def __init__(self, path):
        reader = gguf.GGUFReader(path)
        self.tensors = {tensor.name: tensor for tensor in reader.tensors}

        def field(name):
            return reader.fields["qwen35." + name].contents()

        self.layers = int(field("block_count"))
        self.hidden = int(field("embedding_length"))
        self.heads = int(field("attention.head_count"))
        self.kv_heads = int(field("attention.head_count_kv"))
        self.width = int(field("attention.key_length"))
        self.rotary_width = int(field("rope.dimension_count"))
        self.base = float(field("rope.freq_base"))
        self.epsilon = float(field("attention.layer_norm_rms_epsilon"))
        self.key_heads = int(field("ssm.group_count"))
        self.value_heads = int(field("ssm.time_step_rank"))
        self.state_width = int(field("ssm.state_size"))
        self.conv_width = int(field("ssm.conv_kernel"))
        self.channels = (2 * self.key_heads + self.value_heads) * self.state_width
        self.position = 0
        self.kv, self.recurrent = {}, {}
        self.vocabulary = int(self.tensors["token_embd.weight"].shape[1])

    def weight(self, name, rows=None):
        tensor = self.tensors[name]
        data = tensor.data if rows is None else tensor.data[rows]
        return gguf.dequantize(data, tensor.tensor_type).astype(np.float64)

    def linear(self, x, name, *, output=True):
        matrix = len(x) >= 8
        tensor = self.tensors[name]
        result = []
        # Bound oracle memory even for the large tied embedding/readout table.
        for start in range(0, int(tensor.shape[1]), 8192):
            w = self.weight(name, slice(start, start + 8192))
            if matrix:
                w = bf16(w)
            part = x @ w.T
            result.append(bf16(part) if output else part)
        return np.concatenate(result, axis=-1)

    def norm(self, x, name):
        return (
            x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + self.epsilon) * self.weight(name)
        )

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def rotary(self, x):
        half = self.rotary_width // 2
        frequency = (
            (1 / self.base ** (np.arange(half) / half)).astype(np.float32).astype(np.float64)
        )
        angle = np.arange(self.position, self.position + len(x))[:, None, None] * frequency
        result = x.copy()
        result[..., :half] = x[..., :half] * np.cos(angle) - x[..., half : 2 * half] * np.sin(angle)
        result[..., half : 2 * half] = x[..., half : 2 * half] * np.cos(angle) + x[
            ..., :half
        ] * np.sin(angle)
        return bf16(result)

    def attention(self, a, prefix, layer):
        n, h, kh, d = len(a), self.heads, self.kv_heads, self.width
        qg = self.linear(a, prefix + "attn_q.weight").reshape(n, h, 2, d)
        q = self.rotary(self.norm(qg[:, :, 0], prefix + "attn_q_norm.weight"))
        key = self.rotary(
            self.norm(
                self.linear(a, prefix + "attn_k.weight").reshape(n, kh, d),
                prefix + "attn_k_norm.weight",
            )
        )
        value = self.linear(a, prefix + "attn_v.weight").reshape(n, kh, d)
        if layer in self.kv:
            old_key, old_value = self.kv[layer]
            key, value = np.concatenate((old_key, key)), np.concatenate((old_value, value))
        self.kv[layer] = key, value
        result = np.empty((n, h, d), np.float64)
        for head in range(h):
            kv = head // (h // kh)
            scores = q[:, head] @ key[:, kv].T / np.sqrt(d)
            visible = (
                np.arange(len(key))[None, :] <= np.arange(self.position, self.position + n)[:, None]
            )
            scores = np.where(visible, scores, -np.inf)
            exponent = np.exp(scores - scores.max(axis=-1, keepdims=True))
            probabilities = exponent / exponent.sum(axis=-1, keepdims=True)
            # Prefill's matrix value contraction permits BF16 probability tiles;
            # decode retains FP32 probability/value accumulation across KV runs.
            if n >= 8:
                probabilities = bf16(probabilities)
            result[:, head] = probabilities @ value[:, kv]
        gated = bf16(bf16(result) * self.sigmoid(qg[:, :, 1]))
        return self.linear(gated.reshape(n, -1), prefix + "attn_output.weight")

    def recurrence(self, a, prefix, layer):
        n, hk, hv, d = len(a), self.key_heads, self.value_heads, self.state_width
        qkv = self.linear(a, prefix + "attn_qkv.weight")
        z = self.linear(a, prefix + "attn_gate.weight")
        alpha = self.linear(a, prefix + "ssm_alpha.weight")
        beta = (
            self.sigmoid(self.linear(a, prefix + "ssm_beta.weight"))
            .astype(np.float32)
            .astype(np.float64)
        )
        decay = (
            np.exp(
                self.weight(prefix + "ssm_a")
                * np.logaddexp(0, alpha + self.weight(prefix + "ssm_dt.bias"))
            )
            .astype(np.float32)
            .astype(np.float64)
        )
        history, state = self.recurrent.get(
            layer, (np.zeros((self.conv_width - 1, self.channels)), np.zeros((hv, d, d)))
        )
        convolution = np.concatenate((history, qkv))
        conv_weights = self.weight(prefix + "ssm_conv1d.weight")
        activated = sum(convolution[t : t + n] * conv_weights[:, t] for t in range(self.conv_width))
        activated *= self.sigmoid(activated)
        q, k, v = np.split(activated.reshape(n, 2 * hk + hv, d), (hk, 2 * hk), axis=1)
        q = (
            (q / np.sqrt(np.sum(q * q, axis=-1, keepdims=True) + self.epsilon) / np.sqrt(d))
            .astype(np.float32)
            .astype(np.float64)
        )
        k = (
            (k / np.sqrt(np.sum(k * k, axis=-1, keepdims=True) + self.epsilon))
            .astype(np.float32)
            .astype(np.float64)
        )
        v = v.astype(np.float32).astype(np.float64)
        mapping = np.arange(hv) % hk
        mixed = []
        for row in range(n):
            state = state * decay[row, :, None, None]
            correction = (v[row] - np.einsum("hvk,hk->hv", state, k[row, mapping])) * beta[
                row, :, None
            ]
            state = (
                (state + correction[:, :, None] * k[row, mapping, None, :])
                .astype(np.float32)
                .astype(np.float64)
            )
            mixed.append(
                np.einsum("hvk,hk->hv", state, q[row, mapping])
                .astype(np.float32)
                .astype(np.float64)
            )
        self.recurrent[layer] = convolution[-(self.conv_width - 1) :], state
        mixed = bf16(self.norm(np.stack(mixed), prefix + "ssm_norm.weight")).reshape(n, -1)
        return self.linear(bf16(mixed * z * self.sigmoid(z)), prefix + "ssm_out.weight")

    def advance(self, tokens, *, logits):
        x = bf16(self.weight("token_embd.weight", np.asarray(tokens)))
        for layer in range(self.layers):
            prefix = f"blk.{layer}."
            a = bf16(self.norm(x, prefix + "attn_norm.weight"))
            mixed = (
                self.attention(a, prefix, layer)
                if prefix + "attn_q.weight" in self.tensors
                else self.recurrence(a, prefix, layer)
            )
            x = bf16(x + mixed)
            a = bf16(self.norm(x, prefix + "post_attention_norm.weight"))
            gate = self.linear(a, prefix + "ffn_gate.weight")
            up = self.linear(a, prefix + "ffn_up.weight")
            x = bf16(
                x + self.linear(bf16(gate * self.sigmoid(gate) * up), prefix + "ffn_down.weight")
            )
        self.position += len(tokens)
        if not logits:
            return None
        normalized = bf16(self.norm(x[-1:], "output_norm.weight"))
        output = "output.weight" if "output.weight" in self.tensors else "token_embd.weight"
        return self.linear(normalized, output, output=False).astype(np.float32).ravel()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("tokens", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--history", type=int, default=0)
    parser.add_argument("--chunk", type=int, default=512)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    tokens = np.fromfile(args.tokens, "<i4")
    if not 0 <= args.history < len(tokens) or args.chunk <= 0:
        raise ValueError("invalid reference workload")
    oracle = Oracle(args.model)
    for start in range(0, args.history, args.chunk):
        end = min(start + args.chunk, args.history)
        oracle.advance(tokens[start:end], logits=False)
        print(f"oracle history {end}/{args.history}", flush=True)
    logits = oracle.advance(tokens[args.history :], logits=True)
    if not np.isfinite(logits).all():
        raise ValueError("nonfinite oracle output")
    content = (
        np.asarray((len(tokens), oracle.vocabulary), "<i4").tobytes()
        + tokens.tobytes()
        + logits.tobytes()
    )
    args.output.write_bytes(content)
    args.output.with_suffix(".json").write_text(
        json.dumps(
            dict(
                artifact_sha256=sha(args.model),
                tokens_sha256=sha(args.tokens),
                oracle_source_sha256=sha(__file__),
                reference_sha256=sha(args.output),
                precision="mixed_bf16",
                reference=(
                    "Independent NumPy equations; gguf-0.19 decode; BF16 storage, permitted "
                    "BF16 matrix operands, FP32 recurrent state/logits"
                ),
                history_tokens=args.history,
                history_chunk=args.chunk,
                vocabulary=oracle.vocabulary,
                top1=int(logits.argmax()),
                numpy=np.__version__,
            ),
            indent=2,
        )
        + "\n"
    )
    print(f"saved {args.output}; top1={logits.argmax()}", flush=True)


if __name__ == "__main__":
    main()
