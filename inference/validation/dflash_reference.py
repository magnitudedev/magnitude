#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy==2.5.3", "gguf==0.19.0", "torch==2.14.0", "transformers==5.17.0", "safetensors", "huggingface-hub"]
# ///
"""Independent float32 reference for DFlash and DSpark drafts (GGUF architecture `dflash`).

uv run inference/validation/dflash_reference.py propose --draft D.gguf --target T.gguf --tokens 1,2,3 [--device cuda]
uv run inference/validation/dflash_reference.py check --draft D.gguf --target T.gguf [--device cuda]

Model definition: transformers 5.17.0 `MuseGlimmerAssistantModel` and `DFlashTokenCandidateGenerator`
(DFlash), and the released DSpark code (`RadixArk/Qwen3.8-27B-DSpark@b9a5dbdf`: `dflash.py`
`DFlashDraftModel`, `dspark.py` Markov and confidence heads):
- context: the target's residual entering each of `target_layers` (the released `target_layer_ids`
  are layer *outputs*; the GGUF stores them + 1), float32, concatenated per token, then
  c = RMSNorm(fc c, enc.output_norm) for the n tokens the target has run;
- block: [anchor, mask x (block_size - 1)] at positions n .. n + block_size - 1, embedded with the raw
  target (or the draft's own) embedding table, no embedding norm or scale;
- each layer: q from RMSNorm(x); k, v from concat(c, RMSNorm(x)) (the context is not layer-normed);
  per-head q/k RMSNorm; rotate-half rope at the context and block positions (YaRN when the GGUF
  declares it); attention over context and block with no causal mask (sliding layers keep keys with
  |q - k| <= W); SwiGLU; final RMSNorm; the target's LM head without logit scale or softcap;
- DFlash: block slot i >= 1 predicts position n + i. DSpark (`markov_w1` present, sample from the
  anchor): slot i >= 0 predicts position n + 1 + i with logits + markov_w2 markov_w1[prev_i], prev_0 =
  anchor, prev_{i+1} = the greedy token of slot i, and confidence
  sigmoid(conf_proj [h_i ; markov_w1[prev_i]] + b).

The released DSpark draft of LFM2 uses interleaved rope (`rope_is_neox_style: false`); the converter
reorders its q/k rows and norms to rotate-half order, so rotate-half on the GGUF rows is the same map.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import importlib.util
import json
from pathlib import Path
import sys

import torch

from model_references import open_reference
from reference_cli import device_of
from reference_gguf import Package
from reference_model import (F32, Reference, Rotary, Weights, attention, default_inv_freq, heads, linear, per_layer,
                             rms_norm, silu, yarn_inv_freq)

DSPARK_CODE = ("RadixArk/Qwen3.8-27B-DSpark", "b9a5dbdf03bc999c6c73c426b19c2d9041cea393")


@dataclass
class Proposal:
    positions: list[int]          # target position each slot predicts
    logits: torch.Tensor          # [B, S, V] draft logits per predicting slot (DSpark: Markov-biased)
    tokens: torch.Tensor          # [B, S] greedy proposal per slot
    confidence: torch.Tensor | None  # [B, S] DSpark acceptance probability
    hidden: torch.Tensor          # [B, block_size, H] final-normed block hidden states


class DFlashReference:
    def __init__(self, package: Package, weights: Weights, target: Reference):
        if package.architecture != "dflash":
            raise ValueError(f"not a dflash package: {package.architecture}")
        self.package, self.weights, self.target = package, weights, target
        self.block_count = package.key("block_count")
        self.hidden_size = package.key("embedding_length")
        if self.hidden_size != target.hidden_size:
            raise ValueError("the draft width differs from the target's")
        self.head_count = package.key("attention.head_count")
        self.kv_heads = package.key("attention.head_count_kv")
        self.head_dim = package.key("attention.key_length")
        self.eps = package.key("attention.layer_norm_rms_epsilon")
        self.block_size = package.key("block_size")
        self.target_layers = tuple(package.key("target_layers"))
        if max(self.target_layers) > target.block_count:
            raise ValueError(f"target layer {max(self.target_layers)} beyond the target's {target.block_count}")
        self.mask_token = package.metadata["tokenizer.ggml.mask_token_id"].value
        self.window = package.key("attention.sliding_window", 0)
        self.sliding = [bool(s) for s in per_layer(package.key("attention.sliding_window_pattern", False), self.block_count)]
        self.dspark = weights.has("markov_w1.weight")
        self.from_anchor = bool(package.key("sample_from_anchor", self.dspark))
        if weights.has("d2t"):
            raise ValueError("reduced-vocabulary drafts (d2t) are outside the in-scope catalog")
        base = package.key("rope.freq_base")
        if package.key("rope.scaling.type", "none") == "yarn":
            factor = package.key("rope.scaling.factor")
            self.inv_freq, self.amplitude = yarn_inv_freq(base, self.head_dim, factor,
                                                          package.key("rope.scaling.original_context_length"),
                                                          package.key("rope.scaling.yarn_beta_fast", 32.0),
                                                          package.key("rope.scaling.yarn_beta_slow", 1.0))
        else:
            self.inv_freq, self.amplitude = default_inv_freq(base, self.head_dim), 1.0

    def embed(self, tokens: torch.Tensor) -> torch.Tensor:
        if self.weights.has("token_embd.weight"):
            return self.weights("token_embd.weight")[tokens]
        return self.target.weights("token_embd.weight")[tokens]

    def head(self, hidden: torch.Tensor) -> torch.Tensor:
        return linear(hidden, self.weights("output.weight")) if self.weights.has("output.weight") else self.target.head(hidden)

    def context(self, tokens: torch.Tensor) -> torch.Tensor:
        """Target features of `tokens` [B, n]: concatenated tap residuals [B, n, taps * H]."""
        _, taps = self.target.hidden(tokens, taps=self.target_layers)
        by_layer = dict(zip(sorted(set(self.target_layers)), taps))
        return torch.cat([by_layer[layer] for layer in self.target_layers], dim=-1)

    def block(self, features: torch.Tensor, anchor: torch.Tensor) -> torch.Tensor:
        """Final-normed hidden states [B, block_size, H] of the block after n = features.shape[1] tokens."""
        w = self.weights
        batch, count = features.shape[:2]
        c = rms_norm(linear(features, w("fc.weight")), w("enc.output_norm.weight"), self.eps)
        tokens = torch.cat([anchor[:, None], torch.full((batch, self.block_size - 1), self.mask_token,
                                                        device=anchor.device)], dim=1)
        x = self.embed(tokens)
        positions = torch.arange(count + self.block_size, device=anchor.device)
        rotary = Rotary.build(positions, self.inv_freq, self.amplitude)
        block_rotary = Rotary(rotary.cos[count:], rotary.sin[count:])
        full = torch.ones(self.block_size, count + self.block_size, dtype=torch.bool, device=anchor.device)
        sliding = (positions[count:, None] - positions[None, :]).abs() <= self.window
        for layer in range(self.block_count):
            p = f"blk.{layer}."
            h = rms_norm(x, w(p + "attn_norm.weight"), self.eps)
            kv_input = torch.cat([c, h], dim=1)
            q = block_rotary.apply(rms_norm(heads(linear(h, w(p + "attn_q.weight")), self.head_count), w(p + "attn_q_norm.weight"), self.eps))
            k = rotary.apply(rms_norm(heads(linear(kv_input, w(p + "attn_k.weight")), self.kv_heads), w(p + "attn_k_norm.weight"), self.eps))
            v = heads(linear(kv_input, w(p + "attn_v.weight")), self.kv_heads)
            mixed = attention(q, k, v, sliding if self.sliding[layer] else full, self.head_dim ** -0.5)
            x = x + linear(mixed, w(p + "attn_output.weight"))
            h = rms_norm(x, w(p + "ffn_norm.weight"), self.eps)
            x = x + linear(silu(linear(h, w(p + "ffn_gate.weight"))) * linear(h, w(p + "ffn_up.weight")), w(p + "ffn_down.weight"))
        return rms_norm(x, w("output_norm.weight"), self.eps)

    def propose(self, tokens: torch.Tensor) -> Proposal:
        """One draft pass after `tokens` [B, n + 1]: the target has run tokens[:, :n]; tokens[:, n] is the anchor."""
        count = tokens.shape[1] - 1
        return self.propose_from(self.block(self.context(tokens[:, :count]), tokens[:, count]), tokens[:, count], count)

    def propose_from(self, hidden: torch.Tensor, anchor: torch.Tensor, count: int = 0) -> Proposal:
        """Slot predictions from the block hidden states of a pass after `count` target tokens."""
        if not self.dspark:
            first = 0 if self.from_anchor else 1
            logits = self.head(hidden[:, first:])
            return Proposal([count + first + i + int(self.from_anchor) for i in range(logits.shape[1])], logits,
                            logits.argmax(-1), None, hidden)
        w = self.weights
        base = self.head(hidden)
        previous = anchor
        logits, proposals, confidence = [], [], []
        for slot in range(self.block_size):
            memory = w.rows("markov_w1.weight", previous)
            slot_logits = base[:, slot] + linear(memory, w("markov_w2.weight"))
            features = torch.cat([hidden[:, slot], memory], dim=-1)
            confidence.append(torch.sigmoid(linear(features, w("conf_proj.weight").reshape(1, -1), w("conf_proj.bias")))[:, 0])
            previous = slot_logits.argmax(-1)
            logits.append(slot_logits)
            proposals.append(previous)
        return Proposal([count + 1 + i for i in range(self.block_size)], torch.stack(logits, 1), torch.stack(proposals, 1),
                        torch.stack(confidence, 1), hidden)

    # released-code cross-checks ----------------------------------------------------------------

    def released(self, features: torch.Tensor, anchor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Block hidden states from the released modeling code with this reference's weights, and for
        DSpark the Markov-biased slot logits and confidences from the released heads."""
        batch, count = features.shape[:2]
        tokens = torch.cat([anchor[:, None], torch.full((batch, self.block_size - 1), self.mask_token, device=anchor.device)], 1)
        positions = torch.arange(count + self.block_size, device=anchor.device)[None]
        if not self.dspark:
            from transformers.cache_utils import DFlashCache
            model = self.released_dflash()
            # Called as transformers' DFlashTokenCandidateGenerator calls it on its first draft: a fresh
            # DFlash cache told how many context rows precede the block, and a mask over context + block.
            cache = DFlashCache(config=model.config)
            cache.activate_past_recording()
            cache.set_previous_accepted_tokens(count)
            mask = torch.ones(batch, count + self.block_size, dtype=torch.long, device=anchor.device)
            with torch.no_grad():
                hidden = model(noise_embeds=self.embed(tokens), context_hidden_states=features, position_ids=positions,
                               attention_mask=mask, past_key_values=cache).last_hidden_state
            return hidden, None, None
        model = self.released_dspark()
        with torch.no_grad():
            hidden = model(position_ids=positions, noise_embedding=self.embed(tokens), target_hidden=features,
                           attention_mask=None, is_causal=False)
            base, previous, logits, confidence = self.head(hidden), anchor, [], []
            for slot in range(self.block_size):
                slot_logits = model.markov_head.apply_block_logits(base[:, slot:slot + 1], token_ids=previous[:, None])[:, 0]
                memory = model.markov_head.get_prev_embeddings(previous)
                confidence.append(torch.sigmoid(model.confidence_head(torch.cat([hidden[:, slot], memory], dim=-1))))
                previous = slot_logits.argmax(-1)
                logits.append(slot_logits)
        return hidden, torch.stack(logits, 1), torch.stack(confidence, 1)

    def check_released(self, features: torch.Tensor, anchor: torch.Tensor) -> dict:
        """This reference against the released code on the same context features and anchor."""
        with torch.no_grad():
            hidden = self.block(features, anchor)
            theirs, their_logits, their_confidence = self.released(features, anchor)
            report = {"hidden_relative_max_difference": float((hidden - theirs).abs().max() / theirs.abs().max())}
            passed = report["hidden_relative_max_difference"] < 1e-4
            if self.dspark:
                ours = self.propose_from(hidden, anchor)
                report["logits_relative_max_difference"] = float((ours.logits - their_logits).abs().max() / their_logits.abs().max())
                report["confidence_max_difference"] = float((ours.confidence - their_confidence).abs().max())
                report["proposals_agree"] = bool(torch.equal(ours.logits.argmax(-1), their_logits.argmax(-1)))
                passed &= report["logits_relative_max_difference"] < 1e-4 and report["confidence_max_difference"] < 1e-5
                passed &= report["proposals_agree"]
        return report | {"pass": bool(passed)}

    def layer_parameters(self, prefix: str) -> dict[str, torch.Tensor]:
        w, parameters = self.weights, {}
        for layer in range(self.block_count):
            p, h = f"blk.{layer}.", f"{prefix}layers.{layer}."
            for ours, theirs in [("attn_q", "self_attn.q_proj"), ("attn_k", "self_attn.k_proj"), ("attn_v", "self_attn.v_proj"),
                                 ("attn_output", "self_attn.o_proj"), ("attn_q_norm", "self_attn.q_norm"),
                                 ("attn_k_norm", "self_attn.k_norm"), ("attn_norm", "input_layernorm"),
                                 ("ffn_norm", "post_attention_layernorm"), ("ffn_gate", "mlp.gate_proj"),
                                 ("ffn_up", "mlp.up_proj"), ("ffn_down", "mlp.down_proj")]:
                parameters[h + theirs + ".weight"] = w(p + ours + ".weight")
        return parameters

    def rope_config(self) -> dict:
        if self.package.key("rope.scaling.type", "none") == "yarn":
            return {"rope_type": "yarn", "rope_theta": self.package.key("rope.freq_base"),
                    "factor": self.package.key("rope.scaling.factor"),
                    "original_max_position_embeddings": self.package.key("rope.scaling.original_context_length")}
        return {"rope_type": "default", "rope_theta": self.package.key("rope.freq_base")}

    def common_config(self) -> dict:
        return {"hidden_size": self.hidden_size, "intermediate_size": self.package.key("feed_forward_length"),
                "num_hidden_layers": self.block_count, "num_attention_heads": self.head_count,
                "num_key_value_heads": self.kv_heads, "head_dim": self.head_dim, "rms_norm_eps": self.eps,
                "vocab_size": self.target.vocab_size, "rope_parameters": self.rope_config(), "hidden_act": "silu",
                "layer_types": ["sliding_attention" if s else "full_attention" for s in self.sliding],
                # Unused without sliding layers; the released configs require an integer.
                "sliding_window": self.window or self.package.key("context_length"),
                "block_size": self.block_size, "mask_token_id": self.mask_token,
                "target_layer_ids": [layer - 1 for layer in self.target_layers], "attention_bias": False}

    def released_dflash(self):
        from transformers import AutoConfig, MuseGlimmerAssistantModel
        config = AutoConfig.for_model(model_type="muse_glimmer_assistant", **self.common_config())
        with torch.device(self.weights.device):
            model = MuseGlimmerAssistantModel._from_config(config, dtype=F32, attn_implementation="eager")
        w = self.weights
        parameters = self.layer_parameters("") | {"encoder.fc.weight": w("fc.weight"), "norm.weight": w("output_norm.weight"),
                                                  "encoder.output_norm_enc.weight": w("enc.output_norm.weight")}
        load_exactly(model, parameters)
        return model.eval()

    def released_dspark(self):
        from huggingface_hub import hf_hub_download
        repository, revision = DSPARK_CODE
        backbone = load_module("specforge.modeling.draft.dflash", hf_hub_download(repository, "dflash.py", revision=revision))
        # dspark.py imports its backbone from the SpecForge package it was published from.
        for package in ("specforge", "specforge.modeling", "specforge.modeling.draft"):
            sys.modules.setdefault(package, type(sys)(package))
        sys.modules["specforge.modeling.draft"].dflash = backbone
        module = load_module("dspark", hf_hub_download(repository, "dspark.py", revision=revision))
        config = module.DSparkConfig(**self.common_config(), num_target_layers=self.target.block_count,
                                     markov_rank=self.weights.shape("markov_w1.weight")[1],
                                     markov_head_type="vanilla", enable_confidence_head=True,
                                     confidence_head_with_markov=True,
                                     dflash_config={"target_layer_ids": [layer - 1 for layer in self.target_layers],
                                                    "mask_token_id": self.mask_token, "projector_type": "dspark"})
        config._attn_implementation = "eager"
        with torch.device(self.weights.device):
            model = module.DSparkDraftModel(config).to(F32)
        w = self.weights
        parameters = self.layer_parameters("") | {
            "fc.weight": w("fc.weight"), "hidden_norm.weight": w("enc.output_norm.weight"), "norm.weight": w("output_norm.weight"),
            "markov_head.markov_w1.weight": w("markov_w1.weight"), "markov_head.markov_w2.weight": w("markov_w2.weight"),
            "confidence_head.proj.weight": w("conf_proj.weight").reshape(1, -1), "confidence_head.proj.bias": w("conf_proj.bias")}
        load_exactly(model, parameters)
        return model.eval()


def load_module(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_exactly(model, parameters: dict[str, torch.Tensor]) -> None:
    expected = dict(model.state_dict())
    missing, extra = sorted(set(expected) - set(parameters)), sorted(set(parameters) - set(expected))
    if missing or extra:
        raise ValueError(f"released-code parameters not supplied: {missing}; unknown: {extra}")
    model.load_state_dict(parameters)


def open_draft(draft: Path, target: Path, device: torch.device) -> DFlashReference:
    package = Package.local(draft)
    return DFlashReference(package, Weights(package, device, cache=True), open_reference(target, device, True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("propose", "check"):
        command = commands.add_parser(name)
        command.add_argument("--draft", type=Path, required=True)
        command.add_argument("--target", type=Path, required=True)
        command.add_argument("--device", default="cpu")
        command.add_argument("--json", type=Path)
    commands.choices["propose"].add_argument("--tokens", required=True, help="committed tokens, the last one is the anchor")
    commands.choices["check"].add_argument("--context", type=int, default=24, help="context tokens (random features)")
    commands.choices["check"].add_argument("--seed", type=int, default=20260927)
    options = parser.parse_args()
    device = device_of(options.device)
    if options.command == "propose":
        draft = open_draft(options.draft, options.target, device)
        tokens = torch.tensor([[int(t) for t in options.tokens.split(",")]], device=device)
        with torch.no_grad():
            proposal = draft.propose(tokens)
        report = {"positions": proposal.positions, "proposals": proposal.tokens[0].tolist(),
                  "confidence": None if proposal.confidence is None else proposal.confidence[0].tolist()}
    else:
        # The draft forward given context features does not depend on the target's layers: random
        # features exercise it; the target contributes only its embedding table and head (weights are
        # read on use, so the target's layers are never loaded).
        draft = open_draft(options.draft, options.target, device)
        generator = torch.Generator().manual_seed(options.seed)
        features = torch.randn(1, options.context, len(draft.target_layers) * draft.hidden_size, generator=generator).to(device)
        anchor = torch.randint(0, draft.target.vocab_size, (1,), generator=generator).to(device)
        report = {"context": options.context, **draft.check_released(features, anchor)}
    report = {"draft": str(options.draft), "target": str(options.target), "dspark": draft.dspark, **report}
    text = json.dumps(report, indent=1)
    print(text)
    if options.json:
        options.json.parent.mkdir(parents=True, exist_ok=True)
        options.json.write_text(text + "\n")


if __name__ == "__main__":
    main()
