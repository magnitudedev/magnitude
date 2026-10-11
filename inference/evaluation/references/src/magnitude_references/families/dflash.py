"""Independent float32 reference for DFlash and DSpark drafts (GGUF architecture `dflash`).

`magnitude-references check-draft` compares it with the released draft code. Released-code adapters: Transformers 5.17.0 Muse assistant, an explicitly supplied pinned Z-Lab
DFlash/DFlash2 model.py, and DSpark dflash.py/dspark.py from a supplied source directory
(default RadixArk/Qwen3.8-27B-DSpark@b9a5dbdf). No one adapter defines every checkpoint mask.
Common dense-draft operations:
- context: the target's residual entering each of `target_layers` (the released `target_layer_ids`
  are layer *outputs*; the GGUF stores them + 1), float32, concatenated per token, then
  c = RMSNorm(fc c, enc.output_norm) for the n tokens the target has run;
- block: [anchor, mask x (block_size - 1)] at positions n .. n + block_size - 1, embedded with the raw
  target (or the draft's own) embedding table, no embedding norm or scale;
- each layer: q from RMSNorm(x); k, v from concat(c, RMSNorm(x)) (the context is not layer-normed);
  per-head q/k RMSNorm; rotate-half rope at the context and block positions (YaRN when the GGUF
  declares it); attention with explicit per-layer GGUF causality (no implicit family default);
  sliding layers keep |q - k| < W, and causal layers also require k <= q; SwiGLU; final RMSNorm; the target's LM head without logit scale or softcap;
- DFlash: block slot i >= 1 predicts position n + i. DSpark (`markov_w1` present, sample from the
  anchor): slot i >= 0 predicts position n + 1 + i with logits + markov_w2 markov_w1[prev_i], prev_0 =
  anchor, prev_{i+1} = the greedy token of slot i, and confidence
  sigmoid(conf_proj [h_i ; markov_w1[prev_i]] + b).
- DFlash2 adds grouped dynamic causal convolutions around attention and FFN updates,
  then chains candidate selection using predecessor/successor codebooks and hidden projection.

The released DSpark draft of LFM2 uses interleaved rope (`rope_is_neox_style: false`); the converter
reorders its q/k rows and norms to rotate-half order, so rotate-half on the GGUF rows is the same map.
"""
from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import json
from pathlib import Path
import sys

import torch

from magnitude_references.families import open_reference
from magnitude_references.gguf import Package
from magnitude_references.model import (F32, Reference, Rotary, Weights, attention, default_inv_freq, heads, linear,
                                        per_layer, rms_norm, silu, yarn_inv_freq)


def causal_layers(value: object, count: int) -> tuple[bool, ...]:
    """Explicit scalar BOOL or exactly one BOOL per layer; omission is not a default."""
    if type(value) is bool:
        return (value,) * count
    if isinstance(value, list) and len(value) == count and all(type(v) is bool for v in value):
        return tuple(value)
    raise ValueError("dflash.attention.causal requires BOOL or a BOOL array matching block_count")


def block_mask(count: int, rows: int, *, causal: bool, window: int | None,
               device: torch.device) -> torch.Tensor:
    """Canonical GGUF window counts the current position: a distance of W is excluded."""
    query = count + torch.arange(rows, device=device)[:, None]
    key = torch.arange(count + rows, device=device)[None, :]
    visible = torch.ones(rows, count + rows, dtype=torch.bool, device=device)
    if causal:
        visible &= key <= query
    if window is not None:
        if window <= 0:
            raise ValueError("a sliding layer requires a positive window")
        visible &= (query - key).abs() < window
    return visible


DSPARK_CODE = ("RadixArk/Qwen3.8-27B-DSpark", "b9a5dbdf03bc999c6c73c426b19c2d9041cea393")
# The Z-Lab DFlash/DFlash2 released code checked against (supplied as `source_path`):
# https://raw.githubusercontent.com/z-lab/dflash/07ebd93db9f472af339b644bb70221ad8428328a/dflash/model.py
ZLAB_CODE = ("z-lab/dflash", "07ebd93db9f472af339b644bb70221ad8428328a", "dflash/model.py")


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
        self.causal = causal_layers(package.key("attention.causal", None), self.block_count)
        self.sliding = [bool(s) for s in per_layer(package.key("attention.sliding_window_pattern", False), self.block_count)]
        self.dspark = weights.has("markov_w1.weight")
        self.dflash2 = weights.has("selector_hidden.weight")
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

    def block(self, features: torch.Tensor, anchor: torch.Tensor, rows: int | None = None) -> torch.Tensor:
        """Final-normed hidden states [B, block_size, H] of the block after n = features.shape[1] tokens."""
        w = self.weights
        rows = self.block_size if rows is None else rows
        if not 1 <= rows <= self.block_size:
            raise ValueError("physical rows outside the trained block")
        batch, count = features.shape[:2]
        c = rms_norm(linear(features, w("fc.weight")), w("enc.output_norm.weight"), self.eps)
        tokens = torch.cat([anchor[:, None], torch.full((batch, rows - 1), self.mask_token,
                                                        device=anchor.device)], dim=1)
        x = self.embed(tokens)
        positions = torch.arange(count + rows, device=anchor.device)
        rotary = Rotary.build(positions, self.inv_freq, self.amplitude)
        block_rotary = Rotary(rotary.cos[count:], rotary.sin[count:])
        for layer in range(self.block_count):
            p = f"blk.{layer}."
            h = rms_norm(x, w(p + "attn_norm.weight"), self.eps)
            after_attention = None
            if self.dflash2:
                h, after_attention = self.prepare_convolution(h, p + "attn")
            kv_input = torch.cat([c, h], dim=1)
            q = block_rotary.apply(rms_norm(heads(linear(h, w(p + "attn_q.weight")), self.head_count), w(p + "attn_q_norm.weight"), self.eps))
            k = rotary.apply(rms_norm(heads(linear(kv_input, w(p + "attn_k.weight")), self.kv_heads), w(p + "attn_k_norm.weight"), self.eps))
            v = heads(linear(kv_input, w(p + "attn_v.weight")), self.kv_heads)
            mixed = attention(q, k, v, block_mask(count, rows, causal=self.causal[layer],
                                           window=self.window if self.sliding[layer] else None, device=anchor.device), self.head_dim ** -0.5)
            update = linear(mixed, w(p + "attn_output.weight"))
            if after_attention is not None:
                update = self.convolve(update, after_attention, w(p + "attn_conv_base")[1])
            x = x + update
            h = rms_norm(x, w(p + "ffn_norm.weight"), self.eps)
            after_ffn = None
            if self.dflash2:
                h, after_ffn = self.prepare_convolution(h, p + "ffn")
            update = linear(silu(linear(h, w(p + "ffn_gate.weight"))) * linear(h, w(p + "ffn_up.weight")), w(p + "ffn_down.weight"))
            if after_ffn is not None:
                update = self.convolve(update, after_ffn, w(p + "ffn_conv_base")[1])
            x = x + update
        return rms_norm(x, w("output_norm.weight"), self.eps)

    def convolve(self, hidden: torch.Tensor, dynamic: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
        batch, rows, width = hidden.shape
        group_size = self.package.key("conv_group_size")
        output = torch.zeros_like(hidden)
        for offset in range(min(base.shape[0], rows)):
            coefficients = dynamic[:, offset:, offset].repeat_interleave(group_size, dim=-1)
            output[:, offset:] += hidden[:, :rows-offset] * (base[offset] + coefficients)
        return output

    def prepare_convolution(self, hidden: torch.Tensor, prefix: str):
        batch, rows, width = hidden.shape
        kernel = self.package.key("conv_kernel_size")
        groups = width // self.package.key("conv_group_size")
        dynamic = linear(hidden, self.weights(prefix + "_conv_proj.weight")).reshape(batch, rows, 2, kernel, groups)
        return self.convolve(hidden, dynamic[:, :, 0], self.weights(prefix + "_conv_base")[0]), dynamic[:, :, 1]

    def select_candidates(self, hidden: torch.Tensor, logits: torch.Tensor, anchor: torch.Tensor) -> torch.Tensor:
        scores, ids = logits.topk(self.package.key("selector_top_k"), dim=-1)
        projected = linear(hidden, self.weights("selector_hidden.weight"))
        previous, tokens = anchor, []
        for slot in range(hidden.shape[1]):
            pair = self.weights("selector_predecessor.weight")[previous] * projected[:, slot]
            successor = self.weights("selector_successor.weight")[ids[:, slot]]
            biased = scores[:, slot] + (pair[:, None] * successor).sum(-1)
            previous = ids[:, slot].gather(-1, biased.argmax(-1)[:, None])[:, 0]
            tokens.append(previous)
        return torch.stack(tokens, 1)

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
                            self.select_candidates(hidden[:, first:], logits, anchor) if self.dflash2 else logits.argmax(-1), None, hidden)
        w = self.weights
        base = self.head(hidden)
        previous = anchor
        logits, proposals, confidence = [], [], []
        for slot in range(0 if self.from_anchor else 1, hidden.shape[1]):
            memory = w.rows("markov_w1.weight", previous)
            slot_logits = base[:, slot] + linear(memory, w("markov_w2.weight"))
            features = torch.cat([hidden[:, slot], memory], dim=-1)
            confidence.append(torch.sigmoid(linear(features, w("conf_proj.weight").reshape(1, -1), w("conf_proj.bias")))[:, 0])
            previous = slot_logits.argmax(-1)
            logits.append(slot_logits)
            proposals.append(previous)
        return Proposal([count + 1 + i for i in range(len(logits))], torch.stack(logits, 1), torch.stack(proposals, 1),
                        torch.stack(confidence, 1), hidden)

    # released-code cross-checks ----------------------------------------------------------------

    def released(self, features: torch.Tensor, anchor: torch.Tensor, rows: int | None = None,
                 source: str = "muse", source_path: Path | None = None, source_config: Path | None = None) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Block hidden states from the released modeling code with this reference's weights, and for
        DSpark the Markov-biased slot logits and confidences from the released heads."""
        rows = self.block_size if rows is None else rows
        batch, count = features.shape[:2]
        tokens = torch.cat([anchor[:, None], torch.full((batch, rows - 1), self.mask_token, device=anchor.device)], 1)
        positions = torch.arange(count + rows, device=anchor.device)[None]
        if source == "zlab":
            if self.dspark or source_path is None:
                raise ValueError("zlab check requires its pinned model.py and a DFlash draft")
            model = self.released_zlab(source_path, source_config)
            with torch.no_grad():
                hidden = model(position_ids=positions, noise_embedding=self.embed(tokens), target_hidden=features)
            if self.dflash2:
                tokens, _, _ = model.propose(hidden[:, 1:], anchor, self.head, 0.0)
                return hidden, tokens, None
            return hidden, None, None
        if not self.dspark:
            if source != "muse" or any(self.causal):
                raise ValueError("Muse reference is only valid for explicitly noncausal drafts")
            from transformers.cache_utils import DFlashCache
            model = self.released_dflash()
            # Called as transformers' DFlashTokenCandidateGenerator calls it on its first draft: a fresh
            # DFlash cache told how many context rows precede the block, and a mask over context + block.
            cache = DFlashCache(config=model.config)
            cache.activate_past_recording()
            cache.set_previous_accepted_tokens(count)
            mask = torch.ones(batch, count + rows, dtype=torch.long, device=anchor.device)
            with torch.no_grad():
                hidden = model(noise_embeds=self.embed(tokens), context_hidden_states=features, position_ids=positions,
                               attention_mask=mask, past_key_values=cache).last_hidden_state
            return hidden, None, None
        if source != "dspark" or any(self.causal) or any(self.sliding):
            raise ValueError("released DSpark adapter supports only full noncausal layers")
        model = self.released_dspark(source_path)
        with torch.no_grad():
            hidden = model(position_ids=positions, noise_embedding=self.embed(tokens), target_hidden=features,
                           attention_mask=None, is_causal=False)
            base, previous, logits, confidence = self.head(hidden), anchor, [], []
            for slot in range(0 if self.from_anchor else 1, hidden.shape[1]):
                slot_logits = model.markov_head.apply_block_logits(base[:, slot:slot + 1], token_ids=previous[:, None])[:, 0]
                memory = model.markov_head.get_prev_embeddings(previous)
                confidence.append(torch.sigmoid(model.confidence_head(torch.cat([hidden[:, slot], memory], dim=-1))))
                previous = slot_logits.argmax(-1)
                logits.append(slot_logits)
        return hidden, torch.stack(logits, 1), torch.stack(confidence, 1)

    def check_released(self, features: torch.Tensor, anchor: torch.Tensor, rows: int | None = None,
                       source: str = "muse", source_path: Path | None = None, source_config: Path | None = None) -> dict:
        """This reference against the released code on the same context features and anchor."""
        with torch.no_grad():
            hidden = self.block(features, anchor, rows)
            theirs, their_logits, their_confidence = self.released(features, anchor, rows, source, source_path, source_config)
            report = {"hidden_relative_max_difference": float((hidden - theirs).abs().max() / theirs.abs().max())}
            passed = report["hidden_relative_max_difference"] < 1e-4
            if not self.dspark:
                ours = self.propose_from(hidden, anchor)
                theirs_logits = self.propose_from(theirs, anchor).logits
                report["logits_relative_max_difference"] = float((ours.logits - theirs_logits).abs().max() / theirs_logits.abs().max())
                report["proposals_agree"] = bool(torch.equal(ours.tokens, their_logits if self.dflash2 else theirs_logits.argmax(-1)))
                passed &= report["logits_relative_max_difference"] < 1e-4 and report["proposals_agree"]
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
                "use_sliding_window": any(self.sliding),
                "block_size": self.block_size, "mask_token_id": self.mask_token,
                "target_layer_ids": [layer - 1 for layer in self.target_layers], "attention_bias": False}

    def released_zlab(self, source_path: Path, source_config: Path | None = None):
        from transformers import Qwen3Config
        module = load_module("pinned_zlab_dflash", str(source_path))
        config = Qwen3Config(**self.common_config(), num_target_layers=self.target.block_count,
                            dflash_config={"target_layer_ids": [layer - 1 for layer in self.target_layers]})
        if source_config is not None:
            upstream = json.loads(source_config.read_text())
            if "is_causal" in upstream:
                config.is_causal = upstream["is_causal"]
        # Preserve upstream per-layer is_causal, including explicit source overrides.
        config._attn_implementation = "sdpa"
        with torch.device(self.weights.device):
            if self.dflash2:
                for key in ("conv_kernel_size", "conv_group_size", "selector_rank", "selector_top_k"):
                    setattr(config, key, self.package.key(key))
            model = (module.DFlash2DraftModel(config) if self.dflash2 else module.DFlashDraftModel(config)).to(F32)
        if tuple(layer.self_attn.is_causal for layer in model.layers) != self.causal:
            raise ValueError("GGUF mask differs from pinned Z-Lab reference semantics")
        w = self.weights
        parameters = self.layer_parameters("") | {"fc.weight": w("fc.weight"),
            "hidden_norm.weight": w("enc.output_norm.weight"), "norm.weight": w("output_norm.weight")}
        if self.dflash2:
            for layer in range(self.block_count):
                for ours, theirs in (("attn", "attention"), ("ffn", "mlp")):
                    parameters[f"layers.{layer}.{theirs}_conv.base_kernel"] = w(f"blk.{layer}.{ours}_conv_base")
                    parameters[f"layers.{layer}.{theirs}_conv.kernel_projection.weight"] = w(f"blk.{layer}.{ours}_conv_proj.weight")
            for ours, theirs in (("hidden", "hidden_projection"), ("predecessor", "predecessor_codebook"), ("successor", "successor_codebook")):
                parameters[f"candidate_selector.{theirs}.weight"] = w(f"selector_{ours}.weight")
        load_exactly(model, parameters)
        return model.eval()

    def released_dflash(self):
        from transformers import AutoConfig, MuseGlimmerAssistantModel
        configuration = self.common_config()
        # Transformers Muse specifies an inclusive radius; GGUF/engine specifies a token count.
        if any(self.sliding):
            configuration["sliding_window"] = self.window - 1
        config = AutoConfig.for_model(model_type="muse_glimmer_assistant", **configuration)
        with torch.device(self.weights.device):
            model = MuseGlimmerAssistantModel._from_config(config, dtype=F32, attn_implementation="eager")
        w = self.weights
        parameters = self.layer_parameters("") | {"encoder.fc.weight": w("fc.weight"), "norm.weight": w("output_norm.weight"),
                                                  "encoder.output_norm_enc.weight": w("enc.output_norm.weight")}
        load_exactly(model, parameters)
        return model.eval()

    def released_dspark(self, source_path: Path | None = None):
        from huggingface_hub import hf_hub_download
        repository, revision = DSPARK_CODE
        backbone_path = str(source_path / "dflash.py") if source_path else hf_hub_download(repository, "dflash.py", revision=revision)
        backbone = load_module("specforge.modeling.draft.dflash", backbone_path)
        # dspark.py imports its backbone from the SpecForge package it was published from.
        for package in ("specforge", "specforge.modeling", "specforge.modeling.draft"):
            sys.modules.setdefault(package, type(sys)(package))
        sys.modules["specforge.modeling.draft"].dflash = backbone
        dspark_path = str(source_path / "dspark.py") if source_path else hf_hub_download(repository, "dspark.py", revision=revision)
        module = load_module("dspark", dspark_path)
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


def check_draft(draft: DFlashReference, context: int, seed: int, rows: int | None, source: str,
                source_path: Path | None, source_config: Path | None) -> dict:
    """The draft against the released code on seeded random context features. The draft forward given
    features does not depend on the target's layers; the target contributes only its embedding table
    and head (weights are read on use, so the target's layers are never loaded)."""
    generator = torch.Generator().manual_seed(seed)
    device = draft.weights.device
    features = torch.randn(1, context, len(draft.target_layers) * draft.hidden_size, generator=generator).to(device)
    anchor = torch.randint(0, draft.target.vocab_size, (1,), generator=generator).to(device)
    return {"context": context, "dspark": draft.dspark,
            **draft.check_released(features, anchor, rows, source, source_path, source_config)}
