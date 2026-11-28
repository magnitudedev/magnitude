//! GPU execution of Inkling 2 DSpark (`DraftFusion::SoftmaxMix`,
//! `DraftMask::Learned`, `DraftContextProjection`, `MarkovHead::RnnHidden`).
//!
//! All heavy projections (`Q/K/V/O`, `K_ctx/V_ctx`, `gate/up/down`, `lm_head`,
//! and anchor token `embedding_rows`) execute on the target GPU device through
//! Seismic `project_rows` / `embedding_rows` kernels, keeping the implementation
//! format-agnostic across BF16, F16, Q8, NVFP4, and community quantizations.
//!
//! Algebraic identity for `MarkovHead::RnnHidden`:
//!   `lm_head @ h_draft + (lm_head @ hidden_out_w) @ latent + lm_head @ hidden_out_b`
//!   `== lm_head @ (h_draft + hidden_out_w @ latent + hidden_out_b)`
//! which eliminates load-time `[vocab, rank]` folding (0 ms bind time, 0 MB
//! extra VRAM) and works identically for any `lm_head` quantization format.

use super::{HeadOutput, HeadSelected, HeadSelections};
use crate::programs::native_target_graph::block_attn_res::{
    BoundWeight, clone_charges, prepare_kernel,
};
use crate::{
    DeviceError, HeadLaunchCore, InvariantError, ResidentHead, ResidentOutput, SubmitError,
};
use magnitude_family_contracts::{
    DraftDefinition, DraftFusion, DraftMask, DraftMethod, HeadNorm, MarkovHead, Operator, Rotary,
    RotaryPairing, RotaryPlacement, SublayerIndex, WeightKind, WeightRole, WeightScope,
};
use magnitude_kernels::{
    attention_append_dense, attention_decode, attention_output, dense_expand, dense_output,
    draft_top_k, embedding_rows, project_rows, readout_project_rows, rnn_markov_gate,
    token_confidence, vision_linear, vision_norm,
};
use magnitude_state::LayerRef;
use seismic::{
    BoundNativeGraphPlan, Charges, Device, Element, NativeGraphFamily, NativeGraphFamilySlot,
    NativeGraphPlan, NativeKernel, NativePort, NativeQueue, Tensor,
};
use std::collections::HashMap;
use std::sync::Mutex;

type ProjectKey = (Element, Element, Element, u64, u64, u64, u64);

fn invalid(message: impl Into<String>) -> SubmitError {
    SubmitError::Invariant(InvariantError {
        context: "native draft program",
        detail: message.into(),
    })
}

fn device_err(error: impl std::fmt::Display) -> SubmitError {
    SubmitError::Device(DeviceError::Execution(error.to_string()))
}

fn bf16_tensor(charges: &Charges, shape: &[u64], values: &[f32]) -> Result<Tensor, SubmitError> {
    let bytes: Vec<u8> = values
        .iter()
        .map(|v| ((v.to_bits().wrapping_add(0x7fff + ((v.to_bits() >> 16) & 1))) >> 16) as u16)
        .flat_map(|b| b.to_le_bytes())
        .collect();
    Tensor::from_host(charges, Element::bf16(), shape, &bytes).map_err(device_err)
}

fn f32_tensor(charges: &Charges, shape: &[u64], values: &[f32]) -> Result<Tensor, SubmitError> {
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    Tensor::from_host(charges, Element::f32(), shape, &bytes).map_err(device_err)
}

fn act_bytes(elem: Element, values: &[f32]) -> Result<Vec<u8>, SubmitError> {
    if elem == Element::bf16() {
        Ok(values
            .iter()
            .map(|v| ((v.to_bits().wrapping_add(0x7fff + ((v.to_bits() >> 16) & 1))) >> 16) as u16)
            .flat_map(|b| b.to_le_bytes())
            .collect())
    } else if elem == Element::f16() {
        Ok(values
            .iter()
            .map(|&v| f32_to_f16_bits(v))
            .flat_map(|b| b.to_le_bytes())
            .collect())
    } else {
        Err(invalid(format!("unsupported draft activation element {elem:?}")))
    }
}

fn rotary_tables(rotary: &Rotary, head_dim: u64) -> (u64, u64, u64, u64, Vec<i32>, Vec<f32>, Vec<f32>) {
    match rotary {
        Rotary::None => (0, 0, 0, head_dim, Vec::new(), Vec::new(), Vec::new()),
        Rotary::Interleaved { width, base, .. } => {
            let p = *width / 2;
            let s = head_dim.saturating_sub(2 * p);
            let components = vec![0i32; p as usize];
            let frequencies: Vec<f32> = (0..p)
                .map(|i| base.powf(-2.0 * (i as f64) / (*width as f64)) as f32)
                .collect();
            let amplitudes = vec![1.0f32; p as usize];
            (1, 0, p, s, components, frequencies, amplitudes)
        }
        Rotary::Table { pairs, layout, .. } => {
            let p = pairs.len() as u64;
            let s = head_dim.saturating_sub(2 * p);
            let ra = u64::from(matches!(layout.pairs, RotaryPairing::Adjacent));
            let rt = u64::from(matches!(layout.placement, RotaryPlacement::Trailing));
            let components = vec![0i32; p as usize];
            let frequencies = pairs.iter().map(|pair| pair.frequency as f32).collect();
            let amplitudes = pairs.iter().map(|pair| pair.amplitude as f32).collect();
            (ra, rt, p, s, components, frequencies, amplitudes)
        }
    }
}

fn i32_tensor(charges: &Charges, shape: &[u64], values: &[i32]) -> Result<Tensor, SubmitError> {
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    Tensor::from_host(charges, Element::i32(), shape, &bytes).map_err(device_err)
}

fn read_f32(t: &Tensor) -> Result<Vec<f32>, SubmitError> {
    let bytes = t.read_to_host().map_err(device_err)?;
    if t.element() == Element::f32() {
        Ok(bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect())
    } else if t.element() == Element::bf16() {
        Ok(bytes
            .chunks_exact(2)
            .map(|b| f32::from_bits(u32::from(u16::from_le_bytes([b[0], b[1]])) << 16))
            .collect())
    } else if t.element() == Element::f16() {
        Ok(bytes
            .chunks_exact(2)
            .map(|b| seismic::f16_to_f32(u16::from_le_bytes([b[0], b[1]])))
            .collect())
    } else {
        Err(invalid(format!(
            "unsupported draft tensor element {:?}",
            t.element()
        )))
    }
}

fn rms_norm_in_place(rows: &mut [f32], gamma: &[f32], eps: f32) {
    let width = gamma.len();
    for row in rows.chunks_exact_mut(width) {
        let sq: f32 = row.iter().map(|&v| v * v).sum();
        let inv = 1.0 / (sq / width as f32 + eps).sqrt();
        for (v, &g) in row.iter_mut().zip(gamma.iter()) {
            *v *= inv * g;
        }
    }
}

fn apply_rope(
    data: &mut [f32],
    rows: usize,
    heads: usize,
    head_dim: usize,
    positions: &[i32],
    rotary: &Rotary,
) {
    match rotary {
        Rotary::None => {}
        Rotary::Interleaved { width, base, .. } => {
            let half = (*width as usize) / 2;
            for r in 0..rows {
                let pos = f64::from(positions.get(r).copied().unwrap_or(0));
                for h in 0..heads {
                    let base_idx = (r * heads + h) * head_dim;
                    for i in 0..half {
                        let inv_freq = base.powf(-2.0 * (i as f64) / (*width as f64));
                        let angle = pos * inv_freq;
                        let (sin, cos) = (angle.sin() as f32, angle.cos() as f32);
                        let (i0, i1) = (base_idx + 2 * i, base_idx + 2 * i + 1);
                        let x0 = data[i0];
                        let x1 = data[i1];
                        data[i0] = x0 * cos - x1 * sin;
                        data[i1] = x0 * sin + x1 * cos;
                    }
                }
            }
        }
        Rotary::Table { pairs, layout, .. } => {
            let half = pairs.len();
            let interleaved = matches!(layout.pairs, RotaryPairing::Adjacent);
            let offset = match layout.placement {
                RotaryPlacement::Leading => 0,
                RotaryPlacement::Trailing => head_dim.saturating_sub(2 * half),
            };
            for r in 0..rows {
                let pos = f64::from(positions.get(r).copied().unwrap_or(0));
                for h in 0..heads {
                    let base_idx = (r * heads + h) * head_dim + offset;
                    for (i, pair) in pairs.iter().enumerate() {
                        let angle = pos * pair.frequency;
                        let amp = pair.amplitude as f32;
                        let (sin, cos) = (angle.sin() as f32 * amp, angle.cos() as f32 * amp);
                        let (i0, i1) = if interleaved {
                            (base_idx + 2 * i, base_idx + 2 * i + 1)
                        } else {
                            (base_idx + i, base_idx + half + i)
                        };
                        let x0 = data[i0];
                        let x1 = data[i1];
                        data[i0] = x0 * cos - x1 * sin;
                        data[i1] = x0 * sin + x1 * cos;
                    }
                }
            }
        }
    }
}

fn f32_to_f16_bits(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32 - 127;
    let mant = bits & 0x007f_ffff;
    if exp > 15 {
        return sign | 0x7bff; // clamp to max finite f16
    }
    if exp < -14 {
        if exp < -24 {
            return sign;
        }
        let m = (mant | 0x0080_0000) >> ((-14 - exp) + 13);
        return sign | (m as u16);
    }
    let half_exp = ((exp + 15) as u16) << 10;
    let half_mant = ((mant + 0x1000) >> 13) as u16;
    sign | half_exp.wrapping_add(half_mant)
}

/// Quantize a 2D BF16/F16 `[N, K]` (`K % 32 == 0`) target LM head into a
/// draft-only `q4g32s` (`Q4_0`) tensor (3.56x smaller bandwidth on every
/// speculative draft step; zero impact on target verification fidelity).
fn quantize_lm_head_q4_0(
    charges: &Charges,
    lm_head: &BoundWeight,
) -> Result<Option<BoundWeight>, SubmitError> {
    let elem = lm_head.element();
    if elem != Element::bf16() && elem != Element::f16() {
        return Ok(None);
    }
    let [n, k] = match lm_head.tensor.extents() {
        [n, k] if k % 32 == 0 => [*n as usize, *k as usize],
        _ => return Ok(None),
    };
    let Some(q4_elem) = Element::named("q4g32s") else {
        return Ok(None);
    };
    let Some(src_q4_elem) = Element::named("gguf_q4_0") else {
        return Ok(None);
    };
    let raw = lm_head.tensor.read_to_host().map_err(device_err)?;
    let is_bf16 = elem == Element::bf16();
    let blocks_per_row = k / 32;
    let total_blocks = n * blocks_per_row;
    let mut out = vec![0u8; total_blocks * 18];

    let decode_val = |pair: [u8; 2]| -> f32 {
        let bits = u16::from_le_bytes(pair);
        if is_bf16 {
            f32::from_bits(u32::from(bits) << 16)
        } else {
            seismic::f16_to_f32(bits)
        }
    };

    for blk in 0..total_blocks {
        let src_off = blk * 64;
        let mut vals = [0.0f32; 32];
        let mut amax = 0.0f32;
        for i in 0..32 {
            let v = decode_val([raw[src_off + 2 * i], raw[src_off + 2 * i + 1]]);
            vals[i] = v;
            amax = amax.max(v.abs());
        }
        let d = amax / 7.0f32;
        let inv_d = if d > 0.0 { 1.0 / d } else { 0.0 };
        let dst_off = blk * 18;
        let d_f16 = f32_to_f16_bits(d);
        out[dst_off..dst_off + 2].copy_from_slice(&d_f16.to_le_bytes());
        for i in 0..16 {
            let q0 = ((vals[i] * inv_d).round().clamp(-8.0, 7.0) as i32 + 8) as u8;
            let q1 = ((vals[i + 16] * inv_d).round().clamp(-8.0, 7.0) as i32 + 8) as u8;
            out[dst_off + 2 + i] = (q0 & 0x0f) | ((q1 & 0x0f) << 4);
        }
    }

    let shape = [n as u64, k as u64];
    let Some(repacked) = q4_elem.repack_host(src_q4_elem, &shape, &out) else {
        return Ok(None);
    };
    let Ok(resident_t) = Tensor::from_host(charges, q4_elem, &shape, &repacked) else {
        return Ok(None);
    };
    Ok(Some(BoundWeight {
        tensor: resident_t,
        scale: None,
        bound: None,
    }))
}

/// Sealed GPU graph for one Markov step: `project_rows(lm_head)` -> `draft_top_k(K=1)`
/// -> `token_confidence`. Avoids downloading `[slots, vocab]` logits to CPU!
struct BoundLmSelectGraph {
    plan: NativeGraphPlan,
    bound: BoundNativeGraphPlan,
    _family: NativeGraphFamily,
    slot: NativeGraphFamilySlot,
    source: NativePort,
    candidates: NativePort,
    unary: NativePort,
    confidence: NativePort,
    src_buf: Tensor,
    cand_buf: Tensor,
    unary_buf: Tensor,
    conf_buf: Tensor,
    _owned: Vec<Tensor>,
}

/// Sealed GPU graph for the DSpark entry/injection pass across all `L` draft
/// layers: `project_rows(K_ctx, V_ctx)` + `attention_append_dense` (head norm +
/// RoPE + write directly into the persistent GPU KV slabs).
struct BoundInjectGraph {
    plan: NativeGraphPlan,
    bound: BoundNativeGraphPlan,
    _family: NativeGraphFamily,
    slot: NativeGraphFamilySlot,
    fused_ports: Vec<NativePort>,
    coords_port: NativePort,
    dest_ports: Vec<NativePort>,
    hk_ports: Vec<NativePort>,
    hv_ports: Vec<NativePort>,
    coords_buf: Tensor,
    fused_bufs: Vec<Tensor>,
    dest_bufs: Vec<Tensor>,
    _owned: Vec<Tensor>,
}

/// Sealed GPU graph for the DSpark 4-layer block pass + final `output_norm`:
/// `vision_norm` -> `project_rows(Q,K,V)` -> `attention_decode` (history slabs +
/// fresh bidirectional block) -> `attention_output` -> `dense_expand` ->
/// `dense_output`, ending with `vision_norm(output_norm)`.
struct BoundBlockGraph {
    plan: NativeGraphPlan,
    bound: BoundNativeGraphPlan,
    _family: NativeGraphFamily,
    slot: NativeGraphFamilySlot,
    x_in_port: NativePort,
    coords_port: NativePort,
    fresh_port: NativePort,
    visible_ports: Vec<NativePort>,
    hk_ports: Vec<NativePort>,
    hv_ports: Vec<NativePort>,
    out_port: NativePort,
    x_in_buf: Tensor,
    coords_buf: Tensor,
    fresh_buf: Tensor,
    vis_bufs: Vec<Tensor>,
    out_buf: Tensor,
    _owned: Vec<Tensor>,
}

/// Sealed GPU graph run once before the Markov loop to project both
/// `W_frnn @ f_rnn + rnn_b` (`[S, 3*R]`) and `W_x @ h_draft_steps` (`[steps*S, 3*R]`)
/// in a single compute submission.
struct BoundMarkovPrepGraph {
    plan: NativeGraphPlan,
    bound: BoundNativeGraphPlan,
    _family: NativeGraphFamily,
    slot: NativeGraphFamilySlot,
    frnn_port: NativePort,
    hdraft_port: NativePort,
    frnn_out_port: NativePort,
    x_out_port: NativePort,
    frnn_buf: Tensor,
    hdraft_buf: Tensor,
    frnn_out_buf: Tensor,
    x_out_buf: Tensor,
    _owned: Vec<Tensor>,
}

/// Sealed GPU graph for one Markov step:
/// `vision_linear(latent, hidden_out_w, hidden_out_b, residual=h_draft[k]) -> h_eff [S, H] BF16`
/// -> `readout_project_rows(lm_head)` (or `project_rows`) -> `draft_top_k(K=1)` -> `token_confidence`.
/// Uploads only `[S, R]` BF16 latents per step (no `h_eff` upload, no CPU `hidden_out_w` matvec).
struct BoundMarkovStepGraph {
    plan: NativeGraphPlan,
    bound: BoundNativeGraphPlan,
    _family: NativeGraphFamily,
    slot: NativeGraphFamilySlot,
    latent_port: NativePort,
    hdraft_port: NativePort,
    candidates: NativePort,
    unary: NativePort,
    confidence: NativePort,
    latent_buf: Tensor,
    cand_buf: Tensor,
    unary_buf: Tensor,
    conf_buf: Tensor,
    _owned: Vec<Tensor>,
}

/// Sealed GPU graph for the entire `steps`-long RNN DSpark Markov chain:
/// `vision_linear(f_rnn, W_frnn, rnn_b)` + `vision_linear(h_draft_steps, W_x)` +
/// unrolled `k = 0..steps` of `embedding_rows(markov_emb, prev_tok)` ->
/// `vision_linear(prev_emb, W_e)` + `vision_linear(h_prev_a, W_h)` ->
/// `rnn_markov_gate` -> `vision_linear(latent, hidden_out_w, hidden_out_b, residual=h_draft[k])` ->
/// `readout_project_rows(lm_head, softcap)` (or `project_rows`) -> `draft_top_k(K=1)` ->
/// `token_confidence`.
/// Executes all `steps` in ONE single GPU compute submission with zero CPU↔GPU ping-pongs.
struct BoundMarkovChainGraph {
    plan: NativeGraphPlan,
    bound: BoundNativeGraphPlan,
    _family: NativeGraphFamily,
    slot: NativeGraphFamilySlot,
    frnn_port: NativePort,
    hdraft_act_port: NativePort,
    hdraft_f32_port: NativePort,
    anchor_port: NativePort,
    tokens_out_port: NativePort,
    unary_out_port: NativePort,
    conf_out_port: NativePort,
    frnn_buf: Tensor,
    hdraft_act_buf: Tensor,
    hdraft_f32_buf: Tensor,
    anchor_buf: Tensor,
    tokens_out_buf: Tensor,
    unary_out_buf: Tensor,
    conf_out_buf: Tensor,
    _owned: Vec<Tensor>,
}

/// Physical-history-row-owned `f_rnn` conditioning store.
///
/// Each committed entry row `r` writes its fused `f_rnn [hidden]` vector under
/// its physical history row index `dst >= 0` (the same physical row owned by the
/// head `StateStore` KV arena). Drafting resolves a sequence's `f_rnn` from the
/// last visible committed physical row in its `visible` spans, eliminating
/// transient launch-slot aliasing across repacking, cancellation, fork, and
/// rollback.
struct FRnnStore {
    rows: HashMap<i32, Vec<f32>>,
}

impl FRnnStore {
    fn new() -> Self {
        Self {
            rows: HashMap::new(),
        }
    }

    fn record_row(&mut self, physical_row: i32, f_rnn: Vec<f32>) {
        if physical_row >= 0 {
            self.rows.insert(physical_row, f_rnn);
        }
    }

    #[cfg_attr(not(test), allow(dead_code))]
    fn fork_row(&mut self, from_row: i32, to_row: i32) -> Result<(), SubmitError> {
        let Some(src) = self.rows.get(&from_row).cloned() else {
            return Err(invalid(format!(
                "cannot fork missing committed f_rnn at physical row {from_row}"
            )));
        };
        self.record_row(to_row, src);
        Ok(())
    }

    #[cfg_attr(not(test), allow(dead_code))]
    fn rollback_rows(&mut self, discarded_rows: &[i32]) {
        for &row in discarded_rows {
            self.rows.remove(&row);
        }
    }

    fn last_visible_row(visible: &[[i32; 2]]) -> Option<i32> {
        visible
            .iter()
            .rev()
            .find(|&&[start, end]| end > start)
            .map(|&[_, end]| end - 1)
    }

    fn resolve_visible(&self, visible: &[[i32; 2]]) -> Result<&[f32], SubmitError> {
        let last_row = Self::last_visible_row(visible).ok_or_else(|| {
            invalid("DSpark drafting slot has no committed visible history row for f_rnn")
        })?;
        self.rows
            .get(&last_row)
            .map(Vec::as_slice)
            .ok_or_else(|| {
                invalid(format!(
                    "DSpark drafting slot missing committed f_rnn for physical history row {last_row}"
                ))
            })
    }
}

/// Persistent per-layer KV cache and physical-row-owned `f_rnn` state for `BoundRnnDraft`.
struct RnnDraftCache {
    /// Per layer `l`, `kv[l]` maps physical history row -> `(k [kv_heads*head_dim], v [kv_heads*head_dim])`.
    layers: Vec<HashMap<i32, (Vec<f32>, Vec<f32>)>>,
    /// Per physical history row `dst`, the committed `f_rnn [hidden]`.
    f_rnn: FRnnStore,
}

#[derive(Clone)]
pub(crate) struct PreparedRnnDraft {
    pub(crate) draft: DraftDefinition,
    pub(crate) active_vocab: u64,
    pub(crate) hidden: u64,
    pub(crate) layer_history: Vec<(u64, u32)>,
    device: Device,
}

impl PreparedRnnDraft {
    pub(crate) fn new(
        device: &Device,
        draft: DraftDefinition,
        active_vocab: u64,
        hidden: u64,
        layer_history: Vec<(u64, u32)>,
    ) -> Self {
        Self {
            draft,
            active_vocab,
            hidden,
            layer_history,
            device: device.clone(),
        }
    }

    pub(crate) fn bind(
        &self,
        charges: &Charges,
        resident: &ResidentHead,
    ) -> Result<BoundRnnDraft, SubmitError> {
        let mut weights: HashMap<WeightRole, BoundWeight> = HashMap::new();
        for (&role, w) in resident.weights.iter() {
            weights.insert(role, BoundWeight::from_resident(w));
        }

        // Target LM head sliced to active (unpadded) vocabulary.
        let mut lm_head = match &resident.output {
            ResidentOutput::Packed(w) => {
                let sliced = w
                    .tensor()
                    .slice_leading(0, self.active_vocab)
                    .map_err(device_err)?;
                BoundWeight {
                    tensor: sliced,
                    scale: w.scale().cloned(),
                    bound: w.bound().cloned(),
                }
            }
            ResidentOutput::Progressive(_) => {
                return Err(invalid(
                    "Inkling 2 DSpark requires a packed target LM head",
                ));
            }
        };

        // Quantize BF16/F16 LM head to draft-only Q4_0 (`q4g32s`) when supported
        // by `project_rows` on this device (3.56x smaller bandwidth per draft step).
        if let Ok(Some(q4_lm)) = quantize_lm_head_q4_0(charges, &lm_head) {
            let [n, k] = match q4_lm.tensor.extents() {
                [n, k] => [*n, *k],
                _ => [0, 0],
            };
            if n > 0
                && k > 0
                && prepare_kernel(
                    &self.device,
                    project_rows::native_entry_with(project_rows::Elements {
                        A: Element::bf16(),
                        W: q4_lm.element(),
                        Y: Element::f32(),
                    }),
                    &[("G", 1), ("K", k), ("N", n), ("WS", 0), ("WB", 0)],
                )
                .is_ok()
            {
                lm_head = q4_lm;
            }
        }

        // Prepare GPU `project_rows` kernels for every distinct weight shape/element.
        let mut projects: HashMap<ProjectKey, NativeKernel<project_rows::Entry>> = HashMap::new();
        let mut ensure_project = |w: &BoundWeight| -> Result<(), SubmitError> {
            let [n, k] = match w.tensor.extents() {
                [n, k] => [*n, *k],
                _ => return Ok(()),
            };
            let key: ProjectKey = (
                Element::bf16(),
                w.element(),
                Element::f32(),
                k,
                n,
                w.scale_extent(),
                w.bound_extent(),
            );
            if projects.contains_key(&key) {
                return Ok(());
            }
            let entry = project_rows::native_entry_with(project_rows::Elements {
                A: Element::bf16(),
                W: w.element(),
                Y: Element::f32(),
            });
            if let Ok(kernel) = prepare_kernel(
                &self.device,
                entry,
                &[
                    ("G", 1),
                    ("K", k),
                    ("N", n),
                    ("WS", w.scale_extent()),
                    ("WB", w.bound_extent()),
                ],
            ) {
                projects.insert(key, kernel);
            }
            Ok(())
        };

        ensure_project(&lm_head)?;
        for (role, bw) in &weights {
            if matches!(
                role.kind,
                WeightKind::Query
                    | WeightKind::Key
                    | WeightKind::Value
                    | WeightKind::AttentionOutput
                    | WeightKind::DraftContextKey
                    | WeightKind::DraftContextValue
                    | WeightKind::DenseGate
                    | WeightKind::DenseUp
                    | WeightKind::DenseDown
            ) {
                ensure_project(bw)?;
            }
        }

        // Prepare GPU `embedding_rows` kernel for target embedding table.
        let emb_table = resident.embedding.tensor().clone();
        let emb_entry = embedding_rows::native_entry_with(embedding_rows::Elements {
            EW: emb_table.element(),
            A: Element::bf16(),
        });
        let emb_kernel = prepare_kernel(&self.device, emb_entry, &[("D", self.hidden)])
            .map_err(device_err)?;

        // Pre-read small 1-D / tiny host weights once at bind time (norms, mask,
        // fuse weights, and small Markov RNN weights).
        let read_role = |scope: WeightScope, kind: WeightKind| -> Result<Vec<f32>, SubmitError> {
            let bw = weights
                .get(&WeightRole { scope, kind })
                .ok_or_else(|| invalid(format!("missing draft weight {scope:?}::{kind:?}")))?;
            read_f32(&bw.tensor)
        };

        let mask_emb = match &self.draft.mask {
            DraftMask::Learned(_) => read_role(WeightScope::Draft, WeightKind::DraftMaskEmbedding)?,
            DraftMask::Token(_) => {
                return Err(invalid(
                    "BoundRnnDraft requires DraftMask::Learned (token mask is unsupported)",
                ));
            }
        };
        let output_norm = read_role(WeightScope::Draft, WeightKind::OutputNorm)?;

        let (fuse_weights, fuse_norms) = match &self.draft.fusion {
            DraftFusion::SoftmaxMix { norms, .. } => {
                if norms.len() != self.draft.blocks.len() + 1 {
                    return Err(invalid(format!(
                        "BoundRnnDraft SoftmaxMix fusion requires {} norms (blocks + f_rnn), got {}",
                        self.draft.blocks.len() + 1,
                        norms.len(),
                    )));
                }
                let fw = read_role(WeightScope::Draft, WeightKind::DraftFusion)?;
                let mut fn_vecs = Vec::with_capacity(norms.len());
                for (i, n) in norms.iter().enumerate() {
                    let scope = if i < self.draft.blocks.len() {
                        WeightScope::DraftSublayer(SublayerIndex {
                            block: i as u32,
                            sublayer: 0,
                        })
                    } else {
                        WeightScope::Draft
                    };
                    fn_vecs.push((
                        read_role(scope, WeightKind::DraftFusionNorm)?,
                        n.epsilon as f32,
                    ));
                }
                (fw, fn_vecs)
            }
            DraftFusion::Linear { .. } => {
                return Err(invalid(
                    "BoundRnnDraft requires DraftFusion::SoftmaxMix (linear fusion is unsupported)",
                ));
            }
        };

        let mut layer_norms = Vec::with_capacity(self.draft.blocks.len());
        for i in 0..self.draft.blocks.len() {
            let s0 = WeightScope::DraftSublayer(SublayerIndex {
                block: i as u32,
                sublayer: 0,
            });
            let s1 = WeightScope::DraftSublayer(SublayerIndex {
                block: i as u32,
                sublayer: 1,
            });
            layer_norms.push(LayerNormVectors {
                attn_norm: read_role(s0, WeightKind::InputNorm)?,
                q_norm: read_role(s0, WeightKind::QueryNorm)?,
                k_norm: read_role(s0, WeightKind::KeyNorm)?,
                k_norm_ctx: if self.draft.context_projections.is_some() {
                    Some(read_role(s0, WeightKind::DraftContextKeyNorm)?)
                } else {
                    None
                },
                ffn_norm: read_role(s1, WeightKind::InputNorm)?,
            });
        }

        let markov = match &self.draft.method {
            DraftMethod::DSpark {
                markov: MarkovHead::RnnHidden(cfg),
                ..
            } => {
                let rank = cfg.rank as usize;
                let h = self.hidden as usize;
                let in_dim = 2 * rank + 2 * h;
                let rnn_w = read_role(WeightScope::Draft, WeightKind::MarkovRnnWeight)?;
                let rnn_b = read_role(WeightScope::Draft, WeightKind::MarkovRnnBias)?;
                let hidden_out_w =
                    read_role(WeightScope::Draft, WeightKind::MarkovHiddenOutWeight)?;
                let hidden_out_b = read_role(WeightScope::Draft, WeightKind::MarkovHiddenOutBias)?;
                if rnn_w.len() != 3 * rank * in_dim {
                    return Err(invalid(format!(
                        "MarkovRnnWeight len {} != 3*rank*(2*rank+2*hidden) ({})",
                        rnn_w.len(),
                        3 * rank * in_dim
                    )));
                }
                let mut w_he = Vec::with_capacity(3 * rank * 2 * rank);
                let mut w_h_vals = Vec::with_capacity(3 * rank * rank);
                let mut w_e_vals = Vec::with_capacity(3 * rank * rank);
                let mut w_x_vals = Vec::with_capacity(3 * rank * h);
                let mut w_frnn_vals = Vec::with_capacity(3 * rank * h);
                for o in 0..3 * rank {
                    let row = &rnn_w[o * in_dim..(o + 1) * in_dim];
                    w_he.extend_from_slice(&row[0..2 * rank]);
                    w_h_vals.extend_from_slice(&row[0..rank]);
                    w_e_vals.extend_from_slice(&row[rank..2 * rank]);
                    w_x_vals.extend_from_slice(&row[2 * rank..2 * rank + h]);
                    w_frnn_vals.extend_from_slice(&row[2 * rank + h..in_dim]);
                }
                let r3 = (3 * rank) as u64;
                let hu = self.hidden;
                let ru = rank as u64;
                let w_h_gpu = bf16_tensor(charges, &[r3, ru], &w_h_vals)?;
                let w_e_gpu = bf16_tensor(charges, &[r3, ru], &w_e_vals)?;
                let w_x_gpu = bf16_tensor(charges, &[r3, hu], &w_x_vals)?;
                let w_frnn_gpu = bf16_tensor(charges, &[r3, hu], &w_frnn_vals)?;
                let rnn_b_gpu = f32_tensor(charges, &[1, r3], &rnn_b)?;
                let hidden_out_w_gpu = bf16_tensor(charges, &[hu, ru], &hidden_out_w)?;
                let hidden_out_b_gpu = f32_tensor(charges, &[1, hu], &hidden_out_b)?;
                let markov_emb_gpu = weights
                    .get(&WeightRole {
                        scope: WeightScope::Draft,
                        kind: WeightKind::MarkovEmbedding,
                    })
                    .ok_or_else(|| invalid("missing draft weight Draft::MarkovEmbedding"))?
                    .tensor
                    .clone();
                let markov_emb_kernel = prepare_kernel(
                    &self.device,
                    embedding_rows::native_entry_with(embedding_rows::Elements {
                        EW: markov_emb_gpu.element(),
                        A: Element::bf16(),
                    }),
                    &[("D", ru)],
                )
                .ok();

                let frnn_linear = prepare_kernel(
                    &self.device,
                    vision_linear::native_entry_with(vision_linear::Elements {
                        A: Element::bf16(),
                        W: Element::bf16(),
                        B: Element::f32(),
                        Y: Element::f32(),
                    }),
                    &[
                        ("N", r3),
                        ("K", hu),
                        ("NB", 1),
                        ("NR", 0),
                        ("NG", 0),
                        ("NC", 0),
                    ],
                )
                .ok();
                let x_linear = prepare_kernel(
                    &self.device,
                    vision_linear::native_entry_with(vision_linear::Elements {
                        A: Element::bf16(),
                        W: Element::bf16(),
                        B: Element::f32(),
                        Y: Element::f32(),
                    }),
                    &[
                        ("N", r3),
                        ("K", hu),
                        ("NB", 0),
                        ("NR", 0),
                        ("NG", 0),
                        ("NC", 0),
                    ],
                )
                .ok();
                let he_linear = prepare_kernel(
                    &self.device,
                    vision_linear::native_entry_with(vision_linear::Elements {
                        A: Element::bf16(),
                        W: Element::bf16(),
                        B: Element::f32(),
                        Y: Element::f32(),
                    }),
                    &[
                        ("N", r3),
                        ("K", ru),
                        ("NB", 0),
                        ("NR", 0),
                        ("NG", 0),
                        ("NC", 0),
                    ],
                )
                .ok();
                let out_linear = prepare_kernel(
                    &self.device,
                    vision_linear::native_entry_with(vision_linear::Elements {
                        A: Element::bf16(),
                        W: Element::bf16(),
                        B: Element::f32(),
                        Y: Element::bf16(),
                    }),
                    &[
                        ("N", hu),
                        ("K", ru),
                        ("NB", 1),
                        ("NR", 1),
                        ("NG", 0),
                        ("NC", 0),
                    ],
                )
                .ok();
                let gate_kernel = prepare_kernel(
                    &self.device,
                    rnn_markov_gate::native_entry_with(rnn_markov_gate::Elements {
                        A: Element::bf16(),
                    }),
                    &[],
                )
                .ok();

                MarkovWeights {
                    rank,
                    embedding: read_role(WeightScope::Draft, WeightKind::MarkovEmbedding)?,
                    w_he,
                    rnn_w,
                    rnn_b,
                    hidden_out_w,
                    hidden_out_b,
                    w_h_gpu,
                    w_e_gpu,
                    w_x_gpu,
                    w_frnn_gpu,
                    rnn_b_gpu,
                    hidden_out_w_gpu,
                    hidden_out_b_gpu,
                    markov_emb_gpu,
                    markov_emb_kernel,
                    frnn_linear,
                    x_linear,
                    he_linear,
                    out_linear,
                    gate_kernel,
                }
            }
            _ => {
                return Err(invalid(
                    "BoundRnnDraft requires DraftMethod::DSpark with MarkovHead::RnnHidden",
                ));
            }
        };

        let top_k_kernel = prepare_kernel(&self.device, draft_top_k::native_entry(), &[]).ok();
        let conf_kernel = prepare_kernel(&self.device, token_confidence::native_entry(), &[]).ok();
        let readout_project = match lm_head.tensor.extents() {
            &[v, d] if lm_head.bound_extent() == 0 => prepare_kernel(
                &self.device,
                readout_project_rows::native_entry_with(readout_project_rows::Elements {
                    A: Element::bf16(),
                    OW: lm_head.element(),
                }),
                &[
                    ("V", v),
                    ("D", d),
                    ("S", self.active_vocab.min(v)),
                    ("I", 0),
                    ("WS", lm_head.scale_extent()),
                ],
            )
            .ok(),
            _ => None,
        };
        let empty_f32 =
            Tensor::from_host(charges, Element::f32(), &[0], &[]).map_err(device_err)?;

        let num_layers = self.draft.blocks.len();
        Ok(BoundRnnDraft {
            prepared: self.clone(),
            weights,
            lm_head,
            emb_table,
            emb_kernel,
            readout_project,
            top_k_kernel,
            conf_kernel,
            lm_select_graphs: Mutex::new(HashMap::new()),
            markov_prep_graphs: Mutex::new(HashMap::new()),
            markov_step_graphs: Mutex::new(HashMap::new()),
            markov_chain_graphs: Mutex::new(HashMap::new()),
            inject_graphs: Mutex::new(HashMap::new()),
            block_graphs: Mutex::new(HashMap::new()),
            empty_f32,
            projects,
            mask_emb,
            output_norm,
            fuse_weights,
            fuse_norms,
            layer_norms,
            markov,
            cache: Mutex::new(RnnDraftCache {
                layers: (0..num_layers).map(|_| HashMap::new()).collect(),
                f_rnn: FRnnStore::new(),
            }),
            charges: clone_charges(charges),
        })
    }
}

struct LayerNormVectors {
    attn_norm: Vec<f32>,
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
    k_norm_ctx: Option<Vec<f32>>,
    ffn_norm: Vec<f32>,
}

struct MarkovWeights {
    rank: usize,
    embedding: Vec<f32>,
    /// Contiguous `[3 * rank, 2 * rank]` recurrent/embedding slice `[W_h | W_e]`.
    w_he: Vec<f32>,
    rnn_w: Vec<f32>,
    rnn_b: Vec<f32>,
    hidden_out_w: Vec<f32>,
    hidden_out_b: Vec<f32>,
    w_h_gpu: Tensor,
    w_e_gpu: Tensor,
    w_x_gpu: Tensor,
    w_frnn_gpu: Tensor,
    rnn_b_gpu: Tensor,
    hidden_out_w_gpu: Tensor,
    hidden_out_b_gpu: Tensor,
    markov_emb_gpu: Tensor,
    markov_emb_kernel: Option<NativeKernel<embedding_rows::Entry>>,
    frnn_linear: Option<NativeKernel<vision_linear::Entry>>,
    x_linear: Option<NativeKernel<vision_linear::Entry>>,
    he_linear: Option<NativeKernel<vision_linear::Entry>>,
    out_linear: Option<NativeKernel<vision_linear::Entry>>,
    gate_kernel: Option<NativeKernel<rnn_markov_gate::Entry>>,
}

pub(crate) struct BoundRnnDraft {
    prepared: PreparedRnnDraft,
    weights: HashMap<WeightRole, BoundWeight>,
    lm_head: BoundWeight,
    emb_table: Tensor,
    emb_kernel: NativeKernel<embedding_rows::Entry>,
    readout_project: Option<NativeKernel<readout_project_rows::Entry>>,
    top_k_kernel: Option<NativeKernel<draft_top_k::Entry>>,
    conf_kernel: Option<NativeKernel<token_confidence::Entry>>,
    lm_select_graphs: Mutex<HashMap<u64, BoundLmSelectGraph>>,
    markov_prep_graphs: Mutex<HashMap<(u64, u64), BoundMarkovPrepGraph>>,
    markov_step_graphs: Mutex<HashMap<u64, BoundMarkovStepGraph>>,
    markov_chain_graphs: Mutex<HashMap<(u64, u64), BoundMarkovChainGraph>>,
    inject_graphs: Mutex<HashMap<(u64, Element), BoundInjectGraph>>,
    block_graphs: Mutex<HashMap<(u64, u64, Element), BoundBlockGraph>>,
    empty_f32: Tensor,
    projects: HashMap<ProjectKey, NativeKernel<project_rows::Entry>>,
    mask_emb: Vec<f32>,
    output_norm: Vec<f32>,
    fuse_weights: Vec<f32>,
    fuse_norms: Vec<(Vec<f32>, f32)>,
    layer_norms: Vec<LayerNormVectors>,
    markov: MarkovWeights,
    cache: Mutex<RnnDraftCache>,
    charges: Charges,
}

impl BoundRnnDraft {
    fn project_gpu(&self, input: &[f32], rows: usize, w: &BoundWeight) -> Result<Vec<f32>, SubmitError> {
        if rows == 0 {
            return Ok(Vec::new());
        }
        let [n, k] = match w.tensor.extents() {
            [n, k] => [*n, *k],
            other => return Err(invalid(format!("expected 2D weight, got {other:?}"))),
        };
        let key: ProjectKey = (
            Element::bf16(),
            w.element(),
            Element::f32(),
            k,
            n,
            w.scale_extent(),
            w.bound_extent(),
        );
        let Some(kernel) = self.projects.get(&key) else {
            // Tiny projection (e.g., K % 64 != 0 on CUDA/Vulkan): evaluate directly in F32.
            let w_f32 = read_f32(&w.tensor)?;
            let mut out = vec![0.0f32; rows * (n as usize)];
            let ku = k as usize;
            let nu = n as usize;
            for r in 0..rows {
                let x_row = &input[r * ku..(r + 1) * ku];
                let y_row = &mut out[r * nu..(r + 1) * nu];
                for o in 0..nu {
                    let w_row = &w_f32[o * ku..(o + 1) * ku];
                    let mut sum = 0.0f32;
                    for c in 0..ku {
                        sum += x_row[c] * w_row[c];
                    }
                    y_row[o] = sum;
                }
            }
            return Ok(out);
        };
        // `default_specialization` of `project_rows` is always valid over
        // `1..=VERIFY_ROWS`; chunk larger context-entry batches (e.g. 512 rows)
        // so they never leave the prepared `M` domain.
        let max_chunk = magnitude_kernels::constants::VERIFY_ROWS as usize;
        let ku = k as usize;
        let nu = n as usize;
        let scale_t = w.scale.as_ref().unwrap_or(&self.empty_f32);
        let bound_t = w.bound.as_ref().unwrap_or(&self.empty_f32);
        let mut all_out = Vec::with_capacity(rows * nu);
        let mut start = 0usize;
        while start < rows {
            let chunk = (rows - start).min(max_chunk);
            let src = bf16_tensor(
                &self.charges,
                &[chunk as u64, k],
                &input[start * ku..(start + chunk) * ku],
            )?;
            let out = kernel
                .call(
                    &self.charges,
                    project_rows::Args {
                        source: &src,
                        weight: &w.tensor,
                        weight_scale: scale_t,
                        weight_bound: bound_t,
                    },
                )
                .map_err(|e| device_err(format!("rnn project_gpu(rows={chunk}, k={k}, n={n}): {e}")))?
                .value;
            all_out.extend_from_slice(&read_f32(&out)?);
            start += chunk;
        }
        Ok(all_out)
    }

    /// Run `readout_project_rows(h_eff, lm_head, softcap)` (or `project_rows` when
    /// `softcap == 0`) -> `draft_top_k(K=1)` -> `token_confidence` in a single sealed
    /// GPU graph with a persistent upload `src_buf`, returning `(best_tok, confidence)`
    /// per slot without ever copying `[slots, vocab]` logits back to CPU.
    fn select_lm_head_gpu(
        &self,
        h_eff: &[f32],
        slots: usize,
        softcap: f32,
    ) -> Result<Option<Vec<(usize, f32)>>, SubmitError> {
        let (Some(top_k_k), Some(conf_k)) = (&self.top_k_kernel, &self.conf_kernel) else {
            return Ok(None);
        };
        let m = slots as u64;
        let [v, k] = match self.lm_head.tensor.extents() {
            [v, k] => [*v, *k],
            _ => return Ok(None),
        };
        let use_readout = self.readout_project.is_some();
        if !use_readout && softcap > 0.0 {
            return Ok(None);
        }
        let proj_k_opt = if !use_readout {
            let key: ProjectKey = (
                Element::bf16(),
                self.lm_head.element(),
                Element::f32(),
                k,
                v,
                self.lm_head.scale_extent(),
                self.lm_head.bound_extent(),
            );
            match self.projects.get(&key) {
                Some(pk) => Some(pk),
                None => return Ok(None),
            }
        } else {
            None
        };

        let mut cache = self.lm_select_graphs.lock().expect("lm_select_graphs lock");
        if !cache.contains_key(&m) {
            let device = self.charges.device();
            let mut graph = device.native_graph();
            let source = graph
                .port(Element::bf16(), &[m, k])
                .map_err(device_err)?;
            let weight = graph
                .port(self.lm_head.element(), &[v, k])
                .map_err(device_err)?;
            let scale_t = self.lm_head.scale.as_ref().unwrap_or(&self.empty_f32);
            let scale_p = graph
                .port(scale_t.element(), scale_t.extents())
                .map_err(device_err)?;
            let id_rows_t = i32_tensor(&self.charges, &[m], &(0..m as i32).collect::<Vec<_>>())?;
            let id_rows_p = graph
                .port(Element::i32(), &[m])
                .map_err(device_err)?;
            let mut candidates = graph
                .port(Element::i32(), &[m, 2])
                .map_err(device_err)?;
            let mut unary = graph
                .port(Element::f32(), &[m, 1])
                .map_err(device_err)?;
            let mut confidence = graph
                .port(Element::f32(), &[m])
                .map_err(device_err)?;

            let mut owned = vec![id_rows_t.clone()];
            let mut static_binds: Vec<(NativePort, Tensor)> = vec![
                (weight.clone(), self.lm_head.tensor.clone()),
                (scale_p.clone(), scale_t.clone()),
                (id_rows_p.clone(), id_rows_t),
            ];
            let logits = if let Some(rp_k) = &self.readout_project {
                let active_s = self.prepared.active_vocab.min(v);
                let empty_tok_t =
                    Tensor::from_host(&self.charges, Element::i32(), &[m, 0, active_s], &[])
                        .map_err(device_err)?;
                let empty_tok_p = graph
                    .port(Element::i32(), &[m, 0, active_s])
                    .map_err(device_err)?;
                static_binds.push((empty_tok_p.clone(), empty_tok_t.clone()));
                owned.push(empty_tok_t);
                graph
                    .enqueue(
                        rp_k,
                        readout_project_rows::WorkflowArgs {
                            features: source.tensor().into(),
                            weight: weight.tensor().into(),
                            token_rows: empty_tok_p.tensor().into(),
                            weight_scale: scale_p.tensor().into(),
                            softcap: softcap.into(),
                        },
                    )
                    .map_err(device_err)?
                    .value
            } else {
                let bound_t = self.lm_head.bound.as_ref().unwrap_or(&self.empty_f32);
                let bound_p = graph
                    .port(bound_t.element(), bound_t.extents())
                    .map_err(device_err)?;
                static_binds.push((bound_p.clone(), bound_t.clone()));
                graph
                    .enqueue(
                        proj_k_opt.unwrap(),
                        project_rows::WorkflowArgs {
                            source: source.tensor().into(),
                            weight: weight.tensor().into(),
                            weight_scale: scale_p.tensor().into(),
                            weight_bound: bound_p.tensor().into(),
                        },
                    )
                    .map_err(device_err)?
                    .value
            };
            graph
                .enqueue(
                    top_k_k,
                    draft_top_k::WorkflowArgs {
                        logits: (&logits).into(),
                        candidates: candidates.tensor_mut().into(),
                        unary: unary.tensor_mut().into(),
                    },
                )
                .map_err(device_err)?;
            graph
                .enqueue(
                    conf_k,
                    token_confidence::WorkflowArgs {
                        logits: (&logits).into(),
                        rows: id_rows_p.tensor().into(),
                        selection: candidates.tensor().into(),
                        confidence: confidence.tensor_mut().into(),
                    },
                )
                .map_err(device_err)?;

            let plan = graph.seal().map_err(device_err)?;
            let bind_refs: Vec<(&NativePort, &Tensor)> =
                static_binds.iter().map(|(p, t)| (p, t)).collect();
            let bound = plan
                .bind_static(&self.charges, &bind_refs)
                .map_err(device_err)?;
            let family = NativeGraphFamily::with_launches(
                std::slice::from_ref(&plan),
                &[vec![plan.clone()]],
                1,
                &[],
            )
            .map_err(device_err)?;
            let slot = family.new_slot(&self.charges, &[]).map_err(device_err)?;
            let src_buf =
                Tensor::upload(&self.charges, Element::bf16(), &[m, k]).map_err(device_err)?;
            let cand_buf =
                Tensor::zeros(&self.charges, Element::i32(), &[m, 2]).map_err(device_err)?;
            let unary_buf =
                Tensor::zeros(&self.charges, Element::f32(), &[m, 1]).map_err(device_err)?;
            let conf_buf =
                Tensor::zeros(&self.charges, Element::f32(), &[m]).map_err(device_err)?;
            cache.insert(
                m,
                BoundLmSelectGraph {
                    plan,
                    bound,
                    _family: family,
                    slot,
                    source,
                    candidates,
                    unary,
                    confidence,
                    src_buf,
                    cand_buf,
                    unary_buf,
                    conf_buf,
                    _owned: owned,
                },
            );
        }

        let g = cache.get_mut(&m).unwrap();
        let src_bytes: Vec<u8> = h_eff
            .iter()
            .map(|v| ((v.to_bits().wrapping_add(0x7fff + ((v.to_bits() >> 16) & 1))) >> 16) as u16)
            .flat_map(|b| b.to_le_bytes())
            .collect();
        g.src_buf.write_from_host(&src_bytes).map_err(device_err)?;
        let mut bindings = g.bound.bindings();
        bindings
            .set(&self.charges, &g.source, &g.src_buf)
            .map_err(device_err)?;
        bindings
            .set(&self.charges, &g.candidates, &g.cand_buf)
            .map_err(device_err)?;
        bindings
            .set(&self.charges, &g.unary, &g.unary_buf)
            .map_err(device_err)?;
        bindings
            .set(&self.charges, &g.confidence, &g.conf_buf)
            .map_err(device_err)?;
        let mut active = g.slot.activate(&g.plan).map_err(device_err)?;
        let ready = active
            .attach(bindings, g.plan.new_outputs(&self.charges).map_err(device_err)?)
            .map_err(device_err)?;
        let mut seq = self.charges.device().native_sequence(NativeQueue::Compute);
        let _ = ready.queue(&mut seq).map_err(device_err)?;
        seq.submit()
            .map_err(device_err)?
            .wait()
            .map_err(device_err)?;

        let cand_bytes = g.cand_buf.read_to_host().map_err(device_err)?;
        let conf_bytes = g.conf_buf.read_to_host().map_err(device_err)?;
        let mut out = Vec::with_capacity(slots);
        for s in 0..slots {
            let tok = i32::from_le_bytes(cand_bytes[s * 8..s * 8 + 4].try_into().unwrap())
                .max(0) as usize;
            let prob = f32::from_le_bytes(conf_bytes[s * 4..s * 4 + 4].try_into().unwrap())
                .clamp(0.0, 1.0);
            out.push((tok, prob));
        }
        Ok(Some(out))
    }

    /// Project `rnn_b + W_frnn @ f_rnn` (`[slots, 3*R]`) and `W_x @ h_draft_steps`
    /// (`[steps * slots, 3*R]`) on GPU in a single sealed `NativeGraph` submit,
    /// returning combined `step_biases [steps, slots, 3*R]`.
    fn prep_markov_gpu(
        &self,
        f_rnn_rows: &[f32],
        h_draft_steps: &[f32],
        slots: usize,
        steps: usize,
    ) -> Result<Option<Vec<f32>>, SubmitError> {
        let (Some(frnn_k), Some(x_k)) = (&self.markov.frnn_linear, &self.markov.x_linear) else {
            return Ok(None);
        };
        let m = slots as u64;
        let st = steps as u64;
        let total = m * st;
        let h = self.prepared.hidden;
        let r3 = (3 * self.markov.rank) as u64;
        let key = (m, st);

        let mut cache = self
            .markov_prep_graphs
            .lock()
            .expect("markov_prep_graphs lock");
        if !cache.contains_key(&key) {
            let device = self.charges.device();
            let mut graph = device.native_graph();
            let frnn_port = graph.port(Element::bf16(), &[m, h]).map_err(device_err)?;
            let hdraft_port = graph
                .port(Element::bf16(), &[total, h])
                .map_err(device_err)?;
            let frnn_out_port = graph.port(Element::f32(), &[m, r3]).map_err(device_err)?;
            let x_out_port = graph
                .port(Element::f32(), &[total, r3])
                .map_err(device_err)?;

            let w_frnn_p = graph.port(Element::bf16(), &[r3, h]).map_err(device_err)?;
            let rnn_b_p = graph.port(Element::f32(), &[1, r3]).map_err(device_err)?;
            let w_x_p = graph.port(Element::bf16(), &[r3, h]).map_err(device_err)?;

            let empty_bias_t = f32_tensor(&self.charges, &[0, r3], &[])?;
            let empty_res_f_t = f32_tensor(&self.charges, &[0, m, r3], &[])?;
            let empty_gate_f_t =
                Tensor::from_host(&self.charges, Element::bf16(), &[0, m, r3], &[])
                    .map_err(device_err)?;
            let empty_res_x_t = f32_tensor(&self.charges, &[0, total, r3], &[])?;
            let empty_gate_x_t =
                Tensor::from_host(&self.charges, Element::bf16(), &[0, total, r3], &[])
                    .map_err(device_err)?;

            let empty_f32_p = graph.port(Element::f32(), &[0]).map_err(device_err)?;
            let empty_bias_p = graph.port(Element::f32(), &[0, r3]).map_err(device_err)?;
            let empty_res_f_p = graph
                .port(Element::f32(), &[0, m, r3])
                .map_err(device_err)?;
            let empty_gate_f_p = graph
                .port(Element::bf16(), &[0, m, r3])
                .map_err(device_err)?;
            let empty_res_x_p = graph
                .port(Element::f32(), &[0, total, r3])
                .map_err(device_err)?;
            let empty_gate_x_p = graph
                .port(Element::bf16(), &[0, total, r3])
                .map_err(device_err)?;

            let frnn_out = graph
                .enqueue(
                    frnn_k,
                    vision_linear::WorkflowArgs {
                        x: frnn_port.tensor().into(),
                        weight: w_frnn_p.tensor().into(),
                        bias: rnn_b_p.tensor().into(),
                        residual: empty_res_f_p.tensor().into(),
                        gate: empty_gate_f_p.tensor().into(),
                        minimum: empty_f32_p.tensor().into(),
                        maximum: empty_f32_p.tensor().into(),
                        activation: 0i32.into(),
                    },
                )
                .map_err(device_err)?
                .value;
            let x_out = graph
                .enqueue(
                    x_k,
                    vision_linear::WorkflowArgs {
                        x: hdraft_port.tensor().into(),
                        weight: w_x_p.tensor().into(),
                        bias: empty_bias_p.tensor().into(),
                        residual: empty_res_x_p.tensor().into(),
                        gate: empty_gate_x_p.tensor().into(),
                        minimum: empty_f32_p.tensor().into(),
                        maximum: empty_f32_p.tensor().into(),
                        activation: 0i32.into(),
                    },
                )
                .map_err(device_err)?
                .value;

            graph.store(&frnn_out, &frnn_out_port).map_err(device_err)?;
            graph.store(&x_out, &x_out_port).map_err(device_err)?;

            let plan = graph.seal().map_err(device_err)?;
            let static_binds: Vec<(NativePort, Tensor)> = vec![
                (w_frnn_p, self.markov.w_frnn_gpu.clone()),
                (rnn_b_p, self.markov.rnn_b_gpu.clone()),
                (w_x_p, self.markov.w_x_gpu.clone()),
                (empty_f32_p, self.empty_f32.clone()),
                (empty_bias_p, empty_bias_t.clone()),
                (empty_res_f_p, empty_res_f_t.clone()),
                (empty_gate_f_p, empty_gate_f_t.clone()),
                (empty_res_x_p, empty_res_x_t.clone()),
                (empty_gate_x_p, empty_gate_x_t.clone()),
            ];
            let bind_refs: Vec<(&NativePort, &Tensor)> =
                static_binds.iter().map(|(p, t)| (p, t)).collect();
            let bound = plan
                .bind_static(&self.charges, &bind_refs)
                .map_err(device_err)?;
            let family = NativeGraphFamily::with_launches(
                std::slice::from_ref(&plan),
                &[vec![plan.clone()]],
                1,
                &[],
            )
            .map_err(device_err)?;
            let slot = family.new_slot(&self.charges, &[]).map_err(device_err)?;
            let frnn_buf =
                Tensor::upload(&self.charges, Element::bf16(), &[m, h]).map_err(device_err)?;
            let hdraft_buf =
                Tensor::upload(&self.charges, Element::bf16(), &[total, h]).map_err(device_err)?;
            let frnn_out_buf =
                Tensor::zeros(&self.charges, Element::f32(), &[m, r3]).map_err(device_err)?;
            let x_out_buf =
                Tensor::zeros(&self.charges, Element::f32(), &[total, r3]).map_err(device_err)?;
            cache.insert(
                key,
                BoundMarkovPrepGraph {
                    plan,
                    bound,
                    _family: family,
                    slot,
                    frnn_port,
                    hdraft_port,
                    frnn_out_port,
                    x_out_port,
                    frnn_buf,
                    hdraft_buf,
                    frnn_out_buf,
                    x_out_buf,
                    _owned: vec![
                        empty_bias_t,
                        empty_res_f_t,
                        empty_gate_f_t,
                        empty_res_x_t,
                        empty_gate_x_t,
                    ],
                },
            );
        }

        let g = cache.get_mut(&key).unwrap();
        let frnn_bytes = act_bytes(Element::bf16(), f_rnn_rows)?;
        let hdraft_bytes = act_bytes(Element::bf16(), h_draft_steps)?;
        g.frnn_buf
            .write_from_host(&frnn_bytes)
            .map_err(device_err)?;
        g.hdraft_buf
            .write_from_host(&hdraft_bytes)
            .map_err(device_err)?;
        let mut bindings = g.bound.bindings();
        bindings
            .set(&self.charges, &g.frnn_port, &g.frnn_buf)
            .map_err(device_err)?;
        bindings
            .set(&self.charges, &g.hdraft_port, &g.hdraft_buf)
            .map_err(device_err)?;
        bindings
            .set(&self.charges, &g.frnn_out_port, &g.frnn_out_buf)
            .map_err(device_err)?;
        bindings
            .set(&self.charges, &g.x_out_port, &g.x_out_buf)
            .map_err(device_err)?;
        let mut active = g.slot.activate(&g.plan).map_err(device_err)?;
        let ready = active
            .attach(
                bindings,
                g.plan.new_outputs(&self.charges).map_err(device_err)?,
            )
            .map_err(device_err)?;
        let mut seq = self.charges.device().native_sequence(NativeQueue::Compute);
        let _ = ready.queue(&mut seq).map_err(device_err)?;
        seq.submit()
            .map_err(device_err)?
            .wait()
            .map_err(device_err)?;

        let frnn_out = read_f32(&g.frnn_out_buf)?;
        let mut x_out = read_f32(&g.x_out_buf)?;
        let r3u = 3 * self.markov.rank;
        for k in 0..steps {
            for s in 0..slots {
                let dst = &mut x_out[(k * slots + s) * r3u..(k * slots + s + 1) * r3u];
                let src = &frnn_out[s * r3u..(s + 1) * r3u];
                for (d, &f) in dst.iter_mut().zip(src.iter()) {
                    *d += f;
                }
            }
        }
        Ok(Some(x_out))
    }

    /// Execute one Markov output step in a single sealed GPU graph:
    /// `vision_linear(latent, hidden_out_w, hidden_out_b, residual=h_draft_step)`
    /// -> `readout_project_rows(lm_head, softcap)` (or `project_rows`)
    /// -> `draft_top_k(K=1)` -> `token_confidence`.
    fn step_markov_gpu(
        &self,
        latents: &[f32],
        h_draft_step: &Tensor,
        slots: usize,
        softcap: f32,
    ) -> Result<Option<Vec<(usize, f32)>>, SubmitError> {
        let (Some(out_k), Some(top_k_k), Some(conf_k)) = (
            &self.markov.out_linear,
            &self.top_k_kernel,
            &self.conf_kernel,
        ) else {
            return Ok(None);
        };
        let m = slots as u64;
        let r = self.markov.rank as u64;
        let [v, k] = match self.lm_head.tensor.extents() {
            [v, k] => [*v, *k],
            _ => return Ok(None),
        };
        let use_readout = self.readout_project.is_some();
        if !use_readout && softcap > 0.0 {
            return Ok(None);
        }
        let proj_k_opt = if !use_readout {
            let key: ProjectKey = (
                Element::bf16(),
                self.lm_head.element(),
                Element::f32(),
                k,
                v,
                self.lm_head.scale_extent(),
                self.lm_head.bound_extent(),
            );
            match self.projects.get(&key) {
                Some(pk) => Some(pk),
                None => return Ok(None),
            }
        } else {
            None
        };

        let mut cache = self
            .markov_step_graphs
            .lock()
            .expect("markov_step_graphs lock");
        if !cache.contains_key(&m) {
            let device = self.charges.device();
            let mut graph = device.native_graph();
            let latent_port = graph.port(Element::bf16(), &[m, r]).map_err(device_err)?;
            let hdraft_port = graph
                .port(Element::f32(), &[1, m, k])
                .map_err(device_err)?;
            let out_w_p = graph.port(Element::bf16(), &[k, r]).map_err(device_err)?;
            let out_b_p = graph.port(Element::f32(), &[1, k]).map_err(device_err)?;
            let empty_gate_t =
                Tensor::from_host(&self.charges, Element::bf16(), &[0, m, k], &[])
                    .map_err(device_err)?;
            let empty_gate_p = graph
                .port(Element::bf16(), &[0, m, k])
                .map_err(device_err)?;
            let empty_f32_p = graph.port(Element::f32(), &[0]).map_err(device_err)?;

            let weight = graph
                .port(self.lm_head.element(), &[v, k])
                .map_err(device_err)?;
            let scale_t = self.lm_head.scale.as_ref().unwrap_or(&self.empty_f32);
            let scale_p = graph
                .port(scale_t.element(), scale_t.extents())
                .map_err(device_err)?;
            let id_rows_t = i32_tensor(&self.charges, &[m], &(0..m as i32).collect::<Vec<_>>())?;
            let id_rows_p = graph.port(Element::i32(), &[m]).map_err(device_err)?;
            let mut candidates = graph.port(Element::i32(), &[m, 2]).map_err(device_err)?;
            let mut unary = graph.port(Element::f32(), &[m, 1]).map_err(device_err)?;
            let mut confidence = graph.port(Element::f32(), &[m]).map_err(device_err)?;

            let mut owned = vec![id_rows_t.clone(), empty_gate_t.clone()];
            let mut static_binds: Vec<(NativePort, Tensor)> = vec![
                (out_w_p.clone(), self.markov.hidden_out_w_gpu.clone()),
                (out_b_p.clone(), self.markov.hidden_out_b_gpu.clone()),
                (empty_gate_p.clone(), empty_gate_t),
                (empty_f32_p.clone(), self.empty_f32.clone()),
                (weight.clone(), self.lm_head.tensor.clone()),
                (scale_p.clone(), scale_t.clone()),
                (id_rows_p.clone(), id_rows_t),
            ];

            let h_eff = graph
                .enqueue(
                    out_k,
                    vision_linear::WorkflowArgs {
                        x: latent_port.tensor().into(),
                        weight: out_w_p.tensor().into(),
                        bias: out_b_p.tensor().into(),
                        residual: hdraft_port.tensor().into(),
                        gate: empty_gate_p.tensor().into(),
                        minimum: empty_f32_p.tensor().into(),
                        maximum: empty_f32_p.tensor().into(),
                        activation: 0i32.into(),
                    },
                )
                .map_err(device_err)?
                .value;

            let logits = if let Some(rp_k) = &self.readout_project {
                let active_s = self.prepared.active_vocab.min(v);
                let empty_tok_t =
                    Tensor::from_host(&self.charges, Element::i32(), &[m, 0, active_s], &[])
                        .map_err(device_err)?;
                let empty_tok_p = graph
                    .port(Element::i32(), &[m, 0, active_s])
                    .map_err(device_err)?;
                static_binds.push((empty_tok_p.clone(), empty_tok_t.clone()));
                owned.push(empty_tok_t);
                graph
                    .enqueue(
                        rp_k,
                        readout_project_rows::WorkflowArgs {
                            features: (&h_eff).into(),
                            weight: weight.tensor().into(),
                            token_rows: empty_tok_p.tensor().into(),
                            weight_scale: scale_p.tensor().into(),
                            softcap: softcap.into(),
                        },
                    )
                    .map_err(device_err)?
                    .value
            } else {
                let bound_t = self.lm_head.bound.as_ref().unwrap_or(&self.empty_f32);
                let bound_p = graph
                    .port(bound_t.element(), bound_t.extents())
                    .map_err(device_err)?;
                static_binds.push((bound_p.clone(), bound_t.clone()));
                graph
                    .enqueue(
                        proj_k_opt.unwrap(),
                        project_rows::WorkflowArgs {
                            source: (&h_eff).into(),
                            weight: weight.tensor().into(),
                            weight_scale: scale_p.tensor().into(),
                            weight_bound: bound_p.tensor().into(),
                        },
                    )
                    .map_err(device_err)?
                    .value
            };

            graph
                .enqueue(
                    top_k_k,
                    draft_top_k::WorkflowArgs {
                        logits: (&logits).into(),
                        candidates: candidates.tensor_mut().into(),
                        unary: unary.tensor_mut().into(),
                    },
                )
                .map_err(device_err)?;
            graph
                .enqueue(
                    conf_k,
                    token_confidence::WorkflowArgs {
                        logits: (&logits).into(),
                        rows: id_rows_p.tensor().into(),
                        selection: candidates.tensor().into(),
                        confidence: confidence.tensor_mut().into(),
                    },
                )
                .map_err(device_err)?;

            let plan = graph.seal().map_err(device_err)?;
            let bind_refs: Vec<(&NativePort, &Tensor)> =
                static_binds.iter().map(|(p, t)| (p, t)).collect();
            let bound = plan
                .bind_static(&self.charges, &bind_refs)
                .map_err(device_err)?;
            let family = NativeGraphFamily::with_launches(
                std::slice::from_ref(&plan),
                &[vec![plan.clone()]],
                1,
                &[],
            )
            .map_err(device_err)?;
            let slot = family.new_slot(&self.charges, &[]).map_err(device_err)?;
            let latent_buf =
                Tensor::upload(&self.charges, Element::bf16(), &[m, r]).map_err(device_err)?;
            let cand_buf =
                Tensor::zeros(&self.charges, Element::i32(), &[m, 2]).map_err(device_err)?;
            let unary_buf =
                Tensor::zeros(&self.charges, Element::f32(), &[m, 1]).map_err(device_err)?;
            let conf_buf =
                Tensor::zeros(&self.charges, Element::f32(), &[m]).map_err(device_err)?;
            cache.insert(
                m,
                BoundMarkovStepGraph {
                    plan,
                    bound,
                    _family: family,
                    slot,
                    latent_port,
                    hdraft_port,
                    candidates,
                    unary,
                    confidence,
                    latent_buf,
                    cand_buf,
                    unary_buf,
                    conf_buf,
                    _owned: owned,
                },
            );
        }

        let g = cache.get_mut(&m).unwrap();
        let latent_bytes = act_bytes(Element::bf16(), latents)?;
        g.latent_buf
            .write_from_host(&latent_bytes)
            .map_err(device_err)?;
        let mut bindings = g.bound.bindings();
        bindings
            .set(&self.charges, &g.latent_port, &g.latent_buf)
            .map_err(device_err)?;
        bindings
            .set(&self.charges, &g.hdraft_port, h_draft_step)
            .map_err(device_err)?;
        bindings
            .set(&self.charges, &g.candidates, &g.cand_buf)
            .map_err(device_err)?;
        bindings
            .set(&self.charges, &g.unary, &g.unary_buf)
            .map_err(device_err)?;
        bindings
            .set(&self.charges, &g.confidence, &g.conf_buf)
            .map_err(device_err)?;
        let mut active = g.slot.activate(&g.plan).map_err(device_err)?;
        let ready = active
            .attach(
                bindings,
                g.plan.new_outputs(&self.charges).map_err(device_err)?,
            )
            .map_err(device_err)?;
        let mut seq = self.charges.device().native_sequence(NativeQueue::Compute);
        let _ = ready.queue(&mut seq).map_err(device_err)?;
        seq.submit()
            .map_err(device_err)?
            .wait()
            .map_err(device_err)?;

        let cand_bytes = g.cand_buf.read_to_host().map_err(device_err)?;
        let conf_bytes = g.conf_buf.read_to_host().map_err(device_err)?;
        let mut out = Vec::with_capacity(slots);
        for s in 0..slots {
            let tok = i32::from_le_bytes(cand_bytes[s * 8..s * 8 + 4].try_into().unwrap())
                .max(0) as usize;
            let prob = f32::from_le_bytes(conf_bytes[s * 4..s * 4 + 4].try_into().unwrap())
                .clamp(0.0, 1.0);
            out.push((tok, prob));
        }
        Ok(Some(out))
    }

    /// Execute the entire `steps`-long RNN DSpark Markov chain in ONE single sealed
    /// GPU `NativeGraph` submission (zero CPU↔GPU round-trips between steps).
    /// Returns `Some((tokens, confidences))` in step-major order `[steps * slots]`.
    #[allow(clippy::too_many_arguments)]
    fn run_markov_chain_gpu(
        &self,
        f_rnn_rows: &[f32],
        h_draft_steps: &[f32],
        anchor_tokens: &[i32],
        slots: usize,
        steps: usize,
        softcap: f32,
    ) -> Result<Option<(Vec<i32>, Vec<f32>)>, SubmitError> {
        if slots == 0 || steps == 0 {
            return Ok(Some((Vec::new(), Vec::new())));
        }
        let (
            Some(frnn_k),
            Some(x_k),
            Some(he_k),
            Some(out_k),
            Some(emb_k),
            Some(gate_k),
            Some(top_k_k),
            Some(conf_k),
        ) = (
            &self.markov.frnn_linear,
            &self.markov.x_linear,
            &self.markov.he_linear,
            &self.markov.out_linear,
            &self.markov.markov_emb_kernel,
            &self.markov.gate_kernel,
            &self.top_k_kernel,
            &self.conf_kernel,
        )
        else {
            return Ok(None);
        };
        let m = slots as u64;
        let st = steps as u64;
        let total = m * st;
        let h = self.prepared.hidden;
        let r = self.markov.rank as u64;
        let r3 = 3 * r;
        let [v, lm_k] = match self.lm_head.tensor.extents() {
            [v, k] if *k == h => [*v, *k],
            _ => return Ok(None),
        };
        let [emb_v, emb_r] = match self.markov.markov_emb_gpu.extents() {
            [ev, er] if *er == r => [*ev, *er],
            _ => return Ok(None),
        };
        let use_readout = self.readout_project.is_some();
        if !use_readout && softcap > 0.0 {
            return Ok(None);
        }
        let proj_k_opt = if !use_readout {
            let key: ProjectKey = (
                Element::bf16(),
                self.lm_head.element(),
                Element::f32(),
                lm_k,
                v,
                self.lm_head.scale_extent(),
                self.lm_head.bound_extent(),
            );
            match self.projects.get(&key) {
                Some(pk) => Some(pk),
                None => return Ok(None),
            }
        } else {
            None
        };

        let key = (m, st);
        let mut cache = self
            .markov_chain_graphs
            .lock()
            .expect("markov_chain_graphs lock");
        if !cache.contains_key(&key) {
            let device = self.charges.device();
            let mut graph = device.native_graph();

            // Dynamic run inputs and outputs.
            let frnn_port = graph.port(Element::bf16(), &[m, h]).map_err(device_err)?;
            let hdraft_act_port = graph
                .port(Element::bf16(), &[total, h])
                .map_err(device_err)?;
            let hdraft_f32_port = graph
                .port(Element::f32(), &[st, m, h])
                .map_err(device_err)?;
            let anchor_port = graph.port(Element::i32(), &[m, 2]).map_err(device_err)?;
            let tokens_out_port = graph.port(Element::i32(), &[total, 2]).map_err(device_err)?;
            let unary_out_port = graph.port(Element::f32(), &[total, 1]).map_err(device_err)?;
            let conf_out_port = graph.port(Element::f32(), &[total]).map_err(device_err)?;

            // Static weight ports.
            let w_frnn_p = graph.port(Element::bf16(), &[r3, h]).map_err(device_err)?;
            let rnn_b_p = graph.port(Element::f32(), &[1, r3]).map_err(device_err)?;
            let w_x_p = graph.port(Element::bf16(), &[r3, h]).map_err(device_err)?;
            // `w_h_p` is only referenced for `k >= 1` (`h_prev` is zero at `k == 0`);
            // omit the port when `st == 1` so `graph.seal()` never sees an unused port.
            let w_h_p = if st > 1 {
                Some(graph.port(Element::bf16(), &[r3, r]).map_err(device_err)?)
            } else {
                None
            };
            let w_e_p = graph.port(Element::bf16(), &[r3, r]).map_err(device_err)?;
            let out_w_p = graph.port(Element::bf16(), &[h, r]).map_err(device_err)?;
            let out_b_p = graph.port(Element::f32(), &[1, h]).map_err(device_err)?;
            let markov_emb_p = graph
                .port(self.markov.markov_emb_gpu.element(), &[emb_v, emb_r])
                .map_err(device_err)?;
            let lm_w_p = graph
                .port(self.lm_head.element(), &[v, lm_k])
                .map_err(device_err)?;
            let scale_t = self.lm_head.scale.as_ref().unwrap_or(&self.empty_f32);
            let lm_scale_p = graph
                .port(scale_t.element(), scale_t.extents())
                .map_err(device_err)?;

            // Static constant / zero / empty tensors.
            let empty_dir_t = i32_tensor(&self.charges, &[0, 1], &[])?;
            let id_rows_t = i32_tensor(&self.charges, &[m], &(0..m as i32).collect::<Vec<_>>())?;
            let zero_h_init_t =
                Tensor::zeros(&self.charges, Element::f32(), &[m, r]).map_err(device_err)?;
            let zero_he_h_t =
                Tensor::zeros(&self.charges, Element::f32(), &[m, r3]).map_err(device_err)?;
            let empty_bias_r3_t = f32_tensor(&self.charges, &[0, r3], &[])?;
            let empty_res_s_r3_t = f32_tensor(&self.charges, &[0, m, r3], &[])?;
            let empty_gate_s_r3_t =
                Tensor::from_host(&self.charges, Element::bf16(), &[0, m, r3], &[])
                    .map_err(device_err)?;
            let empty_res_total_r3_t = f32_tensor(&self.charges, &[0, total, r3], &[])?;
            let empty_gate_total_r3_t =
                Tensor::from_host(&self.charges, Element::bf16(), &[0, total, r3], &[])
                    .map_err(device_err)?;
            let empty_gate_s_h_t =
                Tensor::from_host(&self.charges, Element::bf16(), &[0, m, h], &[])
                    .map_err(device_err)?;

            let empty_f32_p = graph.port(Element::f32(), &[0]).map_err(device_err)?;
            let empty_dir_p = graph.port(Element::i32(), &[0, 1]).map_err(device_err)?;
            let id_rows_p = graph.port(Element::i32(), &[m]).map_err(device_err)?;
            let zero_h_init_p = graph.port(Element::f32(), &[m, r]).map_err(device_err)?;
            let zero_he_h_p = graph.port(Element::f32(), &[m, r3]).map_err(device_err)?;
            let empty_bias_r3_p = graph.port(Element::f32(), &[0, r3]).map_err(device_err)?;
            let empty_res_s_r3_p = graph
                .port(Element::f32(), &[0, m, r3])
                .map_err(device_err)?;
            let empty_gate_s_r3_p = graph
                .port(Element::bf16(), &[0, m, r3])
                .map_err(device_err)?;
            let empty_res_total_r3_p = graph
                .port(Element::f32(), &[0, total, r3])
                .map_err(device_err)?;
            let empty_gate_total_r3_p = graph
                .port(Element::bf16(), &[0, total, r3])
                .map_err(device_err)?;
            let empty_gate_s_h_p = graph
                .port(Element::bf16(), &[0, m, h])
                .map_err(device_err)?;

            let mut owned = vec![
                empty_dir_t.clone(),
                id_rows_t.clone(),
                zero_h_init_t.clone(),
                zero_he_h_t.clone(),
                empty_bias_r3_t.clone(),
                empty_res_s_r3_t.clone(),
                empty_gate_s_r3_t.clone(),
                empty_res_total_r3_t.clone(),
                empty_gate_total_r3_t.clone(),
                empty_gate_s_h_t.clone(),
            ];
            let mut static_binds: Vec<(NativePort, Tensor)> = vec![
                (w_frnn_p.clone(), self.markov.w_frnn_gpu.clone()),
                (rnn_b_p.clone(), self.markov.rnn_b_gpu.clone()),
                (w_x_p.clone(), self.markov.w_x_gpu.clone()),
                (w_e_p.clone(), self.markov.w_e_gpu.clone()),
                (out_w_p.clone(), self.markov.hidden_out_w_gpu.clone()),
                (out_b_p.clone(), self.markov.hidden_out_b_gpu.clone()),
                (markov_emb_p.clone(), self.markov.markov_emb_gpu.clone()),
                (lm_w_p.clone(), self.lm_head.tensor.clone()),
                (lm_scale_p.clone(), scale_t.clone()),
                (empty_f32_p.clone(), self.empty_f32.clone()),
                (empty_dir_p.clone(), empty_dir_t),
                (id_rows_p.clone(), id_rows_t),
                (zero_h_init_p.clone(), zero_h_init_t),
                (zero_he_h_p.clone(), zero_he_h_t),
                (empty_bias_r3_p.clone(), empty_bias_r3_t),
                (empty_res_s_r3_p.clone(), empty_res_s_r3_t),
                (empty_gate_s_r3_p.clone(), empty_gate_s_r3_t),
                (empty_res_total_r3_p.clone(), empty_res_total_r3_t),
                (empty_gate_total_r3_p.clone(), empty_gate_total_r3_t),
                (empty_gate_s_h_p.clone(), empty_gate_s_h_t),
            ];
            if let Some(w_h_p) = &w_h_p {
                static_binds.push((w_h_p.clone(), self.markov.w_h_gpu.clone()));
            }

            let (empty_tok_p_opt, lm_bound_p_opt) = if self.readout_project.is_some() {
                let active_s = self.prepared.active_vocab.min(v);
                let empty_tok_t =
                    Tensor::from_host(&self.charges, Element::i32(), &[m, 0, active_s], &[])
                        .map_err(device_err)?;
                let empty_tok_p = graph
                    .port(Element::i32(), &[m, 0, active_s])
                    .map_err(device_err)?;
                static_binds.push((empty_tok_p.clone(), empty_tok_t.clone()));
                owned.push(empty_tok_t);
                (Some(empty_tok_p), None)
            } else {
                let bound_t = self.lm_head.bound.as_ref().unwrap_or(&self.empty_f32);
                let bound_p = graph
                    .port(bound_t.element(), bound_t.extents())
                    .map_err(device_err)?;
                static_binds.push((bound_p.clone(), bound_t.clone()));
                (None, Some(bound_p))
            };

            // Pre-loop: project `base_frnn = W_frnn @ f_rnn + rnn_b` and
            // `base_x_all = W_x @ h_draft_steps` once for all `steps`.
            let base_frnn = graph
                .enqueue(
                    frnn_k,
                    vision_linear::WorkflowArgs {
                        x: frnn_port.tensor().into(),
                        weight: w_frnn_p.tensor().into(),
                        bias: rnn_b_p.tensor().into(),
                        residual: empty_res_s_r3_p.tensor().into(),
                        gate: empty_gate_s_r3_p.tensor().into(),
                        minimum: empty_f32_p.tensor().into(),
                        maximum: empty_f32_p.tensor().into(),
                        activation: 0i32.into(),
                    },
                )
                .map_err(device_err)?
                .value;
            let base_x_all = graph
                .enqueue(
                    x_k,
                    vision_linear::WorkflowArgs {
                        x: hdraft_act_port.tensor().into(),
                        weight: w_x_p.tensor().into(),
                        bias: empty_bias_r3_p.tensor().into(),
                        residual: empty_res_total_r3_p.tensor().into(),
                        gate: empty_gate_total_r3_p.tensor().into(),
                        minimum: empty_f32_p.tensor().into(),
                        maximum: empty_f32_p.tensor().into(),
                        activation: 0i32.into(),
                    },
                )
                .map_err(device_err)?
                .value;

            // Unrolled Markov chain `k = 0..steps`, completely device-resident.
            let mut h_prev = zero_h_init_p.tensor().clone();
            let mut h_prev_a_opt: Option<seismic::WorkflowTensor> = None;
            for k in 0..st {
                let prev_tok = if k == 0 {
                    anchor_port.tensor().slice_leading(0, m)
                } else {
                    tokens_out_port
                        .tensor()
                        .slice_leading((k - 1) * m, k * m)
                };
                let prev_emb = graph
                    .enqueue(
                        emb_k,
                        embedding_rows::WorkflowArgs {
                            table: markov_emb_p.tensor().into(),
                            tokens: (&prev_tok).into(),
                            directory: empty_dir_p.tensor().into(),
                            scale: 1.0f32.into(),
                            normalize: 0i32.into(),
                            epsilon: 1e-6f32.into(),
                        },
                    )
                    .map_err(device_err)?
                    .r0;

                let he_e = graph
                    .enqueue(
                        he_k,
                        vision_linear::WorkflowArgs {
                            x: (&prev_emb).into(),
                            weight: w_e_p.tensor().into(),
                            bias: empty_bias_r3_p.tensor().into(),
                            residual: empty_res_s_r3_p.tensor().into(),
                            gate: empty_gate_s_r3_p.tensor().into(),
                            minimum: empty_f32_p.tensor().into(),
                            maximum: empty_f32_p.tensor().into(),
                            activation: 0i32.into(),
                        },
                    )
                    .map_err(device_err)?
                    .value;

                // At step 0, `h_prev == 0` so `W_h @ h_prev == 0` identically.
                let he_h = if let Some(h_prev_a) = &h_prev_a_opt {
                    graph
                        .enqueue(
                            he_k,
                            vision_linear::WorkflowArgs {
                                x: h_prev_a.into(),
                                weight: w_h_p
                                .as_ref()
                                .expect("w_h_p created for st > 1")
                                .tensor()
                                .into(),
                                bias: empty_bias_r3_p.tensor().into(),
                                residual: empty_res_s_r3_p.tensor().into(),
                                gate: empty_gate_s_r3_p.tensor().into(),
                                minimum: empty_f32_p.tensor().into(),
                                maximum: empty_f32_p.tensor().into(),
                                activation: 0i32.into(),
                            },
                        )
                        .map_err(device_err)?
                        .value
                } else {
                    zero_he_h_p.tensor().clone()
                };

                let base_x_k = base_x_all.slice_leading(k * m, (k + 1) * m);
                let gated = graph
                    .enqueue(
                        gate_k,
                        rnn_markov_gate::WorkflowArgs {
                            h_prev: (&h_prev).into(),
                            base_frnn: (&base_frnn).into(),
                            base_x: (&base_x_k).into(),
                            he_h: (&he_h).into(),
                            he_e: (&he_e).into(),
                        },
                    )
                    .map_err(device_err)?;
                h_prev = gated.r0;
                h_prev_a_opt = Some(gated.r1);
                let latent = gated.r2;

                let hdraft_k = hdraft_f32_port.tensor().slice_leading(k, k + 1);
                let h_eff = graph
                    .enqueue(
                        out_k,
                        vision_linear::WorkflowArgs {
                            x: (&latent).into(),
                            weight: out_w_p.tensor().into(),
                            bias: out_b_p.tensor().into(),
                            residual: (&hdraft_k).into(),
                            gate: empty_gate_s_h_p.tensor().into(),
                            minimum: empty_f32_p.tensor().into(),
                            maximum: empty_f32_p.tensor().into(),
                            activation: 0i32.into(),
                        },
                    )
                    .map_err(device_err)?
                    .value;

                let logits = if let Some(rp_k) = &self.readout_project {
                    graph
                        .enqueue(
                            rp_k,
                            readout_project_rows::WorkflowArgs {
                                features: (&h_eff).into(),
                                weight: lm_w_p.tensor().into(),
                                token_rows: empty_tok_p_opt.as_ref().unwrap().tensor().into(),
                                weight_scale: lm_scale_p.tensor().into(),
                                softcap: softcap.into(),
                            },
                        )
                        .map_err(device_err)?
                        .value
                } else {
                    graph
                        .enqueue(
                            proj_k_opt.unwrap(),
                            project_rows::WorkflowArgs {
                                source: (&h_eff).into(),
                                weight: lm_w_p.tensor().into(),
                                weight_scale: lm_scale_p.tensor().into(),
                                weight_bound: lm_bound_p_opt.as_ref().unwrap().tensor().into(),
                            },
                        )
                        .map_err(device_err)?
                        .value
                };

                let mut step_toks = tokens_out_port.tensor().slice_leading(k * m, (k + 1) * m);
                let mut step_unary = unary_out_port.tensor().slice_leading(k * m, (k + 1) * m);
                graph
                    .enqueue(
                        top_k_k,
                        draft_top_k::WorkflowArgs {
                            logits: (&logits).into(),
                            candidates: (&mut step_toks).into(),
                            unary: (&mut step_unary).into(),
                        },
                    )
                    .map_err(device_err)?;

                let mut step_conf = conf_out_port.tensor().slice_leading(k * m, (k + 1) * m);
                graph
                    .enqueue(
                        conf_k,
                        token_confidence::WorkflowArgs {
                            logits: (&logits).into(),
                            rows: id_rows_p.tensor().into(),
                            selection: (&step_toks).into(),
                            confidence: (&mut step_conf).into(),
                        },
                    )
                    .map_err(device_err)?;
            }

            let plan = graph.seal().map_err(device_err)?;
            let bind_refs: Vec<(&NativePort, &Tensor)> =
                static_binds.iter().map(|(p, t)| (p, t)).collect();
            let bound = plan
                .bind_static(&self.charges, &bind_refs)
                .map_err(device_err)?;
            let family = NativeGraphFamily::with_launches(
                std::slice::from_ref(&plan),
                &[vec![plan.clone()]],
                1,
                &[],
            )
            .map_err(device_err)?;
            let slot = family.new_slot(&self.charges, &[]).map_err(device_err)?;
            let frnn_buf =
                Tensor::upload(&self.charges, Element::bf16(), &[m, h]).map_err(device_err)?;
            let hdraft_act_buf =
                Tensor::upload(&self.charges, Element::bf16(), &[total, h]).map_err(device_err)?;
            let hdraft_f32_buf =
                Tensor::upload(&self.charges, Element::f32(), &[st, m, h]).map_err(device_err)?;
            let anchor_buf =
                Tensor::upload(&self.charges, Element::i32(), &[m, 2]).map_err(device_err)?;
            let tokens_out_buf =
                Tensor::zeros(&self.charges, Element::i32(), &[total, 2]).map_err(device_err)?;
            let unary_out_buf =
                Tensor::zeros(&self.charges, Element::f32(), &[total, 1]).map_err(device_err)?;
            let conf_out_buf =
                Tensor::zeros(&self.charges, Element::f32(), &[total]).map_err(device_err)?;

            cache.insert(
                key,
                BoundMarkovChainGraph {
                    plan,
                    bound,
                    _family: family,
                    slot,
                    frnn_port,
                    hdraft_act_port,
                    hdraft_f32_port,
                    anchor_port,
                    tokens_out_port,
                    unary_out_port,
                    conf_out_port,
                    frnn_buf,
                    hdraft_act_buf,
                    hdraft_f32_buf,
                    anchor_buf,
                    tokens_out_buf,
                    unary_out_buf,
                    conf_out_buf,
                    _owned: owned,
                },
            );
        }

        let g = cache.get_mut(&key).unwrap();
        let frnn_bytes = act_bytes(Element::bf16(), f_rnn_rows)?;
        let hdraft_act_bytes = act_bytes(Element::bf16(), h_draft_steps)?;
        let hdraft_f32_bytes: Vec<u8> = h_draft_steps
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let max_tok = self.prepared.active_vocab.saturating_sub(1) as i32;
        let anchor_bytes: Vec<u8> = anchor_tokens
            .iter()
            .flat_map(|&t| [t.clamp(0, max_tok), 0i32])
            .flat_map(|v| v.to_le_bytes())
            .collect();

        g.frnn_buf
            .write_from_host(&frnn_bytes)
            .map_err(device_err)?;
        g.hdraft_act_buf
            .write_from_host(&hdraft_act_bytes)
            .map_err(device_err)?;
        g.hdraft_f32_buf
            .write_from_host(&hdraft_f32_bytes)
            .map_err(device_err)?;
        g.anchor_buf
            .write_from_host(&anchor_bytes)
            .map_err(device_err)?;

        let mut bindings = g.bound.bindings();
        bindings
            .set(&self.charges, &g.frnn_port, &g.frnn_buf)
            .map_err(device_err)?;
        bindings
            .set(&self.charges, &g.hdraft_act_port, &g.hdraft_act_buf)
            .map_err(device_err)?;
        bindings
            .set(&self.charges, &g.hdraft_f32_port, &g.hdraft_f32_buf)
            .map_err(device_err)?;
        bindings
            .set(&self.charges, &g.anchor_port, &g.anchor_buf)
            .map_err(device_err)?;
        bindings
            .set(&self.charges, &g.tokens_out_port, &g.tokens_out_buf)
            .map_err(device_err)?;
        bindings
            .set(&self.charges, &g.unary_out_port, &g.unary_out_buf)
            .map_err(device_err)?;
        bindings
            .set(&self.charges, &g.conf_out_port, &g.conf_out_buf)
            .map_err(device_err)?;

        let mut active = g.slot.activate(&g.plan).map_err(device_err)?;
        let ready = active
            .attach(
                bindings,
                g.plan.new_outputs(&self.charges).map_err(device_err)?,
            )
            .map_err(device_err)?;
        let mut seq = self.charges.device().native_sequence(NativeQueue::Compute);
        let _ = ready.queue(&mut seq).map_err(device_err)?;
        seq.submit()
            .map_err(device_err)?
            .wait()
            .map_err(device_err)?;

        let tok_bytes = g.tokens_out_buf.read_to_host().map_err(device_err)?;
        let conf_bytes = g.conf_out_buf.read_to_host().map_err(device_err)?;
        let total_usize = steps * slots;
        let mut tokens = Vec::with_capacity(total_usize);
        let mut confs = Vec::with_capacity(total_usize);
        for i in 0..total_usize {
            let tok = i32::from_le_bytes(tok_bytes[i * 8..i * 8 + 4].try_into().unwrap()).max(0);
            let prob = f32::from_le_bytes(conf_bytes[i * 4..i * 4 + 4].try_into().unwrap())
                .clamp(0.0, 1.0);
            tokens.push(tok);
            confs.push(prob);
        }
        Ok(Some((tokens, confs)))
    }

    fn gather_embeddings_gpu(
        &self,
        tokens: &[i32],
        embedding: &crate::device_table::ResidentTable,
    ) -> Result<Vec<f32>, SubmitError> {
        let m = tokens.len();
        if m == 0 {
            return Ok(Vec::new());
        }
        let rows = embedding.indices(tokens).map_err(invalid)?;
        let tok_pairs: Vec<i32> = rows.iter().flat_map(|&r| [r, 0]).collect();
        let tok_t = i32_tensor(&self.charges, &[m as u64, 2], &tok_pairs)?;
        let dir_t = i32_tensor(&self.charges, &[0, 1], &[])?;
        let out = self
            .emb_kernel
            .call(
                &self.charges,
                embedding_rows::Args {
                    table: &self.emb_table,
                    tokens: &tok_t,
                    directory: &dir_t,
                    scale: self.prepared.draft.embedding_scale as f32,
                    normalize: 0,
                    epsilon: 1e-6,
                },
            )
            .map_err(|e| device_err(format!("rnn emb_kernel(m={m}): {e}")))?
            .r0;
        read_f32(&out)
    }

    fn weight(&self, scope: WeightScope, kind: WeightKind) -> Result<&BoundWeight, SubmitError> {
        self.weights
            .get(&WeightRole { scope, kind })
            .ok_or_else(|| invalid(format!("missing draft weight {scope:?}::{kind:?}")))
    }

    /// Extract per-layer GPU KV slabs `(domain, key_slab, val_slab, history_rows, slab_rows)`
    /// from `HeadState::Owned` when dense BF16/F16 slabs are available.
    fn extract_head_slabs(
        &self,
        core: &HeadLaunchCore,
    ) -> Option<Vec<(usize, Tensor, Tensor, u64, u32)>> {
        let num_layers = self.prepared.draft.blocks.len();
        if self.prepared.layer_history.len() < num_layers {
            return None;
        }
        let crate::lanes::HeadState::Owned { advances, .. } = core.state() else {
            return None;
        };
        let planes = advances.first()?.bindings().history;
        let mut out = Vec::with_capacity(num_layers);
        for l in 0..num_layers {
            let layer_ref = LayerRef::Head(l as u32);
            let mut lp = planes.iter().filter(|p| p.layer == layer_ref);
            let k_plane = lp.next()?;
            let v_plane = lp.next()?;
            let elem = k_plane.buffer.element();
            if (elem != Element::bf16() && elem != Element::f16())
                || v_plane.buffer.element() != elem
            {
                return None;
            }
            let (hist_rows, slab_rows) = self.prepared.layer_history[l];
            out.push((
                k_plane.domain.0,
                k_plane.buffer.clone(),
                v_plane.buffer.clone(),
                hist_rows,
                slab_rows,
            ));
        }
        Some(out)
    }

    /// Run the DSpark entry/injection pass across all `L` draft layers in a single
    /// sealed GPU `NativeGraph` submission, writing K/V directly into the GPU KV slabs.
    fn run_inject_gpu(
        &self,
        fused: &[Vec<f32>],
        rows_in: usize,
        entry_coords: &[i32],
        dests_per_layer: &[Vec<i32>],
        slabs: &[(usize, Tensor, Tensor, u64, u32)],
    ) -> Result<(), SubmitError> {
        let draft = &self.prepared.draft;
        let num_layers = draft.blocks.len();
        let h = self.prepared.hidden;
        let m = rows_in as u64;
        let kv_elem = slabs[0].1.element();
        let key = (m, kv_elem);

        let mut map = self.inject_graphs.lock().expect("inject_graphs lock");
        if !map.contains_key(&key) {
            let mut graph = self.charges.device().native_graph();
            let mut static_binds: Vec<(NativePort, Tensor)> = Vec::new();
            let mut owned: Vec<Tensor> = Vec::new();

            let empty_f32_p = graph.port(Element::f32(), &[0]).map_err(device_err)?;
            static_binds.push((empty_f32_p.clone(), self.empty_f32.clone()));

            let coords_p = graph.port(Element::i32(), &[m, 4]).map_err(device_err)?;
            let mut fused_ports = Vec::with_capacity(num_layers);
            let mut dest_ports = Vec::with_capacity(num_layers);
            let mut hk_ports = Vec::with_capacity(num_layers);
            let mut hv_ports = Vec::with_capacity(num_layers);

            for (l, blk) in draft.blocks.iter().enumerate() {
                let Operator::Attention(attn) = &blk.sublayers[0].op else {
                    return Err(invalid("draft block sublayer 0 must be Attention"));
                };
                let s0 = WeightScope::DraftSublayer(SublayerIndex {
                    block: l as u32,
                    sublayer: 0,
                });
                let (wk, wv, kn_vec) = if draft.context_projections.is_some() {
                    (
                        self.weight(s0, WeightKind::DraftContextKey)?,
                        self.weight(s0, WeightKind::DraftContextValue)?,
                        self.layer_norms[l]
                            .k_norm_ctx
                            .as_deref()
                            .ok_or_else(|| invalid("missing DraftContextKeyNorm"))?,
                    )
                } else {
                    (
                        self.weight(s0, WeightKind::Key)?,
                        self.weight(s0, WeightKind::Value)?,
                        self.layer_norms[l].k_norm.as_slice(),
                    )
                };

                let kv_heads = attn.kv_heads;
                let head_dim = attn.width;
                let kv_dim = kv_heads * head_dim;
                let (ra, rt, p, s, r_comp, r_freq, r_amp) = rotary_tables(&attn.rotary, head_dim);
                let k_eps = match &attn.query_norm {
                    HeadNorm::Rms(n) => n.epsilon as f32,
                    _ => 1e-6,
                };
                let (_, _, _, hist_rows, slab_rows) = slabs[l];

                let fused_p = graph.port(kv_elem, &[m, h]).map_err(device_err)?;
                let dest_p = graph.port(Element::i32(), &[m]).map_err(device_err)?;
                let mut hk_p = graph
                    .port(kv_elem, &[hist_rows, kv_heads, head_dim])
                    .map_err(device_err)?;
                let mut hv_p = graph
                    .port(kv_elem, &[hist_rows, kv_heads, head_dim])
                    .map_err(device_err)?;
                fused_ports.push(fused_p.clone());
                dest_ports.push(dest_p.clone());
                hk_ports.push(hk_p.clone());
                hv_ports.push(hv_p.clone());

                let wk_p = graph
                    .port(wk.element(), wk.tensor.extents())
                    .map_err(device_err)?;
                static_binds.push((wk_p.clone(), wk.tensor.clone()));
                let wv_p = graph
                    .port(wv.element(), wv.tensor.extents())
                    .map_err(device_err)?;
                static_binds.push((wv_p.clone(), wv.tensor.clone()));

                let wk_s_p = if let Some(sc) = &wk.scale {
                    let p_s = graph.port(Element::f32(), sc.extents()).map_err(device_err)?;
                    static_binds.push((p_s.clone(), sc.clone()));
                    p_s
                } else {
                    empty_f32_p.clone()
                };
                let wk_b_p = if let Some(bd) = &wk.bound {
                    let p_b = graph.port(Element::f32(), bd.extents()).map_err(device_err)?;
                    static_binds.push((p_b.clone(), bd.clone()));
                    p_b
                } else {
                    empty_f32_p.clone()
                };
                let wv_s_p = if let Some(sc) = &wv.scale {
                    let p_s = graph.port(Element::f32(), sc.extents()).map_err(device_err)?;
                    static_binds.push((p_s.clone(), sc.clone()));
                    p_s
                } else {
                    empty_f32_p.clone()
                };
                let wv_b_p = if let Some(bd) = &wv.bound {
                    let p_b = graph.port(Element::f32(), bd.extents()).map_err(device_err)?;
                    static_binds.push((p_b.clone(), bd.clone()));
                    p_b
                } else {
                    empty_f32_p.clone()
                };

                let pk_k = prepare_kernel(
                    &self.prepared.device,
                    project_rows::native_entry_with(project_rows::Elements {
                        A: kv_elem,
                        W: wk.element(),
                        Y: kv_elem,
                    }),
                    &[
                        ("G", 1),
                        ("K", h),
                        ("N", kv_dim),
                        ("WS", wk.scale_extent()),
                        ("WB", wk.bound_extent()),
                    ],
                )
                .map_err(device_err)?;
                let pv_k = prepare_kernel(
                    &self.prepared.device,
                    project_rows::native_entry_with(project_rows::Elements {
                        A: kv_elem,
                        W: wv.element(),
                        Y: kv_elem,
                    }),
                    &[
                        ("G", 1),
                        ("K", h),
                        ("N", kv_dim),
                        ("WS", wv.scale_extent()),
                        ("WB", wv.bound_extent()),
                    ],
                )
                .map_err(device_err)?;

                let k_out = graph
                    .enqueue(
                        &pk_k,
                        project_rows::WorkflowArgs {
                            source: fused_p.tensor().into(),
                            weight: wk_p.tensor().into(),
                            weight_scale: wk_s_p.tensor().into(),
                            weight_bound: wk_b_p.tensor().into(),
                        },
                    )
                    .map_err(device_err)?
                    .value
                    .reshape(&[1, m, kv_dim]);

                let v_out = graph
                    .enqueue(
                        &pv_k,
                        project_rows::WorkflowArgs {
                            source: fused_p.tensor().into(),
                            weight: wv_p.tensor().into(),
                            weight_scale: wv_s_p.tensor().into(),
                            weight_bound: wv_b_p.tensor().into(),
                        },
                    )
                    .map_err(device_err)?
                    .value
                    .reshape(&[1, m, kv_dim]);

                let kn_t = f32_tensor(&self.charges, &[1, head_dim], kn_vec)?;
                let vn_t = f32_tensor(&self.charges, &[0, head_dim], &[])?;
                let rc_t = i32_tensor(&self.charges, &[p], &r_comp)?;
                let rf_t = f32_tensor(&self.charges, &[p], &r_freq)?;
                let ra_t = f32_tensor(&self.charges, &[p], &r_amp)?;

                let kn_p = graph.port(Element::f32(), &[1, head_dim]).map_err(device_err)?;
                let vn_p = graph.port(Element::f32(), &[0, head_dim]).map_err(device_err)?;
                let rc_p = graph.port(Element::i32(), &[p]).map_err(device_err)?;
                let rf_p = graph.port(Element::f32(), &[p]).map_err(device_err)?;
                let ra_p = graph.port(Element::f32(), &[p]).map_err(device_err)?;
                static_binds.push((kn_p.clone(), kn_t.clone()));
                static_binds.push((vn_p.clone(), vn_t.clone()));
                static_binds.push((rc_p.clone(), rc_t.clone()));
                static_binds.push((rf_p.clone(), rf_t.clone()));
                static_binds.push((ra_p.clone(), ra_t.clone()));
                owned.extend([kn_t, vn_t, rc_t, rf_t, ra_t]);

                let append_k = prepare_kernel(
                    &self.prepared.device,
                    attention_append_dense::native_entry_with(attention_append_dense::Elements {
                        A: kv_elem,
                    }),
                    &[
                        ("KV", kv_heads),
                        ("P", p),
                        ("S", s),
                        ("F", 1),
                        ("N", 1),
                        ("NV", 0),
                        ("RA", ra),
                        ("RT", rt),
                        ("VK", 0),
                        ("KR", 0),
                    ],
                )
                .map_err(device_err)?;

                let append_form = dest_p
                    .tensor()
                    .slice_leading(0, 0)
                    .reshape(&[0, ra, rt, 0, 0]);

                graph
                    .enqueue(
                        &append_k,
                        attention_append_dense::WorkflowArgs {
                            key: (&k_out).into(),
                            value: (&v_out).into(),
                            key_norm: kn_p.tensor().into(),
                            value_norm: vn_p.tensor().into(),
                            rotary_components: rc_p.tensor().into(),
                            rotary_frequencies: rf_p.tensor().into(),
                            rotary_amplitudes: ra_p.tensor().into(),
                            coordinates: coords_p.tensor().into(),
                            destinations: dest_p.tensor().into(),
                            history_key: hk_p.tensor_mut().into(),
                            history_value: hv_p.tensor_mut().into(),
                            form: (&append_form).into(),
                            epsilon: k_eps.into(),
                            slab_rows: slab_rows.into(),
                            key_block: 0i32.into(),
                            key_element: 0i32.into(),
                            key_scale: 0i32.into(),
                            key_amax_floor: 0.0f32.into(),
                        },
                    )
                    .map_err(device_err)?;
            }

            let plan = graph.seal().map_err(device_err)?;
            let bind_refs: Vec<(&NativePort, &Tensor)> =
                static_binds.iter().map(|(p, t)| (p, t)).collect();
            let bound = plan
                .bind_static(&self.charges, &bind_refs)
                .map_err(device_err)?;
            let family = NativeGraphFamily::with_launches(
                std::slice::from_ref(&plan),
                &[vec![plan.clone()]],
                1,
                &[],
            )
            .map_err(device_err)?;
            let slot = family.new_slot(&self.charges, &[]).map_err(device_err)?;
            let coords_buf =
                Tensor::upload(&self.charges, Element::i32(), &[m, 4]).map_err(device_err)?;
            let mut fused_bufs = Vec::with_capacity(num_layers);
            let mut dest_bufs = Vec::with_capacity(num_layers);
            for _ in 0..num_layers {
                fused_bufs.push(
                    Tensor::upload(&self.charges, kv_elem, &[m, h]).map_err(device_err)?,
                );
                dest_bufs.push(
                    Tensor::upload(&self.charges, Element::i32(), &[m]).map_err(device_err)?,
                );
            }
            map.insert(
                key,
                BoundInjectGraph {
                    plan,
                    bound,
                    _family: family,
                    slot,
                    fused_ports,
                    coords_port: coords_p,
                    dest_ports,
                    hk_ports,
                    hv_ports,
                    coords_buf,
                    fused_bufs,
                    dest_bufs,
                    _owned: owned,
                },
            );
        }

        let g = map.get_mut(&key).unwrap();
        let mut bindings = g.bound.bindings();
        let coords_bytes: Vec<u8> = entry_coords.iter().flat_map(|v| v.to_le_bytes()).collect();
        g.coords_buf
            .write_from_host(&coords_bytes)
            .map_err(device_err)?;
        bindings
            .set(&self.charges, &g.coords_port, &g.coords_buf)
            .map_err(device_err)?;
        for l in 0..num_layers {
            let f_bytes = act_bytes(kv_elem, &fused[l])?;
            g.fused_bufs[l]
                .write_from_host(&f_bytes)
                .map_err(device_err)?;
            let d_bytes: Vec<u8> = dests_per_layer[l]
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect();
            g.dest_bufs[l]
                .write_from_host(&d_bytes)
                .map_err(device_err)?;
            bindings
                .set(&self.charges, &g.fused_ports[l], &g.fused_bufs[l])
                .map_err(device_err)?;
            bindings
                .set(&self.charges, &g.dest_ports[l], &g.dest_bufs[l])
                .map_err(device_err)?;
            bindings
                .set(&self.charges, &g.hk_ports[l], &slabs[l].1)
                .map_err(device_err)?;
            bindings
                .set(&self.charges, &g.hv_ports[l], &slabs[l].2)
                .map_err(device_err)?;
        }

        let mut active = g.slot.activate(&g.plan).map_err(device_err)?;
        let ready = active
            .attach(
                bindings,
                g.plan.new_outputs(&self.charges).map_err(device_err)?,
            )
            .map_err(device_err)?;
        let mut seq = self.charges.device().native_sequence(NativeQueue::Compute);
        let _ = ready.queue(&mut seq).map_err(device_err)?;
        seq.submit()
            .map_err(device_err)?
            .wait()
            .map_err(device_err)?;
        Ok(())
    }

    /// Run all `L` DSpark draft transformer layers + final `output_norm` in a
    /// single sealed GPU `NativeGraph` submission over the GPU KV slabs + fresh
    /// bidirectional block rows.
    #[allow(clippy::too_many_arguments)]
    fn run_block_gpu(
        &self,
        x_in: &[f32],
        block_rows: usize,
        coords: &[i32],
        fresh: &[i32],
        visible_per_layer: &[Vec<i32>],
        spans: u64,
        slabs: &[(usize, Tensor, Tensor, u64, u32)],
    ) -> Result<Vec<f32>, SubmitError> {
        let draft = &self.prepared.draft;
        let num_layers = draft.blocks.len();
        let h = self.prepared.hidden;
        let m = block_rows as u64;
        let kv_elem = slabs[0].1.element();
        let key = (m, spans, kv_elem);

        let mut map = self.block_graphs.lock().expect("block_graphs lock");
        if !map.contains_key(&key) {
            let mut graph = self.charges.device().native_graph();
            let mut static_binds: Vec<(NativePort, Tensor)> = Vec::new();
            let mut owned: Vec<Tensor> = Vec::new();

            let empty_f32_p = graph.port(Element::f32(), &[0]).map_err(device_err)?;
            static_binds.push((empty_f32_p.clone(), self.empty_f32.clone()));

            let empty_bias_t = f32_tensor(&self.charges, &[0, h], &[])?;
            let empty_order_t = i32_tensor(&self.charges, &[0, m], &[])?;
            let out_rows_vals: Vec<i32> = (0..block_rows as i32).collect();
            let out_rows_t = i32_tensor(&self.charges, &[m], &out_rows_vals)?;
            let neg_dest_t = i32_tensor(&self.charges, &[m], &vec![-1i32; block_rows])?;
            let empty_inject_t = f32_tensor(&self.charges, &[m, 0], &[])?;
            let empty_sel_rows_t = i32_tensor(&self.charges, &[m, 0], &[])?;

            let empty_bias_p = graph.port(Element::f32(), &[0, h]).map_err(device_err)?;
            let empty_order_p = graph.port(Element::i32(), &[0, m]).map_err(device_err)?;
            let out_rows_p = graph.port(Element::i32(), &[m]).map_err(device_err)?;
            let neg_dest_p = graph.port(Element::i32(), &[m]).map_err(device_err)?;
            let empty_inject_p = graph.port(Element::f32(), &[m, 0]).map_err(device_err)?;
            let empty_sel_rows_p = graph.port(Element::i32(), &[m, 0]).map_err(device_err)?;

            static_binds.push((empty_bias_p.clone(), empty_bias_t.clone()));
            static_binds.push((empty_order_p.clone(), empty_order_t.clone()));
            static_binds.push((out_rows_p.clone(), out_rows_t.clone()));
            static_binds.push((neg_dest_p.clone(), neg_dest_t.clone()));
            static_binds.push((empty_inject_p.clone(), empty_inject_t.clone()));
            static_binds.push((empty_sel_rows_p.clone(), empty_sel_rows_t.clone()));
            owned.extend([
                empty_bias_t,
                empty_order_t,
                out_rows_t,
                neg_dest_t,
                empty_inject_t,
                empty_sel_rows_t,
            ]);

            let x_in_p = graph.port(Element::f32(), &[m, h]).map_err(device_err)?;
            let coords_p = graph.port(Element::i32(), &[m, 4]).map_err(device_err)?;
            let fresh_p = graph.port(Element::i32(), &[m, 2]).map_err(device_err)?;
            let out_port = graph.port(Element::f32(), &[m, h]).map_err(device_err)?;

            let mut visible_ports = Vec::with_capacity(num_layers);
            let mut hk_ports = Vec::with_capacity(num_layers);
            let mut hv_ports = Vec::with_capacity(num_layers);

            let mut x = x_in_p.tensor().clone();
            let eps = draft.output_norm.epsilon as f32;

            for (l, blk) in draft.blocks.iter().enumerate() {
                let Operator::Attention(attn) = &blk.sublayers[0].op else {
                    return Err(invalid("draft block sublayer 0 must be Attention"));
                };
                let s0 = WeightScope::DraftSublayer(SublayerIndex {
                    block: l as u32,
                    sublayer: 0,
                });
                let s1 = WeightScope::DraftSublayer(SublayerIndex {
                    block: l as u32,
                    sublayer: 1,
                });

                let kv_heads = attn.kv_heads;
                let q_heads = attn.heads;
                let g_ratio = q_heads / kv_heads;
                let head_dim = attn.width;
                let q_dim = q_heads * head_dim;
                let kv_dim = kv_heads * head_dim;
                let (ra, rt, p, s, r_comp, r_freq, r_amp) = rotary_tables(&attn.rotary, head_dim);
                let (_, _, _, hist_rows, slab_rows) = slabs[l];

                let vis_p = graph
                    .port(Element::i32(), &[m, spans, 2])
                    .map_err(device_err)?;
                let mut hk_p = graph
                    .port(kv_elem, &[hist_rows, kv_heads, head_dim])
                    .map_err(device_err)?;
                let mut hv_p = graph
                    .port(kv_elem, &[hist_rows, kv_heads, head_dim])
                    .map_err(device_err)?;
                visible_ports.push(vis_p.clone());
                hk_ports.push(hk_p.clone());
                hv_ports.push(hv_p.clone());

                // 1. Attention RMSNorm: x [M, H] f32 -> normed [M, H] kv_elem
                let attn_norm_w = self.weight(s0, WeightKind::InputNorm)?;
                let an_p = graph
                    .port(attn_norm_w.element(), &[1, h])
                    .map_err(device_err)?;
                let an_reshaped = attn_norm_w
                    .tensor
                    .reshape(&[1, h])
                    .map_err(device_err)?;
                static_binds.push((an_p.clone(), an_reshaped.clone()));
                owned.push(an_reshaped);

                let vnorm_k = prepare_kernel(
                    &self.prepared.device,
                    vision_norm::native_entry_with(vision_norm::Elements {
                        NWE: attn_norm_w.element(),
                        NBE: Element::f32(),
                        Y: kv_elem,
                    }),
                    &[("G", 1), ("H", h), ("NW", 1), ("NB", 0), ("NO", 0)],
                )
                .map_err(device_err)?;
                let x3 = x.reshape(&[m, 1, h]);
                let normed = graph
                    .enqueue(
                        &vnorm_k,
                        vision_norm::WorkflowArgs {
                            source: (&x3).into(),
                            weight: an_p.tensor().into(),
                            bias: empty_bias_p.tensor().into(),
                            order: empty_order_p.tensor().into(),
                            epsilon: eps.into(),
                            centered: 0i32.into(),
                            interleave: 0i32.into(),
                            activation: 0i32.into(),
                        },
                    )
                    .map_err(device_err)?
                    .value;

                // 2. Q / K / V projections
                let mut bind_proj = |w: &BoundWeight, out_n: u64| -> Result<seismic::WorkflowTensor, SubmitError> {
                    let wp = graph
                        .port(w.element(), w.tensor.extents())
                        .map_err(device_err)?;
                    static_binds.push((wp.clone(), w.tensor.clone()));
                    let sp = if let Some(sc) = &w.scale {
                        let ps = graph.port(Element::f32(), sc.extents()).map_err(device_err)?;
                        static_binds.push((ps.clone(), sc.clone()));
                        ps
                    } else {
                        empty_f32_p.clone()
                    };
                    let bp = if let Some(bd) = &w.bound {
                        let pb = graph.port(Element::f32(), bd.extents()).map_err(device_err)?;
                        static_binds.push((pb.clone(), bd.clone()));
                        pb
                    } else {
                        empty_f32_p.clone()
                    };
                    let pk = prepare_kernel(
                        &self.prepared.device,
                        project_rows::native_entry_with(project_rows::Elements {
                            A: kv_elem,
                            W: w.element(),
                            Y: kv_elem,
                        }),
                        &[
                            ("G", 1),
                            ("K", h),
                            ("N", out_n),
                            ("WS", w.scale_extent()),
                            ("WB", w.bound_extent()),
                        ],
                    )
                    .map_err(device_err)?;
                    Ok(graph
                        .enqueue(
                            &pk,
                            project_rows::WorkflowArgs {
                                source: (&normed).into(),
                                weight: wp.tensor().into(),
                                weight_scale: sp.tensor().into(),
                                weight_bound: bp.tensor().into(),
                            },
                        )
                        .map_err(device_err)?
                        .value)
                };

                let q = bind_proj(self.weight(s0, WeightKind::Query)?, q_dim)?
                    .reshape(&[m, q_heads, head_dim]);
                let k = bind_proj(self.weight(s0, WeightKind::Key)?, kv_dim)?
                    .reshape(&[1, m, kv_dim]);
                let v = bind_proj(self.weight(s0, WeightKind::Value)?, kv_dim)?
                    .reshape(&[1, m, kv_dim]);

                // 3. GPU attention_decode (history KV slabs + fresh bidirectional block)
                let qn_t = f32_tensor(&self.charges, &[1, head_dim], &self.layer_norms[l].q_norm)?;
                let kn_t = f32_tensor(&self.charges, &[1, head_dim], &self.layer_norms[l].k_norm)?;
                let vn_t = f32_tensor(&self.charges, &[0, head_dim], &[])?;
                let sinks_t = f32_tensor(&self.charges, &[0, q_heads], &[])?;
                let gate_t = Tensor::from_host(&self.charges, kv_elem, &[m, q_heads, 0], &[])
                    .map_err(device_err)?;
                let sel_slab_t =
                    Tensor::from_host(&self.charges, kv_elem, &[0, kv_heads, head_dim], &[])
                        .map_err(device_err)?;
                let rc_t = i32_tensor(&self.charges, &[p], &r_comp)?;
                let rf_t = f32_tensor(&self.charges, &[p], &r_freq)?;
                let ra_t = f32_tensor(&self.charges, &[p], &r_amp)?;

                let qn_p = graph.port(Element::f32(), &[1, head_dim]).map_err(device_err)?;
                let kn_p = graph.port(Element::f32(), &[1, head_dim]).map_err(device_err)?;
                let vn_p = graph.port(Element::f32(), &[0, head_dim]).map_err(device_err)?;
                let sinks_p = graph.port(Element::f32(), &[0, q_heads]).map_err(device_err)?;
                let gate_p = graph.port(kv_elem, &[m, q_heads, 0]).map_err(device_err)?;
                let sel_slab_p = graph
                    .port(kv_elem, &[0, kv_heads, head_dim])
                    .map_err(device_err)?;
                let rc_p = graph.port(Element::i32(), &[p]).map_err(device_err)?;
                let rf_p = graph.port(Element::f32(), &[p]).map_err(device_err)?;
                let ra_p = graph.port(Element::f32(), &[p]).map_err(device_err)?;

                static_binds.push((qn_p.clone(), qn_t.clone()));
                static_binds.push((kn_p.clone(), kn_t.clone()));
                static_binds.push((vn_p.clone(), vn_t.clone()));
                static_binds.push((sinks_p.clone(), sinks_t.clone()));
                static_binds.push((gate_p.clone(), gate_t.clone()));
                static_binds.push((sel_slab_p.clone(), sel_slab_t.clone()));
                static_binds.push((rc_p.clone(), rc_t.clone()));
                static_binds.push((rf_p.clone(), rf_t.clone()));
                static_binds.push((ra_p.clone(), ra_t.clone()));
                owned.extend([qn_t, kn_t, vn_t, sinks_t, gate_t, sel_slab_t, rc_t, rf_t, ra_t]);

                let dec_k = prepare_kernel(
                    &self.prepared.device,
                    attention_decode::native_entry_with(attention_decode::Elements { A: kv_elem }),
                    &[
                        ("KV", kv_heads),
                        ("G", g_ratio),
                        ("P", p),
                        ("S", s),
                        ("I", 0),
                        ("U", 0),
                        ("F", 1),
                        ("NQ", 1),
                        ("N", 1),
                        ("NV", 0),
                        ("SK", 0),
                        ("RA", ra),
                        ("RT", rt),
                        ("RO", 0),
                        ("VK", 0),
                        ("KR", 0),
                        ("AB", 0),
                        ("KS", 0),
                    ],
                )
                .map_err(device_err)?;

                let form_table = neg_dest_p
                    .tensor()
                    .slice_leading(0, 0)
                    .reshape(&[0, ra, rt, 0, 0, 0, 0]);

                let attn_ctx = graph
                    .enqueue(
                        &dec_k,
                        attention_decode::WorkflowArgs {
                            query: (&q).into(),
                            gate: gate_p.tensor().into(),
                            key: (&k).into(),
                            value: (&v).into(),
                            query_norm: qn_p.tensor().into(),
                            key_norm: kn_p.tensor().into(),
                            value_norm: vn_p.tensor().into(),
                            sinks: sinks_p.tensor().into(),
                            rotary_components: rc_p.tensor().into(),
                            rotary_frequencies: rf_p.tensor().into(),
                            rotary_amplitudes: ra_p.tensor().into(),
                            coordinates: coords_p.tensor().into(),
                            visible: vis_p.tensor().into(),
                            fresh: fresh_p.tensor().into(),
                            destinations: neg_dest_p.tensor().into(),
                            selected_rows: empty_sel_rows_p.tensor().into(),
                            selected: sel_slab_p.tensor().into(),
                            history_key: hk_p.tensor_mut().into(),
                            history_value: hv_p.tensor_mut().into(),
                            form: (&form_table).into(),
                            epsilon: eps.into(),
                            scale: (attn.scale as f32).into(),
                            gate_function: 0i32.into(),
                            slab_rows: slab_rows.into(),
                            selected_slab_rows: 1u32.into(),
                            key_block: 0i32.into(),
                            key_element: 0i32.into(),
                            key_scale: 0i32.into(),
                            key_amax_floor: 0.0f32.into(),
                        },
                    )
                    .map_err(device_err)?
                    .value;

                // 4. Attention output projection + residual add
                let wo = self.weight(s0, WeightKind::AttentionOutput)?;
                let wo_p = graph
                    .port(wo.element(), wo.tensor.extents())
                    .map_err(device_err)?;
                static_binds.push((wo_p.clone(), wo.tensor.clone()));
                let wo_s_p = if let Some(sc) = &wo.scale {
                    let ps = graph.port(Element::f32(), sc.extents()).map_err(device_err)?;
                    static_binds.push((ps.clone(), sc.clone()));
                    ps
                } else {
                    empty_f32_p.clone()
                };
                let wo_b_p = if let Some(bd) = &wo.bound {
                    let pb = graph.port(Element::f32(), bd.extents()).map_err(device_err)?;
                    static_binds.push((pb.clone(), bd.clone()));
                    pb
                } else {
                    empty_f32_p.clone()
                };
                let ao_k = prepare_kernel(
                    &self.prepared.device,
                    attention_output::native_entry_with(attention_output::Elements {
                        A: kv_elem,
                        OW: wo.element(),
                    }),
                    &[
                        ("D", h),
                        ("SN", 1),
                        ("IN", 0),
                        ("Q", q_heads),
                        ("W", head_dim),
                        ("OS", wo.scale_extent()),
                        ("OB", wo.bound_extent()),
                    ],
                )
                .map_err(device_err)?;
                x = graph
                    .enqueue(
                        &ao_k,
                        attention_output::WorkflowArgs {
                            hidden: (&x).into(),
                            inject: empty_inject_p.tensor().into(),
                            gated: (&attn_ctx).into(),
                            output_weight: wo_p.tensor().into(),
                            output_scale: wo_s_p.tensor().into(),
                            output_bound: wo_b_p.tensor().into(),
                        },
                    )
                    .map_err(device_err)?
                    .value;

                // 5. Dense SwiGLU FFN (`dense_expand` + `dense_output`)
                let ffn_norm_w = self.weight(s1, WeightKind::InputNorm)?;
                let wg = self.weight(s1, WeightKind::DenseGate)?;
                let wu = self.weight(s1, WeightKind::DenseUp)?;
                let wd = self.weight(s1, WeightKind::DenseDown)?;
                let f_dim = wg.tensor.extents()[0];

                let fn_p = graph
                    .port(ffn_norm_w.element(), &[h])
                    .map_err(device_err)?;
                static_binds.push((fn_p.clone(), ffn_norm_w.tensor.clone()));
                let wg_p = graph
                    .port(wg.element(), wg.tensor.extents())
                    .map_err(device_err)?;
                static_binds.push((wg_p.clone(), wg.tensor.clone()));
                let wu_p = graph
                    .port(wu.element(), wu.tensor.extents())
                    .map_err(device_err)?;
                static_binds.push((wu_p.clone(), wu.tensor.clone()));
                let wd_p = graph
                    .port(wd.element(), wd.tensor.extents())
                    .map_err(device_err)?;
                static_binds.push((wd_p.clone(), wd.tensor.clone()));

                let wg_s_p = if let Some(sc) = &wg.scale {
                    let ps = graph.port(Element::f32(), sc.extents()).map_err(device_err)?;
                    static_binds.push((ps.clone(), sc.clone()));
                    ps
                } else {
                    empty_f32_p.clone()
                };
                let wu_s_p = if let Some(sc) = &wu.scale {
                    let ps = graph.port(Element::f32(), sc.extents()).map_err(device_err)?;
                    static_binds.push((ps.clone(), sc.clone()));
                    ps
                } else {
                    empty_f32_p.clone()
                };
                let wg_b_p = if let Some(bd) = &wg.bound {
                    let pb = graph.port(Element::f32(), bd.extents()).map_err(device_err)?;
                    static_binds.push((pb.clone(), bd.clone()));
                    pb
                } else {
                    empty_f32_p.clone()
                };
                let wu_b_p = if let Some(bd) = &wu.bound {
                    let pb = graph.port(Element::f32(), bd.extents()).map_err(device_err)?;
                    static_binds.push((pb.clone(), bd.clone()));
                    pb
                } else {
                    empty_f32_p.clone()
                };
                let wd_s_p = if let Some(sc) = &wd.scale {
                    let ps = graph.port(Element::f32(), sc.extents()).map_err(device_err)?;
                    static_binds.push((ps.clone(), sc.clone()));
                    ps
                } else {
                    empty_f32_p.clone()
                };
                let wd_b_p = if let Some(bd) = &wd.bound {
                    let pb = graph.port(Element::f32(), bd.extents()).map_err(device_err)?;
                    static_binds.push((pb.clone(), bd.clone()));
                    pb
                } else {
                    empty_f32_p.clone()
                };

                let exp_k = prepare_kernel(
                    &self.prepared.device,
                    dense_expand::native_entry_with(dense_expand::Elements {
                        NW: ffn_norm_w.element(),
                        GW: wg.element(),
                        UW: wu.element(),
                        A: kv_elem,
                    }),
                    &[
                        ("H", h),
                        ("F", f_dim),
                        ("GS", wg.scale_extent()),
                        ("US", wu.scale_extent()),
                        ("GB", wg.bound_extent()),
                        ("UB", wu.bound_extent()),
                        ("LB", 0),
                    ],
                )
                .map_err(device_err)?;
                let glu_limit = out_rows_p
                    .tensor()
                    .slice_leading(0, 0)
                    .reshape(&[0, 0]);
                let prod = graph
                    .enqueue(
                        &exp_k,
                        dense_expand::WorkflowArgs {
                            residual: (&x).into(),
                            norm: fn_p.tensor().into(),
                            gate_weight: wg_p.tensor().into(),
                            up_weight: wu_p.tensor().into(),
                            out_rows: out_rows_p.tensor().into(),
                            eps: eps.into(),
                            activation: 0i32.into(),
                            gate_scale: wg_s_p.tensor().into(),
                            up_scale: wu_s_p.tensor().into(),
                            gate_bound: wg_b_p.tensor().into(),
                            up_bound: wu_b_p.tensor().into(),
                            glu_limit: (&glu_limit).into(),
                        },
                    )
                    .map_err(device_err)?
                    .value;

                let out_k = prepare_kernel(
                    &self.prepared.device,
                    dense_output::native_entry_with(dense_output::Elements {
                        A: kv_elem,
                        DW: wd.element(),
                    }),
                    &[
                        ("H", h),
                        ("F", f_dim),
                        ("DS", wd.scale_extent()),
                        ("DB", wd.bound_extent()),
                    ],
                )
                .map_err(device_err)?;
                x = graph
                    .enqueue(
                        &out_k,
                        dense_output::WorkflowArgs {
                            residual: (&x).into(),
                            product: (&prod).into(),
                            down_weight: wd_p.tensor().into(),
                            out_rows: out_rows_p.tensor().into(),
                            down_scale: wd_s_p.tensor().into(),
                            down_bound: wd_b_p.tensor().into(),
                        },
                    )
                    .map_err(device_err)?
                    .value;
            }

            // Final output RMSNorm: x [M, H] f32 -> x_out [M, H] f32
            let out_norm_w = self.weight(WeightScope::Draft, WeightKind::OutputNorm)?;
            let on_p = graph
                .port(out_norm_w.element(), &[1, h])
                .map_err(device_err)?;
            let on_reshaped = out_norm_w.tensor.reshape(&[1, h]).map_err(device_err)?;
            static_binds.push((on_p.clone(), on_reshaped.clone()));
            owned.push(on_reshaped);

            let out_vnorm_k = prepare_kernel(
                &self.prepared.device,
                vision_norm::native_entry_with(vision_norm::Elements {
                    NWE: out_norm_w.element(),
                    NBE: Element::f32(),
                    Y: Element::f32(),
                }),
                &[("G", 1), ("H", h), ("NW", 1), ("NB", 0), ("NO", 0)],
            )
            .map_err(device_err)?;
            let x3 = x.reshape(&[m, 1, h]);
            let x_out = graph
                .enqueue(
                    &out_vnorm_k,
                    vision_norm::WorkflowArgs {
                        source: (&x3).into(),
                        weight: on_p.tensor().into(),
                        bias: empty_bias_p.tensor().into(),
                        order: empty_order_p.tensor().into(),
                        epsilon: eps.into(),
                        centered: 0i32.into(),
                        interleave: 0i32.into(),
                        activation: 0i32.into(),
                    },
                )
                .map_err(device_err)?
                .value;

            graph.store(&x_out, &out_port).map_err(device_err)?;
            let plan = graph.seal().map_err(device_err)?;
            let bind_refs: Vec<(&NativePort, &Tensor)> =
                static_binds.iter().map(|(p, t)| (p, t)).collect();
            let bound = plan
                .bind_static(&self.charges, &bind_refs)
                .map_err(device_err)?;
            let family = NativeGraphFamily::with_launches(
                std::slice::from_ref(&plan),
                &[vec![plan.clone()]],
                1,
                &[],
            )
            .map_err(device_err)?;
            let slot = family.new_slot(&self.charges, &[]).map_err(device_err)?;
            let x_in_buf =
                Tensor::upload(&self.charges, Element::f32(), &[m, h]).map_err(device_err)?;
            let coords_buf =
                Tensor::upload(&self.charges, Element::i32(), &[m, 4]).map_err(device_err)?;
            let fresh_buf =
                Tensor::upload(&self.charges, Element::i32(), &[m, 2]).map_err(device_err)?;
            let mut vis_bufs = Vec::with_capacity(num_layers);
            for _ in 0..num_layers {
                vis_bufs.push(
                    Tensor::upload(&self.charges, Element::i32(), &[m, spans, 2])
                        .map_err(device_err)?,
                );
            }
            let out_buf = Tensor::zeros(&self.charges, Element::f32(), &[m, h])
                .map_err(device_err)?;
            map.insert(
                key,
                BoundBlockGraph {
                    plan,
                    bound,
                    _family: family,
                    slot,
                    x_in_port: x_in_p,
                    coords_port: coords_p,
                    fresh_port: fresh_p,
                    visible_ports,
                    hk_ports,
                    hv_ports,
                    out_port,
                    x_in_buf,
                    coords_buf,
                    fresh_buf,
                    vis_bufs,
                    out_buf,
                    _owned: owned,
                },
            );
        }

        let g = map.get_mut(&key).unwrap();
        let mut bindings = g.bound.bindings();
        let x_bytes: Vec<u8> = x_in.iter().flat_map(|v| v.to_le_bytes()).collect();
        let coords_bytes: Vec<u8> = coords.iter().flat_map(|v| v.to_le_bytes()).collect();
        let fresh_bytes: Vec<u8> = fresh.iter().flat_map(|v| v.to_le_bytes()).collect();
        g.x_in_buf.write_from_host(&x_bytes).map_err(device_err)?;
        g.coords_buf
            .write_from_host(&coords_bytes)
            .map_err(device_err)?;
        g.fresh_buf
            .write_from_host(&fresh_bytes)
            .map_err(device_err)?;
        bindings
            .set(&self.charges, &g.x_in_port, &g.x_in_buf)
            .map_err(device_err)?;
        bindings
            .set(&self.charges, &g.coords_port, &g.coords_buf)
            .map_err(device_err)?;
        bindings
            .set(&self.charges, &g.fresh_port, &g.fresh_buf)
            .map_err(device_err)?;
        bindings
            .set(&self.charges, &g.out_port, &g.out_buf)
            .map_err(device_err)?;
        for l in 0..num_layers {
            let vis_bytes: Vec<u8> = visible_per_layer[l]
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect();
            g.vis_bufs[l]
                .write_from_host(&vis_bytes)
                .map_err(device_err)?;
            bindings
                .set(&self.charges, &g.visible_ports[l], &g.vis_bufs[l])
                .map_err(device_err)?;
            bindings
                .set(&self.charges, &g.hk_ports[l], &slabs[l].1)
                .map_err(device_err)?;
            bindings
                .set(&self.charges, &g.hv_ports[l], &slabs[l].2)
                .map_err(device_err)?;
        }

        let mut active = g.slot.activate(&g.plan).map_err(device_err)?;
        let ready = active
            .attach(
                bindings,
                g.plan.new_outputs(&self.charges).map_err(device_err)?,
            )
            .map_err(device_err)?;
        let mut seq = self.charges.device().native_sequence(NativeQueue::Compute);
        let _ = ready.queue(&mut seq).map_err(device_err)?;
        seq.submit()
            .map_err(device_err)?
            .wait()
            .map_err(device_err)?;
        read_f32(&g.out_buf)
    }

    pub(crate) fn execute(
        &self,
        core: &HeadLaunchCore,
        embedding: &crate::device_table::ResidentTable,
        row_class_slots: usize,
    ) -> Result<HeadOutput, SubmitError> {
        let draft = &self.prepared.draft;
        let h = self.prepared.hidden as usize;
        let active_vocab = self.prepared.active_vocab as usize;
        let batch = core.batch();
        let entry = batch.upload();
        let block = batch.chain().next();
        let steps = batch.steps();
        let actual_slots = batch.actual_slots();
        let head_slabs = self.extract_head_slabs(core);

        // 1. Entry pass: fuse target taps via DraftFusion::SoftmaxMix into per-layer
        //    ctx[0..L] + f_rnn, project K_ctx/V_ctx on GPU, apply k_norm_ctx + RoPE,
        //    and store into the persistent per-layer draft KV slabs.
        if entry.actual_rows > 0 && !self.fuse_norms.is_empty() {
            let t_cnt = draft.taps.len();
            let rows_in = entry.actual_rows;
            let raw_taps: Vec<f32> = match core.state() {
                crate::lanes::HeadState::Owned {
                    conditioning: crate::HeadConditioning::Features { current, .. },
                    ..
                } => read_f32(current.tensor())?,
                crate::lanes::HeadState::Owned {
                    conditioning: crate::HeadConditioning::Rows(rows),
                    ..
                }
                | crate::lanes::HeadState::TargetReads {
                    conditioning: rows, ..
                } => {
                    let mut out = Vec::with_capacity(rows_in * t_cnt * h);
                    for r in rows {
                        let rb = r.row_bytes();
                        if rb == t_cnt * h * 2 {
                            for b in r.bytes().chunks_exact(2) {
                                out.push(f32::from_bits(
                                    u32::from(u16::from_le_bytes([b[0], b[1]])) << 16,
                                ));
                            }
                        } else if rb == t_cnt * h * 4 {
                            for b in r.bytes().chunks_exact(4) {
                                out.push(f32::from_le_bytes([b[0], b[1], b[2], b[3]]));
                            }
                        } else {
                            return Err(invalid(format!(
                                "unexpected DSpark tap FeatureRows row_bytes={rb} (expected {} or {} for t_cnt={t_cnt}, h={h})",
                                t_cnt * h * 2,
                                t_cnt * h * 4,
                            )));
                        }
                    }
                    out
                }
                crate::lanes::HeadState::Owned {
                    conditioning: crate::HeadConditioning::Front(reads),
                    ..
                } => match &reads.features {
                    crate::FrontFeatures::Rows(features) => read_f32(features.tensor())?,
                    crate::FrontFeatures::Boundary(boundary) => {
                        read_f32(boundary.allocation().tensor().map_err(device_err)?)?
                    }
                },
            };
            if t_cnt > 0 && !raw_taps.is_empty() {
                if raw_taps.len() < rows_in * t_cnt * h {
                    return Err(invalid(format!(
                        "insufficient DSpark tap floats: got {}, need {} (rows_in={rows_in}, t_cnt={t_cnt}, h={h})",
                        raw_taps.len(),
                        rows_in * t_cnt * h,
                    )));
                }
                let mut tap_slices: Vec<Vec<f32>> = vec![vec![0.0f32; rows_in * h]; t_cnt];
                for r in 0..rows_in {
                    for t in 0..t_cnt {
                        let src = &raw_taps[(r * t_cnt + t) * h..(r * t_cnt + t + 1) * h];
                        tap_slices[t][r * h..(r + 1) * h].copy_from_slice(src);
                    }
                }
                let tap_refs: Vec<&[f32]> = tap_slices.iter().map(|v| v.as_slice()).collect();
                let norm_refs: Vec<(&[f32], f32)> = self
                    .fuse_norms
                    .iter()
                    .map(|(v, eps)| (v.as_slice(), *eps))
                    .collect();
                let fused = crate::operators::draft::softmax_mix_fuse(
                    &tap_refs,
                    &self.fuse_weights,
                    &norm_refs,
                    rows_in,
                    h,
                )
                .map_err(|e| invalid(e.to_string()))?;

                let entry_positions: Vec<i32> = (0..rows_in)
                    .map(|r| entry.coordinates.get(r).map_or(0, |c| c[0]))
                    .collect();

                // Record committed f_rnn keyed by each entry row's physical history destination
                // (never by transient launch-slot index).
                if fused.len() > draft.blocks.len() {
                    let hist = entry.histories.first().ok_or_else(|| {
                        invalid("DSpark entry pass has no history domain for f_rnn ownership")
                    })?;
                    let mut cache = self.cache.lock().expect("rnn draft cache lock");
                    let f_rnn_all = &fused[draft.blocks.len()];
                    for r in 0..rows_in {
                        let dst = hist.destinations.get(r).copied().unwrap_or(-1);
                        if dst >= 0 {
                            cache
                                .f_rnn
                                .record_row(dst, f_rnn_all[r * h..(r + 1) * h].to_vec());
                        }
                    }
                }

                // Fast path: write all L layers directly into GPU KV slabs in 1 sealed submit.
                if let Some(slabs) = &head_slabs {
                    let mut entry_coords = vec![0i32; rows_in * 4];
                    for r in 0..rows_in {
                        if let Some(c) = entry.coordinates.get(r) {
                            entry_coords[r * 4..(r + 1) * 4].copy_from_slice(c);
                        }
                    }
                    let mut dests_per_layer = Vec::with_capacity(draft.blocks.len());
                    for l in 0..draft.blocks.len() {
                        let domain = slabs[l].0.min(entry.histories.len().saturating_sub(1));
                        let dests = &entry.histories[domain].destinations;
                        let mut d = vec![-1i32; rows_in];
                        for r in 0..rows_in {
                            d[r] = dests.get(r).copied().unwrap_or(-1);
                        }
                        dests_per_layer.push(d);
                    }
                    self.run_inject_gpu(&fused, rows_in, &entry_coords, &dests_per_layer, slabs)?;
                } else {
                    let mut cache = self.cache.lock().expect("rnn draft cache lock");
                    // Project and cache context K/V for each draft layer on GPU.
                    for (l, blk) in draft.blocks.iter().enumerate() {
                        let Operator::Attention(attn) = &blk.sublayers[0].op else {
                            return Err(invalid("draft block sublayer 0 must be Attention"));
                        };
                        let s0 = WeightScope::DraftSublayer(SublayerIndex {
                            block: l as u32,
                            sublayer: 0,
                        });
                        let (wk, wv, kn) = if draft.context_projections.is_some() {
                            (
                                self.weight(s0, WeightKind::DraftContextKey)?,
                                self.weight(s0, WeightKind::DraftContextValue)?,
                                self.layer_norms[l]
                                    .k_norm_ctx
                                    .as_deref()
                                    .ok_or_else(|| invalid("missing DraftContextKeyNorm"))?,
                            )
                        } else {
                            (
                                self.weight(s0, WeightKind::Key)?,
                                self.weight(s0, WeightKind::Value)?,
                                self.layer_norms[l].k_norm.as_slice(),
                            )
                        };
                        let mut k_ctx = self.project_gpu(&fused[l], rows_in, wk)?;
                        let v_ctx = self.project_gpu(&fused[l], rows_in, wv)?;
                        let kv_heads = attn.kv_heads as usize;
                        let head_dim = attn.width as usize;
                        let k_eps = match &attn.query_norm {
                            HeadNorm::Rms(n) => n.epsilon as f32,
                            _ => 1e-6,
                        };
                        rms_norm_in_place(&mut k_ctx, kn, k_eps);
                        apply_rope(
                            &mut k_ctx,
                            rows_in,
                            kv_heads,
                            head_dim,
                            &entry_positions,
                            &attn.rotary,
                        );

                        // Write into persistent KV cache by physical destination row.
                        let domain = l.min(entry.histories.len().saturating_sub(1));
                        let dests = &entry.histories[domain].destinations;
                        let kv_dim = kv_heads * head_dim;
                        for r in 0..rows_in {
                            let dst = dests.get(r).copied().unwrap_or(-1);
                            if dst >= 0 {
                                cache.layers[l].insert(
                                    dst,
                                    (
                                        k_ctx[r * kv_dim..(r + 1) * kv_dim].to_vec(),
                                        v_ctx[r * kv_dim..(r + 1) * kv_dim].to_vec(),
                                    ),
                                );
                            }
                        }
                    }
                }
            }
        }

        let Some(block_upload) = block else {
            return Ok(HeadOutput {
                selected: HeadSelected::Causal,
                landed_reads: Vec::new(),
                expert_steps: Vec::new(),
            });
        };

        // 2. Drafting pass: assemble `[anchor_emb, mask_emb, ...]` on GPU, run all
        //    L draft transformer blocks (GPU projections + persistent KV + bidirectional
        //    block attention), apply `output_norm`, and run the RNN Markov chain with
        //    GPU `lm_head` projection.
        let block_size = draft.block_rows(steps as u64) as usize;
        let block_rows = actual_slots * block_size;
        let front_anchor_tokens: Option<Vec<i32>> = match core.state() {
            crate::lanes::HeadState::Owned {
                conditioning: crate::HeadConditioning::Front(reads),
                ..
            } => {
                let bytes = reads.selected.tensor().read_to_host().map_err(device_err)?;
                let vals: Vec<i32> = bytes
                    .chunks_exact(4)
                    .map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                    .collect();
                Some(
                    (0..actual_slots)
                        .map(|s| vals.get(s * 2).copied().unwrap_or(0))
                        .collect(),
                )
            }
            _ => None,
        };
        let anchor_tokens: Vec<i32> = front_anchor_tokens.unwrap_or_else(|| {
            (0..actual_slots)
                .map(|s| block_upload.tokens.get(s * block_size).copied().unwrap_or(0))
                .collect()
        });
        let anchor_embs = self.gather_embeddings_gpu(&anchor_tokens, embedding)?;

        let mut x = vec![0.0f32; block_rows * h];
        for s in 0..actual_slots {
            let base = s * block_size;
            x[base * h..(base + 1) * h].copy_from_slice(&anchor_embs[s * h..(s + 1) * h]);
            for k in 1..block_size {
                x[(base + k) * h..(base + k + 1) * h].copy_from_slice(&self.mask_emb);
            }
        }

        let block_positions: Vec<i32> = (0..block_rows)
            .map(|r| block_upload.coordinates.get(r).map_or(0, |c| c[0]))
            .collect();

        if let Some(slabs) = &head_slabs {
            let max_decode = magnitude_kernels::constants::VERIFY_ROWS as usize;
            let slots_per_chunk = (max_decode / block_size.max(1)).max(1);
            let mut slot_start = 0usize;
            while slot_start < actual_slots {
                let chunk_slots = (actual_slots - slot_start).min(slots_per_chunk);
                let chunk_rows = chunk_slots * block_size;
                let row_lo = slot_start * block_size;

                let mut coords = vec![0i32; chunk_rows * 4];
                let mut fresh = vec![0i32; chunk_rows * 2];
                for cs in 0..chunk_slots {
                    let c_lo = cs * block_size;
                    let c_hi = c_lo + block_size;
                    for r in c_lo..c_hi {
                        if let Some(c) = block_upload.coordinates.get(row_lo + r) {
                            coords[r * 4..(r + 1) * 4].copy_from_slice(c);
                        }
                        fresh[r * 2] = c_lo as i32;
                        fresh[r * 2 + 1] = c_hi as i32;
                    }
                }

                // Compute max visible span count across all layers and rows in this chunk.
                let mut spans = 1usize;
                for l in 0..draft.blocks.len() {
                    let domain = slabs[l].0.min(block_upload.histories.len().saturating_sub(1));
                    let visible = &block_upload.histories[domain].visible;
                    for r in 0..chunk_rows {
                        if let Some(sp) = visible.get(row_lo + r) {
                            spans = spans.max(sp.len());
                        }
                    }
                }
                let mut visible_per_layer = Vec::with_capacity(draft.blocks.len());
                for l in 0..draft.blocks.len() {
                    let domain = slabs[l].0.min(block_upload.histories.len().saturating_sub(1));
                    let visible = &block_upload.histories[domain].visible;
                    let mut vis_i32 = vec![0i32; chunk_rows * spans * 2];
                    for r in 0..chunk_rows {
                        if let Some(sp) = visible.get(row_lo + r) {
                            for (si, &[start, end]) in sp.iter().enumerate() {
                                let base = (r * spans + si) * 2;
                                vis_i32[base] = start;
                                vis_i32[base + 1] = end;
                            }
                        }
                    }
                    visible_per_layer.push(vis_i32);
                }

                let x_slice = &x[row_lo * h..(row_lo + chunk_rows) * h];
                let x_chunk_out = self.run_block_gpu(
                    x_slice,
                    chunk_rows,
                    &coords,
                    &fresh,
                    &visible_per_layer,
                    spans as u64,
                    slabs,
                )?;
                x[row_lo * h..(row_lo + chunk_rows) * h].copy_from_slice(&x_chunk_out);
                slot_start += chunk_slots;
            }
        } else {
            {
                let cache = self.cache.lock().expect("rnn draft cache lock");
                for (l, blk) in draft.blocks.iter().enumerate() {
                let Operator::Attention(attn) = &blk.sublayers[0].op else {
                    return Err(invalid("draft block sublayer 0 must be Attention"));
                };
                let s0 = WeightScope::DraftSublayer(SublayerIndex {
                    block: l as u32,
                    sublayer: 0,
                });
                let s1 = WeightScope::DraftSublayer(SublayerIndex {
                    block: l as u32,
                    sublayer: 1,
                });
                let eps = draft.output_norm.epsilon as f32;

                // Attention pre-norm + GPU Q/K/V projections
                let mut normed = x.clone();
                rms_norm_in_place(&mut normed, &self.layer_norms[l].attn_norm, eps);
                let mut q = self.project_gpu(&normed, block_rows, self.weight(s0, WeightKind::Query)?)?;
                let mut k = self.project_gpu(&normed, block_rows, self.weight(s0, WeightKind::Key)?)?;
                let v = self.project_gpu(&normed, block_rows, self.weight(s0, WeightKind::Value)?)?;

                let n_head = attn.heads as usize;
                let n_kv = attn.kv_heads as usize;
                let head_dim = attn.width as usize;
                let group = n_head / n_kv;
                let scale = attn.scale as f32;

                rms_norm_in_place(&mut q, &self.layer_norms[l].q_norm, eps);
                rms_norm_in_place(&mut k, &self.layer_norms[l].k_norm, eps);
                apply_rope(&mut q, block_rows, n_head, head_dim, &block_positions, &attn.rotary);
                apply_rope(&mut k, block_rows, n_kv, head_dim, &block_positions, &attn.rotary);

                // Attention over cached context KV (`visible` spans) + bidirectional block KV
                let domain = l.min(block_upload.histories.len().saturating_sub(1));
                let visible = &block_upload.histories[domain].visible;
                let kv_dim = n_kv * head_dim;
                let q_dim = n_head * head_dim;
                let mut attn_ctx = vec![0.0f32; block_rows * q_dim];

                for s in 0..actual_slots {
                    let slot_lo = s * block_size;
                    let slot_hi = slot_lo + block_size;
                    // Gather cached K/V rows visible to this slot (from row `slot_lo`).
                    let mut hist_kv: Vec<&(Vec<f32>, Vec<f32>)> = Vec::new();
                    if let Some(spans) = visible.get(slot_lo) {
                        for &[start, end] in spans {
                            for pos in start..end {
                                if let Some(entry_kv) = cache.layers[l].get(&pos) {
                                    hist_kv.push(entry_kv);
                                }
                            }
                        }
                    }
                    let total_k = hist_kv.len() + block_size;
                    let mut scores = vec![0.0f32; total_k];
                    for r in slot_lo..slot_hi {
                        for head in 0..n_head {
                            let kv_h = head / group;
                            let q_off = r * q_dim + head * head_dim;
                            let q_vec = &q[q_off..q_off + head_dim];
                            let mut max_s = f32::NEG_INFINITY;
                            for (idx, (hk, _)) in hist_kv.iter().enumerate() {
                                let k_vec = &hk[kv_h * head_dim..(kv_h + 1) * head_dim];
                                let dot: f32 =
                                    q_vec.iter().zip(k_vec.iter()).map(|(&a, &b)| a * b).sum();
                                let sc = dot * scale;
                                scores[idx] = sc;
                                if sc > max_s {
                                    max_s = sc;
                                }
                            }
                            for fk in 0..block_size {
                                let k_off = (slot_lo + fk) * kv_dim + kv_h * head_dim;
                                let k_vec = &k[k_off..k_off + head_dim];
                                let dot: f32 =
                                    q_vec.iter().zip(k_vec.iter()).map(|(&a, &b)| a * b).sum();
                                let sc = dot * scale;
                                scores[hist_kv.len() + fk] = sc;
                                if sc > max_s {
                                    max_s = sc;
                                }
                            }
                            let mut sum_e = 0.0f32;
                            for sc in &mut scores {
                                *sc = (*sc - max_s).exp();
                                sum_e += *sc;
                            }
                            let inv_e = 1.0 / sum_e.max(1e-20);
                            let out_vec = &mut attn_ctx[q_off..q_off + head_dim];
                            for (idx, (_, hv)) in hist_kv.iter().enumerate() {
                                let w = scores[idx] * inv_e;
                                let v_vec = &hv[kv_h * head_dim..(kv_h + 1) * head_dim];
                                for (o, &vv) in out_vec.iter_mut().zip(v_vec.iter()) {
                                    *o = w.mul_add(vv, *o);
                                }
                            }
                            for fk in 0..block_size {
                                let w = scores[hist_kv.len() + fk] * inv_e;
                                let v_off = (slot_lo + fk) * kv_dim + kv_h * head_dim;
                                let v_vec = &v[v_off..v_off + head_dim];
                                for (o, &vv) in out_vec.iter_mut().zip(v_vec.iter()) {
                                    *o = w.mul_add(vv, *o);
                                }
                            }
                        }
                    }
                }

                // Attention output projection on GPU + residual
                let attn_out =
                    self.project_gpu(&attn_ctx, block_rows, self.weight(s0, WeightKind::AttentionOutput)?)?;
                for (xi, &ao) in x.iter_mut().zip(attn_out.iter()) {
                    *xi += ao;
                }

                // SwiGLU FFN on GPU + residual
                let mut ffn_in = x.clone();
                rms_norm_in_place(&mut ffn_in, &self.layer_norms[l].ffn_norm, eps);
                let gate = self.project_gpu(&ffn_in, block_rows, self.weight(s1, WeightKind::DenseGate)?)?;
                let up = self.project_gpu(&ffn_in, block_rows, self.weight(s1, WeightKind::DenseUp)?)?;
                let act: Vec<f32> = gate
                    .iter()
                    .zip(up.iter())
                    .map(|(&g, &u)| (g / (1.0 + (-g).exp())) * u)
                    .collect();
                let ffn_out = self.project_gpu(&act, block_rows, self.weight(s1, WeightKind::DenseDown)?)?;
                for (xi, &fo) in x.iter_mut().zip(ffn_out.iter()) {
                    *xi += fo;
                }
            }
        }

            // Final output RMSNorm over all block rows.
            rms_norm_in_place(&mut x, &self.output_norm, draft.output_norm.epsilon as f32);
        }

        // 3. RNN Markov chain (`k = 0..steps`):
        //    - Resolve each slot's committed `f_rnn` from its last visible physical history row.
        //    - Precompute `rnn_b + W_frnn @ f_rnn + W_x @ h_draft[s, k]` across all steps on GPU
        //      in a single sealed `BoundMarkovPrepGraph` submit.
        //    - For each step `k`, run the small contiguous `[W_h | W_e]` recurrent gate step on
        //      CPU and fuse `vision_linear(hidden_out_w, hidden_out_b, residual=h_draft[k])` +
        //      `readout_project_rows(lm_head)` + `draft_top_k(K=1)` + `token_confidence` in one
        //      sealed `BoundMarkovStepGraph` GPU submit.
        let mut token_out = vec![0i32; steps * row_class_slots * 2];
        let mut conf_out = vec![1.0f32; steps * row_class_slots];
        let softcap = draft.logit_softcap.map(|c| c as f32).filter(|&c| c != 0.0);

        let block_hist = block_upload.histories.first().ok_or_else(|| {
            invalid("DSpark block pass has no history domain for f_rnn ownership")
        })?;
        let frnn_rows: Vec<f32> = {
            let cache = self.cache.lock().expect("rnn draft cache lock");
            let mut rows = Vec::with_capacity(actual_slots * h);
            for s in 0..actual_slots {
                let slot_lo = s * block_size;
                let visible = block_hist.visible.get(slot_lo).ok_or_else(|| {
                    invalid(format!(
                        "DSpark block slot {s} missing visible history spans for f_rnn"
                    ))
                })?;
                let f_rnn = cache.f_rnn.resolve_visible(visible)?;
                rows.extend_from_slice(f_rnn);
            }
            rows
        };

        // Pack step-major `h_draft_steps [steps, actual_slots, H]` once before the loop.
        let mut h_draft_steps = Vec::with_capacity(steps * actual_slots * h);
        for k in 0..steps {
            let offset = (k + 1).min(block_size - 1);
            for s in 0..actual_slots {
                let row_idx = s * block_size + offset;
                h_draft_steps.extend_from_slice(&x[row_idx * h..(row_idx + 1) * h]);
            }
        }

        let m = &self.markov;
        let r3 = 3 * m.rank;
        if let Some((chain_tokens, chain_confs)) = self.run_markov_chain_gpu(
            &frnn_rows,
            &h_draft_steps,
            &anchor_tokens,
            actual_slots,
            steps,
            softcap.unwrap_or(0.0),
        )? {
            for k in 0..steps {
                for s in 0..actual_slots {
                    let src_idx = k * actual_slots + s;
                    let dst_idx = k * row_class_slots + s;
                    token_out[dst_idx * 2] = chain_tokens[src_idx];
                    token_out[dst_idx * 2 + 1] = 0;
                    conf_out[dst_idx] = chain_confs[src_idx];
                }
            }
        } else {
            let gpu_step_biases =
                self.prep_markov_gpu(&frnn_rows, &h_draft_steps, actual_slots, steps)?;
            let cpu_frnn_biases: Vec<Vec<f32>> = if gpu_step_biases.is_none() {
                (0..actual_slots)
                    .map(|s| {
                        let f_rnn = &frnn_rows[s * h..(s + 1) * h];
                        crate::operators::draft::precompute_rnn_frnn_bias(
                            f_rnn, m.rank, h, &m.rnn_w, &m.rnn_b,
                        )
                    })
                    .collect()
            } else {
                Vec::new()
            };

            let h_draft_steps_gpu = if m.out_linear.is_some()
                && self.top_k_kernel.is_some()
                && self.conf_kernel.is_some()
                && (self.readout_project.is_some() || softcap.is_none())
            {
                Some(f32_tensor(
                    &self.charges,
                    &[steps as u64, actual_slots as u64, h as u64],
                    &h_draft_steps,
                )?)
            } else {
                None
            };

            let mut h_states: Vec<Vec<f32>> = vec![vec![0.0f32; m.rank]; actual_slots];
            let mut prev_toks: Vec<usize> = anchor_tokens
                .iter()
                .map(|&t| t.max(0) as usize)
                .collect();

            for k in 0..steps {
                let mut latents = Vec::with_capacity(actual_slots * m.rank);
                for s in 0..actual_slots {
                    let tok = prev_toks[s].min(active_vocab.saturating_sub(1));
                    let prev_emb = &m.embedding[tok * m.rank..(tok + 1) * m.rank];
                    let (h_next, latent) = if let Some(step_biases) = &gpu_step_biases {
                        let b_off = (k * actual_slots + s) * r3;
                        crate::operators::draft::step_rnn_markov_gates_he(
                            &h_states[s],
                            prev_emb,
                            &step_biases[b_off..b_off + r3],
                            &m.w_he,
                        )
                    } else {
                        let hd_off = (k * actual_slots + s) * h;
                        crate::operators::draft::step_rnn_markov_gates_precomputed(
                            &h_states[s],
                            prev_emb,
                            &h_draft_steps[hd_off..hd_off + h],
                            &cpu_frnn_biases[s],
                            &m.rnn_w,
                        )
                    };
                    h_states[s] = h_next;
                    latents.extend_from_slice(&latent);
                }

                if let Some(hd_gpu) = &h_draft_steps_gpu {
                    let hd_step = hd_gpu
                        .slice_leading(k as u64, (k + 1) as u64)
                        .map_err(device_err)?;
                    if let Some(selected) = self.step_markov_gpu(
                        &latents,
                        &hd_step,
                        actual_slots,
                        softcap.unwrap_or(0.0),
                    )? {
                        for (s, (best_tok, prob)) in selected.into_iter().enumerate() {
                            let idx = k * row_class_slots + s;
                            token_out[idx * 2] = best_tok as i32;
                            token_out[idx * 2 + 1] = 0;
                            conf_out[idx] = prob;
                            prev_toks[s] = best_tok;
                        }
                        continue;
                    }
                }

                let mut h_eff = vec![0.0f32; actual_slots * h];
                for s in 0..actual_slots {
                    let hd_off = (k * actual_slots + s) * h;
                    let h_draft = &h_draft_steps[hd_off..hd_off + h];
                    let latent = &latents[s * m.rank..(s + 1) * m.rank];
                    let dst = &mut h_eff[s * h..(s + 1) * h];
                    dst.copy_from_slice(h_draft);
                    for j in 0..h {
                        let w_row = &m.hidden_out_w[j * m.rank..(j + 1) * m.rank];
                        dst[j] += m.hidden_out_b[j]
                            + crate::operators::draft::dot_f32_simd(w_row, latent);
                    }
                }

                if let Some(selected) =
                    self.select_lm_head_gpu(&h_eff, actual_slots, softcap.unwrap_or(0.0))?
                {
                    for (s, (best_tok, prob)) in selected.into_iter().enumerate() {
                        let idx = k * row_class_slots + s;
                        token_out[idx * 2] = best_tok as i32;
                        token_out[idx * 2 + 1] = 0;
                        conf_out[idx] = prob;
                        prev_toks[s] = best_tok;
                    }
                    continue;
                }

                let mut logits = self.project_gpu(&h_eff, actual_slots, &self.lm_head)?;
                if let Some(cap) = softcap {
                    for z in &mut logits {
                        *z = (*z / cap).tanh() * cap;
                    }
                }
                for s in 0..actual_slots {
                    let row = &logits[s * active_vocab..(s + 1) * active_vocab];
                    let mut best_tok = 0usize;
                    let mut best_logit = f32::NEG_INFINITY;
                    for (v, &l) in row.iter().enumerate() {
                        if l > best_logit {
                            best_logit = l;
                            best_tok = v;
                        }
                    }
                    let sum_exp: f32 = row.iter().map(|&l| (l - best_logit).exp()).sum();
                    let prob = (1.0 / sum_exp.max(1e-12)).clamp(0.0, 1.0);
                    let idx = k * row_class_slots + s;
                    token_out[idx * 2] = best_tok as i32;
                    token_out[idx * 2 + 1] = 0;
                    conf_out[idx] = prob;
                    prev_toks[s] = best_tok;
                }
            }
        }

        let tok_bytes: Vec<u8> = token_out.iter().flat_map(|v| v.to_le_bytes()).collect();
        let conf_bytes: Vec<u8> = conf_out.iter().flat_map(|v| v.to_le_bytes()).collect();
        let tokens_t = Tensor::from_host(
            &self.charges,
            Element::i32(),
            &[steps as u64, row_class_slots as u64, 2],
            &tok_bytes,
        )
        .map_err(device_err)?;
        let conf_t = Tensor::from_host(
            &self.charges,
            Element::f32(),
            &[(steps * row_class_slots) as u64],
            &conf_bytes,
        )
        .map_err(device_err)?;
        Ok(HeadOutput {
            selected: HeadSelected::Proposals(HeadSelections {
                tokens: crate::GraphOutputTensor::standalone(tokens_t),
                confidence: Some(crate::GraphOutputTensor::standalone(conf_t)),
            }),
            landed_reads: Vec::new(),
            expert_steps: Vec::new(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operators::draft::{precompute_rnn_frnn_bias, step_rnn_markov_gates_precomputed};

    fn run_markov_steps(
        f_rnn: &[f32],
        h_draft_steps: &[Vec<f32>],
        rank: usize,
        rnn_w: &[f32],
        rnn_b: &[f32],
        emb: &[f32],
    ) -> Vec<(Vec<f32>, Vec<f32>)> {
        let hidden = f_rnn.len();
        let base = precompute_rnn_frnn_bias(f_rnn, rank, hidden, rnn_w, rnn_b);
        let mut h_state = vec![0.0f32; rank];
        let mut out = Vec::with_capacity(h_draft_steps.len());
        for (step, hd) in h_draft_steps.iter().enumerate() {
            let prev_emb = &emb[(step % 4) * rank..((step % 4) + 1) * rank];
            let (h_next, latent) =
                step_rnn_markov_gates_precomputed(&h_state, prev_emb, hd, &base, rnn_w);
            h_state = h_next.clone();
            out.push((h_next, latent));
        }
        out
    }

    #[test]
    fn dspark_frnn_interleaved_fork_rollback_isolation() {
        let (hidden, rank) = (8, 4);
        let in_dim = 2 * rank + 2 * hidden;
        let rnn_w: Vec<f32> = (0..3 * rank * in_dim)
            .map(|i| ((i % 13) as f32 - 6.0) * 0.03)
            .collect();
        let rnn_b: Vec<f32> = (0..3 * rank).map(|i| i as f32 * 0.005).collect();
        let emb: Vec<f32> = (0..4 * rank).map(|i| (i as f32 + 1.0) * 0.04).collect();
        let hd_steps: Vec<Vec<f32>> = (0..3)
            .map(|k| (0..hidden).map(|i| ((k * 7 + i) as f32) * 0.05).collect())
            .collect();

        let f_a0: Vec<f32> = (0..hidden).map(|i| 0.1 * (i as f32 + 1.0)).collect();
        let f_a1: Vec<f32> = (0..hidden).map(|i| 0.25 * (i as f32 + 1.0)).collect();
        let f_a2: Vec<f32> = (0..hidden).map(|i| -0.15 * (i as f32 + 1.0)).collect();
        let f_b0: Vec<f32> = (0..hidden).map(|i| -0.4 * (i as f32 + 1.0)).collect();
        let f_b1: Vec<f32> = (0..hidden).map(|i| 0.7 * (i as f32 + 1.0)).collect();

        // Isolated reference outputs for each committed f_rnn state.
        let ref_a0 = run_markov_steps(&f_a0, &hd_steps, rank, &rnn_w, &rnn_b, &emb);
        let ref_a1 = run_markov_steps(&f_a1, &hd_steps, rank, &rnn_w, &rnn_b, &emb);
        let ref_a2 = run_markov_steps(&f_a2, &hd_steps, rank, &rnn_w, &rnn_b, &emb);
        let ref_b0 = run_markov_steps(&f_b0, &hd_steps, rank, &rnn_w, &rnn_b, &emb);
        let ref_b1 = run_markov_steps(&f_b1, &hd_steps, rank, &rnn_w, &rnn_b, &emb);

        let mut store = FRnnStore::new();

        // Empty visible spans must error rather than silently returning zeros.
        assert!(store.resolve_visible(&[]).is_err());

        // Turn 1: Interleaved batch [slot 0 = seq A (phys row 0), slot 1 = seq B (phys row 64)].
        store.record_row(0, f_a0.clone());
        store.record_row(64, f_b0.clone());
        assert_eq!(
            run_markov_steps(
                store.resolve_visible(&[[0, 1]]).unwrap(),
                &hd_steps,
                rank,
                &rnn_w,
                &rnn_b,
                &emb
            ),
            ref_a0
        );
        assert_eq!(
            run_markov_steps(
                store.resolve_visible(&[[64, 65]]).unwrap(),
                &hd_steps,
                rank,
                &rnn_w,
                &rnn_b,
                &emb
            ),
            ref_b0
        );

        // Turn 2: Swap launch slots [slot 0 = seq B (phys row 65), slot 1 = seq A (phys row 1)].
        store.record_row(65, f_b1.clone());
        store.record_row(1, f_a1.clone());
        assert_eq!(
            run_markov_steps(
                store.resolve_visible(&[[64, 66]]).unwrap(),
                &hd_steps,
                rank,
                &rnn_w,
                &rnn_b,
                &emb
            ),
            ref_b1
        );
        assert_eq!(
            run_markov_steps(
                store.resolve_visible(&[[0, 2]]).unwrap(),
                &hd_steps,
                rank,
                &rnn_w,
                &rnn_b,
                &emb
            ),
            ref_a1
        );

        // Fork seq A at physical row 1 into A_branch1 (appends row 2) and A_branch2 (appends row 128).
        // Both branches initially see shared prefix [0, 2) -> f_a1.
        assert_eq!(store.resolve_visible(&[[0, 2]]).unwrap(), f_a1.as_slice());
        store.record_row(2, f_a2.clone());
        store.fork_row(1, 128).unwrap();
        assert_eq!(
            run_markov_steps(
                store.resolve_visible(&[[0, 3]]).unwrap(),
                &hd_steps,
                rank,
                &rnn_w,
                &rnn_b,
                &emb
            ),
            ref_a2
        );
        assert_eq!(
            run_markov_steps(
                store.resolve_visible(&[[0, 2], [128, 129]]).unwrap(),
                &hd_steps,
                rank,
                &rnn_w,
                &rnn_b,
                &emb
            ),
            ref_a1
        );

        // Rollback A_branch1 from [0, 3) back to checkpoint [0, 2) (discarding row 2).
        store.rollback_rows(&[2]);
        assert!(store.resolve_visible(&[[0, 3]]).is_err());
        assert_eq!(
            run_markov_steps(
                store.resolve_visible(&[[0, 2]]).unwrap(),
                &hd_steps,
                rank,
                &rnn_w,
                &rnn_b,
                &emb
            ),
            ref_a1
        );
        // Seq B remains completely unaffected by A's fork and rollback.
        assert_eq!(
            run_markov_steps(
                store.resolve_visible(&[[64, 66]]).unwrap(),
                &hd_steps,
                rank,
                &rnn_w,
                &rnn_b,
                &emb
            ),
            ref_b1
        );
    }

    #[test]
    fn dspark_sealed_markov_chain_parity() {
        let devices = seismic::Devices::discover(&[]).unwrap();
        let device = devices
            .open_backend(seismic::BackendName::Cuda)
            .or_else(|_| devices.open_backend(seismic::BackendName::Metal))
            .or_else(|_| devices.open_backend(seismic::BackendName::Cpu))
            .unwrap();
        let charges = Charges::tally(&device);
        let (slots, steps, hidden, rank, vocab) = (2usize, 7usize, 64usize, 32usize, 128usize);
        let in_dim = 2 * rank + 2 * hidden;

        let mut state = 0x1234_5678u32;
        let mut next_f32 = |scale: f32| -> f32 {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            ((state >> 8) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0) * scale
        };

        let rnn_w: Vec<f32> = (0..3 * rank * in_dim).map(|_| next_f32(0.15)).collect();
        let rnn_b: Vec<f32> = (0..3 * rank).map(|_| next_f32(0.1)).collect();
        let hidden_out_w: Vec<f32> = (0..hidden * rank).map(|_| next_f32(0.2)).collect();
        let hidden_out_b: Vec<f32> = (0..hidden).map(|_| next_f32(0.05)).collect();
        let markov_emb: Vec<f32> = (0..vocab * rank).map(|_| next_f32(0.3)).collect();
        let lm_head_vals: Vec<f32> = (0..vocab * hidden).map(|_| next_f32(0.4)).collect();
        let f_rnn_rows: Vec<f32> = (0..slots * hidden).map(|_| next_f32(0.5)).collect();
        let h_draft_steps: Vec<f32> = (0..steps * slots * hidden).map(|_| next_f32(0.5)).collect();
        let anchor_tokens = vec![5i32, 42i32];

        let mut w_he = Vec::with_capacity(3 * rank * 2 * rank);
        let mut w_h_vals = Vec::with_capacity(3 * rank * rank);
        let mut w_e_vals = Vec::with_capacity(3 * rank * rank);
        let mut w_x_vals = Vec::with_capacity(3 * rank * hidden);
        let mut w_frnn_vals = Vec::with_capacity(3 * rank * hidden);
        for o in 0..3 * rank {
            let row = &rnn_w[o * in_dim..(o + 1) * in_dim];
            w_he.extend_from_slice(&row[0..2 * rank]);
            w_h_vals.extend_from_slice(&row[0..rank]);
            w_e_vals.extend_from_slice(&row[rank..2 * rank]);
            w_x_vals.extend_from_slice(&row[2 * rank..2 * rank + hidden]);
            w_frnn_vals.extend_from_slice(&row[2 * rank + hidden..in_dim]);
        }

        let (hu, ru, r3, vu) = (hidden as u64, rank as u64, (3 * rank) as u64, vocab as u64);
        let lm_head_t = bf16_tensor(&charges, &[vu, hu], &lm_head_vals).unwrap();
        let markov_emb_t = bf16_tensor(&charges, &[vu, ru], &markov_emb).unwrap();
        let emb_t = bf16_tensor(&charges, &[vu, hu], &lm_head_vals).unwrap();
        let emb_kernel = prepare_kernel(
            &device,
            embedding_rows::native_entry_with(embedding_rows::Elements {
                EW: Element::bf16(),
                A: Element::bf16(),
            }),
            &[("D", hu)],
        )
        .unwrap();
        let markov_emb_kernel = prepare_kernel(
            &device,
            embedding_rows::native_entry_with(embedding_rows::Elements {
                EW: Element::bf16(),
                A: Element::bf16(),
            }),
            &[("D", ru)],
        )
        .unwrap();
        let frnn_linear = prepare_kernel(
            &device,
            vision_linear::native_entry_with(vision_linear::Elements {
                A: Element::bf16(),
                W: Element::bf16(),
                B: Element::f32(),
                Y: Element::f32(),
            }),
            &[("N", r3), ("K", hu), ("NB", 1), ("NR", 0), ("NG", 0), ("NC", 0)],
        )
        .unwrap();
        let x_linear = prepare_kernel(
            &device,
            vision_linear::native_entry_with(vision_linear::Elements {
                A: Element::bf16(),
                W: Element::bf16(),
                B: Element::f32(),
                Y: Element::f32(),
            }),
            &[("N", r3), ("K", hu), ("NB", 0), ("NR", 0), ("NG", 0), ("NC", 0)],
        )
        .unwrap();
        let he_linear = prepare_kernel(
            &device,
            vision_linear::native_entry_with(vision_linear::Elements {
                A: Element::bf16(),
                W: Element::bf16(),
                B: Element::f32(),
                Y: Element::f32(),
            }),
            &[("N", r3), ("K", ru), ("NB", 0), ("NR", 0), ("NG", 0), ("NC", 0)],
        )
        .unwrap();
        let out_linear = prepare_kernel(
            &device,
            vision_linear::native_entry_with(vision_linear::Elements {
                A: Element::bf16(),
                W: Element::bf16(),
                B: Element::f32(),
                Y: Element::bf16(),
            }),
            &[("N", hu), ("K", ru), ("NB", 1), ("NR", 1), ("NG", 0), ("NC", 0)],
        )
        .unwrap();
        let gate_kernel = prepare_kernel(
            &device,
            rnn_markov_gate::native_entry_with(rnn_markov_gate::Elements { A: Element::bf16() }),
            &[],
        )
        .unwrap();
        let readout_project = prepare_kernel(
            &device,
            readout_project_rows::native_entry_with(readout_project_rows::Elements {
                A: Element::bf16(),
                OW: Element::bf16(),
            }),
            &[("V", vu), ("D", hu), ("S", vu), ("I", 0), ("WS", 0)],
        )
        .ok();
        let mut projects = HashMap::new();
        if let Ok(pk) = prepare_kernel(
            &device,
            project_rows::native_entry_with(project_rows::Elements {
                A: Element::bf16(),
                W: Element::bf16(),
                Y: Element::f32(),
            }),
            &[("G", 1), ("K", hu), ("N", vu), ("WS", 0), ("WB", 0)],
        ) {
            projects.insert((Element::bf16(), Element::bf16(), Element::f32(), hu, vu, 0, 0), pk);
        }
        let top_k_kernel = prepare_kernel(&device, draft_top_k::native_entry(), &[]).unwrap();
        let conf_kernel = prepare_kernel(&device, token_confidence::native_entry(), &[]).unwrap();

        let wd = |name: &str, shape: &[u64]| {
            magnitude_family_contracts::WeightDescriptor::stored(name, shape)
        };
        let draft_def = DraftDefinition {
            blocks: Vec::new(),
            block_attention: Vec::new(),
            taps: Vec::new(),
            fusion: DraftFusion::SoftmaxMix {
                weights: wd("fuse.weight", &[1, 1]),
                norms: Vec::new(),
            },
            embedding: magnitude_family_contracts::DraftEmbedding::Target,
            output: None,
            mask: DraftMask::Learned(wd("mask_embd.weight", &[hu])),
            context_projections: None,
            convolutions: None,
            residual: magnitude_family_contracts::ResidualForm::Single,
            output_norm: magnitude_family_contracts::RmsNorm {
                weight: wd("output_norm.weight", &[hu]),
                epsilon: 1e-6,
            },
            embedding_scale: 1.0,
            logit_softcap: None,
            block_size: (steps + 1) as u64,
            layout: magnitude_family_contracts::BlockLayout::AnchorFirst,
            method: DraftMethod::DSpark {
                markov: MarkovHead::RnnHidden(magnitude_family_contracts::RnnMarkovHead {
                    rank: ru,
                    embedding: wd("markov_w1.weight", &[vu, ru]),
                    rnn_proj: magnitude_family_contracts::Affine {
                        weight: wd("rnn_proj.weight", &[r3, 2 * ru + 2 * hu]),
                        bias: wd("rnn_proj.bias", &[r3]),
                    },
                    hidden_out: magnitude_family_contracts::Affine {
                        weight: wd("hidden_out.weight", &[hu, ru]),
                        bias: wd("hidden_out.bias", &[hu]),
                    },
                }),
                confidence: None,
            },
        };

        let bound = BoundRnnDraft {
            prepared: PreparedRnnDraft::new(&device, draft_def, vu, hu, Vec::new()),
            weights: HashMap::new(),
            lm_head: BoundWeight {
                tensor: lm_head_t,
                scale: None,
                bound: None,
            },
            emb_table: emb_t,
            emb_kernel,
            readout_project,
            top_k_kernel: Some(top_k_kernel),
            conf_kernel: Some(conf_kernel),
            lm_select_graphs: Mutex::new(HashMap::new()),
            markov_prep_graphs: Mutex::new(HashMap::new()),
            markov_step_graphs: Mutex::new(HashMap::new()),
            markov_chain_graphs: Mutex::new(HashMap::new()),
            inject_graphs: Mutex::new(HashMap::new()),
            block_graphs: Mutex::new(HashMap::new()),
            empty_f32: f32_tensor(&charges, &[0], &[]).unwrap(),
            projects,
            mask_emb: Vec::new(),
            output_norm: Vec::new(),
            fuse_weights: Vec::new(),
            fuse_norms: Vec::new(),
            layer_norms: Vec::new(),
            markov: MarkovWeights {
                rank,
                embedding: markov_emb.clone(),
                w_he,
                rnn_w: rnn_w.clone(),
                rnn_b: rnn_b.clone(),
                hidden_out_w: hidden_out_w.clone(),
                hidden_out_b: hidden_out_b.clone(),
                w_h_gpu: bf16_tensor(&charges, &[r3, ru], &w_h_vals).unwrap(),
                w_e_gpu: bf16_tensor(&charges, &[r3, ru], &w_e_vals).unwrap(),
                w_x_gpu: bf16_tensor(&charges, &[r3, hu], &w_x_vals).unwrap(),
                w_frnn_gpu: bf16_tensor(&charges, &[r3, hu], &w_frnn_vals).unwrap(),
                rnn_b_gpu: f32_tensor(&charges, &[1, r3], &rnn_b).unwrap(),
                hidden_out_w_gpu: bf16_tensor(&charges, &[hu, ru], &hidden_out_w).unwrap(),
                hidden_out_b_gpu: f32_tensor(&charges, &[1, hu], &hidden_out_b).unwrap(),
                markov_emb_gpu: markov_emb_t,
                markov_emb_kernel: Some(markov_emb_kernel),
                frnn_linear: Some(frnn_linear),
                x_linear: Some(x_linear),
                he_linear: Some(he_linear),
                out_linear: Some(out_linear),
                gate_kernel: Some(gate_kernel),
            },
            cache: Mutex::new(RnnDraftCache {
                layers: Vec::new(),
                f_rnn: FRnnStore::new(),
            }),
            charges,
        };

        let (gpu_tokens, gpu_confs) = bound
            .run_markov_chain_gpu(&f_rnn_rows, &h_draft_steps, &anchor_tokens, slots, steps, 0.0)
            .unwrap()
            .expect("BoundMarkovChainGraph must specialize");
        assert_eq!(gpu_tokens.len(), steps * slots);
        assert_eq!(gpu_confs.len(), steps * slots);

        // Also verify `steps = 1` (single-step chain where `w_h_p` is omitted).
        let (gpu_tokens_1, gpu_confs_1) = bound
            .run_markov_chain_gpu(
                &f_rnn_rows,
                &h_draft_steps[..slots * hidden],
                &anchor_tokens,
                slots,
                1,
                0.0,
            )
            .unwrap()
            .expect("BoundMarkovChainGraph(steps=1) must specialize");
        assert_eq!(gpu_tokens_1, gpu_tokens[..slots]);
        for s in 0..slots {
            assert!((gpu_confs_1[s] - gpu_confs[s]).abs() < 1e-4);
        }

        // CPU reference over all 7 steps.
        let mut h_states = vec![vec![0.0f32; rank]; slots];
        let mut prev_toks: Vec<usize> = anchor_tokens.iter().map(|&t| t as usize).collect();
        let bases: Vec<Vec<f32>> = (0..slots)
            .map(|s| {
                precompute_rnn_frnn_bias(
                    &f_rnn_rows[s * hidden..(s + 1) * hidden],
                    rank,
                    hidden,
                    &rnn_w,
                    &rnn_b,
                )
            })
            .collect();

        for k in 0..steps {
            for s in 0..slots {
                let prev_emb = &markov_emb[prev_toks[s] * rank..(prev_toks[s] + 1) * rank];
                let hd = &h_draft_steps[(k * slots + s) * hidden..(k * slots + s + 1) * hidden];
                let (h_next, latent) = step_rnn_markov_gates_precomputed(
                    &h_states[s],
                    prev_emb,
                    hd,
                    &bases[s],
                    &rnn_w,
                );
                h_states[s] = h_next;
                let mut h_eff = hd.to_vec();
                for j in 0..hidden {
                    h_eff[j] += hidden_out_b[j]
                        + crate::operators::draft::dot_f32_simd(
                            &hidden_out_w[j * rank..(j + 1) * rank],
                            &latent,
                        );
                }
                let mut best_tok = 0usize;
                let mut best_logit = f32::NEG_INFINITY;
                let mut logits = vec![0.0f32; vocab];
                for v in 0..vocab {
                    let l = crate::operators::draft::dot_f32_simd(
                        &lm_head_vals[v * hidden..(v + 1) * hidden],
                        &h_eff,
                    );
                    logits[v] = l;
                    if l > best_logit {
                        best_logit = l;
                        best_tok = v;
                    }
                }
                let sum_exp: f32 = logits.iter().map(|&l| (l - best_logit).exp()).sum();
                let exp_prob = 1.0 / sum_exp;
                let idx = k * slots + s;
                assert_eq!(
                    gpu_tokens[idx] as usize, best_tok,
                    "step {k} slot {s}: token mismatch"
                );
                assert!(
                    (gpu_confs[idx] - exp_prob).abs() < 5e-2,
                    "step {k} slot {s}: conf mismatch gpu={} ref={}",
                    gpu_confs[idx],
                    exp_prob
                );
                prev_toks[s] = best_tok;
            }
        }
    }
}
