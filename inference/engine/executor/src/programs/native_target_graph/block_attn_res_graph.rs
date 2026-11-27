//! Sealed `NativeGraph` execution for `ResidualForm::BlockAttention` (Inkling 2).
//!
//! Replaces the eager per-kernel loop with sealed, statically weight-bound
//! `NativeGraphPlan`s queued directly onto `StepSubmitter`'s `NativeGraphSequence`:
//! - **Zero host syncs** across all 48 layers + exit + readout on decode and
//!   resident/whole-bank prefill (one asynchronous GPU submission per step).
//! - **Zero `stack_sources()` GPU↔CPU copies**: all completed residual blocks
//!   and the running partial live in a single device-resident `[max_sources, M, H]`
//!   F32 buffer indexed via zero-copy `WorkflowTensor::slice_leading` views.
//! - **Fused `SplitQkv` mixing**: `Query`/`Key`/`Value` block-attention and
//!   cross-pass queries are pre-concatenated into `[3, H]` F32 tensors at bind
//!   time so `block_attn_res_mix` runs in a single `Q = 3` launch.
//! - **100% device-side `RelativeBiasProjection`** via `rel_bias_project` on all
//!   backends (eliminating the CUDA/Vulkan CPU fallback for `d_rel = 16`).
//! - **Device-side `TargetTokens::Selected` and DSpark `tap_rows`**: no host
//!   round-trips for selected tokens or target tap capture.

use super::block_attn_res::{
    BlockAttnResBankRun, BlockAttnResLayerState, BoundBlockAttnRes, BoundWeight,
    PreparedBlockAttnRes,
};
use crate::operators::routed::{
    fused_graph::decodes,
    split::{Crossing, Merged, SplitPorts, split as split_shape},
};
use crate::programs::graph::facts::constant;
use magnitude_family_contracts::{
    AttentionGate, AttentionSinks, BlockAttentionNorm, BlockAttentionQueries, BlockAttentionStream,
    BlockTransition, CrossPassInput, Decoder, ExitNorm, GateFunction, HeadNorm,
    HistoryRows, InputNorm, LogScale, MessageControllerPart, Operator, OutputForm, RelativeBias,
    RowProducer, SublayerIndex, TapPoint, ValueNorm, WeightKind, WeightRole, WeightScope,
};
use magnitude_kernels::{
    attention_rel_bias, block_attn_res_mix, causal_short_conv, feature_copy,
    head_l2_clip, message_gate_mix, message_gate_mix_norm, project_rows, readout_features_rows,
    rel_bias_project, rms_concat, stream_join, tap_rows, vision_norm,
};
use seismic::{
    BoundNativeGraphPlan, Charges, Dim, Element, NativeGraph, NativeGraphBindings,
    NativeGraphFamily, NativeGraphFamilySlot, NativeGraphOutputs, NativeGraphPlan, NativePort,
    NativeQueue, ReadyNativeGraphRun, Tensor, WorkflowTensor, WorkflowTensorRef,
};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Device-memory envelope (bytes) required by `BoundBlockAttnRes` for its
/// runtime scratch (`shared_slots` + 2-slot `decode_ring`), allocated from
/// `BoundBlockAttnRes::charges` (`binding_constant_bytes`) alongside
/// `prefill_bufs`.
pub(crate) fn block_attn_res_runtime_scratch_bytes(
    max_rows: u64,
    hidden: u64,
    max_sources: u64,
    max_bank: u64,
) -> u64 {
    let row_f32 = max_rows.saturating_mul(hidden).saturating_mul(4);
    let shared_slots = 20u64.saturating_mul(row_f32);

    // 2-slot `decode_ring` for `m <= VERIFY_ROWS`: each slot owns one
    // `BarStepBuffers` (using `cp_msgs_buf` up to 4 queries/bank-layer instead
    // of `bank_buf` when looped), its `NativeGraphFamilySlot` workspace, and
    // fixed-address `i32` control buffers.
    let decode_m = max_rows.min(magnitude_kernels::constants::VERIFY_ROWS);
    let decode_row_f32 = decode_m.saturating_mul(hidden).saturating_mul(4);
    let decode_cp_and_prelude = if max_bank > 0 {
        max_bank.saturating_mul(4).saturating_add(1)
    } else {
        0
    };
    let decode_bufs_per_slot = max_sources
        .saturating_add(decode_cp_and_prelude)
        .saturating_add(2)
        .saturating_mul(decode_row_f32);
    let decode_workspace_per_slot = 20u64.saturating_mul(decode_row_f32);
    let decode_ctrl_per_slot = decode_m
        .saturating_mul(21)
        .saturating_add(2)
        .saturating_mul(4);
    let decode_ring = 2u64.saturating_mul(
        decode_bufs_per_slot
            .saturating_add(decode_workspace_per_slot)
            .saturating_add(decode_ctrl_per_slot),
    );

    shared_slots.saturating_add(decode_ring)
}

/// Static residual-bank slot schedule for one block, derived once from
/// `BlockAttnResPlan`.
#[derive(Clone, Copy, Debug)]
pub(crate) struct BarBlockSchedule {
    pub transition: Option<BlockTransition>,
    /// Number of active sources in `sources_buf[0..attn_s]` entering Attention.
    pub attn_sources_count: u64,
    /// Slot in `sources_buf` holding `partial` entering Attention.
    pub attn_partial_in_slot: u64,
    /// Slot in `sources_buf` written with `partial` after Attention.
    pub attn_partial_out_slot: u64,
    /// Number of active sources in `sources_buf[0..ffn_s]` entering FFN.
    pub ffn_sources_count: u64,
    /// Slot in `sources_buf` written with `partial` after FFN.
    pub ffn_partial_out_slot: u64,
    /// For `SnapshotPrelude` / `ResetToPreludeAndBank`: slot `P` of `pre_partial`.
    pub prelude_slot: Option<u64>,
    /// For `ResetToPreludeAndBank`: contiguous range `(start, end)` in `sources_buf`
    /// copied into `bank_buf` (or read directly by pass-entry batched cross-pass mix).
    pub bank_src_range: Option<(u64, u64)>,
    /// Active source count in `bank_buf` when `cross_pass != None`.
    pub bank_count: u64,
    /// Leading query offset in `cp_msgs_buf` for Attention's cross-pass message(s)
    /// (length 1 for `Single`, length 3 for `SplitQkv`).
    pub attn_cp_msg_offset: Option<u64>,
    /// Leading query offset in `cp_msgs_buf` for FFN's cross-pass message (length 1).
    pub ffn_cp_msg_offset: Option<u64>,
}

/// Whole-decoder static slot schedule for `sources_buf`, `bank_buf`, and
/// pass-entry batched `cp_msgs_buf`.
#[derive(Clone, Debug)]
pub(crate) struct BarSchedule {
    pub blocks: Vec<BarBlockSchedule>,
    pub max_sources: u64,
    pub max_bank: u64,
    pub exit_sources_count: u64,
    /// Per-block list of static `(WeightScope, BlockAttentionStream)` cross-pass
    /// queries to evaluate in one batched `block_attn_res_mix` on each
    /// `ResetToPreludeAndBank` block (empty for non-reset blocks).
    pub pass_cp_queries: Vec<Vec<(WeightScope, BlockAttentionStream)>>,
    /// Maximum number of cross-pass queries batched in any single pass.
    pub max_pass_cp_queries: u64,
}

impl BarSchedule {
    pub(crate) fn of(
        decoder: &Decoder,
        plan: &crate::operators::block_attn_res::BlockAttnResPlan,
    ) -> Result<Self, String> {
        let mut completed: u64 = 0;
        let mut max_sources: u64 = 1;
        let mut max_bank: u64 = 0;
        let mut prelude_completed: Option<u64> = None;
        let mut cur_bank_count: u64 = 0;
        let mut current_reset_block: Option<usize> = None;
        let mut pass_cp_queries: Vec<Vec<(WeightScope, BlockAttentionStream)>> =
            vec![Vec::new(); decoder.blocks.len()];
        let mut blocks = Vec::with_capacity(decoder.blocks.len());

        for (b_idx, block) in decoder.blocks.iter().enumerate() {
            let sched = plan.blocks[b_idx];
            let mut prelude_slot = None;
            let mut bank_src_range = None;

            match sched.transition {
                None => {}
                Some(BlockTransition::SnapshotPrelude) => {
                    prelude_completed = Some(completed);
                    prelude_slot = Some(completed);
                }
                Some(BlockTransition::ResetToPreludeAndBank) => {
                    let pre_c = prelude_completed
                        .ok_or("ResetToPreludeAndBank without prior SnapshotPrelude")?;
                    let src_lo = pre_c + 1;
                    let src_hi = completed + 1;
                    let bank_len = src_hi.saturating_sub(src_lo);
                    if bank_len == 0 {
                        return Err("ResetToPreludeAndBank produced empty cross-pass bank".into());
                    }
                    prelude_slot = Some(pre_c);
                    bank_src_range = Some((src_lo, src_hi));
                    cur_bank_count = bank_len;
                    max_bank = max_bank.max(bank_len);
                    completed = pre_c;
                    current_reset_block = Some(b_idx);
                }
            }

            let attn_partial_in_slot = completed;
            let attn_sources_count = completed + 1;
            max_sources = max_sources.max(attn_sources_count);

            let attn_sub = &block.sublayers[0];
            let attn_scope = WeightScope::TargetSublayer(SublayerIndex {
                block: b_idx as u32,
                sublayer: 0,
            });
            let attn_cp_msg_offset = if let InputNorm::BlockAttention(attn_in) = &attn_sub.input {
                match &attn_in.cross_pass {
                    CrossPassInput::None => None,
                    CrossPassInput::Single(_) => {
                        let reset_b = current_reset_block
                            .ok_or("Attention cross-pass input without prior ResetToPreludeAndBank")?;
                        let list = &mut pass_cp_queries[reset_b];
                        let off = list.len() as u64;
                        list.push((attn_scope, BlockAttentionStream::Shared));
                        Some(off)
                    }
                    CrossPassInput::SplitQkv { .. } => {
                        let reset_b = current_reset_block
                            .ok_or("Attention cross-pass input without prior ResetToPreludeAndBank")?;
                        let list = &mut pass_cp_queries[reset_b];
                        let off = list.len() as u64;
                        list.push((attn_scope, BlockAttentionStream::Query));
                        list.push((attn_scope, BlockAttentionStream::Key));
                        list.push((attn_scope, BlockAttentionStream::Value));
                        Some(off)
                    }
                }
            } else {
                None
            };

            let attn_partial_out_slot = match attn_sub.output {
                OutputForm::BlockCommit => {
                    completed += 1;
                    completed
                }
                OutputForm::Residual => completed,
                _ => return Err("unsupported attention OutputForm in BlockAttention".into()),
            };
            let ffn_sources_count = completed + 1;
            max_sources = max_sources.max(ffn_sources_count);
            let ffn_partial_out_slot = completed;

            let ffn_cp_msg_offset = if let Some(ffn_sub) = block.sublayers.get(1) {
                let ffn_scope = WeightScope::TargetSublayer(SublayerIndex {
                    block: b_idx as u32,
                    sublayer: 1,
                });
                if let InputNorm::BlockAttention(ffn_in) = &ffn_sub.input {
                    match &ffn_in.cross_pass {
                        CrossPassInput::None => None,
                        CrossPassInput::Single(_) => {
                            let reset_b = current_reset_block
                                .ok_or("FFN cross-pass input without prior ResetToPreludeAndBank")?;
                            let list = &mut pass_cp_queries[reset_b];
                            let off = list.len() as u64;
                            list.push((ffn_scope, BlockAttentionStream::Shared));
                            Some(off)
                        }
                        CrossPassInput::SplitQkv { .. } => {
                            return Err("SplitQkv cross-pass invalid on FFN sublayer".into());
                        }
                    }
                } else {
                    None
                }
            } else {
                None
            };

            blocks.push(BarBlockSchedule {
                transition: sched.transition,
                attn_sources_count,
                attn_partial_in_slot,
                attn_partial_out_slot,
                ffn_sources_count,
                ffn_partial_out_slot,
                prelude_slot,
                bank_src_range,
                bank_count: cur_bank_count,
                attn_cp_msg_offset,
                ffn_cp_msg_offset,
            });
        }

        let exit_sources_count = completed + 1;
        max_sources = max_sources.max(exit_sources_count);
        let max_pass_cp_queries = pass_cp_queries
            .iter()
            .map(|v| v.len() as u64)
            .max()
            .unwrap_or(0);

        Ok(Self {
            blocks,
            max_sources,
            max_bank,
            exit_sources_count,
            pass_cp_queries,
            max_pass_cp_queries,
        })
    }
}

#[inline]
fn use_batched_cp_msgs(schedule: &BarSchedule, m: u64) -> bool {
    decodes(m) && schedule.max_pass_cp_queries > 0
}

/// Cache key identifying a sealed, statically bound `BoundBarFamily`.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct BarLayoutKey {
    pub m: u64,
    pub spans: u64,
    pub slots: u64,
    pub has_taps: bool,
    pub tap_element: Option<Element>,
    /// Per-layer `(history_rows, slab_rows, conv_banks, slab_banks)`.
    pub layer_shapes: Vec<(u64, u32, u64, u32)>,
}

/// Per-step residual-bank buffers (each step in a multi-step prefill chain
/// owns its own residual bank; all steps share one `NativeGraphFamilySlot`
/// because they execute sequentially on the single compute queue; `Exit` writes
/// directly into `ffn_out_buf` once all FFN layers complete).
pub(super) struct BarStepBuffers {
    pub(super) sources_buf: Tensor,
    pub(super) bank_buf: Option<Tensor>,
    pub(super) cp_msgs_buf: Option<Tensor>,
    pub(super) prelude_partial: Option<Tensor>,
    pub(super) ffn_in_buf: Tensor,
    pub(super) ffn_in_3d: Tensor,
    pub(super) ffn_out_buf: Tensor,
}

impl BarStepBuffers {
    /// Exact `(Element, shape)` list of every buffer in one `BarStepBuffers`.
    pub(super) fn extents(
        max_sources: u64,
        max_bank: u64,
        m: u64,
        h: u64,
    ) -> Vec<(Element, Vec<u64>)> {
        let mut out = Vec::with_capacity(5);
        out.push((Element::f32(), vec![max_sources, m, h]));
        if max_bank > 0 {
            out.push((Element::f32(), vec![max_bank, m, h]));
            out.push((Element::f32(), vec![m, h]));
        }
        out.push((Element::f32(), vec![m, h]));
        out.push((Element::f32(), vec![m, h]));
        out
    }

    pub(super) fn tensors(&self) -> impl Iterator<Item = &Tensor> {
        [&self.sources_buf, &self.ffn_in_buf, &self.ffn_out_buf]
            .into_iter()
            .chain(self.bank_buf.iter())
            .chain(self.cp_msgs_buf.iter())
            .chain(self.prelude_partial.iter())
    }

    pub(super) fn allocate(
        charges: &Charges,
        schedule: &BarSchedule,
        m: u64,
        h: u64,
    ) -> Result<Self, String> {
        let batched_cp = use_batched_cp_msgs(schedule, m);
        let sources_buf = Tensor::zeros(charges, Element::f32(), &[schedule.max_sources, m, h])
            .map_err(|e| e.to_string())?;
        let bank_buf = if schedule.max_bank > 0 && !batched_cp {
            Some(
                Tensor::zeros(charges, Element::f32(), &[schedule.max_bank, m, h])
                    .map_err(|e| e.to_string())?,
            )
        } else {
            None
        };
        let cp_msgs_buf = if batched_cp {
            Some(
                Tensor::zeros(
                    charges,
                    Element::f32(),
                    &[schedule.max_pass_cp_queries, m, h],
                )
                .map_err(|e| e.to_string())?,
            )
        } else {
            None
        };
        let prelude_partial = if schedule.max_bank > 0 {
            Some(Tensor::zeros(charges, Element::f32(), &[m, h]).map_err(|e| e.to_string())?)
        } else {
            None
        };
        let ffn_in_buf =
            Tensor::zeros(charges, Element::f32(), &[m, h]).map_err(|e| e.to_string())?;
        let ffn_in_3d = ffn_in_buf.reshape(&[1, m, h]).map_err(|e| e.to_string())?;
        let ffn_out_buf =
            Tensor::zeros(charges, Element::f32(), &[m, h]).map_err(|e| e.to_string())?;
        Ok(Self {
            sources_buf,
            bank_buf,
            cp_msgs_buf,
            prelude_partial,
            ffn_in_buf,
            ffn_in_3d,
            ffn_out_buf,
        })
    }
}

/// Reusable device scratch, persistent control buffers, and `NativeGraphFamilySlot`
/// for one in-flight decode step. Keeping control buffer addresses fixed across steps
/// eliminates per-step CUDA graph parameter rebinding (`0` graph updates/step).
struct BarStepScratch {
    bufs: BarStepBuffers,
    slot: NativeGraphFamilySlot,
    coords_buf: Tensor,
    fresh_buf: Tensor,
    dest_buf: Tensor,
    conv_seg_buf: Tensor,
    conv_stop_buf: Tensor,
    conv_prev_bank_buf: Tensor,
    conv_prev_tape_buf: Tensor,
    conv_next_bank_buf: Tensor,
    vis_bufs: Vec<Tensor>,
}

impl BarStepScratch {
    fn allocate(
        charges: &Charges,
        family: &NativeGraphFamily,
        schedule: &BarSchedule,
        m: u64,
        h: u64,
        slots: u64,
        spans: u64,
    ) -> Result<Self, String> {
        let bufs = BarStepBuffers::allocate(charges, schedule, m, h)?;
        let slot = family.new_slot(charges, &[]).map_err(|e| e.to_string())?;
        let z_i32 = |shape: &[u64]| -> Result<Tensor, String> {
            Tensor::zeros(charges, Element::i32(), shape).map_err(|e| e.to_string())
        };
        let coords_buf = z_i32(&[m, 4])?;
        let fresh_buf = z_i32(&[m, 2])?;
        let dest_buf = z_i32(&[m])?;
        let conv_seg_buf = z_i32(&[slots + 1, 2])?;
        let conv_stop_buf = z_i32(&[slots])?;
        let conv_prev_bank_buf = z_i32(&[slots])?;
        let conv_prev_tape_buf = z_i32(&[slots])?;
        let conv_next_bank_buf = z_i32(&[slots])?;
        let vis_bufs = vec![z_i32(&[m, spans, 2])?];
        Ok(Self {
            bufs,
            slot,
            coords_buf,
            fresh_buf,
            dest_buf,
            conv_seg_buf,
            conv_stop_buf,
            conv_prev_bank_buf,
            conv_prev_tape_buf,
            conv_next_bank_buf,
            vis_bufs,
        })
    }
}

struct BoundBarEntry {
    plan: NativeGraphPlan,
    bound: BoundNativeGraphPlan,
    entry_hidden: NativePort,
    sources_buf: NativePort,
}

type GroupedStepBuf = Crossing<Tensor>;
type GroupedMergedBuf = (Tensor, Tensor, Tensor);

struct BoundBarAttnAndMix {
    plan: NativeGraphPlan,
    bound: BoundNativeGraphPlan,
    sources_buf: NativePort,
    bank_buf: Option<NativePort>,
    cp_msgs_buf: Option<NativePort>,
    prelude_partial: Option<NativePort>,
    prev_ffn_out_buf: Option<NativePort>,
    prev_ffn_post_conv_window: Option<NativePort>,
    ffn_in_buf: Option<NativePort>,
    ffn_in_is_3d: bool,
    coords: NativePort,
    visible: NativePort,
    fresh: NativePort,
    destinations: NativePort,
    conv_segments: Option<NativePort>,
    conv_stop: Option<NativePort>,
    conv_prev_bank: Option<NativePort>,
    conv_prev_tape: Option<NativePort>,
    conv_next_bank: Option<NativePort>,
    history_key: NativePort,
    history_value: NativePort,
    key_conv_window: Option<NativePort>,
    value_conv_window: Option<NativePort>,
    attn_post_conv_window: Option<NativePort>,
    tap_buf: Option<NativePort>,
}

struct BoundBarExit {
    plan: NativeGraphPlan,
    bound: BoundNativeGraphPlan,
    sources_buf: NativePort,
    prev_ffn_out_buf: Option<NativePort>,
    conv_segments: Option<NativePort>,
    conv_stop: Option<NativePort>,
    conv_prev_bank: Option<NativePort>,
    conv_prev_tape: Option<NativePort>,
    conv_next_bank: Option<NativePort>,
    prev_ffn_post_conv_window: Option<NativePort>,
    exit_hidden: NativePort,
    tap_buf: Option<NativePort>,
}

/// Complete sealed & statically weight-bound family of `NativeGraphPlan`s for
/// one concrete `BarLayoutKey`.
pub(crate) struct BoundBarFamily {
    family: NativeGraphFamily,
    entry: BoundBarEntry,
    blocks: Vec<BoundBarAttnAndMix>,
    exit: BoundBarExit,
    _owned_tensors: Vec<Tensor>,
    /// 2-slot ring of reusable scratch buffers for decode (`m <= VERIFY_ROWS`),
    /// preventing any hazard when lookahead overlaps consecutive decode steps.
    decode_ring: Mutex<(usize, Vec<BarStepScratch>)>,
    /// Persistent reusable workspace slot for prefill (`m > VERIFY_ROWS`),
    /// avoiding repeated ~150-300 MB alloc/free churn across prefill chunks
    /// that would otherwise oscillate streamed expert-bank slot counts.
    prefill_slot: Mutex<Option<NativeGraphFamilySlot>>,
    /// Reusable `BarStepBuffers` pool for non-standard prefill chunk sizes
    /// (`m != prefill_rows`, e.g. prompt tail chunks).
    prefill_bufs_pool: Mutex<Vec<BarStepBuffers>>,
    /// Reusable grouped-FFN crossing scratch pool per `SplitLayer` (one per
    /// step in flight during `Front`).
    prefill_grouped_pool: Mutex<
        HashMap<crate::operators::routed::split::SplitLayer<u64>, Vec<GroupedStepBuf>>,
    >,
    /// Reusable grouped-FFN merged scratch pool per `SplitLayer` (one shared
    /// buffer per family, reused sequentially across steps in `Pass` -> `Back`).
    prefill_merged_pool: Mutex<
        HashMap<crate::operators::routed::split::SplitLayer<u64>, Vec<GroupedMergedBuf>>,
    >,
}

/// Uniquely identifies a physical weight tensor view for `BoundBlockAttnRes`'s
/// `fused_proj_cache`: backing allocation identity + element + view layout, plus
/// `WeightRole` whenever the tensor is a sub-view of a larger allocation (since
/// `Tensor` does not expose `byte_offset`).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(super) struct FusedWeightKey {
    storage_id: u64,
    element: Element,
    extents: Vec<u64>,
    strides: Vec<u64>,
    byte_len: u64,
    subview_role: Option<WeightRole>,
}

impl FusedWeightKey {
    fn of(role: WeightRole, tensor: &Tensor) -> Self {
        let storage_id = tensor.observe_storage().identity();
        let byte_len = tensor.byte_len();
        let subview_role = (byte_len < tensor.storage_bytes()).then_some(role);
        Self {
            storage_id,
            element: tensor.element(),
            extents: tensor.extents().to_vec(),
            strides: tensor.strides().to_vec(),
            byte_len,
            subview_role,
        }
    }
}

/// Helper for collecting static bindings (`(NativePort, Tensor)`) while building
/// a single `NativeGraph`.
struct StaticCollector<'a, 'b> {
    charges: &'a Charges,
    fused_proj_cache: &'a Mutex<HashMap<Vec<FusedWeightKey>, Tensor>>,
    owned: &'b mut Vec<Tensor>,
    pairs: Vec<(NativePort, Tensor)>,
}

impl<'a, 'b> StaticCollector<'a, 'b> {
    fn new(
        charges: &'a Charges,
        fused_proj_cache: &'a Mutex<HashMap<Vec<FusedWeightKey>, Tensor>>,
        owned: &'b mut Vec<Tensor>,
    ) -> Self {
        Self {
            charges,
            fused_proj_cache,
            owned,
            pairs: Vec::new(),
        }
    }

    fn bind_tensor(&mut self, graph: &mut NativeGraph, tensor: &Tensor) -> Result<NativePort, String> {
        let port = graph
            .port(tensor.element(), tensor.extents())
            .map_err(|e| e.to_string())?;
        self.pairs.push((port.clone(), tensor.clone()));
        Ok(port)
    }

    fn bind_i32(&mut self, graph: &mut NativeGraph, shape: &[u64], values: &[i32]) -> Result<NativePort, String> {
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let t = Tensor::from_host(self.charges, Element::i32(), shape, &bytes)
            .map_err(|e| e.to_string())?;
        let port = self.bind_tensor(graph, &t)?;
        self.owned.push(t);
        Ok(port)
    }

    fn bind_empty(&mut self, graph: &mut NativeGraph, elem: Element, shape: &[u64]) -> Result<NativePort, String> {
        let t = Tensor::from_host(self.charges, elem, shape, &[]).map_err(|e| e.to_string())?;
        let port = self.bind_tensor(graph, &t)?;
        self.owned.push(t);
        Ok(port)
    }

    fn bind_zeros(&mut self, graph: &mut NativeGraph, elem: Element, shape: &[u64]) -> Result<NativePort, String> {
        let t = Tensor::zeros(self.charges, elem, shape).map_err(|e| e.to_string())?;
        let port = self.bind_tensor(graph, &t)?;
        self.owned.push(t);
        Ok(port)
    }

    fn get_or_fuse_weights(
        &self,
        parts: &[(WeightRole, &Tensor)],
        elem: Element,
        shape: &[u64],
    ) -> Result<Tensor, String> {
        let cache_key: Vec<FusedWeightKey> = parts
            .iter()
            .map(|&(role, tensor)| FusedWeightKey::of(role, tensor))
            .collect();
        let mut cache = self
            .fused_proj_cache
            .lock()
            .expect("bar fused_proj_cache lock");
        if let Some(cached) = cache.get(&cache_key) {
            return Ok(cached.clone());
        }
        let mut raw = Vec::new();
        for &(_, t) in parts {
            raw.extend_from_slice(&t.read_to_host().map_err(|e| e.to_string())?);
        }
        let fused = Tensor::from_host(self.charges, elem, shape, &raw)
            .map_err(|e| e.to_string())?;
        cache.insert(cache_key, fused.clone());
        Ok(fused)
    }

    fn bind_fused_weights(
        &mut self,
        graph: &mut NativeGraph,
        parts: &[(WeightRole, &Tensor)],
        elem: Element,
        shape: &[u64],
    ) -> Result<NativePort, String> {
        let fused = self.get_or_fuse_weights(parts, elem, shape)?;
        self.bind_tensor(graph, &fused)
    }

    fn finish_bind(&self, plan: &NativeGraphPlan, label: &str) -> Result<BoundNativeGraphPlan, String> {
        let refs: Vec<(&NativePort, &Tensor)> = self.pairs.iter().map(|(p, t)| (p, t)).collect();
        plan.bind_static(self.charges, &refs)
            .map_err(|e| format!("BlockAttnRes static bind ({label}): {e}"))
    }
}

fn enqueue_project(
    prepared: &PreparedBlockAttnRes,
    graph: &mut NativeGraph,
    sc: &mut StaticCollector<'_, '_>,
    source: WorkflowTensorRef<'_>,
    source_elem: Element,
    weight: &BoundWeight,
    output: Element,
) -> Result<WorkflowTensor, String> {
    let ws = weight.scale_extent();
    let wb = weight.bound_extent();
    let [n, k] = match weight.extents() {
        [n, k] => [*n, *k],
        other => return Err(format!("project_rows weight expects 2D shape, got {other:?}")),
    };
    let key = (source_elem, weight.element(), output, k, n, ws, wb);
    let kernel = prepared
        .projects
        .get(&key)
        .ok_or_else(|| format!("missing project_rows kernel for {key:?}"))?;
    let w_port = sc.bind_tensor(graph, &weight.tensor)?;
    let scale_port = match &weight.scale {
        Some(s) => sc.bind_tensor(graph, s)?,
        None => sc.bind_empty(graph, Element::f32(), &[0])?,
    };
    let bound_port = match &weight.bound {
        Some(b) => sc.bind_tensor(graph, b)?,
        None => sc.bind_empty(graph, Element::f32(), &[0])?,
    };
    Ok(graph
        .enqueue(
            kernel,
            project_rows::WorkflowArgs {
                source,
                weight: w_port.tensor().into(),
                weight_scale: scale_port.tensor().into(),
                weight_bound: bound_port.tensor().into(),
            },
        )
        .map_err(|e| format!("enqueue project_rows: {e}"))?
        .value)
}

#[allow(clippy::too_many_arguments)]
fn enqueue_cross_pass_one(
    prepared: &PreparedBlockAttnRes,
    graph: &mut NativeGraph,
    sc: &mut StaticCollector<'_, '_>,
    m: u64,
    dest: &seismic::WorkflowTensorView,
    msg: &seismic::WorkflowTensorView,
    pre_cat: Option<WorkflowTensorRef<'_>>,
    scope: WeightScope,
    stream: BlockAttentionStream,
    weights: &HashMap<WeightRole, BoundWeight>,
    norm: Option<(&Tensor, f32)>,
) -> Result<WorkflowTensor, String> {
    let down = weights
        .get(&WeightRole {
            scope,
            kind: WeightKind::MessageController {
                stream,
                part: MessageControllerPart::Down,
            },
        })
        .ok_or("missing message controller down")?;
    let gain_w = weights
        .get(&WeightRole {
            scope,
            kind: WeightKind::MessageController {
                stream,
                part: MessageControllerPart::Gain,
            },
        })
        .ok_or("missing message controller gain")?;
    let forget_w = weights
        .get(&WeightRole {
            scope,
            kind: WeightKind::MessageController {
                stream,
                part: MessageControllerPart::Forget,
            },
        })
        .ok_or("missing message controller forget")?;

    let owned_cat: Option<WorkflowTensor>;
    let cat_ref: WorkflowTensorRef<'_> = match pre_cat {
        Some(c) => c,
        None => {
            owned_cat = Some(
                graph
                    .enqueue(
                        &prepared.rms_concat,
                        rms_concat::WorkflowArgs {
                            a: dest.into(),
                            b: msg.into(),
                            epsilon: (prepared.plan.norm_epsilon as f32).into(),
                        },
                    )
                    .map_err(|e| format!("enqueue rms_concat: {e}"))?
                    .value,
            );
            owned_cat.as_ref().unwrap().into()
        }
    };
    let low = enqueue_project(
        prepared,
        graph,
        sc,
        cat_ref,
        prepared.activation,
        down,
        prepared.activation,
    )?;

    let enqueue_gate = |graph: &mut NativeGraph,
                        sc: &mut StaticCollector<'_, '_>,
                        gain_ref: WorkflowTensorRef<'_>,
                        forget_ref: WorkflowTensorRef<'_>|
     -> Result<WorkflowTensor, String> {
        if let Some((nw, eps)) = norm {
            let nw_port = sc.bind_tensor(graph, nw)?;
            Ok(graph
                .enqueue(
                    &prepared.message_gate_mix_norm,
                    message_gate_mix_norm::WorkflowArgs {
                        dest: dest.into(),
                        msg: msg.into(),
                        gain: gain_ref,
                        forget: forget_ref,
                        norm: nw_port.tensor().into(),
                        epsilon: eps.into(),
                    },
                )
                .map_err(|e| format!("enqueue message_gate_mix_norm: {e}"))?
                .value)
        } else {
            Ok(graph
                .enqueue(
                    &prepared.message_gate_mix,
                    message_gate_mix::WorkflowArgs {
                        dest: dest.into(),
                        msg: msg.into(),
                        gain: gain_ref,
                        forget: forget_ref,
                    },
                )
                .map_err(|e| format!("enqueue message_gate_mix: {e}"))?
                .value)
        }
    };

    // On M = 1 decode, fuse `gain` + `forget` into a single `project_rows` launch
    // when both share element/K with no secondary scale/bound (cuts 128 launches/step).
    if m == 1
        && gain_w.element() == forget_w.element()
        && gain_w.scale.is_none()
        && forget_w.scale.is_none()
        && gain_w.bound.is_none()
        && forget_w.bound.is_none()
        && gain_w.tensor.extents().len() == 2
        && forget_w.tensor.extents().len() == 2
        && gain_w.tensor.extents()[1] == forget_w.tensor.extents()[1]
    {
        let n_g = gain_w.tensor.extents()[0];
        let n_f = forget_w.tensor.extents()[0];
        let k_dim = gain_w.tensor.extents()[1];
        let elem = gain_w.element();
        let fused_key = (prepared.activation, elem, Element::f32(), k_dim, n_g + n_f, 0, 0);
        if let Some(fused_k) = prepared.projects.get(&fused_key) {
            let gain_role = WeightRole {
                scope,
                kind: WeightKind::MessageController {
                    stream,
                    part: MessageControllerPart::Gain,
                },
            };
            let forget_role = WeightRole {
                scope,
                kind: WeightKind::MessageController {
                    stream,
                    part: MessageControllerPart::Forget,
                },
            };
            let w_port = sc.bind_fused_weights(
                graph,
                &[(gain_role, &gain_w.tensor), (forget_role, &forget_w.tensor)],
                elem,
                &[n_g + n_f, k_dim],
            )?;
            let empty_s = sc.bind_empty(graph, Element::f32(), &[0])?;
            let gf = graph
                .enqueue(
                    fused_k,
                    project_rows::WorkflowArgs {
                        source: (&low).into(),
                        weight: w_port.tensor().into(),
                        weight_scale: empty_s.tensor().into(),
                        weight_bound: empty_s.tensor().into(),
                    },
                )
                .map_err(|e| format!("enqueue fused gain+forget project_rows: {e}"))?
                .value;
            let flat = gf.reshape(&[n_g + n_f]);
            let g_v = flat.slice_leading(0, n_g).reshape(&[1, n_g]);
            let f_v = flat.slice_leading(n_g, n_g + n_f).reshape(&[1, n_f]);
            return enqueue_gate(graph, sc, (&g_v).into(), (&f_v).into());
        }
    }
    let gain = enqueue_project(
        prepared,
        graph,
        sc,
        (&low).into(),
        prepared.activation,
        gain_w,
        Element::f32(),
    )?;
    let forget = enqueue_project(
        prepared,
        graph,
        sc,
        (&low).into(),
        prepared.activation,
        forget_w,
        Element::f32(),
    )?;
    enqueue_gate(graph, sc, (&gain).into(), (&forget).into())
}

#[allow(clippy::too_many_arguments)]
fn enqueue_conv(
    prepared: &PreparedBlockAttnRes,
    graph: &mut NativeGraph,
    sc: &mut StaticCollector<'_, '_>,
    input: WorkflowTensorRef<'_>,
    input_is_f32: bool,
    conv_w: &Tensor,
    dyn_taps: Option<WorkflowTensorRef<'_>>,
    window_port: &mut NativePort,
    residual_scale: f32,
    slab_banks: u32,
    segments: &NativePort,
    stop: &NativePort,
    prev_bank: &NativePort,
    prev_tape: &NativePort,
    next_bank: &NativePort,
    m: u64,
    kernel_size: u64,
    output_f32: bool,
) -> Result<WorkflowTensor, String> {
    let cw_port = sc.bind_tensor(graph, conv_w)?;
    let empty_dyn_port;
    let dyn_ref = match dyn_taps {
        Some(d) => d,
        None => {
            empty_dyn_port = sc.bind_empty(graph, Element::f32(), &[m, 0, kernel_size])?;
            empty_dyn_port.tensor().into()
        }
    };
    let kernel = match (input_is_f32, output_f32) {
        (true, true) => &prepared.conv_ff,
        (false, true) => &prepared.conv_af,
        (false, false) => &prepared.conv_aa,
        (true, false) => return Err("unsupported f32->activation causal_short_conv".into()),
    };
    Ok(graph
        .enqueue(
            kernel,
            causal_short_conv::WorkflowArgs {
                input,
                convolution: cw_port.tensor().into(),
                dynamic_taps: dyn_ref,
                segments: segments.tensor().into(),
                stop: stop.tensor().into(),
                previous_bank: prev_bank.tensor().into(),
                previous_tape: prev_tape.tensor().into(),
                following_bank: next_bank.tensor().into(),
                window: window_port.tensor_mut().into(),
                residual_scale: residual_scale.into(),
                slab_banks: slab_banks.into(),
            },
        )
        .map_err(|e| format!("enqueue causal_short_conv: {e}"))?
        .value)
}

#[inline(never)]
#[allow(clippy::too_many_arguments)]
fn build_bound_entry(
    bound_bar: &BoundBlockAttnRes,
    decoder: &Decoder,
    key: &BarLayoutKey,
    owned_tensors: &mut Vec<Tensor>,
    all_plans: &mut Vec<NativeGraphPlan>,
) -> Result<BoundBarEntry, String> {
    let prepared = &bound_bar.prepared;
    let charges = &bound_bar.charges;
    let device = charges.device();
    let schedule = &prepared.schedule;
    let m = key.m;
    let h = decoder.hidden;

    let mut graph = device.native_graph();
    let sc = StaticCollector::new(charges, &bound_bar.fused_proj_cache, owned_tensors);
    let entry_hidden = graph
        .port(Element::f32(), &[m, h])
        .map_err(|e| e.to_string())?;
    let mut sources_buf = graph
        .port(Element::f32(), &[schedule.max_sources, m, h])
        .map_err(|e| e.to_string())?;
    let mut slot0 = sources_buf.tensor_mut().slice_leading(0, 1).reshape(&[m, h]);
    graph
        .enqueue(
            &prepared.f32_copy,
            feature_copy::WorkflowArgs {
                input: entry_hidden.tensor().into(),
                out: (&mut slot0).into(),
            },
        )
        .map_err(|e| format!("enqueue entry feature_copy: {e}"))?;
    let plan = graph.seal().map_err(|e| format!("seal entry: {e}"))?;
    let bound = sc.finish_bind(&plan, "entry")?;
    all_plans.push(plan.clone());
    Ok(BoundBarEntry {
        plan,
        bound,
        entry_hidden,
        sources_buf,
    })
}

/// Fold a completed block's FFN post-conv + `stream_join` + partial store into
/// the start of the following graph (`attn_and_mix[b + 1]` or `exit`), cutting
/// `NativeGraphPlan` launches per step from 146 to 98 on 48-layer models.
#[allow(clippy::too_many_arguments)]
fn enqueue_ffn_post_into_graph(
    prepared: &PreparedBlockAttnRes,
    weights: &HashMap<WeightRole, BoundWeight>,
    graph: &mut NativeGraph,
    sc: &mut StaticCollector<'_, '_>,
    sources_buf: &mut NativePort,
    ffn_sub: &magnitude_family_contracts::Sublayer,
    ffn_scope: WeightScope,
    sched: BarBlockSchedule,
    layer_state: &BlockAttnResLayerState,
    slab_banks: u32,
    conv_ports: Option<(&NativePort, &NativePort, &NativePort, &NativePort, &NativePort)>,
    m: u64,
    h: u64,
) -> Result<(NativePort, Option<NativePort>), String> {
    let ffn_out_buf = graph
        .port(Element::f32(), &[m, h])
        .map_err(|e| e.to_string())?;
    let (ffn_post_conv_window, ffn_out_t) = if let Some(post_conv) = &ffn_sub.post_conv {
        let (cs, cstop, cpb, cpt, cnb) =
            conv_ports.ok_or("missing conv ports for folded ffn post_conv")?;
        let cw = weights
            .get(&WeightRole {
                scope: ffn_scope,
                kind: WeightKind::PostShortConv,
            })
            .ok_or("missing ffn PostShortConv")?;
        let win_t = layer_state
            .ffn_post_conv_window
            .as_ref()
            .ok_or("missing ffn_post_conv_window")?;
        let mut win = graph
            .port(win_t.element(), win_t.extents())
            .map_err(|e| e.to_string())?;
        let out = enqueue_conv(
            prepared,
            graph,
            sc,
            ffn_out_buf.tensor().into(),
            true,
            &cw.tensor,
            None,
            &mut win,
            post_conv.residual_scale as f32,
            slab_banks,
            cs,
            cstop,
            cpb,
            cpt,
            cnb,
            m,
            post_conv.kernel_size,
            true,
        )?;
        (Some(win), Some(out))
    } else {
        (None, None)
    };

    let prev_p = sources_buf
        .tensor()
        .slice_leading(sched.ffn_partial_out_slot, sched.ffn_partial_out_slot + 1)
        .reshape(&[m, h]);
    let empty_inj = sc.bind_empty(graph, Element::f32(), &[m, 0])?;
    let y_arg: WorkflowTensorRef<'_> = match &ffn_out_t {
        Some(t) => t.into(),
        None => ffn_out_buf.tensor().into(),
    };
    let joined = graph
        .enqueue(
            &prepared.stream_join,
            stream_join::WorkflowArgs {
                residual: (&prev_p).into(),
                y: y_arg,
                inject: empty_inj.tensor().into(),
            },
        )
        .map_err(|e| format!("enqueue ffn stream_join: {e}"))?
        .value;
    let mut dst_slot = sources_buf
        .tensor_mut()
        .slice_leading(sched.ffn_partial_out_slot, sched.ffn_partial_out_slot + 1)
        .reshape(&[m, h]);
    graph
        .enqueue(
            &prepared.f32_copy,
            feature_copy::WorkflowArgs {
                input: (&joined).into(),
                out: (&mut dst_slot).into(),
            },
        )
        .map_err(|e| format!("enqueue ffn partial store: {e}"))?;
    Ok((ffn_out_buf, ffn_post_conv_window))
}

#[inline(never)]
#[allow(clippy::too_many_arguments)]
fn build_bound_exit(
    bound_bar: &BoundBlockAttnRes,
    decoder: &Decoder,
    key: &BarLayoutKey,
    tap_count: u64,
    last_ffn: Option<(
        &magnitude_family_contracts::Sublayer,
        WeightScope,
        BarBlockSchedule,
        &BlockAttnResLayerState,
        u32,
    )>,
    owned_tensors: &mut Vec<Tensor>,
    all_plans: &mut Vec<NativeGraphPlan>,
) -> Result<BoundBarExit, String> {
    let prepared = &bound_bar.prepared;
    let charges = &bound_bar.charges;
    let device = charges.device();
    let weights = &bound_bar.weights;
    let schedule = &prepared.schedule;
    let m = key.m;
    let h = decoder.hidden;
    let slots = key.slots;
    let scale_mix = 1.0f32 / (h as f32).sqrt();
    let norm_eps = prepared.plan.norm_epsilon as f32;
    let norm_values = i32::from(prepared.plan.normalize_values);

    let ExitNorm::BlockAttention { .. } = &decoder.exit.norm else {
        return Err("expected ExitNorm::BlockAttention".into());
    };
    let mut graph = device.native_graph();
    let mut sc = StaticCollector::new(charges, &bound_bar.fused_proj_cache, owned_tensors);
    let mut sources_buf = graph
        .port(Element::f32(), &[schedule.max_sources, m, h])
        .map_err(|e| e.to_string())?;
    let exit_hidden = graph
        .port(Element::f32(), &[1, m, h])
        .map_err(|e| e.to_string())?;

    let has_last_conv = last_ffn.is_some_and(|(sub, _, _, _, _)| sub.post_conv.is_some());
    let (conv_segments, conv_stop, conv_prev_bank, conv_prev_tape, conv_next_bank) =
        if has_last_conv {
            (
                Some(graph.port(Element::i32(), &[slots + 1, 2]).map_err(|e| e.to_string())?),
                Some(graph.port(Element::i32(), &[slots]).map_err(|e| e.to_string())?),
                Some(graph.port(Element::i32(), &[slots]).map_err(|e| e.to_string())?),
                Some(graph.port(Element::i32(), &[slots]).map_err(|e| e.to_string())?),
                Some(graph.port(Element::i32(), &[slots]).map_err(|e| e.to_string())?),
            )
        } else {
            (None, None, None, None, None)
        };
    let (prev_ffn_out_buf, prev_ffn_post_conv_window) =
        if let Some((ffn_sub, ffn_scope, ffn_sched, ffn_layer_state, ffn_slab_banks)) = last_ffn {
            let conv_refs = if has_last_conv {
                Some((
                    conv_segments.as_ref().unwrap(),
                    conv_stop.as_ref().unwrap(),
                    conv_prev_bank.as_ref().unwrap(),
                    conv_prev_tape.as_ref().unwrap(),
                    conv_next_bank.as_ref().unwrap(),
                ))
            } else {
                None
            };
            let (out_port, win_port) = enqueue_ffn_post_into_graph(
                prepared,
                weights,
                &mut graph,
                &mut sc,
                &mut sources_buf,
                ffn_sub,
                ffn_scope,
                ffn_sched,
                ffn_layer_state,
                ffn_slab_banks,
                conv_refs,
                m,
                h,
            )?;
            (Some(out_port), win_port)
        } else {
            (None, None)
        };

    let tap_pos = if key.has_taps {
        prepared.tap_points.iter().position(|tp| *tp == TapPoint::Exit)
    } else {
        None
    };
    let tap_buf = if let Some(tp_idx) = tap_pos {
        let tap_elem = key.tap_element.unwrap_or(prepared.activation);
        let tap_k = prepared
            .tap_rows
            .get(&tap_elem)
            .ok_or_else(|| format!("missing tap_rows kernel for {tap_elem:?}"))?;
        let mut tb = graph
            .port(tap_elem, &[m, tap_count * h])
            .map_err(|e| e.to_string())?;
        let idx_port = sc.bind_i32(&mut graph, &[1], &[tp_idx as i32])?;
        let last_slot = schedule.exit_sources_count - 1;
        let cur_partial = sources_buf
            .tensor()
            .slice_leading(last_slot, last_slot + 1)
            .reshape(&[m, h]);
        graph
            .enqueue(
                tap_k,
                tap_rows::WorkflowArgs {
                    residual: (&cur_partial).into(),
                    index: idx_port.tensor().into(),
                    taps: tb.tensor_mut().into(),
                },
            )
            .map_err(|e| format!("enqueue exit tap_rows: {e}"))?;
        Some(tb)
    } else {
        None
    };

    let out_q = weights
        .get(&WeightRole {
            scope: WeightScope::Target,
            kind: WeightKind::BlockAttentionQuery(BlockAttentionStream::Shared),
        })
        .ok_or("missing exit BlockAttentionQuery")?;
    let out_q2 = out_q.tensor.reshape(&[1, h]).map_err(|e| e.to_string())?;
    let out_q_port = sc.bind_tensor(&mut graph, &out_q2)?;
    let final_sources = sources_buf
        .tensor()
        .slice_leading(0, schedule.exit_sources_count);
    let mixed = graph
        .enqueue(
            &prepared.mix,
            block_attn_res_mix::WorkflowArgs {
                sources: (&final_sources).into(),
                queries: out_q_port.tensor().into(),
                epsilon: norm_eps.into(),
                scale: scale_mix.into(),
                normalize_values: norm_values.into(),
            },
        )
        .map_err(|e| format!("enqueue exit mix: {e}"))?
        .value;
    graph
        .store(&mixed, &exit_hidden)
        .map_err(|e| format!("store exit_hidden: {e}"))?;
    let plan = graph.seal().map_err(|e| format!("seal exit: {e}"))?;
    let bound = sc.finish_bind(&plan, "exit")?;
    all_plans.push(plan.clone());
    Ok(BoundBarExit {
        plan,
        bound,
        sources_buf,
        prev_ffn_out_buf,
        conv_segments,
        conv_stop,
        conv_prev_bank,
        conv_prev_tape,
        conv_next_bank,
        prev_ffn_post_conv_window,
        exit_hidden,
        tap_buf,
    })
}

#[allow(clippy::too_many_arguments)]
#[inline(never)]
fn enqueue_ffn_mix_port(
    prepared: &PreparedBlockAttnRes,
    weights: &HashMap<WeightRole, BoundWeight>,
    graph: &mut NativeGraph,
    sc: &mut StaticCollector<'_, '_>,
    sources_buf: &NativePort,
    bank_buf: Option<&NativePort>,
    cp_msgs: Option<&WorkflowTensor>,
    new_partial: WorkflowTensor,
    ffn_sub: &magnitude_family_contracts::Sublayer,
    ffn_scope: WeightScope,
    sched: BarBlockSchedule,
    m: u64,
    h: u64,
    norm_eps: f32,
    scale_mix: f32,
    norm_values: i32,
) -> Result<(NativePort, bool), String> {
    let InputNorm::BlockAttention(ffn_in) = &ffn_sub.input else {
        return Err("expected BlockAttention input on FFN sublayer".into());
    };
    let ffn_sources = sources_buf.tensor().slice_leading(0, sched.ffn_sources_count);
    let (mut x_ffn, mut x_ffn_is_3d, mut x_ffn_is_shared_partial): (WorkflowTensor, bool, bool) =
        match &ffn_in.queries {
            BlockAttentionQueries::Partial => (new_partial, false, true),
            BlockAttentionQueries::Single(_) => {
                let q_w = weights
                    .get(&WeightRole {
                        scope: ffn_scope,
                        kind: WeightKind::BlockAttentionQuery(BlockAttentionStream::Shared),
                    })
                    .ok_or("missing FFN res query")?;
                let q2 = q_w.tensor.reshape(&[1, h]).map_err(|e| e.to_string())?;
                let q_port = sc.bind_tensor(graph, &q2)?;
                let mixed = graph
                    .enqueue(
                        &prepared.mix,
                        block_attn_res_mix::WorkflowArgs {
                            sources: (&ffn_sources).into(),
                            queries: q_port.tensor().into(),
                            epsilon: norm_eps.into(),
                            scale: scale_mix.into(),
                            normalize_values: norm_values.into(),
                        },
                    )
                    .map_err(|e| format!("enqueue FFN mix: {e}"))?
                    .value;
                (mixed, true, false)
            }
            BlockAttentionQueries::SplitQkv { .. } => {
                return Err("SplitQkv invalid on FFN sublayer".into());
            }
        };
    if let CrossPassInput::Single(_) = &ffn_in.cross_pass {
        let owned_msg: Option<WorkflowTensor>;
        let msg_v = if let (Some(cp_all), Some(off)) = (cp_msgs, sched.ffn_cp_msg_offset) {
            cp_all.slice_leading(off, off + 1).reshape(&[m, h])
        } else {
            let b_view = bank_buf
                .ok_or("missing bank_buf for FFN cross-pass")?
                .tensor()
                .slice_leading(0, sched.bank_count);
            let cp_q = weights
                .get(&WeightRole {
                    scope: ffn_scope,
                    kind: WeightKind::CrossPassQuery(BlockAttentionStream::Shared),
                })
                .ok_or("missing FFN cross-pass query")?;
            let cp_q2 = cp_q.tensor.reshape(&[1, h]).map_err(|e| e.to_string())?;
            let cp_q_port = sc.bind_tensor(graph, &cp_q2)?;
            owned_msg = Some(
                graph
                    .enqueue(
                        &prepared.mix,
                        block_attn_res_mix::WorkflowArgs {
                            sources: (&b_view).into(),
                            queries: cp_q_port.tensor().into(),
                            epsilon: norm_eps.into(),
                            scale: scale_mix.into(),
                            normalize_values: norm_values.into(),
                        },
                    )
                    .map_err(|e| format!("enqueue FFN cross-pass mix: {e}"))?
                    .value,
            );
            owned_msg.as_ref().unwrap().reshape(&[m, h])
        };
        let dest_v = x_ffn.reshape(&[m, h]);
        x_ffn = enqueue_cross_pass_one(
            prepared,
            graph,
            sc,
            m,
            &dest_v,
            &msg_v,
            None,
            ffn_scope,
            BlockAttentionStream::Shared,
            weights,
            None,
        )?;
        x_ffn_is_3d = false;
        x_ffn_is_shared_partial = false;
    }
    if !x_ffn_is_shared_partial {
        let ffn_in_port = if x_ffn_is_3d {
            graph
                .port(Element::f32(), &[1, m, h])
                .map_err(|e| e.to_string())?
        } else {
            graph
                .port(Element::f32(), &[m, h])
                .map_err(|e| e.to_string())?
        };
        graph
            .store(&x_ffn, &ffn_in_port)
            .map_err(|e| format!("store ffn_in: {e}"))?;
        Ok((ffn_in_port, x_ffn_is_3d))
    } else {
        let mut ffn_in_port = graph
            .port(Element::f32(), &[m, h])
            .map_err(|e| e.to_string())?;
        let x_ffn_2d = x_ffn.reshape(&[m, h]);
        graph
            .enqueue(
                &prepared.f32_copy,
                feature_copy::WorkflowArgs {
                    input: (&x_ffn_2d).into(),
                    out: ffn_in_port.tensor_mut().into(),
                },
            )
            .map_err(|e| format!("enqueue ffn_in store: {e}"))?;
        Ok((ffn_in_port, false))
    }
}

#[allow(clippy::too_many_arguments)]
#[inline(never)]
fn enqueue_attn_mix_and_norm(
    prepared: &PreparedBlockAttnRes,
    weights: &HashMap<WeightRole, BoundWeight>,
    graph: &mut NativeGraph,
    sc: &mut StaticCollector<'_, '_>,
    sources_buf: &NativePort,
    bank_buf: Option<&NativePort>,
    cp_msgs: Option<&WorkflowTensor>,
    attn_in: &magnitude_family_contracts::BlockAttentionInput,
    attn_scope: WeightScope,
    sched: BarBlockSchedule,
    m: u64,
    h: u64,
    norm_eps: f32,
    scale_mix: f32,
    norm_values: i32,
) -> Result<
    (
        seismic::WorkflowTensorView,
        seismic::WorkflowTensorView,
        seismic::WorkflowTensorView,
        bool,
    ),
    String,
> {
    // Mix residual sources -> (x_q, x_k, x_v).
    let attn_sources = sources_buf.tensor().slice_leading(0, sched.attn_sources_count);
    let partial_in = sources_buf
        .tensor()
        .slice_leading(sched.attn_partial_in_slot, sched.attn_partial_in_slot + 1)
        .reshape(&[m, h]);

    let (mut x_q, mut x_k, mut x_v, split_streams): (
        WorkflowTensor,
        WorkflowTensor,
        WorkflowTensor,
        bool,
    ) = match &attn_in.queries {
        BlockAttentionQueries::Partial => {
            let z = sc.bind_zeros(graph, Element::f32(), &[m, h])?;
            let empty_inj = sc.bind_empty(graph, Element::f32(), &[m, 0])?;
            let p = graph
                .enqueue(
                    &prepared.stream_join,
                    stream_join::WorkflowArgs {
                        residual: (&partial_in).into(),
                        y: z.tensor().into(),
                        inject: empty_inj.tensor().into(),
                    },
                )
                .map_err(|e| e.to_string())?
                .value;
            (p.clone(), p.clone(), p, false)
        }
        BlockAttentionQueries::Single(_) => {
            let q_w = weights
                .get(&WeightRole {
                    scope: attn_scope,
                    kind: WeightKind::BlockAttentionQuery(BlockAttentionStream::Shared),
                })
                .ok_or("missing shared attn res query")?;
            let q_2d = q_w.tensor.reshape(&[1, h]).map_err(|e| e.to_string())?;
            let q_port = sc.bind_tensor(graph, &q_2d)?;
            let mixed = graph
                .enqueue(
                    &prepared.mix,
                    block_attn_res_mix::WorkflowArgs {
                        sources: (&attn_sources).into(),
                        queries: q_port.tensor().into(),
                        epsilon: norm_eps.into(),
                        scale: scale_mix.into(),
                        normalize_values: norm_values.into(),
                    },
                )
                .map_err(|e| format!("enqueue attn single mix: {e}"))?
                .value;
            (mixed.clone(), mixed.clone(), mixed, false)
        }
        BlockAttentionQueries::SplitQkv { .. } => {
            let qq_role = WeightRole {
                scope: attn_scope,
                kind: WeightKind::BlockAttentionQuery(BlockAttentionStream::Query),
            };
            let qk_role = WeightRole {
                scope: attn_scope,
                kind: WeightKind::BlockAttentionQuery(BlockAttentionStream::Key),
            };
            let qv_role = WeightRole {
                scope: attn_scope,
                kind: WeightKind::BlockAttentionQuery(BlockAttentionStream::Value),
            };
            let qq = weights.get(&qq_role).ok_or("missing Q attn res query")?;
            let qk = weights.get(&qk_role).ok_or("missing K attn res query")?;
            let qv = weights.get(&qv_role).ok_or("missing V attn res query")?;
            if decodes(m) {
                let q3_port = sc.bind_fused_weights(
                    graph,
                    &[
                        (qq_role, &qq.tensor),
                        (qk_role, &qk.tensor),
                        (qv_role, &qv.tensor),
                    ],
                    Element::f32(),
                    &[3, h],
                )?;
                let mixed3 = graph
                    .enqueue(
                        &prepared.mix,
                        block_attn_res_mix::WorkflowArgs {
                            sources: (&attn_sources).into(),
                            queries: q3_port.tensor().into(),
                            epsilon: norm_eps.into(),
                            scale: scale_mix.into(),
                            normalize_values: norm_values.into(),
                        },
                    )
                    .map_err(|e| format!("enqueue attn splitQKV mix: {e}"))?
                    .value;
                (mixed3.clone(), mixed3.clone(), mixed3, true)
            } else {
                let mut mix_one = |qw: &BoundWeight| -> Result<WorkflowTensor, String> {
                    let q2 = qw.tensor.reshape(&[1, h]).map_err(|e| e.to_string())?;
                    let qp = sc.bind_tensor(graph, &q2)?;
                    Ok(graph
                        .enqueue(
                            &prepared.mix,
                            block_attn_res_mix::WorkflowArgs {
                                sources: (&attn_sources).into(),
                                queries: qp.tensor().into(),
                                epsilon: norm_eps.into(),
                                scale: scale_mix.into(),
                                normalize_values: norm_values.into(),
                            },
                        )
                        .map_err(|e| format!("enqueue attn splitQKV mix: {e}"))?
                        .value)
                };
                let mq = mix_one(qq)?;
                let mk = mix_one(qk)?;
                let mv = mix_one(qv)?;
                (mq, mk, mv, true)
            }
        }
    };

    let view_stream = |t: &WorkflowTensor, idx: u64, is_3: bool| -> seismic::WorkflowTensorView {
        if is_3 {
            t.slice_leading(idx, idx + 1).reshape(&[m, h])
        } else {
            t.reshape(&[m, h])
        }
    };
    let mut is_q3 =
        decodes(m) && matches!(attn_in.queries, BlockAttentionQueries::SplitQkv { .. });

    // Optional cross-pass bank injection. When the gated output feeds directly
    // into its own `InputNorm`, fuse `message_gate_mix` + `InputNorm` via
    // `message_gate_mix_norm` (eliminating the F32 round-trip and separate
    // `readout_features_rows` launch).
    match &attn_in.cross_pass {
        CrossPassInput::None => {}
        CrossPassInput::Single(_) => {
            let owned_msg: Option<WorkflowTensor>;
            let msg_v = if let (Some(cp_all), Some(off)) = (cp_msgs, sched.attn_cp_msg_offset) {
                cp_all.slice_leading(off, off + 1).reshape(&[m, h])
            } else {
                let b_view = bank_buf
                    .ok_or("missing bank_buf")?
                    .tensor()
                    .slice_leading(0, sched.bank_count);
                let cp_q = weights
                    .get(&WeightRole {
                        scope: attn_scope,
                        kind: WeightKind::CrossPassQuery(BlockAttentionStream::Shared),
                    })
                    .ok_or("missing shared cross-pass query")?;
                let cp_q2 = cp_q.tensor.reshape(&[1, h]).map_err(|e| e.to_string())?;
                let cp_q_port = sc.bind_tensor(graph, &cp_q2)?;
                owned_msg = Some(
                    graph
                        .enqueue(
                            &prepared.mix,
                            block_attn_res_mix::WorkflowArgs {
                                sources: (&b_view).into(),
                                queries: cp_q_port.tensor().into(),
                                epsilon: norm_eps.into(),
                                scale: scale_mix.into(),
                                normalize_values: norm_values.into(),
                            },
                        )
                        .map_err(|e| format!("enqueue attn cross-pass mix: {e}"))?
                        .value,
                );
                owned_msg.as_ref().unwrap().reshape(&[m, h])
            };
            let dest_v = view_stream(&x_q, 0, is_q3);
            if let BlockAttentionNorm::Shared(norm) = &attn_in.norm {
                let nw = weights
                    .get(&WeightRole {
                        scope: attn_scope,
                        kind: WeightKind::InputNorm,
                    })
                    .ok_or("missing shared attn InputNorm")?;
                let out_norm = enqueue_cross_pass_one(
                    prepared,
                    graph,
                    sc,
                    m,
                    &dest_v,
                    &msg_v,
                    None,
                    attn_scope,
                    BlockAttentionStream::Shared,
                    weights,
                    Some((&nw.tensor, norm.epsilon as f32)),
                )?
                .reshape(&[m, h]);
                return Ok((out_norm.clone(), out_norm.clone(), out_norm, split_streams));
            }
            let out = enqueue_cross_pass_one(
                prepared,
                graph,
                sc,
                m,
                &dest_v,
                &msg_v,
                None,
                attn_scope,
                BlockAttentionStream::Shared,
                weights,
                None,
            )?;
            x_q = out.clone();
            x_k = out.clone();
            x_v = out;
            is_q3 = false;
        }
        CrossPassInput::SplitQkv { .. } => {
            let (norm_q, norm_k, norm_v): ((&Tensor, f32), (&Tensor, f32), (&Tensor, f32)) =
                match &attn_in.norm {
                    BlockAttentionNorm::Shared(norm) => {
                        let nw = weights
                            .get(&WeightRole {
                                scope: attn_scope,
                                kind: WeightKind::InputNorm,
                            })
                            .ok_or("missing shared attn InputNorm")?;
                        let eps = norm.epsilon as f32;
                        ((&nw.tensor, eps), (&nw.tensor, eps), (&nw.tensor, eps))
                    }
                    BlockAttentionNorm::SplitQkv { query, key, value } => {
                        let nw_q = weights
                            .get(&WeightRole {
                                scope: attn_scope,
                                kind: WeightKind::BlockAttentionInputNorm(
                                    BlockAttentionStream::Query,
                                ),
                            })
                            .ok_or("missing split Q InputNorm")?;
                        let nw_k = weights
                            .get(&WeightRole {
                                scope: attn_scope,
                                kind: WeightKind::BlockAttentionInputNorm(
                                    BlockAttentionStream::Key,
                                ),
                            })
                            .ok_or("missing split K InputNorm")?;
                        let nw_v = weights
                            .get(&WeightRole {
                                scope: attn_scope,
                                kind: WeightKind::BlockAttentionInputNorm(
                                    BlockAttentionStream::Value,
                                ),
                            })
                            .ok_or("missing split V InputNorm")?;
                        (
                            (&nw_q.tensor, query.epsilon as f32),
                            (&nw_k.tensor, key.epsilon as f32),
                            (&nw_v.tensor, value.epsilon as f32),
                        )
                    }
                };
            if let (Some(cp_all), Some(off)) = (cp_msgs, sched.attn_cp_msg_offset) {
                let dq = view_stream(&x_q, 0, is_q3);
                let dk = view_stream(&x_k, 1, is_q3);
                let dv = view_stream(&x_v, 2, is_q3);
                let mq = cp_all.slice_leading(off, off + 1).reshape(&[m, h]);
                let mk = cp_all.slice_leading(off + 1, off + 2).reshape(&[m, h]);
                let mv = cp_all.slice_leading(off + 2, off + 3).reshape(&[m, h]);
                let cat3_opt = if is_q3 {
                    let d3 = x_q.reshape(&[3 * m, h]);
                    let m3 = cp_all.slice_leading(off, off + 3).reshape(&[3 * m, h]);
                    Some(
                        graph
                            .enqueue(
                                &prepared.rms_concat,
                                rms_concat::WorkflowArgs {
                                    a: (&d3).into(),
                                    b: (&m3).into(),
                                    epsilon: (prepared.plan.norm_epsilon as f32).into(),
                                },
                            )
                            .map_err(|e| format!("enqueue batched rms_concat: {e}"))?
                            .value
                            .reshape(&[3, m, 2 * h]),
                    )
                } else {
                    None
                };
                let cq_v = cat3_opt
                    .as_ref()
                    .map(|c| c.slice_leading(0, 1).reshape(&[m, 2 * h]));
                let ck_v = cat3_opt
                    .as_ref()
                    .map(|c| c.slice_leading(1, 2).reshape(&[m, 2 * h]));
                let cv_v = cat3_opt
                    .as_ref()
                    .map(|c| c.slice_leading(2, 3).reshape(&[m, 2 * h]));
                let xq_norm = enqueue_cross_pass_one(
                    prepared,
                    graph,
                    sc,
                    m,
                    &dq,
                    &mq,
                    cq_v.as_ref().map(|v| v.into()),
                    attn_scope,
                    BlockAttentionStream::Query,
                    weights,
                    Some(norm_q),
                )?
                .reshape(&[m, h]);
                let xk_norm = enqueue_cross_pass_one(
                    prepared,
                    graph,
                    sc,
                    m,
                    &dk,
                    &mk,
                    ck_v.as_ref().map(|v| v.into()),
                    attn_scope,
                    BlockAttentionStream::Key,
                    weights,
                    Some(norm_k),
                )?
                .reshape(&[m, h]);
                let xv_norm = enqueue_cross_pass_one(
                    prepared,
                    graph,
                    sc,
                    m,
                    &dv,
                    &mv,
                    cv_v.as_ref().map(|v| v.into()),
                    attn_scope,
                    BlockAttentionStream::Value,
                    weights,
                    Some(norm_v),
                )?
                .reshape(&[m, h]);
                return Ok((xq_norm, xk_norm, xv_norm, true));
            } else {
                let b_view = bank_buf
                    .ok_or("missing bank_buf")?
                    .tensor()
                    .slice_leading(0, sched.bank_count);
                let cpq_role = WeightRole {
                    scope: attn_scope,
                    kind: WeightKind::CrossPassQuery(BlockAttentionStream::Query),
                };
                let cpk_role = WeightRole {
                    scope: attn_scope,
                    kind: WeightKind::CrossPassQuery(BlockAttentionStream::Key),
                };
                let cpv_role = WeightRole {
                    scope: attn_scope,
                    kind: WeightKind::CrossPassQuery(BlockAttentionStream::Value),
                };
                let cpq = weights.get(&cpq_role).ok_or("missing Q cross-pass query")?;
                let cpk = weights.get(&cpk_role).ok_or("missing K cross-pass query")?;
                let cpv = weights.get(&cpv_role).ok_or("missing V cross-pass query")?;
                if decodes(m) {
                    let cp3_port = sc.bind_fused_weights(
                        graph,
                        &[
                            (cpq_role, &cpq.tensor),
                            (cpk_role, &cpk.tensor),
                            (cpv_role, &cpv.tensor),
                        ],
                        Element::f32(),
                        &[3, h],
                    )?;
                    let msg3 = graph
                        .enqueue(
                            &prepared.mix,
                            block_attn_res_mix::WorkflowArgs {
                                sources: (&b_view).into(),
                                queries: cp3_port.tensor().into(),
                                epsilon: norm_eps.into(),
                                scale: scale_mix.into(),
                                normalize_values: norm_values.into(),
                            },
                        )
                        .map_err(|e| format!("enqueue attn splitQKV cross-pass mix: {e}"))?
                        .value;
                    let dq = view_stream(&x_q, 0, is_q3);
                    let dk = view_stream(&x_k, 1, is_q3);
                    let dv = view_stream(&x_v, 2, is_q3);
                    let mq = msg3.slice_leading(0, 1).reshape(&[m, h]);
                    let mk = msg3.slice_leading(1, 2).reshape(&[m, h]);
                    let mv = msg3.slice_leading(2, 3).reshape(&[m, h]);
                    let cat3_opt = if is_q3 {
                        let d3 = x_q.reshape(&[3 * m, h]);
                        let m3 = msg3.reshape(&[3 * m, h]);
                        Some(
                            graph
                                .enqueue(
                                    &prepared.rms_concat,
                                    rms_concat::WorkflowArgs {
                                        a: (&d3).into(),
                                        b: (&m3).into(),
                                        epsilon: (prepared.plan.norm_epsilon as f32).into(),
                                    },
                                )
                                .map_err(|e| format!("enqueue batched rms_concat: {e}"))?
                                .value
                                .reshape(&[3, m, 2 * h]),
                        )
                    } else {
                        None
                    };
                    let cq_v = cat3_opt
                        .as_ref()
                        .map(|c| c.slice_leading(0, 1).reshape(&[m, 2 * h]));
                    let ck_v = cat3_opt
                        .as_ref()
                        .map(|c| c.slice_leading(1, 2).reshape(&[m, 2 * h]));
                    let cv_v = cat3_opt
                        .as_ref()
                        .map(|c| c.slice_leading(2, 3).reshape(&[m, 2 * h]));
                    let xq_norm = enqueue_cross_pass_one(
                        prepared,
                        graph,
                        sc,
                        m,
                        &dq,
                        &mq,
                        cq_v.as_ref().map(|v| v.into()),
                        attn_scope,
                        BlockAttentionStream::Query,
                        weights,
                        Some(norm_q),
                    )?
                    .reshape(&[m, h]);
                    let xk_norm = enqueue_cross_pass_one(
                        prepared,
                        graph,
                        sc,
                        m,
                        &dk,
                        &mk,
                        ck_v.as_ref().map(|v| v.into()),
                        attn_scope,
                        BlockAttentionStream::Key,
                        weights,
                        Some(norm_k),
                    )?
                    .reshape(&[m, h]);
                    let xv_norm = enqueue_cross_pass_one(
                        prepared,
                        graph,
                        sc,
                        m,
                        &dv,
                        &mv,
                        cv_v.as_ref().map(|v| v.into()),
                        attn_scope,
                        BlockAttentionStream::Value,
                        weights,
                        Some(norm_v),
                    )?
                    .reshape(&[m, h]);
                    return Ok((xq_norm, xk_norm, xv_norm, true));
                } else {
                    let mut one_cp = |dest_t: &WorkflowTensor,
                                      idx: u64,
                                      cp_w: &BoundWeight,
                                      stream: BlockAttentionStream,
                                      stream_norm: (&Tensor, f32)|
                     -> Result<WorkflowTensor, String> {
                        let q2 = cp_w.tensor.reshape(&[1, h]).map_err(|e| e.to_string())?;
                        let qp = sc.bind_tensor(graph, &q2)?;
                        let msg = graph
                            .enqueue(
                                &prepared.mix,
                                block_attn_res_mix::WorkflowArgs {
                                    sources: (&b_view).into(),
                                    queries: qp.tensor().into(),
                                    epsilon: norm_eps.into(),
                                    scale: scale_mix.into(),
                                    normalize_values: norm_values.into(),
                                },
                            )
                            .map_err(|e| format!("enqueue attn cross-pass mix: {e}"))?
                            .value;
                        let dv = view_stream(dest_t, idx, is_q3);
                        let mv = msg.reshape(&[m, h]);
                        enqueue_cross_pass_one(
                            prepared,
                            graph,
                            sc,
                            m,
                            &dv,
                            &mv,
                            None,
                            attn_scope,
                            stream,
                            weights,
                            Some(stream_norm),
                        )
                    };
                    let xq_norm =
                        one_cp(&x_q, 0, cpq, BlockAttentionStream::Query, norm_q)?.reshape(&[m, h]);
                    let xk_norm =
                        one_cp(&x_k, 1, cpk, BlockAttentionStream::Key, norm_k)?.reshape(&[m, h]);
                    let xv_norm =
                        one_cp(&x_v, 2, cpv, BlockAttentionStream::Value, norm_v)?.reshape(&[m, h]);
                    return Ok((xq_norm, xk_norm, xv_norm, true));
                }
            }
        }
    }

    // Input RMSNorm(s).
    let id_rows_port = sc.bind_i32(graph, &[m], &(0..m as i32).collect::<Vec<_>>())?;
    let norm_view = |graph: &mut NativeGraph,
                     sc: &mut StaticCollector<'_, '_>,
                     t: &WorkflowTensor,
                     idx: u64,
                     is_3: bool,
                     nw: &Tensor,
                     eps: f32|
     -> Result<WorkflowTensor, String> {
        let x3 = if is_3 {
            t.slice_leading(idx, idx + 1).reshape(&[m, 1, h])
        } else {
            t.reshape(&[m, 1, h])
        };
        let nw_port = sc.bind_tensor(graph, nw)?;
        Ok(graph
            .enqueue(
                &prepared.norm_rows,
                readout_features_rows::WorkflowArgs {
                    hidden: (&x3).into(),
                    norm: nw_port.tensor().into(),
                    out_rows: id_rows_port.tensor().into(),
                    epsilon: eps.into(),
                },
            )
            .map_err(|e| format!("enqueue readout_features_rows: {e}"))?
            .value)
    };

    let (xq_norm, xk_norm, xv_norm) = match &attn_in.norm {
        BlockAttentionNorm::Shared(norm) => {
            let nw = weights
                .get(&WeightRole {
                    scope: attn_scope,
                    kind: WeightKind::InputNorm,
                })
                .ok_or("missing shared attn InputNorm")?;
            let eps = norm.epsilon as f32;
            if split_streams && is_q3 {
                let x3m = x_q.reshape(&[3 * m, 1, h]);
                let nw_port = sc.bind_tensor(graph, &nw.tensor)?;
                let id3m = sc.bind_i32(
                    graph,
                    &[3 * m],
                    &(0..(3 * m) as i32).collect::<Vec<_>>(),
                )?;
                let normed3 = graph
                    .enqueue(
                        &prepared.norm_rows,
                        readout_features_rows::WorkflowArgs {
                            hidden: (&x3m).into(),
                            norm: nw_port.tensor().into(),
                            out_rows: id3m.tensor().into(),
                            epsilon: eps.into(),
                        },
                    )
                    .map_err(|e| format!("enqueue batched readout_features_rows: {e}"))?
                    .value
                    .reshape(&[3, m, h]);
                (
                    normed3.slice_leading(0, 1).reshape(&[m, h]),
                    normed3.slice_leading(1, 2).reshape(&[m, h]),
                    normed3.slice_leading(2, 3).reshape(&[m, h]),
                )
            } else {
                let nq = norm_view(graph, sc, &x_q, 0, is_q3, &nw.tensor, eps)?;
                if split_streams {
                    let nk = norm_view(graph, sc, &x_k, 1, is_q3, &nw.tensor, eps)?;
                    let nv = norm_view(graph, sc, &x_v, 2, is_q3, &nw.tensor, eps)?;
                    (nq.reshape(&[m, h]), nk.reshape(&[m, h]), nv.reshape(&[m, h]))
                } else {
                    let nqv = nq.reshape(&[m, h]);
                    (nqv.clone(), nqv.clone(), nqv)
                }
            }
        }
        BlockAttentionNorm::SplitQkv { query, key, value } => {
            let nw_q = weights
                .get(&WeightRole {
                    scope: attn_scope,
                    kind: WeightKind::BlockAttentionInputNorm(BlockAttentionStream::Query),
                })
                .ok_or("missing split Q InputNorm")?;
            let nw_k = weights
                .get(&WeightRole {
                    scope: attn_scope,
                    kind: WeightKind::BlockAttentionInputNorm(BlockAttentionStream::Key),
                })
                .ok_or("missing split K InputNorm")?;
            let nw_v = weights
                .get(&WeightRole {
                    scope: attn_scope,
                    kind: WeightKind::BlockAttentionInputNorm(BlockAttentionStream::Value),
                })
                .ok_or("missing split V InputNorm")?;
            (
                norm_view(graph, sc, &x_q, 0, is_q3, &nw_q.tensor, query.epsilon as f32)?
                    .reshape(&[m, h]),
                norm_view(graph, sc, &x_k, 1, is_q3, &nw_k.tensor, key.epsilon as f32)?
                    .reshape(&[m, h]),
                norm_view(graph, sc, &x_v, 2, is_q3, &nw_v.tensor, value.epsilon as f32)?
                    .reshape(&[m, h]),
            )
        }
    };
    Ok((xq_norm, xk_norm, xv_norm, split_streams))
}

#[allow(clippy::too_many_arguments)]
#[inline(never)]
fn enqueue_pass_transition(
    prepared: &PreparedBlockAttnRes,
    weights: &HashMap<WeightRole, BoundWeight>,
    graph: &mut NativeGraph,
    sc: &mut StaticCollector<'_, '_>,
    sources_buf: &mut NativePort,
    bank_buf: &mut Option<NativePort>,
    cp_msgs_buf: &mut Option<NativePort>,
    prelude_partial: &mut Option<NativePort>,
    b_idx: usize,
    sched: BarBlockSchedule,
    batched_cp: bool,
    m: u64,
    h: u64,
    norm_eps: f32,
    scale_mix: f32,
    norm_values: i32,
) -> Result<Option<WorkflowTensor>, String> {
    match sched.transition {
        None => Ok(None),
        Some(BlockTransition::SnapshotPrelude) => {
            let p_slot = sched.prelude_slot.unwrap();
            let src = sources_buf
                .tensor()
                .slice_leading(p_slot, p_slot + 1)
                .reshape(&[m, h]);
            let dst = prelude_partial.as_mut().unwrap();
            graph
                .enqueue(
                    &prepared.f32_copy,
                    feature_copy::WorkflowArgs {
                        input: (&src).into(),
                        out: dst.tensor_mut().into(),
                    },
                )
                .map_err(|e| format!("enqueue SnapshotPrelude copy: {e}"))?;
            Ok(None)
        }
        Some(BlockTransition::ResetToPreludeAndBank) => {
            let (lo, hi) = sched.bank_src_range.unwrap();
            let pass_cp_all_msgs = if batched_cp {
                let queries_list = &prepared.schedule.pass_cp_queries[b_idx];
                if queries_list.is_empty() {
                    None
                } else {
                    let q_count = queries_list.len() as u64;
                    let mut parts: Vec<(WeightRole, &Tensor)> =
                        Vec::with_capacity(queries_list.len());
                    for &(scope, stream) in queries_list {
                        let role = WeightRole {
                            scope,
                            kind: WeightKind::CrossPassQuery(stream),
                        };
                        let w = weights
                            .get(&role)
                            .ok_or_else(|| format!("missing cross-pass query {role:?}"))?;
                        parts.push((role, &w.tensor));
                    }
                    let q_all_port =
                        sc.bind_fused_weights(graph, &parts, Element::f32(), &[q_count, h])?;
                    let src_bank = sources_buf.tensor().slice_leading(lo, hi);
                    let all_msgs = graph
                        .enqueue(
                            &prepared.mix,
                            block_attn_res_mix::WorkflowArgs {
                                sources: (&src_bank).into(),
                                queries: q_all_port.tensor().into(),
                                epsilon: norm_eps.into(),
                                scale: scale_mix.into(),
                                normalize_values: norm_values.into(),
                            },
                        )
                        .map_err(|e| format!("enqueue pass-entry batched cross-pass mix: {e}"))?
                        .value;
                    let cp_port = cp_msgs_buf
                        .as_mut()
                        .ok_or("missing cp_msgs_buf for batched ResetToPreludeAndBank")?;
                    if q_count == prepared.schedule.max_pass_cp_queries {
                        graph
                            .store(&all_msgs, cp_port)
                            .map_err(|e| format!("store cp_msgs_buf: {e}"))?;
                    } else {
                        let src_flat = all_msgs.reshape(&[q_count * m, h]);
                        let mut dst_flat = cp_port
                            .tensor_mut()
                            .slice_leading(0, q_count)
                            .reshape(&[q_count * m, h]);
                        graph
                            .enqueue(
                                &prepared.f32_copy,
                                feature_copy::WorkflowArgs {
                                    input: (&src_flat).into(),
                                    out: (&mut dst_flat).into(),
                                },
                            )
                            .map_err(|e| format!("enqueue cp_msgs_buf store copy: {e}"))?;
                    }
                    Some(all_msgs)
                }
            } else {
                let bank_len = hi - lo;
                let src_bank = sources_buf
                    .tensor()
                    .slice_leading(lo, hi)
                    .reshape(&[bank_len * m, h]);
                let mut dst_bank = bank_buf
                    .as_mut()
                    .unwrap()
                    .tensor_mut()
                    .slice_leading(0, bank_len)
                    .reshape(&[bank_len * m, h]);
                graph
                    .enqueue(
                        &prepared.f32_copy,
                        feature_copy::WorkflowArgs {
                            input: (&src_bank).into(),
                            out: (&mut dst_bank).into(),
                        },
                    )
                    .map_err(|e| format!("enqueue ResetToPrelude bank copy: {e}"))?;
                None
            };

            let p_slot = sched.prelude_slot.unwrap();
            let mut dst_pre = sources_buf
                .tensor_mut()
                .slice_leading(p_slot, p_slot + 1)
                .reshape(&[m, h]);
            let pre_ref = prelude_partial.as_ref().unwrap();
            graph
                .enqueue(
                    &prepared.f32_copy,
                    feature_copy::WorkflowArgs {
                        input: pre_ref.tensor().into(),
                        out: (&mut dst_pre).into(),
                    },
                )
                .map_err(|e| format!("enqueue ResetToPrelude restore copy: {e}"))?;
            Ok(pass_cp_all_msgs)
        }
    }
}

#[allow(clippy::too_many_arguments)]
#[inline(never)]
fn build_bound_attn_and_mix(
    bound_bar: &BoundBlockAttnRes,
    decoder: &Decoder,
    key: &BarLayoutKey,
    b_idx: usize,
    block: &magnitude_family_contracts::Block,
    sched: BarBlockSchedule,
    layer_state: &BlockAttnResLayerState,
    slab_rows: u32,
    slab_banks: u32,
    prev_ffn: Option<(
        &magnitude_family_contracts::Sublayer,
        WeightScope,
        BarBlockSchedule,
        &BlockAttnResLayerState,
        u32,
    )>,
    tap_count: u64,
    is_streamed: bool,
    owned_tensors: &mut Vec<Tensor>,
    all_plans: &mut Vec<NativeGraphPlan>,
) -> Result<BoundBarAttnAndMix, String> {
    let prepared = &bound_bar.prepared;
    let charges = &bound_bar.charges;
    let device = charges.device();
    let weights = &bound_bar.weights;
    let schedule = &prepared.schedule;
    let m = key.m;
    let h = decoder.hidden;
    let spans = key.spans;
    let slots = key.slots;
    let scale_mix = 1.0f32 / (h as f32).sqrt();
    let norm_eps = prepared.plan.norm_epsilon as f32;
    let norm_values = i32::from(prepared.plan.normalize_values);

    let attn_sub = &block.sublayers[0];
    let attn_scope = WeightScope::TargetSublayer(SublayerIndex {
        block: b_idx as u32,
        sublayer: 0,
    });
    let Operator::Attention(attn) = &attn_sub.op else {
        return Err("expected Attention as sublayer 0".into());
    };
    let InputNorm::BlockAttention(attn_in) = &attn_sub.input else {
        return Err("expected BlockAttention input on attention sublayer".into());
    };
    let HistoryRows::Own(RowProducer::Projected(proj_rows)) = &attn.fresh().rows else {
        return Err("expected own projected rows in attention".into());
    };
    let ffn_sub_opt = block.sublayers.get(1);
    let ffn_scope = WeightScope::TargetSublayer(SublayerIndex {
        block: b_idx as u32,
        sublayer: 1,
    });

    let mut graph = device.native_graph();
    let mut sc = StaticCollector::new(charges, &bound_bar.fused_proj_cache, owned_tensors);

            let mut sources_buf = graph
                .port(Element::f32(), &[schedule.max_sources, m, h])
                .map_err(|e| e.to_string())?;
            let prev_ffn_has_conv =
                prev_ffn.is_some_and(|(sub, _, _, _, _)| sub.post_conv.is_some());
            let has_any_conv = prev_ffn_has_conv
                || proj_rows.key_conv.is_some()
                || proj_rows.value_conv.is_some()
                || attn_sub.post_conv.is_some();
            let (conv_segments, conv_stop, conv_prev_bank, conv_prev_tape, conv_next_bank) =
                if has_any_conv {
                    (
                        Some(graph.port(Element::i32(), &[slots + 1, 2]).map_err(|e| e.to_string())?),
                        Some(graph.port(Element::i32(), &[slots]).map_err(|e| e.to_string())?),
                        Some(graph.port(Element::i32(), &[slots]).map_err(|e| e.to_string())?),
                        Some(graph.port(Element::i32(), &[slots]).map_err(|e| e.to_string())?),
                        Some(graph.port(Element::i32(), &[slots]).map_err(|e| e.to_string())?),
                    )
                } else {
                    (None, None, None, None, None)
                };
            let (prev_ffn_out_buf, prev_ffn_post_conv_window) =
                if let Some((p_sub, p_scope, p_sched, p_state, p_slab_banks)) = prev_ffn {
                    let conv_refs = if has_any_conv {
                        Some((
                            conv_segments.as_ref().unwrap(),
                            conv_stop.as_ref().unwrap(),
                            conv_prev_bank.as_ref().unwrap(),
                            conv_prev_tape.as_ref().unwrap(),
                            conv_next_bank.as_ref().unwrap(),
                        ))
                    } else {
                        None
                    };
                    let (out_port, win_port) = enqueue_ffn_post_into_graph(
                        prepared,
                        weights,
                        &mut graph,
                        &mut sc,
                        &mut sources_buf,
                        p_sub,
                        p_scope,
                        p_sched,
                        p_state,
                        p_slab_banks,
                        conv_refs,
                        m,
                        h,
                    )?;
                    (Some(out_port), win_port)
                } else {
                    (None, None)
                };
            let batched_cp = use_batched_cp_msgs(schedule, m);
            let block_uses_cp = !matches!(attn_in.cross_pass, CrossPassInput::None)
                || ffn_sub_opt.is_some_and(|s| {
                    matches!(
                        &s.input,
                        InputNorm::BlockAttention(ba) if !matches!(ba.cross_pass, CrossPassInput::None)
                    )
                });
            let is_reset_with_cp =
                matches!(sched.transition, Some(BlockTransition::ResetToPreludeAndBank))
                    && !schedule.pass_cp_queries[b_idx].is_empty();
            let needs_bank_port = !batched_cp
                && (matches!(sched.transition, Some(BlockTransition::ResetToPreludeAndBank))
                    || block_uses_cp);
            let needs_cp_msgs_port = batched_cp && (is_reset_with_cp || block_uses_cp);
            let mut bank_buf = if needs_bank_port {
                Some(
                    graph
                        .port(Element::f32(), &[schedule.max_bank, m, h])
                        .map_err(|e| e.to_string())?,
                )
            } else {
                None
            };
            let mut cp_msgs_buf = if needs_cp_msgs_port {
                Some(
                    graph
                        .port(Element::f32(), &[schedule.max_pass_cp_queries, m, h])
                        .map_err(|e| e.to_string())?,
                )
            } else {
                None
            };
            let mut prelude_partial = if sched.prelude_slot.is_some() {
                Some(
                    graph
                        .port(Element::f32(), &[m, h])
                        .map_err(|e| e.to_string())?,
                )
            } else {
                None
            };

            // Pass-boundary transitions (on-device `feature_copy` and pass-entry
            // batched cross-pass `block_attn_res_mix` on decode).
            let pass_cp_all_msgs = enqueue_pass_transition(
                prepared,
                weights,
                &mut graph,
                &mut sc,
                &mut sources_buf,
                &mut bank_buf,
                &mut cp_msgs_buf,
                &mut prelude_partial,
                b_idx,
                sched,
                batched_cp,
                m,
                h,
                norm_eps,
                scale_mix,
                norm_values,
            )?;
            let cp_msgs_tensor: Option<&WorkflowTensor> = match (&pass_cp_all_msgs, &cp_msgs_buf) {
                (Some(all_msgs), _) => Some(all_msgs),
                (None, Some(port)) => Some(port.tensor()),
                (None, None) => None,
            };

            // Optional DSpark target tap at Sublayer(b_idx, 0).
            let tap_entry = TapPoint::Sublayer(SublayerIndex {
                block: b_idx as u32,
                sublayer: 0,
            });
            let tap_pos = if key.has_taps {
                prepared.tap_points.iter().position(|tp| *tp == tap_entry)
            } else {
                None
            };
            let tap_buf = if let Some(tp_idx) = tap_pos {
                let tap_elem = key.tap_element.unwrap_or(prepared.activation);
                let tap_k = prepared
                    .tap_rows
                    .get(&tap_elem)
                    .ok_or_else(|| format!("missing tap_rows kernel for {tap_elem:?}"))?;
                let mut tb = graph
                    .port(tap_elem, &[m, tap_count * h])
                    .map_err(|e| e.to_string())?;
                let idx_port = sc.bind_i32(&mut graph, &[1], &[tp_idx as i32])?;
                let cur_partial = sources_buf
                    .tensor()
                    .slice_leading(sched.attn_partial_in_slot, sched.attn_partial_in_slot + 1)
                    .reshape(&[m, h]);
                graph
                    .enqueue(
                        tap_k,
                        tap_rows::WorkflowArgs {
                            residual: (&cur_partial).into(),
                            index: idx_port.tensor().into(),
                            taps: tb.tensor_mut().into(),
                        },
                    )
                    .map_err(|e| format!("enqueue tap_rows block {b_idx}: {e}"))?;
                Some(tb)
            } else {
                None
            };

            let (xq_norm, xk_norm, xv_norm, split_streams) = enqueue_attn_mix_and_norm(
                prepared,
                weights,
                &mut graph,
                &mut sc,
                &sources_buf,
                bank_buf.as_ref(),
                cp_msgs_tensor,
                attn_in,
                attn_scope,
                sched,
                m,
                h,
                norm_eps,
                scale_mix,
                norm_values,
            )?;

            // Dynamic per-step control ports for Attention & Convs.
            let coords = graph
                .port(Element::i32(), &[m, 4])
                .map_err(|e| e.to_string())?;
            let visible = graph
                .port(Element::i32(), &[m, spans, 2])
                .map_err(|e| e.to_string())?;
            let fresh = graph
                .port(Element::i32(), &[m, 2])
                .map_err(|e| e.to_string())?;
            let destinations = graph
                .port(Element::i32(), &[m])
                .map_err(|e| e.to_string())?;

            let w_q = weights
                .get(&WeightRole { scope: attn_scope, kind: WeightKind::Query })
                .ok_or("missing Query weight")?;
            let w_k = weights
                .get(&WeightRole { scope: attn_scope, kind: WeightKind::Key })
                .ok_or("missing Key weight")?;
            let w_v = weights
                .get(&WeightRole { scope: attn_scope, kind: WeightKind::Value })
                .ok_or("missing Value weight")?;
            let w_o = weights
                .get(&WeightRole { scope: attn_scope, kind: WeightKind::AttentionOutput })
                .ok_or("missing AttentionOutput weight")?;

            // Horizontally fuse attention input projections on M = 1 resident decode:
            //   Pass 1 (`!split_streams`): [W_q; W_g; W_r; W_k; W_v] -> 1 launch (-4 calls)
            //   Passes 2/3 (`split_streams`): slice_leading([W_q; W_g; W_r]) -> 1 launch (-2 calls)
            // Cached by physical tensor identity so looped passes share one allocation.
            let mut fused_qgr: Option<(
                seismic::WorkflowTensorView,
                seismic::WorkflowTensorView,
                seismic::WorkflowTensorView,
                Option<seismic::WorkflowTensorView>,
                Option<seismic::WorkflowTensorView>,
            )> = None;
            if m == 1
                && !is_streamed
                && matches!(attn.gate, AttentionGate::Separate { .. })
                && attn.relative_bias.is_some()
            {
                if let (Some(w_g), Some(w_r)) = (
                    weights.get(&WeightRole { scope: attn_scope, kind: WeightKind::AttentionGate }),
                    weights.get(&WeightRole { scope: attn_scope, kind: WeightKind::RelativeQuery }),
                ) {
                    let dense_ok = |a: &BoundWeight, b: &BoundWeight| {
                        a.element() == b.element()
                            && a.scale.is_none()
                            && b.scale.is_none()
                            && a.bound.is_none()
                            && b.bound.is_none()
                            && a.tensor.extents().len() == 2
                            && b.tensor.extents().len() == 2
                            && a.tensor.extents()[1] == b.tensor.extents()[1]
                    };
                    if dense_ok(w_q, w_g) && dense_ok(w_q, w_r) {
                        let elem = w_q.element();
                        let k_dim = w_q.tensor.extents()[1];
                        let n_q = w_q.tensor.extents()[0];
                        let n_g = w_g.tensor.extents()[0];
                        let n_r = w_r.tensor.extents()[0];
                        let n_3 = n_q + n_g + n_r;
                        let kv_ok = dense_ok(w_q, w_k) && dense_ok(w_q, w_v);
                        let n_k = w_k.tensor.extents()[0];
                        let n_v = w_v.tensor.extents()[0];
                        let n_5 = n_3 + n_k + n_v;
                        let key5 = (prepared.activation, elem, prepared.activation, k_dim, n_5, 0, 0);
                        let key3 = (prepared.activation, elem, prepared.activation, k_dim, n_3, 0, 0);
                        let use_5 = !split_streams && kv_ok && prepared.projects.contains_key(&key5);
                        let use_3 = !use_5 && prepared.projects.contains_key(&key3);
                        if use_5 || use_3 {
                            let role_q = WeightRole { scope: attn_scope, kind: WeightKind::Query };
                            let role_g = WeightRole { scope: attn_scope, kind: WeightKind::AttentionGate };
                            let role_r = WeightRole { scope: attn_scope, kind: WeightKind::RelativeQuery };
                            let role_k = WeightRole { scope: attn_scope, kind: WeightKind::Key };
                            let role_v = WeightRole { scope: attn_scope, kind: WeightKind::Value };
                            let fused_tensor: Tensor = if kv_ok && prepared.projects.contains_key(&key5) {
                                let t5 = sc.get_or_fuse_weights(
                                    &[
                                        (role_q, &w_q.tensor),
                                        (role_g, &w_g.tensor),
                                        (role_r, &w_r.tensor),
                                        (role_k, &w_k.tensor),
                                        (role_v, &w_v.tensor),
                                    ],
                                    elem,
                                    &[n_5, k_dim],
                                )?;
                                if use_5 {
                                    t5
                                } else {
                                    t5.slice_leading(0, n_3).map_err(|e| e.to_string())?
                                }
                            } else {
                                sc.get_or_fuse_weights(
                                    &[
                                        (role_q, &w_q.tensor),
                                        (role_g, &w_g.tensor),
                                        (role_r, &w_r.tensor),
                                    ],
                                    elem,
                                    &[n_3, k_dim],
                                )?
                            };
                            let n_tot = if use_5 { n_5 } else { n_3 };
                            let proj_k = if use_5 {
                                &prepared.projects[&key5]
                            } else {
                                &prepared.projects[&key3]
                            };
                            let w_port = sc.bind_tensor(&mut graph, &fused_tensor)?;
                            let empty_s = sc.bind_empty(&mut graph, Element::f32(), &[0])?;
                            let out = graph
                                .enqueue(
                                    proj_k,
                                    project_rows::WorkflowArgs {
                                        source: (&xq_norm).into(),
                                        weight: w_port.tensor().into(),
                                        weight_scale: empty_s.tensor().into(),
                                        weight_bound: empty_s.tensor().into(),
                                    },
                                )
                                .map_err(|e| format!("enqueue fused attn project_rows: {e}"))?
                                .value;
                            let flat = out.reshape(&[n_tot]);
                            let q_v = flat.slice_leading(0, n_q).reshape(&[1, n_q]);
                            let g_v = flat.slice_leading(n_q, n_q + n_g).reshape(&[1, n_g]);
                            let r_v = flat.slice_leading(n_q + n_g, n_3).reshape(&[1, n_r]);
                            let (k_v, v_v) = if use_5 {
                                (
                                    Some(flat.slice_leading(n_3, n_3 + n_k).reshape(&[1, n_k])),
                                    Some(flat.slice_leading(n_3 + n_k, n_5).reshape(&[1, n_v])),
                                )
                            } else {
                                (None, None)
                            };
                            fused_qgr = Some((q_v, g_v, r_v, k_v, v_v));
                        }
                    }
                }
            }

            let q_raw_v: seismic::WorkflowTensorView = match &fused_qgr {
                Some((qv, _, _, _, _)) => qv.clone(),
                None => enqueue_project(prepared, &mut graph, &mut sc, (&xq_norm).into(), prepared.activation, w_q, prepared.activation)?
                    .reshape(&[m, attn.heads * attn.width]),
            };
            let mut k_raw_v: seismic::WorkflowTensorView = match &fused_qgr {
                Some((_, _, _, Some(kv), _)) => kv.clone(),
                _ => enqueue_project(prepared, &mut graph, &mut sc, (&xk_norm).into(), prepared.activation, w_k, prepared.activation)?
                    .reshape(&[m, attn.kv_heads * attn.width]),
            };
            let v_raw_v: seismic::WorkflowTensorView = match &fused_qgr {
                Some((_, _, _, _, Some(vv))) => vv.clone(),
                _ => enqueue_project(prepared, &mut graph, &mut sc, (&xv_norm).into(), prepared.activation, w_v, prepared.activation)?
                    .reshape(&[m, attn.kv_heads * attn.width]),
            };
            let fused_g_v: Option<seismic::WorkflowTensorView> =
                fused_qgr.as_ref().map(|(_, gv, _, _, _)| gv.clone());
            let fused_r_v: Option<seismic::WorkflowTensorView> =
                fused_qgr.as_ref().map(|(_, _, rv, _, _)| rv.clone());

            // Key short conv
            let mut key_conv_window = if let Some(k_conv) = &proj_rows.key_conv {
                let cw = weights
                    .get(&WeightRole { scope: attn_scope, kind: WeightKind::KeyShortConv })
                    .ok_or("missing KeyShortConv")?;
                let dyn_raw = if let Some(dyn_cfg) = &k_conv.dynamic {
                    let dw = weights
                        .get(&WeightRole { scope: attn_scope, kind: WeightKind::KeyShortConvDynamic })
                        .ok_or("missing KeyShortConvDynamic")?;
                    let r = enqueue_project(prepared, &mut graph, &mut sc, (&xk_norm).into(), prepared.activation, dw, Element::f32())?;
                    Some(r.reshape(&[m, dyn_cfg.heads, k_conv.kernel_size]))
                } else {
                    None
                };
                let win_t = layer_state
                    .key_conv_window
                    .as_ref()
                    .ok_or("missing key_conv_window")?;
                let mut win = graph
                    .port(win_t.element(), win_t.extents())
                    .map_err(|e| e.to_string())?;
                let k_conv_out = enqueue_conv(
                    prepared,
                    &mut graph,
                    &mut sc,
                    (&k_raw_v).into(),
                    false,
                    &cw.tensor,
                    dyn_raw.as_ref().map(|v| v.into()),
                    &mut win,
                    k_conv.residual_scale as f32,
                    slab_banks,
                    conv_segments.as_ref().unwrap(),
                    conv_stop.as_ref().unwrap(),
                    conv_prev_bank.as_ref().unwrap(),
                    conv_prev_tape.as_ref().unwrap(),
                    conv_next_bank.as_ref().unwrap(),
                    m,
                    k_conv.kernel_size,
                    false,
                )?;
                k_raw_v = k_conv_out.reshape(&[m, attn.kv_heads * attn.width]);
                Some(win)
            } else {
                None
            };

            // Value short conv -> F32 for unweighted V-RMSNorm
            let (v_for_norm_v, mut value_conv_window) = if let Some(v_conv) = &proj_rows.value_conv {
                let cw = weights
                    .get(&WeightRole { scope: attn_scope, kind: WeightKind::ValueShortConv })
                    .ok_or("missing ValueShortConv")?;
                let dyn_raw = if let Some(dyn_cfg) = &v_conv.dynamic {
                    let dw = weights
                        .get(&WeightRole { scope: attn_scope, kind: WeightKind::ValueShortConvDynamic })
                        .ok_or("missing ValueShortConvDynamic")?;
                    let r = enqueue_project(prepared, &mut graph, &mut sc, (&xv_norm).into(), prepared.activation, dw, Element::f32())?;
                    Some(r.reshape(&[m, dyn_cfg.heads, v_conv.kernel_size]))
                } else {
                    None
                };
                let win_t = layer_state
                    .value_conv_window
                    .as_ref()
                    .ok_or("missing value_conv_window")?;
                let mut win = graph
                    .port(win_t.element(), win_t.extents())
                    .map_err(|e| e.to_string())?;
                let v_out = enqueue_conv(
                    prepared,
                    &mut graph,
                    &mut sc,
                    (&v_raw_v).into(),
                    false,
                    &cw.tensor,
                    dyn_raw.as_ref().map(|v| v.into()),
                    &mut win,
                    v_conv.residual_scale as f32,
                    slab_banks,
                    conv_segments.as_ref().unwrap(),
                    conv_stop.as_ref().unwrap(),
                    conv_prev_bank.as_ref().unwrap(),
                    conv_prev_tape.as_ref().unwrap(),
                    conv_next_bank.as_ref().unwrap(),
                    m,
                    v_conv.kernel_size,
                    true,
                )?;
                (v_out.reshape(&[m, attn.kv_heads, attn.width]), Some(win))
            } else {
                (v_raw_v.reshape(&[m, attn.kv_heads, attn.width]), None)
            };

            let (log_floor, log_alpha) = match attn.log_scale {
                Some(LogScale { floor, alpha }) => (floor as f32, alpha as f32),
                None => (1.0, 0.0),
            };

            // Q head_l2_clip
            let q_3d = q_raw_v.reshape(&[m, attn.heads, attn.width]);
            let q_ready_t: Option<WorkflowTensor> = match &attn.query_norm {
                HeadNorm::L2Clip { threshold, epsilon, scale } => {
                    let sc_port = if scale.is_some() {
                        let w = weights
                            .get(&WeightRole { scope: attn_scope, kind: WeightKind::QueryNorm })
                            .ok_or("missing QueryNorm L2Clip scale")?;
                        sc.bind_tensor(&mut graph, &w.tensor)?
                    } else {
                        sc.bind_empty(&mut graph, Element::f32(), &[0, attn.width])?
                    };
                    Some(
                        graph
                            .enqueue(
                                &prepared.head_l2_clip,
                                head_l2_clip::WorkflowArgs {
                                    x: (&q_3d).into(),
                                    scale: sc_port.tensor().into(),
                                    coordinates: coords.tensor().into(),
                                    threshold: (*threshold as f32).into(),
                                    epsilon: (*epsilon as f32).into(),
                                    log_floor: log_floor.into(),
                                    log_alpha: log_alpha.into(),
                                },
                            )
                            .map_err(|e| format!("enqueue Q head_l2_clip: {e}"))?
                            .value,
                    )
                }
                _ => None,
            };

            // K head_l2_clip
            let k_3d = k_raw_v.reshape(&[m, attn.kv_heads, attn.width]);
            let k_ready_t: Option<WorkflowTensor> = match &proj_rows.key_norm {
                HeadNorm::L2Clip { threshold, epsilon, scale } => {
                    let sc_port = if scale.is_some() {
                        let w = weights
                            .get(&WeightRole { scope: attn_scope, kind: WeightKind::KeyNorm })
                            .ok_or("missing KeyNorm L2Clip scale")?;
                        sc.bind_tensor(&mut graph, &w.tensor)?
                    } else {
                        sc.bind_empty(&mut graph, Element::f32(), &[0, attn.width])?
                    };
                    Some(
                        graph
                            .enqueue(
                                &prepared.head_l2_clip,
                                head_l2_clip::WorkflowArgs {
                                    x: (&k_3d).into(),
                                    scale: sc_port.tensor().into(),
                                    coordinates: coords.tensor().into(),
                                    threshold: (*threshold as f32).into(),
                                    epsilon: (*epsilon as f32).into(),
                                    log_floor: 1.0f32.into(),
                                    log_alpha: 0.0f32.into(),
                                },
                            )
                            .map_err(|e| format!("enqueue K head_l2_clip: {e}"))?
                            .value,
                    )
                }
                _ => None,
            };

            // V unweighted per-head RMSNorm
            let v_ready_view = match proj_rows.value_norm {
                ValueNorm::RmsUnweighted(unw) => {
                    let vn = prepared
                        .v_norms
                        .get(&(attn.kv_heads, attn.width))
                        .ok_or("missing V vision_norm kernel")?;
                    let ew = sc.bind_empty(&mut graph, Element::f32(), &[0, attn.width])?;
                    let eb = sc.bind_empty(&mut graph, Element::f32(), &[0, attn.width])?;
                    let eo = sc.bind_empty(&mut graph, Element::i32(), &[0, m * attn.kv_heads])?;
                    let vn_out = graph
                        .enqueue(
                            vn,
                            vision_norm::WorkflowArgs {
                                source: (&v_for_norm_v).into(),
                                weight: ew.tensor().into(),
                                bias: eb.tensor().into(),
                                order: eo.tensor().into(),
                                epsilon: (unw.epsilon as f32).into(),
                                centered: 0i32.into(),
                                interleave: 0i32.into(),
                                activation: 0i32.into(),
                            },
                        )
                        .map_err(|e| format!("enqueue V vision_norm: {e}"))?
                        .value;
                    vn_out.reshape(&[m, attn.kv_heads, attn.width])
                }
                ValueNorm::None => v_raw_v.reshape(&[m, attn.kv_heads, attn.width]),
            };

            // Relative bias table: fused `head_l2_clip(W_r @ xq_norm) -> rel_bias_project`
            let rel_bias_t: WorkflowTensor = if let Some(RelativeBias {
                d_rel,
                rel_clip,
                ..
            }) = &attn.relative_bias
            {
                let w_rb = weights
                    .get(&WeightRole { scope: attn_scope, kind: WeightKind::RelativeBiasProjection })
                    .ok_or("missing RelativeBiasProjection")?;
                let r_3d = match &fused_r_v {
                    Some(rv) => rv.reshape(&[m, attn.heads, *d_rel]),
                    None => {
                        let w_r = weights
                            .get(&WeightRole { scope: attn_scope, kind: WeightKind::RelativeQuery })
                            .ok_or("missing RelativeQuery")?;
                        enqueue_project(
                            prepared,
                            &mut graph,
                            &mut sc,
                            (&xq_norm).into(),
                            prepared.activation,
                            w_r,
                            prepared.activation,
                        )?
                        .reshape(&[m, attn.heads, *d_rel])
                    }
                };
                let (sc_port, r_threshold, r_epsilon, r_log_floor, r_log_alpha) = match rel_clip {
                    HeadNorm::L2Clip { threshold, epsilon, scale } => {
                        let port = if scale.is_some() {
                            let w = weights
                                .get(&WeightRole { scope: attn_scope, kind: WeightKind::RelativeQueryNorm })
                                .ok_or("missing RelativeQueryNorm")?;
                            sc.bind_tensor(&mut graph, &w.tensor)?
                        } else {
                            sc.bind_empty(&mut graph, Element::f32(), &[0, *d_rel])?
                        };
                        (port, *threshold as f32, *epsilon as f32, log_floor, log_alpha)
                    }
                    _ => (
                        sc.bind_empty(&mut graph, Element::f32(), &[0, *d_rel])?,
                        0.0f32,
                        0.0f32,
                        1.0f32,
                        0.0f32,
                    ),
                };
                let rb_kernel = prepared
                    .rel_bias_projects
                    .get(&(prepared.activation, w_rb.element()))
                    .ok_or_else(|| {
                        format!(
                            "missing rel_bias_project kernel for ({:?}, {:?})",
                            prepared.activation,
                            w_rb.element()
                        )
                    })?;
                let wrb_port = sc.bind_tensor(&mut graph, &w_rb.tensor)?;
                graph
                    .enqueue(
                        rb_kernel,
                        rel_bias_project::WorkflowArgs {
                            r: (&r_3d).into(),
                            weight: wrb_port.tensor().into(),
                            scale: sc_port.tensor().into(),
                            coordinates: coords.tensor().into(),
                            threshold: r_threshold.into(),
                            epsilon: r_epsilon.into(),
                            log_floor: r_log_floor.into(),
                            log_alpha: r_log_alpha.into(),
                        },
                    )
                    .map_err(|e| format!("enqueue rel_bias_project: {e}"))?
                    .value
            } else {
                let p = sc.bind_empty(&mut graph, Element::f32(), &[m, attn.heads, 0])?;
                p.tensor().clone()
            };

            // Dynamic or static sinks
            let sinks_t: WorkflowTensor = match &attn.sinks {
                AttentionSinks::None => {
                    sc.bind_empty(&mut graph, Element::f32(), &[0, attn.heads])?
                        .tensor()
                        .clone()
                }
                AttentionSinks::Static(_) => {
                    let sw = weights
                        .get(&WeightRole { scope: attn_scope, kind: WeightKind::AttentionSinks })
                        .ok_or("missing static AttentionSinks")?;
                    let sw2 = sw.tensor.reshape(&[1, attn.heads]).map_err(|e| e.to_string())?;
                    sc.bind_tensor(&mut graph, &sw2)?.tensor().clone()
                }
                AttentionSinks::Dynamic(_) => {
                    let w_sink = weights
                        .get(&WeightRole { scope: attn_scope, kind: WeightKind::AttentionSinks })
                        .ok_or("missing dynamic AttentionSinks")?;
                    enqueue_project(
                        prepared,
                        &mut graph,
                        &mut sc,
                        (&xq_norm).into(),
                        prepared.activation,
                        w_sink,
                        Element::f32(),
                    )?
                }
            };

            // Output gate
            let (gate_view, gate_fn) = match &attn.gate {
                AttentionGate::None => {
                    let p = sc.bind_empty(&mut graph, prepared.activation, &[m, attn.heads, 0])?;
                    (p.tensor().reshape(&[m, attn.heads, 0]), 0)
                }
                AttentionGate::Separate { function, .. } => {
                    let g_view = match &fused_g_v {
                        Some(gv) => gv.reshape(&[m, attn.heads, attn.width]),
                        None => {
                            let w_g = weights
                                .get(&WeightRole { scope: attn_scope, kind: WeightKind::AttentionGate })
                                .ok_or("missing AttentionGate")?;
                            enqueue_project(
                                prepared,
                                &mut graph,
                                &mut sc,
                                (&xq_norm).into(),
                                prepared.activation,
                                w_g,
                                prepared.activation,
                            )?
                            .reshape(&[m, attn.heads, attn.width])
                        }
                    };
                    let code = match function {
                        GateFunction::Sigmoid => 0,
                        GateFunction::Softplus => 1,
                        GateFunction::Silu => return Err("unsupported SiLU attention gate".into()),
                    };
                    (g_view, code)
                }
                AttentionGate::Interleaved { .. } => {
                    return Err("interleaved gate not supported in attention_rel_bias".into());
                }
            };

            let mut history_key = graph
                .port(layer_state.history_key.element(), layer_state.history_key.extents())
                .map_err(|e| e.to_string())?;
            let mut history_value = graph
                .port(layer_state.history_value.element(), layer_state.history_value.extents())
                .map_err(|e| e.to_string())?;

            let q_arg: WorkflowTensorRef<'_> = match &q_ready_t {
                Some(t) => t.into(),
                None => (&q_3d).into(),
            };
            let k_arg: WorkflowTensorRef<'_> = match &k_ready_t {
                Some(t) => t.into(),
                None => (&k_3d).into(),
            };
            let attended = graph
                .enqueue(
                    &prepared.attention_rel_bias,
                    attention_rel_bias::WorkflowArgs {
                        query: q_arg,
                        gate: (&gate_view).into(),
                        key: k_arg,
                        value: (&v_ready_view).into(),
                        rel_bias: (&rel_bias_t).into(),
                        sinks: (&sinks_t).into(),
                        visible: visible.tensor().into(),
                        fresh: fresh.tensor().into(),
                        destinations: destinations.tensor().into(),
                        history_key: history_key.tensor_mut().into(),
                        history_value: history_value.tensor_mut().into(),
                        scale: (attn.scale as f32).into(),
                        gate_function: gate_fn.into(),
                        slab_rows: slab_rows.into(),
                    },
                )
                .map_err(|e| format!("enqueue attention_rel_bias: {e}"))?
                .value;

            let attn_flat = attended.reshape(&[m, attn.heads * attn.width]);
            let (attn_out, mut attn_post_conv_window) = if let Some(post_conv) = &attn_sub.post_conv {
                let attn_proj = enqueue_project(
                    prepared,
                    &mut graph,
                    &mut sc,
                    (&attn_flat).into(),
                    prepared.activation,
                    w_o,
                    prepared.activation,
                )?;
                let cw = weights
                    .get(&WeightRole { scope: attn_scope, kind: WeightKind::PostShortConv })
                    .ok_or("missing attn PostShortConv")?;
                let win_t = layer_state
                    .attn_post_conv_window
                    .as_ref()
                    .ok_or("missing attn_post_conv_window")?;
                let mut win = graph
                    .port(win_t.element(), win_t.extents())
                    .map_err(|e| e.to_string())?;
                let out = enqueue_conv(
                    prepared,
                    &mut graph,
                    &mut sc,
                    (&attn_proj).into(),
                    false,
                    &cw.tensor,
                    None,
                    &mut win,
                    post_conv.residual_scale as f32,
                    slab_banks,
                    conv_segments.as_ref().unwrap(),
                    conv_stop.as_ref().unwrap(),
                    conv_prev_bank.as_ref().unwrap(),
                    conv_prev_tape.as_ref().unwrap(),
                    conv_next_bank.as_ref().unwrap(),
                    m,
                    post_conv.kernel_size,
                    true,
                )?;
                (out, Some(win))
            } else {
                let out = enqueue_project(
                    prepared,
                    &mut graph,
                    &mut sc,
                    (&attn_flat).into(),
                    prepared.activation,
                    w_o,
                    Element::f32(),
                )?;
                (out, None)
            };

            // Commit or residual-add `attn_out` into `sources_buf[attn_partial_out_slot]`.
            let new_partial = match attn_sub.output {
                OutputForm::BlockCommit => attn_out,
                OutputForm::Residual => {
                    let prev_p = sources_buf
                        .tensor()
                        .slice_leading(sched.attn_partial_in_slot, sched.attn_partial_in_slot + 1)
                        .reshape(&[m, h]);
                    let empty_inj = sc.bind_empty(&mut graph, Element::f32(), &[m, 0])?;
                    graph
                        .enqueue(
                            &prepared.stream_join,
                            stream_join::WorkflowArgs {
                                residual: (&prev_p).into(),
                                y: (&attn_out).into(),
                                inject: empty_inj.tensor().into(),
                            },
                        )
                        .map_err(|e| format!("enqueue attn stream_join: {e}"))?
                        .value
                }
                _ => return Err("unsupported attention OutputForm".into()),
            };
            let mut dst_slot = sources_buf
                .tensor_mut()
                .slice_leading(sched.attn_partial_out_slot, sched.attn_partial_out_slot + 1)
                .reshape(&[m, h]);
            graph
                .enqueue(
                    &prepared.f32_copy,
                    feature_copy::WorkflowArgs {
                        input: (&new_partial).into(),
                        out: (&mut dst_slot).into(),
                    },
                )
                .map_err(|e| format!("enqueue attn partial store: {e}"))?;

            // If block has FFN sublayer, compute FFN `block_attn_res_mix` (+ cross-pass)
            // right here and store `x_ffn` into `ffn_in_buf`!
            let (ffn_in_buf, ffn_in_is_3d) = if let Some(ffn_sub) = ffn_sub_opt {
                let (port, is_3d) = enqueue_ffn_mix_port(
                    prepared,
                    weights,
                    &mut graph,
                    &mut sc,
                    &sources_buf,
                    bank_buf.as_ref(),
                    cp_msgs_tensor,
                    new_partial,
                    ffn_sub,
                    ffn_scope,
                    sched,
                    m,
                    h,
                    norm_eps,
                    scale_mix,
                    norm_values,
                )?;
                (Some(port), is_3d)
            } else {
                (None, false)
            };

            let _ = (&mut key_conv_window, &mut value_conv_window, &mut attn_post_conv_window);
            let plan = graph
                .seal()
                .map_err(|e| format!("seal block {b_idx} attn_and_mix: {e}"))?;
            let bound = sc.finish_bind(&plan, &format!("block {b_idx} attn_and_mix"))?;
            all_plans.push(plan.clone());
            Ok(BoundBarAttnAndMix {
                plan,
                bound,
                sources_buf,
                bank_buf,
                cp_msgs_buf,
                prelude_partial,
                prev_ffn_out_buf,
                prev_ffn_post_conv_window,
                ffn_in_buf,
                ffn_in_is_3d,
                coords,
                visible,
                fresh,
                destinations,
                conv_segments,
                conv_stop,
                conv_prev_bank,
                conv_prev_tape,
                conv_next_bank,
                history_key,
                history_value,
                key_conv_window,
                value_conv_window,
                attn_post_conv_window,
                tap_buf,
            })
}

fn build_bound_family(
    bound_bar: &BoundBlockAttnRes,
    decoder: &Decoder,
    layers: &[BlockAttnResLayerState],
    key: &BarLayoutKey,
) -> Result<BoundBarFamily, String> {
    let prepared = &bound_bar.prepared;
    let charges = &bound_bar.charges;
    let schedule = &prepared.schedule;
    let m = key.m;
    let h = decoder.hidden;
    let slots = key.slots;
    let spans = key.spans;

    let mut owned_tensors: Vec<Tensor> = Vec::new();
    let mut all_plans: Vec<NativeGraphPlan> = Vec::new();
    let is_streamed = bound_bar.streamed_routed.values().any(|b| b.fill.is_some());

    // 1. Entry graph: embedding_rows + optional EntryNorm::Rms -> sources_buf[0]
    let entry = build_bound_entry(bound_bar, decoder, key, &mut owned_tensors, &mut all_plans)?;

    // 2. Per-block `attn_and_mix` graphs (each folds the previous block's FFN
    //    post-conv + stream_join at its start; the routed FFN between them is
    //    already sealed & bound in `bound_bar.streamed_routed`).
    let mut blocks = Vec::with_capacity(decoder.blocks.len());
    let tap_count = prepared.tap_points.len() as u64;

    for (b_idx, block) in decoder.blocks.iter().enumerate() {
        let sched = schedule.blocks[b_idx];
        let layer_state = &layers[b_idx];
        let (_hist_rows, slab_rows, _conv_banks, slab_banks) = key.layer_shapes[b_idx];
        let prev_ffn = if b_idx > 0 {
            decoder.blocks[b_idx - 1].sublayers.get(1).map(|p_sub| {
                let p_scope = WeightScope::TargetSublayer(SublayerIndex {
                    block: (b_idx - 1) as u32,
                    sublayer: 1,
                });
                let p_slab_banks = key.layer_shapes[b_idx - 1].3;
                (
                    p_sub,
                    p_scope,
                    schedule.blocks[b_idx - 1],
                    &layers[b_idx - 1],
                    p_slab_banks,
                )
            })
        } else {
            None
        };

        let attn_and_mix = build_bound_attn_and_mix(
            bound_bar,
            decoder,
            key,
            b_idx,
            block,
            sched,
            layer_state,
            slab_rows,
            slab_banks,
            prev_ffn,
            tap_count,
            is_streamed,
            &mut owned_tensors,
            &mut all_plans,
        )?;
        blocks.push(attn_and_mix);
    }

    // 3. Exit graph: last block's FFN post + optional Exit tap + ExitNorm::BlockAttention -> exit_hidden
    let last_ffn = decoder.blocks.len().checked_sub(1).and_then(|last_idx| {
        decoder.blocks[last_idx].sublayers.get(1).map(|p_sub| {
            let p_scope = WeightScope::TargetSublayer(SublayerIndex {
                block: last_idx as u32,
                sublayer: 1,
            });
            let p_slab_banks = key.layer_shapes[last_idx].3;
            (
                p_sub,
                p_scope,
                schedule.blocks[last_idx],
                &layers[last_idx],
                p_slab_banks,
            )
        })
    });
    let exit = build_bound_exit(
        bound_bar,
        decoder,
        key,
        tap_count,
        last_ffn,
        &mut owned_tensors,
        &mut all_plans,
    )?;

    // One step executes every plan in `all_plans`; pass the full sequence so
    // `NativeGraphFamily`'s CUDA graph budget holds and pins every plan for
    // each concurrent flight (avoiding mid-flight graph eviction under lookahead).
    let sequences: Vec<Vec<NativeGraphPlan>> = vec![all_plans.clone()];
    let decodes_count = usize::from(decodes(m));
    let family = NativeGraphFamily::with_launches(&all_plans, &sequences, decodes_count, &[])
        .map_err(|e| format!("BlockAttnRes NativeGraphFamily: {e}"))?;

    // Pre-allocate a 2-slot decode scratch ring for `m <= VERIFY_ROWS` matching
    // `TargetGraphWorkspaceLease`'s 2-slot `routed_slot` ring so every routed
    // FFN CUDA graph sees fixed `ffn_in_buf`/`ffn_out_buf` addresses (0 updates).
    let mut ring = Vec::new();
    if decodes(m) {
        for _ in 0..2 {
            ring.push(BarStepScratch::allocate(
                charges, &family, schedule, m, h, slots, spans,
            )?);
        }
    }

    Ok(BoundBarFamily {
        family,
        entry,
        blocks,
        exit,
        _owned_tensors: owned_tensors,
        decode_ring: Mutex::new((0, ring)),
        prefill_slot: Mutex::new(None),
        prefill_bufs_pool: Mutex::new(Vec::new()),
        prefill_grouped_pool: Mutex::new(HashMap::new()),
        prefill_merged_pool: Mutex::new(HashMap::new()),
    })
}

/// One step inside a `queue_sealed_chain` wavefront.
pub(crate) struct BarChainStep<'a> {
    pub routed_slot: &'a mut NativeGraphFamilySlot,
    pub layers: Vec<BlockAttnResLayerState>,
    /// Embedded `[M, H]` F32 hidden rows produced by `PreparedTargetEntryGraph`
    /// (handling resident/stored tables, `EntryNotice`, `LandedReadPorts`, and
    /// `EntryNorm::Rms` through the single shared entry path).
    pub entry_hidden: Tensor,
    pub coordinates: &'a [i32],
    pub media_overlays: Vec<(usize, usize, Vec<f32>)>,
    pub visible_per_layer: Vec<Vec<i32>>,
    pub spans: u64,
    pub fresh: Vec<i32>,
    pub destinations: Vec<i32>,
    pub conv_segments: Vec<i32>,
    pub conv_stop: &'a [i32],
    pub conv_prev_bank: &'a [i32],
    pub conv_prev_tape: &'a [i32],
    pub conv_next_bank: &'a [i32],
    pub tap_buf: Option<Tensor>,
    pub bank_run: Option<BlockAttnResBankRun<'a>>,
}

/// Prepared per-step runtime state for one step inside `queue_sealed_chain`.
struct PreparedChainStep {
    m: u64,
    family: Arc<BoundBarFamily>,
    bufs: Option<BarStepBuffers>,
    prefill_idx: Option<usize>,
    ring_idx: Option<usize>,
    coords_t: Tensor,
    fresh_t: Tensor,
    dest_t: Tensor,
    conv_seg_t: Tensor,
    conv_stop_t: Tensor,
    conv_prev_bank_t: Tensor,
    conv_prev_tape_t: Tensor,
    conv_next_bank_t: Tensor,
    vis_cache: Vec<Tensor>,
    vis_for_layer: Vec<usize>,
}

/// Queue a chain of 1 or more `BlockAttnRes` steps onto `queue_compute` (the
/// caller's `StepSubmitter`) in strict layer-by-layer wavefront order so
/// multi-chunk streamed `ScratchGroup` prefills publish every step's `Front`
/// request at layer `l` before any `Fill`/`Pass` at layer `l`, with ZERO host
/// syncs across all layers!
#[inline(never)]
fn prepare_chain_step(
    bound_bar: &BoundBlockAttnRes,
    decoder: &Decoder,
    step: &BarChainStep<'_>,
    s_idx: usize,
    total_steps: usize,
) -> Result<PreparedChainStep, String> {
    let charges = &bound_bar.charges;
    let prepared = &bound_bar.prepared;
    let h = decoder.hidden;
    let upload_i32 = |shape: &[u64], vals: &[i32]| -> Result<Tensor, String> {
        let bytes: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        Tensor::from_host(charges, Element::i32(), shape, &bytes).map_err(|e| e.to_string())
    };
    let m = step.entry_hidden.extents()[0];
    let slots = step.conv_stop.len() as u64;
    if step.layers.len() != decoder.blocks.len() {
        return Err("layer state count disagrees with decoder blocks".into());
    }
    let layer_shapes: Vec<(u64, u32, u64, u32)> = step
        .layers
        .iter()
        .map(|l| {
            let hist_rows = l.history_key.extents()[0];
            let conv_banks = l
                .key_conv_window
                .as_ref()
                .or(l.value_conv_window.as_ref())
                .or(l.attn_post_conv_window.as_ref())
                .or(l.ffn_post_conv_window.as_ref())
                .map_or(0, |t| t.extents()[0]);
            (hist_rows, l.slab_rows, conv_banks, l.slab_banks)
        })
        .collect();
    let key = BarLayoutKey {
        m,
        spans: step.spans,
        slots,
        has_taps: step.tap_buf.is_some() && !prepared.tap_points.is_empty(),
        tap_element: step.tap_buf.as_ref().map(|t| t.element()),
        layer_shapes,
    };
    let family: Arc<BoundBarFamily> = {
        let mut cache = bound_bar.families.lock().expect("bar family cache lock");
        if let Some(existing) = cache.get(&key) {
            existing.clone()
        } else {
            let built = Arc::new(build_bound_family(bound_bar, decoder, &step.layers, &key)?);
            cache.insert(key.clone(), built.clone());
            built
        }
    };
    let (bufs, prefill_idx, ring_idx) = if decodes(m) && total_steps == 1 {
        let mut g = family.decode_ring.lock().expect("bar decode ring lock");
        let idx = g.0 % g.1.len();
        g.0 = g.0.wrapping_add(1);
        (None, None, Some(idx))
    } else if m == bound_bar.prefill_rows && s_idx < bound_bar.prefill_bufs.len() {
        (None, Some(s_idx), None)
    } else {
        let reused = family
            .prefill_bufs_pool
            .lock()
            .expect("bar prefill_bufs_pool lock")
            .pop();
        let allocated = match reused {
            Some(b) => b,
            None => BarStepBuffers::allocate(charges, &prepared.schedule, m, h)?,
        };
        (Some(allocated), None, None)
    };
    let write_i32 = |buf: &mut Tensor, vals: &[i32]| -> Result<(), String> {
        let bytes: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        buf.write_from_host(&bytes).map_err(|e| e.to_string())
    };
    let (
        coords_t,
        fresh_t,
        dest_t,
        conv_seg_t,
        conv_stop_t,
        conv_prev_bank_t,
        conv_prev_tape_t,
        conv_next_bank_t,
        vis_cache,
        vis_for_layer,
    ) = if let Some(ridx) = ring_idx {
        let mut g = family.decode_ring.lock().expect("bar decode ring lock");
        let sc = &mut g.1[ridx];
        write_i32(&mut sc.coords_buf, step.coordinates)?;
        write_i32(&mut sc.fresh_buf, &step.fresh)?;
        write_i32(&mut sc.dest_buf, &step.destinations)?;
        write_i32(&mut sc.conv_seg_buf, &step.conv_segments)?;
        write_i32(&mut sc.conv_stop_buf, step.conv_stop)?;
        write_i32(&mut sc.conv_prev_bank_buf, step.conv_prev_bank)?;
        write_i32(&mut sc.conv_prev_tape_buf, step.conv_prev_tape)?;
        write_i32(&mut sc.conv_next_bank_buf, step.conv_next_bank)?;

        let mut vis_keys: Vec<&[i32]> = Vec::new();
        let mut vis_cache: Vec<Tensor> = Vec::new();
        let mut vis_for_layer: Vec<usize> = Vec::with_capacity(step.visible_per_layer.len());
        for vis in &step.visible_per_layer {
            if let Some(idx) = vis_keys.iter().position(|prev| *prev == vis.as_slice()) {
                vis_for_layer.push(idx);
            } else {
                let idx = vis_cache.len();
                if idx >= sc.vis_bufs.len() {
                    sc.vis_bufs.push(
                        Tensor::zeros(charges, Element::i32(), &[m, step.spans, 2])
                            .map_err(|e| e.to_string())?,
                    );
                }
                write_i32(&mut sc.vis_bufs[idx], vis)?;
                vis_keys.push(vis.as_slice());
                vis_cache.push(sc.vis_bufs[idx].clone());
                vis_for_layer.push(idx);
            }
        }
        (
            sc.coords_buf.clone(),
            sc.fresh_buf.clone(),
            sc.dest_buf.clone(),
            sc.conv_seg_buf.clone(),
            sc.conv_stop_buf.clone(),
            sc.conv_prev_bank_buf.clone(),
            sc.conv_prev_tape_buf.clone(),
            sc.conv_next_bank_buf.clone(),
            vis_cache,
            vis_for_layer,
        )
    } else {
        let coords_t = upload_i32(&[m, 4], step.coordinates)?;
        let fresh_t = upload_i32(&[m, 2], &step.fresh)?;
        let dest_t = upload_i32(&[m], &step.destinations)?;
        let conv_seg_t = upload_i32(&[slots + 1, 2], &step.conv_segments)?;
        let conv_stop_t = upload_i32(&[slots], step.conv_stop)?;
        let conv_prev_bank_t = upload_i32(&[slots], step.conv_prev_bank)?;
        let conv_prev_tape_t = upload_i32(&[slots], step.conv_prev_tape)?;
        let conv_next_bank_t = upload_i32(&[slots], step.conv_next_bank)?;

        let mut vis_keys: Vec<&[i32]> = Vec::new();
        let mut vis_cache: Vec<Tensor> = Vec::new();
        let mut vis_for_layer: Vec<usize> = Vec::with_capacity(step.visible_per_layer.len());
        for vis in &step.visible_per_layer {
            if let Some(idx) = vis_keys.iter().position(|prev| *prev == vis.as_slice()) {
                vis_for_layer.push(idx);
            } else {
                let t = upload_i32(&[m, step.spans, 2], vis)?;
                let idx = vis_cache.len();
                vis_keys.push(vis.as_slice());
                vis_cache.push(t);
                vis_for_layer.push(idx);
            }
        }
        (
            coords_t,
            fresh_t,
            dest_t,
            conv_seg_t,
            conv_stop_t,
            conv_prev_bank_t,
            conv_prev_tape_t,
            conv_next_bank_t,
            vis_cache,
            vis_for_layer,
        )
    };
    Ok(PreparedChainStep {
        m,
        family,
        bufs,
        prefill_idx,
        ring_idx,
        coords_t,
        fresh_t,
        dest_t,
        conv_seg_t,
        conv_stop_t,
        conv_prev_bank_t,
        conv_prev_tape_t,
        conv_next_bank_t,
        vis_cache,
        vis_for_layer,
    })
}

#[inline(never)]
#[allow(clippy::too_many_arguments)]
fn queue_exit_step(
    charges: &Charges,
    ex: &BoundBarExit,
    bufs: &BarStepBuffers,
    ps: &PreparedChainStep,
    last_layer: Option<&BlockAttnResLayerState>,
    slot: &mut NativeGraphFamilySlot,
    tap_buf: Option<&Tensor>,
    queue_compute: &mut impl FnMut(
        ReadyNativeGraphRun<'_>,
        Box<dyn FnOnce(&NativeGraphOutputs) -> Result<(), String> + '_>,
    ) -> Result<(), String>,
) -> Result<Tensor, String> {
    let mut bindings = ex.bound.bindings();
    bindings
        .set(charges, &ex.sources_buf, &bufs.sources_buf)
        .map_err(|e| e.to_string())?;
    if let Some(p) = &ex.prev_ffn_out_buf {
        bindings
            .set(charges, p, &bufs.ffn_out_buf)
            .map_err(|e| e.to_string())?;
    }
    if let Some(p) = &ex.conv_segments {
        bindings.set(charges, p, &ps.conv_seg_t).map_err(|e| e.to_string())?;
    }
    if let Some(p) = &ex.conv_stop {
        bindings.set(charges, p, &ps.conv_stop_t).map_err(|e| e.to_string())?;
    }
    if let Some(p) = &ex.conv_prev_bank {
        bindings.set(charges, p, &ps.conv_prev_bank_t).map_err(|e| e.to_string())?;
    }
    if let Some(p) = &ex.conv_prev_tape {
        bindings.set(charges, p, &ps.conv_prev_tape_t).map_err(|e| e.to_string())?;
    }
    if let Some(p) = &ex.conv_next_bank {
        bindings.set(charges, p, &ps.conv_next_bank_t).map_err(|e| e.to_string())?;
    }
    if let (Some(p), Some(w)) = (
        &ex.prev_ffn_post_conv_window,
        last_layer.and_then(|l| l.ffn_post_conv_window.as_ref()),
    ) {
        bindings.set(charges, p, w).map_err(|e| e.to_string())?;
    }
    // Bind `exit_hidden` to `ffn_in_3d` (disjoint from `prev_ffn_out_buf` which
    // aliases `ffn_out_buf`, avoiding any input/output port aliasing).
    bindings
        .set(charges, &ex.exit_hidden, &bufs.ffn_in_3d)
        .map_err(|e| e.to_string())?;
    if let (Some(p), Some(tb)) = (&ex.tap_buf, tap_buf) {
        bindings.set(charges, p, tb).map_err(|e| e.to_string())?;
    }
    let mut active = slot.activate(&ex.plan).map_err(|e| e.to_string())?;
    let ready = active
        .attach(
            bindings,
            ex.plan.new_outputs(charges).map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())?;
    queue_compute(ready, Box::new(|_| Ok(())))?;
    Ok(bufs.ffn_in_buf.clone())
}

/// Action emitted by [`queue_sealed_chain`] onto the caller's `StepSubmitter`.
pub(crate) enum BarComputeAction<'a> {
    Queue {
        ready: ReadyNativeGraphRun<'a>,
        register: Box<dyn FnOnce(&NativeGraphOutputs) -> Result<(), String> + 'a>,
    },
    /// Flush every step's `Front` submission and, on boundary-parked backends
    /// under `tier_copy_dma`, wait for completion and expose `layer` through
    /// `max_sequence` to `Requests::serve`.
    FlushFront {
        bank: Arc<crate::experts::ExpertBank>,
        layer: usize,
        max_sequence: u64,
    },
    /// Wait for the `Copies` `PrefillFill` DMA transfer for `(layer, sequence,
    /// window)`, queue `Pass`, and flush with tracked generation retirement.
    DmaPass {
        bank: Arc<crate::experts::ExpertBank>,
        group: u64,
        layer: usize,
        place: usize,
        sequence: u64,
        window: u64,
        ready: ReadyNativeGraphRun<'a>,
    },
}

#[inline(never)]
#[allow(clippy::too_many_arguments)]
fn queue_attn_and_mix_step(
    charges: &Charges,
    b_idx: usize,
    a: &BoundBarAttnAndMix,
    bufs: &BarStepBuffers,
    ffn_in_buf: &Tensor,
    ffn_out_buf: &Tensor,
    ps: &PreparedChainStep,
    step: &BarChainStep<'_>,
    slot: &mut NativeGraphFamilySlot,
    on_compute: &mut impl FnMut(BarComputeAction<'_>) -> Result<(), String>,
) -> Result<(), String> {
    let layer_state = &step.layers[b_idx];
    let mut bindings = a.bound.bindings();
    bindings.set(charges, &a.sources_buf, &bufs.sources_buf).map_err(|e| e.to_string())?;
    if let (Some(p), Some(b)) = (&a.bank_buf, &bufs.bank_buf) {
        bindings.set(charges, p, b).map_err(|e| e.to_string())?;
    }
    if let (Some(p), Some(cp)) = (&a.cp_msgs_buf, &bufs.cp_msgs_buf) {
        bindings.set(charges, p, cp).map_err(|e| e.to_string())?;
    }
    if let (Some(p), Some(pre)) = (&a.prelude_partial, &bufs.prelude_partial) {
        bindings.set(charges, p, pre).map_err(|e| e.to_string())?;
    }
    if let Some(p) = &a.prev_ffn_out_buf {
        bindings.set(charges, p, ffn_out_buf).map_err(|e| e.to_string())?;
    }
    if let (Some(p), Some(w)) = (
        &a.prev_ffn_post_conv_window,
        b_idx.checked_sub(1).and_then(|pi| step.layers[pi].ffn_post_conv_window.as_ref()),
    ) {
        bindings.set(charges, p, w).map_err(|e| e.to_string())?;
    }
    if let Some(p) = &a.ffn_in_buf {
        if a.ffn_in_is_3d {
            bindings.set(charges, p, &bufs.ffn_in_3d).map_err(|e| e.to_string())?;
        } else {
            bindings.set(charges, p, ffn_in_buf).map_err(|e| e.to_string())?;
        }
    }
    bindings.set(charges, &a.coords, &ps.coords_t).map_err(|e| e.to_string())?;
    bindings.set(charges, &a.visible, &ps.vis_cache[ps.vis_for_layer[b_idx]]).map_err(|e| e.to_string())?;
    bindings.set(charges, &a.fresh, &ps.fresh_t).map_err(|e| e.to_string())?;
    bindings.set(charges, &a.destinations, &ps.dest_t).map_err(|e| e.to_string())?;
    if let Some(p) = &a.conv_segments {
        bindings.set(charges, p, &ps.conv_seg_t).map_err(|e| e.to_string())?;
    }
    if let Some(p) = &a.conv_stop {
        bindings.set(charges, p, &ps.conv_stop_t).map_err(|e| e.to_string())?;
    }
    if let Some(p) = &a.conv_prev_bank {
        bindings.set(charges, p, &ps.conv_prev_bank_t).map_err(|e| e.to_string())?;
    }
    if let Some(p) = &a.conv_prev_tape {
        bindings.set(charges, p, &ps.conv_prev_tape_t).map_err(|e| e.to_string())?;
    }
    if let Some(p) = &a.conv_next_bank {
        bindings.set(charges, p, &ps.conv_next_bank_t).map_err(|e| e.to_string())?;
    }
    bindings.set(charges, &a.history_key, &layer_state.history_key).map_err(|e| e.to_string())?;
    bindings.set(charges, &a.history_value, &layer_state.history_value).map_err(|e| e.to_string())?;
    if let (Some(p), Some(w)) = (&a.key_conv_window, &layer_state.key_conv_window) {
        bindings.set(charges, p, w).map_err(|e| e.to_string())?;
    }
    if let (Some(p), Some(w)) = (&a.value_conv_window, &layer_state.value_conv_window) {
        bindings.set(charges, p, w).map_err(|e| e.to_string())?;
    }
    if let (Some(p), Some(w)) = (&a.attn_post_conv_window, &layer_state.attn_post_conv_window) {
        bindings.set(charges, p, w).map_err(|e| e.to_string())?;
    }
    if let (Some(p), Some(tb)) = (&a.tap_buf, &step.tap_buf) {
        bindings.set(charges, p, tb).map_err(|e| e.to_string())?;
    }
    let mut active = slot.activate(&a.plan).map_err(|e| e.to_string())?;
    let ready = active
        .attach(bindings, a.plan.new_outputs(charges).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())?;
    on_compute(BarComputeAction::Queue {
        ready,
        register: Box::new(|_| Ok(())),
    })
}

#[inline(never)]
#[allow(clippy::too_many_arguments)]
fn queue_routed_decode_step(
    charges: &Charges,
    b_idx: usize,
    m: u64,
    use_resident: bool,
    ffn_in_buf: &Tensor,
    ffn_out_buf: &Tensor,
    prep_routed: &super::block_attn_res::PreparedStreamedRoutedBlock,
    bound_routed: &super::block_attn_res::BoundStreamedRoutedBlock,
    step: &mut BarChainStep<'_>,
    retained_layer_bufs: &mut Vec<Box<dyn std::any::Any + Send>>,
    on_compute: &mut impl FnMut(BarComputeAction<'_>) -> Result<(), String>,
) -> Result<(), String> {
    let stage = if use_resident {
        prep_routed.resident.get(&m)
    } else {
        prep_routed.decode.get(&m)
    }
    .ok_or_else(|| format!("routed graph missing for block {b_idx} rows {m}"))?;
    let bound = if use_resident {
        bound_routed.resident.get(&m)
    } else {
        bound_routed.decode.get(&m)
    }
    .ok_or_else(|| format!("routed binding missing for block {b_idx} rows {m}"))?;
    let mut bindings = bound.bindings();
    if let Some(port) = &stage.residual {
        bindings.set(charges, port, ffn_in_buf).map_err(|e| e.to_string())?;
    }
    if let Some(port) = &stage.output {
        bindings.set(charges, port, ffn_out_buf).map_err(|e| e.to_string())?;
    }
    let mut host_guesses_owned = None;
    if !use_resident && bound_routed.fill.is_some() {
        let br = step
            .bank_run
            .as_ref()
            .ok_or("missing BlockAttnResBankRun for streamed decode")?;
        for (bp, port) in &stage.banks {
            if bp.part.placed() {
                let tensor = br.bank.placed(bp.scope, bp.part, br.place)?;
                bindings.set(charges, port, &tensor).map_err(|e| e.to_string())?;
            }
        }
        for &(_, port) in &stage.parks {
            bindings.set_park(port, br.bank.parks().timeline()).map_err(|e| e.to_string())?;
        }
        if let Some(inputs) = &stage.step_inputs {
            if let Some(g_port) = &inputs.guesses {
                if let Some(g) = br.guesses {
                    bindings.set(charges, g_port, g).map_err(|e| e.to_string())?;
                } else {
                    let mut flat = vec![0i32; (m as usize) * 2];
                    for (i, pair) in br.host_guesses.iter().take(m as usize).enumerate() {
                        flat[2 * i] = pair[0];
                        flat[2 * i + 1] = pair[1];
                    }
                    let bytes: Vec<u8> = flat.iter().flat_map(|v| v.to_le_bytes()).collect();
                    let g_t = Tensor::from_host(charges, Element::i32(), &[m, 2], &bytes)
                        .map_err(|e| e.to_string())?;
                    bindings.set(charges, g_port, &g_t).map_err(|e| e.to_string())?;
                    host_guesses_owned = Some(g_t);
                }
            }
        }
    }
    if let Some(g_t) = host_guesses_owned {
        retained_layer_bufs.push(Box::new(g_t));
    }
    let mut active = step.routed_slot.activate(&stage.plan).map_err(|e| e.to_string())?;
    if let (Some(inputs), Some(br)) = (&stage.step_inputs, &step.bank_run) {
        let step_bytes: Vec<u8> = magnitude_kernels::step_words(br.sequence, br.front)
            .iter()
            .flat_map(|w| w.to_le_bytes())
            .collect();
        active
            .write_input(&inputs.sequence, &step_bytes)
            .map_err(|e| e.to_string())?;
    }
    let ready = active
        .attach(bindings, stage.plan.new_outputs(charges).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())?;
    if let Some(br) = &step.bank_run {
        let parks = stage.parks.clone();
        let bank_arc = br.bank.clone();
        let seq_val = br.sequence;
        on_compute(BarComputeAction::Queue {
            ready,
            register: Box::new(move |outputs| {
                for &(node, port) in &parks {
                    let Some(value) = outputs.park(port) else { continue };
                    match node {
                        super::GraphTimeline::Park(super::GraphPark { scope, kind }) => {
                            let layer = bank_arc.streamed_layer(scope)?;
                            bank_arc.parks().register(
                                value,
                                seq_val,
                                crate::experts::ParkSite { layer, kind },
                            );
                        }
                        super::GraphTimeline::Publish(scope) => {
                            let layer = bank_arc.streamed_layer(scope)?;
                            bank_arc.parks().publish(value, layer, seq_val);
                        }
                    }
                }
                Ok(())
            }),
        })
    } else {
        on_compute(BarComputeAction::Queue {
            ready,
            register: Box::new(|_| Ok(())),
        })
    }
}

pub(crate) fn queue_sealed_chain(
    bound_bar: &BoundBlockAttnRes,
    decoder: &Decoder,
    steps: &mut [BarChainStep<'_>],
    mut on_compute: impl FnMut(BarComputeAction<'_>) -> Result<(), String>,
) -> Result<(Vec<Tensor>, Box<dyn std::any::Any + Send>), String> {
    macro_rules! queue_compute {
        ($ready:expr, $reg:expr $(,)?) => {
            on_compute(BarComputeAction::Queue {
                ready: $ready,
                register: $reg,
            })
        };
    }
    let charges = &bound_bar.charges;
    let device = charges.device();
    let prepared = &bound_bar.prepared;
    let h = decoder.hidden;

    let upload_i32 = |shape: &[u64], vals: &[i32]| -> Result<Tensor, String> {
        let bytes: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        Tensor::from_host(charges, Element::i32(), shape, &bytes).map_err(|e| e.to_string())
    };
    let step_bytes = |seq: u64, front: u32| -> Vec<u8> {
        magnitude_kernels::step_words(seq, front)
            .iter()
            .flat_map(|w| w.to_le_bytes())
            .collect()
    };

    // Prepare families, scratch buffers, and once-per-step control uploads.
    let mut prepped: Vec<PreparedChainStep> = Vec::with_capacity(steps.len());
    for (s_idx, step) in steps.iter().enumerate() {
        prepped.push(prepare_chain_step(bound_bar, decoder, step, s_idx, steps.len())?);
    }

    // Borrow the decode ring if this is a 1-step decode launch (reused scratch);
    // otherwise allocate ONE shared `NativeGraphFamilySlot` + `ffn_in_buf` +
    // `ffn_out_buf` per distinct `m` across the whole chain (saving ~70+ MiB!).
    let ring_family = (prepped.len() == 1 && prepped[0].ring_idx.is_some())
        .then(|| prepped[0].family.clone());
    let mut ring_guard = ring_family
        .as_ref()
        .map(|f| f.decode_ring.lock().expect("bar decode ring lock"));
    let mut retained_layer_bufs: Vec<Box<dyn std::any::Any + Send>> = Vec::new();
    let mut returned_grouped_bufs: Vec<(
        Arc<BoundBarFamily>,
        crate::operators::routed::split::SplitLayer<u64>,
        GroupedStepBuf,
    )> = Vec::new();
    let mut returned_merged_bufs: Vec<(
        Arc<BoundBarFamily>,
        crate::operators::routed::split::SplitLayer<u64>,
        GroupedMergedBuf,
    )> = Vec::new();
    let mut cached_grouped_bufs: Option<(
        crate::operators::routed::split::SplitLayer<u64>,
        Vec<GroupedStepBuf>,
        HashMap<usize, (Arc<BoundBarFamily>, GroupedMergedBuf)>,
    )> = None;
    let mut shared_slots: HashMap<usize, (Arc<BoundBarFamily>, NativeGraphFamilySlot)> =
        HashMap::new();
    if ring_guard.is_none() {
        let active_fids: Vec<usize> = prepped
            .iter()
            .map(|ps| Arc::as_ptr(&ps.family) as usize)
            .collect();
        for ps in &prepped {
            let fid = Arc::as_ptr(&ps.family) as usize;
            if !shared_slots.contains_key(&fid) {
                let reused = ps
                    .family
                    .prefill_slot
                    .lock()
                    .expect("bar prefill_slot lock")
                    .take();
                let slot = match reused {
                    Some(s) => s,
                    None => {
                        // Release cached prefill slots and scratch pools held by
                        // other chunk-size families so switching chunk sizes (e.g.
                        // m=512 -> m=128) stays within the single-slot budget.
                        if let Ok(families) = bound_bar.families.lock() {
                            for other in families.values() {
                                let ofid = Arc::as_ptr(other) as usize;
                                if active_fids.contains(&ofid) {
                                    continue;
                                }
                                if let Ok(mut g) = other.prefill_slot.lock() {
                                    let _ = g.take();
                                }
                                if let Ok(mut g) = other.prefill_bufs_pool.lock() {
                                    g.clear();
                                }
                                if let Ok(mut g) = other.prefill_grouped_pool.lock() {
                                    g.clear();
                                }
                                if let Ok(mut g) = other.prefill_merged_pool.lock() {
                                    g.clear();
                                }
                            }
                        }
                        ps.family
                            .family
                            .new_slot(charges, &[])
                            .map_err(|e| format!("bar shared_slot(m={}): {e}", ps.m))?
                    }
                };
                shared_slots.insert(fid, (ps.family.clone(), slot));
            }
        }
    }

    macro_rules! step_parts {
        ($ps:expr) => {
            if let Some(idx) = $ps.ring_idx {
                let sc = &mut ring_guard.as_mut().unwrap().1[idx];
                (&sc.bufs, &sc.bufs.ffn_in_buf, &sc.bufs.ffn_out_buf, &mut sc.slot)
            } else {
                let fid = Arc::as_ptr(&$ps.family) as usize;
                let (_, slot) = shared_slots.get_mut(&fid).unwrap();
                let b = match $ps.prefill_idx {
                    Some(pidx) => &bound_bar.prefill_bufs[pidx],
                    None => $ps.bufs.as_ref().unwrap(),
                };
                (b, &b.ffn_in_buf, &b.ffn_out_buf, slot)
            }
        };
    }

    // 1. Entry for every step in the chain.
    for (s_idx, step) in steps.iter().enumerate() {
        let ps = &mut prepped[s_idx];
        let (bufs, _, _, slot) = step_parts!(ps);
        let e = &ps.family.entry;
        let mut bindings = e.bound.bindings();
        bindings.set(charges, &e.entry_hidden, &step.entry_hidden).map_err(|err| err.to_string())?;
        bindings.set(charges, &e.sources_buf, &bufs.sources_buf).map_err(|err| err.to_string())?;
        let mut active = slot.activate(&e.plan).map_err(|err| err.to_string())?;
        let ready = active
            .attach(bindings, e.plan.new_outputs(charges).map_err(|err| err.to_string())?)
            .map_err(|err| err.to_string())?;
        if step.media_overlays.is_empty() {
            queue_compute!(ready, Box::new(|_| Ok(())))?;
        } else {
            let mut seq = device.native_sequence(NativeQueue::Compute);
            let _ = ready.queue(&mut seq).map_err(|err| err.to_string())?;
            seq.submit().map_err(|err| err.to_string())?.wait().map_err(|err| err.to_string())?;
            let mut slot0 = bufs
                .sources_buf
                .slice_leading(0, 1)
                .map_err(|err| err.to_string())?
                .reshape(&[ps.m, h])
                .map_err(|err| err.to_string())?;
            let bytes = slot0.read_to_host().map_err(|err| err.to_string())?;
            let mut host: Vec<f32> = bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect();
            for (start, end, values) in &step.media_overlays {
                let lo = start * h as usize;
                let hi = end * h as usize;
                host[lo..hi].copy_from_slice(values);
            }
            let out_bytes: Vec<u8> = host.iter().flat_map(|v| v.to_le_bytes()).collect();
            slot0.write_from_host(&out_bytes).map_err(|err| err.to_string())?;
        }
    }

    // 2. Layer-by-layer wavefront across all blocks `0..decoder.blocks.len()`.
    for b_idx in 0..decoder.blocks.len() {
        // 2a. AttnAndMix (with folded previous block's FfnPost) for every step.
        for (s_idx, step) in steps.iter().enumerate() {
            let ps = &mut prepped[s_idx];
            let (bufs, ffn_in_buf, ffn_out_buf, slot) = step_parts!(ps);
            queue_attn_and_mix_step(
                charges,
                b_idx,
                &ps.family.blocks[b_idx],
                bufs,
                ffn_in_buf,
                ffn_out_buf,
                ps,
                step,
                slot,
                &mut on_compute,
            )?;
        }

        // 2b. Routed FFN (wavefront for streamed grouped prefill!).
        if let Some(bound_routed) = bound_bar.streamed_routed.get(&b_idx) {
            let prep_routed = prepared
                .streamed_routed
                .get(&b_idx)
                .ok_or_else(|| format!("missing prepared routed block {b_idx}"))?;
            let ffn_scope = WeightScope::TargetSublayer(SublayerIndex {
                block: b_idx as u32,
                sublayer: 1,
            });
            let use_resident = !bound_routed.resident.is_empty()
                && steps
                    .first()
                    .and_then(|s| s.bank_run.as_ref())
                    .is_some_and(|br| br.all_whole);
            let any_streamed_grouped = !use_resident
                && bound_routed.fill.is_some()
                && prepped.iter().any(|ps| !decodes(ps.m));

            if use_resident || !any_streamed_grouped {
                // Resident, whole-bank, or streamed decode: single sealed graph per step.
                for (s_idx, step) in steps.iter_mut().enumerate() {
                    let ps = &mut prepped[s_idx];
                    let (_, ffn_in_buf, ffn_out_buf, _) = step_parts!(ps);
                    queue_routed_decode_step(
                        charges,
                        b_idx,
                        ps.m,
                        use_resident,
                        ffn_in_buf,
                        ffn_out_buf,
                        prep_routed,
                        bound_routed,
                        step,
                        &mut retained_layer_bufs,
                        &mut on_compute,
                    )?;
                }
            } else {
                // Streamed grouped prefill wavefront:
                //   (i)   Front for EVERY step (publishes all N requests at layer b_idx)
                //   (ii)  For each step: per-window Fill + Pass
                //   (iii) For each step: Back
                let Operator::RoutedFfn(routed) = &decoder.blocks[b_idx].sublayers[1].op else {
                    return Err("expected RoutedFfn".into());
                };
                let (split_form, split_layer) = split_shape(routed, decoder);
                let split_layer_dim = constant(split_layer.clone());

                // Reuse per-step crossing buffers AND a single shared merged
                // buffer per family across all layers with the same `split_layer`
                // and across prefills via `BoundBarFamily`'s persistent pools.
                // Sharing one `merged_experts` per family (since `Pass -> Back`
                // runs sequentially per step on the compute stream) avoids
                // allocating N * ~37 MiB per prefill chain.
                if cached_grouped_bufs.as_ref().is_none_or(|(sl, _, _)| *sl != split_layer) {
                    if let Some((old_sl, old_vec, old_merged)) = cached_grouped_bufs.take() {
                        for (ps, buf) in prepped.iter().zip(old_vec.into_iter()) {
                            returned_grouped_bufs.push((ps.family.clone(), old_sl.clone(), buf));
                        }
                        for (_, (fam, mbuf)) in old_merged {
                            returned_merged_bufs.push((fam, old_sl.clone(), mbuf));
                        }
                    }
                    let mut bufs = Vec::with_capacity(steps.len());
                    let mut merged_map: HashMap<usize, (Arc<BoundBarFamily>, GroupedMergedBuf)> =
                        HashMap::new();
                    for ps in &prepped {
                        let m = ps.m;
                        let crossing_bufs = {
                            let reused = ps
                                .family
                                .prefill_grouped_pool
                                .lock()
                                .expect("bar prefill_grouped_pool lock")
                                .get_mut(&split_layer)
                                .and_then(Vec::pop);
                            match reused {
                                Some(b) => b,
                                None => {
                                    let crossing_ext =
                                        Crossing::at(&split_form, &split_layer_dim, &Dim::from(m));
                                    crossing_ext.try_map(|(elem, dims)| {
                                        let exts: Vec<u64> = dims.iter().map(Dim::fixed).collect();
                                        Tensor::zeros(charges, *elem, &exts).map_err(|e| {
                                            format!("bar crossing(m={m}, exts={exts:?}): {e}")
                                        })
                                    })?
                                }
                            }
                        };
                        bufs.push(crossing_bufs);
                        let fid = Arc::as_ptr(&ps.family) as usize;
                        if !merged_map.contains_key(&fid) {
                            let reused = ps
                                .family
                                .prefill_merged_pool
                                .lock()
                                .expect("bar prefill_merged_pool lock")
                                .get_mut(&split_layer)
                                .and_then(Vec::pop);
                            let mbuf = match reused {
                                Some(b) => b,
                                None => {
                                    let merged_ext = Merged::of(routed, decoder, m)?;
                                    let merged_experts = Tensor::zeros(
                                        charges,
                                        merged_ext.experts.0,
                                        &merged_ext.experts.1,
                                    )
                                    .map_err(|e| format!("bar merged_experts(m={m}): {e}"))?;
                                    let merged_inverse = Tensor::zeros(
                                        charges,
                                        merged_ext.inverse.0,
                                        &merged_ext.inverse.1,
                                    )
                                    .map_err(|e| format!("bar merged_inverse(m={m}): {e}"))?;
                                    let table_vals: Vec<i32> = (0..=m as i32).collect();
                                    let merged_table =
                                        upload_i32(&merged_ext.table.1, &table_vals)?;
                                    (merged_experts, merged_inverse, merged_table)
                                }
                            };
                            merged_map.insert(fid, (ps.family.clone(), mbuf));
                        }
                    }
                    cached_grouped_bufs = Some((split_layer.clone(), bufs, merged_map));
                }
                let (_, per_step_bufs, shared_merged_bufs) = cached_grouped_bufs.as_ref().unwrap();

                let bind_dynamic = |br: &BlockAttnResBankRun<'_>,
                                    stage: &super::block_attn_res::PreparedStreamedStage,
                                    place_override: usize,
                                    window: Option<u64>,
                                    bindings: &mut NativeGraphBindings|
                 -> Result<(), String> {
                    for (bp, port) in &stage.banks {
                        if bp.part.placed() {
                            let tensor = br.bank.placed(bp.scope, bp.part, place_override)?;
                            bindings.set(charges, port, &tensor).map_err(|e| e.to_string())?;
                        } else if bp.part.windowed() {
                            let w = window.ok_or("missing window index")?;
                            let tensor = br.bank.windowed(bp.scope, bp.part, w)?;
                            bindings.set(charges, port, &tensor).map_err(|e| e.to_string())?;
                        }
                    }
                    for &(_, port) in &stage.parks {
                        bindings.set_park(port, br.bank.parks().timeline()).map_err(|e| e.to_string())?;
                    }
                    Ok(())
                };

                let register_grouped_parks = |br: &BlockAttnResBankRun<'_>,
                                              stage: &super::block_attn_res::PreparedStreamedStage,
                                              window: Option<u64>,
                                              outputs: &NativeGraphOutputs|
                 -> Result<(), String> {
                    for &(node, port) in &stage.parks {
                        let Some(value) = outputs.park(port) else { continue };
                        match node {
                            super::GraphTimeline::Park(super::GraphPark { scope, kind }) => {
                                let kind = match (window, kind) {
                                    (Some(w), crate::experts::ParkKind::Round(round)) => {
                                        crate::experts::ParkKind::Round(
                                            w * br.bank.plan().window_rounds() + round,
                                        )
                                    }
                                    (_, kind) => kind,
                                };
                                let layer = br.bank.streamed_layer(scope)?;
                                br.bank.parks().register(
                                    value,
                                    br.sequence,
                                    crate::experts::ParkSite { layer, kind },
                                );
                            }
                            super::GraphTimeline::Publish(scope) => {
                                let layer = br.bank.streamed_layer(scope)?;
                                br.bank.parks().publish(value, layer, br.sequence);
                            }
                        }
                    }
                    Ok(())
                };

                // Phase (i): Front for EVERY step in the chain.
                for (s_idx, step) in steps.iter_mut().enumerate() {
                    let ps = &mut prepped[s_idx];
                    let (_, ffn_in_buf, _, _) = step_parts!(ps);
                    let m = ps.m;
                    let br = step.bank_run.as_ref().ok_or("missing BlockAttnResBankRun")?;
                    let (front_s, _, _) = prep_routed
                        .grouped
                        .get(&m)
                        .ok_or_else(|| format!("streamed grouped graphs missing for block {b_idx} rows {m}"))?;
                    let (front_b, _, _) = bound_routed
                        .grouped
                        .get(&m)
                        .ok_or_else(|| format!("streamed grouped bindings missing for block {b_idx} rows {m}"))?;
                    let crossing_bufs = &per_step_bufs[s_idx];
                    let mut bindings = front_b.bindings();
                    if let Some(port) = &front_s.residual {
                        bindings.set(charges, port, ffn_in_buf).map_err(|e| e.to_string())?;
                    }
                    if let Some(SplitPorts::Front(ports)) = &front_s.split {
                        for (port, buf) in ports.iter().zip(crossing_bufs.iter()) {
                            bindings.set(charges, port, buf).map_err(|e| e.to_string())?;
                        }
                    }
                    bind_dynamic(br, front_s, br.place, None, &mut bindings)?;
                    let mut active = step.routed_slot.activate(&front_s.plan).map_err(|e| e.to_string())?;
                    if let Some(inputs) = &front_s.step_inputs {
                        active
                            .write_input(&inputs.sequence, &step_bytes(br.sequence, br.front))
                            .map_err(|e| e.to_string())?;
                    }
                    let ready = active
                        .attach(bindings, front_s.plan.new_outputs(charges).map_err(|e| e.to_string())?)
                        .map_err(|e| e.to_string())?;
                    on_compute(BarComputeAction::Queue {
                        ready,
                        register: Box::new(move |outputs| {
                            register_grouped_parks(br, front_s, None, outputs)
                        }),
                    })?;
                }
                // Flush every step's `Front` to the device before queuing any
                // `Fill`/`Pass` (and expose `Front` on boundary-parked backends
                // under `tier_copy_dma`).
                if let Some(first_br) = steps.first().and_then(|s| s.bank_run.as_ref()) {
                    let layer_idx = first_br.bank.streamed_layer(ffn_scope)?;
                    let max_sequence = steps
                        .iter()
                        .filter_map(|s| s.bank_run.as_ref().map(|br| br.sequence))
                        .max()
                        .unwrap_or(first_br.sequence);
                    on_compute(BarComputeAction::FlushFront {
                        bank: first_br.bank.clone(),
                        layer: layer_idx,
                        max_sequence,
                    })?;
                }

                // Phase (ii) + (iii): Per-step per-window Fill + Pass, followed
                // immediately by Back for that step so every step in the chain
                // shares one `merged_experts` buffer per family on the compute stream.
                for (s_idx, step) in steps.iter_mut().enumerate() {
                    let ps = &mut prepped[s_idx];
                    let m = ps.m;
                    let fid = Arc::as_ptr(&ps.family) as usize;
                    let br = step.bank_run.as_ref().ok_or("missing BlockAttnResBankRun")?;
                    let (_, pass_s, back_s) = prep_routed.grouped.get(&m).unwrap();
                    let (_, pass_b, back_b) = bound_routed.grouped.get(&m).unwrap();
                    let crossing_bufs = &per_step_bufs[s_idx];
                    let (_, (ref merged_experts, ref merged_inverse, ref merged_table)) =
                        shared_merged_bufs[&fid];
                    let layer_idx = br.bank.streamed_layer(ffn_scope)?;
                    let tier_copy_dma = br.bank.tier_copy_dma();
                    let pass_place = br
                        .group_reads
                        .map_or(br.place, |record| br.bank.group_reads_at(record, br.place));
                    for window in 0..br.bank.plan().windows() {
                        if !tier_copy_dma {
                            br.bank.fill_followed(br.sequence, layer_idx, window, false);
                            // Non-DMA Fill (`OnceStage::Fill`).
                            let fill_prep = prep_routed.fill.as_ref().ok_or("missing fill plan")?;
                            let fill_bound = bound_routed.fill.as_ref().ok_or("missing fill bound")?;
                            let mut bindings = fill_bound.bindings();
                            bind_dynamic(br, fill_prep, br.place, Some(window), &mut bindings)?;
                            let mut active = step.routed_slot.activate(&fill_prep.plan).map_err(|e| e.to_string())?;
                            let ready = active
                                .attach(bindings, fill_prep.plan.new_outputs(charges).map_err(|e| e.to_string())?)
                                .map_err(|e| e.to_string())?;
                            if device.backend() == seismic::BackendName::Cuda && device.has_side_queue() {
                                let mut seq = device.native_sequence(NativeQueue::Fill);
                                let outputs = ready.queue(&mut seq).map_err(|e| e.to_string())?;
                                register_grouped_parks(br, fill_prep, Some(window), &outputs)?;
                                let _ = seq.submit().map_err(|e| e.to_string())?;
                            } else {
                                queue_compute!(
                                    ready,
                                    Box::new(move |outputs| {
                                        register_grouped_parks(br, fill_prep, Some(window), outputs)
                                    }),
                                )?;
                            }
                        }
                        // Pass
                        {
                            let mut bindings = pass_b.bindings();
                            if let Some(into) = &pass_s.pass_ports {
                                for (port, buf) in into.crossing.iter().zip(crossing_bufs.pass().iter()) {
                                    bindings.set(charges, port, buf).map_err(|e| e.to_string())?;
                                }
                                bindings.set(charges, &into.experts, merged_experts).map_err(|e| e.to_string())?;
                                bindings.set(charges, &into.inverse, merged_inverse).map_err(|e| e.to_string())?;
                                let rows_t = merged_table.slice_leading(m, m + 1).map_err(|e| e.to_string())?;
                                let first_t = merged_table.slice_leading(0, 1).map_err(|e| e.to_string())?;
                                bindings.set(charges, &into.rows, &rows_t).map_err(|e| e.to_string())?;
                                bindings.set(charges, &into.first, &first_t).map_err(|e| e.to_string())?;
                            }
                            bind_dynamic(br, pass_s, pass_place, Some(window), &mut bindings)?;
                            let mut active = step.routed_slot.activate(&pass_s.plan).map_err(|e| e.to_string())?;
                            let ready = active
                                .attach(bindings, pass_s.plan.new_outputs(charges).map_err(|e| e.to_string())?)
                                .map_err(|e| e.to_string())?;
                            if tier_copy_dma {
                                on_compute(BarComputeAction::DmaPass {
                                    bank: br.bank.clone(),
                                    group: br.sequence.saturating_sub(br.place as u64),
                                    layer: layer_idx,
                                    place: br.place,
                                    sequence: br.sequence,
                                    window,
                                    ready,
                                })?;
                            } else {
                                queue_compute!(ready, Box::new(|_| Ok(())))?;
                            }
                        }
                    }
                    // Back for this step immediately consumes `merged_experts`.
                    let (_, ffn_in_buf, ffn_out_buf, _) = step_parts!(ps);
                    let mut bindings = back_b.bindings();
                    if let Some(port) = &back_s.residual {
                        bindings.set(charges, port, ffn_in_buf).map_err(|e| e.to_string())?;
                    }
                    if let Some(port) = &back_s.output {
                        bindings.set(charges, port, ffn_out_buf).map_err(|e| e.to_string())?;
                    }
                    if let Some(SplitPorts::Back(ports)) = &back_s.split {
                        for (port, buf) in ports.crossing.iter().zip(crossing_bufs.back().iter()) {
                            bindings.set(charges, port, buf).map_err(|e| e.to_string())?;
                        }
                        bindings.set(charges, &ports.experts, merged_experts).map_err(|e| e.to_string())?;
                        bindings.set(charges, &ports.inverse, merged_inverse).map_err(|e| e.to_string())?;
                    }
                    bind_dynamic(br, back_s, br.place, None, &mut bindings)?;
                    let mut active = step.routed_slot.activate(&back_s.plan).map_err(|e| e.to_string())?;
                    let ready = active
                        .attach(bindings, back_s.plan.new_outputs(charges).map_err(|e| e.to_string())?)
                        .map_err(|e| e.to_string())?;
                    queue_compute!(ready, Box::new(|_| Ok(())))?;
                }
            }
        }

    }

    // 3. Exit (with folded last block's FfnPost) for every step.
    let mut exit_hiddens = Vec::with_capacity(steps.len());
    for (s_idx, step) in steps.iter().enumerate() {
        let ps = &mut prepped[s_idx];
        let (bufs, _, _, slot) = step_parts!(ps);
        let exit_hidden = queue_exit_step(
            charges,
            &ps.family.exit,
            bufs,
            ps,
            step.layers.last(),
            slot,
            step.tap_buf.as_ref(),
            &mut |r, reg| {
                on_compute(BarComputeAction::Queue {
                    ready: r,
                    register: reg,
                })
            },
        )?;
        exit_hiddens.push(exit_hidden);
    }

    // Return prefill workspace slots, non-standard `BarStepBuffers`, and grouped
    // FFN scratch to their `BoundBarFamily` on drop (once the submitted sequence
    // completes) so subsequent prefills reuse them with zero alloc/free churn.
    if let Some((sl, vec, merged_map)) = cached_grouped_bufs.take() {
        for (ps, buf) in prepped.iter().zip(vec.into_iter()) {
            returned_grouped_bufs.push((ps.family.clone(), sl.clone(), buf));
        }
        for (_, (fam, mbuf)) in merged_map {
            returned_merged_bufs.push((fam, sl.clone(), mbuf));
        }
    }
    let returned_bufs: Vec<(Arc<BoundBarFamily>, BarStepBuffers)> = prepped
        .iter_mut()
        .filter_map(|ps| ps.bufs.take().map(|b| (ps.family.clone(), b)))
        .collect();
    struct ReturnPrefillResources {
        slots: Vec<(Arc<BoundBarFamily>, NativeGraphFamilySlot)>,
        bufs: Vec<(Arc<BoundBarFamily>, BarStepBuffers)>,
        grouped: Vec<(
            Arc<BoundBarFamily>,
            crate::operators::routed::split::SplitLayer<u64>,
            GroupedStepBuf,
        )>,
        merged: Vec<(
            Arc<BoundBarFamily>,
            crate::operators::routed::split::SplitLayer<u64>,
            GroupedMergedBuf,
        )>,
    }
    impl Drop for ReturnPrefillResources {
        fn drop(&mut self) {
            for (fam, slot) in self.slots.drain(..) {
                if let Ok(mut g) = fam.prefill_slot.lock() {
                    if g.is_none() {
                        *g = Some(slot);
                    }
                }
            }
            for (fam, buf) in self.bufs.drain(..) {
                if let Ok(mut g) = fam.prefill_bufs_pool.lock() {
                    g.push(buf);
                }
            }
            for (fam, sl, gbuf) in self.grouped.drain(..) {
                if let Ok(mut g) = fam.prefill_grouped_pool.lock() {
                    g.entry(sl).or_default().push(gbuf);
                }
            }
            for (fam, sl, mbuf) in self.merged.drain(..) {
                if let Ok(mut g) = fam.prefill_merged_pool.lock() {
                    g.entry(sl).or_default().push(mbuf);
                }
            }
        }
    }

    drop(ring_guard);
    let returned = ReturnPrefillResources {
        slots: shared_slots.into_values().collect(),
        bufs: returned_bufs,
        grouped: returned_grouped_bufs,
        merged: returned_merged_bufs,
    };
    let retained: Box<dyn std::any::Any + Send> =
        Box::new((prepped, returned, retained_layer_bufs));
    Ok((exit_hiddens, retained))
}
