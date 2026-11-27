//! Live execution of `ResidualForm::BlockAttention` (Inkling 2) decoders inside
//! `PreparedTargetGraphs` / `BoundTargetGraphs` / `NativeTargetProgram`.
//!
//! All residual sources (`blocks`, `partial`, `prelude`, `bank`) stay in F32
//! to prevent correlated rounding error accumulation across looped core passes.
//! Supports resident weights (BF16/F16/Q8/NVFP4 with `weight_scale`/`weight_bound`/
//! `expert_scale`) and `ExpertBank` (both whole-home and streamed with Lane B's
//! decoupled `physical_stack` vs `visit_layer`).

use crate::native::{AttestedFeedForward, AttestedTarget};
#[cfg(test)]
use crate::operators;
use crate::operators::{
    block_attn_res::BlockAttnResPlan,
    routed::{
        GeneralRoutedShape,
        fused_graph::decodes,
        graph::{
            GeneralRoutedGraphEntries, RoutedSum, general_pass, general_routed,
            split as general_split,
        },
        split::{PassPorts, RunStage, SplitBuild, SplitPorts, split as split_shape},
        streamed::{BankGraph, BankGraphEntries, LayerBank, StepInputs, scratch_fill},
    },
};
use crate::programs::graph::{
    GraphError,
    draft::GraphDraft,
    facts::{Numbers, constant},
};
use crate::programs::native_constants::{ConstantTensors, GraphConstant};
use crate::{ModelLoadPlan, ResidentTarget, ResidentWeight};
#[cfg(test)]
use magnitude_family_contracts::ExitNorm;
use magnitude_family_contracts::{
    BlockAttentionStream, CrossPassInput, Decoder, EntryNorm, InputNorm, Operator, SublayerIndex,
    TapPoint, WeightKind, WeightRole, WeightScope,
};
use magnitude_kernels::{
    attention_rel_bias, block_attn_res_mix, causal_short_conv, embedding_rows, feature_copy,
    head_l2_clip, message_gate_mix, message_gate_mix_norm, post_norm_residual, project_rows,
    readout_features_rows, rel_bias_project, rms_concat, stream_join, tap_rows, vision_norm,
};
#[cfg(test)]
use seismic::NativeGraphFamilySlot;
use seismic::{
    Argument, BoundNativeGraphPlan, Charges, Device, Dim, Element, NativeGraphPlan, NativeKernel,
    NativePort, NativeSpecialization, NativeTimelinePort, Tensor,
};
use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

/// Bound weight tensor with optional NVFP4 second-level scale and packing bound.
#[derive(Clone)]
pub(crate) struct BoundWeight {
    pub tensor: Tensor,
    pub scale: Option<Tensor>,
    pub bound: Option<Tensor>,
}

impl std::ops::Deref for BoundWeight {
    type Target = Tensor;
    fn deref(&self) -> &Tensor {
        &self.tensor
    }
}

impl BoundWeight {
    #[cfg(test)]
    fn plain(tensor: Tensor) -> Self {
        Self {
            tensor,
            scale: None,
            bound: None,
        }
    }

    pub(crate) fn from_resident(w: &ResidentWeight) -> Self {
        Self {
            tensor: w.tensor().clone(),
            scale: w.scale().cloned(),
            bound: w.bound().cloned(),
        }
    }

    pub(crate) fn element(&self) -> Element {
        self.tensor.element()
    }

    pub(crate) fn scale_extent(&self) -> u64 {
        self.scale.as_ref().map_or(0, |t| t.extents()[0])
    }

    pub(crate) fn bound_extent(&self) -> u64 {
        self.bound.as_ref().map_or(0, |t| t.extents()[0])
    }
}

/// Per-layer persistent KV history and causal short-conv state arenas.
#[derive(Clone)]
pub(crate) struct BlockAttnResLayerState {
    pub history_key: Tensor,
    pub history_value: Tensor,
    pub key_conv_window: Option<Tensor>,
    pub value_conv_window: Option<Tensor>,
    pub attn_post_conv_window: Option<Tensor>,
    pub ffn_post_conv_window: Option<Tensor>,
    pub slab_rows: u32,
    pub slab_banks: u32,
}

/// A single sealed streamed-routed graph stage (decode, grouped front/pass/back,
/// or scratch fill) built via `operators::routed` so `BlockAttnRes` shares the
/// exact streaming/landing protocol with `NativeTargetGraph`.
#[derive(Clone)]
pub(super) struct PreparedStreamedStage {
    pub(super) plan: NativeGraphPlan,
    pub(super) residual: Option<NativePort>,
    pub(super) output: Option<NativePort>,
    pub(super) weights: Vec<(super::WeightPort, NativePort)>,
    pub(super) banks: Vec<(super::BankPort, NativePort)>,
    pub(super) parks: Vec<(
        super::GraphTimeline,
        NativeTimelinePort<seismic::timeline::Parks>,
    )>,
    pub(super) constants: Vec<GraphConstant>,
    pub(super) step_inputs: Option<StepInputs>,
    pub(super) split: Option<SplitPorts>,
    pub(super) pass_ports: Option<PassPorts>,
}

/// Prepared `operators::routed` FFN graphs for one `BlockAttnRes` block
/// (either streamed `ExpertBank`, whole-bank `Home`, or resident `Held`).
#[derive(Clone)]
pub(super) struct PreparedStreamedRoutedBlock {
    pub(super) decode: BTreeMap<u64, PreparedStreamedStage>,
    pub(super) resident: BTreeMap<u64, PreparedStreamedStage>,
    pub(super) grouped: BTreeMap<
        u64,
        (
            PreparedStreamedStage,
            PreparedStreamedStage,
            PreparedStreamedStage,
        ),
    >,
    pub(super) fill: Option<PreparedStreamedStage>,
}

/// Statically bound `operators::routed` FFN plans for one `BlockAttnRes` block.
pub(super) struct BoundStreamedRoutedBlock {
    pub(super) decode: BTreeMap<u64, BoundNativeGraphPlan>,
    pub(super) resident: BTreeMap<u64, BoundNativeGraphPlan>,
    pub(super) grouped: BTreeMap<
        u64,
        (
            BoundNativeGraphPlan,
            BoundNativeGraphPlan,
            BoundNativeGraphPlan,
        ),
    >,
    pub(super) fill: Option<BoundNativeGraphPlan>,
}

/// Active `ExpertBank` step context for one `BlockAttnRes` launch.
pub(crate) struct BlockAttnResBankRun<'a> {
    pub bank: &'a Arc<crate::experts::ExpertBank>,
    pub sequence: u64,
    pub front: u32,
    pub place: usize,
    pub group_reads: Option<usize>,
    pub guesses: Option<&'a Tensor>,
    pub host_guesses: &'a [[i32; 2]],
    pub all_whole: bool,
}

/// Prepared kernels for live execution of a `ResidualForm::BlockAttention`
/// decoder on any Seismic device (Metal, CUDA, Vulkan, CPU).
#[derive(Clone)]
pub(crate) struct PreparedBlockAttnRes {
    pub(super) hidden: u64,
    pub(super) activation: Element,
    pub(crate) plan: BlockAttnResPlan,
    pub(super) schedule: super::block_attn_res_graph::BarSchedule,
    pub(crate) tap_points: Vec<TapPoint>,
    pub(super) mix: NativeKernel<block_attn_res_mix::Entry>,
    pub(super) rms_concat: NativeKernel<rms_concat::Entry>,
    pub(super) message_gate_mix: NativeKernel<message_gate_mix::Entry>,
    pub(super) message_gate_mix_norm: NativeKernel<message_gate_mix_norm::Entry>,
    pub(super) conv_aa: NativeKernel<causal_short_conv::Entry>,
    pub(super) conv_af: NativeKernel<causal_short_conv::Entry>,
    pub(super) conv_ff: NativeKernel<causal_short_conv::Entry>,
    pub(super) head_l2_clip: NativeKernel<head_l2_clip::Entry>,
    pub(super) attention_rel_bias: NativeKernel<attention_rel_bias::Entry>,
    pub(super) stream_join: NativeKernel<stream_join::Entry>,
    pub(super) entry_norm: Option<NativeKernel<post_norm_residual::Entry>>,
    pub(super) embedding: NativeKernel<embedding_rows::Entry>,
    pub(super) norm_rows: NativeKernel<readout_features_rows::Entry>,
    pub(super) f32_copy: NativeKernel<feature_copy::Entry>,
    pub(super) tap_rows: HashMap<Element, NativeKernel<tap_rows::Entry>>,
    pub(super) rel_bias_projects: HashMap<(Element, Element), NativeKernel<rel_bias_project::Entry>>,
    pub(super) v_norms: HashMap<(u64, u64), NativeKernel<vision_norm::Entry>>,
    /// Keyed by `(A, W, Y, K, N, WS, WB)`.
    pub(super) projects:
        HashMap<(Element, Element, Element, u64, u64, u64, u64), NativeKernel<project_rows::Entry>>,
    /// Sealed streamed-`ExpertBank` routed FFN graphs, keyed by target block index.
    pub(super) streamed_routed: BTreeMap<usize, PreparedStreamedRoutedBlock>,
    /// Unique `(SplitLayer, SplitForm)` pairs across banked routed blocks, used
    /// by `PreparedTargetGraphs::group_bytes` to reserve prefill crossing and
    /// merged buffer headroom on elastic bank admission.
    pub(super) split_shapes: HashMap<
        crate::operators::routed::split::SplitLayer<u64>,
        crate::operators::routed::split::SplitForm,
    >,
}

/// Weight-bound `ResidualForm::BlockAttention` executor owned by `BoundTargetGraphs`.
pub(crate) struct BoundBlockAttnRes {
    pub(crate) prepared: Arc<PreparedBlockAttnRes>,
    pub(crate) weights: HashMap<WeightRole, BoundWeight>,
    pub(super) streamed_routed: BTreeMap<usize, BoundStreamedRoutedBlock>,
    _streamed_constants: Vec<Tensor>,
    pub(super) prefill_rows: u64,
    pub(super) prefill_bufs: Vec<super::block_attn_res_graph::BarStepBuffers>,
    /// Pre-allocated 1-place floor streamed grouped-prefill buffers at
    /// `prefill_rows` (committed in `CommittedKind::Arena` so single-place
    /// prefill never lazily OOMs against elastic expert-bank growth).
    pub(super) prefill_floor_grouped: HashMap<
        crate::operators::routed::split::SplitLayer<u64>,
        super::block_attn_res_graph::GroupedStepBuf,
    >,
    pub(super) prefill_floor_merged: HashMap<
        crate::operators::routed::split::SplitLayer<u64>,
        super::block_attn_res_graph::GroupedMergedBuf,
    >,
    pub(super) families: std::sync::Mutex<
        HashMap<
            super::block_attn_res_graph::BarLayoutKey,
            Arc<super::block_attn_res_graph::BoundBarFamily>,
        >,
    >,
    pub(super) fused_proj_cache: std::sync::Mutex<
        HashMap<Vec<super::block_attn_res_graph::FusedWeightKey>, Tensor>,
    >,
    pub(crate) charges: Charges,
}

/// Per-step inputs for `BoundBlockAttnRes::step_hidden`.
#[cfg(test)]
pub(crate) struct BlockAttnResStepInput<'a> {
    pub tokens: &'a [i32],
    /// Position coordinate for each token (`[M, 4] i32`, col 0 = position).
    pub coordinates: &'a [i32],
    /// Optional `(start_row, end_row, features_f32)` media overlays that bypass
    /// `EntryNorm::Rms { text_only: true }`.
    pub media_overlays: &'a [(usize, usize, &'a [f32])],
    /// Visible history ranges `[M, R, 2]` per layer (in `0..T`).
    pub visible_per_layer: &'a [Vec<i32>],
    /// Number of visible spans `R` per row.
    pub spans: u64,
    /// Fresh ranges `[M, 2]` (same for every layer).
    pub fresh: &'a [i32],
    /// Destination KV slots `[M]` per layer (`-1` to skip append).
    pub destinations: &'a [i32],
    /// Recurrent slot controls for `causal_short_conv`.
    pub conv_segments: &'a [i32],
    pub conv_stop: &'a [i32],
    pub conv_prev_bank: &'a [i32],
    pub conv_prev_tape: &'a [i32],
    pub conv_next_bank: &'a [i32],
    /// Optional DSpark target tap points to capture (`state.partial`).
    pub tap_points: &'a [TapPoint],
    /// Active streamed `ExpertBank` step context (when non-whole streamed).
    pub bank_run: Option<BlockAttnResBankRun<'a>>,
}

/// Output of `BoundBlockAttnRes::step_hidden`.
#[cfg(test)]
pub(crate) struct BlockAttnResStepOutput {
    /// Final F32 `[M, hidden]` hidden state after `ExitNorm::BlockAttention`,
    /// ready for `BoundTargetReadoutGraphs`.
    pub exit_hidden: Tensor,
    /// Captured F32 `[M, hidden]` residuals (`state.partial`) in `tap_points` order.
    pub taps: Vec<Tensor>,
}

/// Clone a `Charges` handle preserving each store's ledger entry across every
/// contract (including `Words`) and single-store confinement, so embedded
/// devices never fall back to `Charges::tally` or cross-store entries.
pub(crate) fn clone_charges(charges: &Charges) -> Charges {
    let device = charges.device();
    let by_contract: Vec<(seismic::Contract, Arc<dyn seismic::memory::Entry>)> =
        seismic::Contract::ALL
            .into_iter()
            .filter_map(|c| charges.entry(c).map(|e| (c, e)))
            .collect();
    if by_contract.is_empty() {
        return Charges::none(&device);
    }
    if by_contract.len() < seismic::Contract::ALL.len() {
        let (contract, entry) = by_contract[0].clone();
        debug_assert!(
            by_contract.iter().all(|(_, e)| e.store() == entry.store()),
            "partial Charges spans multiple stores"
        );
        return device.charges_in(contract, entry);
    }
    let entries: Vec<Arc<dyn seismic::memory::Entry>> =
        by_contract.into_iter().map(|(_, e)| e).collect();
    device.charges(move |id| {
        entries
            .iter()
            .find(|e| e.store() == id)
            .cloned()
            .unwrap_or_else(|| {
                panic!("invariant violated: Charges missing entry for store {id:?}")
            })
    })
}

/// Shared 1-node `NativeGraphFamily` builder for `BlockAttnRes` target and
/// RNN DSpark draft programs so they satisfy `NativeGraphFamily`'s non-empty
/// family invariant without sealing per-block graphs.
pub(crate) fn minimal_stream_join_family(
    device: &Device,
    hidden: u64,
) -> Result<seismic::NativeGraphFamily, String> {
    let join = prepare_kernel(
        device,
        stream_join::native_entry(),
        &[("H", hidden), ("SN", 1), ("IN", 0)],
    )?;
    let mut graph = device.native_graph();
    let dims = [("M", 1u64), ("H", hidden), ("SN", 1u64), ("IN", 0u64)];
    let residual = graph
        .input_for(&join, "residual", &dims)
        .map_err(|e| e.to_string())?;
    let y = graph
        .input_for(&join, "y", &dims)
        .map_err(|e| e.to_string())?;
    let inject = graph
        .input_for(&join, "inject", &dims)
        .map_err(|e| e.to_string())?;
    let _ = graph
        .enqueue::<stream_join::Entry>(
            &join,
            stream_join::WorkflowArgs {
                residual: residual.tensor().into(),
                y: y.tensor().into(),
                inject: inject.tensor().into(),
            },
        )
        .map_err(|e| e.to_string())?;
    let plan = graph.seal().map_err(|e| e.to_string())?;
    seismic::NativeGraphFamily::with_launches(
        std::slice::from_ref(&plan),
        &[vec![plan.clone()]],
        1,
        &[],
    )
    .map_err(|e| e.to_string())
}

pub(crate) fn prepare_kernel<E: seismic::Entry>(
    device: &Device,
    entry: seismic::BoundEntry<E>,
    statics: &[(&str, u64)],
) -> Result<NativeKernel<E>, String> {
    let implementation = entry
        .offered(device)
        .map_err(|e| format!("offered {}: {e}", E::NAME))?
        .ok_or_else(|| format!("no implementation for {} on device", E::NAME))?;
    let mut spec = NativeSpecialization::new();
    for name in &implementation.statics {
        let value = statics
            .iter()
            .find(|(candidate, _)| *candidate == name.as_str())
            .map(|(_, v)| *v)
            .ok_or_else(|| format!("prepare {}: missing required static `{name}`", E::NAME))?;
        spec = spec.with_static(name.clone(), value);
    }
    let full_spec = implementation
        .default_specialization(&spec)
        .map_err(|e| format!("default_specialization {}: {e}", E::NAME))?;
    // For Metal `project_rows`, `default_specialization` sets `MATRIX = 1/2`, which
    // dispatches `launch batch` (8-row simdgroup_matrix padding) even on `M = 1`
    // decode. Force `MATRIX = 0` so `M <= VERIFY_ROWS` uses true coalesced GEMV
    // while `M > 16` still uses GEMM/tall, and size `SIMDGROUPS` to saturate all
    // GPU cores for small `N`.
    if E::NAME == "project_rows"
        && device.backend() == seismic::BackendName::Metal
        && full_spec.params().contains_key("MATRIX")
    {
        let mut gemv_spec = full_spec.clone().with_param("MATRIX", 0);
        let n = statics
            .iter()
            .find(|(k, _)| *k == "N")
            .map(|(_, v)| *v)
            .ok_or_else(|| format!("prepare {}: missing required static `N`", E::NAME))?;
        let sg = if n <= 512 { 2 } else if n <= 2048 { 4 } else { 8 };
        for (&(launch_idx, ref pname), _) in full_spec.launch_params() {
            if pname == "SIMDGROUPS" {
                gemv_spec = gemv_spec.with_launch_param(launch_idx, "SIMDGROUPS", sg);
            }
        }
        if let Ok(k) = entry.prepare(device, &gemv_spec) {
            return Ok(k);
        }
    }
    entry
        .prepare(device, &full_spec)
        .map_err(|error| format!("prepare {}: {error}", E::NAME))
}

#[cfg(test)]
fn f32_tensor(charges: &Charges, shape: &[u64], values: &[f32]) -> Result<Tensor, String> {
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    Tensor::from_host(charges, Element::f32(), shape, &bytes).map_err(|e| e.to_string())
}

#[cfg(test)]
fn i32_tensor(charges: &Charges, shape: &[u64], values: &[i32]) -> Result<Tensor, String> {
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    Tensor::from_host(charges, Element::i32(), shape, &bytes).map_err(|e| e.to_string())
}

#[cfg(test)]
fn read_f32(tensor: &Tensor) -> Result<Vec<f32>, String> {
    let bytes = tensor.read_to_host().map_err(|e| e.to_string())?;
    if tensor.element() == Element::f32() {
        Ok(bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect())
    } else if tensor.element() == Element::bf16() {
        Ok(bytes
            .chunks_exact(2)
            .map(|b| f32::from_bits(u32::from(u16::from_le_bytes([b[0], b[1]])) << 16))
            .collect())
    } else if tensor.element() == Element::f16() {
        Ok(bytes
            .chunks_exact(2)
            .map(|b| seismic::f16_to_f32(u16::from_le_bytes([b[0], b[1]])))
            .collect())
    } else {
        Err(format!("unsupported element {:?} in read_f32", tensor.element()))
    }
}

/// Weight metadata used at preparation time (either from `ModelLoadPlan` or
/// directly from uploaded test tensors).
#[derive(Clone, Copy)]
struct WeightMeta {
    element: Element,
    n: u64,
    k: u64,
    scale_extent: u64,
    bound_extent: u64,
}

#[inline(never)]
#[allow(clippy::too_many_arguments)]
fn prepare_non_streamed_routed_block(
    device: &Device,
    load: &ModelLoadPlan,
    ffn_scope: WeightScope,
    hidden: u64,
    ffn_eps: f32,
    is_whole: bool,
    shape: &GeneralRoutedShape,
    gr_entries: GeneralRoutedGraphEntries<'_, seismic::NativeGraph>,
    row_classes: &[u64],
) -> Result<PreparedStreamedRoutedBlock, GraphError> {
    let (topology, numbers) = {
        let bank: LayerBank<'_, '_, seismic::NativeGraph> =
            if is_whole { LayerBank::Home } else { LayerBank::Held };
        let (top, nums) = general_split(load, ffn_scope, shape, &bank)?;
        (Numbers::map(top, &mut Argument::from), constant(nums))
    };
    let mut decode_map = BTreeMap::new();
    for &m in row_classes {
        let rows_dim = Dim::from(m);
        let mut graph = device.native_graph();
        let residual = GraphDraft::port(&mut graph, Element::f32(), &[m, hidden])?;
        let out_port = GraphDraft::port(&mut graph, Element::f32(), &[m, hidden])?;
        let mut weights = Vec::new();
        let mut constants = Vec::new();
        let bank: LayerBank<'_, '_, seismic::NativeGraph> =
            if is_whole { LayerBank::Home } else { LayerBank::Held };
        let (output, step_inputs) = general_routed(
            &mut graph,
            gr_entries,
            bank,
            &topology,
            &numbers,
            load,
            ffn_scope,
            &mut weights,
            &mut constants,
            Some(residual.tensor()),
            RoutedSum::Branch,
            &rows_dim,
            Argument::from(ffn_eps),
            None,
        )?;
        // Non-streamed (`Home`/`Held`) routed layers never call `graph.share`
        // on `output`, so `graph.store` writes directly into `out_port` with
        // zero copy launches.
        graph
            .store(&output, &out_port)
            .map_err(|e| GraphError::from(e.to_string()))?;
        let plan = graph.seal().map_err(|e| GraphError::from(e.to_string()))?;
        decode_map.insert(
            m,
            PreparedStreamedStage {
                plan,
                residual: Some(residual),
                output: Some(out_port),
                weights,
                banks: Vec::new(),
                parks: Vec::new(),
                constants,
                step_inputs,
                split: None,
                pass_ports: None,
            },
        );
    }
    Ok(PreparedStreamedRoutedBlock {
        decode: decode_map,
        resident: BTreeMap::new(),
        grouped: BTreeMap::new(),
        fill: None,
    })
}

#[inline(never)]
#[allow(clippy::too_many_arguments)]
fn prepare_streamed_fill_stage(
    device: &Device,
    load: &ModelLoadPlan,
    ffn_scope: WeightScope,
    bank_entries: BankGraphEntries<'_, seismic::NativeGraph>,
    bank_form: &crate::operators::routed::streamed::BankForm,
    bank_numbers: Option<&crate::operators::routed::streamed::BankNumbers>,
) -> Result<PreparedStreamedStage, GraphError> {
    let mut graph = device.native_graph();
    let mut weights = Vec::new();
    let mut banks = Vec::new();
    let mut parks = Vec::new();
    let mut constants = Vec::new();
    let bank = LayerBank::Streamed(BankGraph {
        entries: bank_entries,
        ports: &mut banks,
        parks: &mut parks,
        unified: None,
    });
    let _ = scratch_fill(
        &mut graph,
        bank,
        bank_form,
        bank_numbers,
        load,
        ffn_scope,
        &mut weights,
        &mut constants,
    )?;
    let plan = graph.seal().map_err(|e| GraphError::from(e.to_string()))?;
    Ok(PreparedStreamedStage {
        plan,
        residual: None,
        output: None,
        weights,
        banks,
        parks,
        constants,
        step_inputs: None,
        split: None,
        pass_ports: None,
    })
}

#[inline(never)]
#[allow(clippy::too_many_arguments)]
fn prepare_streamed_decode_stage<'a>(
    device: &Device,
    load: &ModelLoadPlan,
    ffn_scope: WeightScope,
    m: u64,
    hidden: u64,
    ffn_eps: f32,
    gr_entries: GeneralRoutedGraphEntries<'a, seismic::NativeGraph>,
    bank_entries: BankGraphEntries<'a, seismic::NativeGraph>,
    topology: &crate::operators::routed::graph::GeneralRoutedTopology,
    numbers: &crate::operators::routed::graph::GeneralRoutedLayer,
    f32_copy: &NativeKernel<magnitude_kernels::feature_copy::Entry>,
) -> Result<PreparedStreamedStage, GraphError> {
    let rows_dim = Dim::from(m);
    let mut graph = device.native_graph();
    let residual = GraphDraft::port(&mut graph, Element::f32(), &[m, hidden])?;
    let mut out_port = GraphDraft::port(&mut graph, Element::f32(), &[m, hidden])?;
    let mut weights = Vec::new();
    let mut banks = Vec::new();
    let mut parks = Vec::new();
    let mut constants = Vec::new();
    let bank = LayerBank::Streamed(BankGraph {
        entries: bank_entries,
        ports: &mut banks,
        parks: &mut parks,
        unified: None,
    });
    let (output, step_inputs) = general_routed(
        &mut graph,
        gr_entries,
        bank,
        topology,
        numbers,
        load,
        ffn_scope,
        &mut weights,
        &mut constants,
        Some(residual.tensor()),
        RoutedSum::Branch,
        &rows_dim,
        Argument::from(ffn_eps),
        None,
    )?;
    graph
        .enqueue(
            f32_copy,
            magnitude_kernels::feature_copy::WorkflowArgs {
                input: (&output).into(),
                out: out_port.tensor_mut().into(),
            },
        )
        .map_err(|e| GraphError::from(e.to_string()))?;
    let plan = graph.seal().map_err(|e| GraphError::from(e.to_string()))?;
    Ok(PreparedStreamedStage {
        plan,
        residual: Some(residual),
        output: Some(out_port),
        weights,
        banks,
        parks,
        constants,
        step_inputs,
        split: None,
        pass_ports: None,
    })
}

fn prepare_streamed_routed_blocks(
    device: &Device,
    load: &ModelLoadPlan,
    decoder: &Decoder,
    handles: &AttestedTarget,
    row_classes: &[u64],
) -> Result<BTreeMap<usize, PreparedStreamedRoutedBlock>, GraphError> {
    let bank_plan = load.expert_bank();
    let hidden = decoder.hidden;
    let f32_copy = prepare_kernel(
        device,
        magnitude_kernels::feature_copy::native_entry_with(
            magnitude_kernels::feature_copy::Elements { A: Element::f32() },
        ),
        &[],
    )
    .map_err(GraphError::from)?;
    let mut blocks = BTreeMap::new();
    for (b_idx, block) in decoder.blocks.iter().enumerate() {
        let Some(ffn_sub) = block.sublayers.get(1) else {
            continue;
        };
        let Operator::RoutedFfn(routed) = &ffn_sub.op else {
            continue;
        };
        let ffn_scope = WeightScope::TargetSublayer(SublayerIndex {
            block: b_idx as u32,
            sublayer: 1,
        });
        let has_bank = bank_plan.is_some_and(|b| b.layer(ffn_scope).is_some());
        let has_homes = bank_plan.is_some_and(|b| b.whole() && b.layer(ffn_scope).is_some());
        let gr_kernels = match handles
            .blocks
            .get(b_idx)
            .and_then(|b| b.feed_forward.as_ref())
        {
            Some(AttestedFeedForward::GeneralRouted(k)) => k,
            _ => {
                return Err(format!(
                    "missing GeneralRoutedKernels for routed block {b_idx}"
                )
                .into());
            }
        };
        let ffn_eps = match &ffn_sub.input {
            InputNorm::BlockAttention(ba) => ba.norm.epsilon() as f32,
            InputNorm::Rms(r) => r.epsilon as f32,
            InputNorm::RmsUnweighted(r) => r.epsilon as f32,
            InputNorm::HyperConnectionMix(_) | InputNorm::None => 1e-6,
        };
        let shape = GeneralRoutedShape::of(hidden, routed)
            .map_err(|e| GraphError::from(e.to_string()))?;
        let gr_entries = GeneralRoutedGraphEntries::from(gr_kernels);

        if !has_bank {
            // Resident (`Held`) routed FFN with no expert bank.
            let built = prepare_non_streamed_routed_block(
                device,
                load,
                ffn_scope,
                hidden,
                ffn_eps,
                false,
                &shape,
                gr_entries,
                row_classes,
            )?;
            blocks.insert(b_idx, built);
            continue;
        }

        // Elastic expert banks can start whole (`has_homes`) and shrink below
        // `whole()` at runtime when KV cache grows. Always seal the streamed
        // graphs so runtime shrink works seamlessly, and also seal `Home`
        // fast-path graphs when the bank has home views (`has_homes`).
        let resident_map = if has_homes {
            prepare_non_streamed_routed_block(
                device,
                load,
                ffn_scope,
                hidden,
                ffn_eps,
                true,
                &shape,
                gr_entries,
                row_classes,
            )?
            .decode
        } else {
            BTreeMap::new()
        };

        let bank_kernels = handles
            .blocks
            .get(b_idx)
            .and_then(|b| b.bank.as_ref())
            .ok_or_else(|| {
                GraphError::from(format!(
                    "missing ExpertBankKernels for streamed block {b_idx}"
                ))
            })?;
        let (split_form, split_layer) = split_shape(routed, decoder);
        let split_layer_dim = constant(split_layer);
        let bank_entries = BankGraphEntries::for_backend(bank_kernels, device.backend());

        // Compute topology & numbers once per block.
        let (topology, numbers) = {
            let mut dummy_ports = Vec::new();
            let mut dummy_parks = Vec::new();
            let bank = LayerBank::Streamed(BankGraph {
                entries: bank_entries,
                ports: &mut dummy_ports,
                parks: &mut dummy_parks,
                unified: None,
            });
            let (top, nums) = general_split(load, ffn_scope, &shape, &bank)?;
            (Numbers::map(top, &mut Argument::from), constant(nums))
        };

        // 1. Scratch fill graph (independent of row count).
        let fill = prepare_streamed_fill_stage(
            device,
            load,
            ffn_scope,
            bank_entries,
            &topology.bank,
            numbers.bank.as_ref(),
        )?;

        // 2. Decode (m <= VERIFY_ROWS) and Grouped (m > VERIFY_ROWS) graphs per row class.
        let mut decode_map = BTreeMap::new();
        let mut grouped_map = BTreeMap::new();
        for &m in row_classes {
            let rows_dim = Dim::from(m);
            if decodes(m) {
                let stage = prepare_streamed_decode_stage(
                    device,
                    load,
                    ffn_scope,
                    m,
                    hidden,
                    ffn_eps,
                    gr_entries,
                    bank_entries,
                    &topology,
                    &numbers,
                    &f32_copy,
                )?;
                decode_map.insert(m, stage);
            } else {
                // Front
                let front = {
                    let mut graph = device.native_graph();
                    let residual = GraphDraft::port(&mut graph, Element::f32(), &[m, hidden])?;
                    let mut split_build = SplitPorts::of(
                        &mut graph,
                        RunStage::Front,
                        Some((&split_form, &split_layer_dim)),
                        &rows_dim,
                        &rows_dim,
                    )?
                    .ok_or("expected Front SplitBuild")?;
                    let mut weights = Vec::new();
                    let mut banks = Vec::new();
                    let mut parks = Vec::new();
                    let mut constants = Vec::new();
                    let bank = LayerBank::Streamed(BankGraph {
                        entries: bank_entries,
                        ports: &mut banks,
                        parks: &mut parks,
                        unified: None,
                    });
                    let (_, step_inputs) = general_routed(
                        &mut graph,
                        gr_entries,
                        bank,
                        &topology,
                        &numbers,
                        load,
                        ffn_scope,
                        &mut weights,
                        &mut constants,
                        Some(residual.tensor()),
                        RoutedSum::Branch,
                        &rows_dim,
                        Argument::from(ffn_eps),
                        Some(split_build.split()),
                    )?;
                    let plan = graph.seal().map_err(|e| GraphError::from(e.to_string()))?;
                    PreparedStreamedStage {
                        plan,
                        residual: Some(residual),
                        output: None,
                        weights,
                        banks,
                        parks,
                        constants,
                        step_inputs,
                        split: Some(split_build.into_ports()),
                        pass_ports: None,
                    }
                };
                // Pass
                let pass = {
                    let mut graph = device.native_graph();
                    let (mut pass_ports, blocks_dim) = PassPorts::of(
                        &mut graph,
                        &split_form,
                        &split_layer_dim,
                        &rows_dim,
                        &rows_dim,
                    )?;
                    let mut weights = Vec::new();
                    let mut banks = Vec::new();
                    let mut parks = Vec::new();
                    let mut constants = Vec::new();
                    let bank = LayerBank::Streamed(BankGraph {
                        entries: bank_entries,
                        ports: &mut banks,
                        parks: &mut parks,
                        unified: None,
                    });
                    let _ = general_pass(
                        &mut graph,
                        gr_entries,
                        bank,
                        &topology,
                        &numbers,
                        load,
                        ffn_scope,
                        &mut weights,
                        &mut constants,
                        &mut pass_ports,
                        &blocks_dim,
                        &rows_dim,
                    )?;
                    let plan = graph.seal().map_err(|e| GraphError::from(e.to_string()))?;
                    PreparedStreamedStage {
                        plan,
                        residual: None,
                        output: None,
                        weights,
                        banks,
                        parks,
                        constants,
                        step_inputs: None,
                        split: None,
                        pass_ports: Some(pass_ports),
                    }
                };
                // Back (Combine onto zeros -> ffn_raw)
                let back = {
                    let mut graph = device.native_graph();
                    let residual_port = if routed.shared.is_some() {
                        Some(GraphDraft::port(&mut graph, Element::f32(), &[m, hidden])?)
                    } else {
                        None
                    };
                    let out_port = GraphDraft::port(&mut graph, Element::f32(), &[m, hidden])?;
                    let mut split_build = SplitPorts::of(
                        &mut graph,
                        RunStage::Combine,
                        Some((&split_form, &split_layer_dim)),
                        &rows_dim,
                        &rows_dim,
                    )?
                    .ok_or("expected Combine SplitBuild")?;
                    let residual_tensor = match &residual_port {
                        Some(port) => port.tensor().clone(),
                        None => match &split_build {
                            SplitBuild::Back(bp, _) => bp.crossing.weights.tensor().clone(),
                            _ => unreachable!(),
                        },
                    };
                    let mut weights = Vec::new();
                    let mut banks = Vec::new();
                    let mut parks = Vec::new();
                    let mut constants = Vec::new();
                    let bank = LayerBank::Streamed(BankGraph {
                        entries: bank_entries,
                        ports: &mut banks,
                        parks: &mut parks,
                        unified: None,
                    });
                    let (output, step_inputs) = general_routed(
                        &mut graph,
                        gr_entries,
                        bank,
                        &topology,
                        &numbers,
                        load,
                        ffn_scope,
                        &mut weights,
                        &mut constants,
                        Some(&residual_tensor),
                        RoutedSum::Branch,
                        &rows_dim,
                        Argument::from(ffn_eps),
                        Some(split_build.split()),
                    )?;
                    GraphDraft::store(&mut graph, &output, &out_port)?;
                    let plan = graph.seal().map_err(|e| GraphError::from(e.to_string()))?;
                    PreparedStreamedStage {
                        plan,
                        residual: residual_port,
                        output: Some(out_port),
                        weights,
                        banks,
                        parks,
                        constants,
                        step_inputs,
                        split: Some(split_build.into_ports()),
                        pass_ports: None,
                    }
                };
                grouped_map.insert(m, (front, pass, back));
            }
        }
        blocks.insert(
            b_idx,
            PreparedStreamedRoutedBlock {
                decode: decode_map,
                resident: resident_map,
                grouped: grouped_map,
                fill: Some(fill),
            },
        );
    }
    Ok(blocks)
}

impl PreparedBlockAttnRes {
    /// Prepare all kernels required for live `ResidualForm::BlockAttention`
    /// execution from `ModelLoadPlan` and `AttestedTarget`.
    pub(crate) fn prepare_from_load(
        device: &Device,
        load: &ModelLoadPlan,
        decoder: &Decoder,
        handles: &AttestedTarget,
        row_classes: &[u64],
        tap_points: Vec<TapPoint>,
    ) -> Result<Self, String> {
        let streamed_routed =
            prepare_streamed_routed_blocks(device, load, decoder, handles, row_classes)
                .map_err(|e| e.to_string())?;
        let meta = |role: WeightRole| -> Result<WeightMeta, String> {
            if let Some(wp) = load.weight_by_role(role) {
                let scale_extent = wp.scale_extent();
                let (n, k) = match wp.shape.as_slice() {
                    [rows, cols] => (*rows, *cols),
                    [_, rows, cols] => (*rows, *cols),
                    [len] => (*len, 1),
                    _ => (0, 0),
                };
                return Ok(WeightMeta {
                    element: wp.resident,
                    n,
                    k,
                    scale_extent,
                    bound_extent: wp.bound,
                });
            }
            // Streamed expert weights live in `ExpertBankPlan` rather than target_weights.
            if let Some(bank) = load.expert_bank() {
                if let Some((_, layer)) = bank.layer(role.scope) {
                    let kind = match role.kind {
                        WeightKind::ExpertGate => Some(crate::planning::RoutedTensor::Gate),
                        WeightKind::ExpertUp => Some(crate::planning::RoutedTensor::Up),
                        WeightKind::ExpertDown => Some(crate::planning::RoutedTensor::Down),
                        _ => None,
                    };
                    if let Some(rt) = kind {
                        if let Some(wp) = layer.resident.get(rt) {
                            let scale_extent = wp.scale_extent();
                            let (n, k) = match wp.shape.as_slice() {
                                [_, rows, cols] => (*rows, *cols),
                                [rows, cols] => (*rows, *cols),
                                _ => (0, 0),
                            };
                            return Ok(WeightMeta {
                                element: wp.resident,
                                n,
                                k,
                                scale_extent,
                                bound_extent: wp.bound,
                            });
                        }
                    }
                }
            }
            Err(format!("missing planned weight {role:?}"))
        };
        Self::prepare_with(device, decoder, &meta, streamed_routed, tap_points)
    }

    #[cfg(test)]
    pub(crate) fn prepare_from_weights(
        device: &Device,
        decoder: &Decoder,
        weights: &HashMap<WeightRole, Tensor>,
        tap_points: Vec<TapPoint>,
    ) -> Result<Self, String> {
        let meta = |role: WeightRole| -> Result<WeightMeta, String> {
            weights
                .get(&role)
                .map(|t| {
                    let (n, k) = match t.extents() {
                        [rows, cols] => (*rows, *cols),
                        [_, rows, cols] => (*rows, *cols),
                        [len] => (*len, 1),
                        _ => (0, 0),
                    };
                    WeightMeta {
                        element: t.element(),
                        n,
                        k,
                        scale_extent: 0,
                        bound_extent: 0,
                    }
                })
                .ok_or_else(|| format!("missing weight {role:?}"))
        };
        Self::prepare_with(device, decoder, &meta, BTreeMap::new(), tap_points)
    }

    fn prepare_with(
        device: &Device,
        decoder: &Decoder,
        weight_meta: &dyn Fn(WeightRole) -> Result<WeightMeta, String>,
        streamed_routed: BTreeMap<usize, PreparedStreamedRoutedBlock>,
        tap_points: Vec<TapPoint>,
    ) -> Result<Self, String> {
        let plan = BlockAttnResPlan::of(decoder).map_err(|e| e.to_string())?;
        let schedule = super::block_attn_res_graph::BarSchedule::of(decoder, &plan)?;
        let activation = Element::dense(crate::planning::activation_dtype(decoder.activation_dtype));
        let hidden = decoder.hidden;

        let emb_elem = weight_meta(WeightRole {
            scope: WeightScope::Target,
            kind: WeightKind::Embedding,
        })?
        .element;
        let out_norm_elem = weight_meta(WeightRole {
            scope: WeightScope::Target,
            kind: WeightKind::OutputNorm,
        })?
        .element;
        let out_meta = weight_meta(WeightRole {
            scope: WeightScope::Target,
            kind: WeightKind::Output,
        })?;

        let mix = prepare_kernel(device, block_attn_res_mix::native_entry(), &[])?;
        let rms_concat = prepare_kernel(
            device,
            rms_concat::native_entry_with(rms_concat::Elements { A: activation }),
            &[],
        )?;
        let message_gate_mix = prepare_kernel(device, message_gate_mix::native_entry(), &[])?;
        let message_gate_mix_norm = prepare_kernel(
            device,
            message_gate_mix_norm::native_entry_with(message_gate_mix_norm::Elements {
                NW: out_norm_elem,
                A: activation,
            }),
            &[],
        )?;
        let conv_aa = prepare_kernel(
            device,
            causal_short_conv::native_entry_with(causal_short_conv::Elements {
                A: activation,
                Y: activation,
            }),
            &[],
        )?;
        let conv_af = prepare_kernel(
            device,
            causal_short_conv::native_entry_with(causal_short_conv::Elements {
                A: activation,
                Y: Element::f32(),
            }),
            &[],
        )?;
        let conv_ff = prepare_kernel(
            device,
            causal_short_conv::native_entry_with(causal_short_conv::Elements {
                A: Element::f32(),
                Y: Element::f32(),
            }),
            &[],
        )?;
        let head_l2_clip = prepare_kernel(
            device,
            head_l2_clip::native_entry_with(head_l2_clip::Elements {
                A: activation,
                Y: activation,
            }),
            &[],
        )?;
        let attention_rel_bias = prepare_kernel(
            device,
            attention_rel_bias::native_entry_with(attention_rel_bias::Elements { A: activation }),
            &[],
        )?;
        let stream_join = prepare_kernel(
            device,
            stream_join::native_entry(),
            &[("H", hidden), ("SN", 1), ("IN", 0)],
        )?;
        let entry_norm = match &decoder.entry.norm {
            Some(EntryNorm::Rms { .. }) => {
                let nw = weight_meta(WeightRole {
                    scope: WeightScope::Target,
                    kind: WeightKind::EntryNorm,
                })?
                .element;
                Some(prepare_kernel(
                    device,
                    post_norm_residual::native_entry_with(post_norm_residual::Elements { NW: nw }),
                    &[],
                )?)
            }
            _ => None,
        };
        let embedding = prepare_kernel(
            device,
            embedding_rows::native_entry_with(embedding_rows::Elements {
                EW: emb_elem,
                A: activation,
            }),
            &[("D", hidden)],
        )?;
        let norm_rows = prepare_kernel(
            device,
            readout_features_rows::native_entry_with(readout_features_rows::Elements {
                NW: out_norm_elem,
                A: activation,
            }),
            &[("D", hidden), ("PT", 1), ("NM", hidden)],
        )?;
        let f32_copy = prepare_kernel(
            device,
            feature_copy::native_entry_with(feature_copy::Elements { A: Element::f32() }),
            &[],
        )?;
        let mut tap_rows_map = HashMap::new();
        for tap_elem in [activation, Element::f32()] {
            if !tap_rows_map.contains_key(&tap_elem) {
                if let Ok(k) = prepare_kernel(
                    device,
                    tap_rows::native_entry_with(tap_rows::Elements { A: tap_elem }),
                    &[],
                ) {
                    tap_rows_map.insert(tap_elem, k);
                }
            }
        }
        let mut rel_bias_projects = HashMap::new();

        let mut v_norms = HashMap::new();
        let mut projects = HashMap::new();

        let mut ensure_project = |a: Element, w: WeightMeta, y: Element| -> Result<(), String> {
            let key = (a, w.element, y, w.k, w.n, w.scale_extent, w.bound_extent);
            if !projects.contains_key(&key) {
                let k = prepare_kernel(
                    device,
                    project_rows::native_entry_with(project_rows::Elements {
                        A: a,
                        W: w.element,
                        Y: y,
                    }),
                    &[
                        ("G", 1),
                        ("K", w.k),
                        ("N", w.n),
                        ("WS", w.scale_extent),
                        ("WB", w.bound_extent),
                    ],
                )?;
                projects.insert(key, k);
            }
            Ok(())
        };
        ensure_project(activation, out_meta, Element::f32())?;

        for (b_idx, block) in decoder.blocks.iter().enumerate() {
            let b_u32 = b_idx as u32;
            for (s_idx, sublayer) in block.sublayers.iter().enumerate() {
                let scope = WeightScope::TargetSublayer(SublayerIndex {
                    block: b_u32,
                    sublayer: s_idx as u32,
                });
                // Cross-pass message controller projections
                if let InputNorm::BlockAttention(input) = &sublayer.input {
                    let streams: &[BlockAttentionStream] = match &input.cross_pass {
                        CrossPassInput::None => &[],
                        CrossPassInput::Single(_) => &[BlockAttentionStream::Shared],
                        CrossPassInput::SplitQkv { .. } => &[
                            BlockAttentionStream::Query,
                            BlockAttentionStream::Key,
                            BlockAttentionStream::Value,
                        ],
                    };
                    for &stream in streams {
                        let down_w = weight_meta(WeightRole {
                            scope,
                            kind: WeightKind::MessageController {
                                stream,
                                part: magnitude_family_contracts::MessageControllerPart::Down,
                            },
                        })?;
                        let gain_w = weight_meta(WeightRole {
                            scope,
                            kind: WeightKind::MessageController {
                                stream,
                                part: magnitude_family_contracts::MessageControllerPart::Gain,
                            },
                        })?;
                        let forget_w = weight_meta(WeightRole {
                            scope,
                            kind: WeightKind::MessageController {
                                stream,
                                part: magnitude_family_contracts::MessageControllerPart::Forget,
                            },
                        })?;
                        ensure_project(activation, down_w, activation)?;
                        ensure_project(activation, gain_w, Element::f32())?;
                        ensure_project(activation, forget_w, Element::f32())?;
                        if gain_w.element == forget_w.element
                            && gain_w.k == forget_w.k
                            && gain_w.scale_extent == 0
                            && forget_w.scale_extent == 0
                            && gain_w.bound_extent == 0
                            && forget_w.bound_extent == 0
                        {
                            let mut fused_w = gain_w;
                            fused_w.n = gain_w.n + forget_w.n;
                            let _ = ensure_project(activation, fused_w, Element::f32());
                        }
                    }
                }
                match &sublayer.op {
                    Operator::Attention(attn) => {
                        let v_key = (attn.kv_heads, attn.width);
                        if !v_norms.contains_key(&v_key) {
                            let vn = prepare_kernel(
                                device,
                                vision_norm::native_entry_with(vision_norm::Elements {
                                    NWE: Element::f32(),
                                    NBE: Element::f32(),
                                    Y: activation,
                                }),
                                &[
                                    ("G", attn.kv_heads),
                                    ("H", attn.width),
                                    ("NW", 0),
                                    ("NB", 0),
                                    ("NO", 0),
                                ],
                            )?;
                            v_norms.insert(v_key, vn);
                        }
                        for kind in [
                            WeightKind::Query,
                            WeightKind::Key,
                            WeightKind::Value,
                            WeightKind::AttentionGate,
                            WeightKind::AttentionOutput,
                            WeightKind::RelativeQuery,
                        ] {
                            if let Ok(w) = weight_meta(WeightRole { scope, kind }) {
                                ensure_project(activation, w, activation)?;
                            }
                        }
                        // Prepare horizontally fused attention input projections for
                        // M = 1 decode (`Q + Gate + RelQuery` and `+ Key + Value`).
                        if let (Ok(wq), Ok(wg), Ok(wr)) = (
                            weight_meta(WeightRole { scope, kind: WeightKind::Query }),
                            weight_meta(WeightRole { scope, kind: WeightKind::AttentionGate }),
                            weight_meta(WeightRole { scope, kind: WeightKind::RelativeQuery }),
                        ) {
                            let dense_compat = |a: WeightMeta, b: WeightMeta| {
                                a.element == b.element
                                    && a.k == b.k
                                    && a.scale_extent == 0
                                    && b.scale_extent == 0
                                    && a.bound_extent == 0
                                    && b.bound_extent == 0
                            };
                            if dense_compat(wq, wg) && dense_compat(wq, wr) {
                                let mut fused3 = wq;
                                fused3.n = wq.n + wg.n + wr.n;
                                let _ = ensure_project(activation, fused3, activation);
                                if let (Ok(wk), Ok(wv)) = (
                                    weight_meta(WeightRole { scope, kind: WeightKind::Key }),
                                    weight_meta(WeightRole { scope, kind: WeightKind::Value }),
                                ) {
                                    if dense_compat(wq, wk) && dense_compat(wq, wv) {
                                        let mut fused5 = wq;
                                        fused5.n = wq.n + wg.n + wr.n + wk.n + wv.n;
                                        let _ = ensure_project(activation, fused5, activation);
                                    }
                                }
                            }
                        }
                        for kind in [
                            WeightKind::AttentionSinks,
                            WeightKind::KeyShortConvDynamic,
                            WeightKind::ValueShortConvDynamic,
                        ] {
                            if let Ok(w) = weight_meta(WeightRole { scope, kind }) {
                                ensure_project(activation, w, Element::f32())?;
                            }
                        }
                        if let Ok(w) = weight_meta(WeightRole {
                            scope,
                            kind: WeightKind::RelativeBiasProjection,
                        }) {
                            let rb_key = (activation, w.element);
                            if !rel_bias_projects.contains_key(&rb_key) {
                                let k = prepare_kernel(
                                    device,
                                    rel_bias_project::native_entry_with(rel_bias_project::Elements {
                                        A: activation,
                                        W: w.element,
                                    }),
                                    &[],
                                )?;
                                rel_bias_projects.insert(rb_key, k);
                            }
                            let _ = ensure_project(activation, w, Element::f32());
                        }
                    }
                    _ => {}
                }
            }
        }

        let mut split_shapes = HashMap::new();
        for (&b_idx, block) in &streamed_routed {
            if block.fill.is_some() {
                if let Some(sub) = decoder.blocks.get(b_idx).and_then(|b| b.sublayers.get(1)) {
                    if let Operator::RoutedFfn(routed) = &sub.op {
                        let (sf, sl) = split_shape(routed, decoder);
                        split_shapes.entry(sl).or_insert(sf);
                    }
                }
            }
        }

        Ok(Self {
            hidden: decoder.hidden,
            activation,
            plan,
            schedule,
            tap_points,
            mix,
            rms_concat,
            message_gate_mix,
            message_gate_mix_norm,
            conv_aa,
            conv_af,
            conv_ff,
            head_l2_clip,
            attention_rel_bias,
            stream_join,
            entry_norm,
            embedding,
            norm_rows,
            f32_copy,
            tap_rows: tap_rows_map,
            rel_bias_projects,
            v_norms,
            projects,
            streamed_routed,
            split_shapes,
        })
    }

    /// Build a `NativeGraphFamily` covering all sealed entry + streamed-routed
    /// plans (or a minimal 1-node `stream_join` fallback when empty) so
    /// `TargetGraphWorkspaceLease` is sized to activate any entry or routed stage.
    pub(crate) fn minimal_family(
        &self,
        device: &Device,
        hidden: u64,
        entries: &BTreeMap<(u64, super::EntryTokens), super::entry::PreparedTargetEntryGraph>,
    ) -> Result<seismic::NativeGraphFamily, String> {
        if self.streamed_routed.is_empty() && entries.is_empty() {
            return minimal_stream_join_family(device, hidden);
        }
        let mut plans = Vec::new();
        let mut by_m_decode: BTreeMap<u64, Vec<NativeGraphPlan>> = BTreeMap::new();
        let mut by_m_resident_decode: BTreeMap<u64, Vec<NativeGraphPlan>> = BTreeMap::new();
        let mut by_m_prefill: BTreeMap<u64, Vec<NativeGraphPlan>> = BTreeMap::new();
        let mut by_m_resident_prefill: BTreeMap<u64, Vec<NativeGraphPlan>> = BTreeMap::new();
        let mut fill_plans: Vec<NativeGraphPlan> = Vec::new();
        for (&(m, _), entry) in entries {
            plans.push(entry.plan.clone());
            if crate::operators::routed::fused_graph::decodes(m) {
                by_m_decode.entry(m).or_default().push(entry.plan.clone());
                by_m_resident_decode.entry(m).or_default().push(entry.plan.clone());
            } else {
                by_m_prefill.entry(m).or_default().push(entry.plan.clone());
                by_m_resident_prefill.entry(m).or_default().push(entry.plan.clone());
            }
        }
        for block in self.streamed_routed.values() {
            if let Some(fill) = &block.fill {
                plans.push(fill.plan.clone());
                fill_plans.push(fill.plan.clone());
            }
            for (&m, stage) in &block.decode {
                plans.push(stage.plan.clone());
                if crate::operators::routed::fused_graph::decodes(m) {
                    by_m_decode.entry(m).or_default().push(stage.plan.clone());
                } else {
                    by_m_prefill.entry(m).or_default().push(stage.plan.clone());
                }
            }
            for (&m, stage) in &block.resident {
                plans.push(stage.plan.clone());
                if crate::operators::routed::fused_graph::decodes(m) {
                    by_m_resident_decode.entry(m).or_default().push(stage.plan.clone());
                } else {
                    by_m_resident_prefill.entry(m).or_default().push(stage.plan.clone());
                }
            }
            for (&m, (front, pass, back)) in &block.grouped {
                plans.push(front.plan.clone());
                plans.push(pass.plan.clone());
                plans.push(back.plan.clone());
                let entry = by_m_prefill.entry(m).or_default();
                entry.push(front.plan.clone());
                entry.push(pass.plan.clone());
                entry.push(back.plan.clone());
            }
        }
        let mut sequences: Vec<Vec<NativeGraphPlan>> = by_m_decode.into_values().collect();
        sequences.extend(by_m_resident_decode.into_values());
        let decodes_count = sequences.len();
        sequences.extend(by_m_prefill.into_values());
        sequences.extend(by_m_resident_prefill.into_values());
        if !fill_plans.is_empty() {
            sequences.push(fill_plans);
        }
        seismic::NativeGraphFamily::with_launches(&plans, &sequences, decodes_count, &[])
            .map_err(|e| e.to_string())
    }

    #[cfg(test)]
    fn project(
        &self,
        charges: &Charges,
        source: &Tensor,
        weight: &BoundWeight,
        output: Element,
    ) -> Result<Tensor, String> {
        let ws = weight.scale_extent();
        let wb = weight.bound_extent();
        let [n, k] = match weight.extents() {
            [n, k] => [*n, *k],
            other => return Err(format!("project_rows weight expects 2D shape, got {other:?}")),
        };
        let key = (source.element(), weight.element(), output, k, n, ws, wb);
        if let Some(kernel) = self.projects.get(&key) {
            let empty_scale = f32_tensor(charges, &[0], &[])?;
            let empty_bound = f32_tensor(charges, &[0], &[])?;
            let scale_t = weight.scale.as_ref().unwrap_or(&empty_scale);
            let bound_t = weight.bound.as_ref().unwrap_or(&empty_bound);
            return Ok(kernel
                .call(
                    charges,
                    project_rows::Args {
                        source,
                        weight: &weight.tensor,
                        weight_scale: scale_t,
                        weight_bound: bound_t,
                    },
                )
                .map_err(|e| format!("project_rows: {e}"))?
                .value);
        }
        // Fallback for small non-multiple-of-64 K (e.g. `attn_rel_bias` with d_rel = 16
        // on CUDA/Vulkan where `project_rows` requires K % 64 == 0).
        let m = source.extents()[0] as usize;
        let (k_us, n_us) = (k as usize, n as usize);
        let a_vals = read_f32(source)?;
        let w_vals = read_f32(&weight.tensor)?;
        let mut out = vec![0.0f32; m * n_us];
        for row in 0..m {
            let a_row = &a_vals[row * k_us..(row + 1) * k_us];
            let out_row = &mut out[row * n_us..(row + 1) * n_us];
            for col in 0..n_us {
                let w_row = &w_vals[col * k_us..(col + 1) * k_us];
                let mut acc = 0.0f32;
                for i in 0..k_us {
                    acc = a_row[i].mul_add(w_row[i], acc);
                }
                out_row[col] = acc;
            }
        }
        if output == Element::f32() {
            f32_tensor(charges, &[m as u64, n], &out)
        } else {
            Err(format!("missing project_rows kernel for {key:?}"))
        }
    }

    #[cfg(test)]
    fn rms_norm_rows(
        &self,
        charges: &Charges,
        x: &Tensor,
        norm: &Tensor,
        m: u64,
        h: u64,
        epsilon: f32,
    ) -> Result<Tensor, String> {
        let x3 = x.reshape(&[m, 1, h]).map_err(|e| e.to_string())?;
        let out_rows = i32_tensor(charges, &[m], &(0..m as i32).collect::<Vec<_>>())?;
        Ok(self
            .norm_rows
            .call(
                charges,
                readout_features_rows::Args {
                    hidden: &x3,
                    norm,
                    out_rows: &out_rows,
                    epsilon,
                },
            )
            .map_err(|e| format!("readout_features_rows: {e}"))?
            .value)
    }

}

fn bind_streamed_stage(
    stage: &PreparedStreamedStage,
    b_idx: usize,
    resident: &ResidentTarget,
    charges: &Charges,
    constants: &mut ConstantTensors,
) -> Result<BoundNativeGraphPlan, String> {
    let constant_tensors = stage
        .constants
        .iter()
        .map(|c| Ok((c.port(), constants.tensor(c)?)))
        .collect::<Result<Vec<_>, String>>()?;
    let fixed = stage
        .weights
        .iter()
        .map(|(w, port)| Ok((port, resident.bound(w.role, w.part)?)))
        .chain(
            stage
                .banks
                .iter()
                .filter(|(b, _)| !b.part.placed() && !b.part.windowed())
                .map(|(b, port)| Ok((port, resident.bank_part(*b)?))),
        )
        .chain(constant_tensors.iter().map(|(port, t)| Ok((*port, t))))
        .collect::<Result<Vec<_>, String>>()?;
    stage
        .plan
        .bind_static(charges, &fixed)
        .map_err(|e| format!("BlockAttnRes streamed routed binding block {b_idx}: {e}"))
}

impl BoundBlockAttnRes {
    /// Bind `prepared` to the live `ResidentTarget` (zero-copy `Tensor::clone`
    /// handles + optional `ExpertBank` + statically bound streamed routed plans).
    pub(crate) fn bind(
        prepared: Arc<PreparedBlockAttnRes>,
        charges: &Charges,
        resident: &ResidentTarget,
        prefill_rows: u64,
        prefill_places: usize,
    ) -> Result<Self, String> {
        let mut weights = HashMap::new();
        weights.insert(
            WeightRole {
                scope: WeightScope::Target,
                kind: WeightKind::Embedding,
            },
            BoundWeight::from_resident(&resident.embedding),
        );
        for (&role, w) in resident.sublayers.iter() {
            weights.insert(role, BoundWeight::from_resident(w));
        }
        if let crate::ResidentOutput::Packed(out_w) = &resident.output {
            weights.insert(
                WeightRole {
                    scope: WeightScope::Target,
                    kind: WeightKind::Output,
                },
                BoundWeight::from_resident(out_w),
            );
        }
        let mut constants = ConstantTensors::new(charges);
        let mut streamed_routed = BTreeMap::new();
        for (&b_idx, block) in &prepared.streamed_routed {
            let fill = block
                .fill
                .as_ref()
                .map(|f| bind_streamed_stage(f, b_idx, resident, charges, &mut constants))
                .transpose()?;
            let mut decode = BTreeMap::new();
            for (&m, stage) in &block.decode {
                decode.insert(
                    m,
                    bind_streamed_stage(stage, b_idx, resident, charges, &mut constants)?,
                );
            }
            let mut resident_bound = BTreeMap::new();
            for (&m, stage) in &block.resident {
                resident_bound.insert(
                    m,
                    bind_streamed_stage(stage, b_idx, resident, charges, &mut constants)?,
                );
            }
            let mut grouped = BTreeMap::new();
            for (&m, (front, pass, back)) in &block.grouped {
                grouped.insert(
                    m,
                    (
                        bind_streamed_stage(front, b_idx, resident, charges, &mut constants)?,
                        bind_streamed_stage(pass, b_idx, resident, charges, &mut constants)?,
                        bind_streamed_stage(back, b_idx, resident, charges, &mut constants)?,
                    ),
                );
            }
            streamed_routed.insert(
                b_idx,
                BoundStreamedRoutedBlock {
                    decode,
                    resident: resident_bound,
                    grouped,
                    fill,
                },
            );
        }
        let mut prefill_bufs = Vec::with_capacity(prefill_places);
        for _ in 0..prefill_places {
            prefill_bufs.push(super::block_attn_res_graph::BarStepBuffers::allocate(
                charges,
                &prepared.schedule,
                prefill_rows,
                prepared.hidden,
            )?);
        }
        let mut prefill_floor_grouped = HashMap::new();
        let mut prefill_floor_merged = HashMap::new();
        if prefill_rows > 0 && !prepared.split_shapes.is_empty() {
            use crate::operators::routed::split::{Crossing, Merged};
            let rows_dim = Dim::from(prefill_rows);
            for (split_layer, split_form) in &prepared.split_shapes {
                let split_layer_dim = constant(split_layer.clone());
                let crossing_ext = Crossing::at(split_form, &split_layer_dim, &rows_dim);
                let crossing_bufs = crossing_ext.try_map(|(elem, dims)| {
                    let exts: Vec<u64> = dims.iter().map(Dim::fixed).collect();
                    Tensor::zeros(charges, *elem, &exts)
                        .map_err(|e| format!("bar floor crossing(m={prefill_rows}, exts={exts:?}): {e}"))
                })?;
                prefill_floor_grouped.insert(split_layer.clone(), crossing_bufs);

                let merged_ext = Merged::at(split_form, &split_layer_dim, &rows_dim)?;
                let exts_of = |dims: &[Dim]| -> Vec<u64> { dims.iter().map(Dim::fixed).collect() };
                let merged_experts = Tensor::zeros(
                    charges,
                    merged_ext.experts.0,
                    &exts_of(&merged_ext.experts.1),
                )
                .map_err(|e| format!("bar floor merged_experts(m={prefill_rows}): {e}"))?;
                let merged_inverse = Tensor::zeros(
                    charges,
                    merged_ext.inverse.0,
                    &exts_of(&merged_ext.inverse.1),
                )
                .map_err(|e| format!("bar floor merged_inverse(m={prefill_rows}): {e}"))?;
                let table_vals: Vec<i32> = (0..=prefill_rows as i32).collect();
                let table_bytes: Vec<u8> =
                    table_vals.iter().flat_map(|v| v.to_le_bytes()).collect();
                let merged_table = Tensor::from_host(
                    charges,
                    merged_ext.table.0,
                    &exts_of(&merged_ext.table.1),
                    &table_bytes,
                )
                .map_err(|e| format!("bar floor merged_table(m={prefill_rows}): {e}"))?;
                prefill_floor_merged
                    .insert(split_layer.clone(), (merged_experts, merged_inverse, merged_table));
            }
        }
        Ok(Self {
            prepared,
            weights,
            streamed_routed,
            _streamed_constants: constants.into_tensors(),
            prefill_rows,
            prefill_bufs,
            prefill_floor_grouped,
            prefill_floor_merged,
            families: std::sync::Mutex::new(HashMap::new()),
            fused_proj_cache: std::sync::Mutex::new(HashMap::new()),
            charges: clone_charges(charges),
        })
    }

    pub(super) fn constant_tensors(&self) -> impl Iterator<Item = &Tensor> {
        self.prefill_bufs
            .iter()
            .flat_map(|b| b.tensors())
            .chain(self.prefill_floor_grouped.values().flat_map(|c| c.iter()))
            .chain(
                self.prefill_floor_merged
                    .values()
                    .flat_map(|(experts, inverse, table)| [experts, inverse, table]),
            )
    }

    #[cfg(test)]
    pub(crate) fn from_weights(
        device: &Device,
        prepared: Arc<PreparedBlockAttnRes>,
        raw_weights: HashMap<WeightRole, Tensor>,
    ) -> Self {
        let weights = raw_weights
            .into_iter()
            .map(|(k, t)| (k, BoundWeight::plain(t)))
            .collect();
        Self {
            prepared,
            weights,
            streamed_routed: BTreeMap::new(),
            _streamed_constants: Vec::new(),
            prefill_rows: 0,
            prefill_bufs: Vec::new(),
            prefill_floor_grouped: HashMap::new(),
            prefill_floor_merged: HashMap::new(),
            families: std::sync::Mutex::new(HashMap::new()),
            fused_proj_cache: std::sync::Mutex::new(HashMap::new()),
            charges: Charges::tally(device),
        }
    }

    #[cfg(test)]
    pub(crate) fn step_hidden(
        &self,
        decoder: &Decoder,
        _graph_workspace: Option<&mut NativeGraphFamilySlot>,
        layers: &mut [BlockAttnResLayerState],
        input: &BlockAttnResStepInput<'_>,
    ) -> Result<BlockAttnResStepOutput, String> {
        use super::block_attn_res_graph::{BarChainStep, queue_sealed_chain};
        let device = self.charges.device();
        let fam = self
            .prepared
            .minimal_family(&device, decoder.hidden, &BTreeMap::new())?;
        let mut dummy_slot = fam.new_slot(&self.charges, &[]).map_err(|e| e.to_string())?;
        let emb_w = self
            .weights
            .get(&WeightRole {
                scope: WeightScope::Target,
                kind: WeightKind::Embedding,
            })
            .ok_or("missing Embedding weight")?;
        let m = input.tokens.len() as u64;
        let tok_pairs: Vec<i32> = input.tokens.iter().flat_map(|&t| [t, 0]).collect();
        let tok_t = i32_tensor(&self.charges, &[m, 2], &tok_pairs)?;
        let dir_t = i32_tensor(&self.charges, &[0, 1], &[])?;
        let emb_scale = match decoder.entry.scale {
            magnitude_family_contracts::EmbeddingScale::Unit => 1.0f32,
            magnitude_family_contracts::EmbeddingScale::SqrtHidden => (decoder.hidden as f32).sqrt(),
        };
        let emb_out = self
            .prepared
            .embedding
            .call(
                &self.charges,
                embedding_rows::Args {
                    table: &emb_w.tensor,
                    tokens: &tok_t,
                    directory: &dir_t,
                    scale: emb_scale,
                    normalize: 0,
                    epsilon: 1e-6,
                },
            )
            .map_err(|e| e.to_string())?;
        let entry_hidden = if let (Some(EntryNorm::Rms { norm, .. }), Some(norm_k)) =
            (&decoder.entry.norm, &self.prepared.entry_norm)
        {
            let nw = self
                .weights
                .get(&WeightRole {
                    scope: WeightScope::Target,
                    kind: WeightKind::EntryNorm,
                })
                .ok_or("missing EntryNorm weight")?;
            let zero = f32_tensor(
                &self.charges,
                &[m, decoder.hidden],
                &vec![0.0f32; (m * decoder.hidden) as usize],
            )?;
            let out_rows = i32_tensor(&self.charges, &[m], &(0..m as i32).collect::<Vec<_>>())?;
            norm_k
                .call(
                    &self.charges,
                    post_norm_residual::Args {
                        residual: &zero,
                        projected: &emb_out.r1,
                        norm: &nw.tensor,
                        out_rows: &out_rows,
                        epsilon: norm.epsilon as f32,
                        scale: 1.0f32,
                    },
                )
                .map_err(|e| e.to_string())?
                .value
        } else {
            emb_out.r0
        };
        let tap_count = self.prepared.tap_points.len() as u64;
        let tap_buf = if !input.tap_points.is_empty() && tap_count > 0 {
            Some(
                Tensor::zeros(
                    &self.charges,
                    self.prepared.activation,
                    &[m, tap_count * decoder.hidden],
                )
                .map_err(|e| e.to_string())?,
            )
        } else {
            None
        };
        let mut step = BarChainStep {
            routed_slot: &mut dummy_slot,
            layers: layers.to_vec(),
            entry_hidden,
            coordinates: input.coordinates,
            media_overlays: input
                .media_overlays
                .iter()
                .map(|&(s, e, d)| (s, e, d.to_vec()))
                .collect(),
            visible_per_layer: input.visible_per_layer.to_vec(),
            spans: input.spans,
            fresh: input.fresh.to_vec(),
            destinations: input.destinations.to_vec(),
            conv_segments: input.conv_segments.to_vec(),
            conv_stop: input.conv_stop,
            conv_prev_bank: input.conv_prev_bank,
            conv_prev_tape: input.conv_prev_tape,
            conv_next_bank: input.conv_next_bank,
            tap_buf: tap_buf.clone(),
            bank_run: None,
        };
        let mut seq = device.native_sequence(seismic::NativeQueue::Compute);
        let (exits, _retained) = queue_sealed_chain(
            self,
            decoder,
            std::slice::from_mut(&mut step),
            |action| match action {
                super::block_attn_res_graph::BarComputeAction::Queue { ready, register } => {
                    let outputs = ready.queue(&mut seq).map_err(|e| e.to_string())?;
                    register(&outputs)?;
                    Ok(())
                }
                _ => Ok(()),
            },
        )?;
        seq.submit()
            .map_err(|e| e.to_string())?
            .wait()
            .map_err(|e| e.to_string())?;
        let mut taps = Vec::new();
        if let Some(tb) = tap_buf {
            let flat = read_f32(&tb)?;
            let h = decoder.hidden as usize;
            let tc = tap_count as usize;
            let mu = m as usize;
            for tp in input.tap_points {
                if let Some(idx) = self.prepared.tap_points.iter().position(|p| p == tp) {
                    let mut one = vec![0.0f32; mu * h];
                    for r in 0..mu {
                        let src = &flat[(r * tc + idx) * h..(r * tc + idx + 1) * h];
                        one[r * h..(r + 1) * h].copy_from_slice(src);
                    }
                    taps.push(f32_tensor(&self.charges, &[m, decoder.hidden], &one)?);
                }
            }
        }
        Ok(BlockAttnResStepOutput {
            exit_hidden: exits.into_iter().next().unwrap(),
            taps,
        })
    }

    #[cfg(test)]
    fn compute_logits(
        &self,
        device: &Device,
        decoder: &Decoder,
        exit_hidden: &Tensor,
    ) -> Result<Tensor, String> {
        let charges = Charges::tally(device);
        let m = exit_hidden.extents()[0];
        let h = decoder.hidden;
        let ExitNorm::BlockAttention { norm: out_norm, .. } = &decoder.exit.norm else {
            return Err("expected ExitNorm::BlockAttention".into());
        };
        let out_norm_w = self
            .weights
            .get(&WeightRole {
                scope: WeightScope::Target,
                kind: WeightKind::OutputNorm,
            })
            .ok_or("missing OutputNorm")?;
        let out_w = self
            .weights
            .get(&WeightRole {
                scope: WeightScope::Target,
                kind: WeightKind::Output,
            })
            .ok_or("missing Output")?;
        let exit_normed = self.prepared.rms_norm_rows(
            &charges,
            exit_hidden,
            out_norm_w,
            m,
            h,
            out_norm.epsilon as f32,
        )?;
        let active_vocab = decoder.active_vocabulary();
        let active_out_w = if out_w.extents()[0] > active_vocab {
            let bytes = out_w.read_to_host().map_err(|e| e.to_string())?;
            let row_bytes = bytes.len() / out_w.extents()[0] as usize;
            BoundWeight {
                tensor: Tensor::from_host(
                    &charges,
                    out_w.element(),
                    &[active_vocab, h],
                    &bytes[..active_vocab as usize * row_bytes],
                )
                .map_err(|e| e.to_string())?,
                scale: out_w.scale.clone(),
                bound: out_w.bound.clone(),
            }
        } else {
            out_w.clone()
        };
        let mut logits = self
            .prepared
            .project(&charges, &exit_normed, &active_out_w, Element::f32())?;
        if let Some(cap) = decoder.exit.softcap {
            let cap = cap as f32;
            let vals: Vec<f32> = read_f32(&logits)?
                .into_iter()
                .map(|v| cap * (v / cap).tanh())
                .collect();
            logits = f32_tensor(&charges, &[m, active_vocab], &vals)?;
        }
        Ok(logits)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use magnitude_family_contracts::*;
    use magnitude_state::{KvCodec, ModelStateLayout};
    use seismic::BackendName;

    fn pattern(count: usize, seed: u32, scale: f32) -> Vec<f32> {
        let mut s = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
        (0..count)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 17;
                s ^= s << 5;
                ((s >> 8) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0) * scale
            })
            .collect()
    }

    fn bf16_round(v: f32) -> f32 {
        let bits = v.to_bits();
        let lsb = (bits >> 16) & 1;
        let rounded = bits.wrapping_add(0x7fff + lsb) & 0xffff_0000;
        f32::from_bits(rounded)
    }

    fn bf16_tensor(charges: &Charges, shape: &[u64], values: &[f32]) -> Tensor {
        let bytes: Vec<u8> = values
            .iter()
            .flat_map(|&v| ((bf16_round(v).to_bits() >> 16) as u16).to_le_bytes())
            .collect();
        Tensor::from_host(charges, Element::bf16(), shape, &bytes).unwrap()
    }

    fn make_synthetic_inkling2(looped: bool) -> (Decoder, HashMap<WeightRole, Vec<f32>>) {
        let hidden = 32u64;
        let heads = 4u64;
        let kv_heads = 2u64;
        let head_dim = 8u64;
        let d_rel = 4u64;
        let swa_window = 8u64;
        let rel_extent = 8u64;
        let conv_k = 4u64;
        let experts = 4u64;
        let selected = 2u64;
        let inter = 16u64;
        let vocab_padded = 24u64;
        let vocab_unpadded = 20u64;
        let eps = 1e-6f64;
        let qk_clip = 22.627417f64;
        let rel_clip = 8.0f64;
        let gate_rank = 8u64;
        let block_size = 1u64;

        let mut host_weights: HashMap<WeightRole, Vec<f32>> = HashMap::new();
        let mut seed = 100u32;
        let mut add_w = |role: WeightRole, shape: &[u64], scale: f32, is_f32: bool| {
            seed += 1;
            let s = seed;
            let n: u64 = shape.iter().product();
            let mut vals = pattern(n as usize, s, scale);
            if !is_f32 {
                for v in &mut vals {
                    *v = bf16_round(*v);
                }
            }
            host_weights.insert(role, vals);
            WeightDescriptor::stored(format!("w_{s}"), shape.to_vec())
        };

        let emb_desc = add_w(
            WeightRole {
                scope: WeightScope::Target,
                kind: WeightKind::Embedding,
            },
            &[vocab_padded, hidden],
            0.5,
            false,
        );
        let entry_norm_desc = add_w(
            WeightRole {
                scope: WeightScope::Target,
                kind: WeightKind::EntryNorm,
            },
            &[hidden],
            1.0,
            false,
        );
        let out_q_desc = add_w(
            WeightRole {
                scope: WeightScope::Target,
                kind: WeightKind::BlockAttentionQuery(BlockAttentionStream::Shared),
            },
            &[hidden],
            0.3,
            true,
        );
        let out_norm_desc = add_w(
            WeightRole {
                scope: WeightScope::Target,
                kind: WeightKind::OutputNorm,
            },
            &[hidden],
            1.0,
            false,
        );
        let out_desc = add_w(
            WeightRole {
                scope: WeightScope::Target,
                kind: WeightKind::Output,
            },
            &[vocab_padded, hidden],
            0.25,
            false,
        );

        // Non-looped: 2 blocks (0 = local SWA, 1 = global).
        // Looped: 4 visits (0 = prelude local, 1 = core pass 0 global [SnapshotPrelude],
        //                   2 = core pass 1 global [ResetToPreludeAndBank + CrossPass],
        //                   3 = coda local).
        let n_blocks = if looped { 4 } else { 2 };
        let mut blocks = Vec::new();
        for b in 0..n_blocks {
            let is_global = if looped { b == 1 || b == 2 } else { b == 1 };
            let has_cross_pass = looped && b == 2;
            let transition = if looped && b == 1 {
                Some(BlockTransition::SnapshotPrelude)
            } else if looped && b == 2 {
                Some(BlockTransition::ResetToPreludeAndBank)
            } else {
                None
            };
            let s0 = WeightScope::TargetSublayer(SublayerIndex {
                block: b as u32,
                sublayer: 0,
            });
            let s1 = WeightScope::TargetSublayer(SublayerIndex {
                block: b as u32,
                sublayer: 1,
            });

            let attn_queries = if b == 0 {
                BlockAttentionQueries::Partial
            } else if !is_global {
                BlockAttentionQueries::Single(add_w(
                    WeightRole {
                        scope: s0,
                        kind: WeightKind::BlockAttentionQuery(BlockAttentionStream::Shared),
                    },
                    &[hidden],
                    0.3,
                    true,
                ))
            } else {
                BlockAttentionQueries::SplitQkv {
                    query: add_w(
                        WeightRole {
                            scope: s0,
                            kind: WeightKind::BlockAttentionQuery(BlockAttentionStream::Query),
                        },
                        &[hidden],
                        0.3,
                        true,
                    ),
                    key: add_w(
                        WeightRole {
                            scope: s0,
                            kind: WeightKind::BlockAttentionQuery(BlockAttentionStream::Key),
                        },
                        &[hidden],
                        0.3,
                        true,
                    ),
                    value: add_w(
                        WeightRole {
                            scope: s0,
                            kind: WeightKind::BlockAttentionQuery(BlockAttentionStream::Value),
                        },
                        &[hidden],
                        0.3,
                        true,
                    ),
                }
            };

            let make_cp = |add_w: &mut dyn FnMut(WeightRole, &[u64], f32, bool) -> WeightDescriptor,
                           scope: WeightScope,
                           stream: BlockAttentionStream| {
                CrossPassMessage {
                    query: add_w(
                        WeightRole {
                            scope,
                            kind: WeightKind::CrossPassQuery(stream),
                        },
                        &[hidden],
                        0.3,
                        true,
                    ),
                    controller: MessageController {
                        rank: gate_rank,
                        down: add_w(
                            WeightRole {
                                scope,
                                kind: WeightKind::MessageController {
                                    stream,
                                    part: magnitude_family_contracts::MessageControllerPart::Down,
                                },
                            },
                            &[gate_rank, 2 * hidden],
                            0.2,
                            false,
                        ),
                        gain: add_w(
                            WeightRole {
                                scope,
                                kind: WeightKind::MessageController {
                                    stream,
                                    part: magnitude_family_contracts::MessageControllerPart::Gain,
                                },
                            },
                            &[hidden, gate_rank],
                            0.2,
                            false,
                        ),
                        forget: add_w(
                            WeightRole {
                                scope,
                                kind: WeightKind::MessageController {
                                    stream,
                                    part: magnitude_family_contracts::MessageControllerPart::Forget,
                                },
                            },
                            &[hidden, gate_rank],
                            0.2,
                            false,
                        ),
                    },
                }
            };

            let attn_cross_pass = if has_cross_pass {
                CrossPassInput::SplitQkv {
                    query: Box::new(make_cp(&mut add_w, s0, BlockAttentionStream::Query)),
                    key: Box::new(make_cp(&mut add_w, s0, BlockAttentionStream::Key)),
                    value: Box::new(make_cp(&mut add_w, s0, BlockAttentionStream::Value)),
                }
            } else {
                CrossPassInput::None
            };

            let attn_norm = if !is_global {
                BlockAttentionNorm::Shared(RmsNorm {
                    weight: add_w(
                        WeightRole {
                            scope: s0,
                            kind: WeightKind::InputNorm,
                        },
                        &[hidden],
                        1.0,
                        false,
                    ),
                    epsilon: eps,
                })
            } else {
                BlockAttentionNorm::SplitQkv {
                    query: RmsNorm {
                        weight: add_w(
                            WeightRole {
                                scope: s0,
                                kind: WeightKind::BlockAttentionInputNorm(
                                    BlockAttentionStream::Query,
                                ),
                            },
                            &[hidden],
                            1.0,
                            false,
                        ),
                        epsilon: eps,
                    },
                    key: RmsNorm {
                        weight: add_w(
                            WeightRole {
                                scope: s0,
                                kind: WeightKind::BlockAttentionInputNorm(
                                    BlockAttentionStream::Key,
                                ),
                            },
                            &[hidden],
                            1.0,
                            false,
                        ),
                        epsilon: eps,
                    },
                    value: RmsNorm {
                        weight: add_w(
                            WeightRole {
                                scope: s0,
                                kind: WeightKind::BlockAttentionInputNorm(
                                    BlockAttentionStream::Value,
                                ),
                            },
                            &[hidden],
                            1.0,
                            false,
                        ),
                        epsilon: eps,
                    },
                }
            };

            let q_rows = heads * head_dim;
            let kv_rows = kv_heads * head_dim;
            let wq = add_w(
                WeightRole {
                    scope: s0,
                    kind: WeightKind::Query,
                },
                &[q_rows, hidden],
                0.25,
                false,
            );
            let wk = add_w(
                WeightRole {
                    scope: s0,
                    kind: WeightKind::Key,
                },
                &[kv_rows, hidden],
                0.25,
                false,
            );
            let wv = add_w(
                WeightRole {
                    scope: s0,
                    kind: WeightKind::Value,
                },
                &[kv_rows, hidden],
                0.25,
                false,
            );
            let wo = add_w(
                WeightRole {
                    scope: s0,
                    kind: WeightKind::AttentionOutput,
                },
                &[hidden, q_rows],
                0.25,
                false,
            );
            let wgate = add_w(
                WeightRole {
                    scope: s0,
                    kind: WeightKind::AttentionGate,
                },
                &[q_rows, hidden],
                0.25,
                false,
            );
            let q_clip_w = add_w(
                WeightRole {
                    scope: s0,
                    kind: WeightKind::QueryNorm,
                },
                &[heads, head_dim],
                1.0,
                true,
            );
            let wsink = add_w(
                WeightRole {
                    scope: s0,
                    kind: WeightKind::AttentionSinks,
                },
                &[heads, hidden],
                0.2,
                false,
            );
            let wr = add_w(
                WeightRole {
                    scope: s0,
                    kind: WeightKind::RelativeQuery,
                },
                &[heads * d_rel, hidden],
                0.25,
                false,
            );
            let wr_clip = add_w(
                WeightRole {
                    scope: s0,
                    kind: WeightKind::RelativeQueryNorm,
                },
                &[heads, d_rel],
                1.0,
                true,
            );
            let wr_bias = add_w(
                WeightRole {
                    scope: s0,
                    kind: WeightKind::RelativeBiasProjection,
                },
                &[rel_extent, d_rel],
                0.2,
                false,
            );

            let (k_dyn, v_dyn, kv_scale) = if is_global {
                (
                    Some(DynamicConvTap {
                        proj: add_w(
                            WeightRole {
                                scope: s0,
                                kind: WeightKind::KeyShortConvDynamic,
                            },
                            &[kv_heads * conv_k, hidden],
                            0.2,
                            false,
                        ),
                        heads: kv_heads,
                    }),
                    Some(DynamicConvTap {
                        proj: add_w(
                            WeightRole {
                                scope: s0,
                                kind: WeightKind::ValueShortConvDynamic,
                            },
                            &[kv_heads * conv_k, hidden],
                            0.2,
                            false,
                        ),
                        heads: kv_heads,
                    }),
                    1.0 / ((2 * conv_k + 1) as f64).sqrt(),
                )
            } else {
                (None, None, 1.0 / ((conv_k + 1) as f64).sqrt())
            };

            let k_conv = CausalShortConv {
                kernel_size: conv_k,
                weight: add_w(
                    WeightRole {
                        scope: s0,
                        kind: WeightKind::KeyShortConv,
                    },
                    &[kv_rows, conv_k],
                    0.3,
                    true,
                ),
                dynamic: k_dyn,
                residual_scale: kv_scale,
            };
            let v_conv = CausalShortConv {
                kernel_size: conv_k,
                weight: add_w(
                    WeightRole {
                        scope: s0,
                        kind: WeightKind::ValueShortConv,
                    },
                    &[kv_rows, conv_k],
                    0.3,
                    true,
                ),
                dynamic: v_dyn,
                residual_scale: kv_scale,
            };
            let attn_post_conv = CausalShortConv {
                kernel_size: conv_k,
                weight: add_w(
                    WeightRole {
                        scope: s0,
                        kind: WeightKind::PostShortConv,
                    },
                    &[hidden, conv_k],
                    0.3,
                    true,
                ),
                dynamic: None,
                residual_scale: 1.0 / ((conv_k + 1) as f64).sqrt(),
            };

            let attention = Attention {
                heads,
                kv_heads,
                width: head_dim,
                query: QueryProjection::Full(wq),
                query_keys: QueryKeys::Direct,
                gate: AttentionGate::Separate {
                    weight: wgate,
                    function: GateFunction::Sigmoid,
                    granularity: GateGranularity::Element,
                },
                query_norm: HeadNorm::L2Clip {
                    threshold: qk_clip,
                    epsilon: 1e-6,
                    scale: Some(q_clip_w),
                },
                history: vec![HistorySource::visible(HistoryRows::Own(
                    RowProducer::Projected(ProjectedRows {
                        key: wk,
                        value: ValueSource::Projected(wv),
                        key_conv: Some(k_conv),
                        value_conv: Some(v_conv),
                        key_norm: HeadNorm::L2Clip {
                            threshold: qk_clip,
                            epsilon: 1e-6,
                            scale: None,
                        },
                        value_norm: ValueNorm::RmsUnweighted(UnweightedRms { epsilon: eps }),
                        domain: if is_global {
                            TokenDomain::Token
                        } else {
                            TokenDomain::Window { tokens: swa_window }
                        },
                        rounding: RowRounding::None,
                    }),
                ))],
                rotary: Rotary::None,
                relative_bias: Some(RelativeBias {
                    d_rel,
                    extent: rel_extent,
                    query_proj: wr,
                    rel_clip: HeadNorm::L2Clip {
                        threshold: rel_clip,
                        epsilon: 1e-6,
                        scale: Some(wr_clip),
                    },
                    bias_proj: wr_bias,
                }),
                log_scale: if is_global {
                    Some(LogScale {
                        floor: 2,
                        alpha: 0.1,
                    })
                } else {
                    None
                },
                scale: 1.0 / (head_dim as f64).sqrt(),
                media_rows: MediaRowAttention::Causal,
                sinks: AttentionSinks::Dynamic(wsink),
                output: OutputProjection::Full(wo),
            };

            // FFN sublayer (MoE with SITU)
            let ffn_q = add_w(
                WeightRole {
                    scope: s1,
                    kind: WeightKind::BlockAttentionQuery(BlockAttentionStream::Shared),
                },
                &[hidden],
                0.3,
                true,
            );
            let ffn_cp = if has_cross_pass {
                CrossPassInput::Single(Box::new(make_cp(
                    &mut add_w,
                    s1,
                    BlockAttentionStream::Shared,
                )))
            } else {
                CrossPassInput::None
            };
            let ffn_norm = RmsNorm {
                weight: add_w(
                    WeightRole {
                        scope: s1,
                        kind: WeightKind::InputNorm,
                    },
                    &[hidden],
                    1.0,
                    false,
                ),
                epsilon: eps,
            };
            let router_w = add_w(
                WeightRole {
                    scope: s1,
                    kind: WeightKind::Router,
                },
                &[experts, hidden],
                0.25,
                false,
            );
            let router_b = add_w(
                WeightRole {
                    scope: s1,
                    kind: WeightKind::RouterSelectionBias,
                },
                &[experts],
                0.1,
                true,
            );
            let eg = add_w(
                WeightRole {
                    scope: s1,
                    kind: WeightKind::ExpertGate,
                },
                &[experts, inter, hidden],
                0.25,
                false,
            );
            let eu = add_w(
                WeightRole {
                    scope: s1,
                    kind: WeightKind::ExpertUp,
                },
                &[experts, inter, hidden],
                0.25,
                false,
            );
            let ed = add_w(
                WeightRole {
                    scope: s1,
                    kind: WeightKind::ExpertDown,
                },
                &[experts, hidden, inter],
                0.25,
                false,
            );
            let ffn_post_conv = CausalShortConv {
                kernel_size: conv_k,
                weight: add_w(
                    WeightRole {
                        scope: s1,
                        kind: WeightKind::PostShortConv,
                    },
                    &[hidden, conv_k],
                    0.3,
                    true,
                ),
                dynamic: None,
                residual_scale: 1.0 / ((conv_k + 1) as f64).sqrt(),
            };

            let routed = RoutedFfn {
                experts,
                selected,
                intermediate: inter,
                router: Router {
                    weight: router_w,
                    input: RouterInput::Operator,
                    score: ScoreFunction::Sigmoid,
                    selection: ExpertSelection::TopK {
                        bias: Some(router_b),
                    },
                    media_bias: None,
                    normalization: RouteNormalization::SumPlusEpsilon(1e-20),
                    scale: 8.0,
                },
                expert_up: FeedForwardUp::Gated {
                    activation: ActivationFunction::Silu,
                    limit: Some(GluLimit::softcap(4.0, 25.0).unwrap()),
                    gate: eg,
                    up: eu,
                },
                expert_down: ed,
                expert_scale: None,
                latent: None,
                shared: None,
            };

            blocks.push(Block {
                transition,
                sublayers: vec![
                    Sublayer {
                        input: InputNorm::BlockAttention(BlockAttentionInput {
                            queries: attn_queries,
                            cross_pass: attn_cross_pass,
                            norm: attn_norm,
                        }),
                        op: Operator::Attention(Box::new(attention)),
                        post_conv: Some(attn_post_conv),
                        output: OutputForm::BlockCommit,
                    },
                    Sublayer {
                        input: InputNorm::BlockAttention(BlockAttentionInput {
                            queries: BlockAttentionQueries::Single(ffn_q),
                            cross_pass: ffn_cp,
                            norm: BlockAttentionNorm::Shared(ffn_norm),
                        }),
                        op: Operator::RoutedFfn(Box::new(routed)),
                        post_conv: Some(ffn_post_conv),
                        output: OutputForm::Residual,
                    },
                ],
            });
        }

        let decoder = Decoder {
            activation_dtype: ActivationDType::BF16,
            hidden,
            vocabulary: vocab_padded,
            context_limit: 64,
            residual: ResidualForm::BlockAttention {
                block_size,
                normalize_values: true,
                norm_epsilon: eps,
            },
            entry: EntryForm {
                embedding: emb_desc,
                scale: EmbeddingScale::Unit,
                norm: Some(EntryNorm::Rms {
                    norm: RmsNorm {
                        weight: entry_norm_desc,
                        epsilon: eps,
                    },
                    text_only: true,
                }),
                per_layer: None,
            },
            blocks,
            exit: ExitForm {
                norm: ExitNorm::BlockAttention {
                    query: out_q_desc,
                    norm: RmsNorm {
                        weight: out_norm_desc,
                        epsilon: eps,
                    },
                },
                output: out_desc,
                softcap: Some(30.0),
                unpadded_vocabulary: Some(vocab_unpadded),
            },
        };
        decoder.validate(1).expect("synthetic inkling2 decoder validates");
        (decoder, host_weights)
    }

    fn upload_weights(
        device: &Device,
        decoder: &Decoder,
        host: &HashMap<WeightRole, Vec<f32>>,
    ) -> HashMap<WeightRole, Tensor> {
        let charges = Charges::tally(device);
        let mut all_descs: HashMap<WeightRole, WeightDescriptor> = HashMap::new();
        all_descs.insert(
            WeightRole {
                scope: WeightScope::Target,
                kind: WeightKind::Embedding,
            },
            decoder.entry.embedding.clone(),
        );
        if let Some(EntryNorm::Rms { norm, .. }) = &decoder.entry.norm {
            all_descs.insert(
                WeightRole {
                    scope: WeightScope::Target,
                    kind: WeightKind::EntryNorm,
                },
                norm.weight.clone(),
            );
        }
        for (role, desc) in operators::decoder_weights(decoder) {
            all_descs.insert(role, desc.clone());
        }
        for (role, desc) in operators::exit_weights(decoder) {
            all_descs.insert(role, desc.clone());
        }
        all_descs.insert(
            WeightRole {
                scope: WeightScope::Target,
                kind: WeightKind::Output,
            },
            operators::exit_output_descriptor(decoder)
                .unwrap()
                .into_owned(),
        );

        let mut gpu = HashMap::new();
        for (role, desc) in all_descs {
            let vals = host.get(&role).unwrap_or_else(|| panic!("missing host {role:?}"));
            let n = desc.shape.iter().product::<u64>() as usize;
            let vals = &vals[..n];
            let is_f32 = operators::is_fixed_dense_weight(role.kind, &desc);
            let tensor = if is_f32 {
                f32_tensor(&charges, &desc.shape, vals).unwrap()
            } else {
                bf16_tensor(&charges, &desc.shape, vals)
            };
            gpu.insert(role, tensor);
        }
        gpu
    }

    fn alloc_layers(
        device: &Device,
        decoder: &Decoder,
    ) -> (Vec<seismic::SlabTensor>, Vec<BlockAttnResLayerState>) {
        let charges = Charges::tally(device);
        let t_cap = 16u64;
        let banks = 4u64;
        let pool = seismic::Pool::open(
            &seismic::PoolCharges::one(&charges),
            seismic::Contract::DeviceOnly,
            1 << 20,
            64 << 20,
        )
        .unwrap();
        let mut slabs = Vec::new();
        let layers = decoder
            .blocks
            .iter()
            .map(|block| {
                let Operator::Attention(attn) = &block.sublayers[0].op else {
                    unreachable!()
                };
                let kv_dim = attn.kv_heads * attn.width;
                let kv_regions = [
                    seismic::SlabRegion {
                        element: Element::bf16(),
                        row_shape: vec![attn.kv_heads, attn.width],
                    },
                    seismic::SlabRegion {
                        element: Element::bf16(),
                        row_shape: vec![attn.kv_heads, attn.width],
                    },
                ];
                let kv_slab =
                    seismic::SlabTensor::in_pool(&charges, &pool, t_cap, t_cap, &kv_regions)
                        .unwrap();
                kv_slab.add_slab(seismic::SlabFill::Zeroed).unwrap();
                let history_key = kv_slab.logical_region(0).unwrap();
                let history_value = kv_slab.logical_region(1).unwrap();
                slabs.push(kv_slab);

                let conv_regions = [
                    seismic::SlabRegion {
                        element: Element::f32(),
                        row_shape: vec![4, kv_dim],
                    },
                    seismic::SlabRegion {
                        element: Element::f32(),
                        row_shape: vec![4, kv_dim],
                    },
                    seismic::SlabRegion {
                        element: Element::f32(),
                        row_shape: vec![4, decoder.hidden],
                    },
                    seismic::SlabRegion {
                        element: Element::f32(),
                        row_shape: vec![4, decoder.hidden],
                    },
                ];
                let conv_slab =
                    seismic::SlabTensor::in_pool(&charges, &pool, banks, banks, &conv_regions)
                        .unwrap();
                conv_slab.add_slab(seismic::SlabFill::Zeroed).unwrap();
                let key_conv_window = Some(conv_slab.logical_region(0).unwrap());
                let value_conv_window = Some(conv_slab.logical_region(1).unwrap());
                let attn_post_conv_window = Some(conv_slab.logical_region(2).unwrap());
                let ffn_post_conv_window = Some(conv_slab.logical_region(3).unwrap());
                slabs.push(conv_slab);

                BlockAttnResLayerState {
                    history_key,
                    history_value,
                    key_conv_window,
                    value_conv_window,
                    attn_post_conv_window,
                    ffn_post_conv_window,
                    slab_rows: t_cap as u32,
                    slab_banks: banks as u32,
                }
            })
            .collect();
        (slabs, layers)
    }

    #[test]
    fn inkling2_non_looped_and_looped_forward_end_to_end() {
        let catalog = seismic::Devices::discover(&[]).expect("devices discover");
        for backend in [BackendName::Cpu, BackendName::Metal] {
            let Ok(device) = crate::test_support::open_backend(&catalog, backend) else {
                continue;
            };
            for looped in [false, true] {
                let (decoder, host_w) = make_synthetic_inkling2(looped);

                // Verify state layout derives KV dense history + 4 ConvWindow arenas per block!
                let layout = ModelStateLayout::derive(&decoder, &[], KvCodec::Dense, 1)
                    .expect("state layout derives");
                assert_eq!(
                    layout.target_recurrent.len(),
                    decoder.blocks.len() * 4,
                    "4 CausalShortConv windows per logical visit (looped={looped})"
                );

                // Verify static schedule.
                let plan = BlockAttnResPlan::of(&decoder).expect("schedule builds");
                if looped {
                    assert_eq!(plan.max_bank, 1);
                    assert_eq!(plan.blocks[1].transition, Some(BlockTransition::SnapshotPrelude));
                    assert_eq!(
                        plan.blocks[2].transition,
                        Some(BlockTransition::ResetToPreludeAndBank)
                    );
                } else {
                    assert_eq!(plan.max_bank, 0);
                    assert_eq!(plan.final_sources, 3);
                }

                let gpu_w = upload_weights(&device, &decoder, &host_w);
                let prepared = Arc::new(
                    PreparedBlockAttnRes::prepare_from_weights(
                        &device,
                        &decoder,
                        &gpu_w,
                        vec![
                            TapPoint::Sublayer(SublayerIndex {
                                block: 1,
                                sublayer: 0,
                            }),
                            TapPoint::Exit,
                        ],
                    )
                    .expect("PreparedBlockAttnRes prepares"),
                );
                let bound = BoundBlockAttnRes::from_weights(&device, prepared, gpu_w);
                let (_slabs, mut layers) = alloc_layers(&device, &decoder);

                // Step 1: 3-token prefill (with 1 media overlay on token 1 bypassing EntryNorm)
                let media_row = pattern(decoder.hidden as usize, 999, 0.4);
                let vis_prefill: Vec<Vec<i32>> = vec![vec![0, 0, 0, 0, 0, 0]; decoder.blocks.len()];
                let prefill_out = bound
                    .step_hidden(
                        &decoder,
                        None,
                        &mut layers,
                        &BlockAttnResStepInput {
                            tokens: &[1, 2, 3],
                            coordinates: &[0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0],
                            media_overlays: &[(1, 2, &media_row)],
                            visible_per_layer: &vis_prefill,
                            spans: 1,
                            fresh: &[0, 1, 0, 2, 0, 3],
                            destinations: &[0, 1, 2],
                            conv_segments: &[0, 3, 3, 3],
                            conv_stop: &[3],
                            conv_prev_bank: &[0],
                            conv_prev_tape: &[0],
                            conv_next_bank: &[1],
                            tap_points: &[
                                TapPoint::Sublayer(SublayerIndex {
                                    block: 1,
                                    sublayer: 0,
                                }),
                                TapPoint::Exit,
                            ],
                            bank_run: None,
                        },
                    )
                    .expect("prefill step succeeds");
                let prefill_logits_t = bound
                    .compute_logits(&device, &decoder, &prefill_out.exit_hidden)
                    .expect("prefill logits");
                let prefill_logits = read_f32(&prefill_logits_t).unwrap();
                assert_eq!(
                    prefill_logits.len(),
                    3 * decoder.active_vocabulary() as usize,
                    "logits sliced to unpadded_vocabulary"
                );
                assert!(
                    prefill_logits.iter().all(|v| v.is_finite() && v.abs() <= 30.001),
                    "all logits finite and within softcap=30 ({backend:?}, looped={looped})"
                );
                assert_eq!(prefill_out.taps.len(), 2);

                // Step 2: 1-token decode attending KV [0..3] + fresh [0..1], continuing conv bank 1 -> 2
                let vis_decode: Vec<Vec<i32>> = vec![vec![0, 3]; decoder.blocks.len()];
                let decode_out = bound
                    .step_hidden(
                        &decoder,
                        None,
                        &mut layers,
                        &BlockAttnResStepInput {
                            tokens: &[4],
                            coordinates: &[3, 0, 0, 0],
                            media_overlays: &[],
                            visible_per_layer: &vis_decode,
                            spans: 1,
                            fresh: &[0, 1],
                            destinations: &[3],
                            conv_segments: &[0, 1, 1, 1],
                            conv_stop: &[1],
                            conv_prev_bank: &[1],
                            conv_prev_tape: &[0],
                            conv_next_bank: &[2],
                            tap_points: &[TapPoint::Exit],
                            bank_run: None,
                        },
                    )
                    .expect("decode step succeeds");
                let decode_logits_t = bound
                    .compute_logits(&device, &decoder, &decode_out.exit_hidden)
                    .expect("decode logits");
                let decode_logits = read_f32(&decode_logits_t).unwrap();
                assert_eq!(decode_logits.len(), decoder.active_vocabulary() as usize);
                assert!(
                    decode_logits.iter().all(|v| v.is_finite() && v.abs() <= 30.001),
                    "decode logits finite and within softcap=30 ({backend:?}, looped={looped})"
                );
                // Step 3: 16-row prefill (10 real tokens + 6 padded rows) to exercise M > VERIFY_ROWS
                // (`attention_rel_bias_prefill`, `routed_select_select`, padded row handling).
                let m16_tokens = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 0, 0, 0, 0, 0];
                let mut m16_coords = Vec::with_capacity(16 * 4);
                let mut m16_fresh = Vec::with_capacity(16 * 2);
                let mut m16_dest = Vec::with_capacity(16);
                for r in 0..16i32 {
                    m16_coords.extend_from_slice(&[r.min(9), 0, 0, 0]);
                    if r < 10 {
                        m16_fresh.extend_from_slice(&[0, r + 1]);
                        m16_dest.push(4 + r);
                    } else {
                        m16_fresh.extend_from_slice(&[0, 0]);
                        m16_dest.push(-1);
                    }
                }
                let vis_m16: Vec<Vec<i32>> = vec![vec![0; 16 * 2]; decoder.blocks.len()];
                let m16_out = bound
                    .step_hidden(
                        &decoder,
                        None,
                        &mut layers,
                        &BlockAttnResStepInput {
                            tokens: &m16_tokens,
                            coordinates: &m16_coords,
                            media_overlays: &[],
                            visible_per_layer: &vis_m16,
                            spans: 1,
                            fresh: &m16_fresh,
                            destinations: &m16_dest,
                            conv_segments: &[0, 10, 10, 10],
                            conv_stop: &[10],
                            conv_prev_bank: &[0],
                            conv_prev_tape: &[0],
                            conv_next_bank: &[3],
                            tap_points: &[TapPoint::Exit],
                            bank_run: None,
                        },
                    )
                    .expect("16-row prefill succeeds");
                let m16_logits_t = bound
                    .compute_logits(&device, &decoder, &m16_out.exit_hidden)
                    .expect("m16 logits");
                let m16_logits = read_f32(&m16_logits_t).unwrap();
                assert!(
                    m16_logits.iter().all(|v| v.is_finite() && v.abs() <= 30.001),
                    "16-row prefill logits finite ({backend:?}, looped={looped})"
                );
            }
        }
    }
}
