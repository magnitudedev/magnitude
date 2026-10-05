//! Ordered callable native slots constructed from the one ProgramPlan.
//! Preparation is the only place a checked binding lookup is permitted.

use super::preparation::PreparationInputs;
use super::tuning::{TunedEntry, TuningContext, TuningLimits};
use super::*;
use crate::operators::gated_delta::graph::StepForm;
use crate::operators::routed::fused_graph::DecodeForm;
use crate::SublayerTail;
use crate::{
    ExecutionPlanDraft, FeedForwardProgramSlot, HeadBinding, HeadProjection, ImportProgramSlot,
    MixerProgramSlot,
    PlanError, PlannedDevice, ProgramPlan, ReadoutHead,
};
use magnitude_family_contracts::{ProgressivePlane, SublayerIndex};
use magnitude_kernels::{
    draft_confidence, draft_convolve_input, draft_convolve_residual, draft_gated_rows,
    draft_path_step, draft_top_k, feature_rows, import_dense, post_norm_residual, project_rows,
    repack_weight, tap_rows, widen_rows,
};
use magnitude_state::KvCodec;
use std::{collections::HashSet, rc::Rc};

/// Invocation workspace of the fused routed form's decode kernels.
fn routed_decode_bytes(decode: &RoutedDecodeKernels) -> u128 {
    match decode {
        RoutedDecodeKernels::Expand(expand) => u128::from(expand.invocation_workspace_bytes()),
        RoutedDecodeKernels::SharedRoute { route, choices } => {
            u128::from(route.invocation_workspace_bytes())
                + u128::from(choices.invocation_workspace_bytes())
        }
    }
}

/// Invocation workspace of the recurrent step row classes' kernels.
fn recurrent_step_bytes(step: &RecurrentStepKernels) -> u128 {
    match step {
        RecurrentStepKernels::Step(step) => u128::from(step.invocation_workspace_bytes()),
        RecurrentStepKernels::Convolved { project, step } => {
            u128::from(project.invocation_workspace_bytes())
                + u128::from(step.invocation_workspace_bytes())
        }
    }
}

pub struct AttestedPrograms {
    backend: BackendName,
    tuned: Vec<TunedEntry>,
    pub(super) owner: Tensor,
    report: QualificationReport,
    pub(super) target: AttestedTarget,
    pub(super) head: Option<AttestedHead>,
    pub(super) draft: Option<AttestedDraft>,
    pub(super) vision: Option<AttestedVision>,
    pub(super) state: AttestedState,
    pub(super) imports: Vec<(ImportProgramSlot, AttestedImport)>,
    invocation_workspace_bytes: u64,
    target_graphs: Option<crate::PreparedTargetGraphs>,
    target_readout_graphs: Option<crate::PreparedTargetReadoutGraphs>,
    drafter_graphs: Option<crate::PreparedDrafterGraphs>,
    vision_graphs: Option<Rc<crate::programs::native_vision::PreparedVisionGraphs>>,
    state_graphs: Option<Rc<crate::programs::native_state::PreparedStateCopyGraphs>>,
}

#[derive(Clone)]
pub(crate) struct AttestedTarget {
    pub embedding: NativeKernel<embedding_rows::Entry>,
    pub blocks: Vec<AttestedTargetBlock>,
    pub readout: ReadoutKernels,
    pub features: Option<NativeKernel<readout_features_rows::Entry>>,
    pub shape: NativeKernel<shape_rows::Entry>,
    pub sample: NativeKernel<sample_rows::Entry>,
    /// A separate draft's taps, when one drafts.
    pub taps: Option<TapKernels>,
    /// The per-layer entry, when the model has per-layer inputs.
    pub per_layer: Option<PerLayerEntryKernels>,
}

#[derive(Clone)]
pub(crate) struct AttestedTargetBlock {
    pub mixer: AttestedMixer,
    /// Absent for a lone mixer block.
    pub feed_forward: Option<AttestedFeedForward>,
    /// The post-norm output scales of the block's sublayers.
    pub output_scales: OutputScales,
    /// The per-layer input sublayer's entries, when the block has one.
    pub per_layer: Option<PerLayerKernels>,
}

/// The factor a sublayer's post-norm row op scales its result by: 1, or the
/// layer's output scale (`OutputForm::ScaledPostNorm`), read from the
/// artifact because it is a value of the sealed graph.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct OutputScales {
    pub mixer: f32,
    pub feed_forward: f32,
    pub per_layer: f32,
}

/// The F32 source values of a planned role, read through the tuning weight
/// source.
fn source_f32s(
    load: &crate::ModelLoadPlan,
    tuning: TuningContext<'_>,
    role: magnitude_family_contracts::WeightRole,
) -> Result<Vec<f32>, String> {
    let plan = load
        .weights()
        .find(|weight| weight.role == role)
        .ok_or_else(|| format!("{role:?} is not planned"))?;
    if plan.source != seismic::Element::f32() {
        return Err(format!(
            "{role:?} is stored as {}, not f32",
            plan.source.name()
        ));
    }
    let bytes = tuning.weights.source_bytes(plan)?;
    if bytes.len() % 4 != 0 {
        return Err(format!("{role:?} holds {} bytes", bytes.len()));
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect())
}

/// Reject an artifact whose stored rotary divisors (`rope_freqs`) do not
/// reproduce the rotary tables the family derived from its headers.
fn check_rotary_divisors(
    load: &crate::ModelLoadPlan,
    tuning: TuningContext<'_>,
) -> Result<(), CatalogFailure> {
    for (index, sublayer) in tuning.definition.decoder.sublayers() {
        let magnitude_family_contracts::Operator::Attention(attention) = &sublayer.op else {
            continue;
        };
        let magnitude_family_contracts::Rotary::Table {
            pairs,
            divisors: Some(divisors),
        } = &attention.rotary
        else {
            continue;
        };
        let role = magnitude_family_contracts::WeightRole {
            scope: magnitude_family_contracts::WeightScope::TargetSublayer(index),
            kind: magnitude_family_contracts::WeightKind::RotaryDivisors,
        };
        source_f32s(load, tuning, role)
            .and_then(|stored| {
                crate::operators::attention::check_rotary_divisors(pairs, &divisors.bases, &stored)
            })
            .map_err(|outcome| CatalogFailure::Preparation {
                entry: "attention_decode",
                bindings: format!("rotary divisors of {index:?}"),
                outcome,
            })?;
    }
    Ok(())
}

/// The output scale of the sublayer `index` with `tail`.
fn output_scale(
    load: &crate::ModelLoadPlan,
    tuning: TuningContext<'_>,
    index: SublayerIndex,
    tail: Option<SublayerTail>,
) -> Result<f32, CatalogFailure> {
    let failure = |outcome: String| CatalogFailure::Preparation {
        entry: "post_norm_residual",
        bindings: format!("layer scale of {index:?}"),
        outcome,
    };
    if !matches!(tail, Some(SublayerTail::PostNorm { scaled: true, .. })) {
        return Ok(1.0);
    }
    let role = magnitude_family_contracts::WeightRole {
        scope: magnitude_family_contracts::WeightScope::TargetSublayer(index),
        kind: magnitude_family_contracts::WeightKind::LayerScale,
    };
    match source_f32s(load, tuning, role).map_err(failure)?[..] {
        [scale] => Ok(scale),
        ref values => Err(failure(format!(
            "the layer scale holds {} values",
            values.len()
        ))),
    }
}

#[derive(Clone)]
pub(crate) enum AttestedMixer {
    Attention(AttentionKernels),
    Recurrent(RecurrentKernels),
    StateSpace(StateSpaceKernels),
    ShortConv(ShortConvKernels),
}

#[derive(Clone)]
pub(crate) enum AttestedFeedForward {
    Dense(DenseKernels),
    Routed(RoutedKernels),
    GeneralRouted(GeneralRoutedKernels),
    Parallel(ParallelKernels),
}

#[derive(Clone)]
pub(crate) struct AttestedHead {
    pub blocks: Vec<AttestedHeadBlock>,
    /// Token selection over the draft vocabulary.
    pub shape: NativeKernel<shape_rows::Entry>,
    pub sample: NativeKernel<sample_rows::Entry>,
}

#[derive(Clone)]
pub(crate) struct AttestedHeadBlock {
    pub binding: HeadBinding,
    pub input: NativeKernel<draft_rows::Entry>,
    pub attention: AttentionKernels,
    pub feed_forward: AttestedFeedForward,
    pub features: NativeKernel<readout_features_rows::Entry>,
    pub logits: HeadLogitsKernels,
}

/// A separate draft's entries: per layer in draft order, then the block
/// embedding, the proposing rows' norm and projection, and DSpark's chain.
/// Selection is the target's.
#[derive(Clone)]
pub(crate) struct AttestedDraft {
    pub blocks: Vec<DraftBlockKernels>,
    pub embedding: NativeKernel<embedding_rows::Entry>,
    pub head: NativeKernel<readout_head_rows::Entry>,
    /// Token selection over the readout vocabulary.
    pub shape: NativeKernel<shape_rows::Entry>,
    pub sample: NativeKernel<sample_rows::Entry>,
    /// Widens a device-conditioned entry's target features to F32.
    pub widen: NativeKernel<widen_rows::Entry>,
    pub markov: Option<MarkovKernels>,
    pub dflash2: Option<super::AttestedDflash2>,
}

/// Every kernel of the projector's vision program.
#[derive(Clone)]
pub(crate) struct AttestedVision {
    pub kernels: super::VisionKernels,
}

#[derive(Clone)]
pub(crate) struct AttestedState {
    pub copies: Vec<(Element, NativeKernel<copy_rows::Entry>)>,
    pub conditioning: Option<NativeKernel<conditioning_overlay::Entry>>,
}

#[derive(Clone)]
pub(crate) enum AttestedImport {
    Dense(NativeKernel<import_dense::Entry>),
    Repack(NativeKernel<repack_weight::Entry>),
}

fn missing(entry: &'static str, binding: impl fmt::Debug) -> CatalogFailure {
    CatalogFailure::Qualification {
        entry,
        bindings: format!("{binding:?}"),
        outcome: "ordered program slot was not prepared".into(),
    }
}

fn slot<K, E>(
    handles: &HashMap<K, NativeKernel<E>>,
    binding: K,
    entry: &'static str,
) -> Result<NativeKernel<E>, CatalogFailure>
where
    K: Copy + Eq + std::hash::Hash + fmt::Debug,
    E: seismic::Entry,
{
    handles
        .get(&binding)
        .cloned()
        .ok_or_else(|| missing(entry, binding))
}

impl AttestedPrograms {
    pub fn install_target_readout_graphs(&mut self, graphs: crate::PreparedTargetReadoutGraphs) {
        self.target_readout_graphs = Some(graphs);
    }

    pub fn target_readout_graphs(&self) -> Option<&crate::PreparedTargetReadoutGraphs> {
        self.target_readout_graphs.as_ref()
    }

    pub fn prepare_target_readout_graphs(
        &self,
        device: &Device,
        load: &crate::ModelLoadPlan,
        geometry: &magnitude_family_contracts::Decoder,
        limits: crate::ResourceLimits,
    ) -> Result<crate::PreparedTargetReadoutGraphs, String> {
        crate::programs::graph::readout::PreparedTargetReadoutGraphs::prepare(
            device,
            &self.target,
            load,
            geometry,
            limits,
        )
    }

    /// Seals the head, vision and state-copy graphs. `proposals` is the most
    /// proposals one head transaction drafts (unused without a head).
    #[allow(clippy::too_many_arguments)]
    pub fn prepare_auxiliary_graphs(
        &mut self,
        device: &Device,
        load: &crate::ModelLoadPlan,
        definition: &magnitude_family_contracts::ModelDefinition,
        vision_plan: Option<&crate::VisionProgramPlan>,
        target_state: &crate::StateStorePlan,
        head_state: Option<&crate::StateStorePlan>,
        limits: crate::ResourceLimits,
        proposals: usize,
    ) -> Result<(), String> {
        let row_classes = magnitude_batching::row_classes(limits.max_launch_rows)
            .into_iter()
            .map(|rows| rows as u64)
            .collect::<Vec<_>>();
        let max_rows = *row_classes.last().ok_or_else(|| {
            format!(
                "launch row bound {} has no row class",
                limits.max_launch_rows
            )
        })?;
        if self.vision.is_some() != vision_plan.is_some()
            || self.vision.is_some() != definition.vision.is_some()
        {
            return Err("attested vision, program plan and model definition disagree".into());
        }
        if let (Some(handles), Some(state)) = (&self.head, head_state) {
            let head = definition
                .head
                .as_ref()
                .and_then(|head| head.blocks.first())
                .ok_or("head graph requires a draft head block")?;
            let history = state.sole_history()?;
            let history_rows =
                u64::try_from(history.rows).map_err(|_| "head history rows exceed u64")?;
            let classes = crate::programs::native_head::head_graph_classes(
                limits,
                history_rows,
                history.slab_rows,
                state.span_limit(),
                proposals,
            )?;
            self.drafter_graphs = Some(crate::PreparedDrafterGraphs::Head(Rc::new(
                crate::programs::native_head::PreparedHeadGraphs::prepare(
                    device,
                    handles,
                    load,
                    &definition.decoder,
                    head,
                    classes,
                )
                .map_err(|error| error.to_string())?,
            )));
        }
        if let (Some(handles), Some(state)) = (&self.draft, head_state) {
            let draft = definition
                .draft
                .as_ref()
                .ok_or("draft graphs require a draft")?;
            let geometry =
                crate::programs::native_draft::DraftGeometry::new(definition, state, proposals)?;
            let classes =
                crate::programs::native_draft::draft_graph_classes(limits, proposals, draft)?;
            self.drafter_graphs = Some(crate::PreparedDrafterGraphs::Draft(Rc::new(
                crate::programs::native_draft::PreparedDraftGraphs::prepare(
                    device,
                    handles,
                    &self.target,
                    load,
                    &geometry,
                    classes,
                )
                .map_err(|error| error.to_string())?,
            )));
        }
        if let (Some(handles), Some(vision), Some(_)) =
            (&self.vision, definition.vision.as_ref(), vision_plan)
        {
            let cell = vision.cell_rows();
            max_rows
                .checked_mul(cell)
                .ok_or("vision patch row bound overflow")?;
            self.vision_graphs = Some(Rc::new(
                crate::programs::native_vision::PreparedVisionGraphs::prepare_exact_classes(
                    device,
                    handles,
                    load,
                    vision,
                    (1..=max_rows).map(|outputs| outputs * cell),
                )
                .map_err(|error| error.to_string())?,
            ));
        }
        let state_classes = crate::programs::native_state::state_copy_classes(
            target_state,
            head_state,
            &row_classes,
        )?;
        self.state_graphs = Some(Rc::new(
            crate::programs::native_state::PreparedStateCopyGraphs::prepare(
                device,
                &self.state,
                state_classes,
            )
            .map_err(|error| error.to_string())?,
        ));
        Ok(())
    }

    /// The drafter's graphs: the draft head's or the separate draft's.
    pub fn drafter_graphs(&self) -> Option<&crate::PreparedDrafterGraphs> {
        self.drafter_graphs.as_ref()
    }

    pub fn vision_graphs(
        &self,
    ) -> Option<&Rc<crate::programs::native_vision::PreparedVisionGraphs>> {
        self.vision_graphs.as_ref()
    }

    pub fn state_graphs(
        &self,
    ) -> Option<&Rc<crate::programs::native_state::PreparedStateCopyGraphs>> {
        self.state_graphs.as_ref()
    }

    pub fn install_target_graphs(&mut self, graphs: crate::PreparedTargetGraphs) {
        self.target_graphs = Some(graphs);
    }

    pub fn target_graphs(&self) -> Option<&crate::PreparedTargetGraphs> {
        self.target_graphs.as_ref()
    }

    pub fn prepare_target_graphs(
        &self,
        device: &Device,
        load: &crate::ModelLoadPlan,
        geometry: &magnitude_family_contracts::Decoder,
        state: &crate::StateResourcePlan,
        plan: &crate::TargetProgramPlan,
        limits: crate::ResourceLimits,
    ) -> Result<crate::PreparedTargetGraphs, String> {
        crate::programs::native_target_graph::PreparedTargetGraphs::prepare(
            device,
            &self.target,
            load,
            geometry,
            state,
            plan,
            limits,
        )
    }

    /// Device bytes reserved by the exact native specializations before
    /// construction on `backend`. Duplicate bindings share one prepared
    /// handle, matching the native factory's preparation and the actual
    /// measured charge.
    pub fn planned_invocation_workspace_bytes(
        plan: &ProgramPlan,
        backend: BackendName,
    ) -> Result<u64, PlanError> {
        macro_rules! bytes {
            ($entry:ident) => {
                u128::from(NativeKernel::<$entry::Entry>::planned_invocation_workspace_bytes())
            };
        }
        // The fused routed form's entries: the decode ones by the backend's
        // form.
        let routed_bytes = bytes!(routed_route)
            + match DecodeForm::of(backend).map_err(PlanError::ResourcePlanning)? {
                DecodeForm::Expand => bytes!(routed_expand),
                DecodeForm::SharedRoute => bytes!(routed_route_shared) + bytes!(routed_gate_up),
            }
            + bytes!(routed_output)
            + bytes!(routed_group)
            + bytes!(routed_experts)
            + bytes!(routed_combine);
        fn general_routed_bytes(shape: crate::GeneralRoutedShape) -> u128 {
            let mut bytes = bytes!(routed_select)
                + bytes!(routed_down)
                + bytes!(routed_group)
                + bytes!(routed_scatter)
                + if shape.experts_expansion.gated {
                    bytes!(routed_gate_up) + bytes!(routed_experts)
                } else {
                    bytes!(routed_up) + bytes!(routed_experts_up)
                };
            if let Some((_, expansion)) = shape.shared {
                bytes += bytes!(dense_output)
                    + if expansion.gated {
                        bytes!(dense_expand)
                    } else {
                        bytes!(dense_up)
                    };
            }
            if shape.latent {
                bytes += bytes!(project_rows) + bytes!(dense_output);
            }
            bytes
        }
        let mut charged_imports = HashSet::new();
        let mut charged_copies = HashSet::new();
        let mut charged_mixers = HashSet::new();
        let mut charged_feed_forward = HashSet::new();
        let mut charged_heads = HashSet::new();
        // The native catalog's one-word device identity owner is retained
        // with every prepared group.
        let mut bytes = 4u128;
        for slot in plan.imports() {
            if charged_imports.insert(*slot) {
                bytes += match slot {
                    ImportProgramSlot::Dense { .. } => bytes!(import_dense),
                    ImportProgramSlot::Repack { .. } => bytes!(repack_weight),
                };
            }
        }
        for &element in plan.state().copies() {
            if charged_copies.insert(element) {
                bytes += bytes!(copy_rows);
            }
        }
        bytes += bytes!(conditioning_overlay);
        bytes += bytes!(shape_rows) + bytes!(sample_rows);
        let target = plan.target();
        bytes += bytes!(embedding_rows);
        for block in target.blocks() {
            match block.mixer() {
                MixerProgramSlot::Attention(binding)
                    if charged_mixers.insert(MixerProgramSlot::Attention(binding)) =>
                {
                    bytes += bytes!(attention_project)
                        + match binding.tail {
                            SublayerTail::Residual => bytes!(attention_output),
                            SublayerTail::PostNorm { .. } => {
                                bytes!(project_rows) + bytes!(post_norm_residual)
                            }
                        }
                        + match binding.history {
                            KvCodec::Dense => bytes!(attention_decode) + bytes!(attention_prefill),
                            KvCodec::AffineK8V4 => {
                                bytes!(attention_decode_k8v4) + bytes!(attention_prefill_k8v4)
                            }
                            KvCodec::RotatedK4V4 => {
                                return Err(PlanError::Unsupported("native rotated K4/V4 KV codec"))
                            }
                        }
                }
                MixerProgramSlot::Recurrent(binding)
                    if charged_mixers.insert(MixerProgramSlot::Recurrent(binding)) =>
                {
                    bytes += bytes!(gated_delta_project)
                        + match StepForm::of(backend).map_err(PlanError::ResourcePlanning)? {
                            StepForm::Step => bytes!(gated_delta_step),
                            StepForm::Convolved => {
                                bytes!(gated_delta_project_convolved)
                                    + bytes!(gated_delta_step_convolved)
                            }
                        }
                        + bytes!(gated_delta_chunk)
                        + bytes!(attention_output)
                }
                MixerProgramSlot::StateSpace(binding)
                    if charged_mixers.insert(MixerProgramSlot::StateSpace(binding)) =>
                {
                    bytes += bytes!(attention_project)
                        + bytes!(state_space_step)
                        + bytes!(state_space_chunk)
                        + bytes!(state_space_gate)
                        + bytes!(attention_output)
                }
                MixerProgramSlot::ShortConv(binding)
                    if charged_mixers.insert(MixerProgramSlot::ShortConv(binding)) =>
                {
                    bytes += bytes!(short_conv_project)
                        + bytes!(short_conv_rows)
                        + bytes!(attention_output)
                }
                _ => {}
            }
            match block.feed_forward() {
                Some(FeedForwardProgramSlot::Dense(binding))
                    if charged_feed_forward.insert(FeedForwardProgramSlot::Dense(binding)) =>
                {
                    bytes += bytes!(dense_expand)
                        + match binding.tail {
                            SublayerTail::Residual => bytes!(dense_output),
                            SublayerTail::PostNorm { .. } => {
                                bytes!(project_rows) + bytes!(post_norm_residual)
                            }
                        }
                }
                Some(FeedForwardProgramSlot::Routed(binding))
                    if charged_feed_forward.insert(FeedForwardProgramSlot::Routed(binding)) =>
                {
                    bytes += routed_bytes
                }
                Some(FeedForwardProgramSlot::GeneralRouted(binding))
                    if charged_feed_forward
                        .insert(FeedForwardProgramSlot::GeneralRouted(binding)) =>
                {
                    bytes += general_routed_bytes(binding.shape);
                }
                Some(FeedForwardProgramSlot::Parallel(binding))
                    if charged_feed_forward.insert(FeedForwardProgramSlot::Parallel(binding)) =>
                {
                    bytes += bytes!(dense_expand)
                        + bytes!(project_rows)
                        + general_routed_bytes(binding.routed.shape)
                        + bytes!(moe_tail);
                }
                _ => {}
            }
        }
        let mut charged_per_layer = HashSet::new();
        for binding in target.blocks().iter().filter_map(|block| block.per_layer()) {
            if charged_per_layer.insert(binding) {
                bytes += bytes!(per_layer_gate) + bytes!(project_rows) + bytes!(post_norm_residual);
            }
        }
        if target.per_layer().is_some() {
            bytes += 2 * bytes!(import_dense)
                + bytes!(repack_weight)
                + bytes!(project_rows)
                + bytes!(per_layer_inputs);
        }
        bytes += bytes!(readout_features_rows)
            + match target.readout().head {
                ReadoutHead::Packed { .. } => {
                    bytes!(readout_head_rows) + bytes!(readout_selected_rows)
                }
                ReadoutHead::Progressive => {
                    bytes!(readout_top_rows)
                        + bytes!(readout_refine_rows)
                        + bytes!(readout_exact_rows)
                        + bytes!(readout_planes_rows)
                }
            };
        if target.features().is_some() {
            bytes += bytes!(readout_features_rows);
        }
        if target.taps().is_some() {
            bytes += bytes!(tap_rows) + bytes!(project_rows) + bytes!(feature_rows);
        }
        if let Some(head) = plan.head() {
            // Token selection over the draft vocabulary.
            bytes += bytes!(shape_rows) + bytes!(sample_rows);
            for &binding in head.blocks() {
                if charged_heads.insert(binding) {
                    bytes += bytes!(draft_rows)
                        + bytes!(attention_project)
                        + bytes!(attention_decode)
                        + bytes!(attention_prefill)
                        + bytes!(attention_output)
                        + bytes!(readout_features_rows);
                    bytes += match binding.projection {
                        HeadProjection::Packed(_) => bytes!(head_logits_rows),
                        HeadProjection::Progressive => {
                            bytes!(readout_top_rows)
                                + bytes!(readout_refine_rows)
                                + bytes!(readout_exact_rows)
                                + bytes!(readout_planes_rows)
                        }
                    };
                    match binding.feed_forward {
                        FeedForwardProgramSlot::Dense(_) => {
                            bytes += bytes!(dense_expand) + bytes!(dense_output)
                        }
                        FeedForwardProgramSlot::Routed(_) => bytes += routed_bytes,
                        // `operators::admit` keeps draft heads on the fused
                        // form.
                        FeedForwardProgramSlot::GeneralRouted(_)
                        | FeedForwardProgramSlot::Parallel(_) => {
                            return Err(PlanError::Unsupported(
                                "draft head routed feed-forward form",
                            ))
                        }
                    }
                }
            }
        }
        if let Some(draft) = plan.draft() {
            // One prepared group per distinct binding, apart from the
            // target's.
            let mut attention = HashSet::new();
            let mut dense = HashSet::new();
            for block in draft.blocks() {
                for binding in [block.attention, block.injection] {
                    if attention.insert(binding) {
                        bytes += bytes!(attention_project)
                            + bytes!(attention_output)
                            + bytes!(attention_decode)
                            + bytes!(attention_prefill);
                    }
                }
                if dense.insert(block.feed_forward) {
                    bytes += bytes!(dense_expand) + bytes!(dense_output);
                }
            }
            bytes += bytes!(embedding_rows)
                + bytes!(readout_head_rows)
                + bytes!(shape_rows)
                + bytes!(sample_rows)
                + bytes!(widen_rows);
            if draft.markov().is_some() {
                bytes += bytes!(embedding_rows)
                    + bytes!(dense_output)
                    + bytes!(readout_features_rows)
                    + bytes!(draft_confidence);
            }
            if let Some(binding) = draft.dflash2() {
                let (projections, norms) = super::draft::dflash2_entries(draft, binding);
                bytes += projections.len() as u128 * bytes!(project_rows)
                    + norms.len() as u128 * bytes!(readout_features_rows)
                    + bytes!(draft_convolve_input)
                    + bytes!(draft_convolve_residual)
                    + bytes!(draft_gated_rows)
                    + bytes!(draft_top_k)
                    + 2 * bytes!(embedding_rows)
                    + bytes!(draft_path_step);
            }
        }
        if let Some(vision) = plan.vision() {
            for kernel in vision.kernels() {
                bytes += match kernel.entry {
                    VisionEntry::PatchStem => bytes!(vision_patch_stem),
                    VisionEntry::Norm => bytes!(vision_norm),
                    VisionEntry::Linear => bytes!(vision_linear),
                    VisionEntry::Clamp => bytes!(vision_clamp),
                    VisionEntry::Attention => bytes!(vision_attention),
                    VisionEntry::Pool => bytes!(vision_pool),
                    VisionEntry::Position => bytes!(vision_position),
                    VisionEntry::PostNormResidual => bytes!(post_norm_residual),
                };
            }
        }
        u64::try_from(bytes)
            .map_err(|_| PlanError::Arithmetic("native invocation workspace bytes overflow"))
    }

    /// Prepare every native entry the draft's program plan needs on the
    /// opened device: static values from the model, parameters tuned on the
    /// device, qualification. Errors name the native path and the device's
    /// backend; entries without an implementation for that backend are
    /// reported together.
    pub fn prepare_draft(
        plan: &ExecutionPlanDraft,
        device: &Device,
        tuning: TuningContext<'_>,
    ) -> Result<Self, CatalogError> {
        Self::prepare_for(
            plan.policy().path(),
            plan.device(),
            plan.programs(),
            plan.load(),
            TuningLimits::of(plan.policy().limits(), tuning.definition),
            device,
            tuning,
        )
        .map_err(|failure| CatalogError::native(device.backend(), failure))
    }

    #[allow(clippy::too_many_arguments)]
    fn prepare_for(
        path: ExecutionPath,
        planned_device: &PlannedDevice,
        topology: &ProgramPlan,
        load: &crate::ModelLoadPlan,
        limits: TuningLimits,
        device: &Device,
        tuning: TuningContext<'_>,
    ) -> Result<Self, CatalogFailure> {
        if path != ExecutionPath::Native {
            return Err(CatalogFailure::Preparation {
                entry: "program_factory",
                bindings: "execution path".into(),
                outcome: format!("the native program factory cannot serve the {path} path"),
            });
        }
        if planned_device.selector() != device.info().selector {
            return Err(CatalogFailure::Preparation {
                entry: "program_factory",
                bindings: "selected device".into(),
                outcome: "program plan and opened device differ".into(),
            });
        }
        let definition = tuning.definition;
        let mut prepared = NativePreparationCache::prepare_programs(PreparationInputs {
            device,
            plan: topology,
            load,
            limits,
            tuning,
        })?;
        let tuned = std::mem::take(&mut prepared.tuned);
        let mut cases = Vec::new();
        let mut include = |case| {
            if !cases.contains(&case) {
                cases.push(case);
            }
        };
        if !topology.imports().is_empty() {
            include(QualificationCase::Import);
        }
        if !topology.state().copies().is_empty() {
            include(QualificationCase::State);
        }
        include(QualificationCase::TargetEmbedding);
        for block in topology.target().blocks() {
            include(match block.mixer() {
                MixerProgramSlot::Attention(_) => QualificationCase::TargetAttention,
                MixerProgramSlot::Recurrent(_) => QualificationCase::TargetRecurrent,
                MixerProgramSlot::StateSpace(_) => QualificationCase::TargetStateSpace,
                MixerProgramSlot::ShortConv(_) => QualificationCase::TargetShortConv,
            });
            match block.feed_forward() {
                Some(FeedForwardProgramSlot::Dense(_)) => include(QualificationCase::TargetDense),
                Some(FeedForwardProgramSlot::Routed(_)) => include(QualificationCase::TargetRouted),
                Some(FeedForwardProgramSlot::GeneralRouted(_)) => {
                    include(QualificationCase::TargetGeneralRouted)
                }
                Some(FeedForwardProgramSlot::Parallel(_)) => {
                    include(QualificationCase::TargetDense);
                    include(QualificationCase::TargetGeneralRouted)
                }
                None => {}
            }
        }
        include(QualificationCase::Readout);
        include(QualificationCase::Sampling);
        if topology.head().is_some() {
            include(QualificationCase::Head);
        }
        if topology.vision().is_some() {
            include(QualificationCase::Vision);
        }
        let report = QualificationReport { cases };
        let target_plan = topology.target();
        check_rotary_divisors(load, tuning)?;
        let mut blocks = Vec::with_capacity(target_plan.blocks().len());
        for (index, block) in target_plan.blocks().iter().enumerate() {
            let sublayer = |sublayer| {
                u32::try_from(index)
                    .map(|block| SublayerIndex { block, sublayer })
                    .map_err(|_| CatalogFailure::Preparation {
                        entry: "program_factory",
                        bindings: "target block index".into(),
                        outcome: "the block index exceeds u32".into(),
                    })
            };
            let output_scales = OutputScales {
                mixer: output_scale(
                    load,
                    tuning,
                    sublayer(0)?,
                    match block.mixer() {
                        MixerProgramSlot::Attention(binding) => Some(binding.tail),
                        _ => None,
                    },
                )?,
                feed_forward: output_scale(
                    load,
                    tuning,
                    sublayer(1)?,
                    match block.feed_forward() {
                        Some(FeedForwardProgramSlot::Dense(binding)) => Some(binding.tail),
                        Some(FeedForwardProgramSlot::Parallel(binding)) => {
                            Some(SublayerTail::PostNorm {
                                norm: binding.norm,
                                scaled: binding.scaled,
                            })
                        }
                        _ => None,
                    },
                )?,
                per_layer: output_scale(
                    load,
                    tuning,
                    sublayer(2)?,
                    block.per_layer().map(|binding| binding.tail),
                )?,
            };
            let per_layer = block
                .per_layer()
                .map(|binding| {
                    prepared
                        .target
                        .per_layer
                        .get(&binding)
                        .cloned()
                        .ok_or_else(|| missing("per_layer_stages", binding))
                })
                .transpose()?;
            let mixer = match block.mixer() {
                MixerProgramSlot::Attention(binding) => AttestedMixer::Attention(
                    prepared
                        .target
                        .attention
                        .get(&binding)
                        .cloned()
                        .ok_or_else(|| missing("attention_stages", binding))?,
                ),
                MixerProgramSlot::Recurrent(binding) => AttestedMixer::Recurrent(
                    prepared
                        .target
                        .recurrent
                        .get(&binding)
                        .cloned()
                        .ok_or_else(|| missing("gated_delta_stages", binding))?,
                ),
                MixerProgramSlot::StateSpace(binding) => AttestedMixer::StateSpace(
                    prepared
                        .target
                        .state_space
                        .get(&binding)
                        .cloned()
                        .ok_or_else(|| missing("state_space_stages", binding))?,
                ),
                MixerProgramSlot::ShortConv(binding) => AttestedMixer::ShortConv(
                    prepared
                        .target
                        .short_conv
                        .get(&binding)
                        .cloned()
                        .ok_or_else(|| missing("short_conv_stages", binding))?,
                ),
            };
            let feed_forward = block
                .feed_forward()
                .map(|slot| match slot {
                    FeedForwardProgramSlot::Dense(binding) => prepared
                        .target
                        .dense
                        .get(&binding)
                        .cloned()
                        .map(AttestedFeedForward::Dense)
                        .ok_or_else(|| missing("dense_stages", binding)),
                    FeedForwardProgramSlot::Routed(binding) => prepared
                        .target
                        .routed
                        .get(&binding)
                        .cloned()
                        .map(AttestedFeedForward::Routed)
                        .ok_or_else(|| missing("routed_stages", binding)),
                    FeedForwardProgramSlot::GeneralRouted(binding) => prepared
                        .target
                        .general_routed
                        .get(&binding)
                        .cloned()
                        .map(AttestedFeedForward::GeneralRouted)
                        .ok_or_else(|| missing("general_routed_stages", binding)),
                    FeedForwardProgramSlot::Parallel(binding) => prepared
                        .target
                        .parallel
                        .get(&binding)
                        .cloned()
                        .map(AttestedFeedForward::Parallel)
                        .ok_or_else(|| missing("parallel_stages", binding)),
                })
                .transpose()?;
            blocks.push(AttestedTargetBlock {
                mixer,
                feed_forward,
                output_scales,
                per_layer,
            });
        }
        let target = AttestedTarget {
            embedding: slot(
                &prepared.target.embedding,
                target_plan.embedding(),
                "embedding_rows",
            )?,
            blocks,
            readout: prepared
                .target
                .readout
                .get(&target_plan.readout())
                .cloned()
                .ok_or_else(|| missing("qwen_readout_stages", target_plan.readout()))?,
            features: target_plan
                .features()
                .map(|binding| slot(&prepared.target.features, binding, "readout_features_rows"))
                .transpose()?,
            shape: prepared
                .glue
                .shape_rows
                .clone()
                .ok_or_else(|| missing("shape_rows", "fixed"))?,
            sample: prepared
                .glue
                .sample_rows
                .clone()
                .ok_or_else(|| missing("sample_rows", "fixed"))?,
            taps: target_plan
                .taps()
                .map(|taps| {
                    prepared
                        .target
                        .taps
                        .clone()
                        .ok_or_else(|| missing("tap_stages", taps))
                })
                .transpose()?,
            per_layer: target_plan
                .per_layer()
                .map(|binding| {
                    prepared
                        .target
                        .per_layer_entry
                        .get(&binding)
                        .cloned()
                        .ok_or_else(|| missing("per_layer_entry_stages", binding))
                })
                .transpose()?,
        };
        let head = topology
            .head()
            .map(|head_plan| {
                let handles = prepared
                    .head
                    .as_ref()
                    .ok_or_else(|| missing("head", "enabled"))?;
                let mut blocks = Vec::with_capacity(head_plan.blocks().len());
                for &binding in head_plan.blocks() {
                    blocks.push(AttestedHeadBlock {
                        binding,
                        input: slot(&handles.input, binding, "draft_rows")?,
                        attention: handles
                            .attention
                            .get(&binding)
                            .cloned()
                            .ok_or_else(|| missing("attention_stages", binding))?,
                        feed_forward: match binding.feed_forward {
                            FeedForwardProgramSlot::Dense(_) => AttestedFeedForward::Dense(
                                handles
                                    .dense
                                    .get(&binding)
                                    .cloned()
                                    .ok_or_else(|| missing("dense_stages", binding))?,
                            ),
                            FeedForwardProgramSlot::Routed(_) => AttestedFeedForward::Routed(
                                handles
                                    .routed
                                    .get(&binding)
                                    .cloned()
                                    .ok_or_else(|| missing("routed_stages", binding))?,
                            ),
                            // `operators::admit` keeps draft heads on the
                            // fused routed form.
                            FeedForwardProgramSlot::GeneralRouted(_)
                            | FeedForwardProgramSlot::Parallel(_) => {
                                return Err(CatalogFailure::Preparation {
                                    entry: "routed_select",
                                    bindings: format!("{binding:?}"),
                                    outcome: "draft heads run the fused routed form only".into(),
                                })
                            }
                        },
                        features: slot(&handles.features, binding, "readout_features_rows")?,
                        logits: handles
                            .logits
                            .get(&binding)
                            .cloned()
                            .ok_or_else(|| missing("head_logits", binding))?,
                    });
                }
                Ok::<_, CatalogFailure>(AttestedHead {
                    blocks,
                    shape: handles
                        .shape
                        .clone()
                        .ok_or_else(|| missing("shape_rows", "draft"))?,
                    sample: handles
                        .sample
                        .clone()
                        .ok_or_else(|| missing("sample_rows", "draft"))?,
                })
            })
            .transpose()?;
        let draft = topology
            .draft()
            .map(|draft_plan| {
                let handles = prepared
                    .draft
                    .as_ref()
                    .ok_or_else(|| missing("draft", "enabled"))?;
                let attention = |binding| {
                    handles
                        .attention
                        .get(&binding)
                        .cloned()
                        .ok_or_else(|| missing("draft_attention_stages", binding))
                };
                let blocks =
                    draft_plan
                        .blocks()
                        .iter()
                        .map(|block| {
                            Ok(DraftBlockKernels {
                                attention: attention(block.attention)?,
                                injection: attention(block.injection)?,
                                dense: handles.dense.get(&block.feed_forward).cloned().ok_or_else(
                                    || missing("draft_dense_stages", block.feed_forward),
                                )?,
                            })
                        })
                        .collect::<Result<Vec<_>, CatalogFailure>>()?;
                Ok::<_, CatalogFailure>(AttestedDraft {
                    blocks,
                    embedding: handles
                        .embedding
                        .clone()
                        .ok_or_else(|| missing("embedding_rows", draft_plan.embedding()))?,
                    head: handles
                        .head
                        .clone()
                        .ok_or_else(|| missing("readout_head_rows", "draft"))?,
                    shape: handles
                        .shape
                        .clone()
                        .ok_or_else(|| missing("shape_rows", "draft"))?,
                    sample: handles
                        .sample
                        .clone()
                        .ok_or_else(|| missing("sample_rows", "draft"))?,
                    widen: handles
                        .widen
                        .clone()
                        .ok_or_else(|| missing("widen_rows", "draft"))?,
                    markov: draft_plan
                        .markov()
                        .map(|binding| {
                            handles
                                .markov
                                .clone()
                                .ok_or_else(|| missing("draft_markov_stages", binding))
                        })
                        .transpose()?,
                    dflash2: draft_plan
                        .dflash2()
                        .map(|binding| {
                            handles
                                .dflash2
                                .as_ref()
                                .ok_or_else(|| missing("dflash2_stages", binding.selector))?
                                .attest(draft_plan, binding)
                                .map_err(|entry| missing("dflash2_stages", entry))
                        })
                        .transpose()?,
                })
            })
            .transpose()?;
        let vision = topology
            .vision()
            .map(|vision_plan| {
                let handles = prepared
                    .vision
                    .as_ref()
                    .ok_or_else(|| missing("vision", "enabled"))?;
                if let Some(kernel) = vision_plan
                    .kernels()
                    .iter()
                    .find(|kernel| !handles.contains(kernel))
                {
                    return Err(missing("vision", format!("{kernel:?}")));
                }
                Ok::<_, CatalogFailure>(AttestedVision {
                    kernels: handles.clone(),
                })
            })
            .transpose()?;
        let mut copies = Vec::with_capacity(topology.state().copies().len());
        for &element in topology.state().copies() {
            let handle = match element.dtype() {
                Some(DType::F32) => &prepared.glue.copy_rows_f32,
                Some(DType::F16) => &prepared.glue.copy_rows_f16,
                Some(DType::BF16) => &prepared.glue.copy_rows_bf16,
                Some(DType::U32) => &prepared.glue.copy_rows_u32,
                _ => return Err(missing("copy_rows", element)),
            };
            copies.push((
                element,
                handle
                    .clone()
                    .ok_or_else(|| missing("copy_rows", element))?,
            ));
        }
        let state = AttestedState {
            copies,
            conditioning: Some(
                prepared
                    .glue
                    .conditioning_overlay
                    .clone()
                    .ok_or_else(|| missing("conditioning_overlay", "target conditioning"))?,
            ),
        };
        let mut imports = Vec::with_capacity(topology.imports().len());
        for &binding in topology.imports() {
            let handle = match binding {
                ImportProgramSlot::Dense { source, resident } => AttestedImport::Dense(slot(
                    &prepared.import.import_dense,
                    (source, resident),
                    "import_dense",
                )?),
                ImportProgramSlot::Repack { source, resident } => AttestedImport::Repack(slot(
                    &prepared.import.repack_weight,
                    (source, resident),
                    "repack_weight",
                )?),
            };
            imports.push((binding, handle));
        }
        let invocation_workspace_bytes = Self::sum_invocation_workspace_bytes(&prepared)?;
        let planned_bytes = Self::planned_invocation_workspace_bytes(topology, device.backend())
            .map_err(|error| CatalogFailure::Preparation {
                entry: "program_factory",
                bindings: "native invocation workspace".into(),
                outcome: error.to_string(),
            })?;
        if invocation_workspace_bytes != planned_bytes {
            return Err(CatalogFailure::Qualification {
                entry: "program_factory",
                bindings: "native invocation workspace".into(),
                outcome: format!(
                    "prepared {invocation_workspace_bytes} bytes but the ordered program plan charged {planned_bytes}"
                ),
            });
        }
        let attested = Self {
            backend: device.backend(),
            tuned,
            owner: prepared.owner.clone(),
            report,
            target,
            head,
            draft,
            vision,
            state,
            imports,
            invocation_workspace_bytes,
            target_graphs: None,
            target_readout_graphs: None,
            drafter_graphs: None,
            vision_graphs: None,
            state_graphs: None,
        };
        QualificationView::new(
            &attested,
            topology,
            &definition.decoder,
            definition.vision.as_ref(),
            load,
        )
        .qualify(device)?;
        Ok(attested)
    }

    fn sum_invocation_workspace_bytes(
        prepared: &NativePreparationCache,
    ) -> Result<u64, CatalogFailure> {
        let mut bytes = u128::from(prepared.owner.storage_bytes());
        macro_rules! charge {
            ($iter:expr) => {
                for handle in $iter {
                    bytes += u128::from(handle.invocation_workspace_bytes());
                }
            };
        }
        charge!(prepared.import.import_dense.values());
        charge!(prepared.import.repack_weight.values());
        charge!(prepared.target.embedding.values());
        for handles in prepared.target.attention.values() {
            bytes += u128::from(handles.project.invocation_workspace_bytes())
                + u128::from(handles.history.invocation_workspace_bytes())
                + u128::from(handles.output.invocation_workspace_bytes());
        }
        for handles in prepared.target.recurrent.values() {
            bytes += u128::from(handles.project.invocation_workspace_bytes())
                + recurrent_step_bytes(&handles.step)
                + u128::from(handles.chunk.invocation_workspace_bytes())
                + u128::from(handles.output.invocation_workspace_bytes());
        }
        for handles in prepared.target.state_space.values() {
            bytes += u128::from(handles.project.invocation_workspace_bytes())
                + u128::from(handles.step.invocation_workspace_bytes())
                + u128::from(handles.chunk.invocation_workspace_bytes())
                + u128::from(handles.gate.invocation_workspace_bytes())
                + u128::from(handles.output.invocation_workspace_bytes());
        }
        for handles in prepared.target.short_conv.values() {
            bytes += u128::from(handles.project.invocation_workspace_bytes())
                + u128::from(handles.rows.invocation_workspace_bytes())
                + u128::from(handles.output.invocation_workspace_bytes());
        }
        for handles in prepared.target.dense.values() {
            bytes += u128::from(handles.expand.invocation_workspace_bytes())
                + u128::from(handles.output.invocation_workspace_bytes());
        }
        let general_routed = prepared.target.general_routed.values().chain(
            prepared
                .target
                .parallel
                .values()
                .map(|handles| &handles.routed),
        );
        for handles in prepared.target.parallel.values() {
            bytes += u128::from(handles.expand.invocation_workspace_bytes())
                + u128::from(handles.down.invocation_workspace_bytes())
                + u128::from(handles.tail.invocation_workspace_bytes());
        }
        for handles in prepared.target.per_layer.values() {
            bytes += u128::from(handles.gate.invocation_workspace_bytes())
                + u128::from(handles.output.project.invocation_workspace_bytes())
                + u128::from(handles.output.residual.invocation_workspace_bytes());
        }
        for entry in prepared.target.per_layer_entry.values() {
            bytes += u128::from(entry.round.invocation_workspace_bytes())
                + u128::from(entry.project.invocation_workspace_bytes())
                + match &entry.table {
                    TableConversion::Dense(kernel) => {
                        u128::from(kernel.invocation_workspace_bytes())
                    }
                    TableConversion::Repack(kernel) => {
                        u128::from(kernel.invocation_workspace_bytes())
                    }
                }
                + u128::from(entry.inputs.invocation_workspace_bytes())
                + u128::from(entry.copy.invocation_workspace_bytes());
        }
        for handles in general_routed {
            bytes += u128::from(handles.select.invocation_workspace_bytes())
                + u128::from(handles.down.invocation_workspace_bytes())
                + u128::from(handles.group.invocation_workspace_bytes())
                + u128::from(handles.scatter.invocation_workspace_bytes())
                + match &handles.experts {
                    ExpertKernels::Gated { decode, grouped } => {
                        u128::from(decode.invocation_workspace_bytes())
                            + u128::from(grouped.invocation_workspace_bytes())
                    }
                    ExpertKernels::Plain { decode, grouped } => {
                        u128::from(decode.invocation_workspace_bytes())
                            + u128::from(grouped.invocation_workspace_bytes())
                    }
                };
            if let Some((expansion, output)) = &handles.shared {
                bytes += u128::from(output.invocation_workspace_bytes())
                    + match expansion {
                        DenseExpansionKernel::Gated(kernel) => {
                            u128::from(kernel.invocation_workspace_bytes())
                        }
                        DenseExpansionKernel::Plain(kernel) => {
                            u128::from(kernel.invocation_workspace_bytes())
                        }
                    };
            }
            if let Some((down, up)) = &handles.latent {
                bytes += u128::from(down.invocation_workspace_bytes())
                    + u128::from(up.invocation_workspace_bytes());
            }
        }
        for handles in prepared.target.routed.values() {
            bytes += u128::from(handles.route.invocation_workspace_bytes())
                + routed_decode_bytes(&handles.decode)
                + u128::from(handles.output.invocation_workspace_bytes())
                + u128::from(handles.group.invocation_workspace_bytes())
                + u128::from(handles.experts.invocation_workspace_bytes())
                + u128::from(handles.combine.invocation_workspace_bytes());
        }
        for handles in prepared.target.readout.values() {
            bytes += u128::from(handles.features.invocation_workspace_bytes())
                + match &handles.head {
                    ReadoutHeadKernels::Packed { head, selected } => {
                        u128::from(head.invocation_workspace_bytes())
                            + u128::from(selected.invocation_workspace_bytes())
                    }
                    ReadoutHeadKernels::Progressive(kernels) => {
                        u128::from(kernels.top.invocation_workspace_bytes())
                            + u128::from(kernels.refine.invocation_workspace_bytes())
                            + u128::from(kernels.exact.invocation_workspace_bytes())
                            + u128::from(kernels.planes.invocation_workspace_bytes())
                    }
                };
        }
        charge!(prepared.target.features.values());
        if let Some(taps) = &prepared.target.taps {
            bytes += u128::from(taps.tap.invocation_workspace_bytes())
                + u128::from(taps.fusion.invocation_workspace_bytes())
                + u128::from(taps.features.invocation_workspace_bytes());
        }
        if let Some(head) = &prepared.head {
            charge!(head.input.values());
            for handles in head.attention.values() {
                bytes += u128::from(handles.project.invocation_workspace_bytes())
                    + u128::from(handles.history.invocation_workspace_bytes())
                    + u128::from(handles.output.invocation_workspace_bytes());
            }
            for handles in head.dense.values() {
                bytes += u128::from(handles.expand.invocation_workspace_bytes())
                    + u128::from(handles.output.invocation_workspace_bytes());
            }
            for handles in head.routed.values() {
                bytes += u128::from(handles.route.invocation_workspace_bytes())
                    + routed_decode_bytes(&handles.decode)
                    + u128::from(handles.output.invocation_workspace_bytes())
                    + u128::from(handles.group.invocation_workspace_bytes())
                    + u128::from(handles.experts.invocation_workspace_bytes())
                    + u128::from(handles.combine.invocation_workspace_bytes());
            }
            charge!(head.features.values());
            for logits in head.logits.values() {
                match logits {
                    HeadLogitsKernels::Packed(kernel) => charge!([kernel]),
                    HeadLogitsKernels::Progressive(kernels) => {
                        charge!([&kernels.top]);
                        charge!([&kernels.refine]);
                        charge!([&kernels.exact]);
                        charge!([&kernels.planes]);
                    }
                }
            }
            charge!(head.shape.iter());
            charge!(head.sample.iter());
        }
        if let Some(draft) = &prepared.draft {
            for handles in draft.attention.values() {
                bytes += u128::from(handles.project.invocation_workspace_bytes())
                    + u128::from(handles.history.invocation_workspace_bytes())
                    + u128::from(handles.output.invocation_workspace_bytes());
            }
            for handles in draft.dense.values() {
                bytes += u128::from(handles.expand.invocation_workspace_bytes())
                    + u128::from(handles.output.invocation_workspace_bytes());
            }
            charge!(draft.embedding.iter());
            charge!(draft.head.iter());
            charge!(draft.shape.iter());
            charge!(draft.sample.iter());
            charge!(draft.widen.iter());
            if let Some(markov) = &draft.markov {
                bytes += u128::from(markov.embedding.invocation_workspace_bytes())
                    + u128::from(markov.projection.invocation_workspace_bytes())
                    + u128::from(markov.features.invocation_workspace_bytes())
                    + u128::from(markov.confidence.invocation_workspace_bytes());
            }
            if let Some(dflash2) = &draft.dflash2 {
                charge!(dflash2.norms.values());
                charge!(dflash2.projections.values());
                bytes += u128::from(dflash2.convolve_input.invocation_workspace_bytes())
                    + u128::from(dflash2.convolve_residual.invocation_workspace_bytes())
                    + u128::from(dflash2.gated.invocation_workspace_bytes())
                    + u128::from(dflash2.top_k.invocation_workspace_bytes())
                    + u128::from(dflash2.predecessor.invocation_workspace_bytes())
                    + u128::from(dflash2.successor.invocation_workspace_bytes())
                    + u128::from(dflash2.path.invocation_workspace_bytes());
            }
        }
        if let Some(vision) = &prepared.vision {
            charge!(vision.patch_stem.values());
            charge!(vision.norm.values());
            charge!(vision.linear.values());
            charge!(vision.clamp.values());
            charge!(vision.attention.values());
            charge!(vision.pool.values());
            charge!(vision.position.values());
            charge!(vision.post_norm.values());
        }
        if let Some(handle) = &prepared.glue.shape_rows {
            bytes += u128::from(handle.invocation_workspace_bytes());
        }
        if let Some(handle) = &prepared.glue.sample_rows {
            bytes += u128::from(handle.invocation_workspace_bytes());
        }
        if let Some(handle) = &prepared.glue.conditioning_overlay {
            bytes += u128::from(handle.invocation_workspace_bytes());
        }
        for handle in [
            prepared.glue.copy_rows_f32.as_ref(),
            prepared.glue.copy_rows_f16.as_ref(),
            prepared.glue.copy_rows_bf16.as_ref(),
            prepared.glue.copy_rows_u32.as_ref(),
        ]
        .into_iter()
        .flatten()
        {
            bytes += u128::from(handle.invocation_workspace_bytes());
        }
        u64::try_from(bytes).map_err(|_| CatalogFailure::Preparation {
            entry: "program_factory",
            bindings: "native invocation workspace".into(),
            outcome: "prepared invocation workspace bytes overflow".into(),
        })
    }

    /// Startup-only allowance for qualification. Entries are specialized to
    /// the model's geometry, so fixtures run at it: the weight fixtures of one
    /// qualification hold at most one weight scope's tensors at their
    /// resident representation (one block, or the target's norm and output),
    /// and fixture locals drop before the next binding. Two fixtures differ
    /// from a scope's planned weights: the embedding table is qualified on a
    /// single row, and each head block also holds its draft projection, the
    /// target output restricted to the draft vocabulary. Sixteen MiB more
    /// covers one-row activations and tables, logits rows, checked results
    /// and allocator alignment. Prepared invocation buffers are charged
    /// separately.
    pub fn qualification_peak_bytes(load: &crate::ModelLoadPlan) -> Result<u64, String> {
        use magnitude_family_contracts::{WeightKind, WeightScope};
        const FIXTURES: u64 = 16 * 1024 * 1024;
        /// The fixture set a weight belongs to: one per block (every
        /// sublayer of it), head block, or other weight scope.
        #[derive(Clone, Copy, PartialEq, Eq, Hash)]
        enum FixtureScope {
            Block(u32),
            Head(u32),
            Other(WeightScope),
        }
        let fixture_scope = |scope| match scope {
            WeightScope::TargetSublayer(index)
            | WeightScope::TargetBranch {
                sublayer: index, ..
            } => FixtureScope::Block(index.block),
            WeightScope::HeadBlock(block) => FixtureScope::Head(block),
            WeightScope::HeadSublayer(index)
            | WeightScope::HeadBranch {
                sublayer: index, ..
            } => FixtureScope::Head(index.block),
            other => FixtureScope::Other(other),
        };
        let mut scopes = HashMap::<FixtureScope, u64>::new();
        for weight in load.weights() {
            if weight.role.kind == WeightKind::Embedding {
                continue;
            }
            let scope = scopes.entry(fixture_scope(weight.role.scope)).or_default();
            *scope = scope
                .checked_add(weight.resident_bytes)
                .ok_or("qualification scope byte count overflows")?;
        }
        if load.head().is_some() {
            // The draft projection fixture: the output projection's (or its
            // planes') draft-vocabulary rows.
            let target = |kind| {
                load.target().iter().find(|weight| {
                    weight.role.scope == WeightScope::Target && weight.role.kind == kind
                })
            };
            let fixture = |element: Element, shape: &[u64]| {
                element
                    .canonical_byte_len(shape)
                    .map_err(|error| format!("draft projection fixture: {error}"))
            };
            let projection = match target(WeightKind::Output) {
                Some(output) => {
                    let [vocabulary, hidden] = output.shape[..] else {
                        return Err("target output is not a matrix".into());
                    };
                    fixture(output.resident, &[draft_vocabulary(vocabulary), hidden])?
                }
                None => {
                    let top = target(WeightKind::OutputPlane(ProgressivePlane::Top))
                        .ok_or("head qualification requires the target output")?;
                    let [vocabulary, words] = top.shape[..] else {
                        return Err("target top plane is not a matrix".into());
                    };
                    ProgressivePlane::ALL.into_iter().try_fold(0u64, |bytes, plane| {
                        let element = crate::progressive::element(plane);
                        let plane_bytes =
                            fixture(element, &plane.shape(draft_vocabulary(vocabulary), words * 8))?;
                        bytes
                            .checked_add(plane_bytes)
                            .ok_or_else(|| "draft projection fixture overflows".to_owned())
                    })?
                }
            };
            for (scope, bytes) in &mut scopes {
                if matches!(scope, FixtureScope::Head(_)) {
                    *bytes = bytes
                        .checked_add(projection)
                        .ok_or("qualification scope byte count overflows")?;
                }
            }
        }
        FIXTURES
            .checked_add(scopes.into_values().max().unwrap_or(0))
            .ok_or_else(|| "qualification peak byte count overflows".into())
    }
    pub(crate) fn bind_target(
        &self,
        model: crate::ResidentTarget,
        geometry: magnitude_family_contracts::Decoder,
    ) -> Result<crate::programs::native_target::NativeTargetProgram, CatalogError> {
        (|| -> Result<crate::programs::native_target::NativeTargetProgram, CatalogFailure> {
            if model.blocks != self.target.blocks.len() || model.blocks != geometry.blocks.len() {
                return Err(missing("target", "resident topology"));
            }
            let graphs = self
                .target_graphs
                .as_ref()
                .ok_or_else(|| missing("target", "prepared graphs"))?
                .bind_weights(&model)
                .map_err(|error| CatalogFailure::Preparation {
                    entry: "target_graph",
                    bindings: "resident decoder weights".into(),
                    outcome: error,
                })?;
            let readout_graphs = self
                .target_readout_graphs
                .as_ref()
                .ok_or_else(|| missing("target", "prepared readout graphs"))?
                .clone()
                .bind_weights(&model)
                .map_err(|error| CatalogFailure::Preparation {
                    entry: "target_readout_graph",
                    bindings: "resident readout weights".into(),
                    outcome: error,
                })?;
            crate::programs::native_target::NativeTargetProgram::new(
                model.output_norm.tensor().device(),
                self.state.clone(),
                geometry,
                graphs,
                readout_graphs,
            )
            .map_err(|error| CatalogFailure::Preparation {
                entry: "target_graph_controls",
                bindings: "sealed attention rotary controls".into(),
                outcome: error.to_string(),
            })
        })()
        .map_err(|failure| CatalogError::native(self.backend, failure))
    }
    pub(crate) fn bind_head(
        &self,
        resident: crate::ResidentHead,
        definition: &magnitude_family_contracts::ModelDefinition,
    ) -> Result<crate::programs::native_drafter::NativeDrafterProgram, CatalogError> {
        use crate::programs::native_drafter::NativeDrafterProgram;
        (|| -> Result<NativeDrafterProgram, CatalogFailure> {
            match self
                .drafter_graphs
                .as_ref()
                .ok_or_else(|| missing("head", "prepared native graphs"))?
            {
                crate::PreparedDrafterGraphs::Head(graphs) => {
                    let head = self
                        .head
                        .as_ref()
                        .ok_or_else(|| missing("head", "disabled"))?;
                    if head.blocks.len() != 1
                        || resident.depth != 1
                        || definition
                            .head
                            .as_ref()
                            .is_none_or(|description| description.depth() != 1)
                    {
                        return Err(missing("head", "native single-block topology"));
                    }
                    let graphs = graphs
                        .bind_weights(&resident)
                        .map_err(|error| missing("head", error))?;
                    crate::programs::native_head::NativeHeadProgram::new(
                        definition.decoder.clone(),
                        graphs,
                    )
                    .map(NativeDrafterProgram::Head)
                    .map_err(|error| missing("head", error))
                }
                crate::PreparedDrafterGraphs::Draft(graphs) => {
                    let blocks = definition
                        .draft
                        .as_ref()
                        .ok_or_else(|| missing("draft", "definition"))?
                        .blocks
                        .len();
                    if resident.depth != blocks {
                        return Err(missing("draft", "resident layers"));
                    }
                    let graphs = graphs
                        .bind_weights(&resident)
                        .map_err(|error| missing("draft", error))?;
                    crate::programs::native_draft::NativeDraftProgram::new(
                        definition.clone(),
                        graphs,
                    )
                    .map(NativeDrafterProgram::Draft)
                    .map_err(|error| missing("draft", error))
                }
            }
        })()
        .map_err(|failure| CatalogError::native(self.backend, failure))
    }
    pub(crate) fn bind_vision(
        &self,
        resident: crate::ResidentVision,
        definition: &magnitude_family_contracts::ModelDefinition,
    ) -> Result<crate::programs::native_vision::NativeVisionProgram, CatalogError> {
        (|| -> Result<crate::programs::native_vision::NativeVisionProgram, CatalogFailure> {
            self.vision
                .as_ref()
                .ok_or_else(|| missing("vision", "disabled"))?;
            let description = definition
                .vision
                .as_ref()
                .ok_or_else(|| missing("vision", "description"))?;
            if let Some((role, _)) = description
                .weights()
                .into_iter()
                .find(|(role, _)| resident.weights.get(*role).is_err())
            {
                return Err(missing("vision", format!("resident {role:?}")));
            }
            let graphs = self
                .vision_graphs
                .as_ref()
                .ok_or_else(|| missing("vision", "prepared native graphs"))?
                .clone()
                .bind_weights(&resident)
                .map_err(|error| missing("vision", error))?;
            Ok(crate::programs::native_vision::NativeVisionProgram::new(
                graphs,
            ))
        })()
        .map_err(|failure| CatalogError::native(self.backend, failure))
    }
    pub(crate) fn bind_state(&self) -> crate::programs::native_state::NativeStateProgram {
        crate::programs::native_state::NativeStateProgram::new(
            self.state_graphs
                .as_ref()
                .expect("state graphs prepared before binding")
                .clone(),
        )
    }
    /// Bind one already planned semantic weight to its exact import entry.
    /// This happens during loader construction, never during a numerical call.
    pub(crate) fn bind_import(
        &self,
        weight: &crate::WeightPlan,
    ) -> Result<crate::programs::native_import::NativeImportProgram, CatalogError> {
        (|| -> Result<crate::programs::native_import::NativeImportProgram, CatalogFailure> {
            let requested = if let (Some(source), Some(resident)) =
                (weight.upload.dtype(), weight.resident.dtype())
            {
                ImportProgramSlot::Dense { source, resident }
            } else {
                ImportProgramSlot::Repack {
                    source: weight.upload,
                    resident: weight.resident,
                }
            };
            let (_, handle) = self
                .imports
                .iter()
                .find(|(slot, _)| *slot == requested)
                .ok_or_else(|| missing("weight_import", requested))?;
            Ok(crate::programs::native_import::NativeImportProgram::new(
                weight.storage_identity(),
                handle.clone(),
            ))
        })()
        .map_err(|failure| CatalogError::native(self.backend, failure))
    }
    pub fn invocation_workspace_bytes(&self) -> u64 {
        self.invocation_workspace_bytes
    }
    pub(crate) fn device_storage_bytes(&self) -> Result<u64, &'static str> {
        let mut seen = HashSet::new();
        let mut bytes = self.owner.storage_bytes();
        macro_rules! charge {
            ($handle:expr) => {{
                let handle = $handle;
                if seen.insert(handle.prepared_storage_identity()) {
                    bytes = bytes
                        .checked_add(handle.invocation_workspace_bytes())
                        .ok_or("prepared program charge overflows")?;
                }
            }};
        }
        macro_rules! tail {
            ($output:expr) => {{
                match $output {
                    SublayerOutput::Residual(output) => charge!(output),
                    SublayerOutput::PostNorm(kernels) => {
                        charge!(&kernels.project);
                        charge!(&kernels.residual);
                    }
                }
            }};
        }
        macro_rules! attention {
            ($handles:expr) => {{
                let handles = $handles;
                charge!(&handles.project);
                match &handles.history {
                    super::target::AttentionHistoryKernels::Dense {
                        decode,
                        verify,
                        prefill,
                    } => {
                        charge!(decode);
                        if let Some(verify) = verify {
                            charge!(verify);
                        }
                        charge!(prefill);
                    }
                    super::target::AttentionHistoryKernels::AffineK8V4 {
                        decode,
                        verify,
                        verify_four,
                        verify_eight,
                        prefill,
                        prefill_listed,
                    } => {
                        charge!(decode);
                        if let Some(verify) = verify {
                            charge!(verify);
                        }
                        if let Some(verify_four) = verify_four {
                            charge!(verify_four);
                        }
                        if let Some(verify_eight) = verify_eight {
                            charge!(verify_eight);
                        }
                        charge!(prefill);
                        if let Some(prefill_listed) = prefill_listed {
                            charge!(prefill_listed);
                        }
                    }
                }
                tail!(&handles.output);
            }};
        }
        macro_rules! dense {
            ($handles:expr) => {{
                let handles = $handles;
                charge!(&handles.expand);
                tail!(&handles.output);
            }};
        }
        macro_rules! general_routed {
            ($handles:expr) => {{
                let handles = $handles;
                charge!(&handles.select);
                match &handles.experts {
                    ExpertKernels::Gated { decode, grouped } => {
                        charge!(decode);
                        charge!(grouped);
                    }
                    ExpertKernels::Plain { decode, grouped } => {
                        charge!(decode);
                        charge!(grouped);
                    }
                }
                charge!(&handles.down);
                charge!(&handles.group);
                charge!(&handles.scatter);
                if let Some((expansion, output)) = &handles.shared {
                    match expansion {
                        DenseExpansionKernel::Gated(kernel) => charge!(kernel),
                        DenseExpansionKernel::Plain(kernel) => charge!(kernel),
                    }
                    charge!(output);
                }
                if let Some((down, up)) = &handles.latent {
                    charge!(down);
                    charge!(up);
                }
            }};
        }
        macro_rules! feed_forward {
            ($handles:expr) => {{
                match $handles {
                    AttestedFeedForward::Dense(handles) => dense!(handles),
                    AttestedFeedForward::Routed(handles) => {
                        charge!(&handles.route);
                        match &handles.decode {
                            RoutedDecodeKernels::Expand(expand) => charge!(expand),
                            RoutedDecodeKernels::SharedRoute { route, choices } => {
                                charge!(route);
                                charge!(choices);
                            }
                        }
                        charge!(&handles.output);
                        charge!(&handles.group);
                        charge!(&handles.experts);
                        charge!(&handles.combine);
                    }
                    AttestedFeedForward::GeneralRouted(handles) => general_routed!(handles),
                    AttestedFeedForward::Parallel(handles) => {
                        charge!(&handles.expand);
                        charge!(&handles.down);
                        general_routed!(&handles.routed);
                        charge!(&handles.tail);
                    }
                }
            }};
        }
        charge!(&self.target.embedding);
        for block in &self.target.blocks {
            match &block.mixer {
                AttestedMixer::Attention(handles) => attention!(handles),
                AttestedMixer::Recurrent(handles) => {
                    charge!(&handles.project);
                    match &handles.step {
                        RecurrentStepKernels::Step(step) => charge!(step),
                        RecurrentStepKernels::Convolved { project, step } => {
                            charge!(project);
                            charge!(step);
                        }
                    }
                    charge!(&handles.chunk);
                    charge!(&handles.output);
                }
                AttestedMixer::StateSpace(handles) => {
                    charge!(&handles.project);
                    charge!(&handles.step);
                    charge!(&handles.chunk);
                    charge!(&handles.gate);
                    charge!(&handles.output);
                }
                AttestedMixer::ShortConv(handles) => {
                    charge!(&handles.project);
                    charge!(&handles.rows);
                    charge!(&handles.output);
                }
            }
            if let Some(feed_forward) = &block.feed_forward {
                feed_forward!(feed_forward);
            }
            if let Some(per_layer) = &block.per_layer {
                charge!(&per_layer.gate);
                charge!(&per_layer.output.project);
                charge!(&per_layer.output.residual);
            }
        }
        if let Some(entry) = &self.target.per_layer {
            charge!(&entry.round);
            charge!(&entry.project);
            match &entry.table {
                TableConversion::Dense(kernel) => charge!(kernel),
                TableConversion::Repack(kernel) => charge!(kernel),
            }
            charge!(&entry.inputs);
            charge!(&entry.copy);
        }
        charge!(&self.target.readout.features);
        match &self.target.readout.head {
            ReadoutHeadKernels::Packed { head, selected } => {
                charge!(head);
                charge!(selected);
            }
            ReadoutHeadKernels::Progressive(kernels) => {
                charge!(&kernels.top);
                charge!(&kernels.refine);
                charge!(&kernels.exact);
                charge!(&kernels.planes);
            }
        }
        if let Some(features) = &self.target.features {
            charge!(features);
        }
        if let Some(taps) = &self.target.taps {
            charge!(&taps.tap);
            charge!(&taps.fusion);
            charge!(&taps.features);
        }
        charge!(&self.target.shape);
        charge!(&self.target.sample);
        if let Some(head) = &self.head {
            for block in &head.blocks {
                charge!(&block.input);
                attention!(&block.attention);
                feed_forward!(&block.feed_forward);
                charge!(&block.features);
                match &block.logits {
                    HeadLogitsKernels::Packed(kernel) => charge!(kernel),
                    HeadLogitsKernels::Progressive(kernels) => {
                        charge!(&kernels.top);
                        charge!(&kernels.refine);
                        charge!(&kernels.exact);
                        charge!(&kernels.planes);
                    }
                }
            }
            charge!(&head.shape);
            charge!(&head.sample);
        }
        if let Some(draft) = &self.draft {
            for block in &draft.blocks {
                attention!(&block.attention);
                attention!(&block.injection);
                dense!(&block.dense);
            }
            charge!(&draft.embedding);
            charge!(&draft.head);
            charge!(&draft.shape);
            charge!(&draft.sample);
            charge!(&draft.widen);
            if let Some(markov) = &draft.markov {
                charge!(&markov.embedding);
                charge!(&markov.projection);
                charge!(&markov.features);
                charge!(&markov.confidence);
            }
            if let Some(dflash2) = &draft.dflash2 {
                for layer in &dflash2.layers {
                    charge!(&layer.attention_norm);
                    charge!(&layer.attention_coefficients);
                    charge!(&layer.query);
                    charge!(&layer.key);
                    charge!(&layer.value);
                    charge!(&layer.output);
                    charge!(&layer.feed_forward_norm);
                    charge!(&layer.feed_forward_coefficients);
                    charge!(&layer.gate);
                    charge!(&layer.up);
                    charge!(&layer.down);
                }
                charge!(&dflash2.features);
                charge!(&dflash2.hidden);
                charge!(&dflash2.convolve_input);
                charge!(&dflash2.convolve_residual);
                charge!(&dflash2.gated);
                charge!(&dflash2.top_k);
                charge!(&dflash2.predecessor);
                charge!(&dflash2.successor);
                charge!(&dflash2.path);
            }
        }
        if let Some(vision) = &self.vision {
            let kernels = &vision.kernels;
            for kernel in kernels.patch_stem.values() {
                charge!(kernel);
            }
            for kernel in kernels.norm.values() {
                charge!(kernel);
            }
            for kernel in kernels.linear.values() {
                charge!(kernel);
            }
            for kernel in kernels.clamp.values() {
                charge!(kernel);
            }
            for kernel in kernels.attention.values() {
                charge!(kernel);
            }
            for kernel in kernels.pool.values() {
                charge!(kernel);
            }
            for kernel in kernels.position.values() {
                charge!(kernel);
            }
            for kernel in kernels.post_norm.values() {
                charge!(kernel);
            }
        }
        for (_, copy) in &self.state.copies {
            charge!(copy);
        }
        if let Some(conditioning) = &self.state.conditioning {
            charge!(conditioning);
        }
        for (_, import) in &self.imports {
            match import {
                AttestedImport::Dense(handle) => charge!(handle),
                AttestedImport::Repack(handle) => charge!(handle),
            }
        }
        Ok(bytes)
    }
    pub fn qualification(&self) -> &QualificationReport {
        &self.report
    }
    pub fn belongs_to(&self, device: &Device) -> bool {
        self.owner.belongs_to(device)
    }
    pub const fn path(&self) -> ExecutionPath {
        ExecutionPath::Native
    }
    /// The backend of the device these programs were prepared for.
    pub const fn backend(&self) -> BackendName {
        self.backend
    }
    /// Every entry tuned while these programs were prepared.
    pub fn tuned(&self) -> &[TunedEntry] {
        &self.tuned
    }
}
