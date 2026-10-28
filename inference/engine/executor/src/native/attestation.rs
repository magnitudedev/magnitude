//! Ordered callable native slots constructed from the one ProgramPlan.
//! Preparation is the only place a checked binding lookup is permitted.

use super::preparation::PreparationInputs;
use super::tuning::{TunedEntry, TuningContext, TuningLimits};
use super::*;
use crate::SublayerTail;
use crate::programs::native_target_graph::TARGET_TABLE;
use crate::{
    ExecutionPlanDraft, FeedForwardProgramSlot, HeadBinding, ImportProgramSlot, MixerProgramSlot,
    PlanError, PlannedDevice, ProgramPlan, ReadoutHead,
};
use magnitude_family_contracts::{ProgressivePlane, SublayerIndex, WeightScope};
use magnitude_kernels::{
    cluster_token_rows, draft_top_k, feature_rows, import_dense, packing_bound, place_logits_rows,
    project_rows, readout_project_rows, repack_weight, selection_notice, tap_rows, widen_rows,
};
use std::rc::Rc;

pub struct AttestedPrograms {
    backend: BackendName,
    tuned: Vec<TunedEntry>,
    unprepared: usize,
    readout_choices: Vec<crate::ReadoutClassChoice>,
    pub(super) owner: Tensor,
    report: QualificationReport,
    pub(super) target: AttestedTarget,
    pub(super) head: Option<AttestedHead>,
    pub(super) shared_head: Option<AttestedSharedHead>,
    pub(super) draft: Option<AttestedDraft>,
    pub(super) vision: Option<AttestedVision>,
    pub(super) state: AttestedState,
    pub(super) imports: Vec<(ImportProgramSlot, AttestedImport)>,
    /// The copy of a read's selected rows for the host
    /// (`ProgramPlan::notices_selection`).
    pub(super) notice: Option<NativeKernel<selection_notice::Entry>>,
    target_graphs: Option<std::sync::Arc<crate::PreparedTargetGraphs>>,
    target_readout_graphs: Option<std::sync::Arc<crate::PreparedTargetReadoutGraphs>>,
    drafter_graphs: Option<crate::PreparedDrafterGraphs>,
    vision_graphs: Option<Rc<crate::programs::native_vision::PreparedVisionGraphs>>,
    state_graphs: Option<Rc<crate::programs::native_state::PreparedStateCopyGraphs>>,
}

#[derive(Clone)]
pub(crate) struct AttestedTarget {
    pub residual_features: Option<NativeKernel<feature_rows::Entry>>,
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
    /// The decoder's hyper-connection streams' entries, when it has them.
    pub streams: Option<StreamsKernels>,
    /// The n-gram input's entries, when the block has one.
    pub ngram: Option<NGramKernels>,
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

/// The tails of a block's mixer, feed-forward and per-layer sublayers that
/// decide their output scales (`OutputScales`); `None` for a sublayer whose
/// scale is 1 whatever the artifact holds.
pub(crate) fn output_tails(block: crate::TargetBlockProgramSlot) -> [Option<SublayerTail>; 3] {
    [
        match block.mixer() {
            MixerProgramSlot::Attention(binding) => Some(binding.tail),
            _ => None,
        },
        match block.feed_forward() {
            Some(FeedForwardProgramSlot::Dense(binding)) => Some(binding.tail),
            Some(FeedForwardProgramSlot::Parallel(binding)) => Some(SublayerTail::PostNorm {
                norm: binding.norm,
                scaled: binding.scaled,
            }),
            _ => None,
        },
        block.per_layer().map(|binding| binding.tail),
    ]
}

/// Whether a sublayer of `tail` scales its result by its layer's output
/// scale, a value of the artifact (`output_scale`); otherwise by 1.
pub(crate) fn reads_layer_scale(tail: Option<SublayerTail>) -> bool {
    matches!(tail, Some(SublayerTail::PostNorm { scaled: true, .. }))
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

/// The output scale of the sublayer whose weights `scope` names, with
/// `tail`.
fn output_scale(
    load: &crate::ModelLoadPlan,
    tuning: TuningContext<'_>,
    scope: WeightScope,
    tail: Option<SublayerTail>,
) -> Result<f32, CatalogFailure> {
    let failure = |outcome: String| CatalogFailure::Preparation {
        entry: "post_norm_residual",
        bindings: format!("layer scale of {scope:?}"),
        outcome,
    };
    if !reads_layer_scale(tail) {
        return Ok(1.0);
    }
    let role = magnitude_family_contracts::WeightRole {
        scope,
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
    pub streams: Option<super::StreamsKernels>,
    pub copy: NativeKernel<feature_copy::Entry>,
    pub gather: NativeKernel<feature_rows::Entry>,
    pub binding: HeadBinding,
    pub input: NativeKernel<draft_rows::Entry>,
    pub attention: AttentionKernels,
    pub priming_project: NativeKernel<attention_project::Entry>,
    pub feed_forward: AttestedFeedForward,
    pub features: NativeKernel<readout_features_rows::Entry>,
    pub logits: ProjectionKernels,
}

/// The entries of a draft head reading target history, in pass order: the
/// target's embedding rows, the join of the embedding and hidden halves, the
/// projection into the head's width, its decoder blocks, its normalized
/// output, its own vocabulary table, token selection (the target's, over the
/// whole vocabulary) and the projection back to the target's width.
#[derive(Clone)]
pub(crate) struct AttestedSharedHead {
    pub embedding: NativeKernel<embedding_rows::Entry>,
    pub join: NativeKernel<tap_rows::Entry>,
    pub input: NativeKernel<project_rows::Entry>,
    pub blocks: Vec<AttestedTargetBlock>,
    pub features: NativeKernel<readout_features_rows::Entry>,
    pub projection: ProjectionKernels,
    pub centroids: Option<NativeKernel<readout_project_rows::Entry>>,
    pub cluster_selection: Option<NativeKernel<draft_top_k::Entry>>,
    pub cluster_rows: Option<NativeKernel<cluster_token_rows::Entry>>,
    pub placement: Option<NativeKernel<place_logits_rows::Entry>>,
    pub chain: NativeKernel<project_rows::Entry>,
    pub shape: NativeKernel<shape_rows::Entry>,
    pub sample: NativeKernel<sample_rows::Entry>,
}

/// A separate draft's entries: per layer in draft order, then the block
/// embedding, the proposing rows' norm and projection, and DSpark's chain.
/// Selection is the target's.
#[derive(Clone)]
pub(crate) struct AttestedDraft {
    pub blocks: Vec<DraftBlockKernels>,
    pub embedding: NativeKernel<embedding_rows::Entry>,
    pub head: ProjectionKernels,
    /// Token selection over the readout vocabulary.
    pub shape: NativeKernel<shape_rows::Entry>,
    pub sample: NativeKernel<sample_rows::Entry>,
    /// Widens a device-conditioned entry's target features to F32.
    pub widen: NativeKernel<widen_rows::Entry>,
    pub markov: Option<MarkovKernels>,
    pub convolved: Option<super::AttestedConvolved>,
    pub selector: Option<super::AttestedSelector>,
    pub corrector: Option<super::AttestedCorrector>,
}

/// Every kernel of the projector's vision program.
#[derive(Clone)]
pub(crate) struct AttestedVision {
    pub kernels: super::VisionKernels,
}

#[derive(Clone)]
pub(crate) struct AttestedState {
    pub feature_copy: NativeKernel<feature_copy::Entry>,
    pub copies: Vec<(Element, NativeKernel<copy_rows::Entry>)>,
    pub conditioning: Option<NativeKernel<conditioning_overlay::Entry>>,
}

/// The entries a read of rows the device selects from an imported streamed
/// table runs around its reader (`GatheredTable::read_selected`): the copy
/// of its selection for the host, and the table weight's own import, which
/// converts the rows the host landed into the table's landing slots.
#[derive(Clone)]
pub(crate) struct AttestedLanding {
    pub notice: NativeKernel<selection_notice::Entry>,
    pub import: AttestedImport,
}

/// The entries that make a weight resident: its import, and for a repacked
/// weight with a packing bound (`WeightPlan::bound`) the entry that makes
/// the bound from the imported weight.
#[derive(Clone)]
pub(crate) enum AttestedImport {
    Dense(NativeKernel<import_dense::Entry>),
    Repack {
        repack: NativeKernel<repack_weight::Entry>,
        bound: Option<NativeKernel<packing_bound::Entry>>,
    },
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
    /// Graphs the programs' device has recorded for submissions so far.
    pub(crate) fn recorded_graphs(&self) -> u64 {
        self.owner.device().recorded_graphs()
    }

    pub fn readout_choices(&self) -> &[crate::ReadoutClassChoice] {
        &self.readout_choices
    }

    /// Remove optional physical recipes before graph formation when ordinary
    /// resource admission admits only the canonical packed model.
    pub fn discard_readout_choices(&mut self) -> Result<(), PlanError> {
        if self.target_graphs.is_some()
            || self.target_readout_graphs.is_some()
            || self.drafter_graphs.is_some()
        {
            return Err(PlanError::Topology(
                "readout physical choices are sealed by prepared graphs",
            ));
        }
        if let ReadoutHeadKernels::Qualified { packed, .. } = &self.target.readout.head {
            self.target.readout.head = packed.as_ref().clone();
        }
        if let Some(head) = &mut self.head {
            for block in &mut head.blocks {
                if let ProjectionKernels::Qualified { packed, .. } = &block.logits {
                    block.logits = packed.as_ref().clone();
                    let weight = block.binding.projection.packed();
                    block.binding.projection = ReadoutHead::Packed {
                        weight: weight.element,
                        weight_scale: weight.scale,
                    };
                }
            }
        }
        if let Some(head) = &mut self.shared_head {
            if let ProjectionKernels::Qualified { packed, .. } = &head.projection {
                head.projection = packed.as_ref().clone();
            }
        }
        self.readout_choices.clear();
        Ok(())
    }

    pub fn install_target_readout_graphs(&mut self, graphs: crate::PreparedTargetReadoutGraphs) {
        self.target_readout_graphs = Some(std::sync::Arc::new(graphs));
    }

    pub fn target_readout_graphs(&self) -> Option<&crate::PreparedTargetReadoutGraphs> {
        self.target_readout_graphs.as_deref()
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

    /// Seals head, vision and state-copy graphs for the planned method cap
    /// and exact separate-draft physical widths.
    #[allow(clippy::too_many_arguments)]
    pub fn prepare_auxiliary_graphs(
        &mut self,
        device: &Device,
        load: &crate::ModelLoadPlan,
        definition: &magnitude_family_contracts::ModelDefinition,
        programs: &ProgramPlan,
        state: &crate::StateResourcePlan,
        limits: crate::ResourceLimits,
        method: crate::PlannedMethod,
    ) -> Result<(), String> {
        let proposals = method.draft_rows();
        let vision_plan = programs.vision();
        let (target_state, head_state) = (state.target_state(), state.head_state());
        if self.shared_head.is_some() != programs.shared_head().is_some() {
            return Err("attested and planned draft heads reading target history disagree".into());
        }
        if let (Some(handles), Some(plan)) = (&self.shared_head, programs.shared_head()) {
            let geometry = crate::programs::native_shared_head::SharedHeadGeometry::new(
                definition, state, plan,
            )?;
            let classes =
                crate::programs::native_shared_head::shared_head_graph_classes(limits, proposals)?;
            let landing = self.landing(load, TARGET_TABLE)?;
            self.drafter_graphs = Some(crate::PreparedDrafterGraphs::SharedHead(Rc::new(
                crate::programs::native_shared_head::PreparedSharedHeadGraphs::prepare(
                    device,
                    handles,
                    landing.as_ref(),
                    load,
                    &geometry,
                    classes,
                )
                .map_err(|error| error.to_string())?,
            )));
        }
        let row_classes = magnitude_batching::row_classes(limits.max_launch_rows)
            .into_iter()
            .map(|rows| rows as u64)
            .collect::<Vec<_>>();
        if row_classes.is_empty() {
            return Err(format!(
                "launch row bound {} has no row class",
                limits.max_launch_rows
            ));
        }
        if self.vision.is_some() != vision_plan.is_some()
            || self.vision.is_some() != definition.vision.is_some()
        {
            return Err("attested vision, program plan and model definition disagree".into());
        }
        if let (Some(handles), Some(state)) = (&self.head, head_state) {
            let head = definition
                .head
                .as_ref()
                .ok_or("head graph requires a draft head")?;
            let head = crate::operators::CausalHeadBlock::of(
                head,
                head.blocks
                    .first()
                    .ok_or("head graph requires a draft head block")?,
            )
            .map_err(|error| error.to_string())?;
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
            let landing = self.landing(load, TARGET_TABLE)?;
            self.drafter_graphs = Some(crate::PreparedDrafterGraphs::Head(Rc::new(
                crate::programs::native_head::PreparedHeadGraphs::prepare(
                    device,
                    handles,
                    landing.as_ref(),
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
            let geometry = crate::programs::native_draft::DraftGeometry::new(definition, state)?;
            let classes = crate::programs::native_draft::draft_graph_classes(
                limits,
                proposals,
                &method.block_widths(),
                draft,
            )?;
            let landing = self.landing(load, crate::programs::native_draft::token_table(draft))?;
            self.drafter_graphs = Some(crate::PreparedDrafterGraphs::Draft(Rc::new(
                crate::programs::native_draft::PreparedDraftGraphs::prepare(
                    device,
                    handles,
                    landing.as_ref(),
                    load,
                    programs.draft().ok_or("draft graphs require a draft program")?,
                    &geometry,
                    classes,
                )
                .map_err(|error| error.to_string())?,
            )));
        }
        if let (Some(handles), Some(vision), Some(_)) =
            (&self.vision, definition.vision.as_ref(), vision_plan)
        {
            let patch_rows = crate::programs::native_vision::image_patch_classes(
                limits.max_image_cells,
                vision,
            )?;
            self.vision_graphs = Some(Rc::new(
                crate::programs::native_vision::PreparedVisionGraphs::prepare_classes(
                    device, handles, load, vision, patch_rows,
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

    /// Hold the load's sealed target graphs: its one copy, which the bound
    /// graphs of the target program share.
    pub fn install_target_graphs(&mut self, graphs: crate::PreparedTargetGraphs) {
        self.target_graphs = Some(std::sync::Arc::new(graphs));
    }

    pub fn target_graphs(&self) -> Option<&crate::PreparedTargetGraphs> {
        self.target_graphs.as_deref()
    }

    pub fn prepare_target_graphs(
        &self,
        device: &Device,
        load: &crate::ModelLoadPlan,
        geometry: &magnitude_family_contracts::Decoder,
        state: &crate::StateResourcePlan,
        plan: &crate::TargetProgramPlan,
        limits: crate::ResourceLimits,
        workers: &seismic::Workers,
    ) -> Result<crate::PreparedTargetGraphs, String> {
        let landing = self.landing(load, TARGET_TABLE)?;
        crate::programs::native_target_graph::PreparedTargetGraphs::prepare(
            device,
            &self.target,
            self.notice.as_ref(),
            landing.as_ref(),
            load,
            geometry,
            state,
            plan,
            limits,
            workers,
        )
    }

    /// The device storage the prepared programs hold: their owner
    /// ([`Self::owner`]), one `u32` word. No program entry holds storage of
    /// its own (`Specializer::form`).
    pub const DEVICE_BYTES: u64 = std::mem::size_of::<u32>() as u64;

    /// The prepared programs' device identity owner, retained with every
    /// prepared group: the one word [`Self::DEVICE_BYTES`] charges.
    pub(super) fn owner(device: &Device) -> Result<Tensor, CatalogFailure> {
        Tensor::zeros(device, Element::u32(), &[1]).map_err(|error| {
            CatalogFailure::Preparation {
                entry: "catalog_owner",
                bindings: "A=u32".into(),
                outcome: error.to_string(),
            }
        })
    }

    /// Search every tuning unit of the draft's program plan that has no
    /// stored record, on the opened device within the tuning budget, and
    /// publish each unit's record to `tuning.cache`. Nothing is prepared
    /// for serving: [`Self::prepare_draft`] reads the records. This is the
    /// only search; it runs in a process that ends before the model is
    /// loaded, and holds no more than `room`. Returns every unit's outcome.
    pub fn search_draft(
        plan: &ExecutionPlanDraft,
        device: &Device,
        tuning: TuningContext<'_>,
        room: Option<super::tuning::TuningRoom>,
    ) -> Result<Vec<TunedEntry>, CatalogError> {
        let limits = TuningLimits::of(plan.policy().limits(), tuning.definition);
        Self::preparation_inputs(
            plan.policy().path(),
            plan.device(),
            plan.programs(),
            plan.load(),
            limits,
            plan.policy().limits(),
            plan.policy().method(),
            device,
            tuning,
            room,
        )
        .and_then(NativePreparationCache::search_programs)
        .map_err(|failure| CatalogError::native(device.backend(), failure))
    }

    /// What a walk of `topology` on `device` reads, once the plan and the
    /// opened device are known to agree.
    #[allow(clippy::too_many_arguments)]
    fn preparation_inputs<'a>(
        path: ExecutionPath,
        planned_device: &PlannedDevice,
        topology: &'a ProgramPlan,
        load: &'a crate::ModelLoadPlan,
        limits: TuningLimits,
        selection_limits: crate::ResourceLimits,
        method: crate::PlannedMethod,
        device: &'a Device,
        tuning: TuningContext<'a>,
        room: Option<super::tuning::TuningRoom>,
    ) -> Result<PreparationInputs<'a>, CatalogFailure> {
        if path != ExecutionPath::Native {
            return Err(CatalogFailure::Preparation {
                entry: "program_factory",
                bindings: "execution path".into(),
                outcome: format!("the native program factory cannot serve the {path} path"),
            });
        }
        if planned_device.selector() != device.info().selector() {
            return Err(CatalogFailure::Preparation {
                entry: "program_factory",
                bindings: "selected device".into(),
                outcome: "program plan and opened device differ".into(),
            });
        }
        Ok(PreparationInputs {
            device,
            plan: topology,
            load,
            limits,
            tuning,
            room,
            selection_limits,
            graph_demand: crate::programs::invocation_demand::GraphInvocationDemand::new(
                method,
                super::tuning::workload::terms(limits, &method.draft_widths()),
                tuning.definition.draft.as_ref(),
                selection_limits.max_launch_rows as u64,
            ),
        })
    }

    /// Prepare every native entry the draft's program plan needs on the
    /// opened device, as a load does: static values from the model, each
    /// tuned entry at the choice its stored tuning record names; then
    /// qualification. At a unit without a record it does what `unprepared`
    /// says: stops with [`CatalogFailure::Unprepared`], or prepares the
    /// unit at its canonical default. Nothing is searched, measured or
    /// published. Errors name the native
    /// path and the device's backend; entries without an implementation for
    /// that backend are reported together.
    pub fn prepare_draft(
        plan: &ExecutionPlanDraft,
        device: &Device,
        tuning: TuningContext<'_>,
        unprepared: super::tuning::UnpreparedUnits,
    ) -> Result<Self, CatalogError> {
        Self::prepare_for(
            plan.policy().path(),
            plan.device(),
            plan.programs(),
            plan.load(),
            TuningLimits::of(plan.policy().limits(), tuning.definition),
            plan.policy().limits(),
            plan.policy().method(),
            device,
            tuning,
            unprepared,
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
        selection_limits: crate::ResourceLimits,
        method: crate::PlannedMethod,
        device: &Device,
        tuning: TuningContext<'_>,
        unprepared: super::tuning::UnpreparedUnits,
    ) -> Result<Self, CatalogFailure> {
        let definition = tuning.definition;
        let mut prepared = NativePreparationCache::prepare_programs(
            Self::preparation_inputs(
                path,
                planned_device,
                topology,
                load,
                limits,
                selection_limits,
                method,
                device,
                tuning,
                None,
            )?,
            unprepared,
        )?;
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
        // The decoder blocks of `slots`, whose sublayers' weights `scope`
        // names: the target's, or those of a draft head reading its history.
        let attest_blocks = |slots: &[crate::TargetBlockProgramSlot],
                             scope: fn(SublayerIndex) -> WeightScope|
         -> Result<Vec<AttestedTargetBlock>, CatalogFailure> {
            let mut blocks = Vec::with_capacity(slots.len());
            for (index, block) in slots.iter().enumerate() {
                let sublayer = |sublayer| {
                    u32::try_from(index)
                        .map(|block| scope(SublayerIndex { block, sublayer }))
                        .map_err(|_| CatalogFailure::Preparation {
                            entry: "program_factory",
                            bindings: "target block index".into(),
                            outcome: "the block index exceeds u32".into(),
                        })
                };
                let [mixer_tail, feed_forward_tail, per_layer_tail] = output_tails(*block);
                let output_scales = OutputScales {
                    mixer: output_scale(load, tuning, sublayer(0)?, mixer_tail)?,
                    feed_forward: output_scale(load, tuning, sublayer(1)?, feed_forward_tail)?,
                    per_layer: output_scale(load, tuning, sublayer(2)?, per_layer_tail)?,
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
                let streams = block
                    .streams()
                    .map(|binding| {
                        prepared
                            .target
                            .streams
                            .kernels(binding)
                            .ok_or_else(|| missing("stream_mix", binding))
                    })
                    .transpose()?;
                let ngram = block
                    .ngram()
                    .map(|binding| {
                        prepared
                            .target
                            .ngram
                            .get(&binding)
                            .cloned()
                            .ok_or_else(|| missing("ngram_inject", binding))
                    })
                    .transpose()?;
                blocks.push(AttestedTargetBlock {
                    mixer,
                    feed_forward,
                    output_scales,
                    per_layer,
                    streams,
                    ngram,
                });
            }
            Ok(blocks)
        };
        let blocks = attest_blocks(target_plan.blocks(), WeightScope::TargetSublayer)?;
        let shared_head = topology
            .shared_head()
            .map(|plan| {
                let handles = prepared
                    .shared_head
                    .as_ref()
                    .ok_or_else(|| missing("shared_head", "enabled"))?;
                Ok::<_, CatalogFailure>(AttestedSharedHead {
                    embedding: slot(&prepared.target.embedding, plan.embedding, "embedding_rows")?,
                    join: handles
                        .join
                        .clone()
                        .ok_or_else(|| missing("tap_rows", "head"))?,
                    input: handles
                        .input
                        .clone()
                        .ok_or_else(|| missing("project_rows", "head input"))?,
                    blocks: attest_blocks(&plan.blocks, WeightScope::HeadSublayer)?,
                    features: handles
                        .features
                        .clone()
                        .ok_or_else(|| missing("readout_features_rows", "head"))?,
                    projection: handles
                        .projection
                        .clone()
                        .ok_or_else(|| missing("readout_project_rows", "head"))?,
                    centroids: handles.centroids.clone(),
                    cluster_selection: handles.cluster_selection.clone(),
                    cluster_rows: handles.cluster_rows.clone(),
                    placement: handles.placement.clone(),
                    chain: handles
                        .chain
                        .clone()
                        .ok_or_else(|| missing("project_rows", "head chain"))?,
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
                })
            })
            .transpose()?;
        let target = AttestedTarget {
            residual_features: prepared.target.residual_features.clone(),
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
                    let logits = handles
                        .logits
                        .get(&binding)
                        .cloned()
                        .ok_or_else(|| missing("head_logits", binding))?;
                    let mut attested_binding = binding;
                    if matches!(&logits, ProjectionKernels::Qualified { .. }) {
                        let weight = binding.projection.packed();
                        attested_binding.projection = ReadoutHead::Qualified {
                            weight: weight.element,
                            weight_scale: weight.scale,
                        };
                    }
                    blocks.push(AttestedHeadBlock {
                        streams: binding
                            .streams
                            .map(|_| {
                                handles
                                    .streams
                                    .get(&binding)
                                    .cloned()
                                    .ok_or_else(|| missing("head streams", binding))
                            })
                            .transpose()?,
                        copy: slot(&handles.copy, binding, "feature_copy")?,
                        gather: slot(&handles.gather, binding, "feature_rows")?,
                        binding: attested_binding,
                        input: slot(&handles.input, binding, "draft_rows")?,
                        priming_project: slot(
                            &handles.priming_project,
                            binding,
                            "attention_project",
                        )?,
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
                                });
                            }
                        },
                        features: slot(&handles.features, binding, "readout_features_rows")?,
                        logits,
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
                    convolved: draft_plan
                        .convolutions()
                        .map(|binding| {
                            handles
                                .attest_convolved(draft_plan, binding)
                                .map_err(|entry| missing("draft_convolved_stages", entry))
                        })
                        .transpose()?,
                    selector: draft_plan
                        .selector()
                        .map(|binding| {
                            handles
                                .attest_selector(draft_plan, binding)
                                .map_err(|entry| missing("draft_selector_stages", entry))
                        })
                        .transpose()?,
                    corrector: draft_plan
                        .corrector()
                        .map(|binding| {
                            handles
                                .attest_corrector(draft_plan, binding)
                                .map_err(|entry| missing("draft_corrector_stages", entry))
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
                Some(DType::I32) => &prepared.glue.copy_rows_i32,
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
            feature_copy: prepared
                .glue
                .feature_copy
                .clone()
                .ok_or_else(|| missing("feature_copy", "activation"))?,
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
            let handle = prepared
                .import
                .attested(binding)
                .ok_or_else(|| missing("weight_import", binding))?;
            imports.push((binding, handle));
        }
        let attested = Self {
            backend: device.backend(),
            readout_choices: prepared.readout_choices.clone(),
            tuned,
            unprepared: prepared.unprepared,
            owner: prepared.owner.clone(),
            report,
            target,
            head,
            shared_head,
            draft,
            vision,
            state,
            imports,
            notice: prepared.glue.selection_notice.clone(),
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

    /// Startup-only allowance for qualification. Entries are specialized to
    /// the model's geometry, so fixtures run at it: the weight fixtures of one
    /// qualification hold at most one weight scope's tensors at their
    /// resident representation (one block, or the target's norm and output),
    /// and fixture locals drop before the next binding. Two fixtures differ
    /// from a scope's planned weights: the embedding table is qualified on a
    /// single row, and each head block also holds its draft projection, the
    /// complete target output. Sixteen MiB more
    /// covers one-row activations and tables, logits rows, checked results
    /// and allocator alignment. The programs' own storage
    /// ([`Self::DEVICE_BYTES`]) is charged separately.
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
            // The draft projection fixture: the complete output projection
            // or its complete progressive planes.
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
                    fixture(output.resident, &[vocabulary, hidden])?
                }
                None => {
                    let top = target(WeightKind::OutputPlane(ProgressivePlane::Top))
                        .ok_or("head qualification requires the target output")?;
                    let [vocabulary, groups, _] = top.shape[..] else {
                        return Err("target top plane is not a matrix".into());
                    };
                    let depth = top
                        .progressive_depth()
                        .ok_or("target top plane has no code depth")?;
                    let hidden = groups * 32;
                    ProgressivePlane::ALL
                        .into_iter()
                        .try_fold(0u64, |bytes, plane| {
                            let element = crate::progressive::element(plane);
                            let plane_bytes =
                                fixture(element, &plane.shape(vocabulary, hidden, depth))?;
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
                .bind_weights(&model)
                .map_err(|error| CatalogFailure::Preparation {
                    entry: "target_readout_graph",
                    bindings: "resident readout weights".into(),
                    outcome: error,
                })?;
            crate::programs::native_target::NativeTargetProgram::new(
                model.embedding.tensor().device(),
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
                crate::PreparedDrafterGraphs::SharedHead(graphs) => {
                    self.shared_head
                        .as_ref()
                        .ok_or_else(|| missing("head", "disabled"))?;
                    if resident.depth != 1 {
                        return Err(missing("head", "native single-head topology"));
                    }
                    let graphs = graphs
                        .bind_weights(&resident)
                        .map_err(|error| missing("head", error))?;
                    crate::programs::native_shared_head::NativeSharedHeadProgram::new(
                        definition.decoder.clone(),
                        graphs,
                    )
                    .map(NativeDrafterProgram::SharedHead)
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
    /// The import entry that converts `weight`'s uploaded rows into its
    /// resident storage.
    pub(crate) fn import_entry(
        &self,
        weight: &crate::WeightPlan,
    ) -> Result<AttestedImport, CatalogError> {
        let requested = ImportProgramSlot::of(weight);
        self.imports
            .iter()
            .find(|(slot, _)| *slot == requested)
            .map(|(_, handle)| handle.clone())
            .ok_or_else(|| CatalogError::native(self.backend, missing("weight_import", requested)))
    }

    /// The landing entries of the gathered table `role` of `load`; `None`
    /// of a table without landing slots (`WeightPlan::lands_selected_rows`).
    pub(crate) fn landing(
        &self,
        load: &crate::ModelLoadPlan,
        role: magnitude_family_contracts::WeightRole,
    ) -> Result<Option<AttestedLanding>, String> {
        let plan = load
            .weights()
            .find(|weight| weight.role == role)
            .ok_or_else(|| format!("gathered table {role:?} is not planned"))?;
        if !plan.lands_selected_rows() {
            return Ok(None);
        }
        Ok(Some(AttestedLanding {
            notice: self
                .notice
                .clone()
                .ok_or("a table with landing slots without the selection notice")?,
            import: self
                .import_entry(plan)
                .map_err(|error| error.to_string())?,
        }))
    }

    /// Bind one already planned semantic weight to its exact import entry.
    /// This happens during loader construction, never during a numerical call.
    pub(crate) fn bind_import(
        &self,
        weight: &crate::WeightPlan,
    ) -> Result<crate::programs::native_import::NativeImportProgram, CatalogError> {
        Ok(crate::programs::native_import::NativeImportProgram::new(
            weight.storage_identity(),
            self.import_entry(weight)?,
        ))
    }
    /// The device storage the prepared programs hold, as charged: their
    /// owner's ([`Self::DEVICE_BYTES`]).
    pub(crate) fn device_storage_bytes(&self) -> u64 {
        self.owner.storage_bytes()
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
    /// Every tuned entry these programs were prepared from a stored record
    /// of.
    pub fn tuned(&self) -> &[TunedEntry] {
        &self.tuned
    }

    /// Tuning units these programs serve at their canonical defaults
    /// because no record was stored for them.
    pub fn unprepared(&self) -> usize {
        self.unprepared
    }
}
