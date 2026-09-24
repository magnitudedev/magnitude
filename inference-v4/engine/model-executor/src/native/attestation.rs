//! Ordered callable native slots constructed from the one ProgramPlan.
//! Preparation is the only place a checked binding lookup is permitted.

use super::*;
use crate::{
    ExecutionPlanDraft, FeedForwardProgramSlot, ImportProgramSlot, MixerProgramSlot,
    PlanError, PlannedDevice, ProgramPlan,
};
use super::preparation::PreparationInputs;
use super::tuning::{TunedEntry, TuningContext, TuningLimits};
use magnitude_model_kernels::{import_dense, repack_weight};
use magnitude_model_state::KvCodec;
use std::{collections::HashSet, rc::Rc};

pub struct AttestedPrograms {
    backend: BackendName,
    tuned: Vec<TunedEntry>,
    pub(super) owner: Tensor,
    report: QualificationReport,
    pub(super) target: AttestedTarget,
    pub(super) head: Option<AttestedHead>,
    pub(super) vision: Option<AttestedVision>,
    pub(super) state: AttestedState,
    pub(super) imports: Vec<(ImportProgramSlot, AttestedImport)>,
    invocation_workspace_bytes: u64,
    target_graphs: Option<crate::PreparedTargetGraphs>,
    target_readout_graphs: Option<crate::PreparedTargetReadoutGraphs>,
    head_graphs: Option<Rc<crate::programs::native_head::PreparedHeadGraphs>>,
    vision_graphs: Option<Rc<crate::programs::native_vision::PreparedVisionGraphs>>,
    state_graphs: Option<Rc<crate::programs::native_state::PreparedStateCopyGraphs>>,
}

#[derive(Clone)]
pub(crate) struct AttestedTarget {
    pub embedding: NativeKernel<embedding_rows::Entry>,
    pub blocks: Vec<AttestedTargetBlock>,
    pub readout: ReadoutKernels,
    pub features: Option<NativeKernel<readout_features_rows::Entry>>,
    pub selected: NativeKernel<readout_selected_rows::Entry>,
    pub shape: NativeKernel<shape_rows::Entry>,
    pub sample: NativeKernel<sample_rows::Entry>,
}

#[derive(Clone)]
pub(crate) struct AttestedTargetBlock {
    pub mixer: AttestedMixer,
    pub feed_forward: AttestedFeedForward,
}

#[derive(Clone)]
pub(crate) enum AttestedMixer {
    Attention(AttentionKernels),
    Recurrent(RecurrentKernels),
}

#[derive(Clone)]
pub(crate) enum AttestedFeedForward {
    Dense(DenseKernels),
    Routed(RoutedKernels),
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
    pub input: NativeKernel<qwen_draft_rows::Entry>,
    pub attention: AttentionKernels,
    pub dense: DenseKernels,
    pub features: NativeKernel<readout_features_rows::Entry>,
    pub logits: NativeKernel<head_logits_rows::Entry>,
}

#[derive(Clone)]
pub(crate) struct AttestedVision {
    pub stem: NativeKernel<qwen_vision_stem::Entry>,
    pub blocks: Vec<NativeKernel<qwen_vision_block::Entry>>,
    pub merger: NativeKernel<qwen_vision_merger::Entry>,
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
        geometry: &magnitude_model_contracts::DecoderGeometry,
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
        definition: &magnitude_model_contracts::ModelDefinition,
        target_state: &crate::StateStorePlan,
        head_state: Option<&crate::StateStorePlan>,
        limits: crate::ResourceLimits,
        proposals: usize,
    ) -> Result<(), String> {
        let row_classes = magnitude_model_batching::row_classes(limits.max_batch_rows)
            .into_iter()
            .map(|rows| rows as u64)
            .collect::<Vec<_>>();
        let max_rows = *row_classes
            .last()
            .ok_or_else(|| format!("batch row bound {} has no row class", limits.max_batch_rows))?;
        if let (Some(handles), Some(state)) = (&self.head, head_state) {
            let attention = definition
                .geometry
                .blocks
                .iter()
                .rev()
                .find_map(|block| match &block.mixer {
                    magnitude_model_contracts::MixerGeometry::Attention(geometry) => Some(geometry),
                    _ => None,
                })
                .ok_or("head graph requires target attention geometry")?;
            let history_rows =
                u64::try_from(state.history_rows).map_err(|_| "head history rows exceed u64")?;
            let slot_bound = limits.in_flight_requests.min(limits.max_batch_rows);
            let slot_classes = magnitude_model_batching::row_classes(slot_bound)
                .into_iter()
                .map(|slots| slots as u64)
                .collect::<Vec<_>>();
            let mut classes = Vec::new();
            // Causal-only heads (catch-up after prefill or while not
            // drafting) over any row class.
            for &entry_rows in &row_classes {
                for &slots in slot_classes.iter().filter(|slots| **slots <= entry_rows) {
                    classes.push(crate::programs::native_head::HeadGraphClass {
                        entry_rows,
                        slots,
                        history_rows,
                        steps: 0,
                        shaped: false,
                    });
                }
            }
            // Drafting heads: each slot enters at most `proposals + 1`
            // committed rows (its accepted verification prefix and anchor).
            for &slots in &slot_classes {
                let entry_bound = magnitude_model_batching::row_class(
                    (slots as usize).saturating_mul(proposals + 1).min(max_rows as usize),
                )
                .ok_or("head entry row bound has no class")? as u64;
                for &entry_rows in row_classes
                    .iter()
                    .filter(|rows| **rows >= slots && **rows <= entry_bound)
                {
                    for steps in 1..=proposals as u64 {
                        for shaped in [false, true] {
                            classes.push(crate::programs::native_head::HeadGraphClass {
                                entry_rows,
                                slots,
                                history_rows,
                                steps,
                                shaped,
                            });
                        }
                    }
                }
            }
            self.head_graphs = Some(Rc::new(
                crate::programs::native_head::PreparedHeadGraphs::prepare(
                    device,
                    handles,
                    load,
                    &definition.geometry,
                    attention,
                    classes,
                )
                .map_err(|error| error.to_string())?,
            ));
        }
        if let (Some(handles), Some(vision)) = (&self.vision, definition.vision.as_ref()) {
            let merge = vision
                .geometry
                .merge
                .checked_mul(vision.geometry.merge)
                .ok_or("vision merge area overflow")?;
            let max_patch_rows = max_rows
                .checked_mul(merge)
                .ok_or("vision patch row bound overflow")?;
            let patch_classes = (1..=max_rows).map(|outputs| outputs * merge);
            debug_assert_eq!(patch_classes.clone().last(), Some(max_patch_rows));
            self.vision_graphs = Some(Rc::new(
                crate::programs::native_vision::PreparedVisionGraphs::prepare_exact_classes(
                    device,
                    handles,
                    load,
                    &vision.geometry,
                    definition.geometry.hidden,
                    patch_classes,
                )
                .map_err(|error| error.to_string())?,
            ));
        }
        let mut state_classes = Vec::new();
        for state in std::iter::once(target_state).chain(head_state) {
            for component in &state.history_components {
                for plane in component.planes() {
                    let width = u64::try_from(plane.row_elements)
                        .map_err(|_| "state plane width exceeds u64")?;
                    let extents = vec![
                        u64::try_from(state.history_rows)
                            .map_err(|_| "state history rows exceed u64")?,
                        1,
                        width,
                    ];
                    for &rows in &row_classes {
                        let class = crate::programs::native_state::StateCopyGraphClass {
                            element: Element::dense(plane.dtype),
                            source_extents: extents.clone(),
                            destination_extents: extents.clone(),
                            map_rows: rows,
                        };
                        if !state_classes.contains(&class) {
                            state_classes.push(class);
                        }
                    }
                }
            }
        }
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

    pub fn head_graphs(&self) -> Option<&Rc<crate::programs::native_head::PreparedHeadGraphs>> {
        self.head_graphs.as_ref()
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
        geometry: &magnitude_model_contracts::DecoderGeometry,
        state: &crate::StateResourcePlan,
        limits: crate::ResourceLimits,
    ) -> Result<crate::PreparedTargetGraphs, String> {
        crate::programs::native_target_graph::PreparedTargetGraphs::prepare(
            device,
            &self.target,
            load,
            geometry,
            state,
            limits,
        )
    }

    /// Device bytes reserved by the exact native specializations before
    /// construction. Duplicate bindings share one prepared handle, matching
    /// the native factory's preparation and the actual measured charge.
    pub fn planned_invocation_workspace_bytes(plan: &ProgramPlan) -> Result<u64, PlanError> {
        macro_rules! bytes {
            ($entry:ident) => {
                u128::from(NativeKernel::<$entry::Entry>::planned_invocation_workspace_bytes())
            };
        }
        let mut charged_imports = HashSet::new();
        let mut charged_copies = HashSet::new();
        let mut charged_mixers = HashSet::new();
        let mut charged_feed_forward = HashSet::new();
        let mut charged_heads = HashSet::new();
        let mut charged_vision_blocks = HashSet::new();
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
                    bytes += bytes!(qwen_attention_project)
                        + bytes!(attention_output)
                        + match binding.history {
                            KvCodec::Dense => {
                                bytes!(qwen_attention_decode) + bytes!(qwen_attention_prefill)
                            }
                            KvCodec::AffineK8V4 => {
                                bytes!(qwen_attention_decode_k8v4)
                                    + bytes!(qwen_attention_prefill_k8v4)
                            }
                            KvCodec::RotatedK4V4 => {
                                return Err(PlanError::Unsupported("native rotated K4/V4 KV codec"))
                            }
                        }
                }
                MixerProgramSlot::Recurrent(binding)
                    if charged_mixers.insert(MixerProgramSlot::Recurrent(binding)) =>
                {
                    bytes += bytes!(qwen_recurrent_project)
                        + bytes!(qwen_recurrent_step)
                        + bytes!(qwen_recurrent_chunk)
                        + bytes!(qwen_recurrent_output)
                }
                _ => {}
            }
            match block.feed_forward() {
                FeedForwardProgramSlot::Dense(binding)
                    if charged_feed_forward.insert(FeedForwardProgramSlot::Dense(binding)) =>
                {
                    bytes += bytes!(qwen_dense_expand) + bytes!(qwen_dense_output)
                }
                FeedForwardProgramSlot::Routed(binding)
                    if charged_feed_forward.insert(FeedForwardProgramSlot::Routed(binding)) =>
                {
                    bytes += bytes!(qwen_routed_route)
                        + bytes!(qwen_routed_expand)
                        + bytes!(qwen_routed_output)
                        + bytes!(qwen_routed_group)
                        + bytes!(qwen_routed_experts)
                        + bytes!(qwen_routed_combine)
                }
                _ => {}
            }
        }
        bytes += bytes!(readout_features_rows) + bytes!(readout_head_rows) + bytes!(readout_selected_rows);
        if target.features().is_some() {
            bytes += bytes!(readout_features_rows);
        }
        if let Some(head) = plan.head() {
            // Token selection over the draft vocabulary.
            bytes += bytes!(shape_rows) + bytes!(sample_rows);
            for &binding in head.blocks() {
                if charged_heads.insert(binding) {
                    bytes += bytes!(qwen_draft_rows)
                        + bytes!(qwen_attention_project)
                        + bytes!(qwen_attention_decode)
                        + bytes!(qwen_attention_prefill)
                        + bytes!(attention_output)
                        + bytes!(qwen_dense_expand)
                        + bytes!(qwen_dense_output)
                        + bytes!(readout_features_rows)
                        + bytes!(head_logits_rows);
                }
            }
        }
        if let Some(vision) = plan.vision() {
            bytes += bytes!(qwen_vision_stem);
            for &binding in vision.blocks() {
                if charged_vision_blocks.insert(binding) {
                    bytes += bytes!(qwen_vision_block);
                }
            }
            bytes += bytes!(qwen_vision_merger);
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
        let limits = plan.policy().limits();
        Self::prepare_for(
            plan.policy().path(),
            plan.device(),
            plan.programs(),
            plan.load(),
            TuningLimits {
                max_rows: limits.max_batch_rows as u64,
                max_projected_rows: limits.max_projected_rows as u64,
                context_tokens: tuning.definition.geometry.context_limit,
            },
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
            });
            include(match block.feed_forward() {
                FeedForwardProgramSlot::Dense(_) => QualificationCase::TargetDense,
                FeedForwardProgramSlot::Routed(_) => QualificationCase::TargetRouted,
            });
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
        let mut blocks = Vec::with_capacity(target_plan.blocks().len());
        for block in target_plan.blocks() {
            let mixer = match block.mixer() {
                MixerProgramSlot::Attention(binding) => AttestedMixer::Attention(
                    prepared
                        .target
                        .attention
                        .get(&binding)
                        .cloned()
                        .ok_or_else(|| missing("qwen_attention_stages", binding))?,
                ),
                MixerProgramSlot::Recurrent(binding) => AttestedMixer::Recurrent(
                    prepared
                        .target
                        .recurrent
                        .get(&binding)
                        .cloned()
                        .ok_or_else(|| missing("qwen_recurrent_stages", binding))?,
                ),
            };
            let feed_forward = match block.feed_forward() {
                FeedForwardProgramSlot::Dense(binding) => AttestedFeedForward::Dense(
                    prepared
                        .target
                        .dense
                        .get(&binding)
                        .cloned()
                        .ok_or_else(|| missing("qwen_dense_stages", binding))?,
                ),
                FeedForwardProgramSlot::Routed(binding) => AttestedFeedForward::Routed(
                    prepared
                        .target
                        .routed
                        .get(&binding)
                        .cloned()
                        .ok_or_else(|| missing("qwen_routed_stages", binding))?,
                ),
            };
            blocks.push(AttestedTargetBlock {
                mixer,
                feed_forward,
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
            selected: slot(
                &prepared.target.selected,
                target_plan.readout(),
                "readout_selected_rows",
            )?,
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
                        input: slot(&handles.input, binding, "qwen_draft_rows")?,
                        attention: handles
                            .attention
                            .get(&binding)
                            .cloned()
                            .ok_or_else(|| missing("qwen_attention_stages", binding))?,
                        dense: handles
                            .dense
                            .get(&binding)
                            .cloned()
                            .ok_or_else(|| missing("qwen_dense_stages", binding))?,
                        features: slot(&handles.features, binding, "readout_features_rows")?,
                        logits: slot(&handles.logits, binding, "head_logits_rows")?,
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
        let vision = topology
            .vision()
            .map(|vision_plan| {
                let handles = prepared
                    .vision
                    .as_ref()
                    .ok_or_else(|| missing("vision", "enabled"))?;
                Ok::<_, CatalogFailure>(AttestedVision {
                    stem: slot(&handles.stem, vision_plan.patch(), "qwen_vision_stem")?,
                    blocks: vision_plan
                        .blocks()
                        .iter()
                        .copied()
                        .map(|binding| slot(&handles.blocks, binding, "qwen_vision_block"))
                        .collect::<Result<_, _>>()?,
                    merger: slot(&handles.merger, vision_plan.merger(), "qwen_vision_merger")?,
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
        let state =
            AttestedState {
                copies,
                conditioning: Some(
                    prepared.glue.conditioning_overlay.clone().ok_or_else(|| {
                        missing("conditioning_overlay", "target conditioning")
                    })?,
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
        let planned_bytes =
            Self::planned_invocation_workspace_bytes(topology).map_err(|error| {
                CatalogFailure::Preparation {
                    entry: "program_factory",
                    bindings: "native invocation workspace".into(),
                    outcome: error.to_string(),
                }
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
            vision,
            state,
            imports,
            invocation_workspace_bytes,
            target_graphs: None,
            target_readout_graphs: None,
            head_graphs: None,
            vision_graphs: None,
            state_graphs: None,
        };
        QualificationView::new(
            &attested,
            topology,
            &definition.geometry,
            definition.vision.as_ref().map(|vision| &vision.geometry),
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
                + u128::from(handles.step.invocation_workspace_bytes())
                + u128::from(handles.chunk.invocation_workspace_bytes())
                + u128::from(handles.output.invocation_workspace_bytes());
        }
        for handles in prepared.target.dense.values() {
            bytes += u128::from(handles.expand.invocation_workspace_bytes())
                + u128::from(handles.output.invocation_workspace_bytes());
        }
        for handles in prepared.target.routed.values() {
            bytes += u128::from(handles.route.invocation_workspace_bytes())
                + u128::from(handles.expand.invocation_workspace_bytes())
                + u128::from(handles.output.invocation_workspace_bytes())
                + u128::from(handles.group.invocation_workspace_bytes())
                + u128::from(handles.experts.invocation_workspace_bytes())
                + u128::from(handles.combine.invocation_workspace_bytes());
        }
        for handles in prepared.target.readout.values() {
            bytes += u128::from(handles.features.invocation_workspace_bytes())
                + u128::from(handles.head.invocation_workspace_bytes());
        }
        charge!(prepared.target.features.values());
        charge!(prepared.target.selected.values());
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
            charge!(head.features.values());
            charge!(head.logits.values());
            charge!(head.shape.iter());
            charge!(head.sample.iter());
        }
        if let Some(vision) = &prepared.vision {
            charge!(vision.stem.values());
            charge!(vision.blocks.values());
            charge!(vision.merger.values());
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
    /// qualification hold at most one weight scope's tensors (one block, or
    /// the target's embedding, norm and output), and fixture locals drop
    /// before the next binding. Sixteen MiB more covers one-row activations,
    /// row tables, logits rows, checked results and allocator alignment.
    /// Prepared invocation buffers are charged separately.
    pub fn qualification_peak_bytes(load: &crate::ModelLoadPlan) -> u64 {
        const FIXTURES: u64 = 16 * 1024 * 1024;
        let mut scopes = HashMap::<magnitude_model_contracts::WeightScope, u64>::new();
        for weight in load.weights() {
            *scopes.entry(weight.role.scope).or_default() += weight.resident_bytes;
        }
        FIXTURES + scopes.into_values().max().unwrap_or(0)
    }
    pub(crate) fn target(&self) -> &AttestedTarget {
        &self.target
    }
    pub(crate) fn bind_target(
        &self,
        model: crate::ResidentTarget,
        geometry: magnitude_model_contracts::DecoderGeometry,
    ) -> Result<crate::programs::native_target::NativeTargetProgram, CatalogError> {
        (|| -> Result<crate::programs::native_target::NativeTargetProgram, CatalogFailure> {
        if model.blocks.len() != self.target.blocks.len()
            || model.blocks.len() != geometry.blocks.len()
        {
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
    pub(crate) fn head(&self) -> Option<&AttestedHead> {
        self.head.as_ref()
    }
    pub(crate) fn bind_head(
        &self,
        resident: crate::ResidentHead,
        definition: &magnitude_model_contracts::ModelDefinition,
    ) -> Result<crate::programs::native_head::NativeHeadProgram, CatalogError> {
        (|| -> Result<crate::programs::native_head::NativeHeadProgram, CatalogFailure> {
        let head = self
            .head
            .as_ref()
            .ok_or_else(|| missing("head", "disabled"))?;
        if head.blocks.len() != 1
            || resident.blocks.len() != 1
            || definition
                .head
                .as_ref()
                .is_none_or(|description| description.depth() != 1)
        {
            return Err(missing("head", "native single-block topology"));
        }
        let graphs = self
            .head_graphs
            .as_ref()
            .ok_or_else(|| missing("head", "prepared native graphs"))?
            .bind_weights(&resident)
            .map_err(|error| missing("head", error))?;
        crate::programs::native_head::NativeHeadProgram::new(definition.geometry.clone(), graphs)
            .map_err(|error| missing("head", error))
    })()
        .map_err(|failure| CatalogError::native(self.backend, failure))
    }
    pub(crate) fn vision(&self) -> Option<&AttestedVision> {
        self.vision.as_ref()
    }
    pub(crate) fn bind_vision(
        &self,
        resident: crate::ResidentVision,
        definition: &magnitude_model_contracts::ModelDefinition,
    ) -> Result<crate::programs::native_vision::NativeVisionProgram, CatalogError> {
        (|| -> Result<crate::programs::native_vision::NativeVisionProgram, CatalogFailure> {
        let vision = self
            .vision
            .as_ref()
            .ok_or_else(|| missing("vision", "disabled"))?;
        let description = definition
            .vision
            .as_ref()
            .ok_or_else(|| missing("vision", "description"))?;
        if vision.blocks.len() != resident.blocks.len()
            || vision.blocks.len() != description.blocks.len()
            || resident.patch_embeddings.len() != 2
        {
            return Err(missing("vision", "resident topology"));
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
    pub(crate) fn state(&self) -> &AttestedState {
        &self.state
    }
    pub(crate) fn bind_state(&self) -> crate::programs::native_state::NativeStateProgram {
        crate::programs::native_state::NativeStateProgram::new(
            self.state_graphs
                .as_ref()
                .expect("state graphs prepared before binding")
                .clone(),
        )
    }
    pub(crate) fn bind_state_with_repair(
        &self,
        target: &crate::programs::native_target::NativeTargetProgram,
        store: std::rc::Rc<magnitude_model_state::StateStore>,
    ) -> crate::programs::native_state::NativeStateProgram {
        self.bind_state().with_repair(target.clone(), store)
    }
    pub(crate) fn imports(&self) -> &[(ImportProgramSlot, AttestedImport)] {
        &self.imports
    }
    /// Bind one already planned semantic weight to its exact import entry.
    /// This happens during loader construction, never during a numerical call.
    pub(crate) fn bind_import(
        &self,
        weight: &crate::WeightPlan,
    ) -> Result<crate::programs::native_import::NativeImportProgram, CatalogError> {
        (|| -> Result<crate::programs::native_import::NativeImportProgram, CatalogFailure> {
        let requested = if let (Some(source), Some(resident)) =
            (weight.source.dtype(), weight.resident.dtype())
        {
            ImportProgramSlot::Dense { source, resident }
        } else {
            ImportProgramSlot::Repack {
                source: weight.source,
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
