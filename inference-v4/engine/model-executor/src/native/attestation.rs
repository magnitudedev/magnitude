//! Ordered callable native slots constructed from the one ProgramPlan.
//! Preparation is the only place a checked binding lookup is permitted.

use super::*;
use crate::{
    ExecutionPlan, ExecutionPlanDraft, FeedForwardProgramSlot, ImportProgramSlot, MixerProgramSlot,
    PlanError, PlannedDevice, ProgramPlan,
};
use magnitude_model_kernels::{import_dense, repack_weight};
use std::{collections::HashSet, rc::Rc};

pub struct AttestedPrograms {
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
    pub embedding: NativeKernel<qwen_embedding_rows::Entry>,
    pub blocks: Vec<AttestedTargetBlock>,
    pub readout: ReadoutKernels,
    pub features: Option<NativeKernel<qwen_features_rows::Entry>>,
    pub selected: NativeKernel<qwen_selected_rows::Entry>,
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
}

#[derive(Clone)]
pub(crate) struct AttestedHeadBlock {
    pub input: NativeKernel<qwen_head_rows::Entry>,
    pub attention: AttentionKernels,
    pub dense: DenseKernels,
    pub features: NativeKernel<qwen_features_rows::Entry>,
    pub logits: NativeKernel<head_logits_rows::Entry>,
}

#[derive(Clone)]
pub(crate) struct AttestedVision {
    pub stem: NativeKernel<qwen_vision_stem::Entry>,
    pub blocks: Vec<NativeKernel<qwen_vision_block::Entry>>,
    pub merger: NativeKernel<qwen_vision_merger::Entry>,
    pub output: NativeKernel<qwen_vision_feature_output::Entry>,
}

#[derive(Clone)]
pub(crate) struct AttestedState {
    pub copies: Vec<(Element, NativeKernel<copy_rows::Entry>)>,
    pub gather: Option<NativeKernel<gather_rows::Entry>>,
    pub conditioning: Option<NativeKernel<qwen_conditioning_overlay::Entry>>,
}

#[derive(Clone)]
pub(crate) enum AttestedImport {
    Dense(NativeKernel<import_dense::Entry>),
    Repack(NativeKernel<repack_weight::Entry>),
}

fn missing(entry: &'static str, binding: impl fmt::Debug) -> CatalogError {
    CatalogError::Qualification {
        path: ExecutionPath::NativeMetal,
        entry,
        bindings: format!("{binding:?}"),
        outcome: "ordered program slot was not prepared".into(),
    }
}

fn slot<K, E>(
    handles: &HashMap<K, NativeKernel<E>>,
    binding: K,
    entry: &'static str,
) -> Result<NativeKernel<E>, CatalogError>
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
        crate::programs::native_target_readout_graph::PreparedTargetReadoutGraphs::prepare(
            device,
            &self.target,
            &self.state,
            load,
            geometry,
            limits,
        )
    }

    pub fn prepare_auxiliary_graphs(
        &mut self,
        device: &Device,
        load: &crate::ModelLoadPlan,
        definition: &magnitude_model_contracts::ModelDefinition,
        target_state: &crate::StateStorePlan,
        head_state: Option<&crate::StateStorePlan>,
        limits: crate::ResourceLimits,
    ) -> Result<(), String> {
        let max_rows = u64::try_from(limits.max_batch_rows)
            .map_err(|_| "batch row bound exceeds u64")?
            .checked_next_power_of_two()
            .ok_or("batch row class overflows")?;
        let max_segments = u64::try_from(limits.in_flight_requests)
            .map_err(|_| "request slot bound exceeds u64")?
            .checked_next_power_of_two()
            .ok_or("request slot class overflows")?
            .min(max_rows);
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
            let mut forward_classes = Vec::new();
            let mut project_classes = Vec::new();
            let mut rows = 1u64;
            while rows <= max_rows {
                project_classes.push(crate::programs::native_head::HeadProjectGraphClass { rows });
                let mut segments = 1u64;
                while segments <= max_segments && segments <= rows {
                    forward_classes.push(crate::programs::native_head::HeadForwardGraphClass {
                        rows,
                        segments,
                        history_rows,
                    });
                    segments *= 2;
                }
                rows *= 2;
            }
            let forward = crate::programs::native_head::PreparedHeadForwardGraphs::prepare(
                device,
                handles,
                load,
                &definition.geometry,
                attention,
                self.state
                    .copies
                    .iter()
                    .find(|(element, _)| {
                        *element
                            == match definition.geometry.activation_dtype {
                                magnitude_model_contracts::ActivationDType::F16 => {
                                    seismic::Element::f16()
                                }
                                magnitude_model_contracts::ActivationDType::BF16 => {
                                    seismic::Element::bf16()
                                }
                            }
                    })
                    .or_else(|| self.state.copies.first())
                    .map(|(_, kernel)| kernel)
                    .ok_or_else(|| "head conditioning copy specialization is absent".to_string())?,
                forward_classes,
            )
            .map_err(|error| error.to_string())?;
            let project = crate::programs::native_head::PreparedHeadProjectGraphs::prepare(
                device,
                handles,
                &self.target.shape,
                &self.target.sample,
                load,
                &definition.geometry,
                self.state
                    .copies
                    .iter()
                    .find(|(element, _)| {
                        *element
                            == match definition.geometry.activation_dtype {
                                magnitude_model_contracts::ActivationDType::F16 => {
                                    seismic::Element::f16()
                                }
                                magnitude_model_contracts::ActivationDType::BF16 => {
                                    seismic::Element::bf16()
                                }
                            }
                    })
                    .or_else(|| self.state.copies.first())
                    .map(|(_, kernel)| kernel)
                    .ok_or_else(|| "head feature copy specialization is absent".to_string())?,
                project_classes,
            )
            .map_err(|error| error.to_string())?;
            self.head_graphs = Some(Rc::new(
                crate::programs::native_head::PreparedHeadGraphs::from_parts(forward, project)
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
                    let mut rows = 1u64;
                    while rows <= max_rows {
                        let class = crate::programs::native_state::StateCopyGraphClass {
                            element: Element::dense(plane.dtype),
                            source_extents: extents.clone(),
                            destination_extents: extents.clone(),
                            map_rows: rows,
                        };
                        if !state_classes.contains(&class) {
                            state_classes.push(class);
                        }
                        rows *= 2;
                    }
                }
            }
        }
        let mut segments = 1u64;
        while segments <= max_segments {
            for component in &target_state.recurrent_components {
                let width = component
                    .shape
                    .iter()
                    .try_fold(1u64, |total, extent| {
                        total.checked_mul(u64::try_from(*extent).ok()?)
                    })
                    .ok_or("recurrent component width overflow")?;
                for (source_rows, destination_rows) in [(1, segments), (segments, 1)] {
                    let class = crate::programs::native_state::StateCopyGraphClass {
                        element: Element::dense(component.dtype),
                        source_extents: vec![source_rows, 1, width],
                        destination_extents: vec![destination_rows, 1, width],
                        map_rows: 1,
                    };
                    if !state_classes.contains(&class) {
                        state_classes.push(class);
                    }
                }
            }
            segments *= 2;
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
            &self.state,
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
        bytes += bytes!(qwen_conditioning_overlay) + bytes!(gather_rows);
        bytes += bytes!(shape_rows) + bytes!(sample_rows);
        let target = plan.target();
        bytes += bytes!(qwen_embedding_rows);
        for block in target.blocks() {
            match block.mixer() {
                MixerProgramSlot::Attention(binding)
                    if charged_mixers.insert(MixerProgramSlot::Attention(binding)) =>
                {
                    bytes += bytes!(qwen_attention_normalize)
                        + bytes!(qwen_attention_project)
                        + bytes!(qwen_attention_prepare)
                        + bytes!(qwen_attention_attend)
                        + bytes!(qwen_attention_output)
                }
                MixerProgramSlot::Recurrent(binding)
                    if charged_mixers.insert(MixerProgramSlot::Recurrent(binding)) =>
                {
                    bytes += bytes!(qwen_recurrent_normalize)
                        + bytes!(qwen_recurrent_project)
                        + bytes!(qwen_recurrent_prepare)
                        + bytes!(qwen_recurrent_scan)
                        + bytes!(qwen_recurrent_mix)
                        + bytes!(qwen_recurrent_output)
                }
                _ => {}
            }
            match block.feed_forward() {
                FeedForwardProgramSlot::Dense(binding)
                    if charged_feed_forward.insert(FeedForwardProgramSlot::Dense(binding)) =>
                {
                    bytes += bytes!(qwen_dense_expand)
                        + bytes!(qwen_dense_output)
                        + bytes!(qwen_dense_expand_demanded)
                        + bytes!(qwen_dense_output_demanded)
                }
                FeedForwardProgramSlot::Routed(binding)
                    if charged_feed_forward.insert(FeedForwardProgramSlot::Routed(binding)) =>
                {
                    bytes += bytes!(qwen_routed_normalize)
                        + bytes!(qwen_routed_logits)
                        + bytes!(qwen_routed_select)
                        + bytes!(qwen_routed_expand)
                        + bytes!(qwen_routed_output)
                }
                _ => {}
            }
        }
        bytes += bytes!(qwen_features_rows) + bytes!(head_logits_rows) + bytes!(qwen_selected_rows);
        if target.features().is_some() {
            bytes += bytes!(qwen_features_rows);
        }
        if let Some(head) = plan.head() {
            for &binding in head.blocks() {
                if charged_heads.insert(binding) {
                    bytes += bytes!(qwen_head_rows)
                        + bytes!(qwen_attention_normalize)
                        + bytes!(qwen_attention_project)
                        + bytes!(qwen_attention_prepare)
                        + bytes!(qwen_attention_attend)
                        + bytes!(qwen_attention_output)
                        + bytes!(qwen_dense_expand)
                        + bytes!(qwen_dense_output)
                        + bytes!(qwen_dense_expand_demanded)
                        + bytes!(qwen_dense_output_demanded)
                        + bytes!(qwen_features_rows)
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
            bytes += bytes!(qwen_vision_merger) + bytes!(qwen_vision_feature_output);
        }
        u64::try_from(bytes)
            .map_err(|_| PlanError::Arithmetic("native invocation workspace bytes overflow"))
    }

    pub fn prepare(plan: &ExecutionPlan, device: &Device) -> Result<Self, CatalogError> {
        Self::prepare_for(plan.policy().path(), plan.device(), plan.programs(), device)
    }

    pub fn prepare_draft(plan: &ExecutionPlanDraft, device: &Device) -> Result<Self, CatalogError> {
        Self::prepare_for(plan.policy().path(), plan.device(), plan.programs(), device)
    }

    fn prepare_for(
        path: ExecutionPath,
        planned_device: &PlannedDevice,
        topology: &ProgramPlan,
        device: &Device,
    ) -> Result<Self, CatalogError> {
        if path != ExecutionPath::NativeMetal || device.backend() != BackendName::Metal {
            return Err(CatalogError::Backend {
                path,
                backend: device.backend(),
                outcome: "native program factory requires the planned Metal path".into(),
            });
        }
        if planned_device.selector() != device.info().selector {
            return Err(CatalogError::Preparation {
                path: ExecutionPath::NativeMetal,
                entry: "program_factory",
                bindings: "selected device".into(),
                outcome: "program plan and opened device differ".into(),
            });
        }
        let prepared = NativePreparationCache::prepare_programs(device, topology)?;
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
                "qwen_embedding_rows",
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
                .map(|binding| slot(&prepared.target.features, binding, "qwen_features_rows"))
                .transpose()?,
            selected: slot(
                &prepared.target.selected,
                target_plan.readout(),
                "qwen_selected_rows",
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
                        input: slot(&handles.input, binding, "qwen_head_rows")?,
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
                        features: slot(&handles.features, binding, "qwen_features_rows")?,
                        logits: slot(&handles.logits, binding, "head_logits_rows")?,
                    });
                }
                Ok::<_, CatalogError>(AttestedHead { blocks })
            })
            .transpose()?;
        let vision = topology
            .vision()
            .map(|vision_plan| {
                let handles = prepared
                    .vision
                    .as_ref()
                    .ok_or_else(|| missing("vision", "enabled"))?;
                Ok::<_, CatalogError>(AttestedVision {
                    stem: slot(&handles.stem, vision_plan.patch(), "qwen_vision_stem")?,
                    blocks: vision_plan
                        .blocks()
                        .iter()
                        .copied()
                        .map(|binding| slot(&handles.blocks, binding, "qwen_vision_block"))
                        .collect::<Result<_, _>>()?,
                    merger: slot(&handles.merger, vision_plan.merger(), "qwen_vision_merger")?,
                    output: slot(
                        &handles.output,
                        vision_plan.merger(),
                        "qwen_vision_feature_output",
                    )?,
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
                gather: Some(
                    prepared
                        .glue
                        .gather_rows
                        .clone()
                        .ok_or_else(|| missing("gather_rows", "target selection"))?,
                ),
                conditioning: Some(
                    prepared.glue.conditioning_overlay.clone().ok_or_else(|| {
                        missing("qwen_conditioning_overlay", "target conditioning")
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
                CatalogError::Preparation {
                    path: ExecutionPath::NativeMetal,
                    entry: "program_factory",
                    bindings: "native invocation workspace".into(),
                    outcome: error.to_string(),
                }
            })?;
        if invocation_workspace_bytes != planned_bytes {
            return Err(CatalogError::Qualification {
                path: ExecutionPath::NativeMetal,
                entry: "program_factory",
                bindings: "native invocation workspace".into(),
                outcome: format!(
                    "prepared {invocation_workspace_bytes} bytes but the ordered program plan charged {planned_bytes}"
                ),
            });
        }
        let attested = Self {
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
        QualificationView::new(&attested, topology).qualify(device)?;
        Ok(attested)
    }

    fn sum_invocation_workspace_bytes(
        prepared: &NativePreparationCache,
    ) -> Result<u64, CatalogError> {
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
            bytes += u128::from(handles.normalize.invocation_workspace_bytes())
                + u128::from(handles.project.invocation_workspace_bytes())
                + u128::from(handles.prepare.invocation_workspace_bytes())
                + u128::from(handles.attend.invocation_workspace_bytes())
                + u128::from(handles.output.invocation_workspace_bytes());
        }
        for handles in prepared.target.recurrent.values() {
            bytes += u128::from(handles.normalize.invocation_workspace_bytes())
                + u128::from(handles.project.invocation_workspace_bytes())
                + u128::from(handles.prepare.invocation_workspace_bytes())
                + u128::from(handles.scan.invocation_workspace_bytes())
                + u128::from(handles.mix.invocation_workspace_bytes())
                + u128::from(handles.output.invocation_workspace_bytes());
        }
        for handles in prepared.target.dense.values() {
            bytes += u128::from(handles.expand.invocation_workspace_bytes())
                + u128::from(handles.output.invocation_workspace_bytes())
                + u128::from(handles.expand_demanded.invocation_workspace_bytes())
                + u128::from(handles.output_demanded.invocation_workspace_bytes());
        }
        for handles in prepared.target.routed.values() {
            bytes += u128::from(handles.normalize.invocation_workspace_bytes())
                + u128::from(handles.logits.invocation_workspace_bytes())
                + u128::from(handles.select.invocation_workspace_bytes())
                + u128::from(handles.expand.invocation_workspace_bytes())
                + u128::from(handles.output.invocation_workspace_bytes());
        }
        for handles in prepared.target.readout.values() {
            bytes += u128::from(handles.features.invocation_workspace_bytes())
                + u128::from(handles.logits.invocation_workspace_bytes());
        }
        charge!(prepared.target.features.values());
        charge!(prepared.target.selected.values());
        if let Some(head) = &prepared.head {
            charge!(head.input.values());
            for handles in head.attention.values() {
                bytes += u128::from(handles.normalize.invocation_workspace_bytes())
                    + u128::from(handles.project.invocation_workspace_bytes())
                    + u128::from(handles.prepare.invocation_workspace_bytes())
                    + u128::from(handles.attend.invocation_workspace_bytes())
                    + u128::from(handles.output.invocation_workspace_bytes());
            }
            for handles in head.dense.values() {
                bytes += u128::from(handles.expand.invocation_workspace_bytes())
                    + u128::from(handles.output.invocation_workspace_bytes())
                    + u128::from(handles.expand_demanded.invocation_workspace_bytes())
                    + u128::from(handles.output_demanded.invocation_workspace_bytes());
            }
            charge!(head.features.values());
            charge!(head.logits.values());
        }
        if let Some(vision) = &prepared.vision {
            charge!(vision.stem.values());
            charge!(vision.blocks.values());
            charge!(vision.merger.values());
            charge!(vision.output.values());
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
        if let Some(handle) = &prepared.glue.gather_rows {
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
        u64::try_from(bytes).map_err(|_| CatalogError::Preparation {
            path: ExecutionPath::NativeMetal,
            entry: "program_factory",
            bindings: "native invocation workspace".into(),
            outcome: "prepared invocation workspace bytes overflow".into(),
        })
    }

    /// Startup-only allowance for one bounded qualification fixture.
    /// The target/head/vision fixtures use one row and at most 4-wide hidden
    /// vectors; each repack fixture is one GGUF packet. Fixture locals drop
    /// before the next binding. Sixteen MiB covers all simultaneously live
    /// tensor inputs, allocating checked results, and allocator alignment.
    /// Prepared invocation buffers are charged separately. Recheck this
    /// bound whenever qualification fixture extents change.
    pub const fn qualification_peak_bytes() -> u64 {
        16 * 1024 * 1024
    }
    pub(crate) fn target(&self) -> &AttestedTarget {
        &self.target
    }
    pub(crate) fn bind_target(
        &self,
        model: crate::ResidentTarget,
        geometry: magnitude_model_contracts::DecoderGeometry,
    ) -> Result<crate::programs::native_target::NativeTargetProgram, CatalogError> {
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
            .map_err(|error| CatalogError::Preparation {
                path: ExecutionPath::NativeMetal,
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
            .map_err(|error| CatalogError::Preparation {
                path: ExecutionPath::NativeMetal,
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
        .map_err(|error| CatalogError::Preparation {
            path: ExecutionPath::NativeMetal,
            entry: "target_graph_controls",
            bindings: "sealed attention rotary controls".into(),
            outcome: error.to_string(),
        })
    }
    pub(crate) fn head(&self) -> Option<&AttestedHead> {
        self.head.as_ref()
    }
    pub(crate) fn bind_head(
        &self,
        resident: crate::ResidentHead,
        definition: &magnitude_model_contracts::ModelDefinition,
    ) -> Result<crate::programs::native_head::NativeHeadProgram, CatalogError> {
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
        let attention = definition
            .geometry
            .blocks
            .iter()
            .rev()
            .find_map(|block| match &block.mixer {
                magnitude_model_contracts::MixerGeometry::Attention(geometry) => {
                    Some(geometry.clone())
                }
                _ => None,
            })
            .ok_or_else(|| missing("head", "target attention geometry"))?;
        let graphs = self
            .head_graphs
            .as_ref()
            .ok_or_else(|| missing("head", "prepared native graphs"))?
            .clone()
            .bind_weights(&resident)
            .map_err(|error| missing("head", error))?;
        crate::programs::native_head::NativeHeadProgram::new(
            definition.geometry.clone(),
            attention,
            graphs,
        )
        .map_err(|error| missing("head", error))
    }
    pub(crate) fn vision(&self) -> Option<&AttestedVision> {
        self.vision.as_ref()
    }
    pub(crate) fn bind_vision(
        &self,
        resident: crate::ResidentVision,
        definition: &magnitude_model_contracts::ModelDefinition,
    ) -> Result<crate::programs::native_vision::NativeVisionProgram, CatalogError> {
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
        ExecutionPath::NativeMetal
    }
}
