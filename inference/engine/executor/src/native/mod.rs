mod attestation;
mod draft;
mod glue;
mod head;
mod import;
mod preparation;
mod qualification;
mod specialization;
mod target;
mod tuning;
mod vision;

pub(crate) use attestation::AttestedImport;
pub use attestation::AttestedPrograms;
pub(crate) use attestation::{
    AttestedDraft, AttestedFeedForward, AttestedHead, AttestedHeadBlock, AttestedMixer,
    AttestedState, AttestedTarget, AttestedTargetBlock, AttestedVision, OutputScales,
};
use draft::DraftKernels;
pub(crate) use draft::{
    draft_readout_vocabulary, AttestedDflash2, Dflash2Kernels, Dflash2Projections,
    DraftBlockKernels, MarkovKernels,
};
use glue::GlueKernels;
pub(crate) use head::draft_vocabulary;
use head::HeadKernels;
pub use head::HeadLogitsKernels;
pub(crate) use import::ImportKernels;
pub(crate) use preparation::kernel_requests;
use preparation::NativePreparationCache;
use qualification::QualificationView;
use target::TargetKernels;
pub(crate) use target::{
    AttentionHistoryKernels, AttentionKernels, DenseExpansionKernel, DenseKernels, ExpertKernels,
    GeneralRoutedKernels, ParallelKernels, PerLayerEntryKernels, PerLayerKernels, PostNormKernels,
    ProgressiveReadoutKernels, ReadoutHeadKernels, ReadoutKernels, RecurrentKernels,
    RecurrentStepKernels, RoutedDecodeKernels, RoutedKernels, ShortConvKernels, StateSpaceKernels,
    SublayerOutput, TableConversion, TapKernels,
};
#[cfg(feature = "pinned-tuning")]
pub use tuning::pinned as pinned_tuning;
#[cfg(feature = "tuning-survey")]
pub use tuning::survey as tuning_survey;
pub use tuning::{
    attention_points, row_points, PointShape, TunedEntry, TuningContext, TuningEvent, TuningLimits,
    TuningObserver, TuningOrigin, TuningWeightSource, UnreportedTuning, ZeroTuningWeights,
    ROTATION_LAYERS, TUNING_CONTEXTS, TUNING_ROWS,
};
pub(crate) use vision::VisionKernels;

use crate::{
    AttentionBinding, ExecutionPath, FeedForwardProgramSlot, ImportProgramSlot, MixerProgramSlot,
    ProgramPlan, RecurrentBinding, RoutedBinding, SublayerTail, VisionEntry,
};
use magnitude_kernels::{
    attention_decode, attention_decode_k8v4, attention_output, attention_prefill,
    attention_prefill_k8v4, attention_project, conditioning_overlay, copy_rows, dense_expand,
    dense_output, dense_up, draft_rows, embedding_rows, gated_delta_chunk, gated_delta_project,
    gated_delta_project_convolved, gated_delta_step, gated_delta_step_convolved, head_logits_rows,
    import_dense, moe_tail,
    per_layer_gate, per_layer_inputs, post_norm_residual, project_rows, readout_exact_rows,
    readout_features_rows, readout_head_rows, readout_planes_rows, readout_refine_rows,
    readout_selected_rows, readout_top_rows, repack_weight, routed_combine, routed_down,
    routed_expand, routed_experts, routed_experts_up, routed_gate_up, routed_group, routed_output,
    routed_route, routed_route_shared, routed_scatter, routed_select, routed_up, sample_rows,
    shape_rows,
    short_conv_project, short_conv_rows, state_space_chunk, state_space_gate, state_space_step,
    vision_attention, vision_clamp, vision_linear, vision_norm, vision_patch_stem, vision_pool,
    vision_position,
};
use seismic::{BackendName, DType, Device, Element, NativeKernel, Tensor};
use std::{collections::HashMap, fmt};

/// A failure of the native program catalog, reported with the execution
/// path and the backend of the opened device it was preparing for.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CatalogError {
    pub path: ExecutionPath,
    pub backend: BackendName,
    pub failure: CatalogFailure,
}

impl CatalogError {
    pub(crate) fn native(backend: BackendName, failure: CatalogFailure) -> Self {
        Self {
            path: ExecutionPath::Native,
            backend,
            failure,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CatalogFailure {
    Preparation {
        entry: &'static str,
        bindings: String,
        outcome: String,
    },
    Tuning {
        entry: &'static str,
        bindings: String,
        outcome: String,
    },
    Qualification {
        entry: &'static str,
        bindings: String,
        outcome: String,
    },
    /// The model's statics for an entry lie outside its kernel's domain on
    /// the backend: no configuration of the implementation executes them.
    KernelDomain {
        entry: &'static str,
        statics: Vec<(String, u64)>,
    },
}

impl fmt::Display for CatalogError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let path = self.path;
        let backend = self.backend.as_str();
        match &self.failure {
            CatalogFailure::Preparation {
                entry,
                bindings,
                outcome,
            } => write!(
                formatter,
                "failed to prepare {entry} on {path} ({backend}) with {bindings}: {outcome}"
            ),
            CatalogFailure::Tuning {
                entry,
                bindings,
                outcome,
            } => write!(
                formatter,
                "failed to tune {entry} on {path} ({backend}) with {bindings}: {outcome}"
            ),
            CatalogFailure::Qualification {
                entry,
                bindings,
                outcome,
            } => write!(
                formatter,
                "failed to qualify {entry} on {path} ({backend}) with {bindings}: {outcome}"
            ),
            CatalogFailure::KernelDomain { entry, statics } => {
                let statics = statics
                    .iter()
                    .map(|(name, value)| format!("{name}={value}"))
                    .collect::<Vec<_>>()
                    .join(", ");
                write!(
                    formatter,
                    "{entry} has no admissible {backend} configuration at {statics}"
                )
            }
        }
    }
}

impl std::error::Error for CatalogError {}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum QualificationCase {
    Import,
    State,
    TargetEmbedding,
    TargetAttention,
    TargetRecurrent,
    TargetStateSpace,
    TargetShortConv,
    TargetDense,
    TargetRouted,
    TargetGeneralRouted,
    Readout,
    Sampling,
    Head,
    Vision,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct QualificationReport {
    cases: Vec<QualificationCase>,
}

impl QualificationReport {
    pub fn cases(&self) -> &[QualificationCase] {
        &self.cases
    }
}

fn dense_binding_name(source: DType, resident: DType) -> String {
    format!("E={},U={}", source.name(), resident.name())
}

fn element_binding_name(source: Element, resident: Element) -> String {
    format!("E={},U={}", source.name(), resident.name())
}

#[cfg(test)]
mod qualification_tests {
    use super::*;
    use crate::RecurrentBinding;

    #[test]
    fn packed_recurrent_prepares_with_the_qwen_4b_binding() {
        let catalog = seismic::DeviceCatalog::discover().unwrap();
        let Ok(device) = catalog.open_backend(BackendName::Metal) else {
            return;
        };
        // Native Metal residents use the rows16 layout.
        let rows16 =
            |representation| Element::stored(representation, seismic::Layout::Rows16).unwrap();
        let q4k = rows16("q4k");
        let q5k = rows16("q5k");
        let q8 = rows16("q8g32s");
        let binding = RecurrentBinding {
            key_heads: 16,
            value_heads: 32,
            width: 128,
            convolution_width: 4,
            norm: Element::bf16(),
            qkv: q5k,
            gate: q4k,
            alpha: q8,
            beta: q8,
            recurrent_norm: Element::bf16(),
            output: q5k,
            activation: Element::bf16(),
        };
        let heads = seismic::NativeSpecialization::new()
            .with_static("NK", binding.key_heads)
            .with_static("NV", binding.value_heads)
            .with_static("W", binding.width);
        let projections = heads.clone().with_static("H", 2560);
        let statics = heads.with_static("C", binding.convolution_width);
        /// The declared defaults of `E` at `statics`.
        fn defaults<E: seismic::Entry>(
            device: &seismic::Device,
            statics: &seismic::NativeSpecialization,
        ) -> seismic::NativeSpecialization {
            seismic::generated::native_implementation::<E>(device)
                .unwrap()
                .unwrap()
                .default_specialization(statics)
                .unwrap()
        }
        gated_delta_project::native_for_device_with(
            &device,
            gated_delta_project::Elements {
                NW: binding.norm,
                QW: binding.qkv,
                GW: binding.gate,
                AW: binding.alpha,
                BW: binding.beta,
                A: binding.activation,
            },
            &defaults::<gated_delta_project::Entry>(&device, &projections),
        )
        .unwrap();
        let step = defaults::<gated_delta_step::Entry>(&device, &statics);
        gated_delta_step::native_for_device_with(
            &device,
            gated_delta_step::Elements {
                RN: binding.recurrent_norm,
                A: binding.activation,
            },
            &step,
        )
        .unwrap();
        gated_delta_project_convolved::native_for_device_with(
            &device,
            gated_delta_project_convolved::Elements {
                NW: binding.norm,
                QW: binding.qkv,
                GW: binding.gate,
                AW: binding.alpha,
                BW: binding.beta,
                A: binding.activation,
            },
            &defaults::<gated_delta_project_convolved::Entry>(
                &device,
                &projections
                    .clone()
                    .with_static("C", binding.convolution_width),
            ),
        )
        .unwrap();
        gated_delta_step_convolved::native_for_device_with(
            &device,
            gated_delta_step_convolved::Elements {
                RN: binding.recurrent_norm,
                A: binding.activation,
            },
            &defaults::<gated_delta_step_convolved::Entry>(
                &device,
                &seismic::NativeSpecialization::new()
                    .with_static("NK", binding.key_heads)
                    .with_static("NV", binding.value_heads)
                    .with_static("W", binding.width),
            ),
        )
        .unwrap();
        let chunk = defaults::<gated_delta_chunk::Entry>(&device, &statics);
        gated_delta_chunk::native_for_device_with(
            &device,
            gated_delta_chunk::Elements {
                RN: binding.recurrent_norm,
                A: binding.activation,
            },
            &chunk,
        )
        .unwrap();
        let output = seismic::NativeSpecialization::new()
            .with_static("D", 2560)
            .with_static("Q", binding.value_heads)
            .with_static("W", binding.width);
        attention_output::native_for_device_with(
            &device,
            attention_output::Elements {
                OW: binding.output,
                A: binding.activation,
            },
            &defaults::<attention_output::Entry>(&device, &output),
        )
        .unwrap();
    }
}
