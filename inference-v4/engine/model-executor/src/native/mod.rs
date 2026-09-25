mod attestation;
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
    AttestedFeedForward, AttestedHead, AttestedMixer, AttestedState, AttestedTarget,
    AttestedTargetBlock, AttestedVision,
};
use glue::GlueKernels;
use head::HeadKernels;
pub(crate) use head::draft_vocabulary;
pub(crate) use import::ImportKernels;
use preparation::NativePreparationCache;
use qualification::QualificationView;
use target::TargetKernels;
pub(crate) use target::{
    AttentionHistoryKernels, AttentionKernels, DenseKernels, ReadoutKernels, RecurrentKernels,
    RoutedKernels,
};
pub use tuning::{
    attention_points, row_points, ZeroTuningWeights, PointShape, TunedEntry, TuningContext,
    TuningEvent, TuningLimits, TuningObserver, TuningOrigin, TuningWeightSource, UnreportedTuning, ROTATION_LAYERS,
    TUNING_CONTEXTS, TUNING_ROWS,
};
#[cfg(feature = "pinned-tuning")]
pub use tuning::pinned as pinned_tuning;
#[cfg(feature = "tuning-survey")]
pub use tuning::survey as tuning_survey;
use vision::VisionKernels;

use crate::{
    AttentionShape, ExecutionPath, FeedForwardProgramSlot, ImportProgramSlot, MixerProgramSlot,
    ProgramPlan, RecurrentBinding, RoutedBinding,
};
use magnitude_model_kernels::{
    copy_rows, head_logits_rows, import_dense, gated_attention_decode, gated_attention_decode_k8v4,
    attention_output, gated_attention_prefill, gated_attention_prefill_k8v4,
    gated_attention_project,
    conditioning_overlay, dense_expand, dense_output, draft_rows,
    embedding_rows, readout_features_rows, readout_head_rows, gated_delta_chunk, gated_delta_output,
    gated_delta_project, gated_delta_step, routed_combine, routed_expand,
    routed_experts, routed_group, routed_output, routed_route,
    readout_selected_rows, qwen_vision_block, qwen_vision_merger, qwen_vision_stem, repack_weight, sample_rows, shape_rows,
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
    TargetDense,
    TargetRouted,
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
        let rows16 = |representation| Element::stored(representation, seismic::Layout::Rows16).unwrap();
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
                A: binding.activation,
            },
            &step,
        )
        .unwrap();
        let chunk = defaults::<gated_delta_chunk::Entry>(&device, &statics);
        gated_delta_chunk::native_for_device_with(
            &device,
            gated_delta_chunk::Elements {
                A: binding.activation,
            },
            &chunk,
        )
        .unwrap();
        gated_delta_output::native_for_device_with(
            &device,
            gated_delta_output::Elements {
                RN: binding.recurrent_norm,
                OW: binding.output,
                A: binding.activation,
            },
            &defaults::<gated_delta_output::Entry>(&device, &projections),
        )
        .unwrap();
    }
}
