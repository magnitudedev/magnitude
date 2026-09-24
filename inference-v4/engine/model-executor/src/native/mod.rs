mod attestation;
mod glue;
mod head;
mod import;
mod preparation;
mod qualification;
mod specialization;
mod target;
mod vision;

pub(crate) use attestation::AttestedImport;
pub use attestation::AttestedPrograms;
pub(crate) use attestation::{
    AttestedFeedForward, AttestedHead, AttestedMixer, AttestedState, AttestedTarget,
    AttestedTargetBlock, AttestedVision,
};
use glue::GlueKernels;
use head::HeadKernels;
pub(crate) use import::ImportKernels;
use preparation::NativePreparationCache;
use qualification::QualificationView;
use target::TargetKernels;
pub(crate) use target::{
    AttentionKernels, DenseKernels, ReadoutKernels, RecurrentKernels, RoutedKernels,
};
use vision::VisionKernels;

use crate::{
    AttentionBinding, DenseBinding, EmbeddingBinding, ExecutionPath, FeaturesBinding,
    FeedForwardProgramSlot, ImportProgramSlot, MixerProgramSlot, ProgramPlan, ReadoutBinding,
    RecurrentBinding, RoutedBinding,
};
use magnitude_model_kernels::{
    copy_rows, gather_rows, head_logits_rows, import_dense, qwen_attention_attend,
    qwen_attention_normalize, qwen_attention_output, qwen_attention_prepare,
    qwen_attention_project, qwen_conditioning_overlay, qwen_dense_expand,
    qwen_dense_expand_demanded, qwen_dense_output, qwen_dense_output_demanded, qwen_embedding_rows,
    qwen_features_rows, qwen_head_rows, qwen_recurrent_mix, qwen_recurrent_normalize,
    qwen_recurrent_output, qwen_recurrent_prepare, qwen_recurrent_project, qwen_recurrent_scan,
    qwen_routed_expand, qwen_routed_logits, qwen_routed_normalize, qwen_routed_output,
    qwen_routed_select, qwen_selected_rows, qwen_vision_block, qwen_vision_feature_output,
    qwen_vision_merger, qwen_vision_stem, repack_weight, sample_rows, shape_rows,
};
use seismic::{BackendName, DType, Device, Element, NativeKernel, Tensor};
use std::{collections::HashMap, fmt};

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CatalogError {
    PathUnavailable {
        path: ExecutionPath,
        outcome: String,
    },
    Backend {
        path: ExecutionPath,
        backend: BackendName,
        outcome: String,
    },
    Preparation {
        path: ExecutionPath,
        entry: &'static str,
        bindings: String,
        outcome: String,
    },
    Qualification {
        path: ExecutionPath,
        entry: &'static str,
        bindings: String,
        outcome: String,
    },
}

impl fmt::Display for CatalogError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PathUnavailable { path, outcome } => {
                write!(formatter, "execution path {path} is unavailable: {outcome}")
            }
            Self::Backend {
                path,
                backend,
                outcome,
            } => write!(
                formatter,
                "execution path {path} cannot use {}: {outcome}",
                backend.as_str()
            ),
            Self::Preparation {
                path,
                entry,
                bindings,
                outcome,
            } => write!(
                formatter,
                "failed to prepare {entry} on {path} with {bindings}: {outcome}"
            ),
            Self::Qualification {
                path,
                entry,
                bindings,
                outcome,
            } => write!(
                formatter,
                "failed to qualify {entry} on {path} with {bindings}: {outcome}"
            ),
        }
    }
}

impl std::error::Error for CatalogError {}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StageCallError {
    pub path: ExecutionPath,
    pub entry: &'static str,
    pub bindings: String,
    pub outcome: String,
}

impl fmt::Display for StageCallError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "{} call to {} with {} failed: {}",
            self.path, self.entry, self.bindings, self.outcome
        )
    }
}

impl std::error::Error for StageCallError {}

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
        let q4k = Element::named("q4k").unwrap();
        let q5k = Element::named("q5k").unwrap();
        let q8 = Element::named("q8g32s").unwrap();
        let binding = RecurrentBinding {
            width: 128,
            norm: Element::bf16(),
            qkv: q5k,
            gate: q4k,
            alpha: q8,
            beta: q8,
            recurrent_norm: Element::bf16(),
            output: q5k,
            activation: Element::bf16(),
        };
        qwen_recurrent_project::native_for_device_with(
            &device,
            qwen_recurrent_project::Elements {
                QW: binding.qkv,
                GW: binding.gate,
                AW: binding.alpha,
                BW: binding.beta,
                A: binding.activation,
            }, &seismic::NativeSpecialization::new(),
        )
        .unwrap();
        qwen_recurrent_prepare::native_for_device_with(
            &device,
            qwen_recurrent_prepare::Elements {
                A: binding.activation,
                RN: binding.recurrent_norm,
            }, &seismic::NativeSpecialization::new(),
        )
        .unwrap();
        qwen_recurrent_scan::native_for_device_with(
            &device,
            qwen_recurrent_scan::Elements {
                A: binding.activation,
            }, &seismic::NativeSpecialization::new(),
        )
        .unwrap();
        qwen_recurrent_mix::native_for_device_with(
            &device,
            qwen_recurrent_mix::Elements {
                RN: binding.recurrent_norm,
                A: binding.activation,
            }, &seismic::NativeSpecialization::new(),
        )
        .unwrap();
        qwen_recurrent_output::native_for_device_with(
            &device,
            qwen_recurrent_output::Elements {
                OW: binding.output,
                A: binding.activation,
            }, &seismic::NativeSpecialization::new(),
        )
        .unwrap();
    }

    #[test]
    fn native_scan_preparation_rejects_width_above_thread_local_capacity() {
        let binding = RecurrentBinding {
            width: 256,
            norm: Element::bf16(),
            qkv: Element::bf16(),
            gate: Element::bf16(),
            alpha: Element::bf16(),
            beta: Element::bf16(),
            recurrent_norm: Element::bf16(),
            output: Element::bf16(),
            activation: Element::bf16(),
        };
        assert!(preparation::check_native_recurrent_scan_width(binding).is_ok());
        assert!(matches!(
            preparation::check_native_recurrent_scan_width(RecurrentBinding {
                width: 257,
                ..binding
            }),
            Err(CatalogError::Preparation {
                entry: "qwen_recurrent_scan",
                ..
            })
        ));
    }
}
