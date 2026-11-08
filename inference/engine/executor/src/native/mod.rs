mod attestation;
mod draft;
mod glue;
mod head;
mod import;
mod preparation;
mod qualification;
mod readout;
mod shared_head;
mod specialization;
mod target;
mod tuning;
mod vision;

pub(crate) use attestation::{AttestedImport, AttestedLanding};
pub use attestation::AttestedPrograms;
pub(crate) use attestation::{
    AttestedDraft, AttestedFeedForward, AttestedHead, AttestedHeadBlock, AttestedMixer,
    AttestedSharedHead, AttestedState, AttestedTarget, AttestedTargetBlock, AttestedVision,
    OutputScales, output_tails, reads_layer_scale,
};
use draft::DraftKernels;
pub(crate) use draft::{
    AttestedConvolved, AttestedCorrector, AttestedSelector, ConvolvedKernels, CorrectorKernels,
    DraftBlockKernels, MarkovKernels, SelectorKernels,
};
use glue::GlueKernels;
use head::HeadKernels;
pub(crate) use import::ImportKernels;
use preparation::NativePreparationCache;
pub(crate) use preparation::kernel_requests;
pub(crate) use qualification::admit_site;
use qualification::QualificationView;
pub use readout::ProjectionKernels;
use shared_head::SharedHeadKernels;
use target::TargetKernels;
pub(crate) use target::{
    AttentionHistoryKernels, AttentionKernels, BankCombine, BankMoves, BankRouteForm, BankRouteKernels, DevicePlacement,
    BankSite, SiteAdmission, SiteForm,
    DenseExpansionKernel, DenseKernels, ExpertBankKernels, ExpertKernels, LEADS,
    GeneralRoutedKernels, NGramKernels, ParallelKernels, PerLayerEntryKernels, PerLayerKernels,
    PostNormKernels, ProgressiveReadoutKernels, ReadoutHeadKernels, ReadoutKernels,
    RecurrentKernels, RecurrentStepKernels, RoutedDecodeKernels, RoutedKernels,
    SelectedPrefillKernel, SelectionKernels,
    ShortConvKernels, StateSpaceKernels, StreamsEntryKernels, StreamsKernels,
    SublayerOutput, TableConversion,
    TapKernels,
};
pub use tuning::pinned as pinned_tuning;
#[cfg(feature = "tuning-survey")]
pub use tuning::survey as tuning_survey;
pub(crate) use tuning::workload as tuning_workload;
pub use tuning::{
    MIN_SAMPLE_SECONDS, NumericalProfile, PinnedChoice, PointShape, ROTATION_LAYERS, Receipt,
    ResolvedNumericalProfile, SEARCH_SETTINGS, SEARCH_VERSION, STANDARD_TUNING_BUDGET,
    SearchProgress, TRACE_SCHEMA, TUNING_CONTEXTS, TUNING_ROWS, TracedWork, TunedEntry,
    TuningContext, TuningEvent, TuningLimits, TuningObserver, TuningOrigin, TuningRoom,
    TuningSettings, TuningTrace, TuningWeightSource, UnpreparedUnits, UnreportedTuning,
    ZeroTuningWeights, attention_points, row_points,
};
pub(crate) use vision::VisionKernels;

use crate::{
    AttentionBinding, ExecutionPath, FeedForwardProgramSlot, ImportProgramSlot, MixerProgramSlot,
    ProgramPlan, RecurrentBinding, RoutedBinding, SublayerTail, VisionEntry,
};
use magnitude_kernels::{
    attention_append_k8v4, attention_decode, attention_decode_k8v4, attention_output,
    attention_prefill, attention_prefill_k8v4, attention_project, conditioning_overlay, copy_rows,
    dense_expand, dense_output, dense_up, draft_rows, embedding_rows, feature_copy,
    gated_delta_chunk, gated_delta_project, gated_delta_project_convolved, gated_delta_step,
    gated_delta_step_convolved, import_dense, indexer_select, moe_tail, per_layer_gate,
    per_layer_inputs,
    post_norm_residual, project_rows, readout_exact_rows, readout_features_rows, readout_head_rows,
    readout_planes_rows, readout_project_rows, readout_refine_rows, readout_selected_rows,
    readout_top_rows, repack_weight, routed_combine, routed_down, routed_expand, routed_experts,
    routed_experts_up, routed_gate_up, routed_group, routed_output, routed_route,
    routed_route_shared, routed_scatter, routed_select, routed_up, sample_rows, shape_rows,
    short_conv_project, short_conv_rows, state_space_chunk, state_space_gate, state_space_step,
    vision_attention, vision_clamp, vision_gated_convolution, vision_linear, vision_norm,
    vision_patch_stem, vision_pool, vision_position,
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
    /// Not a failure of the model or the device: a load asked to report
    /// tuning units without a stored record
    /// ([`UnpreparedUnits::Report`](tuning::UnpreparedUnits)) met this one
    /// and stopped before placing anything.
    Unprepared {
        entry: &'static str,
        bindings: String,
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
            CatalogFailure::Unprepared { entry, bindings } => write!(
                formatter,
                "{entry} on {path} ({backend}) with {bindings} has no stored tuning record"
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
        let catalog = seismic::Devices::discover(&[]).unwrap();
        // Only a host with Metal runs it; a Metal executor it has must open.
        if !crate::test_support::available(&catalog, BackendName::Metal) {
            return;
        }
        let device = crate::test_support::open_backend(&catalog, BackendName::Metal).expect("an available Metal executor opens");
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
        let projections = heads
            .clone()
            .with_static("H", 2560)
            .with_static("PT", 1)
            .with_static("NM", 2560);
        let statics = heads.clone().with_static("C", binding.convolution_width);
        /// Prepares `entry` in its default configuration at `statics`, which
        /// its element bindings decide (`where` reads their forms).
        fn prepare<E: seismic::Entry>(
            device: &seismic::Device,
            entry: seismic::BoundEntry<E>,
            statics: &seismic::NativeSpecialization,
        ) {
            let specialization = entry
                .default_specialization(device, statics)
                .unwrap()
                .unwrap();
            entry.prepare(device, &specialization).unwrap();
        }
        prepare(
            &device,
            gated_delta_project::native_entry_with(gated_delta_project::Elements {
                NW: binding.norm,
                QW: binding.qkv,
                GW: binding.gate,
                AW: binding.alpha,
                BW: binding.beta,
                A: binding.activation,
            }),
            // Unbounded projections (no packing bound).
            &projections
                .clone()
                .with_static("QB", 0)
                .with_static("ZB", 0),
        );
        prepare(
            &device,
            gated_delta_step::native_entry_with(gated_delta_step::Elements {
                RN: binding.recurrent_norm,
                A: binding.activation,
            }),
            &statics,
        );
        prepare(
            &device,
            gated_delta_project_convolved::native_entry_with(
                gated_delta_project_convolved::Elements {
                    NW: binding.norm,
                    QW: binding.qkv,
                    GW: binding.gate,
                    AW: binding.alpha,
                    BW: binding.beta,
                    A: binding.activation,
                },
            ),
            &projections.with_static("C", binding.convolution_width),
        );
        prepare(
            &device,
            gated_delta_step_convolved::native_entry_with(gated_delta_step_convolved::Elements {
                RN: binding.recurrent_norm,
                A: binding.activation,
            }),
            &heads,
        );
        prepare(
            &device,
            gated_delta_chunk::native_entry_with(gated_delta_chunk::Elements {
                RN: binding.recurrent_norm,
                A: binding.activation,
            }),
            &statics,
        );
        let output = seismic::NativeSpecialization::new()
            .with_static("D", 2560)
            .with_static("SN", 1)
            .with_static("IN", 0)
            .with_static("Q", binding.value_heads)
            .with_static("W", binding.width)
            .with_static("OS", 0)
            .with_static("OB", 0);
        prepare(
            &device,
            attention_output::native_entry_with(attention_output::Elements {
                OW: binding.output,
                A: binding.activation,
            }),
            &output,
        );
    }
}
