//! Device-free numerical planning.
//!
//! Component, weight, program, capability, and resource authorities are
//! separated below; this module only reexports their stable contracts.

mod assessment;
mod capabilities;
mod components;
mod execution_plan;
mod programs;
mod resources;
mod weights;

pub use assessment::{
    AssessmentFit, AssessmentFitVerdict, AssessmentGraphResourceBounds, AssessmentHeaderBounds,
    AssessmentMemoryBounds, AssessmentMemoryCharge, AssessmentMemoryTerms,
};
pub use capabilities::{CapabilityPlan, PlannedMethod, MAX_DRAFT_PROPOSALS};
pub use components::{ArtifactComponent, ArtifactComponentKind, ComponentPlan, ComponentSelection};
pub use execution_plan::{
    ExecutionPlan, ExecutionPlanDraft, ExecutionPlanner, PlannedDevice, ResolvedPolicy,
};
pub use programs::{
    Dflash2Binding, DraftBlockBinding, DraftProgramPlan, FeedForwardProgramSlot, HeadProgramPlan,
    ImportProgramSlot, MarkovBinding, MixerProgramSlot, ProgramPlan, SelectorBinding,
    StateProgramPlan, TapProgramPlan, TargetBlockProgramSlot, TargetProgramPlan, VisionProgramPlan,
};
pub use resources::{
    GraphSlots, HistoryStorePlan, LayerHistory, NativeGraphCharge, ResourceBytes,
    ResourceCapacity, ResourceLimits, ResourcePlan, ResourcePlanner, StartupSlots,
    StateCapacityPlan, StateResourcePlan, StateStorePlan,
};
pub use weights::{
    resident_element, resident_layout, source_element, AttentionBinding, AttentionShape,
    DenseBinding, DenseBranchBinding, EmbeddingBinding, FeaturesBinding, HeadBinding,
    HostTablePlan, ModelLoadPlan, ParallelBinding, PerLayerBinding, PerLayerEntryBinding, ReadoutBinding,
    DenseScales, RecurrentBinding, RoutedBinding, ScalableWeight, SublayerTail, WeightPlan, WeightScalePlan,
    WeightStorageIdentity,
};

pub(crate) use weights::activation_dtype;
use weights::{
    planned_element, planned_scalable, source_import_peak_bytes, weight_bytes_by_component,
};
#[cfg(test)]
use weights::{resident_dtype, validate_unique_roles};

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use magnitude_artifacts::{
        gguf::{Encoding, TensorDescriptor},
        ArtifactIdentity, ComponentFile, ComponentManifest, PackageIdentity, PackageManifest,
    };
    use magnitude_family_contracts::{
        ActivationDType, ActivationFunction, Attention, AttentionGate, Block, Decoder, DenseFfn,
        EmbeddingScale, EntryForm, ExitForm, ExitNorm, FeedForwardUp, GateFunction, HeadNorm,
        HistoryDomain, HistoryReads, InputNorm, InputSemantics, KeyValue, MediaRowAttention,
        ModelDefinition, Operator, OutputForm, ResidualForm, RmsNorm, Rotary, SublayerIndex,
        Sublayer, ValueNorm, ValueSource, WeightDescriptor, WeightKind,
        WeightRole, WeightScope,
    };
    use magnitude_state::KvCodec;
    use seismic::{DType, Element};

    fn weight(role: WeightRole, name: &str) -> WeightPlan {
        WeightPlan {
            role,
            component: ArtifactComponent {
                kind: ArtifactComponentKind::Target,
                identity: ArtifactIdentity([7; 32]),
            },
            source: Element::f16(),
            upload: Element::f16(),
            resident: Element::f16(),
            shape: vec![1],
            descriptor: WeightDescriptor::stored(name, [1]),
            source_bytes: 2,
            resident_bytes: 2,
            scale: None,
        }
    }

    #[test]
    fn semantic_roles_are_unique_within_an_artifact_component() {
        let role = WeightRole {
            scope: WeightScope::VisionBlock(0),
            kind: WeightKind::Vision(magnitude_family_contracts::VisionWeight::PatchBias),
        };
        let plans = [weight(role, "a"), weight(role, "b")];
        assert!(validate_unique_roles(plans.iter()).is_err());
    }

    #[test]
    fn semantic_roles_select_the_exact_kernel_resident_dtype() {
        let role = |kind| WeightRole {
            scope: WeightScope::TargetSublayer(SublayerIndex {
                block: 0,
                sublayer: 0,
            }),
            kind,
        };
        for kind in [
            WeightKind::QueryNorm,
            WeightKind::KeyNorm,
            WeightKind::RecurrentConvolution,
            WeightKind::RecurrentDecay,
            WeightKind::RecurrentTimeBias,
            WeightKind::SharedRouter,
        ] {
            assert_eq!(resident_dtype(role(kind), DType::BF16), DType::F32);
        }
        for kind in [
            WeightKind::QueryGate,
            WeightKind::RecurrentAlpha,
            WeightKind::Router,
            WeightKind::Vision(magnitude_family_contracts::VisionWeight::Linear {
                site: magnitude_family_contracts::VisionLinearSite::Merger,
                part: magnitude_family_contracts::LinearPart::Weight,
            }),
        ] {
            assert_eq!(resident_dtype(role(kind), DType::BF16), DType::BF16);
        }
    }

    fn descriptor(name: &str, shape: &[u64]) -> WeightDescriptor {
        WeightDescriptor::stored(name, shape)
    }

    /// The smallest geometry every native kernel admits: hidden and
    /// feed-forward widths a multiple of 64 (CUDA K1 k-blocks), an attention
    /// head width a multiple of 32 (attention lanes), and two query heads
    /// sharing one key-value head (Vulkan prefill needs a query group of at
    /// least two).
    pub(crate) fn fixture_definition() -> ModelDefinition {
        const HIDDEN: u64 = 128;
        const WIDTH: u64 = 64;
        const FEATURES: u64 = 128;
        const VOCABULARY: u64 = 256;
        let rms = |name: &str, width| RmsNorm {
            weight: descriptor(name, &[width]),
            epsilon: 1e-6,
        };
        let attention = Attention {
            heads: 2,
            kv_heads: 1,
            width: WIDTH,
            query: descriptor("qg", &[2 * 2 * WIDTH, HIDDEN]),
            gate: AttentionGate::Interleaved {
                function: GateFunction::Sigmoid,
            },
            query_norm: HeadNorm::Rms(rms("qn", WIDTH)),
            key_value: KeyValue::Owned {
                key: descriptor("k", &[WIDTH, HIDDEN]),
                value: ValueSource::Projected(descriptor("v", &[WIDTH, HIDDEN])),
                key_norm: HeadNorm::Rms(rms("kn", WIDTH)),
                value_norm: ValueNorm::None,
                domain: HistoryDomain::Token,
            },
            rotary: Rotary::Interleaved {
                width: WIDTH / 2,
                base: 10_000.0,
                sections: vec![WIDTH / 4],
                axis_pattern: vec![0],
            },
            scale: 1.0 / (WIDTH as f64).sqrt(),
            reads: HistoryReads::Visible,
            media_rows: MediaRowAttention::Causal,
            output: descriptor("o", &[HIDDEN, 2 * WIDTH]),
        };
        let dense = DenseFfn {
            intermediate: FEATURES,
            up: FeedForwardUp::Gated {
                activation: ActivationFunction::Silu,
                gate: descriptor("gate", &[FEATURES, HIDDEN]),
                up: descriptor("up", &[FEATURES, HIDDEN]),
            },
            down: descriptor("down", &[HIDDEN, FEATURES]),
        };
        ModelDefinition {
            family: magnitude_family_contracts::FamilyId("resource-fixture".into()),
            artifact_identity: PackageIdentity {
                target: ArtifactIdentity([7; 32]),
                projector: None,
            },
            inputs: InputSemantics {
                coordinate_axes: 1,
            },
            decoder: Decoder {
                activation_dtype: ActivationDType::BF16,
                hidden: HIDDEN,
                vocabulary: VOCABULARY,
                context_limit: 128,
                residual: ResidualForm::Single,
                entry: EntryForm {
                    embedding: descriptor("embedding", &[VOCABULARY, HIDDEN]),
                    scale: EmbeddingScale::Unit,
                    norm: None,
                    per_layer: None,
                    hash_routing: None,
                },
                blocks: vec![Block {
                    sublayers: vec![
                        Sublayer {
                            input: InputNorm::Rms(rms("input_norm", HIDDEN)),
                            op: Operator::Attention(Box::new(attention)),
                            output: OutputForm::Residual,
                        },
                        Sublayer {
                            input: InputNorm::Rms(rms("ffn_norm", HIDDEN)),
                            op: Operator::DenseFfn(Box::new(dense)),
                            output: OutputForm::Residual,
                        },
                    ],
                }],
                exit: ExitForm {
                    norm: ExitNorm::Rms(rms("output_norm", HIDDEN)),
                    output: descriptor("output", &[VOCABULARY, HIDDEN]),
                    softcap: None,
                },
            },
            head: None,
            vision: None,
            draft: None,
        }
    }

    pub(crate) fn fixture_manifest(definition: &ModelDefinition) -> PackageManifest {
        let decoder = &definition.decoder;
        let head = definition.head.as_ref().map_or_else(Vec::new, |head| {
            crate::operators::head_weights(head).expect("fixture head block indices fit u32")
        });
        let descriptors = std::iter::once(&decoder.entry.embedding)
            .chain(
                crate::operators::decoder_weights(decoder)
                    .into_iter()
                    .chain(head)
                    .map(|(_, descriptor)| descriptor),
            )
            .chain([decoder.exit.norm.weight(), &decoder.exit.output]);
        let mut offset = 0;
        let tensors = descriptors
            .map(|descriptor| {
                let nbytes = descriptor.shape.iter().product::<u64>() * 2;
                let tensor = TensorDescriptor {
                    name: descriptor.name.clone(),
                    shape: descriptor.shape.clone(),
                    encoding: Encoding::F16,
                    offset,
                    nbytes,
                };
                offset += nbytes;
                tensor
            })
            .collect();
        PackageManifest {
            identity: definition.artifact_identity,
            target: ComponentManifest {
                files: vec![ComponentFile {
                    path: "planning-fixture.gguf".into(),
                    size: offset,
                }],
                identity: definition.artifact_identity.target,
                tensors,
            },
            projector: None,
            draft: None,
        }
    }

    /// Build actual checked Seismic graph families. The engine may choose how
    /// many slots to reserve, but no test supplies intermediate tensor shapes.
    fn prepared_resource_plan() -> Option<ResourcePlan> {
        let catalog = seismic::DeviceCatalog::discover().ok()?;
        let selected = crate::platform::select_device(
            &catalog,
            crate::ExecutionPath::Native,
            crate::platform::DeviceRequest::Automatic,
            &crate::platform::MemoryReserves::standard(),
        )
        .ok()?;
        let device = catalog
            .open(catalog.resolve(selected.info.selector).unwrap())
            .unwrap();
        let definition = fixture_definition();
        let manifest = fixture_manifest(&definition);
        let limits = ResourceLimits {
            max_launch_rows: 2,
            max_launch_slots: 2,
            max_projected_rows: 2,
            max_images_per_request: magnitude_artifacts::MAX_IMAGES_PER_REQUEST,
            lookahead: false,
        };
        let capacity_bytes = ResourceCapacity {
            domain_bytes: selected.assessment_capacity_bytes.min(512 * 1024 * 1024),
        };
        let draft = ExecutionPlanner::prepare(
            &selected,
            &manifest,
            &definition,
            ComponentSelection {
                head: false,
                vision: false,
            },
            crate::ExecutionPath::Native,
            PlannedMethod::Plain,
            KvCodec::Dense,
            limits,
        )
        .unwrap();
        let state = ResourcePlanner::state_plan(
            &definition,
            draft.load(),
            draft.policy().method(),
            KvCodec::Dense,
            limits,
            capacity_bytes,
        )
        .unwrap();
        let mut programs = crate::AttestedPrograms::prepare_draft(
            &draft,
            &device,
            crate::TuningContext {
                definition: &definition,
                weights: &crate::ZeroTuningWeights,
                observer: &crate::UnreportedTuning,
                cache: None,
            },
        )
        .unwrap();
        let target = programs
            .prepare_target_graphs(
                &device,
                draft.load(),
                &definition.decoder,
                &state,
                draft.programs().target(),
                limits,
            )
            .unwrap();
        let readout = programs
            .prepare_target_readout_graphs(&device, draft.load(), &definition.decoder, limits)
            .unwrap();
        let assessed_target = crate::programs::native_target_graph::checked_target_family_storage(
            device.backend(),
            draft.load(),
            &definition.decoder,
            &state,
            draft.programs().target(),
            limits,
        )
        .unwrap();
        assert_eq!(
            target.family().workspace_bytes(),
            assessed_target.storage.workspace
        );
        assert_eq!(
            target.family().output_bytes(),
            assessed_target.storage.output
        );
        assert_eq!(
            target.family().upload_bytes(),
            assessed_target.storage.upload
        );
        let assessed_readout = crate::programs::graph::readout::checked_readout_family_storage(
            device.backend(),
            draft.load(),
            &definition.decoder,
            limits,
        )
        .unwrap();
        assert_eq!(
            readout.family().workspace_bytes(),
            assessed_readout.workspace
        );
        assert_eq!(readout.family().output_bytes(), assessed_readout.output);
        assert_eq!(readout.family().upload_bytes(), assessed_readout.upload);
        programs
            .prepare_auxiliary_graphs(
                &device,
                draft.load(),
                &definition,
                draft.programs().vision(),
                state.target_state(),
                state.head_state(),
                limits,
                0,
            )
            .unwrap();
        let copy = programs.state_graphs().unwrap();
        let plan =
            ResourcePlanner::plan_with_state(state, &target, &readout, None, None, copy).unwrap();
        assert_eq!(
            plan.target_graph().workspace_bytes,
            target.workspace_bytes()
        );
        assert_eq!(plan.target_graph().output_bytes, target.output_bytes());
        assert_eq!(
            plan.target_readout_graph().workspace_bytes,
            readout.workspace_bytes()
        );
        assert_eq!(
            plan.target_readout_graph().output_bytes,
            readout.output_bytes()
        );
        assert_eq!(
            plan.state_graph().workspace_bytes,
            copy.workspace_bytes_max()
        );
        assert_eq!(plan.state_graph().output_bytes, copy.output_bytes_max());
        Some(plan)
    }

    #[test]
    fn sealed_graph_footprints_charge_one_request_at_startup() {
        let Some(plan) = prepared_resource_plan() else {
            return;
        };
        let target = plan.target_graph();
        let readout = plan.target_readout_graph();
        let state = plan.state_graph();
        // One launch per lane in flight, whatever the request count; one
        // request's retained readout beside it. More outputs are elastic.
        assert_eq!((target.workspace_slots, target.output_slots), (1, 2));
        assert_eq!((readout.workspace_slots, readout.output_slots), (1, 2));
        assert_eq!((state.workspace_slots, state.output_slots), (1, 1));
        assert!(target.workspace_bytes > 0);
        assert!(target.output_bytes > 0);
        assert!(readout.output_bytes > 0);
        assert_eq!(
            plan.bytes().scratch,
            target.committed_bytes + readout.committed_bytes + state.committed_bytes,
        );
        assert_eq!(plan.steady_committed_bytes(), plan.bytes().total().unwrap());
        assert_eq!(
            plan.startup_peak_bytes(),
            plan.steady_committed_bytes() + plan.qualification_peak_bytes,
        );
    }

    #[test]
    fn sealed_graph_charge_is_admitted_only_when_startup_peak_fits() {
        let Some(plan) = prepared_resource_plan() else {
            return;
        };
        let mut below = plan.clone();
        below.domain_capacity_bytes = plan.startup_peak_bytes() - 1;
        assert!(below.validate().unwrap_err().contains("startup peak"));
        let mut exact = plan;
        exact.domain_capacity_bytes = exact.startup_peak_bytes();
        let admitted_bytes = exact.domain_capacity_bytes;
        assert_eq!(
            exact.validate().unwrap().startup_peak_bytes(),
            admitted_bytes
        );
    }
}
