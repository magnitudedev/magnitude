//! Device-free numerical planning.
//!
//! Component, weight, program, capability, and resource authorities are
//! separated below; this module only reexports their stable contracts.

mod capabilities;
mod components;
mod execution_plan;
mod programs;
mod resources;
mod weights;

pub use capabilities::{CapabilityPlan, PlannedMethod};
pub use components::{ArtifactComponent, ArtifactComponentKind, ComponentPlan, ComponentSelection};
pub use execution_plan::{
    ExecutionPlan, ExecutionPlanDraft, ExecutionPlanner, PlannedDevice, ResolvedPolicy,
};
pub use programs::{
    FeedForwardProgramSlot, HeadProgramPlan, ImportProgramSlot, MixerProgramSlot, ProgramPlan,
    StateProgramPlan, TargetBlockProgramSlot, TargetProgramPlan, VisionProgramPlan,
};
pub use resources::{
    NativeGraphCharge, ResourceBudget, ResourceBytes, ResourceLimits, ResourcePlan,
    ResourcePlanner, RetentionCapacityPlan, StateCapacityPlan, StateResourcePlan, StateStorePlan,
};
pub use weights::{
    resident_element, resident_layout, source_element, AttentionBinding, AttentionShape, DenseBinding,
    EmbeddingBinding,
    FeaturesBinding, HeadBinding, ModelLoadPlan, ReadoutBinding, RecurrentBinding, RoutedBinding,
    VisionBlockBinding, VisionMergerBinding, VisionPatchBinding, WeightPlan, WeightStorageIdentity,
};

use weights::{planned_element, weight_bytes_by_component};
#[cfg(test)]
use weights::{resident_dtype, validate_unique_roles};

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use magnitude_artifacts::{
        gguf::{Encoding, TensorDescriptor},
        ArtifactIdentity, ComponentManifest, PackageIdentity, PackageManifest,
    };
    use magnitude_model_contracts::{
        ActivationDType, AttentionGeometry, AttentionWeights, BlockGeometry, BlockWeights,
        DenseFeedForwardWeights, FeedForwardGeometry, FeedForwardWeights, InputSemantics,
        MixerWeights, ModelDefinition, RotarySemantics, TextCoordinateSemantics, WeightDescriptor,
        WeightKind, WeightRole, WeightScope,
    };
    use magnitude_model_state::KvCodec;
    use seismic::{DType, Element};

    fn weight(role: WeightRole, name: &str) -> WeightPlan {
        WeightPlan {
            role,
            component: ArtifactComponent {
                kind: ArtifactComponentKind::Target,
                identity: ArtifactIdentity([7; 32]),
            },
            source: Element::f16(),
            resident: Element::f16(),
            shape: vec![1],
            descriptor: WeightDescriptor {
                name: name.into(),
                shape: vec![1],
            },
            source_bytes: 2,
            resident_bytes: 2,
        }
    }

    #[test]
    fn semantic_roles_are_unique_within_an_artifact_component() {
        let role = WeightRole {
            scope: WeightScope::VisionBlock(0),
            kind: WeightKind::InputNormWeight,
        };
        let plans = [weight(role, "a"), weight(role, "b")];
        assert!(validate_unique_roles(plans.iter()).is_err());
    }

    #[test]
    fn semantic_roles_select_the_exact_kernel_resident_dtype() {
        let role = |kind| WeightRole {
            scope: WeightScope::TargetBlock(0),
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
            WeightKind::FusedQkvWeight,
            WeightKind::MergerOutput,
        ] {
            assert_eq!(resident_dtype(role(kind), DType::BF16), DType::BF16);
        }
    }

    fn descriptor(name: &str, shape: &[u64]) -> WeightDescriptor {
        WeightDescriptor {
            name: name.into(),
            shape: shape.to_vec(),
        }
    }

    /// The smallest geometry every native kernel admits: hidden and
    /// feed-forward widths a multiple of 64 (CUDA K1 k-blocks), an attention
    /// head width a multiple of 32 (attention lanes), one head.
    pub(crate) fn fixture_definition() -> ModelDefinition {
        const HIDDEN: u64 = 128;
        const WIDTH: u64 = 64;
        const FEATURES: u64 = 128;
        const VOCABULARY: u64 = 256;
        let attention = || AttentionWeights {
            query_gate: descriptor("qg", &[2 * WIDTH, HIDDEN]),
            key: descriptor("k", &[WIDTH, HIDDEN]),
            value: descriptor("v", &[WIDTH, HIDDEN]),
            query_norm: descriptor("qn", &[WIDTH]),
            key_norm: descriptor("kn", &[WIDTH]),
            output: descriptor("o", &[HIDDEN, WIDTH]),
        };
        let dense = || DenseFeedForwardWeights {
            gate: descriptor("gate", &[FEATURES, HIDDEN]),
            up: descriptor("up", &[FEATURES, HIDDEN]),
            down: descriptor("down", &[HIDDEN, FEATURES]),
        };
        ModelDefinition {
            family: magnitude_model_contracts::FamilyId("resource-fixture".into()),
            artifact_identity: PackageIdentity {
                target: ArtifactIdentity([7; 32]),
                projector: None,
            },
            inputs: InputSemantics {
                text_coordinates: TextCoordinateSemantics::ReplicatedPosition,
                coordinate_axes: 1,
            },
            geometry: magnitude_model_contracts::DecoderGeometry {
                activation_dtype: ActivationDType::BF16,
                hidden: HIDDEN,
                vocabulary: VOCABULARY,
                context_limit: 128,
                epsilon: 1e-6,
                blocks: vec![BlockGeometry {
                    mixer: magnitude_model_contracts::MixerGeometry::Attention(AttentionGeometry {
                        heads: 1,
                        kv_heads: 1,
                        width: WIDTH,
                        rotary: RotarySemantics::Interleaved {
                            width: WIDTH / 2,
                            base: 10_000.0,
                            sections: vec![WIDTH / 4],
                            axis_pattern: vec![0],
                        },
                    }),
                    feedforward: FeedForwardGeometry::Dense { intermediate: FEATURES },
                }],
            },
            embedding: descriptor("embedding", &[VOCABULARY, HIDDEN]),
            blocks: vec![BlockWeights {
                input_norm: descriptor("input_norm", &[HIDDEN]),
                mixer: MixerWeights::Attention(Box::new(attention())),
                feedforward_norm: descriptor("ffn_norm", &[HIDDEN]),
                feedforward: FeedForwardWeights::Dense(Box::new(dense())),
            }],
            output_norm: descriptor("output_norm", &[HIDDEN]),
            output: descriptor("output", &[VOCABULARY, HIDDEN]),
            head: None,
            vision: None,
        }
    }

    pub(crate) fn fixture_manifest(definition: &ModelDefinition) -> PackageManifest {
        let block = &definition.blocks[0];
        let MixerWeights::Attention(attention) = &block.mixer else {
            unreachable!()
        };
        let FeedForwardWeights::Dense(dense) = &block.feedforward else {
            unreachable!()
        };
        let descriptors = [
            &definition.embedding,
            &block.input_norm,
            &attention.query_gate,
            &attention.key,
            &attention.value,
            &attention.query_norm,
            &attention.key_norm,
            &attention.output,
            &block.feedforward_norm,
            &dense.gate,
            &dense.up,
            &dense.down,
            &definition.output_norm,
            &definition.output,
        ];
        let mut offset = 0;
        let tensors = descriptors
            .into_iter()
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
                path: "planning-fixture.gguf".into(),
                identity: definition.artifact_identity.target,
                size: offset,
                tensors,
            },
            projector: None,
        }
    }

    /// Build actual checked Seismic graph families. The engine may choose how
    /// many slots to reserve, but no test supplies intermediate tensor shapes.
    fn prepared_resource_plan() -> Option<ResourcePlan> {
        let catalog = seismic::DeviceCatalog::discover().ok()?;
        let selected =
            crate::platform::select_device(
                &catalog,
                crate::ExecutionPath::Native,
                crate::platform::DeviceRequest::Automatic,
            ).ok()?;
        let device = catalog
            .open(catalog.resolve(selected.info.selector).unwrap())
            .unwrap();
        let definition = fixture_definition();
        let manifest = fixture_manifest(&definition);
        let limits = ResourceLimits {
            active_requests: 2,
            in_flight_requests: 2,
            branch_checkpoints: 0,
            max_batch_rows: 2,
            max_projected_rows: 2,
            max_images_per_request: magnitude_artifacts::MAX_IMAGES_PER_REQUEST,
        };
        let budget = ResourceBudget {
            storage_bytes: selected.assessment_capacity_bytes.min(512 * 1024 * 1024),
            retention_bytes: 64 * 1024,
            safety_reserve_bytes: 16 * 1024 * 1024,
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
            budget,
        )
        .unwrap();
        let state =
            ResourcePlanner::state_plan(&definition, draft.load(), KvCodec::Dense, limits, budget)
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
            .prepare_target_graphs(&device, draft.load(), &definition.geometry, &state, limits)
            .unwrap();
        let readout = programs
            .prepare_target_readout_graphs(&device, draft.load(), &definition.geometry, limits)
            .unwrap();
        programs
            .prepare_auxiliary_graphs(
                &device,
                draft.load(),
                &definition,
                state.target_state(),
                state.head_state(),
                limits,
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
    fn sealed_graph_footprints_charge_exact_concurrency() {
        let Some(plan) = prepared_resource_plan() else {
            return;
        };
        let target = plan.target_graph();
        let readout = plan.target_readout_graph();
        let state = plan.state_graph();
        assert_eq!((target.workspace_slots, target.output_slots), (2, 4));
        assert_eq!((readout.workspace_slots, readout.output_slots), (2, 4));
        assert_eq!((state.workspace_slots, state.output_slots), (2, 2));
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
        below.storage_bytes = plan.startup_peak_bytes() - 1;
        assert!(below.validate().unwrap_err().contains("startup peak"));
        let mut exact = plan;
        exact.storage_bytes = exact.startup_peak_bytes();
        let admitted_bytes = exact.storage_bytes;
        assert_eq!(
            exact.validate().unwrap().startup_peak_bytes(),
            admitted_bytes
        );
    }
}
