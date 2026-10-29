//! Device-free numerical planning.
//!
//! Component, weight, program, capability, and resource authorities are
//! separated below; this module only reexports their stable contracts.

mod assessment;
mod capabilities;
mod components;
mod execution_plan;
mod imports;
pub(crate) use imports::{ImportBacking, ImportPlan};
mod programs;
mod resources;
mod weights;

pub use assessment::{
    AssessmentFit, AssessmentFitVerdict, AssessmentGraphResourceBounds, AssessmentHeaderBounds,
    AssessmentMemoryBounds, AssessmentMemoryCharge, AssessmentMemoryTerms, CONVERSATION_CHECKPOINTS,
    DeviceProcessBytes, HostProcessBytes, SealedGraphHostBytes,
};
pub use capabilities::{
    CapabilityPlan, DraftWidths, MAX_DRAFT_PROPOSALS, PlannedMethod, draft_ladder,
};
pub use components::{ArtifactComponent, ArtifactComponentKind, ComponentPlan, ComponentSelection};
pub use execution_plan::{
    BackendPlan, ExecutionPlan, ExecutionPlanDraft, ExecutionPlanner, PlannedDevice, ResolvedPolicy,
};
pub use programs::{
    ConvolutionBinding, ConvolvedLayerBinding, CorrectorBinding, CorrectorLayerBinding,
    DraftBlockBinding, DraftProgramPlan, FeedForwardProgramSlot, HeadProgramPlan,
    ImportProgramSlot, MarkovBinding, MixerProgramSlot, ProgramPlan, SelectorBinding,
    SharedHeadProgramPlan, StateProgramPlan, TapProgramPlan, TargetBlockProgramSlot,
    TargetProgramPlan, VisionProgramPlan,
};
pub use resources::{
    CPU_IMAGE_CELLS, GraphSlots, HistoryStorePlan, LayerHistory, MAX_IMAGE_CELLS, NativeGraphCharge, ResourceBytes,
    ResourceCapacity, ResourceLimits, ResourcePlan, ResourcePlanner, StartupSlots,
    StateCapacityPlan, StateResourcePlan, StateStorePlan, TensorOperations, image_cell_limit,
    max_image_cells, reads_decoded_history,
};
pub use weights::{
    AttentionBinding, AttentionShape, DenseBinding, DenseBranchBinding, DenseScales,
    DirectoryForm, EmbeddingBinding, FeaturesBinding, HeadBinding, HostPeak, HostTablePlan,
    ModelLoadPlan,
    NGramBinding, ParallelBinding, PerLayerBinding, PerLayerEntryBinding, ReadoutBinding,
    ReadoutClassChoice, ReadoutHead, RecurrentBinding, RoutedBinding, ScalableWeight,
    SelectionBinding, SelectionShape, SublayerTail, WeightPlan, WeightScalePlan, WeightStorageIdentity,
    resident_element, resident_elements, resident_layout, source_element,
};

pub(crate) use weights::{activation_dtype, elements, packing_residue, packing_run, value_packets};
use weights::{
    planned_element, planned_readout, planned_scalable,
    weight_bytes_by_component,
};
#[cfg(test)]
use weights::{resident_dtype, validate_unique_roles};

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use magnitude_artifacts::{
        ArtifactIdentity, ComponentFile, ComponentManifest, PackageIdentity, PackageManifest,
        gguf::{Encoding, TensorDescriptor},
    };
    use magnitude_family_contracts::{
        ActivationDType, ActivationFunction, Attention, AttentionGate, Block, Decoder, DenseFfn,
        EmbeddingScale, EntryForm, ExitForm, ExitNorm, FeedForwardUp, GateFunction, HeadNorm,
        HistoryDomain, HistoryReads, InputNorm, InputSemantics, KeyValue, MediaRowAttention,
        ModelDefinition, Operator, OutputForm, ResidualForm, RmsNorm, Rotary, Sublayer,
        SublayerIndex, ValueNorm, ValueSource, WeightDescriptor, WeightKind, WeightRole,
        WeightScope,
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
            bound: 0,
            streamed: None,
        }
    }

    #[test]
    fn streamed_extents_do_not_enlarge_the_import_schedule() {
        let role = WeightRole {
            scope: WeightScope::Target,
            kind: WeightKind::Embedding,
        };
        let mut table = weight(role, "streamed");
        table.shape = vec![1, 1];
        table.descriptor = WeightDescriptor::stored("streamed", [1_000_000, 1]);
        table.streamed = Some(super::weights::StreamedRows {
            table_rows: 1_000_000,
            staging_rows: 1,
            landing_rows: 0,
            directory: DirectoryForm::Dense,
            fill: super::weights::TableFill::Imported,
        });
        let ordinary = weight(
            WeightRole {
                scope: WeightScope::Target,
                kind: WeightKind::OutputNorm,
            },
            "norm",
        );
        let imports = ImportPlan::new([&table, &ordinary]);
        // The table's source is one row. Its million stored rows must not
        // permit the two adjacent ordinary sources to share a larger window.
        assert_eq!(
            imports
                .batch([
                    (0, ImportBacking::Mapped(0..2)),
                    (0, ImportBacking::Mapped(2..4))
                ])
                .unwrap()
                .len,
            1
        );
        assert_eq!(
            imports.peak_bytes().unwrap(),
            ImportPlan::new([&ordinary]).peak_bytes().unwrap()
        );
        // Stored tables upload nothing, regardless of their stored extent.
        table.source_bytes = 0;
        table.streamed.as_mut().unwrap().fill = super::weights::TableFill::Stored;
        assert_eq!(ImportPlan::new([&table]).peak_bytes().unwrap(), 0);
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
            sinks: None,
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
            inputs: InputSemantics { coordinate_axes: 1 },
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
            .chain(
                crate::operators::exit_weights(decoder)
                    .into_iter()
                    .map(|(_, descriptor)| descriptor),
            )
            .chain([&decoder.exit.output]);
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

    #[test]
    fn plain_weight_plan_ignores_an_unadmitted_optional_draft() {
        let definition = fixture_definition();
        let mut manifest = fixture_manifest(&definition);
        let selection = ComponentSelection {
            head: false,
            vision: false,
        };
        let plain =
            ModelLoadPlan::derive(&manifest, &definition, selection, seismic::Layout::Rows32)
                .unwrap();
        manifest.draft = Some(ComponentManifest {
            files: Vec::new(),
            identity: ArtifactIdentity([9; 32]),
            tensors: Vec::new(),
        });
        let fallback =
            ModelLoadPlan::derive(&manifest, &definition, selection, seismic::Layout::Rows32)
                .unwrap();
        assert_eq!(fallback, plain);
    }

    /// The kernel inventory, derived without a device, names exactly the
    /// kernels a load prepares on the selected device.
    #[test]
    fn kernel_inventory_names_every_kernel_a_load_prepares() {
        let Ok(catalog) = seismic::Devices::discover(&[]) else {
            return;
        };
        let Ok(selected) = crate::platform::select_device(
            &catalog,
            crate::ExecutionPath::Native,
            crate::platform::DeviceRequest::Automatic,
            &crate::platform::MemoryReserves::standard(),
        ) else {
            return;
        };
        let device = catalog
            .open(catalog.resolve(selected.info.selector()).unwrap())
            .unwrap();
        let definition = fixture_definition();
        let manifest = fixture_manifest(&definition);
        let limits = ResourceLimits {
            max_launch_rows: 2,
            max_launch_slots: 2,
            max_selected_rows: 2,
            max_drafting_slots: 2,
            exported_logits_rows: 0,
            max_images_per_request: magnitude_artifacts::MAX_IMAGES_PER_REQUEST,
            max_image_cells: 0,
            lookahead: false,
        };
        let selection = ComponentSelection {
            head: false,
            vision: false,
        };
        let draft = ExecutionPlanner::prepare(
            &selected,
            &manifest,
            &definition,
            selection,
            crate::ExecutionPath::Native,
            PlannedMethod::Plain,
            KvCodec::Dense,
            limits,
        )
        .unwrap();
        let (_, recorded) = seismic::record_kernel_requests(|| {
            crate::AttestedPrograms::prepare_draft(
                &draft,
                &device,
                crate::TuningContext {
                    definition: &definition,
                    weights: &crate::ZeroTuningWeights,
                    observer: &crate::UnreportedTuning,
                    cache: None,
                    numerical_profile: crate::ResolvedNumericalProfile::standard(),
                    settings: crate::TuningSettings::STANDARD,
                },
                crate::UnpreparedUnits::ServeDefaults,
            )
            .unwrap()
        });
        let plan = ExecutionPlanner::backend_plan(
            device.backend(),
            &manifest,
            &definition,
            selection,
            crate::ExecutionPath::Native,
            PlannedMethod::Plain,
            KvCodec::Dense,
            limits,
        )
        .unwrap();
        let inventory = crate::kernel_inventory(&definition, &plan)
            .unwrap()
            .into_iter()
            .collect::<std::collections::HashSet<_>>();
        let prepared = recorded
            .into_iter()
            .collect::<std::collections::HashSet<_>>();
        assert!(!prepared.is_empty());
        let named = |requests: std::collections::hash_set::Difference<'_, _, _>| {
            requests
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join("\n")
        };
        assert!(
            inventory == prepared,
            "prepared kernels the inventory omits:\n{}\nlisted kernels the load does not prepare:\n{}",
            named(prepared.difference(&inventory)),
            named(inventory.difference(&prepared)),
        );
    }

    /// A search is held to the room it is given, what the platform lets the
    /// engine claim, and not to what the load is planned to hold.
    #[test]
    fn a_search_is_held_to_the_room_it_is_given() {
        struct Ceiling(std::cell::Cell<Option<Option<u64>>>);
        impl crate::TuningObserver for Ceiling {
            fn event(&self, event: &crate::TuningEvent) {
                if let crate::TuningEvent::Budget(budget) = event {
                    self.0.set(Some(budget.memory.ceiling_bytes));
                }
            }
        }
        let Ok(catalog) = seismic::Devices::discover(&[]) else {
            return;
        };
        let Ok(selected) = crate::platform::select_device(
            &catalog,
            crate::ExecutionPath::Native,
            crate::platform::DeviceRequest::Automatic,
            &crate::platform::MemoryReserves::standard(),
        ) else {
            return;
        };
        let device = catalog
            .open(catalog.resolve(selected.info.selector()).unwrap())
            .unwrap();
        let definition = fixture_definition();
        let manifest = fixture_manifest(&definition);
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
            ResourceLimits {
                max_launch_rows: 2,
                max_launch_slots: 2,
                max_selected_rows: 2,
                max_drafting_slots: 2,
                exported_logits_rows: 0,
                max_images_per_request: magnitude_artifacts::MAX_IMAGES_PER_REQUEST,
                max_image_cells: 0,
                lookahead: false,
            },
        )
        .unwrap();
        let observer = Ceiling(std::cell::Cell::new(None));
        let room = crate::TuningRoom {
            device_bytes: 1 << 40,
            host_bytes: u64::MAX,
            host_shared: true,
        };
        crate::AttestedPrograms::search_draft(
            &draft,
            &device,
            crate::TuningContext {
                definition: &definition,
                weights: &crate::ZeroTuningWeights,
                observer: &observer,
                cache: None,
                numerical_profile: crate::ResolvedNumericalProfile::standard(),
                // The ceiling is set with the mandatory phase: no optional
                // work is needed to read it.
                settings: crate::TuningSettings {
                    budget: std::time::Duration::ZERO,
                    fresh: false,
                },
            },
            Some(room),
        )
        .unwrap();
        let ceiling = observer
            .0
            .get()
            .expect("a search reports what became of its budget");
        assert_eq!(ceiling, Some(room.device_bytes));
    }

    /// Build actual checked Seismic graph families. The engine may choose how
    /// many slots to reserve, but no test supplies intermediate tensor shapes.
    fn prepared_resource_plan() -> Option<ResourcePlan> {
        let catalog = seismic::Devices::discover(&[]).ok()?;
        let selected = crate::platform::select_device(
            &catalog,
            crate::ExecutionPath::Native,
            crate::platform::DeviceRequest::Automatic,
            &crate::platform::MemoryReserves::standard(),
        )
        .ok()?;
        let device = catalog
            .open(catalog.resolve(selected.info.selector()).unwrap())
            .unwrap();
        let definition = fixture_definition();
        let manifest = fixture_manifest(&definition);
        let limits = ResourceLimits {
            max_launch_rows: 2,
            max_launch_slots: 2,
            max_selected_rows: 2,
            max_drafting_slots: 2,
            exported_logits_rows: 0,
            max_images_per_request: magnitude_artifacts::MAX_IMAGES_PER_REQUEST,
            max_image_cells: 0,
            lookahead: false,
        };
        let capacity_bytes = ResourceCapacity {
            domain_bytes: selected.assessment_capacity_bytes.min(512 * 1024 * 1024),
            tensor_operations: TensorOperations::of(selected.info.forms_tensor_operations()),
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
                numerical_profile: crate::ResolvedNumericalProfile::standard(),
                settings: crate::TuningSettings::STANDARD,
            },
            crate::UnpreparedUnits::ServeDefaults,
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
                &seismic::Workers::new(1),
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
                draft.programs(),
                &state,
                limits,
                PlannedMethod::Plain,
            )
            .unwrap();
        let copy = programs.state_graphs().unwrap();
        let plan = ResourcePlanner::plan_with_state(
            state,
            &target,
            &readout,
            None,
            None,
            copy,
        )
        .unwrap();
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
        // One launch per lane in flight, whatever the request count; the
        // target's residual pair; one request's retained readout beside it.
        // More outputs are elastic.
        assert_eq!((target.activations, target.output_slots), (1, 2));
        assert_eq!((readout.activations, readout.output_slots), (1, 2));
        assert_eq!((state.activations, state.output_slots), (1, 1));
        assert!(target.workspace_bytes > 0);
        assert!(target.output_bytes > 0);
        assert!(readout.output_bytes > 0);
        // Workspace is the one arena, charged once at the largest family's.
        let arena = target
            .workspace_bytes
            .max(readout.workspace_bytes)
            .max(state.workspace_bytes);
        assert_eq!(plan.arena_bytes(), arena);
        assert_eq!(
            plan.bytes().scratch,
            arena + target.committed_bytes + readout.committed_bytes + state.committed_bytes,
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

    /// The planning fixture over four hyper-connection streams: its lone
    /// attention mixer reads through a mix site and scatters into the
    /// streams, and its exit is a hyper-connection head, with no output
    /// norm.
    pub(crate) fn hyper_connection_fixture_definition() -> ModelDefinition {
        use magnitude_family_contracts::{HyperConnectionMixing, HyperConnections};
        let mut definition = fixture_definition();
        let decoder = &mut definition.decoder;
        decoder.residual = ResidualForm::HyperConnections(HyperConnections {
            streams: STREAMS,
            mixing: HyperConnectionMixing::LowRankGated { rank: RANK },
        });
        // A dense feed-forward has no streams form: the block is its mixer.
        let sublayers = &mut decoder.blocks[0].sublayers;
        sublayers.truncate(1);
        sublayers[0].input = InputNorm::HyperConnectionMix(site("attn_hc", true));
        sublayers[0].output = OutputForm::Scatter;
        decoder.exit.norm = ExitNorm::HyperConnectionHead(site("output_hc", false));
        definition
    }

    /// The streams and low rank of the hyper-connection fixtures.
    const STREAMS: u64 = 4;
    const RANK: u64 = 64;

    /// A mix site of the fixture geometry, with inject weights when
    /// `injected`.
    fn site(prefix: &str, injected: bool) -> magnitude_family_contracts::HyperConnectionWeights {
        let wide = STREAMS * fixture_definition().decoder.hidden;
        magnitude_family_contracts::HyperConnectionWeights {
            norm: RmsNorm {
                weight: descriptor(&format!("{prefix}_norm"), &[wide]),
                epsilon: 1e-6,
            },
            down: descriptor(&format!("{prefix}_down"), &[RANK, wide]),
            up: descriptor(&format!("{prefix}_up"), &[wide, RANK]),
            inject: injected.then(|| descriptor(&format!("{prefix}_inject"), &[STREAMS, wide])),
        }
    }

    /// A qwen4exp-shaped decoder over the hyper-connection fixture's
    /// streams: a gated delta block whose routed feed-forward is followed by
    /// an n-gram input, then an attention block with a routed feed-forward,
    /// every mixer and feed-forward reading through its mix site.
    pub(crate) fn qwen4exp_fixture_definition() -> ModelDefinition {
        use magnitude_family_contracts::{
            ExpertSelection, GatedDelta, NGramHash, NGramInput, RecurrentHeadMapping,
            RouteNormalization, RoutedFfn, Router, RouterInput, ScoreFunction, SharedExpert,
            SharedExpertGate,
        };
        // Metal routes over a multiple of 32 experts.
        const EXPERTS: u64 = 32;
        const FEATURES: u64 = 64;
        const STATE: u64 = 128;
        const NGRAM_WIDTH: u64 = 32;
        const NGRAM_ROWS: u64 = 64;
        let mut definition = hyper_connection_fixture_definition();
        let decoder = &mut definition.decoder;
        let hidden = decoder.hidden;
        let wide = STREAMS * hidden;
        let rms = |name: &str, width| RmsNorm {
            weight: descriptor(name, &[width]),
            epsilon: 1e-6,
        };
        let silu = |gate: WeightDescriptor, up: WeightDescriptor| FeedForwardUp::Gated {
            activation: ActivationFunction::Silu,
            gate,
            up,
        };
        let mixing = |prefix: &str, op: Operator| Sublayer {
            input: InputNorm::HyperConnectionMix(site(prefix, true)),
            op,
            output: OutputForm::Scatter,
        };
        let routed = |block: u64| {
            let name = |part: &str| format!("blk.{block}.{part}");
            Operator::RoutedFfn(Box::new(RoutedFfn {
                experts: EXPERTS,
                selected: 2,
                intermediate: FEATURES,
                router: Router {
                    weight: descriptor(&name("ffn_gate_inp"), &[EXPERTS, hidden]),
                    input: RouterInput::Operator,
                    score: ScoreFunction::Softmax,
                    selection: ExpertSelection::TopK { bias: None },
                    normalization: RouteNormalization::Sum,
                    scale: 1.0,
                },
                expert_up: silu(
                    descriptor(&name("ffn_gate_exps"), &[EXPERTS, FEATURES, hidden]),
                    descriptor(&name("ffn_up_exps"), &[EXPERTS, FEATURES, hidden]),
                ),
                expert_down: descriptor(&name("ffn_down_exps"), &[EXPERTS, hidden, FEATURES]),
                expert_scale: None,
                latent: None,
                shared: Some(SharedExpert {
                    intermediate: FEATURES,
                    up: silu(
                        descriptor(&name("ffn_gate_shexp"), &[FEATURES, hidden]),
                        descriptor(&name("ffn_up_shexp"), &[FEATURES, hidden]),
                    ),
                    down: descriptor(&name("ffn_down_shexp"), &[hidden, FEATURES]),
                    gate: SharedExpertGate::Sigmoid(descriptor(
                        &name("ffn_gate_inp_shexp"),
                        &[hidden],
                    )),
                }),
            }))
        };
        let (key_heads, value_heads) = (1, 2);
        let channels = (2 * key_heads + value_heads) * STATE;
        let gated_delta = Operator::GatedDelta(Box::new(GatedDelta {
            convolution_width: 4,
            key_heads,
            value_heads,
            width: STATE,
            head_mapping: RecurrentHeadMapping::Tiled,
            query_key_value: descriptor("blk.0.attn_qkv", &[channels, hidden]),
            gate: descriptor("blk.0.attn_gate", &[value_heads * STATE, hidden]),
            alpha: descriptor("blk.0.ssm_alpha", &[value_heads, hidden]),
            beta: descriptor("blk.0.ssm_beta", &[value_heads, hidden]),
            convolution: descriptor("blk.0.ssm_conv1d", &[channels, 4]),
            decay: descriptor("blk.0.ssm_a", &[value_heads]),
            time_bias: descriptor("blk.0.ssm_dt", &[value_heads]),
            norm: rms("blk.0.ssm_norm", STATE),
            output_gate: GateFunction::Sigmoid,
            output: descriptor("blk.0.ssm_out", &[hidden, value_heads * STATE]),
        }));
        let (context, heads_per_order) = (2, 2);
        let heads = context * heads_per_order;
        let ngram = Operator::NGramInput(Box::new(NGramInput {
            hash: NGramHash {
                context,
                heads_per_order,
                multipliers: vec![1, 7, 13],
                sizes: vec![NGRAM_ROWS / heads; heads as usize],
                offsets: (0..heads).map(|head| head * NGRAM_ROWS / heads).collect(),
                reset: 1,
            },
            width: NGRAM_WIDTH,
            table: descriptor("per_layer_token_embd.weight", &[NGRAM_ROWS, NGRAM_WIDTH]),
            key: descriptor("blk.0.ngram_k", &[wide, heads * NGRAM_WIDTH]),
            value: descriptor("blk.0.ngram_v", &[hidden, heads * NGRAM_WIDTH]),
            key_norm: rms("blk.0.ngram_k_norm", wide),
            query_norm: rms("blk.0.ngram_q_norm", wide),
            convolution_norm: rms("blk.0.ngram_conv_norm", wide),
            convolution: descriptor("blk.0.ngram_conv", &[wide, 4]),
            kernel: 4,
            dilation: 1,
        }));
        let attention = decoder.blocks[0].sublayers[0].op.clone();
        decoder.blocks = vec![
            Block {
                sublayers: vec![
                    mixing("blk.0.attn_hc", gated_delta),
                    mixing("blk.0.ffn_hc", routed(0)),
                    Sublayer {
                        input: InputNorm::None,
                        op: ngram,
                        output: OutputForm::Residual,
                    },
                ],
            },
            Block {
                sublayers: vec![
                    mixing("blk.1.attn_hc", attention),
                    mixing("blk.1.ffn_hc", routed(1)),
                ],
            },
        ];
        definition
    }

    /// The fixture manifest of `definition`, its n-gram tables stored as
    /// IQ4_NL blocks (the stored form a streamed table reads).
    pub(crate) fn stored_table_manifest(definition: &ModelDefinition) -> PackageManifest {
        let mut manifest = fixture_manifest(definition);
        let tables = definition
            .decoder
            .sublayers()
            .filter_map(|(_, sublayer)| match &sublayer.op {
                Operator::NGramInput(input) => Some(input.table.name.clone()),
                _ => None,
            })
            .collect::<Vec<_>>();
        // Re-lay the tensors with the tables' block bytes.
        let mut offset = 0;
        for tensor in &mut manifest.target.tensors {
            if tables.contains(&tensor.name) {
                tensor.encoding = Encoding::Iq4Nl;
                tensor.nbytes = tensor.shape.iter().product::<u64>() / 32 * 18;
            }
            tensor.offset = offset;
            offset += tensor.nbytes;
        }
        manifest.target.files[0].size = offset;
        manifest
    }

    /// The qwen4exp fixture without its n-gram input, its tensors encoded
    /// as `encoding` names them (F16 otherwise), and its program plan.
    fn streams_program_plan(
        encoding: impl Fn(&str) -> Encoding,
    ) -> Result<(ModelLoadPlan, ProgramPlan), crate::error::PlanError> {
        let mut definition = qwen4exp_fixture_definition();
        definition.decoder.blocks[0].sublayers.truncate(2);
        let mut manifest = fixture_manifest(&definition);
        let mut offset = 0;
        for tensor in &mut manifest.target.tensors {
            let values = tensor.shape.iter().product::<u64>();
            tensor.encoding = encoding(&tensor.name);
            tensor.nbytes = match tensor.encoding {
                Encoding::Q8_0 => values / 32 * 34,
                _ => values * 2,
            };
            tensor.offset = offset;
            offset += tensor.nbytes;
        }
        manifest.target.files[0].size = offset;
        let selection = ComponentSelection {
            head: false,
            vision: false,
        };
        let load =
            ModelLoadPlan::derive(&manifest, &definition, selection, seismic::Layout::Rows8)
                .unwrap();
        let plan = load.program_plan(&definition, KvCodec::Dense)?;
        Ok((load, plan))
    }

    /// A decoder whose mix sites all plan one representation binds every
    /// block's sites at what the one decoder-wide binding bound (the exit
    /// site's norm, down and up elements, the sublayers' inject element),
    /// the exit site on the last block only; so its streams entries are the
    /// decoder-wide binding's: a sublayer site's and the exit's first stage,
    /// and one second stage.
    #[test]
    fn uniform_streams_blocks_bind_the_decoder_wide_elements() {
        use crate::operators::streams::{InjectingSite, MixSite};
        let (load, plan) = streams_program_plan(|_| Encoding::F16).unwrap();
        let element = |scope, kind| planned_element(load.target(), scope, kind).unwrap();
        let exit = MixSite {
            norm: element(WeightScope::Target, WeightKind::HyperConnectionNorm),
            down: element(WeightScope::Target, WeightKind::HyperConnectionDown),
            up: element(WeightScope::Target, WeightKind::HyperConnectionUp),
        };
        let site = InjectingSite {
            site: exit,
            inject: element(
                WeightScope::TargetSublayer(SublayerIndex {
                    block: 0,
                    sublayer: 0,
                }),
                WeightKind::HyperConnectionInject,
            ),
        };
        let blocks = plan.target().blocks();
        let mut downs = std::collections::HashSet::new();
        let mut ups = std::collections::HashSet::new();
        for (index, block) in blocks.iter().enumerate() {
            let streams = block.streams().unwrap();
            assert_eq!(streams.mixer, site);
            assert_eq!(streams.feed_forward, Some(site));
            assert_eq!(streams.exit, (index + 1 == blocks.len()).then_some(exit));
            for entries in streams.entries().present() {
                downs.insert(entries.down);
                ups.insert(entries.up);
            }
        }
        assert_eq!(downs.len(), 2);
        assert_eq!(ups.len(), 1);
    }

    /// Mix sites whose weights plan different representations (as an
    /// IQ3_M file's do, by layer) plan: each block binds its own sites'
    /// elements, and the exit site its own.
    #[test]
    fn mixed_streams_sites_plan_their_own_elements() {
        let packed = |name: &str| {
            if matches!(name, "blk.1.ffn_hc_down" | "output_hc_up") {
                Encoding::Q8_0
            } else {
                Encoding::F16
            }
        };
        let (load, plan) = streams_program_plan(packed).unwrap();
        let element = |scope, kind| planned_element(load.target(), scope, kind).unwrap();
        let scope = |block, sublayer| WeightScope::TargetSublayer(SublayerIndex { block, sublayer });
        let blocks = plan.target().blocks();
        let [first, last] = [blocks[0].streams().unwrap(), blocks[1].streams().unwrap()];
        let down = |scope| element(scope, WeightKind::HyperConnectionDown);
        assert_ne!(down(scope(1, 1)), down(scope(0, 1)));
        assert_eq!(first.feed_forward.unwrap().site.down, down(scope(0, 1)));
        assert_eq!(last.mixer.site.down, down(scope(1, 0)));
        assert_eq!(last.feed_forward.unwrap().site.down, down(scope(1, 1)));
        let exit = last.exit.unwrap();
        assert_eq!(exit.up, element(WeightScope::Target, WeightKind::HyperConnectionUp));
        assert_ne!(exit.up, last.mixer.site.up);
        assert_eq!(first.exit, None);
    }

    /// Decoders over hyper-connection streams prepare on the device: their
    /// entries (attention, gated delta, routed, the streams' and the n-gram
    /// input's) qualify in the forms their block graphs read and join, their
    /// target graphs seal, and their readout graphs, which read no norm, seal
    /// at the storage their certification charges.
    #[test]
    fn a_streams_decoder_prepares_and_certifies_on_the_device() {
        let Ok(catalog) = seismic::Devices::discover(&[]) else {
            return;
        };
        let attention = hyper_connection_fixture_definition();
        let qwen4exp = qwen4exp_fixture_definition();
        for (definition, manifest) in [
            (&attention, fixture_manifest(&attention)),
            (&qwen4exp, stored_table_manifest(&qwen4exp)),
        ] {
            prepares_on_the_device(&catalog, definition, &manifest);
        }
    }

    fn prepares_on_the_device(
        catalog: &seismic::Devices,
        definition: &ModelDefinition,
        manifest: &PackageManifest,
    ) {
        let limits = ResourceLimits {
            max_launch_rows: 2,
            max_launch_slots: 2,
            max_selected_rows: 2,
            max_drafting_slots: 2,
            exported_logits_rows: 0,
            max_images_per_request: magnitude_artifacts::MAX_IMAGES_PER_REQUEST,
            max_image_cells: 0,
            lookahead: false,
        };
        for backend in [seismic::BackendName::Metal, seismic::BackendName::Cpu] {
            let Ok(selected) = crate::platform::select_device(
                catalog,
                crate::ExecutionPath::Native,
                crate::platform::DeviceRequest::Backend(backend),
                &crate::platform::MemoryReserves::standard(),
            ) else {
                continue;
            };
            let device = catalog
                .open(catalog.resolve(selected.info.selector()).unwrap())
                .unwrap();
            let draft = ExecutionPlanner::prepare(
                &selected,
                manifest,
                definition,
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
                definition,
                draft.load(),
                draft.policy().method(),
                KvCodec::Dense,
                limits,
                ResourceCapacity {
                    domain_bytes: selected.assessment_capacity_bytes.min(512 * 1024 * 1024),
                    tensor_operations: TensorOperations::of(
                        selected.info.forms_tensor_operations(),
                    ),
                },
            )
            .unwrap();
            let programs = crate::AttestedPrograms::prepare_draft(
                &draft,
                &device,
                crate::TuningContext {
                    definition,
                    weights: &crate::ZeroTuningWeights,
                    observer: &crate::UnreportedTuning,
                    cache: None,
                    numerical_profile: crate::ResolvedNumericalProfile::standard(),
                    settings: crate::TuningSettings::STANDARD,
                },
                crate::UnpreparedUnits::ServeDefaults,
            )
            .unwrap_or_else(|error| panic!("{backend:?}: {error}"));
            programs
                .prepare_target_graphs(
                    &device,
                    draft.load(),
                    &definition.decoder,
                    &state,
                    draft.programs().target(),
                    limits,
                    &seismic::Workers::new(1),
                )
                .unwrap_or_else(|error| panic!("{backend:?}: {error}"));
            let readout = programs
                .prepare_target_readout_graphs(&device, draft.load(), &definition.decoder, limits)
                .unwrap_or_else(|error| panic!("{backend:?}: {error}"));
            let assessed = crate::programs::graph::readout::checked_readout_family_storage(
                backend,
                draft.load(),
                &definition.decoder,
                limits,
            )
            .unwrap();
            assert_eq!(
                readout.family().workspace_bytes(),
                assessed.workspace,
                "{backend:?}"
            );
            assert_eq!(
                readout.family().output_bytes(),
                assessed.output,
                "{backend:?}"
            );
            assert_eq!(
                readout.family().upload_bytes(),
                assessed.upload,
                "{backend:?}"
            );
        }
    }
}
