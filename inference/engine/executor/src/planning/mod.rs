//! Device-free numerical planning.
//!
//! Component, weight, program, capability, and resource authorities are
//! separated below; this module only reexports their stable contracts.

mod assessment;
mod bank;
mod capabilities;
mod components;
mod execution_plan;
mod imports;
pub(crate) use imports::ImportPlan;
mod programs;
mod prices;
pub use prices::{PlanPrices, PlanSteps};
pub(crate) use prices::{BankPrices, Prior};
mod prompt_room;
pub(crate) use prompt_room::{pass_ladder, routed_prompt_seconds};
mod resources;
mod weights;

pub use assessment::{
    AssessmentFit, AssessmentFitVerdict, AssessmentGraphResourceBounds, AssessmentHeaderBounds,
    AssessmentMemoryBounds, AssessmentMemoryCharge, AssessmentMemoryTerms, CONVERSATION_CHECKPOINTS,
    DeviceProcessBytes, HostProcessBytes, SealedGraphHostBytes, TargetGraphBounds, UndraftedBounds, VisionGraphBounds, measurement_process_bytes,
};
pub(crate) use assessment::fit_depth;
pub use bank::{
    BankAllocation, BankBinding, BankCounts, BankLayers, BankTable, BankLayout, BankPart, ConveyorCarrier,
    Lender,
    HintWeight, HINT_LEADS, PlacedAllocations, RegionWords,
    ExpertBankPlan, RecordTensor, RecordTensors, RoutedTensor, StagingPool, StoredPart, StoredStack,
    growth_target, BankSubstrate, BankTier, HostTier, LayerSite, Parking, Placer,
};
pub(crate) use bank::{layer_homes, releasing};
#[cfg(test)]
pub(crate) use bank::routed_experts;
pub use capabilities::{
    CapabilityPlan, DraftWidths, MAX_DRAFT_PROPOSALS, PlannedMethod, draft_ladder,
};
pub use components::{ArtifactComponent, ArtifactComponentKind, ComponentPlan, ComponentSelection};
#[cfg(test)]
pub(crate) use execution_plan::bank_substrate;
pub use execution_plan::{
    BackendPlan, BankForm, ChargedDraft, ExecutionPlan, ExecutionPlanDraft, ExecutionPlanner, FloorForms, HeldForm, HeldBound, HeldCharge, NoBank, Placed, Unplaced, WeightBound, WholeFit, PlanRoom, PlannedDevice, SettleRoom, StableRoom,
    ResolvedPolicy,
};
pub use programs::{
    BankedFeedForward, ConvolutionBinding, ConvolvedLayerBinding, CorrectorBinding, CorrectorLayerBinding,
    DraftBlockBinding, DraftProgramPlan, FeedForwardProgramSlot, HeadProgramPlan,
    ImportProgramSlot, MarkovBinding, MixerProgramSlot, ProgramPlan, SelectorBinding,
    SharedHeadProgramPlan, StateProgramPlan, TapProgramPlan, TargetBlockProgramSlot,
    TargetProgramPlan, VisionProgramPlan,
};
pub use resources::{
    ByPass, CPU_IMAGE_CELLS, FlightDepth, GraphSlots, MAX_IMAGE_CELLS, NativeGraphCharge, ResourceBytes,
    PassLaunches, ResourceCapacity, ResourceLimits, ResourcePlan, ResourcePlanner, StartupSlots,
    StateCapacityPlan, StateResourcePlan, StateStorePlan, TensorOperations, image_cell_limit,
    max_image_cells, reads_decoded_history,
};
pub(crate) use resources::{class_bytes, graph_demand, serial_upload_regions};
pub(crate) use weights::{AttentionForm, AttentionWidths, GateForm, RoundingArguments, RoundingWords, SeparateGate, axis};
pub use weights::{
    AttentionBinding, AttentionShape, AttentionValues, CandidatePool, CandidatesBinding,
    DenseBinding, DenseBranchBinding, DenseScales,
    DirectoryForm, EmbeddingBinding, FeaturesBinding, GroupedOutputBinding, GroupedOutputShape,
    HeadBinding, Holding, HostPeak, HostTablePlan, IndexerBinding, LowRankQueryBinding,
    ModelLoadPlan, PickedBinding, PooledBinding,
    NGramBinding, NGramGateForm, NGramRowRule, NGramTaps, ParallelBinding, PerLayerBinding, PerLayerEntryBinding,
    ReadoutBinding,
    ReadoutClassChoice, ReadoutHead, RecurrentBinding, RoundingArgs, RoutedBinding, ScalableWeight,
    SelectionBinding, SelectionShape, SublayerTail, TableFill, TableFlights, WeightPlacement, WeightPlan, WeightScalePlan, WeightStorageIdentity,
    held_storage, resident_element, resident_elements, resident_layout, scaled_source_element,
    rounding_args, source_element, stored_weights,
};

pub(crate) use weights::{
    activation_dtype, elements, leading_bytes, packing_run, resident_storage, stored_row_ranges, value_packets,
};
use weights::{
    planned_element, planned_readout, planned_scalable,
    weight_bytes_by_component,
};
#[cfg(test)]
use weights::{resident_dtype, validate_unique_roles};

#[cfg(any(test, feature = "fixtures"))]
pub(crate) mod fixtures;

#[cfg(test)]
mod grid_records_tests;

#[cfg(test)]
pub(crate) mod tests {
    use super::fixtures::{
        fixture_definition, fixture_manifest, hyper_connection_fixture_definition, qwen4exp_fixture_definition,
        stored_table_manifest,
    };
    use super::*;
    use magnitude_artifacts::{
        ArtifactIdentity, ComponentManifest, PackageManifest, gguf::Encoding,
    };
    use magnitude_family_contracts::{
        ModelDefinition, SublayerIndex, WeightDescriptor, WeightKind, WeightRole, WeightScope,
    };
    use crate::operators::streams::BlockStreams;
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
            holding: Holding::Whole,
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
        let rows = |fill| {
            Holding::Streamed(super::weights::StreamedRows {
                table_rows: 1_000_000,
                staging_rows: 1,
                landing_rows: 0,
                directory: DirectoryForm::Dense,
                fill,
            })
        };
        let flights = super::weights::TableFlights { flights: crate::planning::fixtures::LIMITS.flights(), lanes: 1 };
        table.holding = rows(super::weights::TableFill::Imported(flights));
        let ordinary = weight(
            WeightRole {
                scope: WeightScope::Target,
                kind: WeightKind::OutputNorm,
            },
            "norm",
        );
        let imports = ImportPlan::new([&table, &ordinary]);
        // The table's source is one row. Its million stored rows must not
        // permit the two adjacent ordinary sources to share a larger batch.
        assert_eq!(imports.batch([(0, ordinary.source_bytes), (0, ordinary.source_bytes)]), Some(1));
        assert_eq!(imports.peak_bytes(), ImportPlan::new([&ordinary]).peak_bytes());
        // Stored tables upload nothing, regardless of their stored extent.
        table.source_bytes = 0;
        table.holding = rows(super::weights::TableFill::Stored);
        assert_eq!(ImportPlan::new([&table]).peak_bytes(), 0);
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

    #[test]
    fn plain_weight_plan_ignores_an_unadmitted_optional_draft() {
        let definition = fixture_definition();
        let mut manifest = fixture_manifest(&definition);
        let selection = ComponentSelection {
            head: false,
            vision: false,
        };
        let plain =
            ModelLoadPlan::derive(&manifest, &definition, selection, seismic::Layout::Rows32, crate::PlannedMethod::Plain, crate::planning::fixtures::LIMITS)
                .unwrap();
        manifest.draft = Some(ComponentManifest {
            files: Vec::new(),
            identity: ArtifactIdentity([9; 32]),
            tensors: Vec::new(),
        });
        let fallback =
            ModelLoadPlan::derive(&manifest, &definition, selection, seismic::Layout::Rows32, crate::PlannedMethod::Plain, crate::planning::fixtures::LIMITS)
                .unwrap();
        assert_eq!(fallback, plain);
    }

    /// The kernel inventory, derived without a device, names exactly the
    /// kernels a load prepares on the selected device.
    #[test]
    fn kernel_inventory_names_every_kernel_a_load_prepares() {
        let catalog = seismic::Devices::discover(&[]).expect("devices are discovered");
        let selected = crate::platform::select_device(
            &catalog,
            crate::ExecutionPath::Native,
            crate::platform::DeviceRequest::Automatic,
            &crate::platform::MemoryReserves::of_machine(None),
        )
        .unwrap_or_else(|error| panic!("the host's device is selected: {error}"));
        let device = crate::test_support::open(&catalog, catalog.resolve(selected.info.selector()).unwrap())
            .unwrap_or_else(|error| panic!("the selected device opens: {error}"));
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
            chain_capacity: std::num::NonZeroUsize::MIN,
            conversations: std::num::NonZeroUsize::MIN,
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
        struct Ceiling(std::cell::Cell<Option<(Option<u64>, u64)>>);
        impl crate::TuningObserver for Ceiling {
            fn event(&self, event: &crate::TuningEvent) {
                if let crate::TuningEvent::Budget(budget) = event {
                    self.0.set(Some((budget.memory.ceiling_bytes, budget.memory.began_bytes)));
                }
            }
        }
        /// A room of `.0` bytes in all on the device, whatever it charges.
        struct Total(u64);
        impl crate::TuningRoom for Total {
            fn now(&self, device: &seismic::Device) -> crate::RoomNow {
                crate::RoomNow {
                    device_bytes: self.0.saturating_sub(device.device_ledger().charged),
                    host_bytes: u64::MAX,
                    host_shared: true,
                }
            }
            fn release(&self) -> crate::platform::ReleaseRate {
                crate::platform::ReleaseRate::new(1e9)
            }
        }
        let catalog = seismic::Devices::discover(&[]).expect("devices are discovered");
        let selected = crate::platform::select_device(
            &catalog,
            crate::ExecutionPath::Native,
            crate::platform::DeviceRequest::Automatic,
            &crate::platform::MemoryReserves::of_machine(None),
        )
        .unwrap_or_else(|error| panic!("the host's device is selected: {error}"));
        let device = crate::test_support::open(&catalog, catalog.resolve(selected.info.selector()).unwrap())
            .unwrap_or_else(|error| panic!("the selected device opens: {error}"));
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
                chain_capacity: std::num::NonZeroUsize::MIN,
                conversations: std::num::NonZeroUsize::MIN,
            },
        )
        .unwrap();
        let observer = Ceiling(std::cell::Cell::new(None));
        let room = Total(1 << 40);
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
            Some(&room),
        )
        .unwrap();
        let (ceiling, began) = observer
            .0
            .get()
            .expect("a search reports what became of its budget");
        assert_eq!(ceiling, Some(room.0 - began), "held to the room, less what was charged before it began");
    }

    /// A search's room is read live: a neighbour that takes the room once
    /// the units' mandatory work is done takes it from the search's next
    /// admission, and the optional work it would have started is refused
    /// for its bytes, as the room stood then. A room fixed when the search
    /// began admitted it into memory that was gone (a Prepare worker,
    /// ended by the system with the host's memory wired by its neighbours).
    #[test]
    fn a_search_admits_against_the_room_as_it_shrinks() {
        use std::cell::{Cell, RefCell};
        /// Whether a unit's mandatory work is done, and the domains of the
        /// search's refusal for bytes as its budget report states them
        /// (domain, ceiling, held), none where nothing was refused.
        #[derive(Default)]
        struct Search {
            mandatory: Cell<bool>,
            budget: RefCell<Option<Option<Vec<(&'static str, u64, u64)>>>>,
        }
        impl crate::TuningObserver for Search {
            fn event(&self, event: &crate::TuningEvent) {
                match event {
                    crate::TuningEvent::Mandatory { .. } => self.mandatory.set(true),
                    crate::TuningEvent::Budget(budget) => {
                        let refused = budget.memory.no_room.as_ref().map(|refused| {
                            refused.domains.iter().map(|room| (room.domain, room.ceiling_bytes, room.held_bytes)).collect()
                        });
                        *self.budget.borrow_mut() = Some(refused);
                    }
                    _ => {}
                }
            }
        }
        /// All the room the platform has until the mandatory work is done,
        /// and none after: a neighbour took it.
        struct Taken<'a>(&'a Search);
        impl crate::TuningRoom for Taken<'_> {
            fn now(&self, device: &seismic::Device) -> crate::RoomNow {
                let left = if self.0.mandatory.get() { 0 } else { 1 << 40 };
                crate::RoomNow { device_bytes: left, host_bytes: left, host_shared: crate::memory::ledger::shares_host(device) }
            }
            fn release(&self) -> crate::platform::ReleaseRate {
                crate::platform::ReleaseRate::new(1e9)
            }
        }
        let catalog = seismic::Devices::discover(&[]).expect("devices are discovered");
        let selected = crate::platform::select_device(
            &catalog,
            crate::ExecutionPath::Native,
            crate::platform::DeviceRequest::Automatic,
            &crate::platform::MemoryReserves::of_machine(None),
        )
        .unwrap_or_else(|error| panic!("the host's device is selected: {error}"));
        let device = crate::test_support::open(&catalog, catalog.resolve(selected.info.selector()).unwrap())
            .unwrap_or_else(|error| panic!("the selected device opens: {error}"));
        let definition = fixture_definition();
        let manifest = fixture_manifest(&definition);
        let draft = ExecutionPlanner::prepare(
            &selected,
            &manifest,
            &definition,
            ComponentSelection { head: false, vision: false },
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
                chain_capacity: std::num::NonZeroUsize::MIN,
                conversations: std::num::NonZeroUsize::MIN,
            },
        )
        .unwrap();
        let search = Search::default();
        let room = Taken(&search);
        crate::AttestedPrograms::search_draft(
            &draft,
            &device,
            crate::TuningContext {
                definition: &definition,
                weights: &crate::ZeroTuningWeights,
                observer: &search,
                cache: None,
                numerical_profile: crate::ResolvedNumericalProfile::standard(),
                settings: crate::TuningSettings { budget: std::time::Duration::from_secs(30), fresh: true },
            },
            Some(&room),
        )
        .unwrap();
        assert!(search.mandatory.get(), "the units' mandatory work is done");
        let refused = search.budget.borrow().clone().expect("a search reports what became of its budget");
        let refused = refused.expect("the optional work is refused for the room that is gone");
        assert!(
            refused.iter().any(|&(domain, ceiling, held)| domain == "device" && ceiling == held),
            "refused in the device's room as it stood then, nothing left beside what tuning held: {refused:?}"
        );
    }

    /// Build actual checked Seismic graph families. The engine may choose how
    /// many slots to reserve, but no test supplies intermediate tensor shapes.
    fn prepared_resource_plan() -> ResourcePlan {
        let catalog = seismic::Devices::discover(&[]).expect("devices are discovered");
        let selected = crate::platform::select_device(
            &catalog,
            crate::ExecutionPath::Native,
            crate::platform::DeviceRequest::Automatic,
            &crate::platform::MemoryReserves::of_machine(None),
        )
        .unwrap_or_else(|error| panic!("the host's device is selected: {error}"));
        let device = crate::test_support::open(&catalog, catalog.resolve(selected.info.selector()).unwrap())
            .unwrap_or_else(|error| panic!("the selected device opens: {error}"));
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
            chain_capacity: std::num::NonZeroUsize::MIN,
            conversations: std::num::NonZeroUsize::MIN,
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
        plan
    }

    #[test]
    fn sealed_graph_footprints_charge_one_request_at_startup() {
        let plan = prepared_resource_plan();
        let target = plan.target_graph();
        let readout = plan.target_readout_graph();
        let state = plan.state_graph();
        // One launch per lane in flight, whatever the request count; the
        // target's rotation of a chain of one (its output and the spare);
        // one request's retained readout beside it.
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
            ModelLoadPlan::derive(&manifest, &definition, selection, seismic::Layout::Rows8, crate::PlannedMethod::Plain, crate::planning::fixtures::LIMITS)
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
            let Some(BlockStreams::LowRank(streams)) = block.block_streams() else { panic!("low-rank streams") };
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
        let [first, last] = [0, 1].map(|block| match blocks[block].block_streams() {
            Some(BlockStreams::LowRank(streams)) => streams,
            other => panic!("block {block}'s streams are {other:?}"),
        });
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
    /// at the storage their certification charges; a prompt group's class
    /// of three places holds the target buffers and chunk slots the plan
    /// prices for it.
    #[test]
    fn a_streams_decoder_prepares_and_certifies_on_the_device() {
        let catalog = seismic::Devices::discover(&[]).expect("devices are discovered");
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
            // A chain of three places: a group's class beyond the floor's.
            chain_capacity: std::num::NonZeroUsize::new(3).unwrap(),
            conversations: std::num::NonZeroUsize::MIN,
        };
        let backends = [seismic::BackendName::Metal, seismic::BackendName::Cpu];
        for backend in backends.into_iter().filter(|&backend| crate::test_support::available(catalog, backend)) {
            let selected = crate::platform::select_device(
                catalog,
                crate::ExecutionPath::Native,
                crate::platform::DeviceRequest::Backend(backend),
                &crate::platform::MemoryReserves::of_machine(None),
            )
            .unwrap_or_else(|error| panic!("an available {backend:?} device is selected: {error}"));
            let device = crate::test_support::open(&catalog, catalog.resolve(selected.info.selector()).unwrap())
                .unwrap_or_else(|error| panic!("the selected {backend:?} device opens: {error}"));
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
                .unwrap_or_else(|error| panic!("{backend:?}: {error}"));
            let readout = programs
                .prepare_target_readout_graphs(&device, draft.load(), &definition.decoder, limits)
                .unwrap_or_else(|error| panic!("{backend:?}: {error}"));
            // The plan's held pass, the most places a group's class claims,
            // is the one the load's target family holds.
            let checked = crate::programs::native_target_graph::checked_target_family(
                backend,
                draft.load(),
                &definition.decoder,
                &state,
                draft.programs().target(),
                limits,
                draft.load().pass(limits),
            )
            .unwrap();
            assert_eq!(target.held_pass(), checked.group_bytes.most(), "{backend:?}");
            // The readout the plan commits is the one the load prepares.
            let (assessed, _) = crate::programs::graph::readout::checked_readout_storage(
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
