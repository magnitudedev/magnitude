//! Model-derived memory charges for metadata-only assessment.
//!
//! The standard assessment workload is one conversation at the fit depth
//! `min(context_limit, 100_000)` on a clean load. Its charge is composed of
//! exact header terms (resident weights, history at the fit depth, recurrent
//! banks, native invocation storage) and upper bounds taken from the same
//! checked class enumeration and slot multipliers production preparation
//! uses (graph pools and bound constants, the startup qualification/import
//! peak). The load planner and the state layout remain the authorities for
//! physical representations and codec planes; the platform policy owns each
//! domain's stable capacity and reserve.

use super::resources::{history_row_bytes, recurrent_bank_bytes, NativeGraphCharge};
use super::{
    source_import_peak_bytes, weight_bytes_by_component, ComponentSelection, ModelLoadPlan,
    PlannedMethod, ResourceLimits,
};
use crate::assessment::DomainFit;
use crate::platform::{DomainRole, FitCapacity};
use crate::{AttestedPrograms, GraphError};
use magnitude_family_contracts::ModelDefinition;
use magnitude_state::{BankCapacity, KvCodec, ModelStateLayout};
use seismic::{BackendName, MemoryPoolId};

/// Exact resident and per-state terms of the standard workload, derived
/// without opening a device or reading tensor payloads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AssessmentMemoryTerms {
    pub target_weights: u64,
    pub head_weights: u64,
    pub vision_weights: u64,
    pub history_per_token: u64,
    pub recurrent_per_bank: u64,
    /// Recurrent banks one conversation holds: its accepted bank, every
    /// in-flight successor the planned lookahead keeps, and the pristine seed.
    pub recurrent_banks: u64,
    pub fit_depth: u64,
    /// Host-resident gathered tables (model-family plan §3.7): claims in the
    /// system-RAM domain, never device weights.
    pub host_table_bytes: u64,
}

/// Upper bounds for every nonresident charge of a clean load. The prepared
/// resource bound covers all steady native holdings (invocation storage,
/// graph workspace, output, upload regions and bound constants); the startup
/// bound covers the additional qualification/import peak; the staging bound
/// is the host upload peak of a dedicated device's import.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AssessmentMemoryBounds {
    pub prepared_resource_bytes: u64,
    pub startup_additional_bytes: u64,
    pub staging_upload_bytes: u64,
}

/// Header-derived charges that are known before graph formation. Prepared
/// native invocation storage is exact; the startup charge is the same
/// qualification/import upper bound used by the production resource planner;
/// the staging charge is the largest selected source tensor's import window.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AssessmentHeaderBounds {
    pub prepared_program_bytes: u64,
    pub startup_additional_bytes: u64,
    pub staging_upload_bytes: u64,
}

/// The standard workload's charge in each memory role a load touches.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AssessmentMemoryCharge {
    /// Weights and slab-rounded state at the fit depth, all exact.
    pub exact_resident_bytes: u64,
    pub bounds: AssessmentMemoryBounds,
    /// Everything the device's allocation domain holds at the clean-load peak.
    pub allocation_bytes: u64,
    /// Host RAM a dedicated device's staged import holds at its peak.
    pub staging_bytes: u64,
    /// Host-resident tables: system RAM, which is the allocation domain of a
    /// unified-memory device and the staging domain of a dedicated one.
    pub host_table_bytes: u64,
}

/// A complete fit result: every domain the load touches, and whether the
/// workload fits all of them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AssessmentFit {
    pub verdict: AssessmentFitVerdict,
    pub domains: Vec<DomainFit>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AssessmentFitVerdict {
    Fits,
    /// The domain with the largest deficit limits the load.
    DoesNotFit {
        limiting: MemoryPoolId,
        deficit_bytes: u64,
    },
}

/// Checked graph-pool demand and distinct bound constants from the same class
/// enumeration and slot multipliers as native preparation: an upper bound for
/// every graph pool a clean load commits on the selected backend. Deriving
/// it builds every graph a load prepares, so it fails with
/// `GraphError::KernelDomain` exactly when the model's program calls a kernel
/// outside its domain on the backend.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AssessmentGraphResourceBounds {
    pub target: NativeGraphCharge,
    pub readout: NativeGraphCharge,
    pub head: Option<NativeGraphCharge>,
    pub vision: Option<NativeGraphCharge>,
    pub state: NativeGraphCharge,
    pub binding_constant_bytes: u64,
    pub total_bytes: u64,
}

impl AssessmentGraphResourceBounds {
    pub fn derive(
        definition: &ModelDefinition,
        load: &ModelLoadPlan,
        state: &super::StateResourcePlan,
        method: PlannedMethod,
        codec: KvCodec,
        limits: super::ResourceLimits,
        backend: BackendName,
    ) -> Result<Self, GraphError> {
        let plan = load
            .program_plan(definition, codec)
            .map_err(|error| error.to_string())?;
        if plan.head().is_some() != matches!(method, PlannedMethod::Mtp { .. })
            || plan.draft().is_some() != matches!(method, PlannedMethod::DFlash { .. })
        {
            return Err("assessment method and drafter program disagree".into());
        }
        // The same startup slots a load commits: one request's need.
        let slots = limits.startup_slots();
        let target_graph = crate::programs::native_target_graph::checked_target_family_storage(
            backend,
            load,
            &definition.decoder,
            state,
            plan.target(),
            limits,
        )?;
        let target = NativeGraphCharge::from_checked(
            target_graph.storage,
            crate::programs::native_target_graph::target_runs_per_step(&definition.decoder),
            slots.target,
        )?;
        let readout = NativeGraphCharge::from_checked(
            crate::programs::graph::readout::checked_readout_family_storage(
                backend,
                load,
                &definition.decoder,
                limits,
            )?,
            1,
            slots.readout,
        )?;
        let (head, head_constant_bytes) = match (plan.head(), plan.draft(), state.head_state()) {
            (None, Some(draft_plan), Some(head_state)) => {
                let draft = definition
                    .draft
                    .as_ref()
                    .ok_or("a draft program without a draft")?;
                let geometry = crate::programs::native_draft::DraftGeometry::new(
                    definition,
                    head_state,
                    method.draft_rows(),
                )?;
                let draft_graph = crate::programs::native_draft::checked_draft_family_storage(
                    backend,
                    load,
                    draft_plan,
                    &geometry,
                    crate::programs::native_draft::draft_graph_classes(
                        limits,
                        method.draft_rows(),
                        draft,
                    )?,
                )?;
                (
                    Some(NativeGraphCharge::from_checked(
                        draft_graph.storage,
                        1,
                        slots.head,
                    )?),
                    draft_graph.binding_constant_bytes,
                )
            }
            (Some(head), None, Some(head_state)) => {
                let binding = *head.blocks().first().ok_or("head program has no block")?;
                let head_block = definition
                    .head
                    .as_ref()
                    .and_then(|head| head.blocks.first())
                    .ok_or("head graph requires a draft head block")?;
                let history = head_state.sole_history()?;
                let history_rows =
                    u64::try_from(history.rows).map_err(|_| "head history rows exceed u64")?;
                let head_graph = crate::programs::native_head::checked_head_family_storage(
                    backend,
                    load,
                    &definition.decoder,
                    head_block,
                    binding,
                    crate::programs::native_head::head_graph_classes(
                        limits,
                        history_rows,
                        history.slab_rows,
                        head_state.span_limit(),
                        method.draft_rows(),
                    )?,
                )?;
                (
                    Some(NativeGraphCharge::from_checked(
                        head_graph.storage,
                        1,
                        slots.head,
                    )?),
                    head_graph.binding_constant_bytes,
                )
            }
            (None, None, None) => (None, 0),
            _ => return Err("drafter program and state disagree".into()),
        };
        let (vision, vision_constant_bytes) = match (plan.vision(), definition.vision.as_ref()) {
            (Some(vision_plan), Some(vision_definition)) => {
                let max_rows = magnitude_batching::row_classes(limits.max_launch_rows)
                    .last()
                    .copied()
                    .ok_or("batch row bound has no class")? as u64;
                let merge = vision_definition.cell_rows();
                max_rows
                    .checked_mul(merge)
                    .ok_or("vision patch row bound overflow")?;
                let patch_rows = (1..=max_rows).map(|rows| rows * merge);
                let vision_graph = crate::programs::native_vision::checked_vision_family_resources(
                    backend,
                    load,
                    vision_definition,
                    vision_plan,
                    patch_rows,
                )?;
                (
                    Some(NativeGraphCharge::from_checked(
                        vision_graph.storage,
                        1,
                        slots.vision,
                    )?),
                    vision_graph.binding_constant_bytes,
                )
            }
            (None, None) => (None, 0),
            _ => return Err("vision program and definition disagree".into()),
        };
        let row_classes = magnitude_batching::row_classes(limits.max_launch_rows)
            .into_iter()
            .map(|rows| rows as u64)
            .collect::<Vec<_>>();
        let state = NativeGraphCharge::from_checked(
            crate::programs::native_state::checked_copy_family_storage(
                backend,
                crate::programs::native_state::state_copy_classes(
                    state.target_state(),
                    state.head_state(),
                    &row_classes,
                )?,
            )?,
            1,
            slots.state,
        )?;
        let binding_constant_bytes = target_graph
            .binding_constant_bytes
            .checked_add(head_constant_bytes)
            .and_then(|bytes| bytes.checked_add(vision_constant_bytes))
            .ok_or("checked graph constant charge overflow")?;
        let total_bytes = super::resources::graph_scratch_bytes(
            &[
                Some(&target),
                Some(&readout),
                head.as_ref(),
                vision.as_ref(),
                Some(&state),
            ]
            .into_iter()
            .flatten()
            .collect::<Vec<_>>(),
        )?
        .checked_add(binding_constant_bytes)
        .ok_or("checked graph resource bound overflow")?;
        Ok(Self {
            target,
            readout,
            head,
            vision,
            state,
            binding_constant_bytes,
            total_bytes,
        })
    }
}

impl AssessmentHeaderBounds {
    pub fn derive(
        definition: &ModelDefinition,
        load: &ModelLoadPlan,
        codec: KvCodec,
        backend: BackendName,
    ) -> Result<Self, String> {
        let programs = load
            .program_plan(definition, codec)
            .map_err(|error| error.to_string())?;
        let prepared_program_bytes =
            AttestedPrograms::planned_invocation_workspace_bytes(&programs, backend)
                .map_err(|error| error.to_string())?;
        let largest_source = load
            .weights()
            .map(|weight| weight.source_bytes)
            .max()
            .unwrap_or(0);
        let import_peak = source_import_peak_bytes(largest_source)?;
        let startup_additional_bytes =
            AttestedPrograms::qualification_peak_bytes(load)?.max(import_peak);
        Ok(Self {
            prepared_program_bytes,
            startup_additional_bytes,
            staging_upload_bytes: import_peak,
        })
    }

    /// Complete the memory bounds with the graph-pool upper bound for the
    /// same backend and workload limits.
    pub fn with_graph_resource_bound(
        self,
        graph: &AssessmentGraphResourceBounds,
    ) -> Result<AssessmentMemoryBounds, String> {
        Ok(AssessmentMemoryBounds {
            prepared_resource_bytes: self
                .prepared_program_bytes
                .checked_add(graph.total_bytes)
                .ok_or("assessment prepared resource bound overflow")?,
            startup_additional_bytes: self.startup_additional_bytes,
            staging_upload_bytes: self.staging_upload_bytes,
        })
    }
}

impl AssessmentMemoryTerms {
    /// The standard assessment workload is one conversation at the lesser
    /// of the supported context and 100,000 tokens. Method state and optional
    /// components are included only when selected by the planned method; the
    /// conversation's in-flight banks follow the planned lookahead.
    pub fn derive(
        definition: &ModelDefinition,
        load: &ModelLoadPlan,
        selection: ComponentSelection,
        codec: KvCodec,
        method: PlannedMethod,
        limits: ResourceLimits,
    ) -> Result<Self, String> {
        if selection.head != method.drafts() {
            return Err("assessment method and head selection disagree".into());
        }
        if selection.head != load.head().is_some() || selection.vision != load.vision().is_some() {
            return Err("assessment selection disagrees with the load plan".into());
        }
        let weights = weight_bytes_by_component(load)?;
        let drafter = if selection.head {
            crate::operators::draft::drafter_blocks(definition)
        } else {
            Vec::new()
        };
        let layout =
            ModelStateLayout::derive(&definition.decoder, &drafter, codec, method.draft_rows())?;
        let history_per_token = history_row_bytes(&layout.target_history)?
            .checked_add(history_row_bytes(&layout.head_history)?)
            .ok_or("history row bytes overflow")?;
        let recurrent_per_bank = recurrent_bank_bytes(&layout.target_recurrent)?;
        // The state store's own capacity rule for one live conversation: its
        // accepted bank, its in-flight step (and that step's queued successor
        // under lookahead) and the permanently pristine zero seed.
        let recurrent_banks = BankCapacity {
            active: 1,
            in_flight: 1 + usize::from(limits.lookahead),
            retained: 0,
        }
        .storage_total()
        .map_err(|error| error.to_string())?;
        Ok(Self {
            target_weights: weights[0],
            head_weights: weights[1],
            vision_weights: weights[2],
            history_per_token,
            recurrent_per_bank,
            recurrent_banks: u64::try_from(recurrent_banks)
                .map_err(|_| "assessment bank count exceeds u64")?,
            fit_depth: definition.decoder.context_limit.min(100_000),
            host_table_bytes: load.host_table_bytes()?,
        })
    }

    pub fn history_at_fit_depth(self) -> Result<u64, String> {
        self.history_per_token
            .checked_mul(self.fit_depth)
            .ok_or_else(|| "assessment history bytes overflow".to_owned())
    }

    pub fn recurrent_at_fit_workload(self) -> Result<u64, String> {
        self.recurrent_per_bank
            .checked_mul(self.recurrent_banks)
            .ok_or_else(|| "assessment recurrent bytes overflow".to_owned())
    }

    /// Exact resident bytes of the workload. The state charge includes each
    /// store's address table and whole history and bank slabs from Seismic.
    pub fn exact_resident_bytes(self, state_slab_bytes: u64) -> Result<u64, String> {
        [
            self.target_weights,
            self.head_weights,
            self.vision_weights,
            state_slab_bytes,
        ]
        .into_iter()
        .try_fold(0u64, |total, bytes| {
            total
                .checked_add(bytes)
                .ok_or_else(|| "assessment resident bytes overflow".to_owned())
        })
    }

    /// The charge of each memory role. The allocation domain holds the
    /// weights and every prepared resource, plus the larger of two phases
    /// that never coexist: the workload's state at the fit depth, which grows
    /// only after the load is ready, and the load itself, its startup state
    /// with the qualification/import peak. A dedicated device's staging
    /// domain holds its host upload window.
    pub fn charge(
        self,
        bounds: AssessmentMemoryBounds,
        state_slab_bytes: u64,
        startup_state_bytes: u64,
    ) -> Result<AssessmentMemoryCharge, String> {
        let exact_resident_bytes = self.exact_resident_bytes(state_slab_bytes)?;
        let loading_bytes = startup_state_bytes
            .checked_add(bounds.startup_additional_bytes)
            .ok_or("assessment startup charge overflow")?;
        let allocation_bytes = self
            .exact_resident_bytes(0)?
            .checked_add(bounds.prepared_resource_bytes)
            .and_then(|bytes| bytes.checked_add(state_slab_bytes.max(loading_bytes)))
            .ok_or("assessment allocation charge overflow")?;
        Ok(AssessmentMemoryCharge {
            exact_resident_bytes,
            bounds,
            allocation_bytes,
            staging_bytes: bounds.staging_upload_bytes,
            host_table_bytes: self.host_table_bytes,
        })
    }
}

impl AssessmentMemoryCharge {
    /// Compare the charge with every domain's stable fit capacity
    /// (`platform::fit_capacities`). A domain's remaining bytes are
    /// `capacity − reserve − required`; the workload fits when none is
    /// negative, and otherwise the domain with the largest deficit limits it.
    pub fn assess_fit(
        self,
        capacities: &[(DomainRole, FitCapacity)],
    ) -> Result<AssessmentFit, String> {
        if !capacities
            .iter()
            .any(|(role, _)| *role == DomainRole::Allocation)
        {
            return Err("fit capacities have no allocation domain".into());
        }
        // Host tables are system RAM: the staging domain of a dedicated
        // device, the allocation domain of a unified one.
        let dedicated = capacities
            .iter()
            .any(|(role, _)| *role == DomainRole::Staging);
        let with_tables = |bytes: u64| {
            bytes
                .checked_add(self.host_table_bytes)
                .ok_or_else(|| "domain charge with host tables overflows".to_owned())
        };
        let domains = capacities
            .iter()
            .map(|&(role, capacity)| {
                let required_bytes = match role {
                    DomainRole::Allocation if dedicated => self.allocation_bytes,
                    DomainRole::Allocation => with_tables(self.allocation_bytes)?,
                    DomainRole::Staging => with_tables(self.staging_bytes)?,
                };
                let signed = |bytes: u64| {
                    i64::try_from(bytes).map_err(|_| "domain byte count exceeds i64".to_owned())
                };
                let remaining_bytes = signed(capacity.capacity_bytes)?
                    .checked_sub(signed(capacity.reserve_bytes)?)
                    .and_then(|bytes| bytes.checked_sub(signed(required_bytes).ok()?))
                    .ok_or("domain remaining bytes overflow")?;
                Ok(DomainFit {
                    role,
                    domain: capacity.domain,
                    capacity_bytes: capacity.capacity_bytes,
                    required_bytes,
                    reserve_bytes: capacity.reserve_bytes,
                    remaining_bytes,
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        let verdict = match domains
            .iter()
            .filter(|domain| domain.remaining_bytes < 0)
            .min_by_key(|domain| domain.remaining_bytes)
        {
            None => AssessmentFitVerdict::Fits,
            Some(limiting) => AssessmentFitVerdict::DoesNotFit {
                limiting: limiting.domain,
                deficit_bytes: limiting.remaining_bytes.unsigned_abs(),
            },
        };
        Ok(AssessmentFit { verdict, domains })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use magnitude_kernels::dense_output;
    use seismic::{BackendName, Element, Layout, NativeGraphMetadata};

    #[test]
    fn checked_native_declarations_are_inspectable_without_a_device() {
        let implementation = seismic::generated::native_implementation_for_backend::<
            magnitude_kernels::dense_output::Entry,
        >(BackendName::Cpu)
        .unwrap();
        assert!(implementation.is_some());
        let invalid = seismic::generated::checked_native_binding::<dense_output::Entry>(
            BackendName::Cpu,
            &[
                ("DW", Element::f32()),
                ("A", Element::f32()),
                ("BOGUS", Element::f32()),
            ],
        )
        .unwrap();
        assert!(matches!(
            invalid,
            seismic::NativeBindingCheck::Unsupported(_)
        ));

        let weight = seismic::generated::checked_native_tensor_parameter::<dense_output::Entry>(
            BackendName::Cpu,
            &[("DW", Element::f32()), ("A", Element::f32())],
            "down_weight",
            &[("M", 4), ("O", 2), ("H", 16), ("F", 32), ("DS", 0)],
        )
        .unwrap();
        assert_eq!(
            weight,
            seismic::NativeTensorParameterCheck::Checked(seismic::NativeTensorMetadata {
                element: Element::f32(),
                extents: vec![16, 32],
                canonical_bytes: 16 * 32 * 4,
            })
        );
        let results = seismic::generated::checked_native_tensor_results::<dense_output::Entry>(
            BackendName::Cpu,
            &[("DW", Element::f32()), ("A", Element::f32())],
            &[("M", 4), ("O", 2), ("H", 16), ("F", 32), ("DS", 0)],
        )
        .unwrap();
        assert_eq!(
            results,
            seismic::NativeTensorResultsCheck::Checked(vec![Some(seismic::NativeTensorMetadata {
                element: Element::f32(),
                extents: vec![2, 16],
                canonical_bytes: 2 * 16 * 4,
            })])
        );
        let call = seismic::generated::checked_native_graph_call::<dense_output::Entry>(
            BackendName::Cpu,
            &[("DW", Element::f32()), ("A", Element::f32())],
            &[("M", 4), ("O", 2), ("H", 16), ("F", 32), ("DS", 0)],
        )
        .unwrap();
        let seismic::generated::CheckedNativeGraphCall::Checked {
            parameters,
            results: call_results,
            ..
        } = call
        else {
            panic!("checked dense output call is supported on CPU");
        };
        let seismic::NativeTensorParameterCheck::Checked(weight) = weight else {
            unreachable!()
        };
        assert!(parameters.contains(&Some(weight)));
        assert_eq!(
            seismic::NativeTensorResultsCheck::Checked(call_results),
            results
        );
    }

    #[test]
    fn checked_embedding_graph_distinguishes_upload_from_selected_input() {
        let definition = crate::planning::tests::fixture_definition();
        let manifest = crate::planning::tests::fixture_manifest(&definition);
        let load = ModelLoadPlan::derive(
            &manifest,
            &definition,
            ComponentSelection {
                head: false,
                vision: false,
            },
            Layout::Rows16,
        )
        .unwrap();
        let uploaded = crate::programs::native_target_graph::checked_entry_graph_storage(
            BackendName::Cpu,
            &load,
            &definition.decoder,
            2,
            true,
        )
        .unwrap();
        let selected = crate::programs::native_target_graph::checked_entry_graph_storage(
            BackendName::Cpu,
            &load,
            &definition.decoder,
            2,
            false,
        )
        .unwrap();
        assert_eq!(uploaded.workspace, selected.workspace);
        assert_eq!(uploaded.output, selected.output);
        assert_eq!(uploaded.upload, 2 * 2 * 4);
        assert_eq!(selected.upload, 0);
        assert!(uploaded.workspace > 0);
        assert!(uploaded.output > 0);

        let elements = [("EW", Element::f32()), ("A", Element::f32())];
        let dimensions = [
            ("M", 2),
            ("V", definition.decoder.vocabulary),
            ("D", definition.decoder.hidden),
        ];
        let mut invalid = NativeGraphMetadata::new(BackendName::Cpu);
        let wrong_table = invalid.port(Element::f32(), &[1, 1]).unwrap();
        let tokens = invalid
            .input_for::<magnitude_kernels::embedding_rows::Entry>(&elements, "tokens", &dimensions)
            .unwrap();
        let mismatch = invalid.enqueue::<magnitude_kernels::embedding_rows::Entry>(
            &elements,
            &dimensions,
            magnitude_kernels::embedding_rows::WorkflowArgs {
                table: wrong_table.tensor().into(),
                tokens: tokens.tensor().into(),
                scale: 1.0,
                normalize: 0,
                epsilon: 0.0,
            },
        );
        assert!(matches!(
            mismatch,
            Err(seismic::NativeGraphMetadataError::Call(
                seismic::CallError::Workflow(seismic::WorkflowError::NativeGraphArgumentMismatch {
                    parameter: 0
                })
            ))
        ));
    }

    #[test]
    fn checked_decoder_family_bounds_every_prepared_class_axis() {
        let definition = crate::planning::tests::fixture_definition();
        let manifest = crate::planning::tests::fixture_manifest(&definition);
        let load = ModelLoadPlan::derive(
            &manifest,
            &definition,
            ComponentSelection {
                head: false,
                vision: false,
            },
            Layout::Rows16,
        )
        .unwrap();
        let limits = crate::ResourceLimits {
            max_launch_rows: 2,
            max_launch_slots: 2,
            max_selected_rows: 2,
            max_drafting_slots: 2,
            exported_logits_rows: 0,
            max_images_per_request: 1,
            lookahead: false,
        };
        let state = crate::ResourcePlanner::state_plan(
            &definition,
            &load,
            PlannedMethod::Plain,
            KvCodec::Dense,
            limits,
            crate::ResourceCapacity {
                domain_bytes: 512 * 1024 * 1024,
            },
        )
        .unwrap();
        let plan = load.program_plan(&definition, KvCodec::Dense).unwrap();
        let family = crate::programs::native_target_graph::checked_target_family_storage(
            BackendName::Cpu,
            &load,
            &definition.decoder,
            &state,
            plan.target(),
            limits,
        )
        .unwrap();
        let (block, _) = crate::programs::native_target_graph::checked_block_graph_resources(
            BackendName::Cpu,
            &load,
            &definition.decoder,
            &state,
            plan.target().blocks()[0],
            0,
            2,
            4,
            2,
        )
        .unwrap();
        assert!(family.storage.workspace >= block.workspace);
        assert!(family.storage.output >= block.output);
        assert!(family.storage.upload >= block.upload);
        assert!(family.storage.workspace > 0);
        assert!(family.storage.output > 0);
        let pools = AssessmentGraphResourceBounds::derive(
            &definition,
            &load,
            &state,
            PlannedMethod::Plain,
            KvCodec::Dense,
            limits,
            BackendName::Cpu,
        )
        .unwrap();
        assert_eq!(pools.target.workspace_bytes, family.storage.workspace);
        assert_eq!(pools.target.output_bytes, family.storage.output);
        assert_eq!(pools.target.upload_bytes, family.storage.upload);
        assert_eq!(pools.binding_constant_bytes, family.binding_constant_bytes);
        assert!(pools.binding_constant_bytes > 0);
        assert_eq!(
            pools.target.upload_regions,
            1 + definition.decoder.blocks.len()
        );
        assert!(pools.total_bytes >= pools.target.committed_bytes);
    }

    #[test]
    fn checked_head_family_includes_chained_and_shaped_classes() {
        use magnitude_artifacts::gguf::{Encoding, TensorDescriptor};
        use magnitude_family_contracts::{
            ActivationFunction, ExitNorm, ExpertSelection, FeedForwardUp, Head, HeadBlock,
            InputNorm, Operator, RmsNorm, RouteNormalization, RoutedFfn, Router, RouterInput,
            ScoreFunction, SharedExpert, SharedExpertGate, WeightDescriptor,
        };

        let mut definition = crate::planning::tests::fixture_definition();
        let descriptor =
            |name: &str, shape: &[u64]| WeightDescriptor::stored(format!("head_{name}"), shape);
        let rms = |name: &str| RmsNorm {
            weight: descriptor(name, &[128]),
            epsilon: 1e-6,
        };
        // The head block is the target block (attention, dense) under head
        // names, between its own norms and combine.
        let mut head_layer = definition.decoder.blocks[0].clone();
        let rename = |weight: &mut WeightDescriptor| weight.name = format!("head_{}", weight.name);
        for sublayer in &mut head_layer.sublayers {
            if let InputNorm::Rms(norm) = &mut sublayer.input {
                rename(&mut norm.weight);
            }
            match &mut sublayer.op {
                Operator::Attention(attention) => {
                    rename(&mut attention.query);
                    rename(&mut attention.output);
                    if let magnitude_family_contracts::HeadNorm::Rms(norm) =
                        &mut attention.query_norm
                    {
                        rename(&mut norm.weight);
                    }
                    if let magnitude_family_contracts::KeyValue::Owned {
                        key,
                        value: magnitude_family_contracts::ValueSource::Projected(value),
                        key_norm: magnitude_family_contracts::HeadNorm::Rms(key_norm),
                        ..
                    } = &mut attention.key_value
                    {
                        rename(key);
                        rename(value);
                        rename(&mut key_norm.weight);
                    }
                }
                Operator::DenseFfn(dense) => {
                    if let FeedForwardUp::Gated { gate, up, .. } = &mut dense.up {
                        rename(gate);
                        rename(up);
                    }
                    rename(&mut dense.down);
                }
                _ => unreachable!(),
            }
        }
        definition.head = Some(Head {
            blocks: vec![HeadBlock {
                embedding_norm: rms("embedding_norm"),
                hidden_norm: rms("hidden_norm"),
                combine: descriptor("combine", &[128, 256]),
                block: head_layer,
                output_norm: ExitNorm::Rms(rms("output_norm")),
            }],
        });
        let mut manifest = crate::planning::tests::fixture_manifest(&definition);
        let mut offset = manifest.target.files[0].size;
        let load = ModelLoadPlan::derive(
            &manifest,
            &definition,
            ComponentSelection {
                head: true,
                vision: false,
            },
            Layout::Rows16,
        )
        .unwrap();
        let method = PlannedMethod::Mtp {
            greedy_proposals: 1,
            sampled_proposals: 1,
        };
        let limits = crate::ResourceLimits {
            max_launch_rows: 16,
            max_launch_slots: 2,
            max_selected_rows: 2,
            max_drafting_slots: 2,
            exported_logits_rows: 0,
            max_images_per_request: 1,
            lookahead: false,
        };
        let state = crate::ResourcePlanner::state_plan(
            &definition,
            &load,
            method,
            KvCodec::Dense,
            limits,
            crate::ResourceCapacity {
                domain_bytes: 512 * 1024 * 1024,
            },
        )
        .unwrap();
        let plan = load.program_plan(&definition, KvCodec::Dense).unwrap();
        let head_block = definition.head.as_ref().unwrap().blocks[0].clone();
        let head_state = state.head_state().unwrap();
        let head_history = head_state.sole_history().unwrap();
        let classes = crate::programs::native_head::head_graph_classes(
            limits,
            head_history.rows as u64,
            head_history.slab_rows,
            head_state.span_limit(),
            method.draft_rows(),
        )
        .unwrap();
        assert!(classes.iter().any(|class| class.steps == 1 && class.shaped));
        assert!(classes
            .iter()
            .any(|class| class.entry_rows > crate::operators::routed::fused_graph::DECODE_ROWS));
        let family = crate::programs::native_head::checked_head_family_storage(
            BackendName::Cpu,
            &load,
            &definition.decoder,
            &head_block,
            plan.head().unwrap().blocks()[0],
            classes.clone(),
        )
        .unwrap();
        crate::programs::native_head::verify_head_family_certificates(
            BackendName::Cpu,
            &load,
            &definition.decoder,
            &head_block,
            plan.head().unwrap().blocks()[0],
            &classes,
        )
        .unwrap();
        assert!(family.storage.workspace > 0);
        assert!(family.storage.output > 0);
        let pools = AssessmentGraphResourceBounds::derive(
            &definition,
            &load,
            &state,
            method,
            KvCodec::Dense,
            limits,
            BackendName::Cpu,
        )
        .unwrap();
        assert_eq!(
            pools.head.unwrap().workspace_bytes,
            family.storage.workspace
        );
        let target = crate::programs::native_target_graph::checked_target_family_storage(
            BackendName::Cpu,
            &load,
            &definition.decoder,
            &state,
            plan.target(),
            limits,
        )
        .unwrap();
        assert_eq!(
            pools.binding_constant_bytes,
            target.binding_constant_bytes + family.binding_constant_bytes
        );

        let descriptor = |name: &str, shape: &[u64]| {
            WeightDescriptor::stored(format!("routed_head_{name}"), shape)
        };
        let silu = |gate, up| FeedForwardUp::Gated {
            activation: ActivationFunction::Silu,
            gate,
            up,
        };
        let routed = RoutedFfn {
            experts: 4,
            selected: 2,
            intermediate: 64,
            router: Router {
                weight: descriptor("router", &[4, 128]),
                input: RouterInput::Operator,
                score: ScoreFunction::Softmax,
                selection: ExpertSelection::TopK { bias: None },
                normalization: RouteNormalization::Sum,
                scale: 1.0,
            },
            expert_up: silu(
                descriptor("expert_gate", &[4, 64, 128]),
                descriptor("expert_up", &[4, 64, 128]),
            ),
            expert_down: descriptor("expert_down", &[4, 128, 64]),
            expert_scale: None,
            latent: None,
            shared: Some(SharedExpert {
                intermediate: 64,
                up: silu(
                    descriptor("shared_gate", &[64, 128]),
                    descriptor("shared_up", &[64, 128]),
                ),
                down: descriptor("shared_down", &[128, 64]),
                gate: SharedExpertGate::Sigmoid(descriptor("shared_router", &[128])),
            }),
        };
        for weight in [
            &routed.router.weight,
            routed.expert_up.gate().unwrap(),
            routed.expert_up.up(),
            &routed.expert_down,
            routed.shared.as_ref().unwrap().up.gate().unwrap(),
            routed.shared.as_ref().unwrap().up.up(),
            &routed.shared.as_ref().unwrap().down,
            match &routed.shared.as_ref().unwrap().gate {
                SharedExpertGate::Sigmoid(gate) => gate,
                SharedExpertGate::None => unreachable!(),
            },
        ] {
            let nbytes = weight.shape.iter().product::<u64>() * 2;
            manifest.target.tensors.push(TensorDescriptor {
                name: weight.name.clone(),
                shape: weight.shape.clone(),
                encoding: Encoding::F16,
                offset,
                nbytes,
            });
            offset += nbytes;
        }
        manifest.target.files[0].size = offset;
        let head = &mut definition.head.as_mut().unwrap().blocks[0];
        head.block.sublayers[1].op = Operator::RoutedFfn(Box::new(routed));
        let head_block = head.clone();
        let load = ModelLoadPlan::derive(
            &manifest,
            &definition,
            ComponentSelection {
                head: true,
                vision: false,
            },
            Layout::Rows16,
        )
        .unwrap();
        let plan = load.program_plan(&definition, KvCodec::Dense).unwrap();
        assert!(matches!(
            plan.head().unwrap().blocks()[0].feed_forward,
            crate::FeedForwardProgramSlot::Routed(_)
        ));
        let routed_family = crate::programs::native_head::checked_head_family_storage(
            BackendName::Cpu,
            &load,
            &definition.decoder,
            &head_block,
            plan.head().unwrap().blocks()[0],
            crate::programs::native_head::head_graph_classes(
                limits,
                head_history.rows as u64,
                head_history.slab_rows,
                head_state.span_limit(),
                method.draft_rows(),
            )
            .unwrap(),
        )
        .unwrap();
        assert!(routed_family.storage.workspace > 0);
        let routed_pools = AssessmentGraphResourceBounds::derive(
            &definition,
            &load,
            &state,
            method,
            KvCodec::Dense,
            limits,
            BackendName::Cpu,
        )
        .unwrap();
        assert_eq!(
            routed_pools.head.unwrap().workspace_bytes,
            routed_family.storage.workspace
        );
    }

    #[test]
    fn checked_feature_readout_charges_only_its_export_and_upload() {
        use magnitude_family_contracts::{WeightKind, WeightRole, WeightScope};

        let definition = crate::planning::tests::fixture_definition();
        let manifest = crate::planning::tests::fixture_manifest(&definition);
        let load = ModelLoadPlan::derive(
            &manifest,
            &definition,
            ComponentSelection {
                head: false,
                vision: false,
            },
            Layout::Rows16,
        )
        .unwrap();
        let norm = load
            .weights()
            .find(|weight| {
                weight.role
                    == WeightRole {
                        scope: WeightScope::Target,
                        kind: WeightKind::OutputNorm,
                    }
            })
            .unwrap();
        let storage = crate::programs::graph::readout::checked_features_graph_storage(
            BackendName::Cpu,
            &definition.decoder,
            norm,
            2,
            1,
        )
        .unwrap();
        assert_eq!(storage.workspace, 0);
        assert_eq!(storage.upload, 4);
        assert!(storage.output > 0);
    }

    #[test]
    fn checked_projected_readout_includes_native_scratch() {
        use magnitude_family_contracts::{WeightKind, WeightRole, WeightScope};

        let definition = crate::planning::tests::fixture_definition();
        let manifest = crate::planning::tests::fixture_manifest(&definition);
        let load = ModelLoadPlan::derive(
            &manifest,
            &definition,
            ComponentSelection {
                head: false,
                vision: false,
            },
            Layout::Rows16,
        )
        .unwrap();
        let weight = |kind| {
            load.weights()
                .find(|weight| {
                    weight.role
                        == WeightRole {
                            scope: WeightScope::Target,
                            kind,
                        }
                })
                .unwrap()
        };
        let features = crate::programs::graph::readout::checked_features_graph_storage(
            BackendName::Cpu,
            &definition.decoder,
            weight(WeightKind::OutputNorm),
            2,
            1,
        )
        .unwrap();
        let projected = crate::programs::graph::readout::checked_projected_graph_storage(
            BackendName::Cpu,
            &definition.decoder,
            weight(WeightKind::OutputNorm),
            weight(WeightKind::Output),
            2,
            1,
            1,
        )
        .unwrap();
        assert!(projected.workspace > features.workspace);
        assert!(projected.output > features.output);
        assert!(projected.upload > features.upload);
    }

    #[test]
    fn checked_selection_readout_charges_shaping_controls() {
        use magnitude_family_contracts::{WeightKind, WeightRole, WeightScope};

        let definition = crate::planning::tests::fixture_definition();
        let manifest = crate::planning::tests::fixture_manifest(&definition);
        let load = ModelLoadPlan::derive(
            &manifest,
            &definition,
            ComponentSelection {
                head: false,
                vision: false,
            },
            Layout::Rows16,
        )
        .unwrap();
        let weight = |kind| {
            load.weights()
                .find(|weight| {
                    weight.role
                        == WeightRole {
                            scope: WeightScope::Target,
                            kind,
                        }
                })
                .unwrap()
        };
        let selection = |shaped| {
            crate::programs::graph::readout::checked_selection_graph_storage(
                BackendName::Cpu,
                &definition.decoder,
                weight(WeightKind::OutputNorm),
                weight(WeightKind::Output),
                2,
                1,
                1,
                shaped,
            )
            .unwrap()
        };
        let plain = selection(false);
        let shaped = selection(true);
        assert!(shaped.workspace > plain.workspace);
        assert_eq!(shaped.output, plain.output);
        assert!(shaped.upload > plain.upload);
    }

    #[test]
    fn checked_readout_family_bounds_its_selection_classes() {
        let definition = crate::planning::tests::fixture_definition();
        let manifest = crate::planning::tests::fixture_manifest(&definition);
        let load = ModelLoadPlan::derive(
            &manifest,
            &definition,
            ComponentSelection {
                head: false,
                vision: false,
            },
            Layout::Rows16,
        )
        .unwrap();
        let limits = crate::ResourceLimits {
            max_launch_rows: 2,
            max_launch_slots: 2,
            max_selected_rows: 2,
            max_drafting_slots: 2,
            exported_logits_rows: 0,
            max_images_per_request: 0,
            lookahead: false,
        };
        let family = crate::programs::graph::readout::checked_readout_family_storage(
            BackendName::Cpu,
            &load,
            &definition.decoder,
            limits,
        )
        .unwrap();
        assert!(family.workspace > 0);
        assert!(family.output > 0);
        assert!(family.upload > 0);
    }

    #[test]
    fn checked_vision_family_bounds_exact_patch_classes() {
        use crate::{ArtifactComponent, ArtifactComponentKind, VisionProgramPlan, WeightPlan};
        use magnitude_artifacts::ArtifactIdentity;
        use magnitude_family_contracts::{
            ActivationDType, CellReduction, MergerStage, PositionSampling, VisionActivation,
            VisionAttention, VisionAttentionScale, VisionAttentionSpan, VisionBlock,
            VisionDescription, VisionFeedForward, VisionLinear, VisionMerger, VisionNorm,
            VisionPositions, VisionPreprocessing, VisionResampling, VisionResize, VisionStem,
            VisionUp, WeightDescriptor,
        };

        let f32 = Element::f32();
        let stored = |name: &str, shape: &[u64]| WeightDescriptor::stored(name, shape);
        let linear = |name: &str, outputs: u64, inputs: u64| VisionLinear {
            weight: stored(&format!("{name}.weight"), &[outputs, inputs]),
            bias: Some(stored(&format!("{name}.bias"), &[outputs])),
            clamp: None,
        };
        let layer_norm = |name: &str, width: u64| VisionNorm::Layer {
            weight: stored(&format!("{name}.weight"), &[width]),
            bias: stored(&format!("{name}.bias"), &[width]),
            epsilon: 1e-6,
        };
        // A Qwen3-VL-shaped tower at width 128: one block of one 128-wide
        // head, 2 x 2 cells.
        let description = VisionDescription {
            activation_dtype: ActivationDType::BF16,
            hidden: 128,
            output_hidden: 128,
            preprocessing: VisionPreprocessing {
                resize: VisionResize::PixelBounds {
                    min_pixels: 16,
                    max_pixels: 64,
                },
                resampling: VisionResampling::Bicubic,
                mean: [0.5; 3],
                std: [0.5; 3],
                channels: 3,
                patch: 2,
                merge: 2,
            },
            stem: VisionStem::Patch {
                frames: vec![
                    stored("patch.0", &[128, 3, 2, 2]),
                    stored("patch.1", &[128, 3, 2, 2]),
                ],
                bias: Some(stored("patch.bias", &[128])),
                positions: VisionPositions {
                    table: stored("position", &[16, 128]),
                    sampling: PositionSampling::AlignedCorners { side: 4 },
                },
                norm: None,
            },
            window: None,
            blocks: vec![VisionBlock {
                attention_norm: layer_norm("ln1", 128),
                attention: VisionAttention {
                    heads: 1,
                    width: 128,
                    query: linear("q", 128, 128),
                    key: linear("k", 128, 128),
                    value: linear("v", 128, 128),
                    query_norm: None,
                    key_norm: None,
                    value_norm: None,
                    rotary_base: 10000.0,
                    scale: VisionAttentionScale::InverseSqrtWidth,
                    span: VisionAttentionSpan::Full,
                    output: linear("o", 128, 128),
                },
                attention_post_norm: None,
                feedforward_norm: layer_norm("ln2", 128),
                feedforward: VisionFeedForward {
                    up: VisionUp::Plain(linear("up", 128, 128)),
                    activation: VisionActivation::GeluTanh,
                    down: linear("down", 128, 128),
                },
                feedforward_post_norm: None,
            }],
            merger: VisionMerger {
                norm: Some(layer_norm("post", 128)),
                reduction: CellReduction::Concatenate,
                standardize: None,
                projection_norm: None,
                stages: vec![
                    MergerStage {
                        linear: linear("mm.0", 512, 512),
                        activation: Some(VisionActivation::GeluErf),
                    },
                    MergerStage {
                        linear: linear("mm.2", 128, 512),
                        activation: None,
                    },
                ],
                output_norm: None,
            },
        };
        let component = ArtifactComponent {
            kind: ArtifactComponentKind::Projector,
            identity: ArtifactIdentity([9; 32]),
        };
        let weights = description
            .weights()
            .into_iter()
            .map(|(role, descriptor)| {
                let bytes = descriptor.shape.iter().product::<u64>() * 4;
                WeightPlan {
                    role,
                    component,
                    source: f32,
                    upload: f32,
                    resident: f32,
                    shape: descriptor.shape.clone(),
                    descriptor: descriptor.clone(),
                    source_bytes: bytes,
                    resident_bytes: bytes,
                    scale: None,
                }
            })
            .collect();
        let load = ModelLoadPlan {
            target: Vec::new(),
            head: None,
            vision: Some(weights),
            host_tables: Vec::new(),
        };
        let program = crate::operators::vision::vision_program(&description, &|_| Ok(f32)).unwrap();
        let plan = VisionProgramPlan::new(program.kernels().into_iter().cloned().collect());
        let resources = |classes: &[u64]| {
            crate::programs::native_vision::checked_vision_family_resources(
                BackendName::Cpu,
                &load,
                &description,
                &plan,
                classes.iter().copied(),
            )
            .unwrap()
            .storage
        };
        let single = resources(&[4]);
        let double = resources(&[8]);
        let family = resources(&[4, 8]);
        crate::programs::native_vision::verify_vision_family_certificates(
            BackendName::Cpu,
            &load,
            &description,
            &plan,
            &[4, 8],
        )
        .unwrap();
        assert_eq!(family.workspace, single.workspace.max(double.workspace));
        assert_eq!(family.output, single.output.max(double.output));
        assert_eq!(family.upload, single.upload.max(double.upload));
        assert!(family.workspace > 0 && family.output > 0 && family.upload > 0);
    }

    fn fixture_limits(lookahead: bool) -> crate::ResourceLimits {
        crate::ResourceLimits {
            max_launch_rows: 2,
            max_launch_slots: 2,
            max_selected_rows: 2,
            max_drafting_slots: 2,
            exported_logits_rows: 0,
            max_images_per_request: 1,
            lookahead,
        }
    }

    #[test]
    fn header_terms_include_exact_selected_weights_history_and_bounds() {
        let mut definition = crate::planning::tests::fixture_definition();
        let manifest = crate::planning::tests::fixture_manifest(&definition);
        let selection = ComponentSelection {
            head: false,
            vision: false,
        };
        let load =
            ModelLoadPlan::derive(&manifest, &definition, selection, Layout::Rows16).unwrap();
        let limits = fixture_limits(false);
        let terms = AssessmentMemoryTerms::derive(
            &definition,
            &load,
            selection,
            KvCodec::Dense,
            PlannedMethod::Plain,
            limits,
        )
        .unwrap();

        assert_eq!(
            terms.target_weights,
            weight_bytes_by_component(&load).unwrap()[0]
        );
        assert_eq!(terms.head_weights, 0);
        assert_eq!(terms.vision_weights, 0);
        assert_eq!(terms.history_per_token, 2 * 64 * 2);
        assert_eq!(terms.recurrent_per_bank, 0);
        // Accepted bank, one in-flight successor, pristine seed.
        assert_eq!(terms.recurrent_banks, 3);
        assert_eq!(terms.fit_depth, 128);
        assert_eq!(terms.history_at_fit_depth().unwrap(), 128 * 2 * 64 * 2);
        let pipelined = AssessmentMemoryTerms::derive(
            &definition,
            &load,
            selection,
            KvCodec::Dense,
            PlannedMethod::Plain,
            fixture_limits(true),
        )
        .unwrap();
        assert_eq!(pipelined.recurrent_banks, 4);
        assert!(AssessmentMemoryTerms::derive(
            &definition,
            &load,
            selection,
            KvCodec::Dense,
            PlannedMethod::Mtp {
                greedy_proposals: 1,
                sampled_proposals: 1,
            },
            limits,
        )
        .is_err());

        let header =
            AssessmentHeaderBounds::derive(&definition, &load, KvCodec::Dense, BackendName::Cpu)
                .unwrap();
        assert_eq!(
            header.prepared_program_bytes,
            AttestedPrograms::planned_invocation_workspace_bytes(
                &load.program_plan(&definition, KvCodec::Dense).unwrap(),
                BackendName::Cpu
            )
            .unwrap()
        );
        assert_eq!(
            header.startup_additional_bytes,
            AttestedPrograms::qualification_peak_bytes(&load)
                .unwrap()
                .max(load.target_upload_peak_bytes().unwrap())
        );
        assert_eq!(
            header.staging_upload_bytes,
            load.target_upload_peak_bytes().unwrap()
        );
        let state = crate::ResourcePlanner::state_plan(
            &definition,
            &load,
            PlannedMethod::Plain,
            KvCodec::Dense,
            limits,
            crate::ResourceCapacity {
                domain_bytes: 512 * 1024 * 1024,
            },
        )
        .unwrap();
        let graph = AssessmentGraphResourceBounds::derive(
            &definition,
            &load,
            &state,
            PlannedMethod::Plain,
            KvCodec::Dense,
            limits,
            BackendName::Cpu,
        )
        .unwrap();
        let bounds = header.with_graph_resource_bound(&graph).unwrap();
        assert_eq!(
            bounds.prepared_resource_bytes,
            header.prepared_program_bytes + graph.total_bytes
        );
        assert_eq!(
            bounds.startup_additional_bytes,
            header.startup_additional_bytes
        );
        let state_bytes = state
            .fit_state_bytes(terms.fit_depth, terms.recurrent_banks)
            .unwrap();
        let history = state.target_state().sole_history().unwrap();
        let history_layout = history.slab_layout().unwrap();
        let history_slabs = terms.fit_depth.div_ceil(u64::from(history.slab_rows));
        assert_eq!(
            state_bytes,
            history_layout.address_table_bytes + history_slabs * history_layout.slab_bytes
        );
        assert!(state_bytes > terms.history_at_fit_depth().unwrap());
        let startup_state = state.startup_state_bytes().unwrap();
        let charge = terms.charge(bounds, state_bytes, startup_state).unwrap();
        assert_eq!(
            charge.allocation_bytes,
            terms.exact_resident_bytes(0).unwrap()
                + bounds.prepared_resource_bytes
                + state_bytes.max(startup_state + bounds.startup_additional_bytes)
        );
        assert_eq!(charge.staging_bytes, header.staging_upload_bytes);

        // The fit depth is the context limit capped at 100,000 tokens.
        definition.decoder.context_limit = 262_144;
        let long = AssessmentMemoryTerms::derive(
            &definition,
            &load,
            selection,
            KvCodec::Dense,
            PlannedMethod::Plain,
            limits,
        )
        .unwrap();
        assert_eq!(long.fit_depth, 100_000);
        assert_eq!(
            long.history_at_fit_depth().unwrap(),
            100_000 * terms.history_per_token
        );
    }

    fn fixture_charge(allocation_bytes: u64, staging_bytes: u64) -> AssessmentMemoryCharge {
        AssessmentMemoryCharge {
            exact_resident_bytes: allocation_bytes,
            bounds: AssessmentMemoryBounds {
                prepared_resource_bytes: 0,
                startup_additional_bytes: 0,
                staging_upload_bytes: staging_bytes,
            },
            allocation_bytes,
            staging_bytes,
            host_table_bytes: 0,
        }
    }

    #[test]
    fn workload_charge_composes_exact_terms_and_upper_bounds() {
        let terms = AssessmentMemoryTerms {
            target_weights: 100,
            head_weights: 20,
            vision_weights: 5,
            history_per_token: 2,
            recurrent_per_bank: 10,
            recurrent_banks: 3,
            fit_depth: 10,
            host_table_bytes: 30,
        };
        assert_eq!(terms.exact_resident_bytes(50).unwrap(), 100 + 20 + 5 + 50);
        let bounds = AssessmentMemoryBounds {
            prepared_resource_bytes: 40,
            startup_additional_bytes: 60,
            staging_upload_bytes: 7,
        };
        // The workload's state outgrows the load's startup state and peak:
        // the peak is not charged on top of it.
        let charge = terms.charge(bounds, 90, 10).unwrap();
        assert_eq!(charge.exact_resident_bytes, 100 + 20 + 5 + 90);
        assert_eq!(charge.allocation_bytes, 125 + 40 + 90);
        assert_eq!(charge.staging_bytes, 7);
        // The load's startup state and peak exceed the workload's state.
        let loading = terms.charge(bounds, 50, 10).unwrap();
        assert_eq!(loading.allocation_bytes, 125 + 40 + (10 + 60));
        assert!(terms.charge(bounds, u64::MAX, 10).is_err());
        assert!(terms.charge(bounds, 50, u64::MAX).is_err());
        assert!(AssessmentMemoryTerms {
            history_per_token: u64::MAX,
            ..terms
        }
        .history_at_fit_depth()
        .is_err());
    }

    #[test]
    fn fit_is_decided_per_domain_against_capacity_less_reserve() {
        let host = seismic::DeviceCatalog::discover()
            .unwrap()
            .topology()
            .host_pool()
            .id;
        let domain = |capacity_bytes, reserve_bytes| FitCapacity {
            domain: host,
            kind: seismic::MemoryPoolKind::HostRam,
            capacity_bytes,
            reserve_bytes,
        };
        // Unified memory: one allocation domain.
        let unified = [(DomainRole::Allocation, domain(1_000, 100))];
        let fits = fixture_charge(900, 0).assess_fit(&unified).unwrap();
        assert_eq!(fits.verdict, AssessmentFitVerdict::Fits);
        assert_eq!(
            fits.domains,
            vec![DomainFit {
                role: DomainRole::Allocation,
                domain: host,
                capacity_bytes: 1_000,
                required_bytes: 900,
                reserve_bytes: 100,
                remaining_bytes: 0,
            }]
        );
        let short = fixture_charge(901, 0).assess_fit(&unified).unwrap();
        assert_eq!(
            short.verdict,
            AssessmentFitVerdict::DoesNotFit {
                limiting: host,
                deficit_bytes: 1,
            }
        );
        assert_eq!(short.domains[0].remaining_bytes, -1);

        // A dedicated device also charges its staged upload to host RAM.
        let dedicated = [
            (DomainRole::Allocation, domain(1_000, 100)),
            (DomainRole::Staging, domain(500, 200)),
        ];
        let both = fixture_charge(800, 300).assess_fit(&dedicated).unwrap();
        assert_eq!(both.verdict, AssessmentFitVerdict::Fits);
        assert_eq!(both.domains[1].required_bytes, 300);
        assert_eq!(both.domains[1].remaining_bytes, 0);
        let staging = fixture_charge(800, 350).assess_fit(&dedicated).unwrap();
        assert_eq!(
            staging.verdict,
            AssessmentFitVerdict::DoesNotFit {
                limiting: host,
                deficit_bytes: 50,
            }
        );
        assert_eq!(staging.domains[0].remaining_bytes, 100);
        assert_eq!(staging.domains[1].remaining_bytes, -50);
        // The largest deficit limits the load.
        let worst = fixture_charge(1_000, 310).assess_fit(&dedicated).unwrap();
        assert_eq!(
            worst.verdict,
            AssessmentFitVerdict::DoesNotFit {
                limiting: host,
                deficit_bytes: 100,
            }
        );
        // A reserve above capacity leaves a negative remainder, not an error.
        let tiny = [(DomainRole::Allocation, domain(100, 200))];
        assert_eq!(
            fixture_charge(1, 0).assess_fit(&tiny).unwrap().verdict,
            AssessmentFitVerdict::DoesNotFit {
                limiting: host,
                deficit_bytes: 101,
            }
        );
        assert!(fixture_charge(1, 0)
            .assess_fit(&[(DomainRole::Staging, domain(100, 0))])
            .is_err());

        // Host tables are system RAM: charged to a unified device's
        // allocation domain, and to a dedicated device's staging domain.
        let tables = AssessmentMemoryCharge {
            host_table_bytes: 50,
            ..fixture_charge(800, 300)
        };
        let unified_tables = tables.assess_fit(&unified).unwrap();
        assert_eq!(unified_tables.domains[0].required_bytes, 850);
        let dedicated_tables = tables.assess_fit(&dedicated).unwrap();
        assert_eq!(dedicated_tables.domains[0].required_bytes, 800);
        assert_eq!(dedicated_tables.domains[1].required_bytes, 350);
    }
}
