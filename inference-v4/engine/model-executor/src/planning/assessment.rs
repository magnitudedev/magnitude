//! Exact model-derived memory terms for metadata-only assessment.
//!
//! This is deliberately a demand description, not a fit verdict: scratch,
//! startup transients and the capacity of a particular device are separate
//! inputs. The load planner and the state layout remain the authorities for
//! physical representations and codec planes.

use super::resources::{history_row_bytes, recurrent_bank_bytes, NativeGraphCharge};
use super::{
    source_import_peak_bytes, weight_bytes_by_component, ComponentSelection, ModelLoadPlan,
    PlannedMethod,
};
use crate::AttestedPrograms;
use magnitude_model_contracts::ModelDefinition;
use magnitude_model_kernels::{dense_expand, dense_output, import_dense, repack_weight};
use magnitude_model_state::{BankCapacity, KvCodec, ModelStateLayout};
use seismic::{BackendName, Element};
use std::collections::HashSet;

/// Header-only evidence for a native binding. Checked source can reject an
/// exact binding; successful source checking cannot prove device formation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AssessmentBindingEvidence {
    Unsupported { entry: &'static str, reason: String },
    Unconfirmed,
}

impl AssessmentBindingEvidence {
    /// Inspect import and dense feed-forward bindings from the model's one
    /// program plan. Other slots, native source compilation and device facts
    /// still require preparation, so a positive model verdict is unavailable.
    pub fn inspect_dense_paths(
        definition: &ModelDefinition,
        load: &ModelLoadPlan,
        codec: KvCodec,
        backend: BackendName,
    ) -> Result<Self, String> {
        use super::{FeedForwardProgramSlot, ImportProgramSlot};

        fn check<E: seismic::Entry>(
            backend: BackendName,
            elements: &[(&str, Element)],
        ) -> Result<AssessmentBindingEvidence, String> {
            match seismic::generated::checked_native_binding::<E>(backend, elements)
                .map_err(|error| error.to_string())?
            {
                seismic::NativeBindingCheck::AcceptedByCheckedEntry => {
                    Ok(AssessmentBindingEvidence::Unconfirmed)
                }
                seismic::NativeBindingCheck::Unsupported(reason) => {
                    Ok(AssessmentBindingEvidence::Unsupported {
                        entry: E::NAME,
                        reason,
                    })
                }
            }
        }

        let plan = load
            .program_plan(definition, codec)
            .map_err(|error| error.to_string())?;
        for slot in plan.imports() {
            let evidence = match *slot {
                ImportProgramSlot::Dense { source, resident } => check::<import_dense::Entry>(
                    backend,
                    &[
                        ("E", Element::dense(source)),
                        ("U", Element::dense(resident)),
                    ],
                )?,
                ImportProgramSlot::Repack { source, resident } => {
                    check::<repack_weight::Entry>(backend, &[("E", source), ("U", resident)])?
                }
            };
            if matches!(evidence, Self::Unsupported { .. }) {
                return Ok(evidence);
            }
        }
        let mut checked_dense = HashSet::new();
        for block in plan.target().blocks() {
            let FeedForwardProgramSlot::Dense(binding) = block.feed_forward() else {
                continue;
            };
            if !checked_dense.insert(binding) {
                continue;
            }
            let expand = check::<dense_expand::Entry>(
                backend,
                &[
                    ("NW", binding.norm),
                    ("GW", binding.gate),
                    ("UW", binding.up),
                    ("A", binding.activation),
                ],
            )?;
            if matches!(expand, Self::Unsupported { .. }) {
                return Ok(expand);
            }
            let output = check::<dense_output::Entry>(
                backend,
                &[("DW", binding.down), ("A", binding.activation)],
            )?;
            if matches!(output, Self::Unsupported { .. }) {
                return Ok(output);
            }
        }
        Ok(Self::Unconfirmed)
    }
}

/// Exact resident and per-state terms derived without opening a device or
/// reading tensor payloads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AssessmentMemoryTerms {
    pub target_weights: u64,
    pub head_weights: u64,
    pub vision_weights: u64,
    pub history_per_token: u64,
    pub recurrent_per_bank: u64,
    pub fit_depth: u64,
}

/// Conservative charges for memory that cannot yet be derived from model
/// headers alone. The prepared-resource bound covers all nonresident steady
/// holdings (including scratch, argument, upload and output storage); the
/// startup bound covers additional peak bytes. Both must cover the selected
/// backend and workload.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AssessmentMemoryBounds {
    pub prepared_resource_bytes: u64,
    pub startup_additional_bytes: u64,
}

/// Header-derived charges that are known before graph formation. Prepared
/// native invocation storage is exact; the startup charge is the same
/// qualification/import upper bound used by the production resource planner.
/// Graph pools still need a backend and workload-specific upper bound.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AssessmentHeaderBounds {
    pub prepared_program_bytes: u64,
    pub startup_additional_bytes: u64,
}

/// Checked graph-pool demand and distinct bound constants from the same class
/// enumeration and slot multipliers as native preparation. It does not include
/// runtime external pins or evidence that the backend forms the checked
/// entries, so it cannot by itself authorize a positive fit verdict.
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
    ) -> Result<Self, String> {
        let plan = load
            .program_plan(definition, codec)
            .map_err(|error| error.to_string())?;
        if plan.head().is_some() != matches!(method, PlannedMethod::Mtp { .. }) {
            return Err("assessment method and head program disagree".into());
        }
        let target_launches = limits
            .in_flight_requests
            .checked_add(usize::from(limits.lookahead))
            .ok_or("target launch count overflow")?;
        let target_graph = crate::programs::native_target_graph::checked_target_family_storage(
            backend,
            load,
            &definition.geometry,
            state,
            plan.target(),
            limits,
        )?;
        let target = NativeGraphCharge::from_checked(
            target_graph.storage,
            1 + definition.geometry.blocks.len(),
            target_launches,
            target_launches
                .checked_mul(2)
                .ok_or("target output slot count overflow")?,
        )?;
        let readout = NativeGraphCharge::from_checked(
            crate::programs::graph::readout::checked_readout_family_storage(
                backend,
                load,
                &definition.geometry,
                limits,
            )?,
            1,
            target_launches,
            limits
                .active_requests
                .checked_add(target_launches)
                .ok_or("readout output slot count overflow")?,
        )?;
        let (head, head_constant_bytes) = match (plan.head(), state.head_state()) {
            (Some(head), Some(head_state)) => {
                let binding = *head.blocks().first().ok_or("head program has no block")?;
                let attention = definition
                    .geometry
                    .blocks
                    .iter()
                    .rev()
                    .find_map(|block| match &block.mixer {
                        magnitude_model_contracts::MixerGeometry::Attention(shape) => Some(shape),
                        _ => None,
                    })
                    .ok_or("head graph requires target attention geometry")?;
                let history_rows = u64::try_from(head_state.history_rows)
                    .map_err(|_| "head history rows exceed u64")?;
                let head_graph = crate::programs::native_head::checked_head_family_storage(
                    backend,
                    load,
                    &definition.geometry,
                    attention,
                    binding,
                    crate::programs::native_head::head_graph_classes(
                        limits,
                        history_rows,
                        method.draft_rows(),
                    )?,
                )?;
                (
                    Some(NativeGraphCharge::from_checked(
                        head_graph.storage,
                        1,
                        limits.in_flight_requests,
                        limits
                            .active_requests
                            .checked_add(limits.in_flight_requests)
                            .ok_or("head output slot count overflow")?,
                    )?),
                    head_graph.binding_constant_bytes,
                )
            }
            (None, None) => (None, 0),
            _ => return Err("head program and state disagree".into()),
        };
        let vision = match (plan.vision(), definition.vision.as_ref()) {
            (Some(vision_plan), Some(vision_definition)) => {
                let max_rows = magnitude_model_batching::row_classes(limits.max_batch_rows)
                    .last()
                    .copied()
                    .ok_or("batch row bound has no class")? as u64;
                let merge = vision_definition
                    .geometry
                    .merge
                    .checked_mul(vision_definition.geometry.merge)
                    .ok_or("vision merge area overflow")?;
                max_rows
                    .checked_mul(merge)
                    .ok_or("vision patch row bound overflow")?;
                let patch_rows = (1..=max_rows).map(|rows| rows * merge);
                Some(NativeGraphCharge::from_checked(
                    crate::programs::native_vision::checked_vision_family_storage(
                        backend,
                        load,
                        &vision_definition.geometry,
                        definition.geometry.hidden,
                        vision_plan,
                        patch_rows,
                    )?,
                    1,
                    limits.in_flight_requests,
                    state.retention().retained_media_features,
                )?)
            }
            (None, None) => None,
            _ => return Err("vision program and definition disagree".into()),
        };
        let row_classes = magnitude_model_batching::row_classes(limits.max_batch_rows)
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
            limits.in_flight_requests,
            limits.in_flight_requests,
        )?;
        let binding_constant_bytes = target_graph
            .binding_constant_bytes
            .checked_add(head_constant_bytes)
            .ok_or("checked graph constant charge overflow")?;
        let total_bytes = target
            .committed_bytes
            .checked_add(readout.committed_bytes)
            .and_then(|bytes| bytes.checked_add(head.map_or(0, |charge| charge.committed_bytes)))
            .and_then(|bytes| bytes.checked_add(vision.map_or(0, |charge| charge.committed_bytes)))
            .and_then(|bytes| bytes.checked_add(state.committed_bytes))
            .and_then(|bytes| bytes.checked_add(binding_constant_bytes))
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
    ) -> Result<Self, String> {
        let programs = load
            .program_plan(definition, codec)
            .map_err(|error| error.to_string())?;
        let prepared_program_bytes =
            AttestedPrograms::planned_invocation_workspace_bytes(&programs)
                .map_err(|error| error.to_string())?;
        let largest_source = load
            .weights()
            .map(|weight| weight.source_bytes)
            .max()
            .unwrap_or(0);
        let import_peak = source_import_peak_bytes(largest_source)?;
        let startup_additional_bytes =
            AttestedPrograms::qualification_peak_bytes(load).max(import_peak);
        Ok(Self {
            prepared_program_bytes,
            startup_additional_bytes,
        })
    }

    /// Complete the memory bound once the Seismic graph pools have a proven
    /// upper bound for the same backend and one-conversation workload.
    pub fn with_graph_resource_bound(
        self,
        graph_resource_bytes: u64,
    ) -> Result<AssessmentMemoryBounds, String> {
        Ok(AssessmentMemoryBounds {
            prepared_resource_bytes: self
                .prepared_program_bytes
                .checked_add(graph_resource_bytes)
                .ok_or("assessment prepared resource bound overflow")?,
            startup_additional_bytes: self.startup_additional_bytes,
        })
    }
}

/// A positive fit verdict is possible only with bounds for every nonresident
/// charge. An upper bound above capacity is inconclusive: actual use may fit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AssessmentFitVerdict {
    DoesNotFit,
    FitsWithinBounds,
    Unconfirmed,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AssessmentFit {
    pub verdict: AssessmentFitVerdict,
    pub exact_resident_bytes: u64,
    /// Exact resident bytes plus any separately proven fixed holdings.
    pub exact_required_lower_bound_bytes: u64,
    pub conservative_required_bytes: Option<u64>,
    pub capacity_bytes: u64,
}

/// Cost of one shipped default streaming-kernel launch, inferred from two
/// device measurements at different resident byte counts. This keeps the
/// fixed launch cost separate from weight traffic in a decode prediction.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StreamingCost {
    pub launch_seconds: f64,
    pub seconds_per_byte: f64,
}

impl StreamingCost {
    pub fn from_samples(
        small_bytes: u64,
        small_seconds: f64,
        large_bytes: u64,
        large_seconds: f64,
    ) -> Result<Self, String> {
        if small_bytes == 0
            || large_bytes <= small_bytes
            || !small_seconds.is_finite()
            || !large_seconds.is_finite()
            || small_seconds <= 0.0
            || large_seconds <= small_seconds
        {
            return Err("streaming measurements do not establish a positive size slope".into());
        }
        let seconds_per_byte = (large_seconds - small_seconds) / (large_bytes - small_bytes) as f64;
        let launch_seconds = small_seconds - seconds_per_byte * small_bytes as f64;
        if !seconds_per_byte.is_finite()
            || seconds_per_byte <= 0.0
            || !launch_seconds.is_finite()
            || launch_seconds < 0.0
        {
            return Err("streaming measurements do not establish a nonnegative launch cost".into());
        }
        Ok(Self {
            launch_seconds,
            seconds_per_byte,
        })
    }

    pub fn predict(self, bytes: u64) -> Result<f64, String> {
        let seconds = self.launch_seconds + self.seconds_per_byte * bytes as f64;
        if seconds.is_finite() && seconds > 0.0 {
            Ok(seconds)
        } else {
            Err("streaming prediction is outside the finite positive time domain".into())
        }
    }
}

impl AssessmentMemoryTerms {
    /// The standard assessment workload is one conversation at the lesser
    /// of the supported context and 100,000 tokens. Method state and optional
    /// components are included only when selected by the planned method.
    pub fn derive(
        definition: &ModelDefinition,
        load: &ModelLoadPlan,
        selection: ComponentSelection,
        codec: KvCodec,
        method: PlannedMethod,
    ) -> Result<Self, String> {
        if selection.head != matches!(method, PlannedMethod::Mtp { .. }) {
            return Err("assessment method and head selection disagree".into());
        }
        if selection.head != load.head().is_some() || selection.vision != load.vision().is_some() {
            return Err("assessment selection disagrees with the load plan".into());
        }
        let weights = weight_bytes_by_component(load)?;
        let head_depth = if selection.head {
            definition
                .head
                .as_ref()
                .ok_or("selected draft head is absent from the model definition")?
                .depth()
        } else {
            0
        };
        let layout =
            ModelStateLayout::derive(&definition.geometry, head_depth, codec, method.draft_rows())?;
        let history_per_token = history_row_bytes(&layout.target_history)?
            .checked_add(history_row_bytes(&layout.head_history)?)
            .ok_or("history row bytes overflow")?;
        let recurrent_per_bank = recurrent_bank_bytes(&layout.target_recurrent)?;
        Ok(Self {
            target_weights: weights[0],
            head_weights: weights[1],
            vision_weights: weights[2],
            history_per_token,
            recurrent_per_bank,
            fit_depth: definition.geometry.context_limit.min(100_000),
        })
    }

    pub fn history_at_fit_depth(self) -> Result<u64, String> {
        self.history_per_token
            .checked_mul(self.fit_depth)
            .ok_or_else(|| "assessment history bytes overflow".to_owned())
    }

    /// One active conversation needs its accepted bank, one in-flight
    /// successor and the permanently pristine zero seed. Use the state
    /// store's capacity rule rather than a separate assessment constant.
    pub fn recurrent_at_fit_workload(self) -> Result<u64, String> {
        let banks = BankCapacity {
            active: 1,
            in_flight: 1,
            retained: 0,
        }
        .storage_total()
        .map_err(|error| error.to_string())?;
        self.recurrent_per_bank
            .checked_mul(u64::try_from(banks).map_err(|_| "assessment bank count exceeds u64")?)
            .ok_or_else(|| "assessment recurrent bytes overflow".to_owned())
    }

    /// A definite no-fit result is possible when even these exact resident
    /// terms exceed stable capacity. A value below capacity is not a fit
    /// verdict: prepared scratch and startup peaks remain unaccounted for.
    pub fn minimum_required_bytes(self) -> Result<u64, String> {
        [
            self.target_weights,
            self.head_weights,
            self.vision_weights,
            self.history_at_fit_depth()?,
            self.recurrent_at_fit_workload()?,
        ]
        .into_iter()
        .try_fold(0u64, |total, bytes| {
            total
                .checked_add(bytes)
                .ok_or_else(|| "assessment minimum resident bytes overflow".to_owned())
        })
    }

    /// Compare against the capacity returned by `platform::assessment_capacity`.
    /// Missing prepared-resource or startup bounds cannot establish a
    /// positive fit.
    pub fn assess_fit(
        self,
        capacity_bytes: u64,
        bounds: Option<AssessmentMemoryBounds>,
    ) -> Result<AssessmentFit, String> {
        let exact_resident_bytes = self.minimum_required_bytes()?;
        let conservative_required_bytes = bounds
            .map(|bounds| {
                exact_resident_bytes
                    .checked_add(bounds.prepared_resource_bytes)
                    .and_then(|bytes| bytes.checked_add(bounds.startup_additional_bytes))
                    .ok_or_else(|| "assessment conservative memory charge overflow".to_owned())
            })
            .transpose()?;
        let verdict = if exact_resident_bytes > capacity_bytes {
            AssessmentFitVerdict::DoesNotFit
        } else if conservative_required_bytes.is_some_and(|bytes| bytes <= capacity_bytes) {
            AssessmentFitVerdict::FitsWithinBounds
        } else {
            AssessmentFitVerdict::Unconfirmed
        };
        Ok(AssessmentFit {
            verdict,
            exact_resident_bytes,
            exact_required_lower_bound_bytes: exact_resident_bytes,
            conservative_required_bytes,
            capacity_bytes,
        })
    }

    /// The header also proves the exact native invocation workspace charge.
    /// It strengthens a no-fit conclusion, while the unknown graph pools
    /// still prevent a positive fit conclusion.
    pub fn assess_with_header_bounds(
        self,
        capacity_bytes: u64,
        header: AssessmentHeaderBounds,
    ) -> Result<AssessmentFit, String> {
        let exact_resident_bytes = self.minimum_required_bytes()?;
        let lower_bound = exact_resident_bytes
            .checked_add(header.prepared_program_bytes)
            .ok_or("assessment exact lower bound overflow")?;
        Ok(AssessmentFit {
            verdict: if lower_bound > capacity_bytes {
                AssessmentFitVerdict::DoesNotFit
            } else {
                AssessmentFitVerdict::Unconfirmed
            },
            exact_resident_bytes,
            exact_required_lower_bound_bytes: lower_bound,
            conservative_required_bytes: None,
            capacity_bytes,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use seismic::{BackendName, Layout, NativeGraphMetadata};

    #[test]
    fn checked_native_declarations_are_inspectable_without_a_device() {
        let implementation = seismic::generated::native_implementation_for_backend::<
            magnitude_model_kernels::dense_output::Entry,
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
            &[("M", 4), ("O", 2), ("H", 16), ("F", 32)],
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
            &[("M", 4), ("O", 2), ("H", 16), ("F", 32)],
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
            &definition.geometry,
            2,
            true,
        )
        .unwrap();
        let selected = crate::programs::native_target_graph::checked_entry_graph_storage(
            BackendName::Cpu,
            &load,
            &definition.geometry,
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
            ("V", definition.geometry.vocabulary),
            ("D", definition.geometry.hidden),
        ];
        let mut invalid = NativeGraphMetadata::new(BackendName::Cpu);
        let wrong_table = invalid.port(Element::f32(), &[1, 1]).unwrap();
        let tokens = invalid
            .input_for::<magnitude_model_kernels::embedding_rows::Entry>(
                &elements,
                "tokens",
                &dimensions,
            )
            .unwrap();
        let mismatch = invalid.enqueue::<magnitude_model_kernels::embedding_rows::Entry>(
            &elements,
            &dimensions,
            magnitude_model_kernels::embedding_rows::WorkflowArgs {
                table: wrong_table.tensor().into(),
                tokens: tokens.tensor().into(),
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
            max_retained_entries: 1,
            active_requests: 1,
            in_flight_requests: 2,
            branch_checkpoints: 0,
            max_batch_rows: 2,
            max_projected_rows: 2,
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
            &definition.geometry,
            &state,
            plan.target(),
            limits,
        )
        .unwrap();
        let (block, _) = crate::programs::native_target_graph::checked_block_graph_resources(
            BackendName::Cpu,
            &load,
            &definition.geometry,
            &state,
            plan.target().blocks()[0],
            0,
            2,
            4,
            2,
            state.target_state().history_rows as u64,
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
            1 + definition.geometry.blocks.len()
        );
        assert!(pools.total_bytes >= pools.target.committed_bytes);
    }

    #[test]
    fn checked_head_family_includes_chained_and_shaped_classes() {
        use magnitude_artifacts::gguf::{Encoding, TensorDescriptor};
        use magnitude_model_contracts::{
            AttentionWeights, DenseFeedForwardWeights, HeadBlock, HeadWeights, WeightDescriptor,
        };

        let mut definition = crate::planning::tests::fixture_definition();
        let descriptor = |name: &str, shape: &[u64]| WeightDescriptor {
            name: format!("head_{name}"),
            shape: shape.to_vec(),
        };
        let block = HeadBlock {
            embedding_norm: descriptor("embedding_norm", &[128]),
            hidden_norm: descriptor("hidden_norm", &[128]),
            combine: descriptor("combine", &[128, 256]),
            input_norm: descriptor("input_norm", &[128]),
            attention: AttentionWeights {
                query_gate: descriptor("query_gate", &[128, 128]),
                key: descriptor("key", &[64, 128]),
                value: descriptor("value", &[64, 128]),
                query_norm: descriptor("query_norm", &[64]),
                key_norm: descriptor("key_norm", &[64]),
                output: descriptor("attention_output", &[128, 64]),
            },
            feedforward_norm: descriptor("feedforward_norm", &[128]),
            feedforward_geometry: magnitude_model_contracts::FeedForwardGeometry::Dense {
                intermediate: 128,
            },
            feedforward: magnitude_model_contracts::FeedForwardWeights::Dense(Box::new(
                DenseFeedForwardWeights {
                    gate: descriptor("gate", &[128, 128]),
                    up: descriptor("up", &[128, 128]),
                    down: descriptor("down", &[128, 128]),
                },
            )),
            output_norm: descriptor("output_norm", &[128]),
        };
        let mut manifest = crate::planning::tests::fixture_manifest(&definition);
        let magnitude_model_contracts::FeedForwardWeights::Dense(dense) = &block.feedforward else {
            unreachable!()
        };
        let descriptors = [
            &block.embedding_norm,
            &block.hidden_norm,
            &block.combine,
            &block.input_norm,
            &block.attention.query_gate,
            &block.attention.key,
            &block.attention.value,
            &block.attention.query_norm,
            &block.attention.key_norm,
            &block.attention.output,
            &block.feedforward_norm,
            &dense.gate,
            &dense.up,
            &dense.down,
            &block.output_norm,
        ];
        let mut offset = manifest.target.size;
        for descriptor in descriptors {
            let nbytes = descriptor.shape.iter().product::<u64>() * 2;
            manifest.target.tensors.push(TensorDescriptor {
                name: descriptor.name.clone(),
                shape: descriptor.shape.clone(),
                encoding: Encoding::F16,
                offset,
                nbytes,
            });
            offset += nbytes;
        }
        manifest.target.size = offset;
        definition.head = Some(HeadWeights {
            blocks: vec![block],
        });
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
            max_retained_entries: 1,
            active_requests: 1,
            in_flight_requests: 2,
            branch_checkpoints: 0,
            max_batch_rows: 16,
            max_projected_rows: 2,
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
        let attention = match &definition.geometry.blocks[0].mixer {
            magnitude_model_contracts::MixerGeometry::Attention(shape) => shape,
            _ => unreachable!(),
        };
        let head_state = state.head_state().unwrap();
        let classes = crate::programs::native_head::head_graph_classes(
            limits,
            head_state.history_rows as u64,
            method.draft_rows(),
        )
        .unwrap();
        assert!(classes.iter().any(|class| class.steps == 1 && class.shaped));
        assert!(classes
            .iter()
            .any(|class| class.entry_rows > crate::programs::graph::routed::DECODE_ROWS));
        let family = crate::programs::native_head::checked_head_family_storage(
            BackendName::Cpu,
            &load,
            &definition.geometry,
            attention,
            plan.head().unwrap().blocks()[0],
            classes,
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
            &definition.geometry,
            &state,
            plan.target(),
            limits,
        )
        .unwrap();
        assert_eq!(
            pools.binding_constant_bytes,
            target.binding_constant_bytes + family.binding_constant_bytes
        );

        let descriptor = |name: &str, shape: &[u64]| WeightDescriptor {
            name: format!("routed_head_{name}"),
            shape: shape.to_vec(),
        };
        let routed = magnitude_model_contracts::RoutedFeedForwardWeights {
            router: descriptor("router", &[4, 128]),
            shared_router: descriptor("shared_router", &[128]),
            expert_gate: descriptor("expert_gate", &[4, 64, 128]),
            expert_up: descriptor("expert_up", &[4, 64, 128]),
            expert_down: descriptor("expert_down", &[4, 128, 64]),
            shared_gate: descriptor("shared_gate", &[64, 128]),
            shared_up: descriptor("shared_up", &[64, 128]),
            shared_down: descriptor("shared_down", &[128, 64]),
        };
        for weight in [
            &routed.router,
            &routed.shared_router,
            &routed.expert_gate,
            &routed.expert_up,
            &routed.expert_down,
            &routed.shared_gate,
            &routed.shared_up,
            &routed.shared_down,
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
        manifest.target.size = offset;
        let head = &mut definition.head.as_mut().unwrap().blocks[0];
        head.feedforward_geometry = magnitude_model_contracts::FeedForwardGeometry::Routed(
            magnitude_model_contracts::ExpertGeometry {
                count: 4,
                selected: 2,
                intermediate: 64,
                shared_intermediate: 64,
                normalize_selected: true,
            },
        );
        head.feedforward = magnitude_model_contracts::FeedForwardWeights::Routed(Box::new(routed));
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
            &definition.geometry,
            attention,
            plan.head().unwrap().blocks()[0],
            crate::programs::native_head::head_graph_classes(
                limits,
                head_state.history_rows as u64,
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
        use magnitude_model_contracts::{WeightKind, WeightRole, WeightScope};

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
            &definition.geometry,
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
        use magnitude_model_contracts::{WeightKind, WeightRole, WeightScope};

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
            &definition.geometry,
            weight(WeightKind::OutputNorm),
            2,
            1,
        )
        .unwrap();
        let projected = crate::programs::graph::readout::checked_projected_graph_storage(
            BackendName::Cpu,
            &definition.geometry,
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
        use magnitude_model_contracts::{WeightKind, WeightRole, WeightScope};

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
                &definition.geometry,
                weight(WeightKind::OutputNorm),
                weight(WeightKind::Output),
                2,
                1,
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
            max_retained_entries: 2,
            active_requests: 2,
            in_flight_requests: 2,
            branch_checkpoints: 0,
            max_batch_rows: 2,
            max_projected_rows: 2,
            max_images_per_request: 0,
            lookahead: false,
        };
        let family = crate::programs::graph::readout::checked_readout_family_storage(
            BackendName::Cpu,
            &load,
            &definition.geometry,
            limits,
        )
        .unwrap();
        assert!(family.workspace > 0);
        assert!(family.output > 0);
        assert!(family.upload > 0);
    }

    #[test]
    fn checked_vision_family_bounds_exact_patch_classes() {
        use crate::{
            ArtifactComponent, ArtifactComponentKind, VisionBlockBinding, VisionMergerBinding,
            VisionPatchBinding, VisionProgramPlan, WeightPlan,
        };
        use magnitude_artifacts::ArtifactIdentity;
        use magnitude_model_contracts::{
            ActivationDType, VisionActivation, VisionGeometry, WeightDescriptor, WeightKind,
            WeightRole, WeightScope,
        };

        let f32 = Element::f32();
        let bf16 = Element::bf16();
        let component = ArtifactComponent {
            kind: ArtifactComponentKind::Projector,
            identity: ArtifactIdentity([9; 32]),
        };
        let mut weights = Vec::new();
        let mut add = |scope, kind, shape: &[u64]| {
            let name = format!("{scope:?}:{kind:?}");
            let bytes = shape.iter().product::<u64>() * 4;
            weights.push(WeightPlan {
                role: WeightRole { scope, kind },
                component,
                source: f32,
                resident: f32,
                shape: shape.to_vec(),
                descriptor: WeightDescriptor {
                    name,
                    shape: shape.to_vec(),
                },
                source_bytes: bytes,
                resident_bytes: bytes,
            });
        };
        let vision = WeightScope::Vision;
        add(
            WeightScope::VisionPatch(0),
            WeightKind::PatchEmbedding,
            &[128, 3, 2, 2],
        );
        add(
            WeightScope::VisionPatch(1),
            WeightKind::PatchEmbedding,
            &[128, 3, 2, 2],
        );
        add(vision, WeightKind::PatchBias, &[128]);
        add(vision, WeightKind::PositionEmbedding, &[16, 128]);
        let block = WeightScope::VisionBlock(0);
        for kind in [
            WeightKind::InputNormWeight,
            WeightKind::InputNormBias,
            WeightKind::AttentionOutputBias,
            WeightKind::FeedForwardNormWeight,
            WeightKind::FeedForwardNormBias,
            WeightKind::FeedForwardUpBias,
            WeightKind::FeedForwardDownBias,
        ] {
            add(block, kind, &[128]);
        }
        add(block, WeightKind::FusedQkvWeight, &[384, 128]);
        add(block, WeightKind::FusedQkvBias, &[384]);
        add(block, WeightKind::AttentionOutput, &[128, 128]);
        add(block, WeightKind::DenseUp, &[128, 128]);
        add(block, WeightKind::DenseDown, &[128, 128]);
        add(vision, WeightKind::NormWeight, &[128]);
        add(vision, WeightKind::NormBias, &[128]);
        add(vision, WeightKind::MergerHidden, &[512, 512]);
        add(vision, WeightKind::MergerHiddenBias, &[512]);
        add(vision, WeightKind::MergerOutput, &[128, 512]);
        add(vision, WeightKind::MergerOutputBias, &[128]);
        let load = ModelLoadPlan {
            target: Vec::new(),
            head: None,
            vision: Some(weights),
        };
        let geometry = VisionGeometry {
            activation_dtype: ActivationDType::BF16,
            depth: 1,
            hidden: 128,
            intermediate: 128,
            heads: 1,
            patch: 2,
            merge: 2,
            image_size: 8,
            output_hidden: 128,
            epsilon: 1e-6,
            table_side: 4,
            temporal_patch: 2,
            channels: 3,
            block_activation: VisionActivation::GeluTanh,
            merger_activation: VisionActivation::GeluErf,
        };
        let plan = VisionProgramPlan::new(
            VisionPatchBinding {
                temporal_weight_0: f32,
                temporal_weight_1: f32,
                bias: f32,
                position: f32,
            },
            vec![VisionBlockBinding {
                input_norm_weight: f32,
                input_norm_bias: f32,
                qkv_weight: f32,
                qkv_bias: f32,
                attention_output: f32,
                attention_output_bias: f32,
                feedforward_norm_weight: f32,
                feedforward_norm_bias: f32,
                up: f32,
                up_bias: f32,
                down: f32,
                down_bias: f32,
                activation: bf16,
            }],
            VisionMergerBinding {
                output_norm_weight: f32,
                output_norm_bias: f32,
                hidden: f32,
                hidden_bias: f32,
                output: f32,
                output_bias: f32,
                activation: bf16,
            },
        );
        let single = crate::programs::native_vision::checked_vision_family_storage(
            BackendName::Cpu,
            &load,
            &geometry,
            128,
            &plan,
            [4],
        )
        .unwrap();
        let double = crate::programs::native_vision::checked_vision_family_storage(
            BackendName::Cpu,
            &load,
            &geometry,
            128,
            &plan,
            [8],
        )
        .unwrap();
        let family = crate::programs::native_vision::checked_vision_family_storage(
            BackendName::Cpu,
            &load,
            &geometry,
            128,
            &plan,
            [4, 8],
        )
        .unwrap();
        assert_eq!(family.workspace, single.workspace.max(double.workspace));
        assert_eq!(family.output, single.output.max(double.output));
        assert_eq!(family.upload, single.upload.max(double.upload));
        assert!(family.workspace > 0 && family.output > 0 && family.upload > 0);
    }

    #[test]
    fn header_terms_include_exact_selected_weights_and_history() {
        let definition = crate::planning::tests::fixture_definition();
        let manifest = crate::planning::tests::fixture_manifest(&definition);
        let selection = ComponentSelection {
            head: false,
            vision: false,
        };
        let load =
            ModelLoadPlan::derive(&manifest, &definition, selection, Layout::Rows16).unwrap();
        assert_eq!(
            AssessmentBindingEvidence::inspect_dense_paths(
                &definition,
                &load,
                KvCodec::Dense,
                BackendName::Cpu,
            )
            .unwrap(),
            AssessmentBindingEvidence::Unconfirmed,
        );
        let terms = AssessmentMemoryTerms::derive(
            &definition,
            &load,
            selection,
            KvCodec::Dense,
            PlannedMethod::Plain,
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
        assert_eq!(terms.fit_depth, 128);
        assert_eq!(terms.history_at_fit_depth().unwrap(), 128 * 2 * 64 * 2);
        assert!(AssessmentMemoryTerms::derive(
            &definition,
            &load,
            selection,
            KvCodec::Dense,
            PlannedMethod::Mtp {
                greedy_proposals: 1,
                sampled_proposals: 1,
            },
        )
        .is_err());

        let header = AssessmentHeaderBounds::derive(&definition, &load, KvCodec::Dense).unwrap();
        assert_eq!(
            header.prepared_program_bytes,
            AttestedPrograms::planned_invocation_workspace_bytes(
                &load.program_plan(&definition, KvCodec::Dense).unwrap()
            )
            .unwrap()
        );
        assert_eq!(
            header.startup_additional_bytes,
            AttestedPrograms::qualification_peak_bytes(&load)
                .max(load.target_upload_peak_bytes().unwrap())
        );
        let complete = header.with_graph_resource_bound(42).unwrap();
        assert_eq!(
            complete.prepared_resource_bytes,
            header.prepared_program_bytes + 42
        );
        assert_eq!(
            complete.startup_additional_bytes,
            header.startup_additional_bytes
        );
        assert!(AssessmentHeaderBounds {
            prepared_program_bytes: u64::MAX,
            startup_additional_bytes: 0,
        }
        .with_graph_resource_bound(1)
        .is_err());
    }

    #[test]
    fn streaming_measurements_reject_unresolved_or_nonphysical_costs() {
        let cost = StreamingCost::from_samples(2_000_000, 0.001, 30_000_000, 0.0038).unwrap();
        assert!((cost.launch_seconds - 0.0008).abs() < 1e-12);
        assert!((cost.predict(10_000_000).unwrap() - 0.0018).abs() < 1e-12);
        assert!(StreamingCost::from_samples(2_000_000, 0.001, 30_000_000, 0.001).is_err());
        assert!(StreamingCost::from_samples(2_000_000, 0.001, 30_000_000, 0.03).is_err());
        assert!(StreamingCost::from_samples(2_000_000, f64::NAN, 30_000_000, 0.0038).is_err());
    }

    #[test]
    fn fit_requires_complete_bounds_and_only_exact_bytes_prove_no_fit() {
        let terms = AssessmentMemoryTerms {
            target_weights: 100,
            head_weights: 20,
            vision_weights: 0,
            history_per_token: 2,
            recurrent_per_bank: 10,
            fit_depth: 10,
        };
        let exact = terms.minimum_required_bytes().unwrap();
        assert_eq!(exact, 170);
        assert_eq!(
            terms.assess_fit(169, None).unwrap().verdict,
            AssessmentFitVerdict::DoesNotFit
        );
        assert_eq!(
            terms.assess_fit(200, None).unwrap().verdict,
            AssessmentFitVerdict::Unconfirmed
        );
        let header = AssessmentHeaderBounds {
            prepared_program_bytes: 31,
            startup_additional_bytes: 100,
        };
        let rejected = terms.assess_with_header_bounds(200, header).unwrap();
        assert_eq!(rejected.exact_resident_bytes, 170);
        assert_eq!(rejected.exact_required_lower_bound_bytes, 201);
        assert_eq!(rejected.verdict, AssessmentFitVerdict::DoesNotFit);
        assert_eq!(
            terms
                .assess_with_header_bounds(201, header)
                .unwrap()
                .verdict,
            AssessmentFitVerdict::Unconfirmed
        );
        let bounds = AssessmentMemoryBounds {
            prepared_resource_bytes: 10,
            startup_additional_bytes: 20,
        };
        assert_eq!(
            terms.assess_fit(200, Some(bounds)).unwrap().verdict,
            AssessmentFitVerdict::FitsWithinBounds
        );
        assert_eq!(
            terms.assess_fit(199, Some(bounds)).unwrap().verdict,
            AssessmentFitVerdict::Unconfirmed
        );
        assert_eq!(
            terms
                .assess_fit(199, Some(bounds))
                .unwrap()
                .conservative_required_bytes,
            Some(200)
        );
    }
}
