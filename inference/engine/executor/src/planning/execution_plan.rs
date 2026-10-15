//! One allocation-free, checked numerical startup decision.

use super::{
    ArtifactComponent, ArtifactComponentKind, CapabilityPlan, ComponentPlan, ComponentSelection,
    MAX_DRAFT_PROPOSALS, ModelLoadPlan, PlannedMethod, ProgramPlan, ReadoutClassChoice,
    ResourceLimits, ResourcePlan, WeightPlan,
};
use crate::{ExecutionPath, error::PlanError, platform::SelectedDevice};
use magnitude_artifacts::PackageManifest;
use magnitude_family_contracts::ModelDefinition;
use magnitude_state::KvCodec;
use seismic::{BackendName, DeviceSelector};

/// The planned device by Seismic identity. The selector, not a name or
/// ordinal, is what the executing process resolves and opens.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlannedDevice {
    selector: DeviceSelector,
    backend: BackendName,
    name: String,
    assessment_capacity_bytes: u64,
    tensor_operations: bool,
}

impl PlannedDevice {
    pub fn selector(&self) -> DeviceSelector {
        self.selector
    }

    pub fn backend(&self) -> BackendName {
        self.backend
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    /// Stable capacity of the selected allocation domain at planning time.
    pub fn assessment_capacity_bytes(&self) -> u64 {
        self.assessment_capacity_bytes
    }

    /// Whether the device forms Metal tensor operations
    /// (`SelectedDevice::tensor_operations`).
    pub fn tensor_operations(&self) -> bool {
        self.tensor_operations
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResolvedPolicy {
    path: ExecutionPath,
    method: PlannedMethod,
    codec: KvCodec,
    selection: ComponentSelection,
    limits: ResourceLimits,
}

impl ResolvedPolicy {
    pub fn path(&self) -> ExecutionPath {
        self.path
    }

    pub fn method(&self) -> PlannedMethod {
        self.method
    }

    pub fn codec(&self) -> KvCodec {
        self.codec
    }

    pub fn selection(&self) -> ComponentSelection {
        self.selection
    }

    pub fn limits(&self) -> ResourceLimits {
        self.limits
    }
}

/// The only startup value a program factory and allocator should consume.
/// Every field is immutable after the checked constructor returns.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExecutionPlan {
    device: PlannedDevice,
    components: ComponentPlan,
    load: ModelLoadPlan,
    programs: ProgramPlan,
    resources: ResourcePlan,
    capabilities: CapabilityPlan,
    policy: ResolvedPolicy,
}

/// Model topology on a backend: everything planning derives without a
/// device. A load's [`ExecutionPlanDraft`] is this plan on its selected
/// device.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BackendPlan {
    backend: BackendName,
    components: ComponentPlan,
    load: ModelLoadPlan,
    programs: ProgramPlan,
    capabilities: CapabilityPlan,
    policy: ResolvedPolicy,
}

impl BackendPlan {
    pub fn backend(&self) -> BackendName {
        self.backend
    }

    pub fn programs(&self) -> &ProgramPlan {
        &self.programs
    }

    pub fn policy(&self) -> &ResolvedPolicy {
        &self.policy
    }

    pub fn load(&self) -> &ModelLoadPlan {
        &self.load
    }
}

/// Model topology and device choice before physical storage admission. Native
/// programs and their Seismic-owned graph storage can be prepared from this
/// value; only `admit` constructs an executable plan with resource ownership.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExecutionPlanDraft {
    device: PlannedDevice,
    components: ComponentPlan,
    load: ModelLoadPlan,
    programs: ProgramPlan,
    capabilities: CapabilityPlan,
    policy: ResolvedPolicy,
}

impl ExecutionPlanDraft {
    pub fn device(&self) -> &PlannedDevice {
        &self.device
    }

    pub fn programs(&self) -> &ProgramPlan {
        &self.programs
    }

    pub fn policy(&self) -> &ResolvedPolicy {
        &self.policy
    }

    pub fn load(&self) -> &ModelLoadPlan {
        &self.load
    }

    /// Retain only empirically selected optional readout views before graph
    /// resources or persistent residency are admitted. The canonical packed
    /// projection remains available to every class.
    pub fn finalize_readout(
        mut self,
        definition: &ModelDefinition,
        choices: &[ReadoutClassChoice],
    ) -> Result<Self, PlanError> {
        let canonical = self.clone();
        self.load = self
            .load
            .finalize_readout(choices)
            .map_err(PlanError::InvalidDefinition)?;
        self.programs = self.load.program_plan(definition, self.policy.codec)?;
        if !choices.is_empty() {
            let context_limit = u32::try_from(definition.decoder.context_limit)
                .map_err(|_| PlanError::Arithmetic("model context exceeds assessment range"))?;
            let assessment =
                crate::assessment::prepare_execution_assessment(definition, &self, context_limit)
                    .map_err(|error| PlanError::ResourcePlanning(error.to_string()))?;
            if assessment.memory_charge().allocation_bytes > self.device.assessment_capacity_bytes {
                return canonical.finalize_readout(definition, &[]);
            }
        }
        Ok(self)
    }

    /// The weight cap (spec 6.1). The resident plan when the standard
    /// workload fits the device with every weight resident and `cap` (when
    /// given) covers every weight; otherwise the streamed plan with the
    /// largest bank that fits both, or a refusal naming the limit.
    ///
    /// A cap never raises what the device permits, and a cap below the
    /// model's weights streams even a model that fits: artificial capping and
    /// "does not fit" are one path.
    pub fn with_weight_cap(
        self,
        definition: &ModelDefinition,
        cap: Option<u64>,
    ) -> Result<Self, PlanError> {
        let charge = |draft: &Self| -> Result<(u64, u64), PlanError> {
            let context_limit = u32::try_from(definition.decoder.context_limit)
                .map_err(|_| PlanError::Arithmetic("model context exceeds assessment range"))?;
            let assessment =
                crate::assessment::prepare_execution_assessment(definition, draft, context_limit)
                    .map_err(|error| PlanError::ResourcePlanning(error.to_string()))?;
            let [target, head, vision] = super::weights::weight_bytes_by_component(&draft.load)
                .map_err(PlanError::ResourcePlanning)?;
            Ok((
                assessment.memory_charge().allocation_bytes,
                target + head + vision,
            ))
        };
        let capacity = self.device.assessment_capacity_bytes;
        let (allocation, weights) = charge(&self)?;
        if allocation <= capacity && cap.is_none_or(|cap| cap >= weights) {
            return Ok(self);
        }
        let selected = definition_selected(definition)?;
        let streamed = |slots: u64| -> Result<Self, PlanError> {
            let mut draft = self.clone();
            draft.load = draft
                .load
                .with_shared_expert_bank(slots, selected)
                .map_err(PlanError::InvalidDefinition)?;
            draft.programs = draft.load.program_plan(definition, draft.policy.codec)?;
            Ok(draft)
        };
        // Assess each supported capacity directly. Directory storage,
        // padded backing and graph workspaces belong to the candidate;
        // neither weight nor total charge is inferred from a payload slope.
        let probe = streamed(1)?;
        let experts = probe.load.expert_bank().expect("streamed bank").experts;
        let smallest = 1; // each shared group separately admits its launch union
        let mut slots = experts - 1;
        while slots >= smallest {
            let candidate = streamed(slots)?;
            // Weight bytes are exact metadata and monotone with capacity.
            // Reject over-cap candidates before constructing graph-resource
            // assessment. Every retained candidate still passes the original
            // complete allocation assessment; no measured peak substitutes it.
            let [target,head,vision]=super::weights::weight_bytes_by_component(&candidate.load)
                .map_err(PlanError::ResourcePlanning)?;
            let weights=target+head+vision;
            if cap.is_some_and(|cap|weights>cap) {
                slots-=1;
                continue;
            }
            let (allocation, assessed_weights) = charge(&candidate)?;
            if assessed_weights!=weights {
                return Err(PlanError::ResourcePlanning("weight metadata changed during bank assessment".into()));
            }
            if allocation <= capacity && cap.is_none_or(|cap| weights <= cap) {
                eprintln!("magnitude-engine: streamed load: {slots} of {experts} expert slots per routed layer (weights {weights}, allocation {allocation}, weight cap {cap:?}, device capacity {capacity})");
                return Ok(candidate);
            }
            slots-=1;
        }
        // Preserve the detailed minimum-bank refusal after the exhaustive
        // descending search finds no admissible capacity.
        let minimum = streamed(smallest)?;
        let (allocation, weights) = charge(&minimum)?;
        if allocation <= capacity && cap.is_none_or(|cap| weights <= cap) {
            return Ok(minimum);
        }
        Err(PlanError::ResourcePlanning(format!(
            "the smallest expert bank ({smallest} slots) needs {weights} weight bytes and {allocation} total allocation bytes; weight cap {cap:?}, device capacity {capacity}"
        )))
    }

    pub fn admit(self, resources: ResourcePlan) -> Result<ExecutionPlan, PlanError> {
        let planned_weight_bytes = resources
            .bytes()
            .target_weights
            .checked_add(resources.bytes().head_weights)
            .and_then(|bytes| bytes.checked_add(resources.bytes().vision_weights))
            .ok_or(PlanError::Arithmetic("resident weight byte count overflow"))?;
        if planned_weight_bytes == 0 {
            return Err(PlanError::Topology(
                "resource plan has no resident target weights",
            ));
        }
        Ok(ExecutionPlan {
            device: self.device,
            components: self.components,
            load: self.load,
            programs: self.programs,
            resources,
            capabilities: self.capabilities,
            policy: self.policy,
        })
    }
}

/// The experts a routed layer selects per row (`K`), the same for every
/// routed layer a bank streams.
fn definition_selected(definition: &ModelDefinition) -> Result<u64, PlanError> {
    let mut selected = None;
    for (_, sublayer) in definition.decoder.sublayers() {
        if let magnitude_family_contracts::Operator::RoutedFfn(routed) = &sublayer.op {
            match selected {
                None => selected = Some(routed.selected),
                Some(k) if k == routed.selected => {}
                Some(_) => {
                    return Err(PlanError::Unsupported(
                        "streamed routed layers selecting different expert counts",
                    ));
                }
            }
        }
    }
    selected.ok_or(PlanError::Unsupported("a streamed load without routed layers"))
}

impl ExecutionPlan {
    pub fn device(&self) -> &PlannedDevice {
        &self.device
    }

    pub fn components(&self) -> &ComponentPlan {
        &self.components
    }

    pub fn weights(&self) -> impl Iterator<Item = &WeightPlan> {
        self.load.weights()
    }

    pub fn load(&self) -> &ModelLoadPlan {
        &self.load
    }

    pub fn programs(&self) -> &ProgramPlan {
        &self.programs
    }

    pub fn resources(&self) -> &ResourcePlan {
        &self.resources
    }

    pub fn capabilities(&self) -> CapabilityPlan {
        self.capabilities
    }

    pub fn policy(&self) -> &ResolvedPolicy {
        &self.policy
    }
}

pub struct ExecutionPlanner;

impl ExecutionPlanner {
    #[allow(clippy::too_many_arguments)]
    pub fn prepare(
        device: &SelectedDevice,
        manifest: &PackageManifest,
        definition: &ModelDefinition,
        selection: ComponentSelection,
        path: ExecutionPath,
        method: PlannedMethod,
        codec: KvCodec,
        limits: ResourceLimits,
    ) -> Result<ExecutionPlanDraft, PlanError> {
        let BackendPlan {
            backend,
            components,
            load,
            programs,
            capabilities,
            policy,
        } = Self::backend_plan(
            device.info.backend,
            manifest,
            definition,
            selection,
            path,
            method,
            codec,
            limits,
        )?;
        Ok(ExecutionPlanDraft {
            device: PlannedDevice {
                selector: device.info.selector,
                backend,
                name: device.info.name.clone(),
                assessment_capacity_bytes: device.assessment_capacity_bytes,
                tensor_operations: device.tensor_operations,
            },
            components,
            load,
            programs,
            capabilities,
            policy,
        })
    }

    /// The model's plan on `backend`, which is all the planner reads of a
    /// device: [`Self::prepare`] adds the selected device to it.
    #[allow(clippy::too_many_arguments)]
    pub fn backend_plan(
        backend: BackendName,
        manifest: &PackageManifest,
        definition: &ModelDefinition,
        selection: ComponentSelection,
        path: ExecutionPath,
        method: PlannedMethod,
        codec: KvCodec,
        limits: ResourceLimits,
    ) -> Result<BackendPlan, PlanError> {
        if path == ExecutionPath::Native && codec == KvCodec::RotatedK4V4 {
            return Err(PlanError::Unsupported("native rotated K4/V4 KV codec"));
        }
        if selection.head != method.drafts() {
            return Err(PlanError::Topology("method and head selection disagree"));
        }
        match method {
            PlannedMethod::Mtp { .. }
                if definition.head.is_none() || definition.draft.is_some() =>
            {
                return Err(PlanError::Topology(
                    "MTP drafts with the definition's head alone",
                ));
            }
            PlannedMethod::DFlash { .. }
                if definition.draft.is_none() || definition.head.is_some() =>
            {
                return Err(PlanError::Topology(
                    "DFlash drafts with the definition's separate draft alone",
                ));
            }
            _ => {}
        }
        let layout = super::resident_layout(path, backend);
        let load = ModelLoadPlan::derive(manifest, definition, selection, layout)
            .map_err(PlanError::InvalidDefinition)?;
        // The head in progressive planes, where the backend reads them.
        let load = if path == ExecutionPath::Native
            && crate::programs::graph::readout::reads_progressive_heads(backend)
                .map_err(PlanError::ResourcePlanning)?
        {
            load.with_readout_candidates(definition)
                .map_err(PlanError::InvalidDefinition)?
        } else {
            load
        };
        let programs = load.program_plan(definition, codec)?;
        let target = ArtifactComponent {
            kind: ArtifactComponentKind::Target,
            identity: manifest.target.identity,
        };
        let vision = if selection.vision {
            manifest
                .projector
                .as_ref()
                .map(|projector| ArtifactComponent {
                    kind: ArtifactComponentKind::Projector,
                    identity: projector.identity,
                })
        } else {
            None
        };
        if selection.vision && vision.is_none() {
            return Err(PlanError::Topology(
                "enabled vision has no projector component",
            ));
        }
        // The head block chains on the device, so the draft width is bounded
        // by the sealed head graphs, not by the head's depth.
        let max_draft_proposals = match method {
            PlannedMethod::Plain => None,
            PlannedMethod::Mtp {
                greedy_proposals,
                sampled_proposals,
            } => {
                if greedy_proposals == 0
                    || sampled_proposals == 0
                    || greedy_proposals.max(sampled_proposals) > MAX_DRAFT_PROPOSALS
                {
                    return Err(PlanError::Unsupported("MTP proposal width"));
                }
                Some(greedy_proposals.max(sampled_proposals))
            }
            // One block pass drafts every proposal, so the width is bounded
            // by the draft's block, not by sealed chains.
            PlannedMethod::DFlash {
                proposals,
                generated,
                ..
            } => {
                let block = definition.draft.as_ref().map_or(
                    0,
                    magnitude_family_contracts::DraftDefinition::max_proposals,
                );
                if proposals == 0 || u64::from(proposals) > block || generated > proposals {
                    return Err(PlanError::Unsupported("DFlash proposal width"));
                }
                Some(proposals)
            }
        };
        if programs.drafts_with_head() != matches!(method, PlannedMethod::Mtp { .. })
            || programs.draft().is_some() != matches!(method, PlannedMethod::DFlash { .. })
            || programs.vision().is_some() != selection.vision
        {
            return Err(PlanError::Topology(
                "enabled components and program slots disagree",
            ));
        }
        // The drafter's component: the target's embedded head, or the
        // separate draft.
        let head = match method {
            PlannedMethod::Plain => None,
            PlannedMethod::Mtp { .. } => Some(target),
            PlannedMethod::DFlash { .. } => Some(ArtifactComponent {
                kind: ArtifactComponentKind::Draft,
                identity: manifest
                    .draft
                    .as_ref()
                    .ok_or(PlanError::Topology("DFlash has no draft component"))?
                    .identity,
            }),
        };
        Ok(BackendPlan {
            backend,
            components: ComponentPlan {
                target,
                head,
                vision,
            },
            load,
            programs,
            capabilities: CapabilityPlan {
                max_draft_proposals,
                vision: selection.vision,
            },
            policy: ResolvedPolicy {
                path,
                method,
                codec,
                selection,
                limits,
            },
        })
    }
}
