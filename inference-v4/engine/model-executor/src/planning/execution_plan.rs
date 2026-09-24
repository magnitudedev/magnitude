//! One allocation-free, checked numerical startup decision.

use super::{
    ArtifactComponent, ArtifactComponentKind, CapabilityPlan, ComponentPlan, ComponentSelection,
    ModelLoadPlan, PlannedMethod, ProgramPlan, ResourceBudget, ResourceLimits, ResourcePlan,
    WeightPlan, MAX_DRAFT_PROPOSALS,
};
use crate::{
    error::{CapacityError, PlanError, ResourceKind},
    platform::SelectedDevice,
    ExecutionPath,
};
use magnitude_artifacts::PackageManifest;
use magnitude_model_contracts::ModelDefinition;
use magnitude_model_state::KvCodec;
use seismic::{BackendName, DeviceSelector};

/// The planned device by Seismic identity. The selector, not a name or
/// ordinal, is what the executing process resolves and opens.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlannedDevice {
    selector: DeviceSelector,
    backend: BackendName,
    name: String,
    assessment_capacity_bytes: u64,
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

    /// Stable capacity minus the planning reserve at planning time.
    pub fn assessment_capacity_bytes(&self) -> u64 {
        self.assessment_capacity_bytes
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResolvedPolicy {
    path: ExecutionPath,
    method: PlannedMethod,
    codec: KvCodec,
    selection: ComponentSelection,
    limits: ResourceLimits,
    budget: ResourceBudget,
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

    pub fn budget(&self) -> ResourceBudget {
        self.budget
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

    pub fn policy(&self) -> ResolvedPolicy {
        self.policy
    }

    pub fn load(&self) -> &ModelLoadPlan {
        &self.load
    }

    pub fn admit(self, resources: ResourcePlan) -> Result<ExecutionPlan, PlanError> {
        let planned_weight_bytes = resources
            .bytes()
            .target_weights
            .checked_add(resources.bytes().head_weights)
            .and_then(|bytes| bytes.checked_add(resources.bytes().vision_weights))
            .ok_or(PlanError::Arithmetic("resident weight byte count overflow"))?;
        if planned_weight_bytes == 0
            || resources
                .bytes()
                .total()
                .map_err(PlanError::ResourcePlanning)?
                > self.policy.budget.storage_bytes
        {
            return Err(PlanError::Topology(
                "resource charges disagree with storage policy",
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

    pub fn policy(&self) -> ResolvedPolicy {
        self.policy
    }
}

pub struct ExecutionPlanner;

impl ExecutionPlanner {
    pub fn prepare(
        device: &SelectedDevice,
        manifest: &PackageManifest,
        definition: &ModelDefinition,
        selection: ComponentSelection,
        path: ExecutionPath,
        method: PlannedMethod,
        codec: KvCodec,
        limits: ResourceLimits,
        budget: ResourceBudget,
    ) -> Result<ExecutionPlanDraft, PlanError> {
        if path == ExecutionPath::Native && codec == KvCodec::RotatedK4V4 {
            return Err(PlanError::Unsupported("native rotated K4/V4 KV codec"));
        }
        if budget.storage_bytes > device.assessment_capacity_bytes {
            return Err(PlanError::Resource(CapacityError {
                resource: ResourceKind::DeviceMemory,
                required: budget.storage_bytes,
                available: device.assessment_capacity_bytes,
            }));
        }
        if selection.head != matches!(method, PlannedMethod::Mtp { .. }) {
            return Err(PlanError::Topology("method and head selection disagree"));
        }
        match method {
            PlannedMethod::Mtp { .. } if limits.branch_checkpoints < limits.active_requests => {
                return Err(PlanError::Topology(
                    "MTP branch checkpoint bound is below active requests",
                ));
            }
            _ => {}
        }
        let load = ModelLoadPlan::derive(
            manifest,
            definition,
            selection,
            super::resident_layout(path, device.info.backend),
        )
        .map_err(PlanError::InvalidDefinition)?;
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
        };
        if programs.head().is_some() != selection.head
            || programs.vision().is_some() != selection.vision
        {
            return Err(PlanError::Topology(
                "enabled components and program slots disagree",
            ));
        }
        Ok(ExecutionPlanDraft {
            device: PlannedDevice {
                selector: device.info.selector,
                backend: device.info.backend,
                name: device.info.name.clone(),
                assessment_capacity_bytes: device.assessment_capacity_bytes,
            },
            components: ComponentPlan {
                target,
                head: selection.head.then_some(target),
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
                budget,
            },
        })
    }
}
