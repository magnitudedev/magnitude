//! Logical operation grouping for one native physical executor domain.
//!
//! The physical domain owns programs, resident state, in-flight advances and
//! reconciliation. This module only keeps compatible operations together and
//! identifies the lane the service must submit next.

use magnitude_model_executor::{
    Completion, DomainError, DomainRequirements, GroupKey, HeadFlight, NativeFamily, Operation,
    ProgramFamily, ProjectFlight, ReservedResources, StateFlight, TargetFlight, VisionFlight,
};
pub use magnitude_model_executor::{DomainCheckpoint, ExecutorDomain};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DomainLane {
    Target,
    Head,
    Project,
    Repair,
    Encoder,
}

impl DomainLane {
    fn for_operation(operation: &Operation) -> Self {
        match operation {
            Operation::Forward { .. } => Self::Target,
            Operation::Head { .. } => Self::Head,
            Operation::Project { .. } => Self::Project,
            Operation::Repair { .. } => Self::Repair,
            Operation::Encode { .. } => Self::Encoder,
        }
    }
}

pub struct OperationGroup {
    lane: DomainLane,
    key: GroupKey,
    operations: Vec<Operation>,
}

impl OperationGroup {
    pub fn lane(&self) -> DomainLane {
        self.lane
    }
    pub fn key(&self) -> &GroupKey {
        &self.key
    }
    pub fn operations(&self) -> &[Operation] {
        &self.operations
    }
}

/// Preserve input order while coalescing adjacent compatible operations. A
/// request may occupy only one slot in a group; another operation starts a new
/// group so dependencies never leap across an intervening lane.
pub fn group<F: ProgramFamily>(
    domain: &ExecutorDomain<F>,
    operations: Vec<Operation>,
) -> Result<Vec<OperationGroup>, String> {
    let mut groups: Vec<OperationGroup> = Vec::new();
    for operation in operations {
        operation.validate().map_err(|error| error.to_string())?;
        let lane = DomainLane::for_operation(&operation);
        let key = domain.group_key(&operation)?;
        let request = operation.request();
        let existing = groups.last_mut().filter(|group| {
            group.lane == lane
                && group.key == key
                && group.lane != DomainLane::Encoder
                && group
                    .operations
                    .iter()
                    .all(|item| item.request() != request)
        });
        if let Some(existing) = existing {
            existing.operations.push(operation);
        } else {
            groups.push(OperationGroup {
                lane,
                key,
                operations: vec![operation],
            });
        }
    }
    Ok(groups)
}

pub enum DomainFlight<F: ProgramFamily = NativeFamily> {
    Target(TargetFlight<F::TargetSubmission>),
    Head(HeadFlight<F::HeadSubmission>),
    Project(ProjectFlight<F::ProjectSubmission>),
    Vision(VisionFlight<F::VisionSubmission>),
    Repair(StateFlight<F::StateSubmission>),
}

pub fn submit_group<F: ProgramFamily>(
    domain: &mut ExecutorDomain<F>,
    group: &OperationGroup,
) -> Result<DomainFlight<F>, DomainError> {
    let reservation = domain.reserve(group.operations.clone())?;
    let (operations, resources) = reservation.into_parts();
    let submitted = match (group.lane, resources) {
        (DomainLane::Target, ReservedResources::Target(reservation)) => domain
            .submit_target(operations, reservation)
            .map(DomainFlight::Target),
        (DomainLane::Head, ReservedResources::Head(graph_workspace, graph_output, advances)) => {
            domain
                .submit_head(operations, graph_workspace, graph_output, advances)
                .map(DomainFlight::Head)
        }
        (DomainLane::Project, ReservedResources::Head(graph_workspace, graph_output, advances)) => {
            if !advances.is_empty() {
                return Err(DomainError::Invariant(
                    magnitude_model_executor::InvariantError {
                        context: "reserved domain submission",
                        detail: "projection reservation unexpectedly owns state advances".into(),
                    },
                ));
            }
            domain
                .submit_project(operations, graph_workspace, graph_output)
                .map(DomainFlight::Project)
        }
        (DomainLane::Repair, ReservedResources::Repair(reservation)) => {
            let [Operation::Repair { request, .. }] = operations.as_slice() else {
                return Err(DomainError::Input(
                    "repair group must contain one operation".into(),
                ));
            };
            domain
                .submit_repair(*request, reservation)
                .map(DomainFlight::Repair)
        }
        (DomainLane::Encoder, ReservedResources::Vision(workspace, output)) => {
            let [operation @ Operation::Encode { .. }] = operations.as_slice() else {
                return Err(DomainError::Input(
                    "vision group must contain one encode operation".into(),
                ));
            };
            domain
                .submit_vision(operation.clone(), workspace, output)
                .map(DomainFlight::Vision)
        }
        _ => Err(DomainError::Invariant(
            magnitude_model_executor::InvariantError {
                context: "reserved domain submission",
                detail: "reserved resource lane differs from operation group".into(),
            },
        )),
    };
    submitted.map_err(|error| match error {
        DomainError::Capacity(capacity) => {
            DomainError::Invariant(magnitude_model_executor::InvariantError {
                context: "reserved domain submission",
                detail: format!("reserved capacity became unavailable: {capacity}"),
            })
        }
        other => other,
    })
}

pub fn requirements<F: ProgramFamily>(
    domain: &ExecutorDomain<F>,
    group: &OperationGroup,
) -> Result<DomainRequirements, DomainError> {
    domain.requirements(&group.operations)
}

impl<F: ProgramFamily> DomainFlight<F> {
    pub fn completion(&mut self) -> &mut dyn Completion {
        match self {
            Self::Target(flight) => flight.completion(),
            Self::Head(flight) => flight.completion(),
            Self::Project(flight) => flight.completion(),
            Self::Vision(flight) => flight.completion(),
            Self::Repair(flight) => flight.completion(),
        }
    }
}
