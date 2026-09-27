//! Logical operation grouping for one native physical executor domain.
//!
//! The physical domain owns programs, resident state, in-flight advances and
//! reconciliation. This module only keeps compatible operations together and
//! identifies the lane the service must submit next.

use magnitude_executor::{
    Completion, DomainError, DomainRequirements, GroupKey, HeadFlight, NativeFamily, Operation,
    ProgramFamily, ReservedResources, TargetFlight, VisionFlight,
};
pub use magnitude_executor::{DomainCheckpoint, ExecutorDomain};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DomainLane {
    Target,
    Head,
    Encoder,
}

impl DomainLane {
    fn for_operation(operation: &Operation) -> Self {
        match operation {
            Operation::Forward { .. } => Self::Target,
            Operation::Head { .. } => Self::Head,
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
    pub fn into_operations(self) -> Vec<Operation> {
        self.operations
    }

    /// Keep a capacity-limited launch's last request for a later physical
    /// launch without changing its logical round or operation order.
    pub fn split_last(mut self) -> Result<(Self, Self), Self> {
        if self.operations.len() < 2 {
            return Err(self);
        }
        let last = self
            .operations
            .pop()
            .expect("group has at least two operations");
        let tail = Self {
            lane: self.lane,
            key: self.key.clone(),
            operations: vec![last],
        };
        Ok((self, tail))
    }
}

/// Preserve input order while coalescing adjacent compatible operations. A
/// request may occupy only one slot in a group; another operation starts a new
/// group so dependencies never leap across an intervening lane. The domain
/// validates every operation when a group is reserved.
pub fn group<F: ProgramFamily>(
    domain: &ExecutorDomain<F>,
    operations: Vec<Operation>,
) -> Vec<OperationGroup> {
    let mut groups: Vec<OperationGroup> = Vec::new();
    for operation in operations {
        let lane = DomainLane::for_operation(&operation);
        let key = domain.group_key(&operation);
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
    groups
}

pub enum DomainFlight<F: ProgramFamily = NativeFamily> {
    Target(TargetFlight<F::TargetSubmission>),
    Head(HeadFlight<F::HeadSubmission>),
    Vision(VisionFlight<F::VisionSubmission>),
}

pub fn submit_group<F: ProgramFamily>(
    domain: &mut ExecutorDomain<F>,
    group: &OperationGroup,
) -> Result<DomainFlight<F>, DomainError> {
    let operations = group.operations.as_slice();
    let resources = domain.reserve(operations)?.into_resources();
    let submitted = match (group.lane, resources) {
        (DomainLane::Target, ReservedResources::Target(reservation)) => domain
            .submit_target(operations, reservation)
            .map(DomainFlight::Target),
        (DomainLane::Head, ReservedResources::Head(graph_workspace, graph_output, advances)) => {
            domain
                .submit_head(operations, graph_workspace, graph_output, advances)
                .map(DomainFlight::Head)
        }
        (DomainLane::Encoder, ReservedResources::Vision(workspace, output)) => {
            let [operation @ Operation::Encode { .. }] = operations else {
                return Err(DomainError::Input(
                    "vision group must contain one encode operation".into(),
                ));
            };
            domain
                .submit_vision(operation, workspace, output)
                .map(DomainFlight::Vision)
        }
        _ => Err(DomainError::Invariant(
            magnitude_executor::InvariantError {
                context: "reserved domain submission",
                detail: "reserved resource lane differs from operation group".into(),
            },
        )),
    };
    submitted.map_err(|error| match error {
        DomainError::Capacity(capacity) => {
            DomainError::Invariant(magnitude_executor::InvariantError {
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
            Self::Vision(flight) => flight.completion(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use magnitude_executor::{
        CommittedClass, Demand, ExecutableKind, ProgramIdentity, RequestId, TokenId, WorkKind,
    };

    #[test]
    fn capacity_split_preserves_request_order_and_launch_identity() {
        let key = GroupKey {
            program_identity: ProgramIdentity::new("target").unwrap(),
            executable: ExecutableKind::Target,
            commitment: CommittedClass::AllCommitted,
        };
        let operation = |id| Operation::Forward {
            request: RequestId(id),
            kind: WorkKind::Decode,
            tokens: vec![TokenId(1)],
            position: 0,
            conditioning: None,
            demand: Demand::NONE,
            select: Vec::new(),
            committed: 1,
        };
        let group = OperationGroup {
            lane: DomainLane::Target,
            key: key.clone(),
            operations: vec![operation(1), operation(2), operation(3)],
        };
        let (leading, trailing) = group.split_last().ok().unwrap();
        assert_eq!(leading.key(), &key);
        assert_eq!(trailing.key(), &key);
        assert_eq!(
            leading
                .operations()
                .iter()
                .map(Operation::request)
                .collect::<Vec<_>>(),
            vec![RequestId(1), RequestId(2)]
        );
        assert_eq!(
            trailing
                .operations()
                .iter()
                .map(Operation::request)
                .collect::<Vec<_>>(),
            vec![RequestId(3)]
        );
        assert!(trailing.split_last().is_err());
    }
}
