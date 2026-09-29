//! Closed numerical execution failures at the program boundary.

use magnitude_family_contracts::{DeferredForm, WeightRole};
use std::{error, fmt};

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DeviceError {
    Lost(String),
    Execution(String),
    Transfer(String),
}

impl fmt::Display for DeviceError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Lost(source) => write!(formatter, "device lost: {source}"),
            Self::Execution(source) => write!(formatter, "device execution failed: {source}"),
            Self::Transfer(source) => write!(formatter, "device transfer failed: {source}"),
        }
    }
}

impl error::Error for DeviceError {}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PlanError {
    InvalidDefinition(String),
    Unsupported(&'static str),
    /// A sublayer operator the executor does not implement at `position`.
    UnsupportedOperator {
        operator: &'static str,
        position: &'static str,
    },
    /// A form the family contract defines only as an interface point.
    Deferred(DeferredForm),
    /// A weight with a resident second-level scale, bound by an entry that
    /// has no accumulator-scale port for it.
    UnportedScale(WeightRole),
    Topology(&'static str),
    Arithmetic(&'static str),
    ResourcePlanning(String),
    Resource(CapacityError),
}

impl fmt::Display for PlanError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidDefinition(detail) => formatter.write_str(detail),
            Self::Unsupported(feature) => write!(formatter, "unsupported {feature}"),
            Self::UnsupportedOperator { operator, position } => {
                write!(formatter, "unsupported {operator} operator as {position}")
            }
            Self::Deferred(form) => write!(formatter, "unsupported {} (deferred)", form.name()),
            Self::UnportedScale(role) => write!(
                formatter,
                "unsupported second-level scale of {:?}/{:?}: its entry has no \
                 accumulator-scale port",
                role.scope, role.kind
            ),
            Self::Topology(detail) => formatter.write_str(detail),
            Self::Arithmetic(detail) => write!(formatter, "plan arithmetic failed: {detail}"),
            Self::ResourcePlanning(detail) => {
                write!(formatter, "resource planning failed: {detail}")
            }
            Self::Resource(source) => source.fmt(formatter),
        }
    }
}

impl error::Error for PlanError {}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CapacityError {
    pub resource: ResourceKind,
    pub required: u64,
    pub available: u64,
}

impl fmt::Display for CapacityError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "{:?} requires {} units; {} available",
            self.resource, self.required, self.available
        )
    }
}

impl error::Error for CapacityError {}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InvariantError {
    pub context: &'static str,
    pub detail: String,
}

impl fmt::Display for InvariantError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}: {}", self.context, self.detail)
    }
}

impl error::Error for InvariantError {}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SubmitError {
    Device(DeviceError),
    Invariant(InvariantError),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResourceKind {
    DeviceMemory,
    /// Host RAM a dedicated device stages its uploads through.
    HostStaging,
    StateRows,
    RecurrentBanks,
    Workspace,
    Output,
    Feature,
}

impl fmt::Display for SubmitError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Device(source) => source.fmt(formatter),
            Self::Invariant(source) => source.fmt(formatter),
        }
    }
}

impl error::Error for SubmitError {}
