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

/// Weights a sequence's step named could not be read from the model's
/// artifact: a stored table's rows (`StoredTable`) or a streamed layer's
/// experts (`ExpertBank`). The step read other values in their place, so
/// the sequence fails; nothing of a step other than the one that named them
/// is affected.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WeightReadError {
    pub detail: String,
}

impl fmt::Display for WeightReadError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "model weights could not be read: {}", self.detail)
    }
}

impl error::Error for WeightReadError {}

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
    /// A kernel call of the load's programs at statics outside its kernel's
    /// domain on the backend, found while charging the load's memory: the
    /// model's shapes are beyond what the engine's kernels execute.
    KernelDomain(seismic::KernelDomainViolation),
    Resource(CapacityError),
    /// The load's weight bound (a cap or a device room) is below the
    /// weights a load holds with its expert bank at its minimum: each group's least records, what its steps need
    /// resident at once to make progress (`ExpertBankPlan::least_records`).
    WeightBound { bound: crate::WeightBound, needed: u64 },
    /// The load's routed experts cannot be streamed.
    Unstreamed(UnstreamedExperts),
    /// The plan's settle has no prices for its holdings
    /// (`planning::PlanPrices`): the machine's standing figures are not
    /// measured yet (`pending`), or cannot be, with why. A load of it waits
    /// for the measurement job, as its assessment does.
    Unpriced { pending: bool, reason: String },
}

/// Why a load's routed experts cannot be streamed through an expert bank.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum UnstreamedExperts {
    /// The backend lacks an entry of the bank of one of the model's routed
    /// forms, or the shared-route decode form a fused bank-served decode
    /// takes.
    Backend(seismic::BackendName),
    /// The device reserves no address ranges, so a bank cannot resize.
    Unreserved(seismic::BackendName),
    /// The device reserves no address ranges where its backend reserves
    /// on other devices, or discovery could not read how, for `reason`
    /// (`seismic::Limits::reserving`: a Vulkan device with one queue in its
    /// compute family, a granule that could not be read).
    Unreservable { backend: seismic::BackendName, reason: String },
    /// The device holds no host tier, which every miss lands in
    /// (`platform::host_tier`), for `reason`.
    Untiered { backend: seismic::BackendName, reason: String },
    /// The device's single allocation holds `most` slots of a record
    /// tensor, fewer than the bank's `floor`.
    Allocation { floor: u64, most: u64 },
    /// The model routes no experts.
    Unrouted,
    /// A routed layer whose record form (gated or up-only) or selected
    /// experts differ from the first streamed layer's, which the bank's
    /// entries and words share.
    Shape(magnitude_family_contracts::SublayerIndex),
    /// A routed layer that selects in another order than the first streamed
    /// layer (form, score function, selection bias, router input, or the
    /// elements of the weights a route reads of the next layer): a layer's
    /// hints rank the next layer's choices by its own.
    SelectionOrder(magnitude_family_contracts::SublayerIndex),
    /// An expert stack whose experts are not each one range of its stored
    /// experts' bytes, uploaded as stored, nor contiguous in its resident
    /// form.
    Record(WeightRole),
    /// The expert stacks of a routed layer disagree on their experts.
    Geometry(WeightRole),
}

impl fmt::Display for UnstreamedExperts {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Backend(backend) => {
                write!(formatter, "{} has no expert-streaming entries", backend.as_str())
            }
            Self::Unreserved(backend) => write!(
                formatter,
                "this {} device reserves no address ranges for an expert bank",
                backend.as_str()
            ),
            Self::Unreservable { backend, reason } => write!(
                formatter,
                "this {} device reserves no address ranges for an expert bank: {reason}",
                backend.as_str()
            ),
            Self::Untiered { backend, reason } => write!(
                formatter,
                "this {} device holds no host tier for an expert bank's misses to land in: {reason}",
                backend.as_str()
            ),
            Self::Allocation { floor, most } => write!(
                formatter,
                "one allocation holds {most} slots of a record tensor, below the floor of {floor}"
            ),
            Self::Unrouted => formatter.write_str("the model routes no experts"),
            Self::Shape(index) => write!(
                formatter,
                "routed layer {index:?} differs in record form or selected experts from the first streamed layer"
            ),
            Self::SelectionOrder(index) => write!(
                formatter,
                "routed layer {index:?} selects in another order than the first streamed layer"
            ),
            Self::Record(role) => write!(
                formatter,
                "{:?}/{:?} does not hold each expert as one range of its stored bytes",
                role.scope, role.kind
            ),
            Self::Geometry(role) => write!(
                formatter,
                "{:?}/{:?} differs from its layer's expert geometry",
                role.scope, role.kind
            ),
        }
    }
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
            Self::KernelDomain(violation) => violation.fmt(formatter),
            Self::Resource(source) => source.fmt(formatter),
            Self::WeightBound { bound, needed } => write!(
                formatter,
                "{bound} is below the {needed} bytes the load's weights need with its expert bank \
                 at its minimum, the records its steps need at once"
            ),
            Self::Unstreamed(reason) => write!(formatter, "experts cannot stream: {reason}"),
            Self::Unpriced { pending: true, reason } => write!(formatter, "the plan's prices are pending: {reason}"),
            Self::Unpriced { pending: false, reason } => write!(formatter, "the plan has no prices: {reason}"),
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
