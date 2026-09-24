//! Cloneable RAII leases for executor-owned device resources.
//!
//! A lease owns its allocation directly through `Rc`; dropping the final clone
//! releases the tensor. There is no registry, numeric handle space, explicit
//! release operation, or side-channel keeper object.

use crate::{GraphOutputTensor, ResourceDomainId};
use magnitude_model_contracts::PreparedVisionInput;
use seismic::{Device, Tensor};
use std::{
    cell::Cell,
    cmp::Ordering,
    fmt,
    hash::{Hash, Hasher},
    ops::Range,
    rc::Rc,
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConditioningRange {
    pub destination: Range<usize>,
}

/// Retained results keep the pool claim alive until the final consumer drops
/// its reference. An owned tensor remains available for non-pooled callers.
enum TensorBacking {
    Owned(Tensor),
    Retained(Tensor, Rc<RetentionClaim>),
    Graph(GraphOutputTensor),
}

impl TensorBacking {
    fn tensor(&self) -> &Tensor {
        match self {
            Self::Owned(tensor) => tensor,
            Self::Retained(tensor, _) => tensor,
            Self::Graph(tensor) => tensor.tensor(),
        }
    }
}

/// The claim returns its exact device charge when the last retained feature
/// reference drops. Source pool claims are independent of this allocation.
pub(crate) struct RetentionClaim {
    used: Rc<Cell<u64>>,
    bytes: u64,
}

impl RetentionClaim {
    pub(crate) fn new(used: Rc<Cell<u64>>, bytes: u64) -> Self {
        Self { used, bytes }
    }
}

impl Drop for RetentionClaim {
    fn drop(&mut self) {
        self.used.set(self.used.get() - self.bytes);
    }
}

impl fmt::Debug for TensorBacking {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("TensorBacking")
            .field("element", &self.tensor().element())
            .field("extents", &self.tensor().extents())
            .finish()
    }
}

#[derive(Debug)]
pub(crate) struct FeatureAllocation {
    tensor: Option<TensorBacking>,
    rows: usize,
    width: usize,
}

impl FeatureAllocation {
    pub fn tensor(&self) -> Result<&Tensor, ResourceError> {
        self.tensor
            .as_ref()
            .map(TensorBacking::tensor)
            .ok_or(ResourceError::LogicalLease)
    }
    pub const fn rows(&self) -> usize {
        self.rows
    }
    pub const fn width(&self) -> usize {
        self.width
    }
    pub fn slice(&self, range: Range<usize>) -> Result<Tensor, ResourceError> {
        if range.start >= range.end || range.end > self.rows {
            return Err(ResourceError::InvalidRange {
                start: range.start,
                end: range.end,
                rows: self.rows,
            });
        }
        self.tensor()?
            .slice_leading(range.start as u64, range.end as u64)
            .map_err(|error| ResourceError::Tensor(error.to_string()))
    }
}

#[derive(Debug)]
pub(crate) struct ConditioningAllocation {
    tensor: Option<TensorBacking>,
    rows: usize,
    width: usize,
    ranges: Vec<ConditioningRange>,
}

impl ConditioningAllocation {
    pub fn tensor(&self) -> Result<&Tensor, ResourceError> {
        self.tensor
            .as_ref()
            .map(TensorBacking::tensor)
            .ok_or(ResourceError::LogicalLease)
    }
    pub const fn rows(&self) -> usize {
        self.rows
    }
    pub const fn width(&self) -> usize {
        self.width
    }
    pub fn ranges(&self) -> &[ConditioningRange] {
        &self.ranges
    }
}

#[derive(Debug)]
pub(crate) struct LogitsAllocation {
    tensor: Option<TensorBacking>,
    rows: usize,
    vocabulary: usize,
}

impl LogitsAllocation {
    pub fn tensor(&self) -> Result<&Tensor, ResourceError> {
        self.tensor
            .as_ref()
            .map(TensorBacking::tensor)
            .ok_or(ResourceError::LogicalLease)
    }
    pub const fn rows(&self) -> usize {
        self.rows
    }
    pub const fn vocabulary(&self) -> usize {
        self.vocabulary
    }
}

macro_rules! lease {
    ($name:ident, $allocation:ty) => {
        #[derive(Clone)]
        pub struct $name {
            domain: ResourceDomainId,
            allocation: Rc<$allocation>,
        }
        impl $name {
            pub fn domain(&self) -> &ResourceDomainId {
                &self.domain
            }
            pub(crate) fn allocation(&self) -> &$allocation {
                &self.allocation
            }
            pub fn same_allocation(&self, other: &Self) -> bool {
                self.domain == other.domain && Rc::ptr_eq(&self.allocation, &other.allocation)
            }
            fn address(&self) -> usize {
                Rc::as_ptr(&self.allocation) as usize
            }
        }
        impl fmt::Debug for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.debug_struct(stringify!($name))
                    .field("domain", &self.domain)
                    .finish_non_exhaustive()
            }
        }
        impl PartialEq for $name {
            fn eq(&self, other: &Self) -> bool {
                self.same_allocation(other)
            }
        }
        impl Eq for $name {}
        impl Hash for $name {
            fn hash<H: Hasher>(&self, state: &mut H) {
                self.domain.hash(state);
                self.address().hash(state);
            }
        }
        impl PartialOrd for $name {
            fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
                Some(self.cmp(other))
            }
        }
        impl Ord for $name {
            fn cmp(&self, other: &Self) -> Ordering {
                self.domain
                    .cmp(&other.domain)
                    .then_with(|| self.address().cmp(&other.address()))
            }
        }
    };
}

lease!(FeatureRef, FeatureAllocation);
lease!(ConditioningRef, ConditioningAllocation);
lease!(LogitsRef, LogitsAllocation);

impl FeatureRef {
    /// Tensor-free RAII lease for contract tests and non-device executors.
    pub fn logical(
        domain: ResourceDomainId,
        rows: usize,
        width: usize,
    ) -> Result<Self, ResourceError> {
        if rows == 0 || width == 0 {
            return Err(ResourceError::InvalidFeatureShape);
        }
        Ok(Self {
            domain,
            allocation: Rc::new(FeatureAllocation {
                tensor: None,
                rows,
                width,
            }),
        })
    }
}

impl ConditioningRef {
    pub fn logical(
        domain: ResourceDomainId,
        rows: usize,
        width: usize,
        ranges: Vec<ConditioningRange>,
    ) -> Result<Self, ResourceError> {
        validate_ranges(rows, &ranges)?;
        if width == 0 {
            return Err(ResourceError::InvalidConditioningShape);
        }
        Ok(Self {
            domain,
            allocation: Rc::new(ConditioningAllocation {
                tensor: None,
                rows,
                width,
                ranges,
            }),
        })
    }
}

impl LogitsRef {
    /// Read the leased `[rows, vocabulary]` F32 logits to the host, waiting
    /// for the work that produced them.
    pub fn read_to_host(&self) -> Result<Vec<f32>, ResourceError> {
        let bytes = self
            .allocation
            .tensor()?
            .read_to_host()
            .map_err(|error| ResourceError::Tensor(error.to_string()))?;
        Ok(bytes
            .chunks_exact(4)
            .map(|word| f32::from_le_bytes(word.try_into().expect("four bytes")))
            .collect())
    }

    pub fn rows(&self) -> usize {
        self.allocation.rows()
    }

    pub fn vocabulary(&self) -> usize {
        self.allocation.vocabulary()
    }

    pub fn logical(
        domain: ResourceDomainId,
        rows: usize,
        vocabulary: usize,
    ) -> Result<Self, ResourceError> {
        if rows == 0 || vocabulary == 0 {
            return Err(ResourceError::InvalidLogitsShape);
        }
        Ok(Self {
            domain,
            allocation: Rc::new(LogitsAllocation {
                tensor: None,
                rows,
                vocabulary,
            }),
        })
    }
}

struct ImageAllocation {
    patches: usize,
    prepared: Option<PreparedVisionInput>,
}

#[derive(Clone)]
pub struct ImageRef {
    domain: ResourceDomainId,
    allocation: Rc<ImageAllocation>,
}

impl ImageRef {
    pub fn new(domain: ResourceDomainId, patches: usize) -> Result<Self, ResourceError> {
        if patches == 0 {
            return Err(ResourceError::InvalidImageShape);
        }
        Ok(Self {
            domain,
            allocation: Rc::new(ImageAllocation {
                patches,
                prepared: None,
            }),
        })
    }
    pub fn prepared(
        domain: ResourceDomainId,
        input: PreparedVisionInput,
    ) -> Result<Self, ResourceError> {
        let patches = input
            .grid()
            .into_iter()
            .try_fold(1usize, usize::checked_mul)
            .ok_or(ResourceError::InvalidImageShape)?;
        if patches == 0 || input.pixels().shape().first().copied() != Some(patches) {
            return Err(ResourceError::InvalidImageShape);
        }
        Ok(Self {
            domain,
            allocation: Rc::new(ImageAllocation {
                patches,
                prepared: Some(input),
            }),
        })
    }
    pub fn domain(&self) -> &ResourceDomainId {
        &self.domain
    }
    pub fn patches(&self) -> usize {
        self.allocation.patches
    }
    pub fn prepared_input(&self) -> Result<PreparedVisionInput, ResourceError> {
        self.allocation
            .prepared
            .clone()
            .ok_or(ResourceError::LogicalLease)
    }
    fn address(&self) -> usize {
        Rc::as_ptr(&self.allocation) as usize
    }
}

impl fmt::Debug for ImageRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ImageRef")
            .field("domain", &self.domain)
            .field("patches", &self.patches())
            .finish_non_exhaustive()
    }
}
impl PartialEq for ImageRef {
    fn eq(&self, other: &Self) -> bool {
        self.domain == other.domain && Rc::ptr_eq(&self.allocation, &other.allocation)
    }
}
impl Eq for ImageRef {}
impl Hash for ImageRef {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.domain.hash(state);
        self.address().hash(state);
    }
}
impl PartialOrd for ImageRef {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for ImageRef {
    fn cmp(&self, other: &Self) -> Ordering {
        self.domain
            .cmp(&other.domain)
            .then_with(|| self.address().cmp(&other.address()))
    }
}

#[derive(Clone)]
pub struct ResourceDomain {
    id: ResourceDomainId,
    device: Rc<Device>,
}

impl ResourceDomain {
    pub fn new(id: ResourceDomainId, device: Rc<Device>) -> Self {
        Self { id, device }
    }
    pub fn id(&self) -> &ResourceDomainId {
        &self.id
    }
    pub fn device(&self) -> &Device {
        &self.device
    }
    fn validate(&self, tensor: &Tensor) -> Result<(), ResourceError> {
        tensor
            .belongs_to(&self.device)
            .then_some(())
            .ok_or(ResourceError::ForeignDevice)
    }
    pub fn publish_features(&self, tensor: Tensor) -> Result<FeatureRef, ResourceError> {
        self.publish_feature_backing(TensorBacking::Owned(tensor))
    }
    pub(crate) fn publish_retained_features(
        &self,
        tensor: Tensor,
        claim: Rc<RetentionClaim>,
    ) -> Result<FeatureRef, ResourceError> {
        self.publish_feature_backing(TensorBacking::Retained(tensor, claim))
    }
    pub fn publish_graph_features(
        &self,
        tensor: GraphOutputTensor,
    ) -> Result<FeatureRef, ResourceError> {
        self.publish_feature_backing(TensorBacking::Graph(tensor))
    }
    fn publish_feature_backing(&self, tensor: TensorBacking) -> Result<FeatureRef, ResourceError> {
        self.validate(tensor.tensor())?;
        let [rows, width] = tensor.tensor().extents() else {
            return Err(ResourceError::InvalidFeatureShape);
        };
        let (rows, width) = (
            usize::try_from(*rows).map_err(|_| ResourceError::InvalidFeatureShape)?,
            usize::try_from(*width).map_err(|_| ResourceError::InvalidFeatureShape)?,
        );
        if rows == 0 || width == 0 {
            return Err(ResourceError::InvalidFeatureShape);
        }
        Ok(FeatureRef {
            domain: self.id.clone(),
            allocation: Rc::new(FeatureAllocation {
                tensor: Some(tensor),
                rows,
                width,
            }),
        })
    }
    pub fn publish_conditioning(
        &self,
        tensor: Tensor,
        ranges: Vec<ConditioningRange>,
    ) -> Result<ConditioningRef, ResourceError> {
        self.publish_conditioning_backing(TensorBacking::Owned(tensor), ranges)
    }
    fn publish_conditioning_backing(
        &self,
        tensor: TensorBacking,
        ranges: Vec<ConditioningRange>,
    ) -> Result<ConditioningRef, ResourceError> {
        self.validate(tensor.tensor())?;
        let [rows, width] = tensor.tensor().extents() else {
            return Err(ResourceError::InvalidConditioningShape);
        };
        let (rows, width) = (
            usize::try_from(*rows).map_err(|_| ResourceError::InvalidConditioningShape)?,
            usize::try_from(*width).map_err(|_| ResourceError::InvalidConditioningShape)?,
        );
        if rows == 0 || width == 0 {
            return Err(ResourceError::InvalidConditioningShape);
        }
        validate_ranges(rows, &ranges)?;
        Ok(ConditioningRef {
            domain: self.id.clone(),
            allocation: Rc::new(ConditioningAllocation {
                tensor: Some(tensor),
                rows,
                width,
                ranges,
            }),
        })
    }
    pub fn publish_logits(&self, tensor: Tensor) -> Result<LogitsRef, ResourceError> {
        self.publish_logits_backing(TensorBacking::Owned(tensor))
    }
    pub fn publish_graph_logits(
        &self,
        tensor: GraphOutputTensor,
    ) -> Result<LogitsRef, ResourceError> {
        self.publish_logits_backing(TensorBacking::Graph(tensor))
    }
    fn publish_logits_backing(&self, tensor: TensorBacking) -> Result<LogitsRef, ResourceError> {
        self.validate(tensor.tensor())?;
        let [rows, vocabulary] = tensor.tensor().extents() else {
            return Err(ResourceError::InvalidLogitsShape);
        };
        let (rows, vocabulary) = (
            usize::try_from(*rows).map_err(|_| ResourceError::InvalidLogitsShape)?,
            usize::try_from(*vocabulary).map_err(|_| ResourceError::InvalidLogitsShape)?,
        );
        if rows == 0 || vocabulary == 0 {
            return Err(ResourceError::InvalidLogitsShape);
        }
        Ok(LogitsRef {
            domain: self.id.clone(),
            allocation: Rc::new(LogitsAllocation {
                tensor: Some(tensor),
                rows,
                vocabulary,
            }),
        })
    }
    pub fn validate_feature(&self, lease: &FeatureRef) -> Result<(), ResourceError> {
        self.validate_domain(lease.domain())
    }
    pub fn validate_conditioning(&self, lease: &ConditioningRef) -> Result<(), ResourceError> {
        self.validate_domain(lease.domain())
    }
    pub fn validate_logits(&self, lease: &LogitsRef) -> Result<(), ResourceError> {
        self.validate_domain(lease.domain())
    }
    pub fn feature_span(&self, span: &crate::FeatureSpan) -> Result<Tensor, ResourceError> {
        self.validate_feature(&span.features)?;
        let end = span
            .start
            .checked_add(span.count)
            .ok_or(ResourceError::InvalidRange {
                start: span.start,
                end: usize::MAX,
                rows: span.features.allocation().rows(),
            })?;
        span.features.allocation().slice(span.start..end)
    }
    fn validate_domain(&self, domain: &ResourceDomainId) -> Result<(), ResourceError> {
        (&self.id == domain)
            .then_some(())
            .ok_or(ResourceError::ForeignDomain)
    }
}

fn validate_ranges(rows: usize, ranges: &[ConditioningRange]) -> Result<(), ResourceError> {
    let valid = rows > 0
        && !ranges.is_empty()
        && ranges
            .iter()
            .all(|r| r.destination.start < r.destination.end && r.destination.end <= rows)
        && ranges
            .windows(2)
            .all(|p| p[0].destination.end <= p[1].destination.start);
    valid
        .then_some(())
        .ok_or(ResourceError::InvalidConditioningRanges)
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ResourceError {
    ForeignDevice,
    ForeignDomain,
    LogicalLease,
    InvalidImageShape,
    InvalidFeatureShape,
    InvalidConditioningShape,
    InvalidConditioningRanges,
    InvalidLogitsShape,
    InvalidRange {
        start: usize,
        end: usize,
        rows: usize,
    },
    Tensor(String),
}
impl fmt::Display for ResourceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}
impl std::error::Error for ResourceError {}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn logical_leases_are_raii_and_domain_checked() {
        let domain = ResourceDomainId::new("a").unwrap();
        let a = FeatureRef::logical(domain.clone(), 2, 4).unwrap();
        let b = a.clone();
        assert!(a.same_allocation(&b));
        assert_eq!(a.allocation().rows(), 2);
        assert_eq!(a.domain(), &domain);
    }
}
