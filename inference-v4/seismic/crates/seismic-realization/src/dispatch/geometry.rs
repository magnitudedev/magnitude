//! Dispatch equations interpreted by concrete realization and symbolic export.
//! This algebra describes geometry only; it supplies no legality or timing claims.
use super::TilePlacement;

pub trait Algebra {
    type Value: Copy;
    type Error;
    fn constant(&mut self, value: u64) -> Result<Self::Value, Self::Error>;
    fn product(&mut self, a: Self::Value, b: Self::Value) -> Result<Self::Value, Self::Error>;
    fn ceil_div(&mut self, a: Self::Value, b: Self::Value) -> Result<Self::Value, Self::Error>;
    fn maximum(&mut self, a: Self::Value, b: Self::Value) -> Result<Self::Value, Self::Error>;
}

pub struct Mapping<V> {
    pub counts: Vec<V>,
    pub strides: Vec<V>,
    pub work_items: V,
}

/// Caller establishes one positive step per extent. Empty mappings retain the
/// concrete ABI's zero strides, including axes preceding a zero extent.
pub fn mapping<A: Algebra>(
    a: &mut A,
    extents: &[u64],
    steps: &[A::Value],
) -> Result<Mapping<A::Value>, A::Error> {
    assert_eq!(extents.len(), steps.len());
    let mut counts = Vec::with_capacity(extents.len());
    for (&extent, &step) in extents.iter().zip(steps) {
        let extent = a.constant(extent)?;
        counts.push(a.ceil_div(extent, step)?);
    }
    let zero = a.constant(0)?;
    let mut strides = vec![zero; extents.len()];
    let mut work_items = zero;
    if !extents.contains(&0) {
        work_items = a.constant(1)?;
        for index in (0..counts.len()).rev() {
            strides[index] = work_items;
            work_items = a.product(work_items, counts[index])?;
        }
    }
    Ok(Mapping {
        counts,
        strides,
        work_items,
    })
}

pub struct Dispatch<V> {
    pub groups: V,
    pub threads_per_group: V,
    pub dispatched_lanes: V,
}
pub fn dispatch<A: Algebra>(
    a: &mut A,
    work: A::Value,
    lanes: A::Value,
    items: A::Value,
) -> Result<Dispatch<A::Value>, A::Error> {
    let threads_per_group = a.product(lanes, items)?;
    let groups = a.ceil_div(work, items)?;
    let dispatched_lanes = a.product(groups, threads_per_group)?;
    Ok(Dispatch {
        groups,
        threads_per_group,
        dispatched_lanes,
    })
}

pub struct Storage<V> {
    pub private_elements_per_lane: V,
    pub shared_elements_per_item: V,
    pub private_bytes_per_lane: V,
    pub shared_bytes_per_group: V,
}
pub fn storage<A: Algebra>(
    a: &mut A,
    capacity: A::Value,
    bytes: u64,
    placement: &TilePlacement,
    lanes: A::Value,
    items: A::Value,
) -> Result<Storage<A::Value>, A::Error> {
    let zero = a.constant(0)?;
    let one = a.constant(1)?;
    let (private, shared) = match placement {
        TilePlacement::Replicated => (a.maximum(capacity, one)?, zero),
        TilePlacement::Distributed => {
            let per_lane = a.ceil_div(capacity, lanes)?;
            (a.maximum(per_lane, one)?, zero)
        }
        TilePlacement::GroupShared => (zero, a.maximum(capacity, one)?),
    };
    let bytes = a.constant(bytes)?;
    let private_bytes_per_lane = a.product(private, bytes)?;
    let shared_bytes = a.product(shared, bytes)?;
    let shared_bytes_per_group = a.product(shared_bytes, items)?;
    Ok(Storage {
        private_elements_per_lane: private,
        shared_elements_per_item: shared,
        private_bytes_per_lane,
        shared_bytes_per_group,
    })
}

pub(super) struct Concrete;
impl Algebra for Concrete {
    type Value = u64;
    type Error = String;
    fn constant(&mut self, value: u64) -> Result<u64, String> {
        Ok(value)
    }
    fn product(&mut self, a: u64, b: u64) -> Result<u64, String> {
        a.checked_mul(b)
            .ok_or_else(|| "dispatch geometry product overflow".into())
    }
    fn ceil_div(&mut self, a: u64, b: u64) -> Result<u64, String> {
        if b == 0 {
            return Err("dispatch geometry divisor must be positive".into());
        }
        Ok(a.div_ceil(b))
    }
    fn maximum(&mut self, a: u64, b: u64) -> Result<u64, String> {
        Ok(a.max(b))
    }
}
