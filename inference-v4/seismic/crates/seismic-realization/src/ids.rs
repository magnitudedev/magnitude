//! Identity vocabulary of the closed compiler (frozen by C0).
//!
//! Three families:
//!
//! 1. Occurrence-qualified logical identities (owner O1). A region-local
//!    logical id is never used alone outside its graph; every reference is
//!    qualified by `OwnedGraphKey`.
//! 2. Strategy-local construction identities (owners S1/D1/K1/M1). Sparse
//!    `u32` newtypes allocated during formation.
//! 3. Dense sealed indices (owner P1/N1). Produced only by the physical seal,
//!    stored in `DenseMap`, and in-bounds by construction.

use seismic_lang::logical::{GraphValueId, LogicalStorageId, LogicalViewId, NodeRef, RegionPath, StateTokenId};
use seismic_lang::abi::RangeEndpoint;
use seismic_lang::types::ValuePath;
use std::marker::PhantomData;

macro_rules! sparse_ids {
    ($($(#[$meta:meta])* $name:ident),+ $(,)?) => {$(
        $(#[$meta])*
        #[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
        pub struct $name(pub u32);
    )+};
}

sparse_ids!(
    /// One statically expanded call occurrence in the acyclic occurrence forest.
    OccurrenceId,
    /// One physical strategy of one occurrence (ordinal within the occurrence).
    StrategyId,
    /// One closed kernel block of one strategy.
    BlockId,
    /// One structured schedule step of one strategy.
    StepId,
    /// One residence of one strategy's residence graph.
    ResidenceId,
    /// One executor scalar slot (device-produced control/status scalar).
    ExecutorScalarSlotId,
    /// One declared tuning/placement parameter.
    PlanParamId,
    /// One solver decision exported by a strategy (algorithm/residence choice).
    ChoiceVarId,
    /// One external value entering a kernel block.
    KernelInputId,
    /// One kernel-local SSA value.
    KernelSsaId,
    /// One iteration axis value of a kernel block.
    KernelAxisId,
    /// One external destination of a kernel block.
    KernelOutputId,
    /// One kernel-local addressable storage of a kernel block.
    KernelLocalId,
    /// One status field allocated during formation (template identity).
    StatusFieldTemplateId,
    /// One canonical logical value identity of the occurrence forest.
    CanonicalValueId,
    /// One canonical logical storage identity of the occurrence forest.
    CanonicalStorageId,
    /// One canonical semantic leaf of the occurrence forest.
    CanonicalLeafId,
    /// One derived invocation value of the invocation contract.
    InvocationValueId,
);

macro_rules! dense_ids {
    ($($(#[$meta:meta])* $name:ident),+ $(,)?) => {$(
        $(#[$meta])*
        #[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
        pub struct $name(u32);

        impl $name {
            /// Positional index into the owning dense table.
            pub fn index(self) -> usize {
                self.0 as usize
            }
        }

        impl DenseIndex for $name {
            fn from_index(index: usize) -> Self {
                $name(u32::try_from(index).expect("dense index fits u32"))
            }
            fn index(self) -> usize {
                self.0 as usize
            }
        }
    )+};
}

dense_ids!(
    /// One launch of the sealed physical schedule.
    LaunchIx,
    /// One retained call of the sealed physical schedule.
    CallIx,
    /// One conditional of the sealed physical schedule.
    BranchIx,
    /// One repeat of the sealed physical schedule.
    RepeatIx,
    /// One executor guard of the sealed physical schedule.
    GuardIx,
    /// One physical storage of the sealed global storage table.
    StorageIx,
    /// One executor scalar slot of the sealed plan.
    ScalarSlotIx,
    /// One public buffer binding of the root ABI.
    BufferSlot,
    /// One by-value scalar of the root ABI.
    ScalarSlot,
    /// One field of the compiler-owned result scalar block.
    ResultFieldIx,
    /// One field of the root status block.
    StatusFieldIx,
    /// One native fact folded into a launch handle.
    NativeFactIx,
);

/// Index trait of dense sealed tables. Only this crate constructs values
/// from raw indices (the physical seal).
pub trait DenseIndex: Copy {
    fn from_index(index: usize) -> Self;
    fn index(self) -> usize;
}

/// A dense, contiguous, in-bounds-by-construction table. Constructed only by
/// the physical/native seal (`pub(crate)`); indexing is infallible.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DenseMap<I: DenseIndex, T> {
    items: Vec<T>,
    marker: PhantomData<I>,
}

impl<I: DenseIndex, T> DenseMap<I, T> {
    pub(crate) fn from_vec(items: Vec<T>) -> DenseMap<I, T> {
        DenseMap {
            items,
            marker: PhantomData,
        }
    }

    pub fn len(&self) -> usize {
        self.items.len()
    }

    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = (I, &T)> + '_ {
        self.items
            .iter()
            .enumerate()
            .map(|(index, item)| (I::from_index(index), item))
    }

    pub fn ids(&self) -> impl Iterator<Item = I> + '_ {
        (0..self.items.len()).map(I::from_index)
    }

    pub fn values(&self) -> impl Iterator<Item = &T> + '_ {
        self.items.iter()
    }
}

impl<I: DenseIndex, T> std::ops::Index<I> for DenseMap<I, T> {
    type Output = T;
    fn index(&self, id: I) -> &T {
        &self.items[id.index()]
    }
}

// ---------------------------------------------------------------------------
// Occurrence-qualified logical identities (owner O1)
// ---------------------------------------------------------------------------

/// One immutable logical graph definition instantiated at one exact call
/// occurrence.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct OwnedGraphKey {
    pub occurrence: OccurrenceId,
    pub logical_alternative: u32,
}

/// One occurrence of one physical strategy's root: the occurrence and the
/// logical alternative it realizes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct OwnedOccurrence {
    pub occurrence: OccurrenceId,
    pub logical_alternative: u32,
}

/// One region in an occurrence-owned graph instance.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct OwnedRegionRef {
    pub graph: OwnedGraphKey,
    pub region: RegionPath,
}

/// One node in an occurrence-owned graph instance.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct OwnedNodeRef {
    pub graph: OwnedGraphKey,
    pub node: NodeRef,
}

/// One value in an occurrence-owned graph instance.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct OwnedValueRef {
    pub graph: OwnedGraphKey,
    pub value: GraphValueId,
}

/// One logical storage in an occurrence-owned graph instance.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct OwnedStorageRef {
    pub graph: OwnedGraphKey,
    pub storage: LogicalStorageId,
}

/// One logical view in an occurrence-owned graph instance.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct OwnedViewRef {
    pub graph: OwnedGraphKey,
    pub view: LogicalViewId,
}

/// One state token in an occurrence-owned graph instance.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct OwnedStateRef {
    pub graph: OwnedGraphKey,
    pub state: StateTokenId,
}

/// One region result in an occurrence-owned graph instance.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct OwnedRegionResultRef {
    pub region: OwnedRegionRef,
    pub ordinal: u32,
}

/// One physical leaf of an occurrence-owned logical value. A range remains
/// one semantic leaf but expands to two kernel scalars distinguished by
/// `endpoint`; every other leaf kind uses `None`.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct OwnedValueLeafRef {
    pub value: OwnedValueRef,
    pub path: ValuePath,
    pub endpoint: Option<RangeEndpoint>,
}

/// One safety obligation: the `index`-th obligation of one owned node.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ObligationRef {
    pub node: OwnedNodeRef,
    pub index: u32,
}
