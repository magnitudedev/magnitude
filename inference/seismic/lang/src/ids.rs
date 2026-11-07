//! Scope-correct semantic identities.
//!
//! An ordinal is never an identity on its own. Every module-, program-,
//! schema-, function-, or region-local handle carries the runtime owner that
//! allocated it. Owners are deliberately not serialized: a checked bundle
//! carries ordinals, and one decode rebinds them to owners allocated fresh
//! for it (`crate::wire`). Stable content identities serialize as they are.

use crate::wire;
use serde::de::Error as _;
use serde::ser::Error as _;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::fmt;
use std::marker::PhantomData;
use std::num::NonZeroU64;
use std::sync::atomic::{AtomicU64, Ordering};

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct Owner(NonZeroU64);

impl Owner {
    fn fresh() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(1);
        let value = NEXT
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
                current.checked_add(1).filter(|next| *next != 0)
            })
            .unwrap_or_else(|_| panic!("semantic owner id space exhausted"));
        let value = NonZeroU64::new(value)
            .unwrap_or_else(|| panic!("semantic owner allocator produced zero"));
        Self(value)
    }
}

impl fmt::Debug for Owner {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "owner#{}", self.0)
    }
}

/// Runtime identity of one decoded or source-checked module.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ModuleId(Owner);

impl ModuleId {
    pub(crate) fn fresh() -> Self {
        Self(Owner::fresh())
    }
}

impl fmt::Debug for ModuleId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ModuleId({:?})", self.0)
    }
}

/// Runtime identity of one monomorphized semantic program.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct ProgramId(Owner);

impl ProgramId {
    pub(crate) fn fresh() -> Self {
        Self(Owner::fresh())
    }
}

/// Runtime identity of one call schema.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct SchemaId(Owner);

impl SchemaId {
    pub(crate) fn fresh() -> Self {
        Self(Owner::fresh())
    }
}

macro_rules! module_id {
    ($(#[$doc:meta])* $name:ident) => {
        $(#[$doc])*
        #[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
        pub struct $name { module: ModuleId, ordinal: u32 }
        impl $name {
            pub(crate) const fn new(module: ModuleId, ordinal: u32) -> Self { Self { module, ordinal } }
            pub(crate) const fn module(self) -> ModuleId { self.module }
            pub(crate) const fn index(self) -> usize { self.ordinal as usize }
        }
        impl fmt::Debug for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "{}({:?}, #{})", stringify!($name), self.module, self.ordinal)
            }
        }
        impl Serialize for $name {
            fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
                wire::encode_module(self.module).map_err(S::Error::custom)?;
                self.ordinal.serialize(serializer)
            }
        }
        impl<'de> Deserialize<'de> for $name {
            fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
                let ordinal = u32::deserialize(deserializer)?;
                let module = wire::decode_module(ordinal).map_err(D::Error::custom)?;
                Ok(Self::new(module, ordinal))
            }
        }
    };
}

module_id!(/// One exported entry in a checked module.
    EntryId);

macro_rules! program_id {
    ($(#[$doc:meta])* $name:ident, $kind:ident) => {
        $(#[$doc])*
        #[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
        pub struct $name { program: ProgramId, ordinal: u32 }
        impl $name {
            pub(crate) const fn new(program: ProgramId, ordinal: u32) -> Self { Self { program, ordinal } }
            pub(crate) const fn program(self) -> ProgramId { self.program }
            pub(crate) const fn index(self) -> usize { self.ordinal as usize }
        }
        impl fmt::Debug for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "{}({:?}, #{})", stringify!($name), self.program.0, self.ordinal)
            }
        }
        impl Serialize for $name {
            fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
                wire::encode_program(self.program).map_err(S::Error::custom)?;
                self.ordinal.serialize(serializer)
            }
        }
        impl<'de> Deserialize<'de> for $name {
            fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
                let ordinal = u32::deserialize(deserializer)?;
                let program = wire::decode_program(wire::ProgramHandle::$kind, ordinal)
                    .map_err(D::Error::custom)?;
                Ok(Self::new(program, ordinal))
            }
        }
    };
}

program_id!(/// One monomorphized function family.
    FamilyId, Family);
program_id!(/// One monomorphized function body.
    FunctionId, Function);

macro_rules! schema_id {
    ($(#[$doc:meta])* $name:ident) => {
        $(#[$doc])*
        #[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
        pub struct $name { schema: SchemaId, ordinal: u32 }
        impl $name {
            pub(crate) const fn new(schema: SchemaId, ordinal: u32) -> Self { Self { schema, ordinal } }
            pub(crate) const fn schema(self) -> SchemaId { self.schema }
            pub(crate) const fn index(self) -> usize { self.ordinal as usize }
        }
        impl fmt::Debug for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "{}({:?}, #{})", stringify!($name), self.schema.0, self.ordinal)
            }
        }
        impl Serialize for $name {
            fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
                let slot = wire::encode_schema(self.schema).map_err(S::Error::custom)?;
                (slot, self.ordinal).serialize(serializer)
            }
        }
        impl<'de> Deserialize<'de> for $name {
            fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
                let (slot, ordinal) = <(u32, u32)>::deserialize(deserializer)?;
                let schema = wire::decode_schema(slot).map_err(D::Error::custom)?;
                Ok(Self::new(schema, ordinal))
            }
        }
    };
}

schema_id!(/// One symbolic call-schema dimension.
    DimensionId);
schema_id!(/// One call-schema parameter.
    ParameterId);

/// One structured region owned by a function.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RegionId {
    function: FunctionId,
    ordinal: u32,
}

impl RegionId {
    pub(crate) const fn new(function: FunctionId, ordinal: u32) -> Self {
        Self { function, ordinal }
    }
    pub const fn function(self) -> FunctionId {
        self.function
    }
    pub const fn index(self) -> usize {
        self.ordinal as usize
    }
}

impl fmt::Debug for RegionId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "RegionId({:?}, #{})", self.function, self.ordinal)
    }
}

/// One function-global SSA value.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct SemanticValueId {
    function: FunctionId,
    ordinal: u32,
}

impl SemanticValueId {
    pub(crate) const fn new(function: FunctionId, ordinal: u32) -> Self {
        Self { function, ordinal }
    }
    pub const fn function(self) -> FunctionId {
        self.function
    }
    pub const fn index(self) -> usize {
        self.ordinal as usize
    }
}

impl fmt::Debug for SemanticValueId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SemanticValueId({:?}, #{})", self.function, self.ordinal)
    }
}

/// A node within a region.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NodeId {
    region: RegionId,
    ordinal: u32,
}

impl NodeId {
    pub(crate) const fn new(region: RegionId, ordinal: u32) -> Self {
        Self { region, ordinal }
    }
    pub const fn region(self) -> RegionId {
        self.region
    }
    pub(crate) const fn ordinal(self) -> usize {
        self.ordinal as usize
    }
}

impl fmt::Debug for NodeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "NodeId({:?}, #{})", self.region, self.ordinal)
    }
}

/// One lexical loop binder owned by its loop-body region.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BinderId {
    region: RegionId,
    ordinal: u32,
}

impl BinderId {
    pub(crate) const fn new(region: RegionId, ordinal: u32) -> Self {
        Self { region, ordinal }
    }
    pub(crate) const fn region(self) -> RegionId {
        self.region
    }
}

impl fmt::Debug for BinderId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "BinderId({:?}, #{})", self.region, self.ordinal)
    }
}

/// Process-global registry identities. An identity is its entry's key
/// (`registry::Table::key`): a digest of what names the entry among the
/// rows of its table, so registering another entry changes no existing
/// identity, nor anything compared, hashed, ordered, printed or serialized
/// by it. It also carries the entry's position in this process's table,
/// which only indexes that table ([`Self::position`]).
macro_rules! registry_id {
    ($(#[$doc:meta])* $name:ident, $table:ident) => {
        $(#[$doc])*
        #[derive(Clone, Copy)]
        pub struct $name {
            key: u32,
            position: u32,
        }
        impl $name {
            pub(crate) const fn new(key: u32, position: u32) -> Self {
                Self { key, position }
            }
            /// The entry's stable key, unchanged by any other entry the
            /// registry adds: what persisted identities hold.
            pub const fn key(self) -> u32 {
                self.key
            }
            /// The entry's position in its registry table in this process:
            /// an index into that table alone, which another entry the
            /// registry adds can move, so nothing persisted holds it.
            pub const fn position(self) -> usize {
                self.position as usize
            }
        }
        impl PartialEq for $name {
            fn eq(&self, other: &Self) -> bool {
                self.key == other.key
            }
        }
        impl Eq for $name {}
        impl PartialOrd for $name {
            fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
                Some(self.cmp(other))
            }
        }
        impl Ord for $name {
            fn cmp(&self, other: &Self) -> std::cmp::Ordering {
                self.key.cmp(&other.key)
            }
        }
        impl std::hash::Hash for $name {
            fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
                self.key.hash(state)
            }
        }
        impl fmt::Debug for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "{}#{:08x}", stringify!($name), self.key)
            }
        }
        /// Registry identities serialize as their key, and decoding
        /// rejects a key no entry of the table has.
        impl Serialize for $name {
            fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
                self.key.serialize(serializer)
            }
        }
        impl<'de> Deserialize<'de> for $name {
            fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
                let key = u32::deserialize(deserializer)?;
                crate::registry::position(crate::registry::Table::$table, key)
                    .map(|position| Self::new(key, position))
                    .ok_or_else(|| D::Error::custom(concat!(stringify!($name), " outside the registry")))
            }
        }
    };
}

registry_id!(/// One registered capability namespace.
    CapabilityId, Capability);
registry_id!(/// One registered typed intrinsic signature.
    IntrinsicId, Intrinsic);
registry_id!(/// One registered element representation.
    RepresentationId, Representation);
registry_id!(/// One exact registered conversion between element representations.
    RepresentationConversionId, RepresentationConversion);

/// Content-derived stable identity of a semantic entity.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct StableId<Kind> {
    digest: [u8; 32],
    kind: PhantomData<Kind>,
}

impl<Kind> StableId<Kind> {
    pub(crate) const fn new(digest: [u8; 32]) -> Self {
        Self {
            digest,
            kind: PhantomData,
        }
    }
    pub const fn digest(&self) -> &[u8; 32] {
        &self.digest
    }
}

impl<Kind> Serialize for StableId<Kind> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.digest.serialize(serializer)
    }
}

impl<'de, Kind> Deserialize<'de> for StableId<Kind> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        <[u8; 32]>::deserialize(deserializer).map(Self::new)
    }
}

impl<Kind> fmt::Debug for StableId<Kind> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for byte in &self.digest[..8] {
            write!(f, "{byte:02x}")?;
        }
        write!(f, "…")
    }
}

pub mod stable {
    #[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
    pub enum Module {}
    #[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
    pub enum Function {}
    #[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
    pub enum Entry {}
    #[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
    pub enum Content {}
}

pub type ModuleHash = StableId<stable::Module>;
pub type StableFunctionId = StableId<stable::Function>;
pub type StableEntryId = StableId<stable::Entry>;
/// What an entry's meaning depends on: its family's declarations and the
/// transitive closure of what they may call, with module constants as
/// their values, under this compiler and registry. Unlike its
/// [`StableEntryId`], nothing else in its module is part of it.
pub type EntryContent = StableId<stable::Content>;
