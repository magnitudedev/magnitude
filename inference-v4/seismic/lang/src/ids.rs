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

/// Process-global registry identities, scoped by `REGISTRY_REVISION`.
macro_rules! registry_id {
    ($(#[$doc:meta])* $name:ident, $table:ident) => {
        $(#[$doc])*
        #[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
        pub struct $name(u32);
        impl $name {
            pub(crate) const fn new(index: u32) -> Self { Self(index) }
            pub(crate) const fn index(self) -> usize { self.0 as usize }
        }
        impl fmt::Debug for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "{}#{}", stringify!($name), self.0)
            }
        }
        /// Registry identities serialize as their index; the bundle header
        /// pins `REGISTRY_REVISION`, and decoding rejects an index outside
        /// the table.
        impl Serialize for $name {
            fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
                self.0.serialize(serializer)
            }
        }
        impl<'de> Deserialize<'de> for $name {
            fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
                let index = u32::deserialize(deserializer)?;
                if (index as usize) < crate::registry::table_len(crate::registry::Table::$table) {
                    Ok(Self(index))
                } else {
                    Err(D::Error::custom(concat!(stringify!($name), " outside the registry")))
                }
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
}

pub type ModuleHash = StableId<stable::Module>;
pub type StableFunctionId = StableId<stable::Function>;
pub type StableEntryId = StableId<stable::Entry>;
