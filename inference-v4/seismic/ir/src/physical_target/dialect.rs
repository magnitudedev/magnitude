//! Pure vocabulary required to construct and inspect executable operations.
//!
//! Implementing this contract requires no compiler, device, native artifact,
//! execution service, or performance model. Intrinsic ownership and numerical
//! rules are consumed by the same checked kernel builders as ordinary ops.

use super::{IntrinsicIdentityBuilder, IntrinsicNumericalSemantics};
use seismic_lang::registry::BackendName;
use std::fmt;

pub trait PhysicalDialect: Sized + 'static + fmt::Debug {
    const NAME: BackendName;

    /// Selected pure dispatch request, shared unchanged by IR, reflection and execution.
    type LaunchDescriptor: Clone
        + fmt::Debug
        + PartialEq
        + Eq
        + std::hash::Hash
        + crate::identity::CanonicalIdentity
        + Send
        + Sync
        + 'static;
    fn ordinary_launch() -> Self::LaunchDescriptor;
    /// Resolve a source participation requirement during construction.
    fn launch_for_participation(
        _facts: &Self::Facts,
        requirement: crate::schedule::LaunchParticipation,
    ) -> Option<Self::LaunchDescriptor> {
        match requirement {
            crate::schedule::LaunchParticipation::Independent => Some(Self::ordinary_launch()),
            crate::schedule::LaunchParticipation::CooperativeGrid => None,
        }
    }

    /// Backend kernel intrinsic operations admitted in typed kernel IR.
    /// Parameterized by backend so a Metal intrinsic cannot appear in a CUDA
    /// kernel (§7.3).
    type Intrinsic: Clone + fmt::Debug + Send + Sync + 'static;

    /// Backend-specific target facts beyond the common limits (SIMD width,
    /// compute capability, language version, ...). Complete before planning.
    type Facts: Clone + fmt::Debug + PartialEq + Send + Sync + 'static;

    /// Canonically identifies one backend intrinsic without Debug rendering
    /// or opaque backend-supplied bytes.
    fn write_intrinsic_identity(
        intrinsic: &Self::Intrinsic,
        identity: &mut IntrinsicIdentityBuilder,
    );

    /// Exact numerical semantics of the concrete intrinsic instruction this
    /// target emits. The registry supplies target-independent source
    /// semantics; this hook closes target facts such as accumulator rounding
    /// and subnormal handling before the solver derives admissibility.
    fn intrinsic_numerics(
        facts: &Self::Facts,
        signature: &seismic_lang::registry::IntrinsicSignature,
        intrinsic: &Self::Intrinsic,
    ) -> IntrinsicNumericalSemantics;

    /// Addressable-resource handles referenced by one concrete intrinsic.
    /// Core uses this projection to prove that every lease has an owner and
    /// that operation-lifetime state belongs to exactly one emitted op.
    fn intrinsic_addressable_resources(
        intrinsic: &Self::Intrinsic,
    ) -> Vec<crate::kernel::ops::AddressableResourceHandle>;
}
