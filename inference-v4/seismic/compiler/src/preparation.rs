//! Preparation: the candidate domain, the mandatory general member, the
//! designated default members and every member realized since, and the
//! publication that selects among them. Preparation measures nothing.
//!
//! Interim form (until A4-L4): the domain borrows its device description and
//! compiler registry, so the preparation target borrows them for `'ctx`.

use crate::candidate_domain::CandidateCoordinate;
use crate::target::CompilerRegistry;
use seismic_lang::precision::PrecisionPolicy;
use seismic_target::{DeviceDescription, NativeCompiler, TargetFamily};

#[derive(Clone, Debug, Default)]
pub struct PreparationOptions {
    /// The only preparation input besides the entry and device (D1).
    pub precision: PrecisionPolicy,
}

impl PreparationOptions {
    pub fn new(precision: PrecisionPolicy) -> Self {
        Self { precision }
    }
}

pub struct PreparationTarget<'ctx, T: TargetFamily, C: NativeCompiler<T>> {
    pub device: &'ctx DeviceDescription<T>,
    pub registry: &'ctx CompilerRegistry<T>,
    pub compiler: &'ctx C,
    pub context: &'ctx C::Context,
}

/// Identity of one realized member within one preparation. Never reused.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct MemberId(u32);

impl MemberId {
    pub(crate) fn new(index: usize) -> Self {
        Self(u32::try_from(index).expect("a preparation realizes at most u32::MAX members"))
    }
    pub(crate) fn index(self) -> usize {
        self.0 as usize
    }
}

#[derive(Clone, Debug)]
pub enum MemberRealization {
    Ready(MemberId),
    /// The domain identified the coordinate with the earlier coordinate `of`;
    /// `member` is the member realized for `of`, which serves both.
    Duplicate {
        of: CandidateCoordinate,
        member: MemberId,
    },
}
