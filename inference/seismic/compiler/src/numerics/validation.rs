//! Adversarial diagnostic cases. Observations never grant selection eligibility.
use super::ObservedInvocation;
use seismic_lang::interp::OracleOutcome;

/// Versioned, dedicated numerical corpus, independent of performance seeds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ValidationCase {
    FiniteRandom,
    Zeros,
    AlternatingExtremes,
    SmallMagnitude,
    SpecialValues,
}
impl ValidationCase {
    pub const CORPUS: [(Self, u64); 6] = [
        (Self::FiniteRandom, 0x714d_631a_c083_fa19),
        (Self::FiniteRandom, 0x623e_d4b5_1059_b712),
        (Self::Zeros, 0xa913_6e25_d142_c081),
        (Self::AlternatingExtremes, 0x57dc_32ae_9b14_8f03),
        (Self::SmallMagnitude, 0xd820_413e_a671_52fb),
        (Self::SpecialValues, 0x326b_98c1_4ead_705f),
    ];
}
#[derive(Debug)]
pub struct ValidationObservation {
    pub reference: OracleOutcome,
    pub actual: ObservedInvocation,
    /// Remaining shared interpreter/comparison work allowance.
    pub remaining_work: u64,
}
