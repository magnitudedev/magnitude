//! Instruction-set tiers.
//!
//! A tier is a closed set of target features. Each tier is a zero-sized token
//! type implementing [`Isa`]; a value of it can only be obtained from
//! detection ([`Isa::detect`]) or, in generated code, from the one
//! `#[target_feature]` boundary that the device's detected tier discharges
//! ([`Isa::assume_detected`]). Holding a token is the proof that its features
//! are present.
//!
//! Everything below a boundary is `#[inline(always)]` and generic over the
//! tier, so it compiles with the boundary's features: plain loops over
//! fixed-width lanes vectorize to the tier's registers, and intrinsics called
//! inside inline into it.

pub use seismic_cpu_tiers::Tier;

/// An instruction-set tier token.
///
/// # Safety
/// An implementation's `detect` returns a value only when every feature of
/// `TIER` ([`Tier::features`]) is present, and `assume_detected` is called
/// only where that holds.
pub unsafe trait Isa: Copy + Send + Sync + 'static {
    const TIER: Tier;
    /// The tier's quantized-dot mode ([`Tier::quantized_dots`]).
    const QUANTIZED_DOTS: u8 = Self::TIER.quantized_dots();

    /// The token, when the host has the tier's features.
    fn detect() -> Option<Self>;

    /// The token without detection.
    ///
    /// # Safety
    /// The caller guarantees the tier's features are present: generated
    /// code calls it only in a function selected for a device whose detected
    /// tier includes this one.
    unsafe fn assume_detected() -> Self;
}

/// The tier tokens of this architecture, one per tier of
/// `seismic_cpu_tiers::tiers!`, each detected by its tier's features.
macro_rules! tokens {
    ($($token:ident $name:ident $arch:tt [$($feature:tt),*];)*) => {
        $(
            #[cfg(target_arch = $arch)]
            #[doc = concat!("The `", stringify!($token), "` tier token.")]
            #[derive(Clone, Copy, Debug)]
            pub struct $token(());

            // SAFETY: `detect` checks every feature of the tier.
            #[cfg(target_arch = $arch)]
            unsafe impl Isa for $token {
                const TIER: Tier = Tier::$token;

                fn detect() -> Option<Self> {
                    Tier::$token.is_detected().then_some($token(()))
                }

                unsafe fn assume_detected() -> Self {
                    $token(())
                }
            }
        )*
    };
}

seismic_cpu_tiers::tiers!(tokens);

