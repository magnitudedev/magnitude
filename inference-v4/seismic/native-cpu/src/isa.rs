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

/// A tier of the target architecture, as data: the name recorded in device
/// identity and tuning keys.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Tier {
    X86V2,
    X86V3,
    X86V4,
    X86V4Vnni,
    Neon,
}

impl Tier {
    /// Every tier of every architecture, as code generators enumerate them.
    pub const EVERY: [Tier; 5] = [
        Tier::X86V2,
        Tier::X86V3,
        Tier::X86V4,
        Tier::X86V4Vnni,
        Tier::Neon,
    ];

    /// The `target_arch` of the tier.
    pub const fn arch(self) -> &'static str {
        match self {
            Tier::X86V2 | Tier::X86V3 | Tier::X86V4 | Tier::X86V4Vnni => "x86_64",
            Tier::Neon => "aarch64",
        }
    }

    /// The name of the tier's token type in this crate.
    pub const fn token(self) -> &'static str {
        match self {
            Tier::X86V2 => "X86V2",
            Tier::X86V3 => "X86V3",
            Tier::X86V4 => "X86V4",
            Tier::X86V4Vnni => "X86V4Vnni",
            Tier::Neon => "Neon",
        }
    }

    /// Every tier of the architecture this crate is compiled for, lowest
    /// first.
    pub const fn all() -> &'static [Tier] {
        #[cfg(target_arch = "x86_64")]
        {
            &[Tier::X86V2, Tier::X86V3, Tier::X86V4, Tier::X86V4Vnni]
        }
        #[cfg(target_arch = "aarch64")]
        {
            &[Tier::Neon]
        }
    }

    pub const fn name(self) -> &'static str {
        match self {
            Tier::X86V2 => "x86v2",
            Tier::X86V3 => "x86v3",
            Tier::X86V4 => "x86v4",
            Tier::X86V4Vnni => "x86v4vnni",
            Tier::Neon => "neon",
        }
    }

    /// The `target_feature` set of the tier, as generated boundaries enable
    /// it.
    pub const fn features(self) -> &'static str {
        match self {
            Tier::X86V2 => X86V2_FEATURES,
            Tier::X86V3 => X86V3_FEATURES,
            Tier::X86V4 => X86V4_FEATURES,
            Tier::X86V4Vnni => X86V4VNNI_FEATURES,
            Tier::Neon => NEON_FEATURES,
        }
    }

    /// Whether the host has every feature of the tier.
    pub fn is_detected(self) -> bool {
        match self {
            #[cfg(target_arch = "x86_64")]
            Tier::X86V2 => X86V2::detect().is_some(),
            #[cfg(target_arch = "x86_64")]
            Tier::X86V3 => X86V3::detect().is_some(),
            #[cfg(target_arch = "x86_64")]
            Tier::X86V4 => X86V4::detect().is_some(),
            #[cfg(target_arch = "x86_64")]
            Tier::X86V4Vnni => X86V4Vnni::detect().is_some(),
            #[cfg(target_arch = "aarch64")]
            Tier::Neon => Neon::detect().is_some(),
            #[allow(unreachable_patterns)]
            _ => false,
        }
    }

    /// The highest tier of this architecture the host supports.
    pub fn detected() -> Option<Tier> {
        Tier::all()
            .iter()
            .rev()
            .copied()
            .find(|tier| tier.is_detected())
    }

    /// Every tier at or below `self` that the host supports: the tier axis a
    /// device offers the tuner.
    pub fn at_or_below(self) -> impl Iterator<Item = Tier> {
        Tier::all()
            .iter()
            .copied()
            .filter(move |tier| *tier <= self && tier.is_detected())
    }
}

pub const X86V2_FEATURES: &str = "sse3,ssse3,sse4.1,sse4.2,popcnt";
pub const X86V3_FEATURES: &str =
    "sse3,ssse3,sse4.1,sse4.2,popcnt,avx,avx2,fma,f16c,bmi1,bmi2,lzcnt,movbe";
pub const X86V4_FEATURES: &str =
    "sse3,ssse3,sse4.1,sse4.2,popcnt,avx,avx2,fma,f16c,bmi1,bmi2,lzcnt,movbe,avx512f,avx512cd,avx512bw,avx512dq,avx512vl";
pub const X86V4VNNI_FEATURES: &str =
    "sse3,ssse3,sse4.1,sse4.2,popcnt,avx,avx2,fma,f16c,bmi1,bmi2,lzcnt,movbe,avx512f,avx512cd,avx512bw,avx512dq,avx512vl,avx512vnni";
pub const NEON_FEATURES: &str = "neon";

/// An instruction-set tier token.
///
/// # Safety
/// An implementation's `detect` returns a value only when every feature in
/// `FEATURES` is present, and `assume_detected` is called only where that
/// holds.
pub unsafe trait Isa: Copy + Send + Sync + 'static {
    const TIER: Tier;
    const FEATURES: &'static str;

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

macro_rules! tier {
    ($name:ident, $tier:ident, $features:ident, [$($feature:tt),*], $detect:ident) => {
        #[doc = concat!("The `", stringify!($name), "` tier token.")]
        #[derive(Clone, Copy, Debug)]
        pub struct $name(());

        // SAFETY: `detect` checks every feature of `FEATURES`.
        unsafe impl Isa for $name {
            const TIER: Tier = Tier::$tier;
            const FEATURES: &'static str = $features;

            fn detect() -> Option<Self> {
                ($(std::arch::$detect!($feature))&&*).then_some($name(()))
            }

            unsafe fn assume_detected() -> Self {
                $name(())
            }
        }
    };
}

#[cfg(target_arch = "x86_64")]
tier!(
    X86V2,
    X86V2,
    X86V2_FEATURES,
    ["sse3", "ssse3", "sse4.1", "sse4.2", "popcnt"],
    is_x86_feature_detected
);
#[cfg(target_arch = "x86_64")]
tier!(
    X86V3,
    X86V3,
    X86V3_FEATURES,
    [
        "sse3", "ssse3", "sse4.1", "sse4.2", "popcnt", "avx", "avx2", "fma", "f16c", "bmi1",
        "bmi2", "lzcnt", "movbe"
    ],
    is_x86_feature_detected
);
#[cfg(target_arch = "x86_64")]
tier!(
    X86V4,
    X86V4,
    X86V4_FEATURES,
    [
        "sse3", "ssse3", "sse4.1", "sse4.2", "popcnt", "avx", "avx2", "fma", "f16c", "bmi1",
        "bmi2", "lzcnt", "movbe", "avx512f", "avx512cd", "avx512bw", "avx512dq", "avx512vl"
    ],
    is_x86_feature_detected
);
#[cfg(target_arch = "x86_64")]
tier!(
    X86V4Vnni,
    X86V4Vnni,
    X86V4VNNI_FEATURES,
    [
        "sse3",
        "ssse3",
        "sse4.1",
        "sse4.2",
        "popcnt",
        "avx",
        "avx2",
        "fma",
        "f16c",
        "bmi1",
        "bmi2",
        "lzcnt",
        "movbe",
        "avx512f",
        "avx512cd",
        "avx512bw",
        "avx512dq",
        "avx512vl",
        "avx512vnni"
    ],
    is_x86_feature_detected
);
#[cfg(target_arch = "aarch64")]
tier!(
    Neon,
    Neon,
    NEON_FEATURES,
    ["neon"],
    is_aarch64_feature_detected
);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_host_has_a_tier_and_every_lower_one() {
        let detected = Tier::detected().expect("every supported host has the floor tier");
        let below = detected.at_or_below().collect::<Vec<_>>();
        assert_eq!(below.last(), Some(&detected));
        assert_eq!(below.first(), Tier::all().first());
    }

    #[test]
    fn feature_strings_are_cumulative() {
        for pair in Tier::all().windows(2) {
            assert!(
                pair[1].features().starts_with(pair[0].features()),
                "{:?}",
                pair
            );
        }
    }
}
