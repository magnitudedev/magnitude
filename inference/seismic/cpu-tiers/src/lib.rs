//! The CPU instruction-set tiers as data: each tier's name, architecture,
//! target features, widths and detection, and the version of the CPU
//! library. The CPU library (`seismic-native-cpu`) re-exports them; code
//! generators read them here, without the library.
//!
//! [`tiers!`] is the one definition of every tier: [`Tier`] and the
//! library's tier tokens and `#[target_feature]` boundaries are all
//! expanded from it.

/// The version of the CPU library, part of every CPU implementation digest.
pub const VERSION: &str = concat!(env!("CARGO_PKG_VERSION"), "+cpu-library-6");

/// Passes every tier, lowest first within its architecture, to the macro
/// `$callback`, one `Token module "target_arch" ["feature", ..];` per tier:
/// its token type's name, its lower-case name (`Tier::name`), its
/// `target_arch`, and its target features, each a cumulative superset of the
/// tier below it.
#[macro_export]
macro_rules! tiers {
    ($callback:ident) => {
        $callback! {
            X86V2 x86v2 "x86_64" ["sse3", "ssse3", "sse4.1", "sse4.2", "popcnt"];
            X86V3 x86v3 "x86_64" [
                "sse3", "ssse3", "sse4.1", "sse4.2", "popcnt", "avx", "avx2", "fma", "f16c",
                "bmi1", "bmi2", "lzcnt", "movbe"
            ];
            X86V4 x86v4 "x86_64" [
                "sse3", "ssse3", "sse4.1", "sse4.2", "popcnt", "avx", "avx2", "fma", "f16c",
                "bmi1", "bmi2", "lzcnt", "movbe", "avx512f", "avx512cd", "avx512bw", "avx512dq",
                "avx512vl"
            ];
            X86V4Vnni x86v4vnni "x86_64" [
                "sse3", "ssse3", "sse4.1", "sse4.2", "popcnt", "avx", "avx2", "fma", "f16c",
                "bmi1", "bmi2", "lzcnt", "movbe", "avx512f", "avx512cd", "avx512bw", "avx512dq",
                "avx512vl", "avx512vnni"
            ];
            Neon neon "aarch64" ["neon"];
            NeonDot neondot "aarch64" ["neon", "dotprod"];
        }
    };
}

/// Whether the host has target feature `$feature` of this architecture.
#[cfg(target_arch = "x86_64")]
macro_rules! has {
    ($feature:tt) => {
        std::arch::is_x86_feature_detected!($feature)
    };
}
#[cfg(target_arch = "aarch64")]
macro_rules! has {
    ($feature:tt) => {
        std::arch::is_aarch64_feature_detected!($feature)
    };
}

macro_rules! define {
    ($($token:ident $name:ident $arch:tt [$first:tt $(, $feature:tt)*];)*) => {
        /// A tier of the target architecture, as data: the name recorded in
        /// device identity and tuning keys.
        #[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
        pub enum Tier {
            $($token),*
        }

        impl Tier {
            /// Every tier of every architecture, as code generators
            /// enumerate them.
            pub const EVERY: [Tier; [$(Tier::$token),*].len()] = [$(Tier::$token),*];

            /// Every tier of the architecture this crate is compiled for,
            /// lowest first.
            pub const fn all() -> &'static [Tier] {
                &[$(#[cfg(target_arch = $arch)] Tier::$token),*]
            }

            /// The `target_arch` of the tier.
            pub const fn arch(self) -> &'static str {
                match self {
                    $(Tier::$token => $arch),*
                }
            }

            /// The name of the tier's token type in the CPU library.
            pub const fn token(self) -> &'static str {
                match self {
                    $(Tier::$token => stringify!($token)),*
                }
            }

            pub const fn name(self) -> &'static str {
                match self {
                    $(Tier::$token => stringify!($name)),*
                }
            }

            /// The `target_feature` set of the tier, as generated boundaries
            /// enable it.
            pub const fn features(self) -> &'static str {
                match self {
                    $(Tier::$token => concat!($first $(, ",", $feature)*)),*
                }
            }

            /// Whether the host has every feature of the tier.
            pub fn is_detected(self) -> bool {
                match self {
                    $(
                        #[cfg(target_arch = $arch)]
                        Tier::$token => has!($first) $(&& has!($feature))*,
                    )*
                    #[allow(unreachable_patterns)]
                    _ => false,
                }
            }
        }
    };
}

tiers!(define);

impl Tier {
    /// Bytes of the tier's widest vector, which its bodies load and store
    /// whole: SSE's 128 bits, AVX2's 256, AVX-512's 512, NEON's 128.
    pub const fn vector_bytes(self) -> usize {
        match self {
            Tier::X86V2 | Tier::Neon | Tier::NeonDot => 16,
            Tier::X86V3 => 32,
            Tier::X86V4 | Tier::X86V4Vnni => 64,
        }
    }

    /// The widest vector of any of `tiers`.
    pub const fn widest_vector(tiers: &[Tier]) -> usize {
        let (mut widest, mut tier) = (0, 0);
        while tier < tiers.len() {
            if tiers[tier].vector_bytes() > widest {
                widest = tiers[tier].vector_bytes();
            }
            tier += 1;
        }
        widest
    }

    /// How the tier's quantized (INT8) dots multiply: 0 in portable integer
    /// loops, 1 with vector byte products (AVX2, or NEON `sdot`), 2 with
    /// AVX-512 VNNI.
    pub const fn quantized_dots(self) -> u8 {
        match self {
            Tier::X86V2 | Tier::Neon => 0,
            Tier::X86V3 | Tier::X86V4 | Tier::NeonDot => 1,
            Tier::X86V4Vnni => 2,
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
