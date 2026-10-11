//! The caller's numerical requirement for a load: α, how many rounding steps
//! of floating storage a configuration may move a kernel's output from its
//! default's. There are no per-operation tolerances and no precision
//! declarations.
use seismic::{BoundEntry, Entry, Limit, NumericalRequirement};

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NumericalProfile {
    pub alpha: Limit,
}

/// A checked profile.
#[derive(Clone, Debug, PartialEq)]
pub struct ResolvedNumericalProfile {
    alpha: f64,
}

/// The standard α: set to admit every technique with a measured effect on
/// speed. The packed weight forms measure 3.1 to 6.7 from their defaults; 8
/// admits them.
const STANDARD_ALPHA: Limit = Limit::from_finite(8.0);

/// The element parameter a kernel's activations are stored as, where the
/// kernel has one.
const ACTIVATION_ELEMENT: &str = "A";

impl ResolvedNumericalProfile {
    /// α: the largest error of a floating result or written state, in
    /// rounding steps of the type that governs it, a configuration may have
    /// against its unit's default.
    pub fn alpha(&self) -> f64 {
        self.alpha
    }

    /// The engine's standard profile, resolved.
    pub fn standard() -> &'static Self {
        static STANDARD: std::sync::OnceLock<ResolvedNumericalProfile> =
            std::sync::OnceLock::new();
        STANDARD.get_or_init(|| {
            NumericalProfile::standard()
                .resolve()
                .expect("the standard profile resolves")
        })
    }

    /// What the candidates of the unit of `entry` are held to under this
    /// profile: within α rounding steps, measured against the entry's
    /// activation type beside each subject's own.
    pub(super) fn requirement<E: Entry>(&self, entry: &BoundEntry<E>) -> NumericalRequirement {
        NumericalRequirement::RoundingSteps {
            alpha: self.alpha,
            activation: entry
                .element(ACTIVATION_ELEMENT)
                .and_then(|element| element.dtype())
                .filter(|dtype| dtype.is_float()),
        }
    }
}

impl NumericalProfile {
    /// The engine's standard profile.
    pub fn standard() -> Self {
        Self {
            alpha: STANDARD_ALPHA,
        }
    }

    pub fn resolve(&self) -> Result<ResolvedNumericalProfile, String> {
        if self.alpha.get() <= 0.0 {
            return Err("numerical profile α must be positive".into());
        }
        Ok(ResolvedNumericalProfile {
            alpha: self.alpha.get(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn profiles_resolve_to_their_alpha_and_refuse_a_nonpositive_one() {
        let looser = NumericalProfile {
            alpha: Limit::from_finite(16.5),
        };
        assert_eq!(looser.resolve().unwrap().alpha(), 16.5);
        assert!(NumericalProfile { alpha: Limit::ZERO }.resolve().is_err());
        let parsed: NumericalProfile = serde_json::from_str(r#"{"alpha":0.5}"#).unwrap();
        assert_eq!(parsed.resolve().unwrap().alpha(), 0.5);
        assert_eq!(
            ResolvedNumericalProfile::standard(),
            &NumericalProfile::standard().resolve().unwrap()
        );
    }
}
