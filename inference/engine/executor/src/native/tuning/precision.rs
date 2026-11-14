//! The caller's numerical requirement for a load: α, how many rounding steps
//! of floating storage a configuration may move a kernel's output from its
//! default's, and whether a streamed load's misses may be computed on the
//! CPU, in its own arithmetic. There are no per-operation tolerances and no
//! precision declarations.
use seismic::{BoundEntry, Entry, Limit, NumericalRequirement};

/// A load's profile. It crosses the worker wire in the load's manifest, so
/// it serializes every field; a profile file states only what differs from
/// the standard ([`Self::read`]).
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NumericalProfile {
    pub alpha: Limit,
    /// Whether a streamed layer's CPU site may compute its misses: its
    /// arithmetic is admitted against α by its own qualification. Without,
    /// no site verdict is requested or used and every miss is the device's.
    pub cpu_sites: bool,
}

/// A checked profile.
#[derive(Clone, Debug, PartialEq)]
pub struct ResolvedNumericalProfile {
    alpha: f64,
    cpu_sites: bool,
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
        self.requirement_of(entry.element(ACTIVATION_ELEMENT).and_then(|element| element.dtype()))
    }

    /// What a unit whose activations are stored as `activation` is held to
    /// under this profile: within α rounding steps, measured against that
    /// type (a floating one) beside each subject's own.
    pub(crate) fn requirement_of(&self, activation: Option<seismic::DType>) -> NumericalRequirement {
        NumericalRequirement::RoundingSteps { alpha: self.alpha, activation: activation.filter(|dtype| dtype.is_float()) }
    }

    /// What a streamed layer's CPU site, whose pool computes in
    /// `activation`, is held to under this profile ([`Self::requirement_of`]);
    /// none where the profile approves no CPU site
    /// ([`NumericalProfile::cpu_sites`]), so no verdict is requested or read
    /// and every miss is the device's.
    pub(crate) fn site_requirement(&self, activation: Option<seismic::DType>) -> Option<NumericalRequirement> {
        self.cpu_sites.then(|| self.requirement_of(activation))
    }
}

impl NumericalProfile {
    /// The engine's standard profile.
    pub fn standard() -> Self {
        Self {
            alpha: STANDARD_ALPHA,
            cpu_sites: true,
        }
    }

    /// The profile file at `path` (`--numerical-profile`).
    pub fn read(path: &std::path::Path) -> Result<Self, String> {
        std::fs::read(path)
            .map_err(|error| error.to_string())
            .and_then(|bytes| Self::from_json(&bytes))
            .map_err(|error| format!("numerical profile {}: {error}", path.display()))
    }

    /// A profile file's: the standard profile with the fields the file's
    /// object states in place of its own, so `{"cpu_sites": false}` is the
    /// standard α with no CPU sites. A field the profile does not have is
    /// refused.
    fn from_json(bytes: &[u8]) -> Result<Self, String> {
        let serde_json::Value::Object(stated) = serde_json::from_slice(bytes).map_err(|error| error.to_string())? else {
            return Err("a numerical profile is a JSON object".into());
        };
        let serde_json::Value::Object(mut profile) =
            serde_json::to_value(Self::standard()).expect("a profile serializes")
        else {
            unreachable!("a profile serializes as an object")
        };
        profile.extend(stated);
        serde_json::from_value(serde_json::Value::Object(profile)).map_err(|error| error.to_string())
    }

    pub fn resolve(&self) -> Result<ResolvedNumericalProfile, String> {
        if self.alpha.get() <= 0.0 {
            return Err("numerical profile α must be positive".into());
        }
        Ok(ResolvedNumericalProfile {
            alpha: self.alpha.get(),
            cpu_sites: self.cpu_sites,
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
            cpu_sites: true,
        };
        assert_eq!(looser.resolve().unwrap().alpha(), 16.5);
        assert!(NumericalProfile { alpha: Limit::ZERO, cpu_sites: true }.resolve().is_err());
        let parsed = NumericalProfile::from_json(br#"{"alpha":0.5}"#).unwrap();
        assert_eq!(parsed.resolve().unwrap().alpha(), 0.5);
        assert!(parsed.resolve().unwrap().site_requirement(None).is_some(), "a profile approves CPU sites unless it says otherwise");
        let unsited = NumericalProfile::from_json(br#"{"cpu_sites":false}"#).unwrap();
        assert_eq!(unsited, NumericalProfile { cpu_sites: false, ..NumericalProfile::standard() });
        assert_eq!(unsited.resolve().unwrap().site_requirement(None), None);
        assert!(NumericalProfile::from_json(br#"{"cpu_site":false}"#).is_err(), "an unknown field is refused");
        assert!(NumericalProfile::from_json(b"[]").is_err());
        assert_eq!(
            ResolvedNumericalProfile::standard(),
            &NumericalProfile::standard().resolve().unwrap()
        );
    }
}
