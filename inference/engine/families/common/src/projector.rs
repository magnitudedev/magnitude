//! Which projectors a family binds to a target: the one rule every vision
//! family applies before interpreting a projector header.

use magnitude_artifacts::{
    UnsupportedReason,
    gguf::{Directory, PROJECTOR_ARCHITECTURE, Scalar, Value},
};
use magnitude_family_contracts::FamilyError;
use std::{error, fmt};

/// Why a header given as a projector is not one the family binds to the
/// target.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ProjectorIdentityError {
    /// The header is not a CLIP mmproj with a vision encoder: nothing the
    /// engine can build as a projector.
    NotProjector,
    /// A vision projector that declares no type: nothing names it, so
    /// nothing can be read from it.
    Untyped,
    /// A vision projector of a type (`found`) the family does not read:
    /// another model's, which disagrees with the target. `read` are the
    /// types the family reads.
    Foreign {
        found: String,
        read: Vec<&'static str>,
    },
    /// A projector whose features (`projector` wide) are not the target's
    /// embedding width: it cannot feed this decoder. Names (a base model's)
    /// never decide; a fine-tune keeps its base's width.
    Width { projector: u64, target: u64 },
}

impl ProjectorIdentityError {
    /// `Unreadable` for a header that is not a projector or declares no
    /// type; `TargetMismatch` for a projector that disagrees with the target.
    pub fn reasons(&self) -> Vec<UnsupportedReason> {
        match self {
            Self::NotProjector | Self::Untyped => vec![UnsupportedReason::Unreadable],
            Self::Foreign { .. } | Self::Width { .. } => vec![UnsupportedReason::TargetMismatch {
                detail: self.to_string(),
            }],
        }
    }
}

impl fmt::Display for ProjectorIdentityError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotProjector => {
                formatter.write_str("the projector is not a CLIP mmproj with a vision encoder")
            }
            Self::Untyped => formatter.write_str("the projector declares no type"),
            Self::Foreign { found, read } => write!(
                formatter,
                "a {found} projector is not one this model reads ({})",
                read.join(", ")
            ),
            Self::Width { projector, target } => write!(
                formatter,
                "projector width {projector} differs from the target's {target}"
            ),
        }
    }
}

impl error::Error for ProjectorIdentityError {}

impl From<ProjectorIdentityError> for FamilyError {
    fn from(error: ProjectorIdentityError) -> Self {
        FamilyError::unsupported(error.reasons(), error.to_string())
    }
}

/// The keys a projector declares its type under: which one depends on the
/// model, and no projector declares both.
const TYPE_KEYS: [&str; 2] = ["clip.projector_type", "clip.vision.projector_type"];

/// The type of a projector the family binds to a target of `target_width`:
/// `directory` is a CLIP mmproj with a vision encoder whose type is one
/// `types` names, and whose declared feature width
/// (`clip.vision.projection_dim`) is the target's. `types` is the family's
/// one table of the types it reads, each with the reader it selects. A
/// projector that declares no width is the family's reader to refuse.
pub fn projector_type<K: Copy>(
    directory: &Directory,
    types: &[(&'static str, K)],
    target_width: u64,
) -> Result<K, ProjectorIdentityError> {
    if directory.value("general.architecture").and_then(Value::string)
        != Some(PROJECTOR_ARCHITECTURE)
        || directory.value("general.type").and_then(Value::string) != Some("mmproj")
        || directory.value("clip.has_vision_encoder") != Some(&Value::Scalar(Scalar::Bool(true)))
    {
        return Err(ProjectorIdentityError::NotProjector);
    }
    let Some(found) = TYPE_KEYS
        .iter()
        .find_map(|key| directory.value(key).and_then(Value::string))
    else {
        return Err(ProjectorIdentityError::Untyped);
    };
    let Some(kind) = types
        .iter()
        .find(|(name, _)| *name == found)
        .map(|(_, kind)| *kind)
    else {
        return Err(ProjectorIdentityError::Foreign {
            found: found.to_owned(),
            read: types.iter().map(|(name, _)| *name).collect(),
        });
    };
    match directory
        .value("clip.vision.projection_dim")
        .and_then(Value::unsigned)
    {
        Some(projector) if projector != target_width => Err(ProjectorIdentityError::Width {
            projector,
            target: target_width,
        }),
        _ => Ok(kind),
    }
}
