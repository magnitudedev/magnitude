//! The registered model families (integration spec §4.3). Recognition runs
//! over every family; exactly one must claim a package's architecture.
//! Whether that family admits the file's feature set is its inspection's
//! decision, which names every feature it refuses.

use crate::error::{UnsupportedModel, UnsupportedReason};
use magnitude_artifacts::gguf::{Directory, Value};
use magnitude_family_contracts::ModelFamily;
use magnitude_family_gemma4::Gemma4Family;
use magnitude_family_lfm2::Lfm2Family;
use magnitude_family_llama::LlamaFamily;
use magnitude_family_muse_glimmer::MuseGlimmerFamily;
use magnitude_family_nemotron_h::NemotronHFamily;
use magnitude_family_qwen35::Qwen35Family;
use magnitude_family_qwen4exp::Qwen4ExpFamily;

static FAMILIES: &[&dyn ModelFamily] = &[
    &Qwen35Family,
    &Qwen4ExpFamily,
    &LlamaFamily,
    &NemotronHFamily,
    &Lfm2Family,
    &Gemma4Family,
    &MuseGlimmerFamily,
];

/// The architectures a separate drafter's GGUF may declare: a DFlash-family
/// block drafter, a Gemma 4 assistant and a Qwen4 prediction head
/// ([`crate::host::bind_draft`] interprets each).
const DRAFTER_ARCHITECTURES: &[&str] = &[
    magnitude_family_dflash::ARCHITECTURE,
    magnitude_family_gemma4::ASSISTANT_ARCHITECTURE,
    magnitude_family_qwen4exp::FAMILY_ID,
];

/// The `general.architecture` values the engine recognizes, as targets and
/// as separate drafters. Recognition is not admission: a recognized file
/// may still be refused for its feature set.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Architectures {
    pub targets: Vec<&'static str>,
    pub drafters: Vec<&'static str>,
}

pub fn architectures() -> Architectures {
    Architectures {
        targets: FAMILIES
            .iter()
            .flat_map(|family| family.architectures().iter().copied())
            .collect(),
        drafters: DRAFTER_ARCHITECTURES.to_vec(),
    }
}

/// The one family that claims `target`'s architecture.
pub fn recognize(target: &Directory) -> Result<&'static dyn ModelFamily, UnsupportedModel> {
    let Some(architecture) = target
        .value("general.architecture")
        .and_then(Value::string)
        .filter(|architecture| !architecture.is_empty())
    else {
        return Err(UnsupportedModel::unreadable(
            "the header names no architecture",
        ));
    };
    let mut matches = FAMILIES
        .iter()
        .copied()
        .filter(|family| family.recognizes(target));
    match (matches.next(), matches.next()) {
        (Some(family), None) => Ok(family),
        (None, _) => Err(UnsupportedModel::reason(
            UnsupportedReason::Architecture {
                architecture: architecture.to_owned(),
            },
            format!("no registered model family recognizes architecture {architecture:?}"),
        )),
        (Some(first), Some(second)) => Err(UnsupportedModel::reason(
            UnsupportedReason::Architecture {
                architecture: architecture.to_owned(),
            },
            format!(
                "architecture {architecture:?} is claimed by both the {} and {} families",
                first.name(),
                second.name()
            ),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use magnitude_artifacts::gguf::{ByteOrder, Metadata, Scalar};

    fn declaring(architecture: &str) -> Directory {
        Directory {
            version: 3,
            byte_order: ByteOrder::Little,
            alignment: 32,
            data_offset: 0,
            metadata: vec![Metadata {
                name: "general.architecture".into(),
                value: Value::Scalar(Scalar::String(architecture.into())),
            }],
            tensors: Vec::new(),
        }
    }

    #[test]
    fn an_unrecognized_architecture_is_named() {
        for architecture in ["deepseek4", "qwen3"] {
            let Err(unsupported) = recognize(&declaring(architecture)) else {
                panic!("no family recognizes {architecture}");
            };
            assert_eq!(
                unsupported.reasons,
                [UnsupportedReason::Architecture {
                    architecture: architecture.into()
                }]
            );
        }
    }

    #[test]
    fn a_header_naming_no_architecture_is_unreadable_never_an_empty_architecture() {
        let mut silent = declaring("");
        for metadata in [silent.metadata.clone(), Vec::new()] {
            silent.metadata = metadata;
            let Err(unsupported) = recognize(&silent) else {
                panic!("nothing recognizes a header without an architecture");
            };
            assert_eq!(unsupported.reasons, [UnsupportedReason::Unreadable]);
        }
    }

    #[test]
    fn every_listed_architecture_is_recognized_by_exactly_one_family() {
        let architectures = architectures();
        for architecture in &architectures.targets {
            assert!(
                recognize(&declaring(architecture)).is_ok(),
                "{architecture}"
            );
        }
        assert!(architectures.targets.contains(&"llama"));
        assert!(architectures.drafters.contains(&"dflash"));
        assert!(!architectures.targets.contains(&"dflash"));
    }
}
