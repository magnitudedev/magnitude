//! The registered model families (integration spec §4.3). Recognition runs
//! over every family; exactly one must claim a package.

use crate::error::UnsupportedModel;
use magnitude_artifacts::gguf::{Directory, Value};
use magnitude_family_contracts::ModelFamily;
use magnitude_family_qwen35::Qwen35Family;

static FAMILIES: &[&dyn ModelFamily] = &[&Qwen35Family];

/// The one family that recognizes `target`.
pub fn recognize(target: &Directory) -> Result<&'static dyn ModelFamily, UnsupportedModel> {
    let mut matches = FAMILIES
        .iter()
        .copied()
        .filter(|family| family.recognizes(target));
    let architecture = || {
        target
            .value("general.architecture")
            .and_then(Value::string)
            .unwrap_or("<missing>")
            .to_owned()
    };
    match (matches.next(), matches.next()) {
        (Some(family), None) => Ok(family),
        (None, _) => Err(UnsupportedModel::Family {
            reason: format!(
                "no registered model family recognizes architecture {:?}",
                architecture()
            ),
        }),
        (Some(first), Some(second)) => Err(UnsupportedModel::Family {
            reason: format!(
                "architecture {:?} is claimed by both the {} and {} families",
                architecture(),
                first.name(),
                second.name()
            ),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use magnitude_artifacts::gguf::{ByteOrder, Metadata, Scalar};

    #[test]
    fn an_unrecognized_architecture_names_the_family_once() {
        let directory = Directory {
            version: 3,
            byte_order: ByteOrder::Little,
            alignment: 32,
            data_offset: 0,
            metadata: vec![Metadata {
                name: "general.architecture".into(),
                value: Value::Scalar(Scalar::String("gemma4".into())),
            }],
            tensors: Vec::new(),
        };
        let Err(unsupported) = recognize(&directory) else {
            panic!("no family recognizes gemma4");
        };
        assert_eq!(
            unsupported.to_string(),
            "unsupported model family: no registered model family recognizes architecture \"gemma4\""
        );
    }
}
