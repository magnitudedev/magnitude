//! Strict header reading names every key and tensor it refuses, each as a
//! typed reason.

use magnitude_artifacts::{
    gguf::{ByteOrder, Directory, Encoding, Metadata as Entry, Scalar, TensorDescriptor, Value},
    UnsupportedReason,
};
use magnitude_family_common::{HeaderError, Metadata, Tensors};
use magnitude_family_contracts::FamilyError;

fn directory(keys: &[&str], tensors: &[&str]) -> Directory {
    Directory {
        version: 3,
        byte_order: ByteOrder::Little,
        alignment: 32,
        data_offset: 0,
        metadata: keys
            .iter()
            .map(|key| Entry {
                name: (*key).into(),
                value: Value::Scalar(Scalar::Unsigned(1)),
            })
            .collect(),
        tensors: tensors
            .iter()
            .map(|name| TensorDescriptor {
                name: (*name).into(),
                shape: vec![32],
                encoding: Encoding::F32,
                offset: 0,
                nbytes: 128,
            })
            .collect(),
    }
}

#[test]
fn every_unknown_architecture_key_is_named() {
    let header = directory(
        &[
            "arch.block_count",
            "arch.sliding_window",
            "general.name",
            "arch.rope.scaling.type",
        ],
        &[],
    );
    let Err(error) = Metadata::new(&header, "arch", &["block_count"]) else {
        panic!("unknown keys are refused");
    };
    assert_eq!(
        error,
        HeaderError::UnknownMetadata(vec![
            "arch.sliding_window".into(),
            "arch.rope.scaling.type".into()
        ])
    );
    assert_eq!(
        FamilyError::from(error).reasons,
        [
            UnsupportedReason::UnrecognizedMetadata {
                key: "arch.sliding_window".into()
            },
            UnsupportedReason::UnrecognizedMetadata {
                key: "arch.rope.scaling.type".into()
            },
        ]
    );
}

#[test]
fn every_unbound_tensor_is_named() {
    let header = directory(&[], &["bound.weight", "extra.weight", "rope_freqs.weight"]);
    let mut tensors = Tensors::new(&header);
    tensors.bind("bound.weight", &[32]).unwrap();
    let error = tensors.finish().unwrap_err();
    assert_eq!(
        FamilyError::from(error).reasons,
        [
            UnsupportedReason::UnrecognizedTensor {
                name: "extra.weight".into()
            },
            UnsupportedReason::UnrecognizedTensor {
                name: "rope_freqs.weight".into()
            },
        ]
    );
}

#[test]
fn a_malformed_header_names_no_reason() {
    let error = FamilyError::from(HeaderError::MissingMetadata("arch.block_count".into()));
    assert!(error.reasons.is_empty());
    assert_eq!(error.detail, "missing metadata arch.block_count");
}
