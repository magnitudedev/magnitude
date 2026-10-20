//! Strict header reading names every key and tensor it refuses, each as a
//! typed reason.

use magnitude_artifacts::{
    gguf::{ByteOrder, Directory, Encoding, Metadata as Entry, Scalar, TensorDescriptor, Value},
    UnsupportedReason,
};
use magnitude_family_common::{architecture_keys, HeaderError, Metadata, Tensors};
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
fn only_the_architecture_namespace_is_read_strictly() {
    let header = directory(
        &[
            "arch.block_count",
            "general.quantized_by",
            "mradermacher.quantize_version",
            "unsloth.version",
            "imatrix.chunks_count",
            "quantize.imatrix.file",
            "split.count",
            "other_arch.block_count",
            "archive.block_count",
            "tokenizer.ggml.model",
        ],
        &[],
    );
    assert!(Metadata::new(&header, "arch", &["block_count"]).is_ok());
    assert_eq!(
        architecture_keys(&header, "arch")
            .map(|(key, entry)| (key, entry.name.as_str()))
            .collect::<Vec<_>>(),
        [("block_count", "arch.block_count")]
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
fn a_header_missing_a_key_the_family_needs_names_it() {
    let error = FamilyError::from(HeaderError::MissingMetadata("arch.block_count".into()));
    assert_eq!(
        error.reasons,
        [UnsupportedReason::MissingMetadata {
            key: "arch.block_count".into()
        }]
    );
    assert_eq!(error.detail, "missing metadata arch.block_count");
}
