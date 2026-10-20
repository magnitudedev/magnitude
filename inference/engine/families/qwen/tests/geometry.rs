//! The Qwen metadata reader and geometry name what they refuse.

use magnitude_artifacts::{
    UnsupportedReason,
    gguf::{ByteOrder, Directory, Metadata as Entry, Scalar, Value},
};
use magnitude_family_qwen::{GEOMETRY_KEYS, Geometry, Metadata};

fn header(metadata: &[(&str, Value)]) -> Directory {
    Directory {
        version: 3,
        byte_order: ByteOrder::Little,
        alignment: 32,
        data_offset: 0,
        metadata: metadata
            .iter()
            .map(|(name, value)| Entry {
                name: (*name).into(),
                value: value.clone(),
            })
            .collect(),
        tensors: Vec::new(),
    }
}

#[test]
fn rope_scaling_is_named() {
    let header = header(&[(
        "qwen35.rope.scaling.type",
        Value::Scalar(Scalar::String("yarn".into())),
    )]);
    let error = Geometry::read(
        &Metadata::new(&header, "qwen35", GEOMETRY_KEYS).unwrap(),
        false,
    )
    .unwrap_err();
    assert_eq!(
        error.reasons,
        [UnsupportedReason::RopeScaling {
            kind: "yarn".into()
        }]
    );
}

#[test]
fn every_unknown_key_of_the_family_is_named() {
    let header = header(&[
        ("general.name", Value::Scalar(Scalar::String("x".into()))),
        (
            "qwen35.embedding_length",
            Value::Scalar(Scalar::Unsigned(8)),
        ),
        (
            "qwen35.attention.sliding_window",
            Value::Scalar(Scalar::Unsigned(8)),
        ),
        (
            "qwen35.rope.scaling.factor",
            Value::Scalar(Scalar::Float(4.0)),
        ),
    ]);
    let Err(error) = Metadata::new(&header, "qwen35", GEOMETRY_KEYS) else {
        panic!("unknown keys are refused");
    };
    assert_eq!(
        error.reasons,
        [
            UnsupportedReason::UnrecognizedMetadata {
                key: "qwen35.attention.sliding_window".into()
            },
            UnsupportedReason::UnrecognizedMetadata {
                key: "qwen35.rope.scaling.factor".into()
            },
        ]
    );
}

#[test]
fn missing_and_malformed_keys_are_named() {
    let header = header(&[(
        "qwen35.embedding_length",
        Value::Scalar(Scalar::String("wide".into())),
    )]);
    let m = Metadata::new(&header, "qwen35", GEOMETRY_KEYS).unwrap();
    assert_eq!(
        m.integer("attention.head_count").unwrap_err().reasons,
        [UnsupportedReason::MissingMetadata {
            key: "qwen35.attention.head_count".into()
        }]
    );
    assert_eq!(
        m.integer("embedding_length").unwrap_err().reasons,
        [UnsupportedReason::UnrecognizedMetadata {
            key: "qwen35.embedding_length".into()
        }]
    );
}
