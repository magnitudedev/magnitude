//! The Qwen geometry names the rope scaling it refuses.

use magnitude_artifacts::{
    gguf::{ByteOrder, Directory, Metadata as Entry, Scalar, Value},
    UnsupportedReason,
};

#[test]
fn rope_scaling_is_named() {
    let header = Directory {
        version: 3,
        byte_order: ByteOrder::Little,
        alignment: 32,
        data_offset: 0,
        metadata: vec![Entry {
            name: "qwen35.rope.scaling.type".into(),
            value: Value::Scalar(Scalar::String("yarn".into())),
        }],
        tensors: Vec::new(),
    };
    let error = magnitude_family_qwen::Geometry::read(
        &magnitude_family_qwen::Metadata::new(&header, "qwen35"),
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
