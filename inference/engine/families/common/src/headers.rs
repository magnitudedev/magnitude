//! Exact GGUF headers of catalog components, as dumped under
//! `inference/validation/results/catalog-headers/` (metadata plus the full
//! tensor directory; long metadata arrays keep only their head), for family
//! tests.

use magnitude_artifacts::gguf::{
    ByteOrder, Directory, Encoding, Metadata, Scalar, TensorDescriptor, Value,
};
use serde_json::Value as Json;
use std::path::PathBuf;

pub fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../validation/results/catalog-headers")
}

/// Every dumped component: `(model, role, file)`.
pub fn index() -> Vec<(String, String, String)> {
    let index: Json =
        serde_json::from_slice(&std::fs::read(root().join("index.json")).unwrap()).unwrap();
    index
        .as_array()
        .unwrap()
        .iter()
        .map(|row| {
            let field = |name: &str| row[name].as_str().unwrap().to_owned();
            (field("model"), field("role"), field("file"))
        })
        .collect()
}

fn scalar(value: &Json) -> Scalar {
    match value {
        Json::Bool(value) => Scalar::Bool(*value),
        Json::String(value) => Scalar::String(value.clone()),
        Json::Number(number) => {
            if let Some(value) = number.as_u64() {
                Scalar::Unsigned(value)
            } else if let Some(value) = number.as_i64() {
                Scalar::Signed(value)
            } else {
                Scalar::Float(number.as_f64().unwrap())
            }
        }
        other => panic!("metadata scalar expected, received {other}"),
    }
}

fn value(value: &Json) -> Value {
    match value {
        Json::Array(values) => Value::Array(values.iter().map(scalar).collect()),
        Json::Object(truncated) => Value::Array(
            truncated["head"]
                .as_array()
                .unwrap()
                .iter()
                .map(scalar)
                .collect(),
        ),
        other => Value::Scalar(scalar(other)),
    }
}

fn encoding(name: &str) -> Encoding {
    let id = match name {
        "F32" => 0,
        "F16" => 1,
        "Q4_0" => 2,
        "Q5_0" => 6,
        "Q5_1" => 7,
        "Q8_0" => 8,
        "Q3_K" => 11,
        "Q4_K" => 12,
        "Q5_K" => 13,
        "Q6_K" => 14,
        "IQ4_NL" => 20,
        "IQ3_S" => 21,
        "IQ4_XS" => 23,
        "I32" => 26,
        "BF16" => 30,
        "MXFP4" => 39,
        "NVFP4" => 40,
        "Q1_0" => 41,
        other => panic!("unmapped GGML type {other}"),
    };
    Encoding::try_from(id).unwrap()
}

/// The directory of one dumped component file.
pub fn directory(file: &str) -> Directory {
    let path = root().join(file);
    let header: Json = serde_json::from_slice(
        &std::fs::read(&path).unwrap_or_else(|error| panic!("{path:?}: {error}")),
    )
    .unwrap();
    let metadata = header["metadata"]
        .as_object()
        .unwrap()
        .iter()
        .map(|(name, item)| Metadata {
            name: name.clone(),
            value: value(item),
        })
        .collect();
    let tensors = header["tensors"]
        .as_array()
        .unwrap()
        .iter()
        .map(|tensor| {
            let encoding = encoding(tensor["type"].as_str().unwrap());
            // Dumped in GGML storage order; the directory is outermost first.
            let mut shape = tensor["shape"]
                .as_array()
                .unwrap()
                .iter()
                .map(|extent| extent.as_u64().unwrap())
                .collect::<Vec<_>>();
            let elements = shape.iter().product::<u64>();
            shape.reverse();
            TensorDescriptor {
                name: tensor["name"].as_str().unwrap().to_owned(),
                shape,
                encoding,
                offset: 0,
                nbytes: elements / encoding.block_elements() * encoding.block_bytes(),
            }
        })
        .collect();
    Directory {
        version: 3,
        byte_order: ByteOrder::Little,
        alignment: 32,
        data_offset: 0,
        metadata,
        tensors,
    }
}

/// Sets metadata `name` to `value`, replacing any recorded value.
pub fn set(directory: &mut Directory, name: &str, value: Value) {
    directory.metadata.retain(|entry| entry.name != name);
    directory.metadata.push(Metadata {
        name: name.into(),
        value,
    });
}

pub fn remove_tensor(directory: &mut Directory, name: &str) {
    let before = directory.tensors.len();
    directory.tensors.retain(|tensor| tensor.name != name);
    assert_eq!(directory.tensors.len() + 1, before, "{name} is recorded once");
}

pub fn reshape_tensor(directory: &mut Directory, name: &str, shape: &[u64]) {
    directory
        .tensors
        .iter_mut()
        .find(|tensor| tensor.name == name)
        .unwrap_or_else(|| panic!("{name} is recorded"))
        .shape = shape.into();
}

/// Adds an F32 tensor.
pub fn add_tensor(directory: &mut Directory, name: &str, shape: &[u64]) {
    directory.tensors.push(TensorDescriptor {
        name: name.into(),
        shape: shape.into(),
        encoding: Encoding::F32,
        offset: 0,
        nbytes: shape.iter().product::<u64>() * 4,
    });
}
