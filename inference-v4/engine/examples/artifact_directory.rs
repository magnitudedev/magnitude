//! Inspect metadata only; this does not establish full model execution support.
use magnitude_artifacts::{gguf, ArtifactIdentity, FileSource, PackageIdentity};
use magnitude_model_qwen35 as qwen35;
use serde_json::{json, Value};
fn scalar(s: &gguf::Scalar) -> Value {
    match s {
        gguf::Scalar::String(v) => json!(v),
        gguf::Scalar::Bool(v) => json!(v),
        gguf::Scalar::Unsigned(v) => json!(v),
        gguf::Scalar::Signed(v) => json!(v),
        gguf::Scalar::Float(v) => json!(v),
    }
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    for path in std::env::args().skip(1) {
        let source = FileSource::open(&path)?;
        let d = gguf::read_directory(&mut source.reader(), gguf::DEFAULT_HEADER_LIMIT)?;
        // Directory-only validation: no content identity has been established.
        let mut model = serde_json::to_value(qwen35::inspect_components(
            &d,
            None,
            PackageIdentity {
                target: ArtifactIdentity([0; 32]),
                projector: None,
            },
        )?)?;
        model.as_object_mut().unwrap().remove("artifact_identity");
        let metadata = d
            .metadata
            .iter()
            .map(|m| {
                (
                    m.name.clone(),
                    match &m.value {
                        gguf::Value::Scalar(s) => scalar(s),
                        gguf::Value::Array(a) => Value::Array(a.iter().map(scalar).collect()),
                    },
                )
            })
            .collect::<serde_json::Map<_, _>>();
        let value = json!({"path":path,"format":"gguf","qwen_description":model,"file_size":source.size(),"version":d.version,"byte_order":format!("{:?}",d.byte_order),"alignment":d.alignment,"data_offset":d.data_offset,"metadata":metadata,"tensors":d.tensors.iter().map(|t|json!({"name":t.name,"shape":t.shape,"encoding":t.encoding as u32,"offset":t.offset,"nbytes":t.nbytes})).collect::<Vec<_>>()});
        println!("{}", serde_json::to_string(&value)?);
    }
    Ok(())
}
