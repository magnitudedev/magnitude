//! Assessment experiment: every planned weight's resident representation and
//! bytes for a native backend layout, derived from the artifact header alone
//! (no weights are read, no device is opened). Prints one JSON document.
//!
//! Usage: assessment_weights <target.gguf> <metal|cuda|vulkan>

use magnitude_artifacts::Package;
use magnitude_model_executor::{resident_layout, ComponentSelection, ExecutionPath, ModelLoadPlan};
use magnitude_model_qwen35 as qwen35;
use seismic::BackendName;
use serde_json::json;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [path, backend] = args.as_slice() else {
        return Err("usage: assessment_weights <target.gguf> <metal|cuda|vulkan>".into());
    };
    let backend = BackendName::parse(backend).ok_or("unknown backend")?;
    let package = Package::open_without_projector(path)?;
    let definition = qwen35::inspect_package(&package)?;
    let load = ModelLoadPlan::derive(
        &package.manifest(),
        &definition,
        ComponentSelection {
            head: false,
            vision: false,
        },
        resident_layout(ExecutionPath::Native, backend),
    )?;
    let weights = load
        .target()
        .iter()
        .map(|weight| {
            json!({
                "scope": format!("{:?}", weight.role.scope),
                "kind": format!("{:?}", weight.role.kind),
                "tensor": weight.descriptor.name,
                "source": weight.source.name(),
                "resident": weight.resident.name(),
                "shape": weight.shape,
                "resident_bytes": weight.resident_bytes,
            })
        })
        .collect::<Vec<_>>();
    let geometry = serde_json::to_value(&definition.geometry)?;
    println!(
        "{}",
        serde_json::to_string(&json!({
            "path": path,
            "backend": backend.as_str(),
            "geometry": geometry,
            "weights": weights,
        }))?
    );
    Ok(())
}
