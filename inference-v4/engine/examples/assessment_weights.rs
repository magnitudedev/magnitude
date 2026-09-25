//! Assessment experiment: every planned weight's resident representation and
//! bytes for a native backend layout, derived from the artifact header alone
//! (no weights are read, no device is opened). Prints one JSON document.
//!
//! Usage: assessment_weights <target.gguf> <metal|cuda|vulkan>

use magnitude_artifacts::PackageHeaders;
use magnitude_model_executor::{
    assessment::AssessmentDecodeDemand, resident_layout, AssessmentBindingEvidence,
    AssessmentHeaderBounds, AssessmentMemoryTerms, ComponentSelection, ExecutionPath,
    ModelLoadPlan, PlannedMethod,
};
use magnitude_model_qwen35 as qwen35;
use magnitude_model_state::KvCodec;
use seismic::BackendName;
use serde_json::json;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [path, backend] = args.as_slice() else {
        return Err("usage: assessment_weights <target.gguf> <metal|cuda|vulkan>".into());
    };
    let backend = BackendName::parse(backend).ok_or("unknown backend")?;
    let headers = PackageHeaders::open(path, None)?;
    let definition = qwen35::inspect_components(headers.target(), None, headers.identity())?;
    let selection = ComponentSelection {
        head: false,
        vision: false,
    };
    let load = ModelLoadPlan::derive_headers(
        &headers,
        &definition,
        selection,
        resident_layout(ExecutionPath::Native, backend),
    )?;
    let memory = AssessmentMemoryTerms::derive(
        &definition,
        &load,
        selection,
        KvCodec::AffineK8V4,
        PlannedMethod::Plain,
    )?;
    let header_bounds = AssessmentHeaderBounds::derive(&definition, &load, KvCodec::AffineK8V4)?;
    let demand = AssessmentDecodeDemand::from_model(&definition, &load, KvCodec::AffineK8V4)?;
    let dense_output_demands = demand
        .dense_output
        .into_iter()
        .map(|demand| {
            json!({
                "weight": demand.weight.name(),
                "activation": demand.activation.name(),
                "launches": demand.launches,
                "resident_bytes": demand.resident_bytes,
            })
        })
        .collect::<Vec<_>>();
    let dense_expand_demands = demand
        .dense_expand
        .into_iter()
        .map(|demand| {
            json!({
                "norm": demand.norm.name(),
                "gate": demand.gate.name(),
                "up": demand.up.name(),
                "activation": demand.activation.name(),
                "launches": demand.launches,
                "resident_bytes": demand.resident_bytes,
            })
        })
        .collect::<Vec<_>>();
    let attention_project_demands = demand
        .attention_project
        .into_iter()
        .map(|demand| {
            json!({
                "shape": {
                    "hidden": demand.shape.hidden,
                    "kv_heads": demand.shape.kv_heads,
                    "group": demand.shape.group,
                    "rotary_pairs": demand.shape.rotary_pairs,
                    "width": demand.shape.width,
                },
                "norm": demand.norm.name(),
                "query_gate": demand.query_gate.name(),
                "key": demand.key.name(),
                "value": demand.value.name(),
                "activation": demand.activation.name(),
                "launches": demand.launches,
                "resident_bytes": demand.resident_bytes,
            })
        })
        .collect::<Vec<_>>();
    let readout = demand.readout;
    let binding = AssessmentBindingEvidence::inspect_dense_paths(
        &definition,
        &load,
        KvCodec::AffineK8V4,
        backend,
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
            "dense_path_binding": match binding {
                AssessmentBindingEvidence::Unsupported { entry, reason } => json!({
                    "status": "unsupported", "entry": entry, "reason": reason,
                }),
                AssessmentBindingEvidence::Unconfirmed => json!({"status": "unconfirmed"}),
            },
            "dense_output_demands": dense_output_demands,
            "dense_expand_demands": dense_expand_demands,
            "attention_project_demands": attention_project_demands,
            "readout_demand": {
                "norm": readout.norm.name(),
                "weight": readout.weight.name(),
                "activation": readout.activation.name(),
                "launches": 1,
                "resident_bytes": readout.resident_bytes,
            },
            "memory": {
                "target_weights": memory.target_weights,
                "head_weights": memory.head_weights,
                "vision_weights": memory.vision_weights,
                "fit_depth": memory.fit_depth,
                "history_at_fit_depth": memory.history_at_fit_depth()?,
                "recurrent_at_fit_workload": memory.recurrent_at_fit_workload()?,
                "exact_resident_bytes": memory.minimum_required_bytes()?,
                "exact_resident_and_program_bytes": memory
                    .minimum_required_bytes()?
                    .checked_add(header_bounds.prepared_program_bytes)
                    .ok_or("assessment exact lower bound overflow")?,
                "prepared_program_bytes": header_bounds.prepared_program_bytes,
                "startup_additional_bound_bytes": header_bounds.startup_additional_bytes,
                "graph_resource_bound": null,
            },
            "weights": weights,
        }))?
    );
    Ok(())
}
