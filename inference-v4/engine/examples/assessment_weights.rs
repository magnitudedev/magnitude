//! Assessment experiment: a model's plain decode demand (every launch of one
//! step, keyed by its measured class) and its planned weights, derived from
//! the artifact header alone (no weights are read). With a basis file (as
//! `assessment_measure` stores it), the demand is estimated against that
//! basis at the given depths; the basis is taken at its recorded identity,
//! which the output reports. Prints one JSON document.
//!
//! Usage: assessment_weights <target.gguf> <metal|cuda|vulkan|cpu>
//!            [dense|affine-k8v4] [basis.json depth...]

use magnitude_artifacts::PackageHeaders;
use magnitude_model_executor::{
    assessment::{
        estimate_performance, parse_basis, performance_depths, BasisIdentity, ClassMeasurement,
        DecodeDemand,
    },
    resident_layout, ComponentSelection, ExecutionPath, ModelLoadPlan,
};
use magnitude_model_qwen35 as qwen35;
use magnitude_model_state::KvCodec;
use seismic::BackendName;
use serde_json::{json, Value};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (path, backend, rest) = match args.as_slice() {
        [path, backend, rest @ ..] => (path, backend, rest),
        _ => {
            return Err("usage: assessment_weights <target.gguf> <backend> \
                        [dense|affine-k8v4] [basis.json depth...]"
                .into())
        }
    };
    let backend = BackendName::parse(backend).ok_or("unknown backend")?;
    let codec = match rest.first() {
        Some(name) => name.parse::<KvCodec>()?,
        None => KvCodec::AffineK8V4,
    };
    let headers = PackageHeaders::open(path, None)?;
    let definition = qwen35::inspect_components(headers.target(), None, headers.identity())?;
    let load = ModelLoadPlan::derive_headers(
        &headers,
        &definition,
        ComponentSelection {
            head: false,
            vision: false,
        },
        resident_layout(ExecutionPath::Native, backend),
    )?;
    let demand = DecodeDemand::from_model(&definition, &load, codec)?;
    let terms = demand
        .terms
        .iter()
        .map(|term| {
            json!({
                "class": term.key.class.name(),
                "bindings": term.key.bindings.iter().map(|element| element.name()).collect::<Vec<_>>(),
                "geometry": term.key.geometry,
                "launches": term.launches,
                "bytes": term.bytes,
                "launch_bytes": term.launch_bytes,
                "bytes_per_context_token": term.bytes_per_context_token,
            })
        })
        .collect::<Vec<_>>();
    let estimate = match rest.get(1) {
        None => json!(null),
        Some(file) => {
            let depths = rest[2..]
                .iter()
                .map(|depth| depth.parse::<u32>())
                .collect::<Result<Vec<_>, _>>()?;
            let document = serde_json::from_slice::<Value>(&std::fs::read(file)?)?;
            let recorded = &document["identity"];
            let text = |field: &str| {
                recorded[field]
                    .as_str()
                    .map(str::to_owned)
                    .ok_or(format!("basis identity has no {field}"))
            };
            let identity = BasisIdentity {
                engine_build: text("engine_build")?,
                backend: text("backend")?,
                device: text("device")?,
                protocol_version: u32::try_from(
                    recorded["protocol_version"]
                        .as_u64()
                        .ok_or("basis identity has no protocol version")?,
                )?,
            };
            if identity.backend != backend.as_str() {
                return Err("the basis was measured on another backend".into());
            }
            let basis = parse_basis(&document, &identity).ok_or("not a current basis file")?;
            let unmeasured = demand.unmeasured(&basis);
            if unmeasured.is_empty() {
                let context_limit = u32::try_from(definition.geometry.context_limit)?;
                let estimates = estimate_performance(
                    &demand,
                    &basis,
                    &performance_depths(context_limit, &depths),
                )?;
                json!(estimates
                    .iter()
                    .map(|estimate| json!({
                        "term_median_ms": demand
                            .terms
                            .iter()
                            .map(|term| {
                                let Some(ClassMeasurement::Measured { cost, .. }) =
                                    basis.get(&term.key)
                                else {
                                    unreachable!("every term is measured");
                                };
                                let bytes = term.bytes
                                    + term.bytes_per_context_token
                                        * u64::from(estimate.context_tokens);
                                json!([
                                    term.key.class.name(),
                                    term.key.bindings.iter().map(|element| element.name()).collect::<Vec<_>>(),
                                    cost.seconds(term.launches, bytes, term.launch_bytes).median * 1e3,
                                ])
                            })
                            .collect::<Vec<_>>(),
                        "context_tokens": estimate.context_tokens,
                        "lower_tokens_per_second": estimate.lower_tokens_per_second,
                        "estimated_tokens_per_second": estimate.estimated_tokens_per_second,
                        "upper_tokens_per_second": estimate.upper_tokens_per_second,
                        "confidence": format!("{:?}", estimate.confidence),
                    }))
                    .collect::<Vec<_>>())
            } else {
                json!({
                    "incompatible": unmeasured
                        .iter()
                        .map(|(key, reason)| json!({
                            "key": format!("{key:?}"),
                            "reason": reason,
                        }))
                        .collect::<Vec<_>>(),
                })
            }
        }
    };
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
    println!(
        "{}",
        serde_json::to_string(&json!({
            "path": path,
            "backend": backend.as_str(),
            "kv_codec": codec.identity(),
            "geometry": serde_json::to_value(&definition.geometry)?,
            "decode_demand": terms,
            "estimate": estimate,
            "weights": weights,
        }))?
    );
    Ok(())
}
