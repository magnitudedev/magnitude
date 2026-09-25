//! One device measurement class for model assessment, with no model file.
//!
//! Usage: assessment_measure <metal|cuda|vulkan|cpu> <dense_output|dense_expand> <resident-element> [activation]
//! Example: assessment_measure metal dense_expand q4k@rows16 bf16

use magnitude_model_executor::assessment::DeviceMeasurementRunner;
use seismic::{BackendName, DeviceCatalog, Element};
use serde_json::json;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if !(3..=4).contains(&args.len()) {
        return Err(
            "usage: assessment_measure <backend> <entry> <resident-element> [activation]".into(),
        );
    }
    let backend = BackendName::parse(&args[0]).ok_or("unknown backend")?;
    let entry = &args[1];
    let weight = Element::named(&args[2]).ok_or("unknown resident element")?;
    let activation = args
        .get(3)
        .map(|name| Element::named(name).ok_or("unknown activation element"))
        .transpose()?
        .unwrap_or_else(Element::bf16);
    let device = DeviceCatalog::discover()?.open_backend(backend)?;
    let runner = DeviceMeasurementRunner::new(&device);
    let (device_identity, timing_protocol, small, large, cost) = match entry.as_str() {
        "dense_output" => {
            let result = runner.measure_dense_output(weight, activation)?;
            (
                result.device_identity,
                result.timing_protocol,
                result.small,
                result.large,
                result.cost,
            )
        }
        "dense_expand" => {
            let result = runner.measure_dense_expand(weight, activation)?;
            (
                result.device_identity,
                result.timing_protocol,
                result.small,
                result.large,
                result.cost,
            )
        }
        _ => return Err("entry must be dense_output or dense_expand".into()),
    };
    let point = |p: &magnitude_model_executor::assessment::StreamingPoint| {
        json!({
            "hidden": p.hidden,
            "features": p.features,
            "weight_bytes": p.weight_bytes,
            "native_artifact": &p.native_artifact,
            "median_seconds": p.timing.median,
            "deviation_seconds": p.timing.deviation,
            "samples_seconds": p.timing.samples,
            "rotation_bytes": p.timing.rotation_bytes,
        })
    };
    println!(
        "{}",
        serde_json::to_string(&json!({
            "device": device.info().selector.to_string(),
            "device_identity": &device_identity,
            "backend": backend.as_str(),
            "entry": entry,
            "timing_protocol": {
                "samples": timing_protocol.samples,
                "min_sample_seconds": timing_protocol.min_sample_seconds,
            },
            "weight": weight.name(),
            "activation": activation.name(),
            "small": point(&small),
            "large": point(&large),
            "cost": match &cost {
                Ok(cost) => json!({
                    "status": "measured",
                    "launch_seconds": cost.launch_seconds,
                    "seconds_per_byte": cost.seconds_per_byte,
                }),
                Err(reason) => json!({"status": "unmeasured", "reason": reason}),
            },
        }))?
    );
    Ok(())
}
