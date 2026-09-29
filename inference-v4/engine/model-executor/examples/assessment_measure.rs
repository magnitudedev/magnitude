//! Measure the complete declared assessment basis of one backend with no
//! model file, print each class as it completes and the total wall time, and
//! store the basis JSON in a directory.
//!
//! Usage: assessment_measure <metal|cuda|vulkan|cpu> <basis-dir>

use magnitude_model_executor::assessment::{
    basis_file_name, measure_basis_observed, measurement_plan, store_basis, BasisIdentity,
    ClassMeasurement, CostModel,
};
use magnitude_model_executor::platform::MemoryReserves;
use seismic::{BackendName, DeviceCatalog};
use std::path::PathBuf;
use std::time::Instant;

const ENGINE_BUILD: &str = concat!(env!("CARGO_PKG_NAME"), "@", env!("CARGO_PKG_VERSION"));

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    let [backend, directory] = args.as_slice() else {
        return Err("usage: assessment_measure <metal|cuda|vulkan|cpu> <basis-dir>".into());
    };
    let backend = BackendName::parse(backend).ok_or("unknown backend")?;
    let directory = PathBuf::from(directory);
    let catalog = DeviceCatalog::discover()?;
    let device = catalog.open_backend(backend)?;
    let identity = BasisIdentity::for_device(&device, ENGINE_BUILD);
    let plan = measurement_plan(backend);
    eprintln!(
        "measuring {} classes on {} ({})",
        plan.len(),
        identity.device,
        identity.engine_build
    );
    let began = Instant::now();
    let basis = measure_basis_observed(
        &catalog,
        &device,
        MemoryReserves::standard(),
        identity,
        |key, measurement, profile| {
            let bindings = key
                .bindings
                .iter()
                .map(|element| element.name())
                .collect::<Vec<_>>()
                .join(",");
            let geometry = key
                .geometry
                .iter()
                .map(|(name, value)| format!("{name}={value}"))
                .collect::<Vec<_>>()
                .join(",");
            let result = match measurement {
                ClassMeasurement::Unsupported { reason } => format!("UNSUPPORTED {reason}"),
                ClassMeasurement::Measured { points, cost } => {
                    let model = match &cost.model {
                        CostModel::PerLaunch { seconds } => {
                            format!("{:.2} us/launch", seconds * 1e6)
                        }
                        CostModel::Linear(linear) => format!(
                            "{:.2} us/launch + {:.3} ps/byte ({:.1} GB/s)",
                            linear.launch_seconds * 1e6,
                            linear.seconds_per_byte * 1e12,
                            1e-9 / linear.seconds_per_byte
                        ),
                        CostModel::Curve(curve) => curve
                            .iter()
                            .map(|(bytes, seconds)| {
                                format!(
                                    "{:.1}MB {:.1}us ({:.0} GB/s)",
                                    *bytes as f64 / 1e6,
                                    seconds * 1e6,
                                    *bytes as f64 / seconds / 1e9
                                )
                            })
                            .collect::<Vec<_>>()
                            .join(", "),
                    };
                    let bytes = points
                        .iter()
                        .map(|point| point.bytes.to_string())
                        .collect::<Vec<_>>()
                        .join("/");
                    format!(
                        "{model}; bytes {bytes}; slow {:.3} fast {:.3}",
                        cost.slow_factor, cost.fast_factor
                    )
                }
            };
            eprintln!(
                "{:>7.3}s (form {:.3} alloc {:.3} time {:.3}) {} [{bindings}] [{geometry}] {result}",
                profile.total.as_secs_f64(),
                profile.formation.as_secs_f64(),
                profile.allocation.as_secs_f64(),
                profile.timing.as_secs_f64(),
                key.class.name(),
            );
        },
    )?;
    let seconds = began.elapsed().as_secs_f64();
    let unsupported = basis
        .classes
        .iter()
        .filter(|(_, measurement)| matches!(measurement, ClassMeasurement::Unsupported { .. }))
        .count();
    store_basis(&directory, &basis)?;
    let name = basis_file_name(&basis.identity).ok_or("basis identity names no backend")?;
    eprintln!(
        "measured {} classes ({unsupported} unsupported) in {seconds:.2}s; basis {}",
        basis.classes.len(),
        directory.join(name).display()
    );
    Ok(())
}
