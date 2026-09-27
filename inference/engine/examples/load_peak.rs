//! Load one package in process at its full supported context and report the
//! load preview (the planned startup peak), the ready census and the
//! process's peak physical footprint. Development evidence for the planner.
//!
//! `load_peak --model TARGET.gguf [--projector MMPROJ.gguf] --cache-dir DIR [--device metal]`

use magnitude_engine::{
    composition::{start_in_process, EngineConfiguration},
    options::{standard_service_limits, ModelPolicy, PackageOptions, ProjectorSelection},
};
use magnitude_executor::{
    platform::{DeviceRequest, MemoryReserves},
    ExecutionPath,
};
use std::{path::PathBuf, time::Instant};

fn main() -> Result<(), String> {
    let mut model = None;
    let mut projector = ProjectorSelection::Disabled;
    let mut cache_dir = None;
    let mut device = DeviceRequest::Automatic;
    let mut args = std::env::args().skip(1);
    while let Some(flag) = args.next() {
        let mut value = || args.next().ok_or_else(|| format!("{flag} requires a value"));
        match flag.as_str() {
            "--model" => model = Some(PathBuf::from(value()?)),
            "--projector" => projector = ProjectorSelection::Explicit(PathBuf::from(value()?)),
            "--cache-dir" => cache_dir = Some(PathBuf::from(value()?)),
            "--device" => device = value()?.parse().map_err(|error| format!("{error}"))?,
            other => return Err(format!("unknown flag {other}")),
        }
    }
    let resolved = EngineConfiguration {
        package: PackageOptions {
            target: model.ok_or("--model is required")?,
            projector,
            draft: None,
        },
        model: ModelPolicy::default(),
        context_tokens: None,
        service: standard_service_limits(),
        path: ExecutionPath::Native,
        device,
        kernel_cache: cache_dir,
        reserves: MemoryReserves::standard(),
    }
    .resolve()
    .map_err(|error| error.to_string())?;
    let catalog = seismic::DeviceCatalog::discover().map_err(|error| error.to_string())?;
    let preview = resolved
        .manifest
        .preview(&catalog)
        .map_err(|error| error.to_string())?;
    println!("preview {preview:?}");
    let started = Instant::now();
    let engine = start_in_process(resolved, |progress| eprintln!("load: {progress:?}"))
        .map_err(|error| error.to_string())?;
    println!("ready in {:.2} s", started.elapsed().as_secs_f64());
    println!("resources {:?}", engine.ready_info().resources);
    println!("census {:?}", engine.ready_info().census);
    drop(engine);
    Ok(())
}
