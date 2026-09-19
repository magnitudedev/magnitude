//! Full-artifact cold-path comparison using the existing functional test machine.
//! This is not a calibrated hardware performance or optimality benchmark.
#[cfg(target_os = "macos")]
#[allow(dead_code)]
#[path = "../tests/support/decoder_hardware.rs"]
mod decoder_hardware;

#[cfg(target_os = "macos")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use seismic_engine::models::qwen35::loading::Model;
    use seismic_runtime::{
        tuner::{Algorithm, NeighborhoodOptions},
        Device,
    };
    use serde_json::json;
    use std::{
        rc::Rc,
        time::{Duration, Instant},
    };

    let args: Vec<_> = std::env::args().collect();
    if args.len() != 4 {
        return Err("usage: qwen_tuning_compare ARTIFACT exact|joint OUTPUT_JSON".into());
    }
    let method = args[2].as_str();
    let mut settings = decoder_hardware::metal();
    settings.search.options.algorithm = match method {
        "exact" => Algorithm::Exact,
        "joint" => Algorithm::Neighborhood(NeighborhoodOptions {
            seed: 0,
            max_neighborhood_variables: 8,
            population_size: 1,
            exploration: false,
            restart_after: u64::MAX,
            ..Default::default()
        }),
        _ => return Err("method must be exact or joint".into()),
    };
    settings.search.limits.time = Some(Duration::from_secs(1));
    settings.search.limits.work = u64::MAX;
    settings.search.limits.memory_bytes = Some(512 * 1024 * 1024);
    let device = Rc::new(Device::metal()?);
    eprintln!("ready: full Qwen artifact, {method}, functional hardware profile");
    let start = Instant::now();
    let mut stages = Vec::new();
    let mut stage = "artifact_open";
    let mut compiled_kernels = None;
    let result: Result<(), String> = (|| {
        let before = Instant::now();
        let model = Model::open(&args[1])?;
        stages.push(json!({"stage":stage,"seconds":before.elapsed().as_secs_f64()}));
        eprintln!(
            "artifact opened after {:.3}s",
            start.elapsed().as_secs_f64()
        );
        stage = "weight_import_and_decoder_construction";
        let before = Instant::now();
        let mut decoder = model.load(device, settings, 256, 1)?;
        stages.push(json!({"stage":stage,"seconds":before.elapsed().as_secs_f64()}));
        compiled_kernels = Some(decoder.compiled_kernel_count());
        eprintln!("decoder loaded after {:.3}s", start.elapsed().as_secs_f64());
        stage = "cold_prefill";
        let mut state = decoder.state_store().create()?;
        let before = Instant::now();
        let (advance, _) = decoder
            .prefill_batched(&mut state, &[1u32; 32])
            .map_err(|e| format!("{e}"))?;
        let _ = advance.logits();
        advance.commit()?;
        stages.push(json!({"stage":stage,"seconds":before.elapsed().as_secs_f64()}));
        compiled_kernels = Some(decoder.compiled_kernel_count());
        stage = "cold_decode";
        let before = Instant::now();
        let (advance, _) = decoder
            .prefill_batched(&mut state, &[2u32])
            .map_err(|e| format!("{e}"))?;
        let _ = advance.logits();
        advance.commit()?;
        stages.push(json!({"stage":stage,"seconds":before.elapsed().as_secs_f64()}));
        compiled_kernels = Some(decoder.compiled_kernel_count());
        Ok(())
    })();
    let elapsed = start.elapsed().as_secs_f64();
    let record = json!({
        "method": method, "artifact": args[1], "elapsed_seconds": elapsed,
        "status": if result.is_ok() { "cold_path_completed" } else { "failed" },
        "last_stage": stage, "completed_stages": stages,
        "error": result.err(), "compiled_kernels": compiled_kernels,
        "search_budget_seconds_per_request": 1,
        "hardware_profile": "existing hypothetical single-service decoder fixture",
        "hardware_quality_eligible": false,
        "timing_scope": "cold artifact open/import/construction/prefill/decode; includes execution if reached, excludes process/device/profile setup",
        "context_capacity":256, "prompt_tokens":32, "decode_tokens":1
    });
    std::fs::write(&args[3], serde_json::to_vec_pretty(&record)?)?;
    println!("{}", record);
    Ok(())
}

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("This experiment requires Metal on macOS.");
    std::process::exit(1);
}
