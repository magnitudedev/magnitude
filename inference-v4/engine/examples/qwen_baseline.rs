//! Full-model measurement with ordinary automatic selection. Only a search budget is
//! supplied; no physical implementation, tiling, or candidate flags are accepted.
//! The report records each entry's selection status and estimates as estimates.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use seismic_engine::models::qwen35::baseline::Baseline;
    use seismic_runtime::{plan::Settings, Device};
    use std::{path::Path, rc::Rc};
    let args = std::env::args().collect::<Vec<_>>();
    let mut args = args;
    // `--device cpu|cuda|metal` may appear anywhere; the positional arguments are unchanged.
    let target = match args.iter().position(|a| a == "--device") {
        None => "metal".to_string(),
        Some(at) => {
            let value = args.get(at + 1).cloned().ok_or("--device requires cpu, cuda or metal")?;
            args.drain(at..at + 2);
            value
        }
    };
    // Likewise `--strategy exact|greedy`: how selection improves the seed (default exact).
    let strategy = match args.iter().position(|a| a == "--strategy") {
        None => seismic_runtime::Strategy::Exact,
        Some(at) => {
            let strategy = match args.get(at + 1).map(String::as_str) {
                Some("exact") => seismic_runtime::Strategy::Exact,
                Some("greedy") => seismic_runtime::Strategy::Greedy,
                _ => return Err("--strategy requires exact or greedy".into()),
            };
            args.drain(at..at + 2);
            strategy
        }
    };
    if args.len() != 6 && args.len() != 7 {
        return Err("usage: qwen_baseline ARTIFACT CONTEXT PROMPT_IDS CONTINUATION_IDS OUTPUT_JSON [exact|unconstrained] [--device cpu|cuda|metal] [--strategy exact|greedy]".into());
    }
    let precision = match args.get(6).map(String::as_str) {
        None | Some("exact") => seismic_lang::precision::PrecisionPolicy::Exact,
        Some("unconstrained") => seismic_lang::precision::PrecisionPolicy::Unconstrained,
        Some(other) => return Err(format!("unknown precision `{other}`").into()),
    };
    let device = Rc::new(Device::open(&target)?);
    let device_name = match device.facts() {
        #[cfg(target_os = "macos")]
        seismic_runtime::DeviceFacts::Metal(facts) => facts.name,
        seismic_runtime::DeviceFacts::Cpu(facts) => format!("host CPU, {} workers", facts.workers),
        seismic_runtime::DeviceFacts::Cuda(facts) => facts.name,
    };
    let parse = |s: &str| s.split(',').map(str::parse::<u32>).collect::<Result<Vec<_>, _>>();
    let prompt = parse(&args[3])?;
    let continuation = parse(&args[4])?;
    let settings = Settings { precision: precision.clone(), strategy, ..Settings::default() };
    let result = (|| {
        eprintln!("loading artifact and selecting numerical imports");
        let mut baseline = Baseline::load(Path::new(&args[1]), device, settings.clone(), args[2].parse().map_err(|e| format!("context: {e}"))?)?;
        eprintln!("starting cold automatic full-model forward");
        baseline.measure(&prompt, &continuation)
    })();
    let record = match &result {
        Ok(report) => serde_json::json!({"status":"measured", "report": report}),
        Err(error) => serde_json::json!({"status":"not_measured", "error": error}),
    };
    let record = serde_json::json!({
        "result": record,
        "device": device_name,
        "backend": target,
        "precision": format!("{precision:?}"),
        "strategy": format!("{strategy:?}"),
        "budget": {"work": settings.budget.work, "seconds": settings.budget.time.map(|t| t.as_secs_f64())},
    });
    std::fs::write(&args[5], serde_json::to_vec_pretty(&record)?)?;
    result.map(|_| ()).map_err(Into::into)
}
