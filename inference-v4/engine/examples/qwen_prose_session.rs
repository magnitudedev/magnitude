//! Long-context session measurement with ordinary automatic selection: the session-bench
//! prose fixture prefilled in fixed chunks, then greedy decode. Only the search budget's
//! strategy and precision policy are supplied; no implementation flags exist.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use seismic_engine::models::qwen35::session::Session;
    use seismic_runtime::{plan::Settings, Device, Strategy};
    use std::{path::Path, rc::Rc};
    const USAGE: &str = "usage: qwen_prose_session ARTIFACT OUTPUT_JSON [--device metal|cpu|cuda] [--strategy exact|greedy] [--precision exact|unconstrained] [--context-tokens 16384] [--prefill-chunk 512] [--decode 256] [--fixture PATH]";
    let mut args = std::env::args().skip(1).collect::<Vec<_>>();
    let mut option = |name: &str| -> Result<Option<String>, String> {
        let Some(at) = args.iter().position(|a| a == name) else { return Ok(None) };
        let value = args.get(at + 1).cloned().ok_or(format!("{name} requires a value\n{USAGE}"))?;
        args.drain(at..at + 2);
        Ok(Some(value))
    };
    let target = option("--device")?.unwrap_or("metal".into());
    let strategy = match option("--strategy")?.as_deref() {
        None | Some("exact") => Strategy::Exact,
        Some("greedy") => Strategy::Greedy,
        Some(other) => return Err(format!("unknown strategy `{other}`").into()),
    };
    let precision = match option("--precision")?.as_deref() {
        None | Some("exact") => seismic_lang::precision::PrecisionPolicy::Exact,
        Some("unconstrained") => seismic_lang::precision::PrecisionPolicy::Unconstrained,
        Some(other) => return Err(format!("unknown precision `{other}`").into()),
    };
    let number = |value: Option<String>, default: usize| value.map_or(Ok(default), |v| v.parse::<usize>().map_err(|e| format!("`{v}`: {e}")));
    let context = number(option("--context-tokens")?, 16384)?;
    let chunk = number(option("--prefill-chunk")?, 512)?;
    let decode = number(option("--decode")?, 256)?;
    let fixture = match option("--fixture")? {
        Some(path) => path,
        None => format!("{}/.cache/magnitude/benchmarks/sources/9a6844ac0703853720010787c7b6c70b0020f1ab1862dcd74452fa46474d1215/moby-dick.txt", std::env::var("HOME")?),
    };
    let [artifact, output] = args.as_slice() else { return Err(USAGE.into()) };
    let device = Rc::new(Device::open(&target)?);
    let device_name = match device.facts() {
        #[cfg(target_os = "macos")]
        seismic_runtime::DeviceFacts::Metal(facts) => facts.name,
        seismic_runtime::DeviceFacts::Cpu(facts) => format!("host CPU, {} workers", facts.workers),
        seismic_runtime::DeviceFacts::Cuda(facts) => facts.name,
    };
    let settings = Settings { precision: precision.clone(), strategy, ..Settings::default() };
    let result = (|| {
        let text = std::fs::read(&fixture).map_err(|e| format!("{fixture}: {e}"))?;
        eprintln!("loading artifact and selecting numerical imports");
        let mut session = Session::load(Path::new(artifact), device, settings.clone(), context)?;
        eprintln!("starting cold chunked prefill");
        session.measure(&text, chunk, decode)
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
        "fixture": fixture,
        "budget": {"work": settings.budget.work, "seconds": settings.budget.time.map(|t| t.as_secs_f64())},
    });
    std::fs::write(output, serde_json::to_vec_pretty(&record)?)?;
    if let Ok(report) = &result {
        eprintln!(
            "solve {:.2} s, compile {:.2} s, first token {:.2} s, prefill warm {:.1} tok/s, decode warm {:.2} tok/s",
            report.solve_seconds_total, report.compile_seconds_total, report.cold_seconds_to_first_token, report.prefill_warm.tokens_per_second, report.decode_warm.tokens_per_second
        );
    }
    result.map(|_| ()).map_err(Into::into)
}
