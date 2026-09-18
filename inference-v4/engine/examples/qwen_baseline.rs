//! Full-model measurement with ordinary automatic selection and explicit hardware
//! inputs. No physical implementation, tiling, or candidate flags are accepted.
#[cfg(target_os = "macos")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use seismic_accounting::{selection::Budget, workload::DerivationLimits};
    use seismic_engine::models::qwen35::baseline::Baseline;
    use seismic_runtime::{Device, plan::Settings, tuner::{Form, Hardware}};
    use std::{path::Path, rc::Rc};
    let args = std::env::args().collect::<Vec<_>>();
    if args.len() != 7 {
        return Err("usage: qwen_baseline ARTIFACT METAL_HARDWARE_JSON CONTEXT PROMPT_IDS CONTINUATION_IDS OUTPUT_JSON".into());
    }
    #[derive(serde::Deserialize)]
    #[serde(deny_unknown_fields)]
    struct Profile {
        device_name: String,
        evidence: Vec<String>,
        assumptions: Vec<String>,
        hardware: seismic_metal::model::Hardware,
    }
    let profile_bytes = std::fs::read(&args[2])?;
    let profile: Profile = serde_json::from_slice(&profile_bytes)?;
    profile.hardware.validate()?;
    if profile.evidence.is_empty() || profile.assumptions.is_empty() {
        return Err("hardware input requires identified evidence and explicit modeling assumptions".into());
    }
    let device = Rc::new(Device::metal()?);
    let seismic_runtime::DeviceFacts::Metal(facts) = device.facts() else { unreachable!() };
    if facts.name != profile.device_name { return Err("hardware profile does not identify this device".into()); }
    let parse = |s: &str| s.split(',').map(str::parse::<u32>).collect::<Result<Vec<_>, _>>();
    let prompt = parse(&args[4])?;
    let continuation = parse(&args[5])?;
    let settings = Settings {
        hardware: Hardware::Metal(profile.hardware), form: Form::Metal,
        derivation_limits: DerivationLimits { instructions: 10_000_000, operations: 1_000_000 },
        search: Budget { nodes: 100_000, schedule_assignments: 1_000_000 },
    };
    let result = (|| {
        eprintln!("loading artifact and selecting numerical imports");
        let mut baseline = Baseline::load(Path::new(&args[1]), device, settings, args[3].parse().map_err(|e| format!("context: {e}"))?)?;
        eprintln!("starting cold automatic full-model forward");
        baseline.measure(&prompt, &continuation)
    })();
    use sha2::{Digest, Sha256};
    let record = match &result {
        Ok(report) => serde_json::json!({"status":"measured", "report": report}),
        Err(error) => serde_json::json!({"status":"not_measured", "error": error}),
    };
    let profile_sha256 = Sha256::digest(&profile_bytes).iter().map(|byte| format!("{byte:02x}")).collect::<String>();
    let record = serde_json::json!({"result": record, "hardware_profile_sha256": profile_sha256, "evidence": profile.evidence, "assumptions": profile.assumptions});
    std::fs::write(&args[6], serde_json::to_vec_pretty(&record)?)?;
    result.map(|_| ()).map_err(Into::into)
}
#[cfg(not(target_os = "macos"))]
fn main() { eprintln!("This baseline entry point requires Metal on macOS."); std::process::exit(1); }
