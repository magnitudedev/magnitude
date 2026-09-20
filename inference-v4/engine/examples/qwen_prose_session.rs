//! Long-context session measurement with ordinary automatic selection: the session-bench
//! prose fixture prefilled in fixed chunks, then greedy decode. Only the search budget and
//! precision policy are supplied; no implementation flags exist.

use seismic_lang::precision::{EvidenceRequirement, Limit, PrecisionPolicy, Tolerance};

const USAGE: &str = "usage: qwen_prose_session ARTIFACT OUTPUT_JSON [--device metal|cpu|cuda] --precision exact|bounded|unconstrained [--atol V --rtol V --relative-floor V --ulps N|- --evidence proven|qualified] [--context-tokens 16384] [--prefill-chunk 512] [--decode 256] [--fixture PATH]\n\
bounded precision requires all five bounded-policy options; `--ulps -` makes the numerical envelope authoritative";

struct Options {
    artifact: String,
    output: String,
    target: String,
    precision: PrecisionPolicy,
    context: usize,
    chunk: usize,
    decode: usize,
    fixture: String,
}

#[derive(Default)]
struct BoundedOptions {
    absolute: Option<Limit>,
    relative: Option<Limit>,
    relative_floor: Option<Limit>,
    ulps: Option<Option<u64>>,
    evidence: Option<EvidenceRequirement>,
}

fn parse_options(args: impl IntoIterator<Item = String>) -> Result<Options, String> {
    let mut args = args.into_iter();
    let mut positional = Vec::new();
    let mut target = None;
    let mut precision = None;
    let mut bounded = BoundedOptions::default();
    let mut context = None;
    let mut chunk = None;
    let mut decode = None;
    let mut fixture = None;

    fn set<T>(slot: &mut Option<T>, value: T, flag: &str) -> Result<(), String> {
        if slot.replace(value).is_some() {
            Err(format!("duplicate option `{flag}`"))
        } else {
            Ok(())
        }
    }
    fn value(args: &mut impl Iterator<Item = String>, flag: &str) -> Result<String, String> {
        args.next()
            .ok_or_else(|| format!("{flag} requires a value\n{USAGE}"))
    }
    fn limit(value: String, flag: &str) -> Result<Limit, String> {
        let parsed = value
            .parse::<f64>()
            .map_err(|_| format!("bad {flag} numerical limit `{value}`"))?;
        Limit::new(parsed)
    }
    fn count(value: String, flag: &str) -> Result<usize, String> {
        value
            .parse::<usize>()
            .map_err(|e| format!("bad {flag} value `{value}`: {e}"))
    }

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--device" => {
                let parsed = value(&mut args, &arg)?;
                if !matches!(parsed.as_str(), "metal" | "cpu" | "cuda") {
                    return Err(format!(
                        "bad --device `{parsed}`; expected metal, cpu or cuda"
                    ));
                }
                set(&mut target, parsed, &arg)?;
            }
            "--precision" => set(&mut precision, value(&mut args, &arg)?, &arg)?,
            "--atol" => {
                let parsed = limit(value(&mut args, &arg)?, &arg)?;
                set(&mut bounded.absolute, parsed, &arg)?;
            }
            "--rtol" => {
                let parsed = limit(value(&mut args, &arg)?, &arg)?;
                set(&mut bounded.relative, parsed, &arg)?;
            }
            "--relative-floor" => {
                let parsed = limit(value(&mut args, &arg)?, &arg)?;
                set(&mut bounded.relative_floor, parsed, &arg)?;
            }
            "--ulps" => {
                let text = value(&mut args, &arg)?;
                let parsed = if text == "-" {
                    None
                } else {
                    Some(
                        text.parse::<u64>()
                            .map_err(|_| format!("bad --ulps limit `{text}`"))?,
                    )
                };
                set(&mut bounded.ulps, parsed, &arg)?;
            }
            "--evidence" => {
                let parsed = match value(&mut args, &arg)?.as_str() {
                    "proven" => EvidenceRequirement::Proven,
                    "qualified" => EvidenceRequirement::Qualified,
                    other => {
                        return Err(format!(
                            "bad --evidence `{other}`; expected proven or qualified"
                        ))
                    }
                };
                set(&mut bounded.evidence, parsed, &arg)?;
            }
            "--context-tokens" => {
                let parsed = count(value(&mut args, &arg)?, &arg)?;
                set(&mut context, parsed, &arg)?;
            }
            "--prefill-chunk" => {
                let parsed = count(value(&mut args, &arg)?, &arg)?;
                set(&mut chunk, parsed, &arg)?;
            }
            "--decode" => {
                let parsed = count(value(&mut args, &arg)?, &arg)?;
                set(&mut decode, parsed, &arg)?;
            }
            "--fixture" => set(&mut fixture, value(&mut args, &arg)?, &arg)?,
            flag if flag.starts_with("--") => {
                return Err(format!("unknown option `{flag}`\n{USAGE}"))
            }
            _ => positional.push(arg),
        }
    }

    let precision_name = precision.ok_or_else(|| format!("--precision is required\n{USAGE}"))?;
    let has_bounded_options = bounded.absolute.is_some()
        || bounded.relative.is_some()
        || bounded.relative_floor.is_some()
        || bounded.ulps.is_some()
        || bounded.evidence.is_some();
    let precision = match precision_name.as_str() {
        "exact" | "unconstrained" if has_bounded_options => {
            return Err(format!(
                "bounded-policy options require `--precision bounded`, not `--precision {precision_name}`"
            ))
        }
        "exact" => PrecisionPolicy::Exact,
        "unconstrained" => PrecisionPolicy::Unconstrained,
        "bounded" => {
            let mut missing = Vec::new();
            if bounded.absolute.is_none() {
                missing.push("--atol");
            }
            if bounded.relative.is_none() {
                missing.push("--rtol");
            }
            if bounded.relative_floor.is_none() {
                missing.push("--relative-floor");
            }
            if bounded.ulps.is_none() {
                missing.push("--ulps");
            }
            if bounded.evidence.is_none() {
                missing.push("--evidence");
            }
            if !missing.is_empty() {
                return Err(format!(
                    "bounded precision requires explicit {}; use zero values when zero is intended",
                    missing.join(", ")
                ));
            }
            let mut policy = PrecisionPolicy::bounded(Tolerance {
                absolute: bounded.absolute.expect("checked above"),
                relative: bounded.relative.expect("checked above"),
                relative_floor: bounded.relative_floor.expect("checked above"),
                ulps: bounded.ulps.expect("checked above"),
            });
            let PrecisionPolicy::Bounded { evidence, .. } = &mut policy else {
                unreachable!()
            };
            *evidence = bounded.evidence.expect("checked above");
            policy
        }
        other => {
            return Err(format!(
                "bad --precision `{other}`; expected exact, bounded or unconstrained"
            ))
        }
    };

    let [artifact, output] = positional.as_slice() else {
        return Err(USAGE.into());
    };
    let fixture = match fixture {
        Some(fixture) => fixture,
        None => format!(
            "{}/.cache/magnitude/benchmarks/sources/9a6844ac0703853720010787c7b6c70b0020f1ab1862dcd74452fa46474d1215/moby-dick.txt",
            std::env::var("HOME").map_err(|e| format!("HOME is required for the default fixture: {e}"))?
        ),
    };
    Ok(Options {
        artifact: artifact.clone(),
        output: output.clone(),
        target: target.unwrap_or_else(|| "metal".into()),
        precision,
        context: context.unwrap_or(16384),
        chunk: chunk.unwrap_or(512),
        decode: decode.unwrap_or(256),
        fixture,
    })
}

fn tolerance_json(tolerance: &Tolerance) -> serde_json::Value {
    serde_json::json!({
        "absolute": tolerance.absolute.get(),
        "relative": tolerance.relative.get(),
        "relative_floor": tolerance.relative_floor.get(),
        "ulps": tolerance.ulps,
    })
}

fn precision_json(policy: &PrecisionPolicy) -> serde_json::Value {
    match policy {
        PrecisionPolicy::Exact => serde_json::json!({"mode": "exact"}),
        PrecisionPolicy::Unconstrained => serde_json::json!({"mode": "unconstrained"}),
        PrecisionPolicy::Bounded {
            default,
            outputs,
            evidence,
            specials,
            inputs,
        } => serde_json::json!({
            "mode": "bounded",
            "default": tolerance_json(default),
            "outputs": outputs.iter().map(|(name, tolerance)| (name.clone(), tolerance_json(tolerance))).collect::<serde_json::Map<_, _>>(),
            "evidence": match evidence {
                EvidenceRequirement::Proven => "proven",
                EvidenceRequirement::Qualified => "qualified",
            },
            "specials": {
                "nan": specials.nan,
                "infinity": specials.infinity,
                "signed_zero": specials.signed_zero,
                "subnormal": specials.subnormal,
            },
            "inputs": inputs.iter().map(|(name, range)| (name.clone(), serde_json::json!({
                "minimum": range.minimum.get(),
                "maximum": range.maximum.get(),
            }))).collect::<serde_json::Map<_, _>>(),
        }),
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    use seismic_engine::models::qwen35::session::Session;
    use seismic_runtime::{plan::Settings, Device};
    use std::{path::Path, rc::Rc};

    let options = parse_options(std::env::args().skip(1))?;
    let device = Rc::new(Device::open(&options.target)?);
    let device_name = match device.facts() {
        #[cfg(target_os = "macos")]
        seismic_runtime::DeviceFacts::Metal(facts) => facts.name,
        seismic_runtime::DeviceFacts::Cpu(facts) => {
            format!("host CPU, {} workers", facts.workers)
        }
        seismic_runtime::DeviceFacts::Cuda(facts) => facts.name,
    };
    let settings = Settings {
        precision: options.precision.clone(),
        ..Settings::default()
    };
    let result = (|| {
        let text =
            std::fs::read(&options.fixture).map_err(|e| format!("{}: {e}", options.fixture))?;
        eprintln!("loading artifact and selecting numerical imports");
        let mut session = Session::load(
            Path::new(&options.artifact),
            device,
            settings.clone(),
            options.context,
        )?;
        eprintln!("starting cold chunked prefill");
        session.measure(&text, options.chunk, options.decode)
    })();
    let record = match &result {
        Ok(report) => serde_json::json!({"status":"measured", "report": report}),
        Err(error) => serde_json::json!({"status":"not_measured", "error": error}),
    };
    let record = serde_json::json!({
        "result": record,
        "device": device_name,
        "backend": options.target,
        "precision": precision_json(&options.precision),
        "fixture": options.fixture,
        "session": {
            "context_tokens": options.context,
            "prefill_chunk": options.chunk,
            "decode": options.decode,
        },
        "budget": {"work": settings.budget.work, "seconds": settings.budget.time.map(|t| t.as_secs_f64())},
    });
    std::fs::write(&options.output, serde_json::to_vec_pretty(&record)?)?;
    if let Ok(report) = &result {
        eprintln!(
            "compile {:.2} s, first token {:.2} s, prefill warm {:.1} tok/s, decode warm {:.2} tok/s",
            report.compile_seconds_total, report.cold_seconds_to_first_token, report.prefill_warm.tokens_per_second, report.decode_warm.tokens_per_second
        );
    }
    result.map(|_| ()).map_err(Into::into)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(extra: &[&str]) -> Result<Options, String> {
        let mut args = vec!["artifact", "output"];
        args.extend_from_slice(extra);
        parse_options(args.into_iter().map(str::to_owned))
    }

    #[test]
    fn parses_each_policy_without_implicit_bounded_values() {
        assert_eq!(
            parse(&["--precision", "exact"]).unwrap().precision,
            PrecisionPolicy::Exact
        );
        assert_eq!(
            parse(&["--precision", "unconstrained"]).unwrap().precision,
            PrecisionPolicy::Unconstrained
        );
        let bounded = parse(&[
            "--precision",
            "bounded",
            "--atol",
            "0.000001",
            "--rtol",
            "0.0001",
            "--relative-floor",
            "0.00000001",
            "--ulps",
            "-",
            "--evidence",
            "proven",
        ])
        .unwrap();
        assert_eq!(
            bounded.precision,
            PrecisionPolicy::Bounded {
                default: Tolerance {
                    absolute: Limit::new(0.000001).unwrap(),
                    relative: Limit::new(0.0001).unwrap(),
                    relative_floor: Limit::new(0.00000001).unwrap(),
                    ulps: None,
                },
                outputs: Default::default(),
                evidence: EvidenceRequirement::Proven,
                specials: Default::default(),
                inputs: Default::default(),
            }
        );
    }

    #[test]
    fn rejects_partial_or_contradictory_bounded_options() {
        let incomplete = parse(&["--precision", "bounded", "--atol", "0.1"])
            .err()
            .unwrap();
        assert!(incomplete.contains("--rtol"));
        assert!(incomplete.contains("--relative-floor"));
        assert!(incomplete.contains("--ulps"));
        assert!(incomplete.contains("--evidence"));

        let contradictory = parse(&["--precision", "exact", "--atol", "0.1"])
            .err()
            .unwrap();
        assert!(contradictory.contains("require `--precision bounded`"));
    }

    #[test]
    fn serializes_the_complete_resolved_bounded_policy() {
        let options = parse(&[
            "--precision",
            "bounded",
            "--atol",
            "0",
            "--rtol",
            "0.001",
            "--relative-floor",
            "0.01",
            "--ulps",
            "4",
            "--evidence",
            "qualified",
        ])
        .unwrap();
        assert_eq!(
            precision_json(&options.precision),
            serde_json::json!({
                "mode": "bounded",
                "default": {"absolute": 0.0, "relative": 0.001, "relative_floor": 0.01, "ulps": 4},
                "outputs": {},
                "evidence": "qualified",
                "specials": {"nan": true, "infinity": true, "signed_zero": true, "subnormal": true},
                "inputs": {},
            })
        );
    }
}
