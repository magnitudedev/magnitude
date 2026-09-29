//! A8's scenarios on Metal under shader validation (design A8 §2.5, G-A8-new-13 (b)).
//!
//! An out-of-bounds device load on Metal does not fault: it returns garbage,
//! so a differential check alone cannot see a missing edge-tile predicate or
//! a hoisted access. Shader validation reports every such access in the
//! command buffer's logs, and the Metal submission panics on a log entry.
//! Metal reads `MTL_SHADER_VALIDATION` at the process's first Metal call, so
//! this binary sets it once before any scenario opens a device. It is a
//! separate test binary because the variable applies to the whole process.
#![cfg(target_os = "macos")]

mod common;

use common::{corpus_path, run_public, Selection};
use seismic::BackendName;
use seismic_corpus::scenario::load_all;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::Once;

static VALIDATION: Once = Once::new();

#[test]
fn a8_scenarios_under_shader_validation() {
    VALIDATION.call_once(|| std::env::set_var("MTL_SHADER_VALIDATION", "1"));
    let scenarios: Vec<String> = load_all(&corpus_path("scenarios"))
        .into_iter()
        .filter(|scenario| {
            scenario.name.as_str().starts_with("areas/A8/")
                && scenario.backends.contains(BackendName::Metal)
        })
        .map(|scenario| scenario.name.as_str().to_owned())
        .collect();
    assert!(!scenarios.is_empty(), "scenarios/areas/A8 holds no Metal scenario");
    let mut failures = Vec::new();
    for name in &scenarios {
        for (selection, label) in [(Selection::Baseline, "baseline"), (Selection::Ranked, "ranked")] {
            let run = catch_unwind(AssertUnwindSafe(|| run_public(name, BackendName::Metal, selection)));
            if let Err(panic) = run {
                let message = panic
                    .downcast_ref::<String>()
                    .map(String::as_str)
                    .or_else(|| panic.downcast_ref::<&str>().copied())
                    .unwrap_or("non-string panic payload");
                failures.push(format!("{name} ({label}): {message}"));
            }
        }
    }
    assert!(
        failures.is_empty(),
        "{} of {} scenario run(s) failed under shader validation:\n{}",
        failures.len(),
        2 * scenarios.len(),
        failures.join("\n")
    );
}
