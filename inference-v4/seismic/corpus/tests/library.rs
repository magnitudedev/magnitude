//! The standard library and `engine/model-kernels/kernels` entries through the public API
//! (design A10 §2.8; `engine.invocations` is A9's Tier 0).
use crate::common::{
    check_call, corpus_path, device, element_named, policy_name, prepare, Selection,
};
use seismic::dynamic::Module;
use seismic::BackendName;
use seismic_corpus::library::{self, LibraryFile};
use seismic_corpus::scenario::PolicySet;
use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

fn std_module() -> Module {
    Module::source("", "std-entries.seismic", true, None).expect("std loads")
}

fn engine_module() -> Module {
    let directory = corpus_path("../../engine/model-kernels/kernels");
    let mut paths: Vec<PathBuf> = std::fs::read_dir(&directory)
        .unwrap_or_else(|e| panic!("{}: {e}", directory.display()))
        .map(|entry| entry.expect("engine kernels entry").path())
        .filter(|path| {
            path.extension()
                .is_some_and(|extension| extension == "seismic")
        })
        .collect();
    paths.sort();
    Module::load(&paths, true).unwrap_or_else(|e| panic!("engine kernels load: {e}"))
}

fn names(module: &Module) -> BTreeSet<String> {
    module
        .functions()
        .iter()
        .map(|f| f.name().to_owned())
        .collect()
}

#[test]
fn every_library_entry_has_an_invocation() {
    let std_entries = names(&std_module());
    let engine_entries: BTreeSet<String> = names(&engine_module())
        .difference(&std_entries)
        .cloned()
        .collect();
    let mut missing = Vec::new();
    for (file, entries) in [
        ("library/std.invocations", &std_entries),
        ("library/engine.invocations", &engine_entries),
    ] {
        let invoked: BTreeSet<String> = library::load(&corpus_path(file))
            .invocations
            .into_iter()
            .map(|invocation| invocation.entry)
            .collect();
        missing.extend(
            entries
                .difference(&invoked)
                .map(|entry| format!("{file}: {entry}")),
        );
    }
    assert!(
        missing.is_empty(),
        "{} entr(ies) without an invocation line:\n{}",
        missing.len(),
        missing.join("\n")
    );
}

/// Every line under `exact` and `unconstrained`, baseline selection; all failures reported at once.
fn run_library(module: &Module, file: &str, backend: BackendName) {
    let library: LibraryFile = library::load(&corpus_path(file));
    let device = device(backend);
    let mut failures = Vec::new();
    for invocation in &library.invocations {
        let context = format!("{file}:{} {}", invocation.line, invocation.entry);
        let function = match module.function(&invocation.entry) {
            Ok(function) => function,
            Err(e) => {
                failures.push(format!("{context}: {e}"));
                continue;
            }
        };
        let elements: BTreeMap<String, _> = invocation
            .elements
            .iter()
            .map(|(name, element)| (name.clone(), element_named(element)))
            .collect();
        for policy in PolicySet::BOTH.policies() {
            let context = format!("{context} {}", policy_name(&policy));
            match prepare(
                &function,
                &device,
                elements.clone(),
                &policy,
                &Selection::Baseline,
            ) {
                Ok((kernel, _)) => {
                    if let Err(e) =
                        check_call(&kernel, &device, &invocation.arguments, None, &policy)
                    {
                        failures.push(format!("{context}: {e}"));
                    }
                }
                Err(e) => failures.push(format!("{context}: preparation: {e}")),
            }
        }
    }
    assert!(
        failures.is_empty(),
        "{} failure(s):\n{}",
        failures.len(),
        failures.join("\n")
    );
}

#[test]
fn std_entries_cpu() {
    run_library(&std_module(), "library/std.invocations", BackendName::Cpu);
}

#[cfg(target_os = "macos")]
#[test]
fn std_entries_metal() {
    run_library(&std_module(), "library/std.invocations", BackendName::Metal);
}

#[test]
fn engine_entries_cpu() {
    run_library(
        &engine_module(),
        "library/engine.invocations",
        BackendName::Cpu,
    );
}

#[cfg(target_os = "macos")]
#[test]
fn engine_entries_metal() {
    run_library(
        &engine_module(),
        "library/engine.invocations",
        BackendName::Metal,
    );
}
