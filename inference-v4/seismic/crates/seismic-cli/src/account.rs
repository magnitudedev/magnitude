//! Inspect the same IR-derived accounts used by compiler development.

use crate::{load_program, options};
use seismic_lang::program::Program;
use std::collections::HashMap;

pub fn account(args: &[String]) -> Result<(), String> {
    let options = options(args)?;
    let program = load_program(&options)?;
    let name = options.function.as_deref().ok_or("--fn is required")?;
    print_work(&program, name, &options.shapes, &options.elements, options.analysis_steps)?;
    if matches!(options.target.as_str(), "cpu" | "cuda") {
        let lowered = seismic_lang::lower::lower_specialized(&program, name, &options.target, &options.shapes, &options.elements, &seismic_lang::lower::Options{piece:options.piece})?;
        let dispatch = if options.target == "cuda" {
            seismic_realization::Dispatch::ParallelRoot
        } else {
            seismic_realization::Dispatch::Sequential
        };
        let realization = seismic_compiler::scalar_candidate(
            &lowered,
            seismic_realization::CallConv::SystemV,
            seismic_realization::ScalarOptions {
                dispatch,
                loads: options.loads,
            },
        )?;
        print_scalar(&seismic_accounting::realization::scalar(&realization));
    }
    Ok(())
}

pub fn print_work(
    program: &Program,
    name: &str,
    shapes: &HashMap<String, i64>,
    elements: &HashMap<String,seismic_lang::types::Elem>,
    analysis_steps: usize,
) -> Result<(), String> {
    let derived =
        seismic_accounting::derive_specialized(program, name, shapes, elements, &Default::default(), analysis_steps)?;
    let account = derived.work;
    println!(
        "portable-algorithm work: {name} ({})",
        if account.is_exact() {
            "exact operation counts"
        } else {
            "conditional or incomplete"
        }
    );
    println!("whole-axis reference streaming; semantic operations, not issued instructions or necessary lower-bound work");
    for term in &account.terms {
        println!(
            "  {}:{} {:?}: {:?}{}",
            term.function,
            term.byte_offset,
            term.kind,
            term.count,
            if term.conditions.is_empty() {
                String::new()
            } else {
                format!(" when {}", term.conditions.join(" and "))
            }
        );
    }
    for reason in &account.unavailable {
        println!("  unavailable: {reason}");
    }
    let memory = derived.memory;
    println!(
        "diagnostic access regions: {} ({} analysis steps; budget {})",
        if memory.is_exact() {
            "exact union"
        } else {
            "incomplete coverage"
        },
        memory.analysis_steps,
        analysis_steps
    );
    println!("  alias assumption: unbound parameters/planes use separate named backings; this is not an invocation binding or necessary-traffic proof");
    for (backing, access) in memory.accesses.iter() {
        println!(
            "  {backing}: {} bytes read, {} bytes written",
            access.reads.bytes(),
            access.writes.bytes()
        );
    }
    for reason in &memory.unavailable {
        println!("  unavailable: {reason}");
    }
    println!("physical-path traffic, realization cost, and roofline: unavailable (residency/profile/realization not yet bound)");
    Ok(())
}

pub fn print_scalar(account: &seismic_accounting::realization::ScalarAccount) {
    println!("scalar SSA realization: {} invocations, {} private scratch bytes/invocation, {:?} bytes/dispatch",account.invocation_count,account.scratch_bytes_per_invocation,account.scratch_bytes_per_dispatch);
    for assumption in &account.assumptions {
        println!("  scope: {assumption}");
    }
    for (object, traffic) in &account.traffic {
        println!(
            "  {object:?}: {:?} requested read bytes, {:?} requested write bytes",
            traffic.reads, traffic.writes
        );
    }
    let mut operations =
        std::collections::BTreeMap::<String, seismic_accounting::quantity::Count>::new();
    for term in &account.instructions {
        let name = format!(
            "{} ({}) -> ({})",
            term.primitive
                .map(|p| format!("{p:?}"))
                .unwrap_or_else(|| term.opcode.clone()),
            term.operand_types.join(","),
            term.result_types.join(",")
        );
        let count = operations
            .entry(name)
            .or_insert(seismic_accounting::quantity::Count::Exact(0));
        *count = count.add(&term.count);
    }
    for (name, count) in operations {
        println!("  {name}: {count:?} instances");
    }
    for reason in &account.unavailable {
        println!("  unavailable: {reason}");
    }
    println!("native instruction mapping, physical transactions, resource prediction and selection: unavailable (backend/profile evidence not yet bound)");
}
