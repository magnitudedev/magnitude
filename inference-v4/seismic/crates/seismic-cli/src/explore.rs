//! Inspect the admitted expansion/materialization space and actual emitted forms.
use crate::{load_program, options};
use seismic_lang::lower::{
    alternatives::{Space, Specialization},
    Options,
};
use seismic_realization::{CallConv, Dispatch, ScalarOptions};
use serde_json::json;
use sha2::{Digest, Sha256};

pub fn explore(args: &[String]) -> Result<(), String> {
    let mut filtered = Vec::new();
    let mut budget = None;
    let mut i = 0;
    while i < args.len() {
        if args[i] == "--candidate-budget" {
            if budget.is_some() {
                return Err("duplicate --candidate-budget".into());
            }
            i += 1;
            budget = Some(
                args.get(i)
                    .ok_or("--candidate-budget requires a positive integer")?
                    .parse::<usize>()
                    .map_err(|_| "invalid --candidate-budget")?,
            );
        } else {
            filtered.push(args[i].clone());
        }
        i += 1;
    }
    let budget = budget
        .filter(|n| *n > 0)
        .ok_or("explore requires an explicit positive --candidate-budget")?;
    let options = options(&filtered)?;
    let program = load_program(&options)?;
    let entry = options.function.as_deref().ok_or("--fn is required")?;
    let lowering = Options {
        piece: options.piece,
    };
    let mut space = Space::new(Specialization {
        program: &program,
        entry,
        backend: &options.target,
        shapes: &options.shapes,
        elements: &options.elements,
        options: &lowering,
    });
    let mut attempted = 0;
    let mut emitted = 0;
    for attempt in space.by_ref().take(budget) {
        attempted += 1;
        let decisions = attempt.steps.iter().map(|s| json!({
            "site": format!("{:?}", s.domain.kind),
            "domain": s.domain.alternatives.iter().map(|a| format!("{a:?}")).collect::<Vec<_>>(),
            "selected": format!("{:?}", s.selected),
        })).collect::<Vec<_>>();
        let realization = attempt.result.and_then(|lowered| {
            match options.target.as_str() {
                "cpu" | "cuda" => {
                    let config = ScalarOptions {
                        dispatch: if options.target == "cpu" { Dispatch::Sequential } else { Dispatch::ParallelRoot },
                        loads: options.loads,
                    };
                    let sequence = seismic_compiler::scalar_sequence(&lowered, CallConv::SystemV, config)?;
                    let phases = sequence.phases.iter().map(|phase| {
                        let account = seismic_accounting::realization::scalar(&phase.program);
                        let graph = seismic_realization::graph::Graph::scalar(&phase.program);
                        let ir = phase.program.function.display().to_string();
                        json!({
                            "source_statement": phase.source_statement,
                            "scalar_ir_sha256": digest(ir.as_bytes()),
                            "invocations": account.invocation_count,
                            "private_scratch_bytes_per_invocation": account.scratch_bytes_per_invocation,
                            "instructions": account.instructions.iter().map(|i|json!({"opcode":i.opcode,"count":format!("{:?}",i.count)})).collect::<Vec<_>>(),
                            "requested_traffic": account.traffic.iter().map(|(object,traffic)|json!({"object":format!("{object:?}"),"reads":format!("{:?}",traffic.reads),"writes":format!("{:?}",traffic.writes)})).collect::<Vec<_>>(),
                            "data_dependencies": graph.instructions.iter().map(|i|json!({
                                "instruction":i.id.to_string(),
                                "inputs":i.inputs.iter().map(|v|json!({"value":v.value.to_string(),"definition":format!("{:?}",v.definition),"type":v.ty.to_string()})).collect::<Vec<_>>(),
                                "outputs":i.outputs.iter().map(|(v,t)|json!({"value":v.to_string(),"type":t.to_string()})).collect::<Vec<_>>(),
                                "encoding":format!("{:?}",i.encoding),
                            })).collect::<Vec<_>>(),
                            "control_edges":graph.edges.iter().map(|e|json!({"from":e.from.to_string(),"to":e.destination.to_string(),"arguments":e.arguments.iter().map(|(a,b)|json!([a.to_string(),b.to_string()])).collect::<Vec<_>>()})).collect::<Vec<_>>(),
                            "memory_order":graph.memory_order.iter().map(|(a,b)|json!([a.to_string(),b.to_string()])).collect::<Vec<_>>(),
                            "unavailable": account.unavailable,
                        })
                    }).collect::<Vec<_>>();
                    Ok(json!({"form":"scalar_ssa_before_native_optimization","phases":phases}))
                }
                "metal" => metal(&lowered, &options),
                other => Err(format!("unsupported exploration backend `{other}`")),
            }
        });
        let record = match realization {
            Ok(realization) => {
                emitted += 1;
                json!({"kind":"candidate","index":attempted-1,"decisions":decisions,"realization":realization})
            }
            Err(error) => {
                json!({"kind":"candidate","index":attempted-1,"decisions":decisions,"error":error})
            }
        };
        println!("{record}");
    }
    println!(
        "{}",
        json!({
            "kind":"coverage", "scope":"admitted construct expansions and proven producer materializations at the supplied stream capacity",
            "attempted":attempted,"emitted":emitted,"exhausted":space.exhausted(),
            "candidate_budget":budget,"failed_attempts":attempted-emitted,
            "optimality_certified":false,
            "uncovered":["other stream capacities","placement and parallel mappings","native scheduling and code generation alternatives","physical execution cost and optimality"],
        })
    );
    Ok(())
}

#[cfg(target_os = "macos")]
fn metal(
    lowered: &seismic_lang::lowered_ir::LoweredIr,
    options: &crate::Options,
) -> Result<serde_json::Value, String> {
    let device = seismic_metal::runtime::Device::open()?;
    let facts = device.info();
    let emitted = seismic_metal::msl::emit_with(
        lowered,
        seismic_metal::execution::Config {
            loads: options.loads,
            sg_per_tg: options.sg_per_tg,
            piece: options.piece,
            per_item: options.per_item,
            split: options.split,
            tile_piece: None,
            max_threads_per_threadgroup: facts
                .max_threads_per_threadgroup
                .try_into()
                .map_err(|_| "thread limit overflow")?,
            max_threadgroup_bytes: facts
                .max_threadgroup_bytes
                .try_into()
                .map_err(|_| "storage limit overflow")?,
        },
    )?;
    Ok(json!({"form":"emitted_msl_before_native_optimization",
        "source_sha256":digest(emitted.source.as_bytes()),
        "launches":emitted.launches.iter().map(|l|json!({"kernel":l.kernel,"dispatch":format!("{:?}",l.dispatch),"declared_tiles":format!("{:?}",l.tiles)})).collect::<Vec<_>>(),
        "physical_account":"unavailable",
    }))
}
#[cfg(not(target_os = "macos"))]
fn metal(
    _: &seismic_lang::lowered_ir::LoweredIr,
    _: &crate::Options,
) -> Result<serde_json::Value, String> {
    Err("Metal exploration needs macOS device capabilities".into())
}

fn digest(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}
