//! Structured realization inspection, using the device's queried resource limits.
#[cfg(target_os = "macos")]
pub fn inspect(args: &[String]) -> Result<(), String> {
    use crate::{load_program, options};
    use seismic_lang::lower::{lower_specialized, Options};
    use seismic_metal::{family::GroupFamily, msl, runtime::Device};
    use serde_json::json;
    use sha2::{Digest, Sha256};
    let options = options(args)?;
    if options.target != "metal" {
        return Err("group-family inspection currently requires Metal".into());
    }
    let device = Device::open()?;
    let facts = device.info();
    let program = load_program(&options)?;
    let name = options.function.as_deref().ok_or("--fn is required")?;
    let start = std::time::Instant::now();
    let lowered = lower_specialized(
        &program,
        name,
        "metal",
        &options.shapes,
        &options.elements,
        &Options {
            piece: options.piece,
        },
    )?;
    let config = seismic_metal::execution::Config {
            loads: options.loads,
        sg_per_tg: options.sg_per_tg,
        piece: options.piece,
        per_item: options.per_item,
        split: options.split,
        tile_piece: None,
        max_threads_per_threadgroup: i64::try_from(facts.max_threads_per_threadgroup)
            .map_err(|_| "thread limit overflow")?,
        max_threadgroup_bytes: i64::try_from(facts.max_threadgroup_bytes)
            .map_err(|_| "storage limit overflow")?,
    };
    let family = GroupFamily::derive(seismic_metal::execution::prepare(&lowered, seismic_metal::execution::Config { sg_per_tg: 1, ..config })?)?;
    let constraints=family.constraints.iter().map(|c|json!({"launch":c.launch,"resource":c.resource,"units_per_item":c.units_per_item,"capacity":c.capacity,"maximum_items_per_group":c.maximum_items})).collect::<Vec<_>>();
    let groups=family.groupings().map(|g|json!({"items_per_group":g.items_per_group,"launches":g.launches.iter().zip(&g.shared_bytes_per_group).map(|(d,shared)|json!({"work_items":d.work_items,"groups":d.groups,"threads_per_group":d.threads_per_group,"padding_lanes":d.padding_lanes(),"declared_threadgroup_bytes":shared})).collect::<Vec<_>>()})).collect::<Vec<_>>();
    let tiles=family.execution().memory().launches().iter().enumerate().map(|(index,l)|json!({"launch":index,"tiles":l.arrays.iter().map(|a| { let t=&a.declaration; json!({"symbol":t.symbol,"dtype":t.dtype.name(),"element_capacity":t.capacity,"placement":format!("{:?}",t.placement)}) }).collect::<Vec<_>>()})).collect::<Vec<_>>();
    let scratch = family.execution().memory().scratch().iter().map(|s| json!({"buffer":s.index,"phase":s.phase,"variable":s.variable,"dtype":s.dtype.name(),"elements_per_item":s.elements_per_item,"work_items":s.work_items,"parts":s.parts,"bytes":s.bytes,"producer":s.producer,"consumer":s.consumer})).collect::<Vec<_>>();
    let launch_dependencies = family.execution().memory().launches().iter().enumerate().map(|(index,l)| json!({"launch":index,"predecessor":l.predecessor})).collect::<Vec<_>>();
    let dispatches = family.execution().phases().iter().flat_map(|p| std::iter::once(p.dispatch.clone()).chain(p.merge_dispatch.clone())).collect::<Vec<_>>();
    let storage_account = seismic_accounting::storage::derive(family.execution().memory(), &dispatches)?;
    let count_json = |count: &seismic_accounting::quantity::Count| match count.bounds() {
        Some((lo,hi)) => json!({"lower":lo,"upper":hi}),
        None => json!({"unknown":format!("{count:?}")}),
    };
    let storage_account = json!({"items_per_group":1,"assumptions":storage_account.assumptions,"retained_scratch_bytes":count_json(&storage_account.retained_scratch_bytes),
        "peak_required_scratch_bytes_for_serial_launches":count_json(&storage_account.peak_required_scratch_bytes_for_serial_launches),
        "launches":storage_account.launches.iter().map(|l| json!({"declared_private_array_bytes_per_lane":count_json(&l.declared_private_array_bytes_per_lane),
            "declared_shared_array_bytes_per_group":count_json(&l.declared_shared_array_bytes_per_group),"static_barrier_sites":count_json(&l.static_barrier_sites),
            "barrier_executions":count_json(&l.barrier_executions),"native_private_bytes_per_lane":count_json(&l.native_private_bytes_per_lane),
            "unmodeled_fragment_operations":l.unmodeled_fragment_operations.iter().map(|id| id.0).collect::<Vec<_>>()})).collect::<Vec<_>>()});
    let analysis_seconds = start.elapsed().as_secs_f64();
    let compile_start = std::time::Instant::now();
    let selected = u64::try_from(options.sg_per_tg)
        .map_err(|_| "invalid requested group count".to_string())
        .and_then(|groups| family.select(groups))
        .and_then(|execution| msl::emit_execution(&execution));
    let (selected_hash, native) = match selected {
        Ok(emitted) => {
            let hash = Sha256::digest(emitted.source.as_bytes()).iter().map(|b| format!("{b:02x}")).collect::<String>();
            (Some(hash), device.compile(emitted))
        }
        Err(error) => (None, Err(error)),
    };
    let (native_facts,error)=match native {Ok(p)=>(Some(p.facts.iter().map(|f|json!({"kernel":f.kernel,"execution_width":f.execution_width,"max_threads_per_group":f.max_threads_per_group,"static_threadgroup_bytes":f.static_threadgroup_bytes})).collect::<Vec<_>>()),None),Err(error)=>(None,Some(error))};
    println!(
        "{}",
        json!({"kind":"metal_group_family","selected_msl_sha256":selected_hash,"kernel":name,"shapes":options.shapes,"elements":options.elements.iter().map(|(n,e)|(n,e.to_string())).collect::<std::collections::BTreeMap<_,_>>(),"device":format!("{facts:?}"),"stream_piece":options.piece,"per_item":options.per_item,"constraints":constraints,"groupings":groups,"declared_tiles":tiles,"scratch":scratch,"storage_account":storage_account,"launch_dependencies":launch_dependencies,"requested_items_per_group":options.sg_per_tg,"native_pipeline_facts":native_facts,"candidate_error":error,"analysis_seconds":analysis_seconds,"native_compile_seconds":compile_start.elapsed().as_secs_f64(),"performance_prediction":null,"limitations":["group order is deterministic enumeration, not a performance preference","declared private arrays are not registers or occupancy","family analysis and selection use only IR; native compilation separately qualifies the requested grouping","no GPU execution or measurement performed"]})
    );
    Ok(())
}
#[cfg(not(target_os = "macos"))]
pub fn inspect(_: &[String]) -> Result<(), String> {
    Err("Metal grouping inspection requires macOS device queries".into())
}
