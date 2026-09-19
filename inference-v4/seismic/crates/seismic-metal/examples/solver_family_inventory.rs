//! Read-only Gate 0 inventory. No selection, native emission or device execution.
//! The source-derived grouping family remains unresolved; earlier diagnostic
//! preparation choices are stated explicitly rather than called full coverage.
use seismic_accounting::{
    schedule::{self, structured::Node},
    workload::{Allocation, BufferBinding, DerivationLimits, ScalarWorkload},
};
use seismic_lang::{
    ir::LoadMode,
    program::{compile, SourceFile},
    Scope,
};
use seismic_metal::{
    execution::{self, Config},
    family::GroupFamily,
    model::{self, Hardware, Service, Timing, Units},
};
use seismic_realization::{dispatch::TilePlacement, LoadStrategy};
use std::sync::Arc;

fn lower(text: &str, entry: &str, n: i64) -> seismic_lang::lowered_ir::LoweredIr {
    let program = compile(
        &[SourceFile {
            path: "solver-inventory.seismic.portable".into(),
            scope: Scope::Portable,
            text: text.into(),
        }],
        &[],
    )
    .unwrap();
    seismic_lang::lower::lower(
        &program,
        entry,
        "metal",
        &std::collections::HashMap::from([("N".into(), n)]),
    )
    .unwrap()
}

fn grouping_inventory() {
    let source = lower("fn evaluate(x: tensor[5,64] f32, middle: tensor[5,64] f32, out: tensor[5] f32):\n  for row in parallel:\n    a = load(x[row])\n    y = tile[64] f32\n    for i in owned(y): y[i] = a[(i+1)%64]\n    store(y,middle[row])\n  for row in parallel:\n    y = tile[1] f32\n    for i in owned(y): y[i] = middle[row,0]\n    store(y,out[row:row+1])\n", "evaluate", 5);
    let config = Config {
        loads: LoadStrategy::Materialize,
        sg_per_tg: 1,
        max_threads_per_threadgroup: 96,
        max_threadgroup_bytes: 1024,
        ..Default::default()
    };
    let execution = execution::prepare_with_choices(
        &source,
        config,
        &mut |_, _| Ok(LoadMode::Materialize),
        &mut |_| Ok(TilePlacement::GroupShared),
        &mut |decision| Ok(decision.diagnostic()),
    )
    .unwrap();
    let family = Arc::new(GroupFamily::derive(execution).unwrap());
    println!("family=two-launch-shared-storage scope=unresolved-groupings earlier-choices=diagnostic-materialized-group-shared");
    for constraint in &family.constraints {
        println!(
            "  launch={} resource={} units_per_item={} capacity={} maximum_items={}",
            constraint.launch,
            constraint.resource,
            constraint.units_per_item,
            constraint.capacity,
            constraint.maximum_items
        );
    }
    let mut builder = magnitude_solver::model::ModelBuilder::new();
    let dispatch = family.append_dispatch(&mut builder, "inventory", &[]).unwrap();
    for (launch, geometry) in dispatch.launches.iter().enumerate() {
        println!("  launch={launch} grouping_domain={:?}", geometry.items_per_group.bounds());
    }

}

fn structure(node: &Node) -> (u64, u128, Vec<u64>) {
    match node {
        Node::Operation(_) => (1, 1, Vec::new()),
        Node::Compose { children, .. } => children.iter().fold(
            (1, 0, Vec::new()),
            |(nodes, logical, mut repeats), child| {
                let (n, l, r) = structure(child);
                repeats.extend(r);
                (nodes + n, logical + l, repeats)
            },
        ),
        Node::Repeat { count, body, .. } => {
            let (n, l, mut r) = structure(body);
            r.push(*count);
            (n + 1, l * *count as u128, r)
        }
        Node::Scope { body, .. } => {
            let (n, l, r) = structure(body);
            (n + 1, l, r)
        }
    }
}

fn repetition_inventory(count: i64) {
    let source = lower("fn write[N](out: tensor[N] f32):\n  for row in parallel:\n    y = tile[1] f32\n    for i in owned(y): y[i] = 3.0\n    store(y, out[row:row+1])\n", "write", count);
    let execution = execution::prepare(&source, Config::default()).unwrap();
    let hardware = Hardware {
        identity: "inventory hypothetical pooled machine, not device timing".into(),
        timebase: schedule::Timebase {
            seconds_numerator: 1,
            seconds_denominator: 1,
        },
        resources: vec![schedule::Resource {
            name: "service".into(),
            capacity: 1024,
            unit: schedule::CapacityUnit::Slots,
        }],
        resident_groups: 2,
        resident_shared_bytes: 65536,
        timings: model::requirements(&execution)
            .unwrap()
            .primitives
            .into_iter()
            .map(|primitive| Timing {
                primitive,
                latency: 1,
                services: vec![Service {
                    resource: 0,
                    offset: 0,
                    duration: 1,
                    units: Units::PerLane(1),
                }],
            })
            .collect(),
    };
    let bytes = count as u64 * 4;
    let workload = ScalarWorkload {
        identity: format!("inventory repetition N={count}"),
        integer_domains: Vec::new(),
        allocations: vec![Allocation {
            id: 1,
            bytes,
            alignment: 4,
            known_bytes: Default::default(),
        }],
        buffers: vec![BufferBinding {
            allocation: 1,
            offset: 0,
            bytes,
        }],
        scalars: Vec::new(),
    };
    let limits = DerivationLimits {
        instructions: 1000,
        operations: 1000,
    };
    let started = std::time::Instant::now();
    let structured = model::structured_execution(&execution, &hardware, &workload, limits).unwrap();
    let micros = started.elapsed().as_micros();
    let (nodes, logical, repetitions) = structure(&structured.root);
    println!("family=repeated-write N={count} retained_nodes={nodes} logical_operations={logical} repeat_counts={repetitions:?} construction_us={micros} unresolved={:?}", structured.unmapped);
    println!("  boundary=whole-group-residency+service-profile+launch-order; objective=conditional-model-completion; schedule-witness=analysis-only");
}

fn main() {
    grouping_inventory();
    for count in [8, 1024, 1_000_000_003] {
        repetition_inventory(count);
    }
}
