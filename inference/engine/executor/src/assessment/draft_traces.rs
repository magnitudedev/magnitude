//! Every served catalog target with a separate draft, under every draft
//! width policy a load may serve it with (adaptive, and each fixed budget
//! up to its proposals), on every backend, in every way its load holds its
//! experts (resident, and banked at its floor where they stream) and with
//! every KV codec it may be served with, outlines its draft graphs from the
//! family's graph traces (`seismic::trace`: recorded once per topology,
//! answering every later model its guards and shapes admit) exactly as the
//! load lays them out: the family's storage, census, binding constants and
//! host heap, and each class's storage and host heaps.

use super::catalog::{bank_reservations, served_drafted_targets, LIMITS};
use crate::programs::native_constants::CheckedGraphResources;
use crate::programs::native_draft::{
    certified_draft_family, draft_graph_classes, draft_traces_recorded, DraftGeometry, DraftGraphClass,
};
use crate::{
    resident_layout, ComponentSelection, DraftWidths, ExecutionPath, ModelLoadPlan, PlannedMethod,
    ResourceCapacity, ResourcePlanner,
};
use magnitude_state::{kv_codec_policy, KvCodec};
use seismic::{BackendName, NativeGraphLayout, NativeGraphOutline};
use std::collections::BTreeSet;
use std::time::{Duration, Instant};

/// The codecs a native load may serve: the native path refuses rotated
/// K4/V4 (`ExecutionPlan::backend_plan`).
const NATIVE_CODECS: [KvCodec; 2] = [KvCodec::Dense, KvCodec::AffineK8V4];

fn every_served_draft_outlines_its_graphs_from_traces(backend: BackendName) {
    let selection = ComponentSelection { head: true, vision: false };
    let (mut rows, mut failures) = (0, Vec::new());
    let (mut answered, mut recorded) = ((0u32, Duration::ZERO), (0u32, Duration::ZERO));
    let mut laid_out = (0u32, Duration::ZERO);
    let (mut methods, mut codecs) = (BTreeSet::new(), BTreeSet::new());
    for (name, definition, manifest) in served_drafted_targets() {
        // A separate draft only: a head drafts through its own families.
        let Some(bound) = definition.draft.as_ref() else {
            continue;
        };
        methods.insert(format!("{:?}", bound.method.variant()));
        let served_codecs = NATIVE_CODECS
            .into_iter()
            .filter(|&codec| kv_codec_policy(&definition.decoder, Some(codec)).is_ok())
            .collect::<Vec<_>>();
        codecs.extend(served_codecs.iter().map(|codec| format!("{codec:?}")));
        let proposals = u8::try_from(bound.max_proposals()).unwrap();
        let widths = std::iter::once(DraftWidths::Adaptive).chain((0..=proposals).map(DraftWidths::Fixed));
        for widths in widths {
            let method = PlannedMethod::DFlash { proposals, widths };
            let line = format!("{name} {widths:?} {}", backend.as_str());
            let layout = resident_layout(ExecutionPath::Native, backend);
            // A target that serves derives its drafted load on every backend.
            let load = match ModelLoadPlan::derive(&manifest, &definition, selection, layout, method, LIMITS) {
                Ok(load) => load,
                Err(error) => {
                    failures.push(format!("{line}: its drafted load: {error}"));
                    continue;
                }
            };
            let banked = bank_reservations(&definition, backend).ok().map(|substrate| {
                load.bank_floor(&definition, 1, LIMITS.flights(), substrate).and_then(|floor| {
                    load.clone().with_expert_bank(&definition, floor, 1, LIMITS.flights(), substrate)
                })
            });
            let loads = std::iter::once(("resident", Ok(load))).chain(banked.map(|banked| ("banked", banked)));
            for (form, load) in loads {
                let load = match load {
                    Ok(load) => load,
                    Err(error) => {
                        failures.push(format!("{line} {form}: its load: {error}"));
                        continue;
                    }
                };
                for &codec in &served_codecs {
                    let line = format!("{line} {form} {codec:?}");
                    let state = ResourcePlanner::state_plan(
                        &definition,
                        &load,
                        method,
                        codec,
                        LIMITS,
                        ResourceCapacity {
                            domain_bytes: 64 << 30,
                            tensor_operations: crate::TensorOperations::Absent,
                        },
                    )
                    .unwrap_or_else(|error| panic!("{line}: its state plans: {error}"));
                    let programs = load
                        .program_plan(&definition, codec)
                        .unwrap_or_else(|error| panic!("{line}: its programs plan: {error}"));
                    let plan =
                        programs.draft().unwrap_or_else(|| panic!("{line}: a separate draft plans a program"));
                    let geometry = DraftGeometry::new(&definition, state.head_state().unwrap())
                        .unwrap_or_else(|error| panic!("{line}: its draft geometry: {error}"));
                    let classes =
                        draft_graph_classes(LIMITS, method.draft_rows(), &method.block_widths(), bound)
                            .unwrap_or_else(|error| panic!("{line}: its draft classes: {error}"));
                    let traces = draft_traces_recorded();
                    let began = Instant::now();
                    let outlined =
                        certified_draft_family::<NativeGraphOutline>(backend, &load, plan, &geometry, &classes);
                    let elapsed = began.elapsed();
                    let tally = match draft_traces_recorded() == traces {
                        true => &mut answered,
                        false => &mut recorded,
                    };
                    *tally = (tally.0 + 1, tally.1 + elapsed);
                    let began = Instant::now();
                    let certified =
                        certified_draft_family::<NativeGraphLayout>(backend, &load, plan, &geometry, &classes);
                    laid_out = (laid_out.0 + 1, laid_out.1 + began.elapsed());
                    rows += 1;
                    match (outlined, certified) {
                        (Ok(outlined), Ok(certified)) => compare(&line, outlined, certified, &mut failures),
                        (outlined, certified) => failures.push(format!(
                            "{line}: outlined {:?}, laid out {:?}",
                            outlined.err(),
                            certified.err()
                        )),
                    }
                }
            }
        }
    }
    let mean = |(count, total): (u32, Duration)| total.checked_div(count.max(1)).unwrap_or_default();
    println!(
        "{rows} rows over draft methods {methods:?} and KV codecs {codecs:?}, adaptive and every fixed width; \
         {} traces recorded for {} families ({:?} each), {} families answered by traces ({:?} each); \
         the load's layouts {:?} each",
        draft_traces_recorded(),
        recorded.0,
        mean(recorded),
        answered.0,
        mean(answered),
        mean(laid_out)
    );
    assert!(rows > 0);
    assert!(answered.0 > 0, "no family was answered by another's traces");
    assert!(failures.is_empty(), "{} of {rows} rows differ:\n{}", failures.len(), failures.join("\n"));
}

#[test]
#[ignore = "the served catalog's headers: run on a rig host"]
fn every_served_draft_outlines_its_graphs_from_traces_on_cpu() {
    every_served_draft_outlines_its_graphs_from_traces(BackendName::Cpu);
}

#[test]
#[ignore = "the served catalog's headers: run on a rig host"]
fn every_served_draft_outlines_its_graphs_from_traces_on_metal() {
    every_served_draft_outlines_its_graphs_from_traces(BackendName::Metal);
}

#[test]
#[ignore = "the served catalog's headers: run on a rig host"]
fn every_served_draft_outlines_its_graphs_from_traces_on_cuda() {
    every_served_draft_outlines_its_graphs_from_traces(BackendName::Cuda);
}

#[test]
#[ignore = "the served catalog's headers: run on a rig host"]
fn every_served_draft_outlines_its_graphs_from_traces_on_vulkan() {
    every_served_draft_outlines_its_graphs_from_traces(BackendName::Vulkan);
}

/// The outlined family against the laid out one, field by field: the
/// family's resources, and each class's storage and host heaps.
fn compare<L: crate::programs::graph::draft::Certified, M: crate::programs::graph::draft::Certified>(
    line: &str,
    (outline, outlines): (CheckedGraphResources, std::collections::BTreeMap<DraftGraphClass, L>),
    (layout, layouts): (CheckedGraphResources, std::collections::BTreeMap<DraftGraphClass, M>),
    failures: &mut Vec<String>,
) {
    fn family(
        resources: &CheckedGraphResources,
    ) -> (&seismic::NativeGraphStorageBytes, &seismic::LaunchGraphs, u64, &seismic::HostHeap) {
        (&resources.storage, &resources.graphs, resources.binding_constant_bytes, &resources.host_heap)
    }
    if family(&outline) != family(&layout) {
        failures.push(format!("{line}: outlined family {:?}, laid out {:?}", family(&outline), family(&layout)));
    }
    if outlines.len() != layouts.len() {
        failures.push(format!("{line}: {} classes outlined, {} laid out", outlines.len(), layouts.len()));
    }
    for (class, laid) in &layouts {
        let Some(outlined) = outlines.get(class) else {
            failures.push(format!("{line}: class {class:?} is not outlined"));
            continue;
        };
        let outlined = (outlined.storage_bytes(), outlined.plan_heap(), outlined.bound_heap());
        let laid = (laid.storage_bytes(), laid.plan_heap(), laid.bound_heap());
        if outlined != laid {
            failures.push(format!("{line}: class {class:?} outlined {outlined:?}, laid out {laid:?}"));
        }
    }
}
