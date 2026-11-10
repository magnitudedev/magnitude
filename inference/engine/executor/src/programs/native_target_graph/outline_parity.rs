//! The target family's outline (what assessment charges, from the family's
//! graph traces) against the layouts a load of the same plan seals, field by
//! field.

use super::certify::{TargetFamilyCertificate, certify_target_and_undrafted};
use super::classes::Topology;
use super::entry::EntryTokens;
use crate::{
    ModelLoadPlan, PassLaunches, ResourceLimits, StateResourcePlan, TargetProgramPlan,
    programs::graph::draft::Certified,
};
use magnitude_family_contracts::Decoder;
use seismic::{BackendName, HostHeap, NativeGraphLayout, NativeGraphOutline, NativeGraphStorageBytes};
use std::collections::BTreeMap;
use std::fmt::Debug;
use std::time::{Duration, Instant};

/// Where the family of `plan` and its undrafted family (`certify_target_and_undrafted`)
/// outlined differ from the same families laid out: one line per field
/// that differs, none where every field is equal. Either certification
/// failing is a difference unless both fail alike. With the time the
/// outline took, and the layouts.
#[allow(clippy::too_many_arguments)]
pub(crate) fn target_outline_differences(
    backend: BackendName,
    load: &ModelLoadPlan,
    geometry: &Decoder,
    state: &StateResourcePlan,
    plan: &TargetProgramPlan,
    limits: ResourceLimits,
    pass: PassLaunches,
) -> (Vec<String>, [Duration; 2], usize) {
    let began = Instant::now();
    let outlined = certify_target_and_undrafted::<NativeGraphOutline>(backend, load, geometry, state, plan, limits, pass);
    let outlining = began.elapsed();
    let began = Instant::now();
    let laid = certify_target_and_undrafted::<NativeGraphLayout>(backend, load, geometry, state, plan, limits, pass);
    let laying = began.elapsed();
    let graphs = outlined.as_ref().map_or(0, |outlined| outlined_graphs(&outlined[..]));
    let differences = match (outlined, laid) {
        (Ok(outlined), Ok(laid)) => ["drafted", "undrafted"]
            .into_iter()
            .zip(outlined.into_iter().zip(laid))
            .flat_map(|(family, (outlined, laid))| {
                let mut differences = Vec::new();
                compare(&mut differences, &outlined, &laid, limits);
                differences.into_iter().map(move |line| format!("{family}: {line}"))
            })
            .collect(),
        (Err(outlined), Err(laid)) if outlined == laid => Vec::new(),
        (outlined, laid) => vec![format!(
            "outlined {:?}, laid out {:?}",
            outlined.err(),
            laid.err()
        )],
    };
    (differences, [outlining, laying], graphs)
}

/// What a certified graph charges: its storage and its plan and binding heaps.
fn charge<L: Certified>(layout: &L) -> (NativeGraphStorageBytes, HostHeap, HostHeap) {
    (layout.storage_bytes(), layout.plan_heap(), layout.bound_heap())
}

fn field<T: PartialEq + Debug>(differences: &mut Vec<String>, name: &str, outlined: T, laid: T) {
    if outlined != laid {
        differences.push(format!("{name}: outlined {outlined:?}, laid out {laid:?}"));
    }
}

fn blocks<L: Certified>(
    blocks: &[BTreeMap<Topology, super::certify::CertifiedBlockGraph<L>>],
) -> Vec<BTreeMap<Topology, ((NativeGraphStorageBytes, HostHeap, HostHeap), HostHeap)>> {
    blocks
        .iter()
        .map(|graphs| {
            graphs
                .iter()
                .map(|(&topology, graph)| (topology, (charge(&graph.layout), graph.owned)))
                .collect()
        })
        .collect()
}

fn compare(
    differences: &mut Vec<String>,
    outlined: &TargetFamilyCertificate<NativeGraphOutline>,
    laid: &TargetFamilyCertificate<NativeGraphLayout>,
    limits: ResourceLimits,
) {
    let d = differences;
    field(d, "storage", outlined.resources.storage, laid.resources.storage);
    field(d, "resource graphs", &outlined.resources.graphs, &laid.resources.graphs);
    field(
        d,
        "binding constants",
        outlined.resources.binding_constant_bytes,
        laid.resources.binding_constant_bytes,
    );
    field(d, "constant and plan heap", outlined.resources.host_heap, laid.resources.host_heap);
    field(d, "launch graphs", &outlined.graphs, &laid.graphs);
    field(d, "sealed", outlined.sealed, laid.sealed);
    field(d, "upload regions", &outlined.upload_regions, &laid.upload_regions);
    field(d, "passes", outlined.passes, laid.passes);
    field(d, "group bytes", group_bytes(outlined, limits), group_bytes(laid, limits));
    field(d, "entries", entries(&outlined.entries), entries(&laid.entries));
    field(
        d,
        "per-layer entry",
        outlined.per_layer.as_ref().map(charge),
        laid.per_layer.as_ref().map(charge),
    );
    field(d, "blocks", blocks(&outlined.blocks), blocks(&laid.blocks));
    field(d, "untapped blocks", blocks(&outlined.untapped), blocks(&laid.untapped));
}

/// What its prompt groups' own buffers hold at each pass size's places
/// (`TargetFamilyCertificate::checked`).
fn group_bytes<L>(certificate: &TargetFamilyCertificate<L>, limits: ResourceLimits) -> Vec<Result<u64, String>> {
    (1..=PassLaunches::whole_chain(limits).get())
        .map(|places| certificate.group.group_bytes(&certificate.served, places))
        .collect()
}

fn entries<L: Certified>(
    entries: &BTreeMap<EntryTokens, L>,
) -> Vec<(EntryTokens, (NativeGraphStorageBytes, HostHeap, HostHeap))> {
    entries.iter().map(|(&source, layout)| (source, charge(layout))).collect()
}

/// The traces the target family's graphs recorded so far, entries' and
/// blocks' (`NativeGraphTraces::recorded`).
pub(crate) fn recorded_traces() -> usize {
    super::entry::ENTRY_TRACES.recorded() + super::block::BLOCK_TRACES.recorded()
}

/// How many graphs the outlined drafted and undrafted families of `plan`
/// hold: each an answer of the traces, or a recording.
pub(crate) fn outlined_graphs(certificates: &[TargetFamilyCertificate<NativeGraphOutline>]) -> usize {
    certificates
        .iter()
        .map(|family| {
            family.entries.len()
                + family.blocks.iter().chain(&family.untapped).map(BTreeMap::len).sum::<usize>()
        })
        .sum()
}
