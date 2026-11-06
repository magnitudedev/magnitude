use super::{
    GroupRows,
    block::{BlockGraphInputs, checked_block_graph_draft},
    classes::{
        BlockOutputScales, BlockServed, FamilyOutline, Passes, StagedClass, Topology, UnreadScales,
        block_class_slice, block_listings, block_routed, block_served, block_split, block_stages,
        block_upload_regions, chained_once_runs, class_topology, enumerate_block_graphs, LaunchRuns, once_class, once_graphs,
        slot_scores,
    },
    entry::{
        EntryTokens, TARGET_TABLE, embedding_weight, entry_graph_topology, keeps_token_rows,
        notice_bytes, notice_rows, notifies_selection, target_weight,
    },
    heap::{
        GraphCounts, PlannedHeap, PlannedSealedGraphs, SealedGraphsHeap, block_binding_storage,
        block_graph_storage,
    },
    prepared::{BlockGraphShape, block_scopes},
    weights::{CheckedLandingEntries, activation},
};
use crate::{
    ModelLoadPlan, ResourceLimits, StateResourcePlan, TargetProgramPlan,
    operators::{
        block, paired_block,
        per_layer::graph::{
            CheckedPerLayerEntryEntries, per_layer_entry, per_layer_entry_class_slice,
        },
        routed::{
            split::{OnceStage, RunStage, Stage},
            streamed,
        },
    },
    programs::{
        graph::{GraphError, RowForm},
        native_constants::{CheckedGraphFamilyResources, CheckedGraphResources, GraphConstant},
    },
};
use magnitude_family_contracts::{Decoder, WeightKind};
use seismic::{
    BackendName, GraphCensus, HostHeap, LaunchGraphs, NativeGraphClassSlice, NativeGraphLayout,
    NativeGraphMetadata,
};
use std::collections::BTreeMap;

/// The same decoder class ladder as `PreparedTargetGraphs::prepare`. The
/// family takes independent maxima because its workspace, output and upload
/// holdings may peak in different members.
#[cfg(test)]
pub(crate) fn checked_target_family_storage(
    backend: BackendName,
    load: &ModelLoadPlan,
    geometry: &Decoder,
    state: &StateResourcePlan,
    plan: &TargetProgramPlan,
    limits: ResourceLimits,
) -> Result<CheckedGraphResources, GraphError> {
    certify_target_family(backend, load, geometry, state, plan, limits)
        .map(|certificate| certificate.resources)
}

/// The target family a load of `plan` prepares, as its assessment charges
/// it ([`CheckedTargetFamily`]), from the one enumeration the sealer
/// follows.
pub(crate) fn checked_target_family(
    backend: BackendName,
    load: &ModelLoadPlan,
    geometry: &Decoder,
    state: &StateResourcePlan,
    plan: &TargetProgramPlan,
    limits: ResourceLimits,
) -> Result<CheckedTargetFamily, GraphError> {
    certify_target_family(backend, load, geometry, state, plan, limits)?.checked(limits)
}

/// [`checked_target_family`] of `plan`'s family and of the same target's
/// without its drafter's writes (`certify_target_and_undrafted`).
pub(crate) fn checked_target_and_undrafted(
    backend: BackendName,
    load: &ModelLoadPlan,
    geometry: &Decoder,
    state: &StateResourcePlan,
    plan: &TargetProgramPlan,
    limits: ResourceLimits,
) -> Result<[CheckedTargetFamily; 2], GraphError> {
    let [target, undrafted] = certify_target_and_undrafted(backend, load, geometry, state, plan, limits)?;
    Ok([target.checked(limits)?, undrafted.checked(limits)?])
}

/// What the load's assessment charges of the target family, from the one
/// enumeration the sealer follows ([`checked_target_family`]).
pub(crate) struct CheckedTargetFamily {
    /// The family's storage.
    pub resources: CheckedGraphResources,
    /// The graphs a load of it seals and the host heap they hold.
    pub sealed: PlannedSealedGraphs,
    /// The upload regions of a workspace slot
    /// (`PreparedTargetGraphs::upload_regions`).
    pub upload_regions: Vec<u64>,
    /// What its graphs are made of (`TargetFamilyCertificate::graphs`).
    pub graphs: LaunchGraphs,
    /// The bytes the own buffers of a prompt group of the plan's chain
    /// places hold beside the floor's (`GroupRows::group_bytes`), granted
    /// when a group opens.
    pub group_bytes: u64,
}

pub(super) struct TargetFamilyCertificate {
    pub(super) outline: FamilyOutline,
    /// How each block is served (`block_served`).
    pub(super) served: Vec<BlockServed>,
    /// What its prompt groups' buffers span.
    pub(super) group: GroupRows,
    pub(super) resources: CheckedGraphResources,
    /// What the family's decode and largest launches replay
    /// (`seismic::NativeGraphFamily::graphs`), bounded from the layouts the
    /// plans seal into.
    pub(super) graphs: LaunchGraphs,
    /// The graphs a load seals into these layouts, and their host heap.
    pub(super) sealed: PlannedSealedGraphs,
    /// The upload regions of a workspace slot: one for each graph run that
    /// writes input one step queues on it before any completes, each
    /// holding that run's input until it completes. The embedding entry's,
    /// the per-layer entry's of a per-layer model, each the largest input
    /// its layouts place, the launch's controls, which its blocks bind
    /// (`LaunchControlLayout`), and each block's (`block_upload_regions`).
    pub(super) upload_regions: Vec<u64>,
    pub(super) entries: BTreeMap<EntryTokens, NativeGraphLayout>,
    /// The per-layer entry graph's layout, of a per-layer entry.
    pub(super) per_layer: Option<NativeGraphLayout>,
    /// Per block, what each row regime's classes seal into, those that
    /// list their launch's history row tiles apart.
    /// Per block, its certified graphs by topology (`class_topology`).
    pub(super) blocks: Vec<BTreeMap<Topology, CertifiedBlockGraph>>,
    /// The passes a group runs (`group_pass_rows`).
    pub(super) passes: Passes,
}

impl TargetFamilyCertificate {
    /// What the assessment charges of it, its prompt groups of `limits`'
    /// chain places.
    fn checked(self, limits: ResourceLimits) -> Result<CheckedTargetFamily, GraphError> {
        let group_bytes = self.group.group_bytes(&self.served, limits.chain_capacity.get())?;
        Ok(CheckedTargetFamily {
            resources: self.resources,
            sealed: self.sealed,
            upload_regions: self.upload_regions,
            graphs: self.graphs,
            group_bytes,
        })
    }
}

/// What every class of one block's row regime prepares to: one topology.
#[derive(Clone)]
pub(super) struct CertifiedBlockGraph {
    /// The layout its plans seal into.
    pub(super) layout: NativeGraphLayout,
    /// The host heap of the vectors each of its prepared graphs owns
    /// (`owned_vectors_heap`). Every weight, constant and state port is a
    /// port of the graph, and the layout holds its classes to its ports,
    /// so the template's lengths are each class's.
    pub(super) owned: HostHeap,
}

pub(super) fn certify_target_family(
    backend: BackendName,
    load: &ModelLoadPlan,
    geometry: &Decoder,
    state: &StateResourcePlan,
    plan: &TargetProgramPlan,
    limits: ResourceLimits,
) -> Result<TargetFamilyCertificate, GraphError> {
    let outline = FamilyOutline::of(backend, load, geometry, state, plan, limits)?;
    FamilyLayouts::of_entries(backend, load, geometry, plan, &outline)?
        .certify(outline, load, geometry, state, plan, limits)
}

/// The family of `plan`, and the family of the same target without its
/// drafter's writes (`FamilyOutline::undrafted`), what a load of the target
/// alone seals: from one certification of the layouts they share, each
/// family's enumeration, bindings, upload regions, census and floor extents
/// its own. A plan that drafts with nothing has one family, twice.
pub(super) fn certify_target_and_undrafted(
    backend: BackendName,
    load: &ModelLoadPlan,
    geometry: &Decoder,
    state: &StateResourcePlan,
    plan: &TargetProgramPlan,
    limits: ResourceLimits,
) -> Result<[TargetFamilyCertificate; 2], GraphError> {
    let outline = FamilyOutline::of(backend, load, geometry, state, plan, limits)?;
    let undrafted = outline.undrafted();
    let mut layouts = FamilyLayouts::of_entries(backend, load, geometry, plan, &outline)?;
    Ok([
        layouts.certify(outline, load, geometry, state, plan, limits)?,
        layouts.certify(undrafted, load, geometry, state, plan, limits)?,
    ])
}

/// What a certified graph adds to its family's resources: its storage and
/// constants as a template certified at its rows' classes, or as the one
/// class of a block's pass or fill.
#[derive(Clone)]
enum Included {
    Classes(Vec<GraphConstant>, Vec<u64>),
    Once(Vec<GraphConstant>),
}

/// One block shape's certified graphs by topology, with what each adds to
/// its family's resources.
type ShapeLayouts = BTreeMap<Topology, (CertifiedBlockGraph, Included)>;

/// The layouts a target's families seal into, each certified once: the
/// entries', and each block shape's (`BlockGraphShape`, which decides
/// every graph a block prepares, the sealer sharing one plan between the
/// blocks of a shape): the launches' own, and its group's at the passes
/// the group runs. The families of one target differ only in which shape
/// each block is of, so each reads its blocks' layouts here.
struct FamilyLayouts {
    entries: BTreeMap<EntryTokens, NativeGraphLayout>,
    per_layer: Option<NativeGraphLayout>,
    launches: Vec<(BlockGraphShape, ShapeLayouts)>,
    groups: Vec<(BlockGraphShape, Passes, ShapeLayouts)>,
    /// The passes a group runs beside each workspace arena found: a
    /// function of the arena alone, since a block's pass is of its shape
    /// without its drafter's writes, of which every family has each
    /// (`group_pass_rows`).
    passes: Vec<(u64, Passes)>,
}

impl FamilyLayouts {
    /// The entry graphs' layouts, the same in every family of the target.
    fn of_entries(
        backend: BackendName,
        load: &ModelLoadPlan,
        geometry: &Decoder,
        plan: &TargetProgramPlan,
        outline: &FamilyOutline,
    ) -> Result<Self, GraphError> {
        let FamilyOutline { ref row_classes, largest_rows, .. } = *outline;
        let mut entries = BTreeMap::new();
        let landing = CheckedLandingEntries::of(load, TARGET_TABLE)?;
        for source in [EntryTokens::Uploaded, EntryTokens::Selected] {
            let weight = embedding_weight(load)?;
            let elements = [("EW", weight.resident), ("A", activation(geometry))];
            let graph = entry_graph_topology(
                NativeGraphMetadata::new_template(backend),
                &elements,
                notifies_selection(load).then_some(&[][..]),
                landing.as_ref().map(CheckedLandingEntries::entries),
                keeps_token_rows(load),
                weight,
                geometry,
                largest_rows,
                source,
            )?
            .graph;
            let layout = graph
                .seal_template()
                .and_then(|template| {
                    template.certify(&[
                        NativeGraphClassSlice::new().dimension("M", row_classes.iter().copied())
                    ])
                })
                .map_err(|error| format!("target entry graph: {error}"))?;
            entries.insert(source, layout);
        }
        let per_layer = match (plan.per_layer(), &geometry.entry.per_layer) {
            (None, None) => None,
            (Some(binding), Some(entry)) => {
                let checked = CheckedPerLayerEntryEntries::new(binding);
                let mut graph = NativeGraphMetadata::new_template(backend);
                per_layer_entry(
                    &mut graph,
                    checked.entries(),
                    target_weight(load, WeightKind::PerLayerModelProjection)?,
                    target_weight(load, WeightKind::PerLayerProjectionNorm)?,
                    binding,
                    entry,
                    largest_rows,
                )?;
                let layout = graph
                    .seal_template()
                    .and_then(|template| {
                        template.certify(&[per_layer_entry_class_slice(row_classes.iter().copied())])
                    })
                    .map_err(|error| format!("per-layer entry graph: {error}"))?;
                Some(layout)
            }
            _ => return Err("the per-layer entry's binding and geometry disagree".into()),
        };
        Ok(Self { entries, per_layer, launches: Vec::new(), groups: Vec::new(), passes: Vec::new() })
    }

    /// The family of `outline`, its blocks' layouts certified where no
    /// family certified their shape's.
    fn certify(
        &mut self,
        outline: FamilyOutline,
        load: &ModelLoadPlan,
        geometry: &Decoder,
        state: &StateResourcePlan,
        plan: &TargetProgramPlan,
        limits: ResourceLimits,
    ) -> Result<TargetFamilyCertificate, GraphError> {
        let backend = outline.backend;
        let FamilyOutline { ref row_classes, largest_rows, max_slots, max_segments, tap_width, ref tapped, .. } = outline;
        let distinct_blocks = (0..geometry.blocks.len())
            .filter(|&index| outline.first_of_shape(index) == index)
            .collect::<Vec<_>>();
        let mut family = CheckedGraphFamilyResources::new();
        for layout in self.entries.values().chain(&self.per_layer) {
            family.include(layout.storage_bytes(), []);
        }
        let entries = self.entries.clone();
        let per_layer = self.per_layer.clone();
        // A block's pass and fill, of the one class of `rows` rows; a fill reads
        // no rows.
        let once = |index: usize, stage: OnceStage, rows: u64| {
            let slot = plan.blocks()[index];
            let paired = paired_block(&geometry.blocks[index]).map_err(|error| error.to_string())?;
            let (graph, constants, owned) = checked_block_graph_draft(
                NativeGraphMetadata::new_template(backend),
                slot,
                None,
                BlockGraphInputs::of_class(load, geometry, state, index, once_class(rows, stage), rows),
            )?;
            let slice = match stage {
                OnceStage::Pass => block::feed_forward_class_slice(
                    &paired,
                    rows,
                    rows,
                    NativeGraphClassSlice::new().dimension("M", [rows]),
                )?,
                OnceStage::Fill => NativeGraphClassSlice::new(),
            };
            let layout = graph
                .seal_template()
                .and_then(|template| template.certify(&[slice]))
                .map_err(|error| format!("target block {index}'s {stage:?}: {error}"))?;
            Ok::<_, GraphError>((layout, constants, owned))
        };
        // A block's run graphs of each row regime: its launches' own (`group`
        // false) or its group's combine (true), of a pass of `passes`.
        let runs = |index: usize, group: bool, passes: Passes| {
            let mut graphs = ShapeLayouts::new();
            let slot = plan.blocks()[index];
            let paired = paired_block(&geometry.blocks[index]).map_err(|error| error.to_string())?;
            let split = block_split(load, geometry, index, limits)?.is_some();
            let resident = streamed::resident_form(load, block_scopes(index)?[1]);
            let mut regimes: BTreeMap<RowForm, Vec<u64>> = BTreeMap::new();
            for &rows in row_classes {
                regimes.entry(RowForm::of(rows)).or_default().push(rows);
            }
            for rows in regimes.into_values() {
                let largest = *rows.last().expect("a regime holds a row class");
                // A regime's listing classes are their own topology (another
                // kernel of the entry, and the list's input), over every listed
                // tile count; so are a selected read's classes that score (the
                // indexer's selection and the selected attention forms).
                let topologies = [(false, false), (true, false)]
                    .into_iter()
                    .chain(slot_scores(slot).then_some((false, true)));
                for (listed, scored) in topologies {
                    let listings = match listed {
                        false => Vec::new(),
                        true => block_listings(state, backend, slot, index, largest)?.tiles,
                    };
                    let mut listing_rows = Vec::new();
                    for &rows in &rows {
                        if !listed
                            || !block_listings(state, backend, slot, index, rows)?
                                .tiles
                                .is_empty()
                        {
                            listing_rows.push(rows);
                        }
                    }
                    let rows = listing_rows;
                    if rows.is_empty() {
                        continue;
                    }
                    let stages = RunStage::sealed(split, resident, largest).iter().filter(|&&stage| (stage == RunStage::Combine) == group);
                    for &stage in stages {
                        // The regime's class the template is drafted at.
                        let drafted = (
                            (largest, 1, 1, listings.last().copied().unwrap_or(0), u64::from(scored)),
                            Stage::Run(stage),
                        );
                        let (graph, constants, owned) = checked_block_graph_draft(
                            NativeGraphMetadata::new_template(backend),
                            slot,
                            tap_width
                                .map(|width| (width, tapped[index].positions()))
                                .filter(|(_, positions)| positions.any()),
                            BlockGraphInputs {
                                residual: outline.exports_residual(index),
                                ..BlockGraphInputs::of_class(load, geometry, state, index, drafted, passes.rows())
                            },
                        )?;
                        let slices = rows
                            .iter()
                            .map(|&rows| {
                                let grouped = if stage == RunStage::Combine { passes.rows() } else { rows };
                                block_class_slice(&paired, rows, grouped, max_slots, max_segments, &listings, scored)
                            })
                            .collect::<Result<Vec<_>, _>>()?;
                        let layout = graph
                            .seal_template()
                            .and_then(|template| template.certify(&slices))
                            .map_err(|error| format!("target block {index}: {error}"))?;
                        graphs.insert(
                            class_topology(drafted),
                            (CertifiedBlockGraph { layout, owned }, Included::Classes(constants, rows.clone())),
                        );
                    }
                }
            }
            Ok::<_, GraphError>(graphs)
        };
        // Two phases: the launches' own graphs, which fix the arena the
        // group's pass must fit in (`group_pass_rows`); then the group's
        // graphs, of the pass's rows: its backs, pass and fills. Before the
        // arena is fixed, a pass may cover the whole chain.
        let chain = Passes { region: largest_rows, launches: limits.chain_capacity.get() };
        let mut blocks = vec![ShapeLayouts::new(); geometry.blocks.len()];
        for &index in &distinct_blocks {
            let shape = &outline.shapes[index];
            let graphs = match self.launches.iter().find(|(certified, _)| certified == shape) {
                Some((_, graphs)) => graphs.clone(),
                None => {
                    let graphs = runs(index, false, chain)?;
                    self.launches.push((shape.clone(), graphs.clone()));
                    graphs
                }
            };
            blocks[index] = graphs;
        }
        let arena = blocks
            .iter()
            .flat_map(BTreeMap::values)
            .map(|(graph, _)| graph.layout.storage_bytes().workspace)
            .chain(entries.values().chain(&per_layer).map(|layout| layout.storage_bytes().workspace))
            .max()
            .expect("a target family has entry graphs");
        let passes = match self.passes.iter().find(|&&(found, _)| found == arena) {
            Some(&(_, passes)) => passes,
            None => {
                let passes = group_pass_rows(&distinct_blocks, geometry, chain, arena, |index, rows| {
                    Ok(once(index, OnceStage::Pass, rows)?.0.storage_bytes().workspace)
                })?;
                self.passes.push((arena, passes));
                passes
            }
        };
        for &index in &distinct_blocks {
            let shape = &outline.shapes[index];
            let graphs = match self.groups.iter().find(|(certified, at, _)| certified == shape && *at == passes) {
                Some((.., graphs)) => graphs.clone(),
                None => {
                    let mut graphs = runs(index, true, passes)?;
                    let streams = streamed::streams(load, block_scopes(index)?[1]);
                    for (stage, rows) in once_graphs(streams, block_split(load, geometry, index, limits)?.is_some(), passes) {
                        let (layout, constants, owned) = once(index, stage, rows)?;
                        graphs.insert(
                            class_topology(once_class(rows, stage)),
                            (CertifiedBlockGraph { layout, owned }, Included::Once(constants)),
                        );
                    }
                    self.groups.push((shape.clone(), passes, graphs.clone()));
                    graphs
                }
            };
            blocks[index].extend(graphs);
        }
        for (graph, included) in distinct_blocks.iter().flat_map(|&index| blocks[index].values()) {
            let storage = graph.layout.storage_bytes();
            match included {
                Included::Classes(constants, rows) => family.include_classes(storage, constants, rows)?,
                Included::Once(constants) => family.include(storage, constants.iter().cloned()),
            }
        }
        let served = block_served(load, geometry, limits, outline.chain_rows, passes)?;
        for index in 0..blocks.len() {
            let first = outline.first_of_shape(index);
            if first != index {
                blocks[index] = blocks[first].clone();
            }
        }
        let blocks = blocks
            .into_iter()
            .map(|graphs| graphs.into_iter().map(|(topology, (graph, _))| (topology, graph)).collect::<BTreeMap<_, _>>())
            .collect::<Vec<_>>();
        // The graphs a load seals: two entries and the per-layer entry per row
        // class, and the enumerated block plans, each into its regime's layout.
        // Layer scales the plan does not know are taken both ways, which
        // bounds the sealed block plans from both sides.
        let enumerate = |unread| {
            enumerate_block_graphs(&outline, &served, passes, state, plan, BlockOutputScales::Planned(unread))
        };
        let enumeration = enumerate(UnreadScales::Distinct)?;
        let least = match enumeration.exact {
            true => None,
            false => Some(enumerate(UnreadScales::Equal)?),
        };
        let count = |sealing: &[(usize, StagedClass)]| {
            let mut planned = PlannedHeap::default();
            for _ in row_classes {
                for layout in entries.values().chain(&per_layer) {
                    planned.seal(layout)?;
                }
            }
            let mut sealed = GraphCounts::default();
            for &(index, class) in sealing {
                planned.seal_block(
                    blocks[index]
                        .get(&class_topology(class))
                        .ok_or("a block class has no certified layout")?,
                )?;
                sealed.count(class.1);
            }
            Ok::<_, GraphError>((planned, sealed))
        };
        let (planned, sealed_blocks) = count(&enumeration.sealing)?;
        // The bindings a load makes of them: one for each entry graph, and one
        // for every block's every class, whichever block sealed its plan.
        let mut bound = HostHeap::default();
        let mut binding = |layout: &NativeGraphLayout| {
            bound = bound
                .plus(layout.bound_heap())
                .ok_or("planned graph binding heap overflows")?;
            Ok::<_, GraphError>(())
        };
        for _ in row_classes {
            for layout in entries.values().chain(&per_layer) {
                binding(layout)?;
            }
        }
        let mut block_bindings = vec![GraphCounts::default(); blocks.len()];
        let bound_classes = enumeration.sealing.iter().copied().chain(
            enumeration
                .sharing
                .iter()
                .map(|&(index, class, _)| (index, class)),
        );
        for (index, class) in bound_classes {
            binding(
                &blocks[index]
                    .get(&class_topology(class))
                    .ok_or("a block class has no certified layout")?
                    .layout,
            )?;
            block_bindings[index].count(class.1);
        }
        let bound = bound
            .plus(block_binding_storage(&block_bindings)?)
            .ok_or("planned graph binding heap overflows")?;
        // Every block holds a graph of each of its classes, whichever block
        // sealed it; what differs with the count is the sealed graphs' shared
        // allocations.
        let graphs = |(planned, sealed_blocks): (PlannedHeap, GraphCounts)| {
            Ok::<_, GraphError>(SealedGraphsHeap {
                graphs: planned.graphs,
                plans: planned.plans,
                storage: block_graph_storage(&block_bindings, sealed_blocks)?,
                owned: planned.owned,
            })
        };
        let at_most = graphs((planned, sealed_blocks))?;
        let at_least = match &least {
            None => at_most,
            Some(least) => graphs(count(&least.sealing)?)?,
        };
        let sealed = PlannedSealedGraphs {
            at_most,
            at_least,
            bound,
            constants: family.constant_heap()?,
        };
        // The floor group's buffers and the pass's, and the selection notice,
        // which the family binds with its constants
        // (`PreparedTargetGraphs::binding_constant_bytes`).
        let mut resources = family.finish()?;
        let notice = notice_bytes(notice_rows(load, largest_rows))?;
        let group_rows = GroupRows::of(load, geometry, plan, tap_width, largest_rows);
        let floor = group_rows.extents(&served, 1);
        let (group, pass) = (floor.bytes()?, floor.pass_bytes()?);
        resources.binding_constant_bytes = resources
            .binding_constant_bytes
            .checked_add(group)
            .and_then(|bytes| bytes.checked_add(pass))
            .and_then(|bytes| bytes.checked_add(notice))
            .ok_or("target binding charge overflows")?;
        // A graph writes the input its topology's layout places: every class
        // of the topology seals into it.
        let upload = |layout: &NativeGraphLayout| layout.storage_bytes().upload;
        let largest = |layouts: &mut dyn Iterator<Item = &NativeGraphLayout>| {
            layouts.map(upload).max().filter(|&bytes| bytes > 0)
        };
        let upload_regions = largest(&mut entries.values())
            .into_iter()
            .chain(largest(&mut per_layer.iter()))
            .chain([enumeration.controls.region_bytes()])
            .chain(blocks.iter().enumerate().flat_map(|(index, graphs)| {
                block_upload_regions(
                    row_classes.iter().copied(),
                    |rows| block_stages(&served, index, rows),
                    // A run's place holds each graph that takes it: a whole
                    // decode block's either form's.
                    |rows, stage| {
                        graphs
                            .iter()
                            .filter(|(&(form, _, _, of, _), _)| {
                                form == RowForm::of(rows) && matches!(of, Stage::Run(of) if of.place() == stage)
                            })
                            .map(|(_, graph)| upload(&graph.layout))
                            .max()
                            .unwrap_or(0)
                    },
                )
            }))
            .collect();
        // What the family's launches replay, before any plan is sealed
        // (`seismic::NativeGraphFamily::graphs`): a bound the sealed family
        // keeps within, which a load charges before it seals. Every plan of a
        // layout shares one graph, whatever its class (on CUDA a node for each
        // launch a plan of it makes, which a layout's every declared launch
        // bounds; elsewhere a run's own, which its layout's recording bounds), so
        // a launch replays, of each layout, a graph for each run of it it
        // makes: both entries' and the per-layer entry's of its rows, each
        // block's runs of its class (each sharing an earlier block's plans its
        // own run), and, where the class splits at a block (`RunStage::Front`),
        // every graph the block runs once a group (`Stage::Once`, its fill and
        // its passes): a class served whole, as decode is, runs none. A block's
        // layouts are its shape's first block's. The first launch class, the
        // fewest rows, is the decode launch.
        let overflow = || GraphError::from("a target family's graph census overflows");
        #[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
        enum Graphed {
            Entry(EntryTokens),
            PerLayer,
            Block(usize, Topology),
        }
        let staged = enumeration
            .sealing
            .iter()
            .copied()
            .chain(enumeration.sharing.iter().map(|&(index, class, _)| (index, class)))
            .collect::<Vec<_>>();
        let launch_runs = LaunchRuns::of(
            &served,
            staged
                .iter()
                .map(|&(index, staged)| (index, staged, Graphed::Block(outline.first_of_shape(index), class_topology(staged)))),
        );
        // Each launch's runs of each layout, the first launch class's (the
        // fewest rows, decode's) its decode, and of each layout the most any
        // launch makes its union.
        let mut most = BTreeMap::<Graphed, u64>::new();
        let mut decode = None;
        for &class in &enumeration.launch_classes {
            let mut runs = BTreeMap::<Graphed, u64>::new();
            for &source in entries.keys() {
                *runs.entry(Graphed::Entry(source)).or_default() += 1;
            }
            if per_layer.is_some() {
                *runs.entry(Graphed::PerLayer).or_default() += 1;
            }
            for &graphed in launch_runs.of_class(class) {
                *runs.entry(graphed).or_default() += 1;
            }
            for (&graphed, &count) in &runs {
                let most = most.entry(graphed).or_default();
                *most = (*most).max(count);
            }
            decode.get_or_insert(runs);
        }
        let census = |runs: &BTreeMap<Graphed, u64>| -> Result<GraphCensus, GraphError> {
            let mut census = GraphCensus::default();
            for (&graphed, &count) in runs {
                let layout = match graphed {
                    Graphed::Entry(source) => &entries[&source],
                    Graphed::PerLayer => per_layer.as_ref().ok_or("a per-layer run has no layout")?,
                    Graphed::Block(index, topology) => {
                        &blocks[index].get(&topology).ok_or("a block class has no certified layout")?.layout
                    }
                };
                census = census.plus(layout.storage_bytes().recorded.times(count).ok_or_else(overflow)?).ok_or_else(overflow)?;
            }
            Ok(census)
        };
        // What a chain of each length beyond one launch further runs of the
        // graphs its split blocks run once a group (`chained_once_runs`),
        // each block's own once classes, at its shape's first block's
        // layouts: none where one graph serves every run of its shape in a
        // submission.
        let chain = |launches: usize| {
            let further = chained_once_runs(passes, launches);
            let mut runs = BTreeMap::<Graphed, u64>::new();
            for &(index, (class, stage)) in &staged {
                let Stage::Once(once) = stage else { continue };
                if let Some(&(_, count)) = further.iter().find(|&&(key, _)| key == (once, class.0)) {
                    *runs.entry(Graphed::Block(outline.first_of_shape(index), class_topology((class, stage)))).or_default() += count;
                }
            }
            census(&runs)
        };
        let chained = match backend.graphs_update_in_flight() {
            true => Vec::new(),
            false => (2..=limits.chain_capacity.get()).map(chain).collect::<Result<Vec<_>, _>>()?,
        };
        let graphs = LaunchGraphs {
            decode: census(&decode.ok_or("a target family has no launch class")?)?,
            union: census(&most)?,
            chained,
        };
        Ok(TargetFamilyCertificate {
            outline,
            served,
            group: group_rows,
            resources,
            graphs,
            sealed,
            upload_regions,
            entries,
            per_layer,
            blocks,
            passes,
        })
    }
}

/// The passes a group runs: of the most whole launches, of at most
/// `chain`'s, whose pass needs no more workspace (`pass_workspace` of block
/// and rows) at any of the routed `blocks` than `arena`, the workspace the
/// launches' own graphs need. A merge set of more runs splits into passes
/// of these rows, so the pass's grouping and expert scratch share the
/// arena's bytes and add none. A pass needs more workspace with more rows;
/// one launch's pass is the least a group runs, as a launch of its own
/// would.
fn group_pass_rows(
    blocks: &[usize],
    geometry: &Decoder,
    chain: Passes,
    arena: u64,
    mut pass_workspace: impl FnMut(usize, u64) -> Result<u64, GraphError>,
) -> Result<Passes, GraphError> {
    let mut passes = chain;
    for &index in blocks {
        if block_routed(geometry, index)?.is_none() {
            continue;
        }
        // The most launches that fit: at least one, at most those found.
        let (mut fits, mut over) = (1, passes.launches + 1);
        while over - fits > 1 {
            let middle = (fits + over) / 2;
            match pass_workspace(index, passes.of_chain(middle))? <= arena {
                true => fits = middle,
                false => over = middle,
            }
        }
        passes.launches = fits;
    }
    Ok(passes)
}
