//! What a streamed routed layer adds to its routed form's graph, fused or
//! general (`routed.seismic`'s streaming section):
//!
//! ```text
//! decode rows:  [step] → route + resolve + assign (each choice's source by
//!               the step-stamped map: a bank record, a host tier record
//!               the GPU reads in place or the CPU computes, or absent) →
//!               gated experts' early gate/up (the choices the route
//!               sourced) → landing launch (convert each absent expert's
//!               record as its read lands, where the bank places; source each
//!               absent choice where it landed) → site request (the CPU's
//!               choices and the rows, across a link) → the late gate/up
//!               (the choices the landing sourced; up-only experts' whole
//!               expansion), each choice read where its source names, by
//!               the resident gate/up's arithmetic → the projections → the sum,
//!               waiting for the CPU's answer and adding it (a whole
//!               group's layer: the resident combine)
//! larger rows:  front: route → scratch request
//!               fill:  each round's landing launch: arrive, convert each
//!                      conveyed record as it lands → seed copy
//!               pass:  over a merge set's rows, the resident grouped
//!                      entries over the bank's records where the map
//!                      holds an expert for the step (a hit), else the
//!                      scratch tier
//!               back:  combine
//! ```
//!
//! A bank-served launch's streamed layers run in order, so its first
//! streamed layer sets the step for all of them. A scratch-served layer's
//! front and fill ([`super::split::Stage`]) read their own run's sequence: a prompt group
//! runs every step's front of a layer, then each merge set's fills, pass
//! and backs.

use crate::native::{BankRouteKernels, ExpertBankKernels};
use crate::planning::{
    BankBinding, BankPart, BankedFeedForward, ExpertBankPlan, RecordTensors, RoutedTensor,
};
use crate::programs::graph::{GraphError, draft::GraphDraft};
use crate::programs::graph::facts::Numbers;
use crate::programs::native_constants::GraphConstant;

/// What a host port of the gate's records of `ports` (a round's conveyor)
/// binds: the gate's, or of an up-only record, none of its up's (a view of
/// no records, `GV` 0), in the element its entries bind the gate's
/// (`CheckedBankEntries`).
pub(crate) enum GateRecords<'p> {
    Gated(&'p WorkflowTensor),
    Up(WorkflowTensorView),
}

impl<'p> GateRecords<'p> {
    pub(crate) fn of(ports: &'p RecordTensors<NativePort>) -> Self {
        match &ports.gate {
            Some(gate) => Self::Gated(gate.tensor()),
            None => Self::Up(ports.up.tensor().slice_leading(0, 0)),
        }
    }

    pub(crate) fn arg(&self) -> WorkflowTensorRef<'_> {
        match self {
            Self::Gated(gate) => (*gate).into(),
            Self::Up(view) => view.into(),
        }
    }
}
use crate::experts::ParkKind;
use crate::programs::native_target_graph::{
    BankPort, GraphPark, GraphTimeline, WeightPart, WeightPort, bank_port, weight_port,
};
use crate::ModelLoadPlan;
use crate::programs::native_target_graph::PlannedWeights;
use magnitude_family_contracts::{WeightRole, WeightScope};
use magnitude_kernels::{
    routed_copy, routed_down_project, routed_down_sited, routed_gate_up, routed_gate_up_part,
    routed_land, routed_list_grid, routed_output_project,
    routed_output_sited, routed_output_streamed, routed_pass_map, routed_round, routed_route_streamed,
    routed_scratch_grids, routed_scratch_request, routed_seed_commit, routed_select_streamed,
    routed_site_request, routed_stage_host, routed_step,
};
use seismic::{
    Dim, Element, NativeGraph, NativePort, NativeTimelinePort, WorkflowTensor,
    WorkflowTensorRef, WorkflowTensorView,
};

/// A streamed layer's decode route and resolve by its routed form
/// (`native::BankRouteKernels`).
pub(crate) enum BankRoute<'a, G: GraphDraft + 'a> {
    Fused {
        activation: Element,
        route: G::Ranged<'a, routed_route_streamed::Entry>,
    },
    General {
        activation: Element,
        select: G::Binding<'a, routed_select_streamed::Entry>,
    },
}

impl<'a, G: GraphDraft + 'a> Copy for BankRoute<'a, G> {}
impl<'a, G: GraphDraft + 'a> Clone for BankRoute<'a, G> {
    fn clone(&self) -> Self {
        *self
    }
}

/// A streamed layer's bank entries; `seed` per routed tensor of the record.
pub(crate) struct BankGraphEntries<'a, G: GraphDraft + 'a> {
    pub route: BankRoute<'a, G>,
    pub land: G::Binding<'a, routed_land::Entry>,
    pub step: G::Binding<'a, routed_step::Entry>,
    pub scratch_request: G::Binding<'a, routed_scratch_request::Entry>,
    pub round: G::Binding<'a, routed_round::Entry>,
    pub scratch_grids: G::Binding<'a, routed_scratch_grids::Entry>,
    pub pass_map: G::Binding<'a, routed_pass_map::Entry>,
    /// The device's seed copy into the bank's records, whoever places
    /// (`DevicePlacement`).
    pub seeding: BankSeedingEntries<'a, G>,
    /// The CPU site's request, where the layer has a site.
    pub site: Option<BankSiteEntries<'a, G>>,
    /// The split decode combine (`native::ExpertBankKernels::combine`),
    /// decoupled from `site` for general routed layers so UMA layers can
    /// overlap early-hit `routed_down_project` with `routed_land`.
    pub combine: Option<BankCombineEntry<'a, G>>,
    /// A gated layer's gate/up passes, its held choices' before the tier
    /// wait and the wait's after it (`native::ExpertBankKernels::gate_up`);
    /// none of up-only experts.
    pub gate_up: Option<G::Ranged<'a, routed_gate_up_part::Entry>>,
    /// Whether the target backend's `routed_down_project` kernel supports
    /// masked-write skip semantics (`mode == 1` hit-only / `mode == 2` skip
    /// `code == -1` without overwriting `published`).
    pub masked_down_project: bool,
    /// Unified device-classified 3-tier decode staging + self-gate
    /// (`routed_stage_host`, 4 CTAs x 256 threads), present on CUDA `Placer::Tier`.
    pub stage_host: Option<G::Binding<'a, routed_stage_host::Entry>>,
}

/// A streamed layer's CPU site request entry (`native::BankSite`).
pub(crate) struct BankSiteEntries<'a, G: GraphDraft + 'a> {
    pub request: G::Binding<'a, routed_site_request::Entry>,
}

/// A streamed layer's combine with the CPU's answer, by its routed form: its
/// projections (by row range), then their sum waiting for the answer
/// (`native::BankCombine`).
pub(crate) enum BankCombineEntry<'a, G: GraphDraft + 'a> {
    FusedInline(G::Ranged<'a, routed_output_streamed::Entry>),
    Fused {
        project: G::Ranged<'a, routed_output_project::Entry>,
        sited: G::Binding<'a, routed_output_sited::Entry>,
    },
    General {
        project: G::Ranged<'a, routed_down_project::Entry>,
        sited: G::Binding<'a, routed_down_sited::Entry>,
    },
}

impl<'a, G: GraphDraft + 'a> Copy for BankCombineEntry<'a, G> {}
impl<'a, G: GraphDraft + 'a> Clone for BankCombineEntry<'a, G> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<'a, G: GraphDraft + 'a> Copy for BankSiteEntries<'a, G> {}
impl<'a, G: GraphDraft + 'a> Clone for BankSiteEntries<'a, G> {
    fn clone(&self) -> Self {
        *self
    }
}

/// A prompt group's seeding of the bank by the device: each routed tensor's
/// copy on the seed list's grid, the grid, then the commit of each seeded
/// record; the host maps the seeds once the step has completed.
pub(crate) struct BankSeedingEntries<'a, G: GraphDraft + 'a> {
    pub seed: RecordTensors<G::Binding<'a, routed_copy::Entry>>,
    pub seed_commit: G::Binding<'a, routed_seed_commit::Entry>,
    pub list_grid: G::Binding<'a, routed_list_grid::Entry>,
}

impl<'a, G: GraphDraft + 'a> Copy for BankSeedingEntries<'a, G> {}
impl<'a, G: GraphDraft + 'a> Clone for BankSeedingEntries<'a, G> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<'a, G: GraphDraft + 'a> Copy for BankGraphEntries<'a, G> {}
impl<'a, G: GraphDraft + 'a> Clone for BankGraphEntries<'a, G> {
    fn clone(&self) -> Self {
        *self
    }
}

/// Whether `backend`'s `routed_down_project` kernel implements masked-write
/// skip semantics (`mode == 1` hit-only / `mode == 2` skip `code == -1`
/// without overwriting `published`). Only CUDA implements skip semantics
/// (`cuda/lib/routed/down.cuh`); Metal (`routed_down_project.metal:23`) and
/// Vulkan (`routed_down_project.comp:11`) unconditionally overwrite every
/// `(m, k)` cell and must NEVER seal or execute `RunStage::DecodeHits`.
pub(crate) fn routed_down_project_supports_masked_write(backend: seismic::BackendName) -> bool {
    matches!(backend, seismic::BackendName::Cuda)
}

impl<'a, G: GraphDraft + 'a> BankGraphEntries<'a, G> {
    /// Whether this layer's entries support split hit/miss down projection
    /// (`DecodeHits` + `DecodeBack` inside the unified streamed decode graph).
    pub(crate) fn can_split_hits(&self) -> bool {
        self.masked_down_project
            && self.gate_up.is_some()
            && self.site.is_some()
            && matches!(self.combine, Some(BankCombineEntry::General { .. }))
    }
}

impl<'a> BankGraphEntries<'a, NativeGraph> {
    pub(crate) fn for_backend(kernels: &'a ExpertBankKernels, backend: seismic::BackendName) -> Self {
        Self {
            route: match &kernels.route {
                BankRouteKernels::Fused { activation, route } => BankRoute::Fused {
                    activation: *activation,
                    route,
                },
                BankRouteKernels::General { activation, select } => BankRoute::General {
                    activation: *activation,
                    select,
                },
            },
            land: &kernels.land,
            step: &kernels.step,
            scratch_request: &kernels.scratch_request,
            round: &kernels.round,
            scratch_grids: &kernels.scratch_grids,
            pass_map: &kernels.pass_map,
            seeding: BankSeedingEntries {
                seed: kernels.placement.seed.as_ref(),
                seed_commit: &kernels.placement.seed_commit,
                list_grid: &kernels.placement.list_grid,
            },
            site: kernels.site.as_ref().map(|site| BankSiteEntries {
                request: &site.request,
            }),
            combine: kernels.combine.as_ref().map(|combine| match combine {
                crate::native::BankCombine::FusedInline(inline) => BankCombineEntry::FusedInline(inline),
                crate::native::BankCombine::Fused { project, sited } => BankCombineEntry::Fused { project, sited },
                crate::native::BankCombine::General { project, sited } => BankCombineEntry::General { project, sited },
            }),
            gate_up: kernels.gate_up.as_ref(),
            masked_down_project: routed_down_project_supports_masked_write(backend),
            stage_host: kernels.stage_host.as_ref(),
        }
    }
}

/// The route's element assignments by routed form.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum CheckedRoute {
    Fused {
        route: [(&'static str, Element); 6],
    },
    General {
        select: [(&'static str, Element); 4],
    },
}

/// The bank entries' element assignments, from a layer's feed-forward
/// binding (either routed form) and its bank's.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct CheckedBankEntries {
    route: CheckedRoute,
    /// `routed_land`'s: each routed tensor's stored and resident elements plus
    /// the activation element for inline site publication.
    land: [(&'static str, Element); 7],
    /// `routed_round`'s: each routed tensor's stored and resident elements.
    landing: [(&'static str, Element); 6],
    none: [(&'static str, Element); 0],
    /// The seed copy's.
    seed: RecordTensors<[(&'static str, Element); 1]>,
    /// The CPU site's request, where the layer has a site
    /// (`native::BankSite::served`).
    site: Option<CheckedSite>,
    /// The split decode combine (`native::BankCombine::served`).
    combine: Option<CheckedCombine>,
    /// The gate/up passes', of gated experts.
    gate_up: Option<[(&'static str, Element); 3]>,
    /// Whether the target backend's `routed_down_project` supports masked-write
    /// skip semantics (`routed_down_project_supports_masked_write`).
    masked_down_project: bool,
    /// Unified device-classified 3-tier decode staging (`routed_stage_host`).
    stage_host: Option<[(&'static str, Element); 3]>,
}

/// A CPU site's request element assignments.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct CheckedSite {
    request: [(&'static str, Element); 1],
}

/// A combine's projections' and sum's element assignments.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum CheckedCombine {
    FusedInline([(&'static str, Element); 3]),
    Fused([(&'static str, Element); 3], [(&'static str, Element); 1]),
    General([(&'static str, Element); 2], [(&'static str, Element); 2]),
}

impl CheckedBankEntries {
    /// Of a layer of feed-forward `slot` whose bank is `bank` on `backend`.
    pub(crate) fn new(
        slot: BankedFeedForward,
        bank: BankBinding,
        backend: seismic::BackendName,
    ) -> Result<Self, String> {
        let RecordTensors { gate, up, down } = bank.record;
        // What a landing launch converts from (staging, a round's conveyor)
        // and into: an up-only record's gate views are empty (`GV`, `GS`
        // 0), in its up's elements.
        let activation = match slot {
            BankedFeedForward::Routed(binding) => binding.activation,
            BankedFeedForward::GeneralRouted(binding)
            | BankedFeedForward::Parallel(crate::ParallelBinding { routed: binding, .. }) => binding.activation,
        };
        let land = [
            ("SG", gate.map_or(up.source, |gate| gate.source)),
            ("SU", up.source),
            ("SD", down.source),
            ("UG", gate.map_or(up.resident, |gate| gate.resident)),
            ("UU", up.resident),
            ("UD", down.resident),
            ("A", activation),
        ];
        let landing = [
            ("SG", gate.map_or(up.source, |gate| gate.source)),
            ("SU", up.source),
            ("SD", down.source),
            ("UG", gate.map_or(up.resident, |gate| gate.resident)),
            ("UU", up.resident),
            ("UD", down.resident),
        ];
        let general = |binding: crate::GeneralRoutedBinding| CheckedRoute::General {
            select: [
                ("NW", binding.norm),
                ("RNW", binding.router_norm),
                ("RW", binding.router),
                ("A", binding.activation),
            ],
        };
        let gate_up = match slot {
            BankedFeedForward::Routed(binding) => {
                Some([("EGW", binding.expert_gate), ("EUW", binding.expert_up), ("A", binding.activation)])
            }
            BankedFeedForward::GeneralRouted(binding) | BankedFeedForward::Parallel(crate::ParallelBinding { routed: binding, .. }) => {
                // An up-only layer has no passes; a gated one's gate
                // element is its resident gate/up's.
                let gate = binding.expert_gate.unwrap_or(binding.expert_up);
                binding.shape.experts_expansion.gated.then_some([("EGW", gate), ("EUW", binding.expert_up), ("A", binding.activation)])
            }
        };
        let route = match slot {
            BankedFeedForward::Routed(binding) => CheckedRoute::Fused {
                route: [
                    ("NW", binding.norm),
                    ("RW", binding.router),
                    ("SRW", Element::f32()),
                    ("SGW", binding.shared_gate),
                    ("SUW", binding.shared_up),
                    ("A", binding.activation),
                ],
            },
            BankedFeedForward::GeneralRouted(binding) => general(binding),
            BankedFeedForward::Parallel(binding) => general(binding.routed),
        };
        let site = crate::native::BankSite::served(slot, bank, backend)?.then(|| CheckedSite {
            request: [("A", activation)],
        });
        let combine = crate::native::BankCombine::served(slot, bank, backend).then(|| match slot {
            BankedFeedForward::Routed(binding) => {
                if <routed_output_streamed::Entry as seismic::EntryContract>::contract_node_shape(backend).is_ok() {
                    CheckedCombine::FusedInline([
                        ("EDW", binding.expert_down),
                        ("SDW", binding.shared_down),
                        ("A", binding.activation),
                    ])
                } else {
                    CheckedCombine::Fused(
                        [("EDW", binding.expert_down), ("SDW", binding.shared_down), ("A", binding.activation)],
                        [("A", binding.activation)],
                    )
                }
            }
            BankedFeedForward::GeneralRouted(binding) | BankedFeedForward::Parallel(crate::ParallelBinding { routed: binding, .. }) => {
                CheckedCombine::General(
                    [("EDW", binding.expert_down), ("A", binding.activation)],
                    [("A", binding.activation), ("R", super::graph::sum_element(&binding))],
                )
            }
        });
        let stage_host = (bank.tier > 0
            && <routed_stage_host::Entry as seismic::EntryContract>::contract_node_shape(backend).is_ok())
        .then(|| {
            [
                ("UG", gate.map_or(up.resident, |g| g.resident)),
                ("UU", up.resident),
                ("UD", down.resident),
            ]
        });
        Ok(Self {
            route,
            land,
            landing,
            none: [],
            seed: bank.record.map(|_, tensor| [("U", tensor.resident)]),
            site,
            combine,
            gate_up,
            masked_down_project: routed_down_project_supports_masked_write(backend),
            stage_host,
        })
    }

    pub(crate) fn entries<G: crate::programs::graph::draft::ElementBoundDraft>(
        &self,
    ) -> BankGraphEntries<'_, G> {
        BankGraphEntries {
            route: match &self.route {
                CheckedRoute::Fused { route } => BankRoute::Fused {
                    activation: route[5].1,
                    route: G::bind_ranged(&route[..]),
                },
                CheckedRoute::General { select } => BankRoute::General {
                    activation: select[3].1,
                    select: G::bind(&select[..]),
                },
            },
            land: G::bind(&self.land),
            step: G::bind(&self.none),
            scratch_request: G::bind(&self.none),
            round: G::bind(&self.landing),
            scratch_grids: G::bind(&self.none),
            pass_map: G::bind(&self.none),
            seeding: BankSeedingEntries {
                seed: self.seed.as_ref().map(|_, seed| G::bind(&seed[..])),
                seed_commit: G::bind(&self.none[..]),
                list_grid: G::bind(&self.none[..]),
            },
            site: self.site.as_ref().map(|site| BankSiteEntries {
                request: G::bind(&site.request[..]),
            }),
            combine: self.combine.as_ref().map(|combine| match combine {
                CheckedCombine::FusedInline(inline) => {
                    BankCombineEntry::FusedInline(G::bind_ranged(&inline[..]))
                }
                CheckedCombine::Fused(project, sited) => BankCombineEntry::Fused {
                    project: G::bind_ranged(&project[..]),
                    sited: G::bind(&sited[..]),
                },
                CheckedCombine::General(project, sited) => BankCombineEntry::General {
                    project: G::bind_ranged(&project[..]),
                    sited: G::bind(&sited[..]),
                },
            }),
            gate_up: self.gate_up.as_ref().map(|gate_up| G::bind_ranged(&gate_up[..])),
            masked_down_project: self.masked_down_project,
            stage_host: self.stage_host.as_ref().map(|sh| G::bind(&sh[..])),
        }
    }
}

/// Ports and timeline nodes declared by a unified CUDA 3-tier streamed decode
/// graph (`RunStage::Whole` with `routed_stage_host` + top-level `WAIT_VALUE_32`
/// + tail `WRITE_VALUE_32(completed_gen)`).
#[derive(Clone, Debug)]
pub(crate) struct UnifiedDecodePorts {
    pub front_sources: NativePort,
    pub back_sources: NativePort,
    pub ready_slot: NativePort,
    pub land_gate: NativeTimelinePort<seismic::timeline::Gate>,
    pub completed_gen: Option<NativeTimelinePort<seismic::timeline::Reached>>,
    /// `true` when `DecodeHits` runs before `land_gate` and `back_sources`
    /// masks VRAM hits to `-1` (general routed with `can_split_hits()`).
    pub mask_hits_back: bool,
}

/// A streamed layer's bank as its block's graph binds it: its entries, and
/// the bank ports and timeline nodes the graph declares.
pub(crate) struct BankGraph<'p, 'a, G: GraphDraft + 'a> {
    pub entries: BankGraphEntries<'a, G>,
    pub ports: &'p mut Vec<(BankPort, NativePort)>,
    /// The park node before each call that waits for the host, and the
    /// publication after each site request, each with its layer's scope and
    /// kind (`crate::experts::Parks`).
    pub parks: &'p mut Vec<(GraphTimeline, NativeTimelinePort<seismic::timeline::Parks>)>,
    /// Slot populated when this block seals the unified CUDA 3-tier streamed
    /// decode graph (`RunStage::Whole` on partial `Placer::Tier`).
    pub unified: Option<&'p mut Option<UnifiedDecodePorts>>,
}

/// How a routed layer's graph reads its experts: from the stacks a load
/// that holds them whole imported (`Held`); through the bank that streams
/// them (`Streamed`); or, a streamed layer's resident form
/// (`RunStage::Resident`), in place at its whole bank's home records, as a
/// held layer reads its stacks (`Home`).
pub(crate) enum LayerBank<'p, 'a, G: GraphDraft + 'a> {
    Held,
    Streamed(BankGraph<'p, 'a, G>),
    Home,
}

impl<'a, G: GraphDraft + 'a> LayerBank<'_, 'a, G> {
    /// The part of the planned expert stacks the layer's graph binds: the
    /// layer's home records in its resident form, else the planned values (a
    /// streamed layer's its group's records, `ResidentTarget::bound`).
    pub(crate) fn stacks(&self) -> WeightPart {
        match self {
            Self::Home => WeightPart::Home,
            Self::Held | Self::Streamed(_) => WeightPart::Values,
        }
    }

    /// Whether the layer's experts are its bank's.
    pub(crate) fn banked(&self) -> bool {
        !matches!(self, Self::Held)
    }
}

/// A landing launch (`routed_land`) of `args`, after the park node it
/// waits on, which it returns: the one place a landing launch gets its
/// park, for the load's layers and the assessment's points alike, so a
/// point cannot build the launch without it. Conditioned on
/// `policy[POLICY_RESOLVED]` being clear (`LaunchEncoding::WhileClear`): a
/// resolved layer can never miss a file read, so neither `routed_land` nor
/// its `Park` (signal + command-buffer cut + host wait on Metal) is encoded.
pub(crate) fn land_launch<'a, G: GraphDraft + 'a>(
    graph: &mut G,
    entry: G::Binding<'a, routed_land::Entry>,
    dimensions: &[(&str, impl Clone + Into<Dim>)],
    args: routed_land::WorkflowArgs<'_>,
    policy: &NativePort,
) -> Result<NativeTimelinePort<seismic::timeline::Parks>, GraphError> {
    let park = graph.timeline(seismic::timeline::Park);
    graph.enqueue_under(
        entry,
        dimensions,
        args,
        policy,
        magnitude_kernels::constants::POLICY_RESOLVED,
        seismic::LaunchEncoding::WhileClear,
    )?;
    Ok(park)
}

/// The bank that serves the routed layer of `scope`, and the layer's place
/// among its layers; none of a layer `load` keeps resident.
pub(crate) fn served(load: &ModelLoadPlan, scope: WeightScope) -> Option<(&ExpertBankPlan, usize)> {
    let bank = load.expert_bank()?;
    Some((bank, bank.layer(scope)?.0))
}

/// `XS`: the rows of the sources by which a routed layer's decode entries
/// read each choice ([`DecodeReads`]): of a layer its bank serves
/// (`streamed`, [`served`]), the step's source rows; of a resident layer
/// none, each expert read from its stacks by index.
pub(crate) fn source_rows(streamed: bool) -> u64 {
    match streamed {
        true => magnitude_kernels::constants::SOURCE_ROWS,
        false => 0,
    }
}

/// Whether a routed layer may decode in the Expand form: one that reads its
/// experts through its bank (`streamed`) does so in the route's own launch,
/// which only the shared-route form does.
pub(crate) fn expand_form(streamed: bool) -> Result<(), &'static str> {
    match streamed {
        true => Err("a streamed layer's decode needs the shared-route form"),
        false => Ok(()),
    }
}

/// How a routed layer's graph reads its experts, beside its numbers: from
/// the stacks a load holds whole (`Held`), through the bank that streams
/// them (`Streamed`, its bank's structure), or in its streamed layer's
/// resident form at its whole bank's home records (`Home`). The graph's
/// [`LayerBank`] binds the same form.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) enum BankForm {
    Held,
    Streamed(BankTopology),
    Home,
}

/// What decides a streamed layer's graph beside its numbers
/// ([`BankNumbers`]): whether it has a CPU site and sets its run's step,
/// whether its load fronts, the lead its route predicts at (`AHEAD`), what
/// each hint lead binds, the layer whose policy words its hints bind, who
/// places into its bank, and its records' resident and stored elements.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct BankTopology {
    pub sited: bool,
    pub sets_step: bool,
    /// Whether its load's decode passes run front rows
    /// (`ModelLoadPlan::fronts`): the predicate that gives a decode graph's
    /// step its guesses port ([`StreamedLayer::step`]), as it gives a group
    /// the kept guesses a host-token step binds there.
    pub fronts: bool,
    pub placer: crate::planning::Placer,
    /// Whether prefill copies host-tier experts into scratch via producer-
    /// enqueued hardware DMA (`ExpertBankPlan::tier_copy_dma`) rather than SM
    /// `routed_copy`.
    pub tier_copy_dma: bool,
    pub record: RecordTensors<Element>,
    pub stored: RecordTensors<Element>,
}

/// Every number a streamed layer's graph is built of
/// (`ExpertBankPlan::binding` and the bank's layout), in the bank plan's
/// names: `BANK`, `E`, the slot map's experts, `K`, `T`, a round's
/// `L`, the widest layer's experts, `V`, `TR`, the scratch rounds,
/// `FLIGHT` and `P`, its tables' slots and widths, `RING`, `GM`, the bank's
/// choices (a decode landing list's entries), the first seeded record, and
/// each routed tensor's `[N, K]`.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct BankNumbers<D = Dim> {
    pub records: D,
    /// Resident record count before the tail VRAM decode-stage scratch slots
    /// (`plan.records(group)`). `DecodeHits` reads `[0 .. stage_base)` while
    /// `routed_stage_host` writes `[stage_base .. records)` as disjoint slices
    /// so `DagOrder` forks them concurrently in the CUDA graph.
    pub stage_base: D,
    /// Tail VRAM scratch slots (`plan.decode_stage_slots()` = `16` on discrete
    /// `Placer::Tier`, `0` otherwise), equal to `records - stage_base`.
    pub stage_slots: D,
    pub experts: D,
    pub layer_experts: D,
    pub selected: D,
    pub staging: D,
    pub round_records: D,
    pub widest: D,
    pub window_slots: D,
    pub window_rounds: D,
    pub layer_window: D,
    pub conveyor: D,
    pub tier: D,
    pub rounds: D,
    pub flight: D,
    pub choices: D,
    pub file_width: D,
    pub file_slots: D,
    pub ring_words: D,
    pub site_slots: D,
    pub site_width: D,
    pub scratch_slots: D,
    pub scratch_lanes: D,
    pub map_words: D,
    pub seeded_from: D,
    pub matrix: RecordTensors<[D; 2]>,
}

impl<D> Numbers<D> for BankNumbers<D> {
    type Of<E> = BankNumbers<E>;
    fn map<E>(self, f: &mut impl FnMut(D) -> E) -> BankNumbers<E> {
        BankNumbers {
            records: f(self.records),
            stage_base: f(self.stage_base),
            stage_slots: f(self.stage_slots),
            experts: f(self.experts),
            layer_experts: f(self.layer_experts),
            selected: f(self.selected),
            staging: f(self.staging),
            round_records: f(self.round_records),
            widest: f(self.widest),
            window_slots: f(self.window_slots),
            window_rounds: f(self.window_rounds),
            layer_window: f(self.layer_window),
            conveyor: f(self.conveyor),
            tier: f(self.tier),
            rounds: f(self.rounds),
            flight: f(self.flight),
            choices: f(self.choices),
            file_width: f(self.file_width),
            file_slots: f(self.file_slots),
            ring_words: f(self.ring_words),
            site_slots: f(self.site_slots),
            site_width: f(self.site_width),
            scratch_slots: f(self.scratch_slots),
            scratch_lanes: f(self.scratch_lanes),
            map_words: f(self.map_words),
            seeded_from: f(self.seeded_from),
            matrix: self.matrix.map(|_, [n, k]| [f(n), f(k)]),
        }
    }
}

/// The form and numbers of the routed layer of `scope` whose graph reads
/// its experts as `bank` does: the only reading of the load's bank plan.
pub(crate) fn split<G: GraphDraft>(
    load: &ModelLoadPlan,
    scope: WeightScope,
    bank: &LayerBank<'_, '_, G>,
) -> Result<(BankForm, Option<BankNumbers<u64>>), GraphError> {
    let (plan, index) = match (served(load, scope), bank) {
        (Some(_), LayerBank::Home) => return Ok((BankForm::Home, None)),
        (None, LayerBank::Held) => return Ok((BankForm::Held, None)),
        (Some(served), LayerBank::Streamed(_)) => served,
        _ => return Err("a routed layer's bank entries disagree with its plan".into()),
    };
    let binding = plan.binding(index)?;
    let layer = &plan.layers()[index];
    let (words, file, site, scratch) =
        (binding.words, binding.words.file(), binding.words.site(), binding.words.scratch());
    let topology = BankTopology {
        sited: plan.sited(index),
        sets_step: plan.sets_step(index),
        // Only target decode passes carry front-row guesses into `routed_step`;
        // drafter layers (`DraftSublayer`, `HeadSublayer`) run with `front = 0`
        // and never bind a `guesses` port.
        fronts: load.fronts_target() && !layer.is_draft(),
        placer: binding.placer,
        tier_copy_dma: plan.tier_copy_dma(),
        record: binding.record.map(|_, tensor| tensor.resident),
        stored: binding.record.map(|_, tensor| tensor.source),
    };
    let numbers = BankNumbers {
        records: binding.records,
        stage_base: plan.records(layer.group),
        stage_slots: plan.decode_stage_slots(),
        experts: binding.experts,
        layer_experts: layer.experts,
        selected: binding.selected,
        staging: binding.staging,
        round_records: binding.round_records,
        widest: binding.widest,
        window_slots: plan.window_slots(),
        window_rounds: plan.window_rounds(),
        layer_window: plan.layer_window(index),
        conveyor: binding.conveyor,
        tier: binding.tier,
        rounds: words.rounds,
        flight: words.flight,
        choices: words.choices,
        file_width: file.width(),
        file_slots: file.slots(),
        ring_words: words.ring_words(),
        site_slots: site.slots(),
        site_width: site.width(),
        scratch_slots: scratch.slots(),
        scratch_lanes: scratch.lanes(),
        map_words: plan.group_experts(layer.group),
        seeded_from: plan.seeded_from(layer.group),
        matrix: binding.record.map(|_, tensor| tensor.matrix),
    };
    Ok((BankForm::Streamed(topology), Some(numbers)))
}

/// A routed layer the load streams: its bank's structure and numbers, its
/// bank's entries, and the bank ports and timeline nodes its graph declares.
pub(crate) struct StreamedLayer<'p, 'a, G: GraphDraft + 'a, W: PlannedWeights + ?Sized> {
    planned: &'p W,
    topology: &'p BankTopology,
    numbers: &'p BankNumbers,
    scope: WeightScope,
    entries: BankGraphEntries<'a, G>,
    ports: &'p mut Vec<(BankPort, NativePort)>,
    parks: &'p mut Vec<(GraphTimeline, NativeTimelinePort<seismic::timeline::Parks>)>,
    unified: Option<&'p mut Option<UnifiedDecodePorts>>,
}

/// A gated layer's resident gate/up, which its streamed decode gate/up
/// computes each choice by ([`StreamedLayer::gate_up`]): its arguments, at
/// its dimensions.
pub(crate) struct ResidentGateUp<'d, A> {
    pub args: A,
    pub dimensions: &'d [(&'static str, Dim)],
}

/// The extents a streamed layer's decode entries take beside its bank's.
pub(crate) struct DecodeShape<'d> {
    /// `M`: the launch's rows.
    pub rows: &'d Dim,
}

impl<'p, 'a, G: GraphDraft + 'a, W: PlannedWeights + ?Sized> StreamedLayer<'p, 'a, G, W> {
    /// The routed layer of `scope` of `form` with its `bank`'s graph, when
    /// its graph reads its experts through the bank that streams them; none
    /// of a layer the load holds whole, nor of a streamed layer's resident
    /// form.
    pub(crate) fn of(
        planned: &'p W,
        form: &'p BankForm,
        numbers: Option<&'p BankNumbers>,
        scope: WeightScope,
        bank: LayerBank<'p, 'a, G>,
    ) -> Result<Option<Self>, GraphError> {
        match (form, numbers, bank) {
            (BankForm::Home, None, LayerBank::Home) | (BankForm::Held, None, LayerBank::Held) => Ok(None),
            (BankForm::Streamed(topology), Some(numbers), LayerBank::Streamed(BankGraph { mut entries, ports, parks, unified })) => {
                // The plan alone says which layers have a CPU site
                // (`ExpertBankPlan::sited`): a layer encodes the load's site
                // entries exactly there, and a sited layer needs them.
                // General split combine (`routed_down_project` +
                // `routed_down_sited`) is kept even when unsited so UMA layers
                // overlap early-hit down projections with `routed_land`.
                entries.site = match (topology.sited, entries.site) {
                    (true, None) => return Err("a layer with a CPU site lacks its site entries".into()),
                    (sited, site) => site.filter(|_| sited),
                };
                if topology.sited && entries.combine.is_none() {
                    return Err("a layer with a CPU site lacks its combine entries".into());
                }
                entries.combine = entries.combine.filter(|combine| {
                    topology.sited || matches!(combine, BankCombineEntry::General { .. })
                });
                Ok(Some(Self { planned, topology, numbers, scope, entries, ports, parks, unified }))
            }
            _ => Err("a routed layer's bank entries disagree with its plan".into()),
        }
    }

    /// Its landing launch's threadgroups while it lands
    /// (`Placer::land_groups`).
    pub(crate) fn land_groups(&self) -> i32 {
        self.topology.placer.land_groups()
    }

    /// Whether the layer sets its run's step word
    /// (`ExpertBankPlan::sets_step`), a scratch-served run's in its front.
    pub(crate) fn sets_step(&self) -> bool {
        self.topology.sets_step
    }

    pub(crate) fn numbers(&self) -> &'p BankNumbers {
        self.numbers
    }

    /// The layer's decode route and resolve by its routed form.
    pub(crate) fn route(&self) -> BankRoute<'a, G> {
        self.entries.route
    }

    /// What the route's tier port binds of `ports` (`TM`, and the tier map,
    /// or a graph constant of no words without a host tier).
    pub(crate) fn route_tier(
        &mut self,
        graph: &mut G,
        ports: &DecodePorts,
        constants: &mut Vec<GraphConstant>,
    ) -> Result<RouteTier, GraphError> {
        Ok(match &ports.tier {
            Some(tier) => RouteTier { experts: self.numbers.layer_experts.clone(), map: tier.map.tensor().clone() },
            None => RouteTier {
                experts: Dim::from(0),
                map: GraphConstant::empty_of(graph, Element::i32(), &[0], constants)?,
            },
        })
    }

    /// Whether this layer is sealing the unified CUDA 3-tier streamed decode
    /// graph (`routed_stage_host` + top-level `WAIT_VALUE_32` + tail
    /// `WRITE_VALUE_32(completed_gen)`).
    pub(crate) fn is_unified_decode(&self) -> bool {
        self.unified.is_some()
            && self.entries.stage_host.is_some()
            && self.topology.placer == crate::planning::Placer::Tier
            && !self.numbers.tier.is_zero()
    }

    /// Whether this layer's entries support split hit/miss down projection
    /// (`DecodeHits` before `land_gate` and `DecodeBack` after `land_gate`).
    pub(crate) fn can_split_hits(&self) -> bool {
        self.entries.can_split_hits()
    }

    /// Record the unified decode ports and top-level `WAIT_VALUE_32` gate node
    /// into the block graph's `UnifiedDecodePorts` slot.
    pub(crate) fn record_unified_decode(
        &mut self,
        front_sources: NativePort,
        back_sources: NativePort,
        ready_slot: NativePort,
        land_gate: NativeTimelinePort<seismic::timeline::Gate>,
        mask_hits_back: bool,
    ) {
        if let Some(slot) = &mut self.unified {
            **slot = Some(UnifiedDecodePorts {
                front_sources,
                back_sources,
                ready_slot,
                land_gate,
                completed_gen: None,
                mask_hits_back,
            });
        }
    }

    /// Enqueue the unified 4-CTA x 256-thread `routed_stage_host` kernel that
    /// classifies `front_sources` into `VRAM_HIT` / `HOST_STAGE` / `HOST_WORK`,
    /// copies SM-staged host-tier experts into VRAM scratch `stage_base + s`,
    /// populates `back_sources`, and either self-gates `land_gate` (when
    /// `host_work == 0`, zero CPU wakeup) or publishes `ready_slot`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn enqueue_stage_host(
        &mut self,
        graph: &mut G,
        ports: &DecodePorts,
        rows: &Dim,
        front_sources: &WorkflowTensor,
        back_sources: &mut WorkflowTensor,
        ready_slot: &mut WorkflowTensor,
        records: RecordTensors<&mut NativePort>,
        constants: &mut Vec<GraphConstant>,
    ) -> Result<(), GraphError> {
        let entry = self
            .entries
            .stage_host
            .ok_or("enqueue_stage_host called without routed_stage_host entry")?;
        let tier = ports
            .tier
            .as_ref()
            .ok_or("enqueue_stage_host requires host tier records")?;
        let numbers = self.numbers;
        let [f_dim, h_dim] = numbers.matrix.up.clone();
        // Bind `routed_stage_host` to the disjoint tail VRAM scratch slice
        // `[stage_base .. records)` (`E = stage_slots`) so `DagOrder`
        // sees zero RAW/WAR conflict with `DecodeHits` reading `[0 .. stage_base)`
        // and forks them as parallel sibling branches before `WAIT_VALUE_32`.
        let (tier_gate, mut expert_gate_view, gv, gd) = match (&tier.records.gate, records.gate) {
            (Some(tg), Some(eg)) => (
                tg.tensor().clone(),
                eg.tensor().slice_leading(&numbers.stage_base, &numbers.records),
                numbers.tier.clone(),
                numbers.stage_slots.clone(),
            ),
            _ => {
                let empty_t = GraphConstant::empty_of(
                    graph,
                    self.topology.record.up,
                    &[Dim::from(0), f_dim.clone(), h_dim.clone()],
                    constants,
                )?;
                let empty_e = GraphConstant::empty_of(
                    graph,
                    self.topology.record.up,
                    &[Dim::from(0), f_dim.clone(), h_dim.clone()],
                    constants,
                )?
                .slice_leading(0, 0);
                (empty_t, empty_e, Dim::from(0), Dim::from(0))
            }
        };
        let mut expert_up_view = records
            .up
            .tensor()
            .slice_leading(&numbers.stage_base, &numbers.records);
        let mut expert_down_view = records
            .down
            .tensor()
            .slice_leading(&numbers.stage_base, &numbers.records);
        let dims = [
            ("M", rows.clone()),
            ("K", numbers.selected.clone()),
            ("GV", gv),
            ("TR", numbers.tier.clone()),
            ("GD", gd),
            ("E", numbers.stage_slots.clone()),
            ("F", f_dim),
            ("H", h_dim),
        ];
        graph.enqueue(
            entry,
            &dims,
            routed_stage_host::WorkflowArgs {
                front_sources: front_sources.into(),
                back_sources: back_sources.into(),
                policy: ports.policy.tensor().into(),
                step: ports.step.tensor().into(),
                land_grid: ports.land_grid.tensor().into(),
                ready_slot: ready_slot.into(),
                tier_gate: (&tier_gate).into(),
                tier_up: tier.records.up.tensor().into(),
                tier_down: tier.records.down.tensor().into(),
                expert_gate: (&mut expert_gate_view).into(),
                expert_up: (&mut expert_up_view).into(),
                expert_down: (&mut expert_down_view).into(),
            },
        )?;
        Ok(())
    }

    /// Construct [`DecodeReads`] over an explicit `sources` view (`front_sources`
    /// for `DecodeHits`, or `back_sources` for `DecodeBack`).
    pub(crate) fn reads_for_sources(
        &self,
        graph: &mut G,
        ports: &DecodePorts,
        sources: WorkflowTensorView,
        constants: &mut Vec<GraphConstant>,
    ) -> Result<DecodeReads, GraphError> {
        self.reads_inner(
            graph,
            ports.tier.as_ref().map(|tier| &tier.records),
            sources,
            Some(ports.policy.tensor().clone()),
            constants,
        )
    }

    /// The key records its decode route writes (`keys`): every row's key of
    /// each of the layer's experts, which placement ranks by.
    pub(crate) fn route_keys(&mut self, graph: &mut G) -> Result<NativePort, GraphError> {
        self.part(graph, BankPart::Keys)
    }

    /// Where a decode launch reads each choice ([`DecodeReads`]): the
    /// route's `sources` (the step's source rows), and the group's tier
    /// records, or without a tier graph constants of no records in their
    /// form (the bank's records are reserved, whose views bind as their
    /// reservation's).
    pub(crate) fn reads(
        &self,
        graph: &mut G,
        ports: &DecodePorts,
        locals: &RouteLocals,
        constants: &mut Vec<GraphConstant>,
    ) -> Result<DecodeReads, GraphError> {
        self.reads_inner(
            graph,
            ports.tier.as_ref().map(|tier| &tier.records),
            locals.snapshot(),
            Some(ports.policy.tensor().clone()),
            constants,
        )
    }

    /// Run `routed_gate_up_part` for a single decode pass (`late == 2` VRAM
    /// hits reading `front_sources`, `late == 3` sourced misses reading masked
    /// `back_sources`, or `late == 4` sourced non-hits reading unmasked
    /// `back_sources`) into `product` (allocating a new local if `into` is `None`).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn gate_up_part_mode<'x>(
        &mut self,
        graph: &mut G,
        args: routed_gate_up::WorkflowArgs<'x>,
        dimensions: &[(&str, Dim)],
        rows: &Dim,
        step: &WorkflowTensor,
        routing: &WorkflowTensor,
        late: i32,
        into: Option<WorkflowTensor>,
    ) -> Result<WorkflowTensor, GraphError> {
        let entry = G::pick(
            self.entries
                .gate_up
                .expect("a gated streamed layer's bank holds its gate/up passes (`bank_gate_up`)"),
            rows,
        );
        let numbers = self.numbers;
        let dimensions =
            [dimensions, &[("FLIGHT", numbers.flight.clone()), ("P", numbers.choices.clone())]].concat();
        let mut product_local;
        let mut product_existing;
        let product_ref: &mut WorkflowTensor = match into {
            Some(existing) => {
                product_existing = existing;
                &mut product_existing
            }
            None => {
                product_local = graph.local_for(entry, "product", &dimensions)?;
                product_local.tensor_mut()
            }
        };
        let routed_gate_up::WorkflowArgs {
            normalized,
            routes,
            sources,
            expert_gate,
            expert_up,
            tier_gate,
            tier_up,
            activation,
            gate_scale,
            up_scale,
            glu_limit,
        } = args;
        graph.enqueue(
            entry,
            &dimensions,
            routed_gate_up_part::WorkflowArgs {
                normalized,
                routes,
                sources,
                expert_gate,
                expert_up,
                tier_gate,
                tier_up,
                activation,
                gate_scale,
                up_scale,
                glu_limit,
                routing: routing.into(),
                step: step.into(),
                late: late.into(),
                product: product_ref.into(),
            },
        )?;
        Ok(product_ref.clone())
    }

    fn reads_inner(
        &self,
        graph: &mut G,
        tier_records: Option<&RecordTensors<NativePort>>,
        sources: WorkflowTensorView,
        policy: Option<WorkflowTensor>,
        constants: &mut Vec<GraphConstant>,
    ) -> Result<DecodeReads, GraphError> {
        let numbers = self.numbers;
        let (tier, count) = match tier_records {
            Some(records) => (
                records.as_ref().map(|_, port| port.tensor().slice_leading(0, &numbers.tier)),
                numbers.tier.clone(),
            ),
            None => (
                self.topology.record.zip(numbers.matrix.clone()).try_map(|_, (resident, [n, k])| {
                    Ok::<_, GraphError>(
                        GraphConstant::empty_of(graph, resident, &[Dim::from(0), n, k], constants)?.slice_leading(0, 0),
                    )
                })?,
                Dim::from(0),
            ),
        };
        Ok(DecodeReads { sources, rows: Dim::from(source_rows(true)), tier, records: count, policy })
    }
    /// The port of `part` of this layer's bank.
    pub(crate) fn part(&mut self, graph: &mut G, part: BankPart) -> Result<NativePort, GraphError> {
        bank_port(graph, self.planned, self.scope, part, self.ports)
    }

    /// The port of `part(kind)` of each routed tensor of the record.
    fn parts(
        &mut self,
        graph: &mut G,
        part: impl Fn(RoutedTensor) -> BankPart,
    ) -> Result<RecordTensors<NativePort>, GraphError> {
        let (planned, scope) = (self.planned, self.scope);
        let ports = &mut *self.ports;
        self.topology.record.try_map(|kind, _| bank_port(graph, planned, scope, part(kind), ports))
    }

    /// The run's step word (its chain place's), which every streamed entry
    /// of the run reads: set to the run's sequence (`routed_step`) where the
    /// layer sets it, with the sequence input the run writes and, of a
    /// decode graph of `rows` rows (`tokens`) of a load that fronts, the
    /// guesses its front rows run on, whose kinds the step words carry; a
    /// graph whose steps run no front row (a prompt's front, or any graph
    /// of a load that does not front) has none.
    pub(crate) fn step(
        &mut self,
        graph: &mut G,
        tokens: Option<&Dim>,
        constants: &mut Vec<GraphConstant>,
    ) -> Result<(NativePort, Option<StepInputs>), GraphError> {
        let entries = self.entries;
        let mut step = self.part(graph, BankPart::Step)?;
        if !self.sets_step() {
            return Ok((step, None));
        }
        let tokens = tokens.filter(|_| self.topology.fronts);
        // Every port of the entry and its launch are laid out at its one T:
        // the graph's token rows, none (T = 0) where it has none.
        let dimensions = [("T", tokens.cloned().unwrap_or_else(|| Dim::from(0)))];
        let sequence = graph.input_for(entries.step, "sequence", &dimensions)?;
        let guesses = match tokens {
            Some(_) => Some(graph.port_for(entries.step, "guesses", &dimensions)?),
            None => None,
        };
        let guessed = match &guesses {
            Some(guesses) => guesses.tensor().clone(),
            None => GraphConstant::empty_of(graph, Element::i32(), &[Dim::from(0), Dim::from(2)], constants)?,
        };
        graph.enqueue(
            entries.step,
            &dimensions,
            routed_step::WorkflowArgs {
                step: step.tensor_mut().into(),
                sequence: sequence.tensor().into(),
                guesses: (&guessed).into(),
            },
        )?;
        Ok((step, Some(StepInputs { sequence, guesses })))
    }

    /// The park node before this layer's call that waits for the host, of
    /// `kind`, which each run binds (`crate::experts::Parks`).
    fn park(&mut self, graph: &mut G, kind: ParkKind) {
        let port = graph.timeline(seismic::timeline::Park);
        self.parked(kind, port);
    }

    /// `port`, a park node before this layer's call of `kind`, for each run
    /// to bind.
    fn parked(&mut self, kind: ParkKind, port: NativeTimelinePort<seismic::timeline::Parks>) {
        self.parks.push((GraphTimeline::Park(GraphPark { scope: self.scope, kind }), port));
    }

    /// The publication of this layer's site request, the call enqueued last,
    /// whose writes the CPU pool reads while the device goes on
    /// (`seismic::timeline::Publish`): a submission boundary where parks are,
    /// nothing where kernels wait in place.
    fn publish(&mut self, graph: &mut G) {
        let port = graph.timeline(seismic::timeline::Publish);
        self.parks.push((GraphTimeline::Publish(self.scope), port));
    }

    /// The decode landing launch of the choices the route's file request
    /// names absent (`routed_land`), after its park. Where the bank places,
    /// its threadgroups convert each absent expert's record from staging
    /// into the bank record the stacks read as its read lands, in landing
    /// order, each parked only on the host's count of the layer's landing
    /// list, until the step's file slot is answered with every entry; then
    /// it writes each absent choice's sources from the slot's answer, where
    /// the route assigned the expert: the CPU's, or the GPU's in place.
    /// Where the host tier places, it converts nothing (its staging and
    /// records views of none) and only sources. It declares the slot and
    /// the list host-written, so the experts after it read what it wrote on
    /// every backend. It launches on the grid the route wrote (`land_grid`):
    /// none where its step requested nothing.
    pub(crate) fn land(
        &mut self,
        graph: &mut G,
        ports: LandPorts<'_>,
        site_choices: Option<&StepChoices<'_>>,
        shape: &DecodeShape,
        constants: &mut Vec<GraphConstant>,
    ) -> Result<(), GraphError> {
        let LandPorts { decode: DecodePorts { step, policy, file, land_grid, site, .. }, sources, records } = ports;
        let inline_site = site
            .as_mut()
            .zip(site_choices)
            .map(|(site, choices)| (&mut site.table, &mut site.rows, choices));
        self.land_inner(graph, step, policy, file, land_grid, sources, records, inline_site, shape, constants)
    }

    /// Publishes a site request already written by `routed_route_streamed` or
    /// `routed_land` where parks are submission boundaries (Vulkan); a no-op
    /// on backends that park in kernels (CUDA, Metal) or unsited layers.
    pub(crate) fn publish_site(&mut self, graph: &mut G) {
        if self.entries.site.is_some() {
            self.publish(graph);
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn land_inner(
        &mut self,
        graph: &mut G,
        step: &NativePort,
        policy: &NativePort,
        file: &mut NativePort,
        land_grid: &NativePort,
        sources: &mut WorkflowTensor,
        records: RecordTensors<&mut NativePort>,
        inline_site: Option<(&mut NativePort, &mut NativePort, &StepChoices<'_>)>,
        shape: &DecodeShape,
        constants: &mut Vec<GraphConstant>,
    ) -> Result<(), GraphError> {
        let numbers = self.numbers;
        let mut list = self.part(graph, BankPart::LandingList)?;
        // Where the bank places, its staging and the bank's records; where
        // the tier places, views of none of either (`V`, `S` 0): graph
        // constants of no records in the stored form, and the stacks'.
        let (from, into) = match self.topology.placer {
            crate::planning::Placer::Bank => (numbers.staging.clone(), numbers.records.clone()),
            crate::planning::Placer::Tier => (Dim::from(0), Dim::from(0)),
        };
        let staging = match self.topology.placer {
            crate::planning::Placer::Bank => {
                self.parts(graph, |kind| BankPart::Staging { kind })?.map(|_, port| port.tensor().slice_leading(0, &from))
            }
            crate::planning::Placer::Tier => self.topology.stored.zip(numbers.matrix.clone()).try_map(|_, (stored, [n, k])| {
                Ok::<_, GraphError>(GraphConstant::empty_of(graph, stored, &[Dim::from(0), n, k], constants)?.slice_leading(0, 0))
            })?,
        };
        let mut destination = records.map(|_, port| port.tensor().slice_leading(0, &into));
        let gated = |records: &Dim| match self.topology.record.gate {
            Some(_) => records.clone(),
            None => Dim::from(0),
        };
        let [features, hidden] = numbers.matrix.up.clone();
        let activation = match self.entries.route {
            BankRoute::Fused { activation, .. } | BankRoute::General { activation, .. } => activation,
        };
        let (sm, sf, sr) = match &inline_site {
            Some(_) => (shape.rows.clone(), numbers.site_slots.clone(), numbers.site_width.clone()),
            None => (Dim::from(0), Dim::from(0), Dim::from(0)),
        };
        let empty_routes = inline_site
            .is_none()
            .then(|| GraphConstant::empty_of(graph, Element::i32(), &[Dim::from(0), numbers.selected.clone()], constants))
            .transpose()?;
        let empty_weights = inline_site
            .is_none()
            .then(|| GraphConstant::empty_of(graph, Element::f32(), &[Dim::from(0), numbers.selected.clone()], constants))
            .transpose()?;
        let empty_norm = inline_site
            .is_none()
            .then(|| GraphConstant::empty_of(graph, activation, &[Dim::from(0), hidden.clone()], constants))
            .transpose()?;
        let mut empty_site = inline_site
            .is_none()
            .then(|| GraphConstant::empty_port(graph, Element::i32(), &[Dim::from(0), Dim::from(0)], constants))
            .transpose()?;
        let mut empty_site_rows = inline_site
            .is_none()
            .then(|| {
                GraphConstant::empty_port(
                    graph,
                    Element::f32(),
                    &[Dim::from(0), Dim::from(magnitude_kernels::constants::VERIFY_ROWS), hidden.clone()],
                    constants,
                )
            })
            .transpose()?;
        let dimensions = [
            ("M", shape.rows.clone()),
            ("K", numbers.selected.clone()),
            ("FW", numbers.file_width.clone()),
            ("FLIGHT", numbers.file_slots.clone()),
            ("L", numbers.choices.clone()),
            ("V", from.clone()),
            ("GV", gated(&from)),
            ("S", into.clone()),
            ("GS", gated(&into)),
            ("F", features),
            ("H", hidden),
            ("SM", sm),
            ("SF", sf),
            ("SR", sr),
        ];
        let staging_gate = match &staging.gate {
            Some(gate) => gate.clone(),
            None => staging.up.slice_leading(0, 0),
        };
        let mut gate = match destination.gate.take() {
            Some(gate) => gate,
            None => destination.up.slice_leading(0, 0),
        };
        let (routes_ref, weights_ref, norm_ref, site_ref, site_rows_ref) = match inline_site {
            Some((site_table, site_rows, choices)) => (
                choices.routes.into(),
                choices.weights.into(),
                choices.input.into(),
                site_table.tensor_mut().into(),
                site_rows.tensor_mut().into(),
            ),
            None => (
                empty_routes.as_ref().unwrap().into(),
                empty_weights.as_ref().unwrap().into(),
                empty_norm.as_ref().unwrap().into(),
                empty_site.as_mut().unwrap().tensor_mut().into(),
                empty_site_rows.as_mut().unwrap().tensor_mut().into(),
            ),
        };
        let park = land_launch(
            graph,
            self.entries.land,
            &dimensions,
            routed_land::WorkflowArgs {
                step: step.tensor().into(),
                file: file.tensor_mut().into(),
                sources: (&mut *sources).into(),
                policy: policy.tensor().into(),
                landing: list.tensor_mut().into(),
                staging_gate: (&staging_gate).into(),
                staging_up: (&staging.up).into(),
                staging_down: (&staging.down).into(),
                gate: (&mut gate).into(),
                up: (&mut destination.up).into(),
                down: (&mut destination.down).into(),
                grid: land_grid.tensor().into(),
                routes: routes_ref,
                weights: weights_ref,
                normalized: norm_ref,
                site: site_ref,
                site_rows: site_rows_ref,
            },
            policy,
        )?;
        self.parked(ParkKind::Land, park);
        Ok(())
    }

    /// A decode launch's site request, where the layer has a site
    /// (`routed_site_request`): after its route and its landing launch, before
    /// its gate/up, the step's CPU choices of the route's snapshot
    /// (`sources`), their experts (`routes`) and weights (`weights`, as its
    /// combine weighs them) and the rows its experts project (`normalized`),
    /// into the step's slot of its site table, which the CPU pool polls. A
    /// step with no CPU choice writes nothing. Encoded while the layer's group is not
    /// whole (`policy[POLICY_WHOLE]` clear): a whole group misses nothing.
    /// Its publication follows it: where parks are submission boundaries,
    /// the request is the host's only once a submission ends, so one ends
    /// right after it (taking a value only while the request launches), and
    /// the pool computes while the device runs its own choices.
    pub(crate) fn site_request(
        &mut self,
        graph: &mut G,
        ports: &mut DecodePorts,
        routes: &WorkflowTensor,
        weights: &WorkflowTensor,
        sources: &WorkflowTensorView,
        normalized: &WorkflowTensor,
        shape: &DecodeShape,
    ) -> Result<(), GraphError> {
        let (Some(entries), Some(site)) = (self.entries.site, &mut ports.site) else {
            return Ok(());
        };
        self.site_request_inner(
            graph,
            entries,
            &ports.step,
            &ports.policy,
            &mut site.table,
            &mut site.rows,
            routes,
            weights,
            sources,
            normalized,
            shape,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn site_request_inner(
        &mut self,
        graph: &mut G,
        entries: BankSiteEntries<'a, G>,
        step: &NativePort,
        policy: &NativePort,
        site_table: &mut NativePort,
        site_rows: &mut NativePort,
        routes: &WorkflowTensor,
        weights: &WorkflowTensor,
        sources: &WorkflowTensorView,
        normalized: &WorkflowTensor,
        shape: &DecodeShape,
    ) -> Result<(), GraphError> {
        let numbers = self.numbers;
        graph.enqueue_under(
            entries.request,
            &[
                ("M", shape.rows.clone()),
                ("H", numbers.matrix.down[0].clone()),
                ("K", numbers.selected.clone()),
                ("FLIGHT", numbers.site_slots.clone()),
                ("R", numbers.site_width.clone()),
            ],
            routed_site_request::WorkflowArgs {
                step: step.tensor().into(),
                routes: routes.into(),
                sources: sources.into(),
                weights: weights.into(),
                normalized: normalized.into(),
                site: site_table.tensor_mut().into(),
                site_rows: site_rows.tensor_mut().into(),
            },
            policy,
            magnitude_kernels::constants::POLICY_WHOLE,
            seismic::LaunchEncoding::WhileClear,
        )?;
        self.publish(graph);
        Ok(())
    }

    /// The decode launch's landing of the choices its route left absent
    /// ([`Self::land`]), which converts the device's misses into `records`
    /// as their reads land and sources every absent choice into the step's
    /// sources, then its site request of `choices` ([`Self::site_request`]).
    /// Every entry after it may read any choice.
    pub(crate) fn settle(
        &mut self,
        graph: &mut G,
        step: &mut RoutedStep,
        records: RecordTensors<&mut NativePort>,
        choices: &StepChoices<'_>,
        constants: &mut Vec<GraphConstant>,
    ) -> Result<(), GraphError> {
        let shape = DecodeShape { rows: choices.rows };
        let mut sources = step.locals.sources.tensor().clone();
        let inline_site = matches!(self.entries.route, BankRoute::Fused { .. });
        self.land(
            graph,
            LandPorts { decode: &mut step.ports, sources: &mut sources, records },
            inline_site.then_some(choices),
            &shape,
            constants,
        )?;
        if inline_site {
            self.publish_site(graph);
            Ok(())
        } else {
            let snapshot = step.locals.snapshot();
            self.site_request(graph, &mut step.ports, choices.routes, choices.weights, &snapshot, choices.input, &shape)
        }
    }

    /// A gated layer's decode gate/up around its landing ([`Self::settle`]), in
    /// two passes of `routed_gate_up_part` over one product, which it
    /// returns: the early pass before the landing, the first writer of the
    /// product, computes the choices the route sourced, so they run while
    /// the host reads the absent ones; the late pass after the landing
    /// computes the choices the landing sourced, encoded only while
    /// the layer's group is not resolved at its route
    /// (`policy[POLICY_RESOLVED]` clear: a resolved group has none; this
    /// condition is that rule's one owner). Each takes its resident
    /// gate/up's `args` at its `dimensions`, and the step's routing record
    /// and step. Each choice's product is computed once, by the resident
    /// gate/up's arithmetic, so the product is its one pass's whatever the
    /// bank held.
    pub(crate) fn gate_up<'x>(
        &mut self,
        graph: &mut G,
        step: &mut RoutedStep,
        ResidentGateUp { args, dimensions }: ResidentGateUp<'_, impl Fn() -> routed_gate_up::WorkflowArgs<'x>>,
        records: RecordTensors<&mut NativePort>,
        choices: &StepChoices<'_>,
        constants: &mut Vec<GraphConstant>,
    ) -> Result<WorkflowTensor, GraphError> {
        let entry = G::pick(self.entries.gate_up.expect("a gated streamed layer's bank holds its gate/up passes (`bank_gate_up`)"), choices.rows);
        let numbers = self.numbers;
        let dimensions =
            [dimensions, &[("FLIGHT", numbers.flight.clone()), ("P", numbers.choices.clone())]].concat();
        let mut product = graph.local_for(entry, "product", &dimensions)?;
        let pass = |graph: &mut G, step: &RoutedStep, late: bool, product: &mut NativePort| {
            let routed_gate_up::WorkflowArgs {
                normalized,
                routes,
                sources,
                expert_gate,
                expert_up,
                tier_gate,
                tier_up,
                activation,
                gate_scale,
                up_scale,
                glu_limit,
            } = args();
            let args = routed_gate_up_part::WorkflowArgs {
                normalized,
                routes,
                sources,
                expert_gate,
                expert_up,
                tier_gate,
                tier_up,
                activation,
                gate_scale,
                up_scale,
                glu_limit,
                routing: step.routing.tensor().into(),
                step: step.ports.step.tensor().into(),
                late: i32::from(late).into(),
                product: product.tensor_mut().into(),
            };
            match late {
                false => graph.enqueue(entry, &dimensions, args).map(drop),
                true => graph
                    .enqueue_under(
                        entry,
                        &dimensions,
                        args,
                        &step.ports.policy,
                        magnitude_kernels::constants::POLICY_RESOLVED,
                        seismic::LaunchEncoding::WhileClear,
                    )
                    .map(drop),
            }
        };
        pass(graph, step, false, &mut product)?;
        self.settle(graph, step, records, choices, constants)?;
        pass(graph, step, true, &mut product)?;
        Ok(product.tensor().clone())
    }

    /// A gated general layer's decode `gate_up` AND `down_project` around its
    /// landing ([`Self::settle`]), overlapping both hits' gate/up AND hits'
    /// down projections with the host's SSD read:
    /// `routed_gate_up_part(late=0)` ->
    /// `routed_down_project(mode=0) [WhileClear(combine_policy)]` ->
    /// `self.settle` ->
    /// `routed_gate_up_part(late=1) [WhileClear(POLICY_RESOLVED)]` ->
    /// `routed_down_project(mode=3) [WhileClear(POLICY_RESOLVED)]`.
    /// On `POLICY_RESOLVED` (unsited) or `POLICY_WHOLE` (sited) layers, zero
    /// split-down launches are encoded and the caller runs `routed_down` alone.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn gate_up_and_down_project<'x>(
        &mut self,
        graph: &mut G,
        step: &mut RoutedStep,
        ResidentGateUp { args, dimensions }: ResidentGateUp<'_, impl Fn() -> routed_gate_up::WorkflowArgs<'x>>,
        down_dimensions: &[(&str, Dim)],
        expert_down: &WorkflowTensor,
        tier_down: &WorkflowTensorView,
        sources: &WorkflowTensorView,
        records: RecordTensors<&mut NativePort>,
        choices: &StepChoices<'_>,
        constants: &mut Vec<GraphConstant>,
    ) -> Result<(WorkflowTensor, WorkflowTensor), GraphError> {
        let gate_up_entry = G::pick(
            self.entries
                .gate_up
                .expect("a gated streamed layer's bank holds its gate/up passes (`bank_gate_up`)"),
            choices.rows,
        );
        let Some(BankCombineEntry::General { project, .. }) = self.entries.combine else {
            return Err("gate_up_and_down_project requires general combine entries".into());
        };
        let project_entry = G::pick(project, choices.rows);
        let numbers = self.numbers;
        let routing_dims = [("FLIGHT", numbers.flight.clone()), ("P", numbers.choices.clone())];
        let gate_up_dims = [dimensions, &routing_dims].concat();
        let down_dims = [down_dimensions, &routing_dims].concat();
        let mut product = graph.local_for(gate_up_entry, "product", &gate_up_dims)?;
        let mut published = graph.local_for(project_entry, "published", &down_dims)?;
        let combine_policy = if step.ports.site.is_some() {
            magnitude_kernels::constants::POLICY_WHOLE
        } else {
            magnitude_kernels::constants::POLICY_RESOLVED
        };
        let gate_up_pass = |graph: &mut G, step: &RoutedStep, late: bool, product: &mut NativePort| {
            let routed_gate_up::WorkflowArgs {
                normalized,
                routes,
                sources,
                expert_gate,
                expert_up,
                tier_gate,
                tier_up,
                activation,
                gate_scale,
                up_scale,
                glu_limit,
            } = args();
            let args = routed_gate_up_part::WorkflowArgs {
                normalized,
                routes,
                sources,
                expert_gate,
                expert_up,
                tier_gate,
                tier_up,
                activation,
                gate_scale,
                up_scale,
                glu_limit,
                routing: step.routing.tensor().into(),
                step: step.ports.step.tensor().into(),
                late: i32::from(late).into(),
                product: product.tensor_mut().into(),
            };
            match late {
                false => graph.enqueue(gate_up_entry, &gate_up_dims, args).map(drop),
                true => graph
                    .enqueue_under(
                        gate_up_entry,
                        &gate_up_dims,
                        args,
                        &step.ports.policy,
                        magnitude_kernels::constants::POLICY_RESOLVED,
                        seismic::LaunchEncoding::WhileClear,
                    )
                    .map(drop),
            }
        };
        let down_pass = |graph: &mut G,
                         step: &RoutedStep,
                         mode: i32,
                         policy_word: u64,
                         product: &NativePort,
                         published: &mut NativePort| {
            graph
                .enqueue_under(
                    project_entry,
                    &down_dims,
                    routed_down_project::WorkflowArgs {
                        product: product.tensor().into(),
                        routes: choices.routes.into(),
                        sources: sources.into(),
                        expert_down: expert_down.into(),
                        tier_down: tier_down.into(),
                        routing: step.routing.tensor().into(),
                        step: step.ports.step.tensor().into(),
                        mode: mode.into(),
                        published: published.tensor_mut().into(),
                    },
                    &step.ports.policy,
                    policy_word,
                    seismic::LaunchEncoding::WhileClear,
                )
                .map(drop)
        };
        gate_up_pass(graph, step, false, &mut product)?;
        down_pass(graph, step, 0, combine_policy, &product, &mut published)?;
        self.settle(graph, step, records, choices, constants)?;
        gate_up_pass(graph, step, true, &mut product)?;
        down_pass(
            graph,
            step,
            3,
            magnitude_kernels::constants::POLICY_RESOLVED,
            &product,
            &mut published,
        )?;
        Ok((product.tensor().clone(), published.tensor().clone()))
    }

    /// The layer's combine with the CPU's answer and the dimensions of its
    /// site table (`FLIGHT`, `SW`), where the layer has a site: its
    /// projections, then its sum, which waits for the answer when a choice
    /// of the step is the CPU's.
    pub(crate) fn site_combine(&self) -> Option<(BankCombineEntry<'a, G>, [(&'static str, Dim); 2])> {
        self.entries.site?;
        let combine = self.entries.combine?;
        let numbers = self.numbers;
        Some((combine, [("FLIGHT", numbers.site_slots.clone()), ("SW", numbers.site_width.clone())]))
    }

    /// The layer's general split combine (`routed_down_project` +
    /// `routed_down_sited`), available on any streamed general layer whose
    /// backend declares both entries (including unsited UMA layers).
    pub(crate) fn general_combine(
        &self,
    ) -> Option<(G::Ranged<'a, routed_down_project::Entry>, G::Binding<'a, routed_down_sited::Entry>)> {
        match self.entries.combine {
            Some(BankCombineEntry::General { project, sited }) => Some((project, sited)),
            _ => None,
        }
    }

    /// The site's park, between the combine's projections and its sum
    /// (`ParkKind::Site`): the device's projections run before it, beside
    /// the pool.
    pub(crate) fn park_site(&mut self, graph: &mut G) {
        self.park(graph, ParkKind::Site);
    }

    /// A scratch-served launch's front's request of the experts its `rows`
    /// rows' `routes` name, from its chain place's flags, in its step's slot
    /// of the layer's scratch table, of the run's step (`step`): the step
    /// inputs the run writes where the front sets it.
    pub(crate) fn request(
        &mut self,
        graph: &mut G,
        routes: &WorkflowTensor,
        rows: &Dim,
        constants: &mut Vec<GraphConstant>,
    ) -> Result<Option<StepInputs>, GraphError> {
        let entries = self.entries;
        let numbers = self.numbers;
        // The flags are the widest layer's (`BankPart::ScratchFlags`).
        let dimensions = [
            ("M", rows.clone()),
            ("K", numbers.selected.clone()),
            ("E", numbers.widest.clone()),
            ("FLIGHT", numbers.scratch_slots.clone()),
            ("LANES", numbers.scratch_lanes.clone()),
        ];
        let (step, inputs) = self.step(graph, None, constants)?;
        let mut slots = self.part(graph, BankPart::ScratchTable)?;
        let mut flags = self.part(graph, BankPart::ScratchFlags)?;
        let resident = self.part(graph, BankPart::Resident)?;
        graph.enqueue(
            entries.scratch_request,
            &dimensions,
            routed_scratch_request::WorkflowArgs {
                routes: routes.into(),
                step: step.tensor().into(),
                flags: flags.tensor_mut().into(),
                scratch: slots.tensor_mut().into(),
                resident: resident.tensor().into(),
            },
        )?;
        Ok(inputs)
    }

    /// Where a merge set's pass reads the layer's experts: its group's
    /// records, at the records its map holds for the step (every expert
    /// while its group is whole), and the scratch tier, slot `e` holding
    /// cold expert `e` (its steps' fills: `fill_scratch`). The layer's map
    /// rows and the step, resolved into device memory before the pass
    /// (`routed_pass_map`), which its expert entries read; the tier and the
    /// group's records; and where the pass's grouping counts each expert's
    /// choices, its group's reads (`BankPart::GroupReads`).
    pub(crate) fn pass_sources(
        &mut self,
        graph: &mut G,
    ) -> Result<(WorkflowTensor, WorkflowTensor, RecordTensors<NativePort>, Dim, NativePort, Dim), GraphError> {
        let map = self.part(graph, BankPart::SlotMap)?;
        let window = self.part(graph, BankPart::WindowTable)?;
        let step = self.part(graph, BankPart::Step)?;
        let entry = self.entries.pass_map;
        let dimensions = [("E", self.numbers.experts.clone())];
        let mut resolved = graph.local_for(entry, "resolved", &dimensions)?;
        let mut pass_step = graph.local_for(entry, "pass_step", &dimensions)?;
        graph.enqueue(
            entry,
            &dimensions,
            routed_pass_map::WorkflowArgs {
                map: map.tensor().into(),
                window: window.tensor().into(),
                step: step.tensor().into(),
                resolved: resolved.tensor_mut().into(),
                pass_step: pass_step.tensor_mut().into(),
            },
        )?;
        Ok((
            resolved.tensor().clone(),
            pass_step.tensor().clone(),
            self.parts(graph, |kind| BankPart::Scratch { kind })?,
            self.numbers.records.clone(),
            self.part(graph, BankPart::GroupReads)?,
            self.numbers.layer_window.clone(),
        ))
    }

    /// A scratch-served step's fill of the scratch tier with the cold
    /// experts its front requested (`OnceStage::Fill`): each round's landing
    /// launch, which arrives at the round and converts each conveyed record
    /// as its read lands, beside a host tier the copy of the experts it
    /// keeps whole after round 0, then a prompt group's seed copy, each
    /// reading the step word its front set or read. Every datum the host
    /// serves a round with is read by these launches, after the round's
    /// park in this graph, from host memory in place (`routed_round`): no
    /// answer needs a submission to the device. Its merge set's pass
    /// then reads each expert where the layer's map holds it for the step
    /// (a hit, in its group's records) or the scratch tier (slot `e` holding
    /// cold expert `e`: `pass_sources`); the host maps the seeds once their
    /// copy has completed. It writes no input, so its run holds no upload
    /// region. Its last write: the seed commit's commits, or where nothing
    /// seeds the scratch tier it fills.
    pub(crate) fn fill_scratch(
        &mut self,
        graph: &mut G,
        weights: &mut Vec<(WeightPort, NativePort)>,
        constants: &mut Vec<GraphConstant>,
    ) -> Result<WorkflowTensor, GraphError> {
        let entries = self.entries;
        let numbers = self.numbers;
        let experts = &numbers.experts;
        let (scratch_slots, scratch_lanes) = (&numbers.scratch_slots, &numbers.scratch_lanes);
        let [features, hidden] = &numbers.matrix.up;
        // An up-only record's gate conveyor and scratch are empty (`GV`,
        // `GS` 0), views of its up's.
        let gated = |records: &Dim| match self.topology.record.gate {
            Some(_) => records.clone(),
            None => Dim::from(0),
        };
        // A round's landing launch's dimensions.
        let landing = [
            ("FLIGHT", scratch_slots.clone()),
            ("LANES", scratch_lanes.clone()),
            ("L", numbers.round_records.clone()),
            ("V", numbers.conveyor.clone()),
            ("GV", gated(&numbers.conveyor)),
            ("S", numbers.layer_window.clone()),
            ("GS", gated(&numbers.layer_window)),
            ("F", features.clone()),
            ("H", hidden.clone()),
            ("E", numbers.window_slots.clone()),
            ("EL", experts.clone()),
        ];
        let step = self.part(graph, BankPart::Step)?;
        let window = self.part(graph, BankPart::Window)?;
        let mut slots = self.part(graph, BankPart::ScratchTable)?;
        let mut scratch = self.parts(graph, |kind| BankPart::Scratch { kind })?;
        let conveyor = self.parts(graph, |kind| BankPart::Conveyor { kind })?;
        let seeds = self.part(graph, BankPart::SeedList)?;
        let tiered = self.part(graph, BankPart::TierList)?;
        // Beside a host tier, its records, which the tier copy reads in
        // place, and the grid of the tier list it launches on and the list
        // staged in device memory it reads. When `tier_copy_dma` is enabled
        // (CUDA with a linked host tier), host-tier -> scratch copies run
        // via producer-enqueued hardware DMA (`cuMemcpyHtoDAsync`) on the fill
        // stream, so no SM zero-copy kernel is emitted.
        let mut tier = match numbers.tier.is_zero() || self.topology.tier_copy_dma {
            true => None,
            false => Some((
                self.parts(graph, |kind| BankPart::Tier { kind })?,
                self.part(graph, BankPart::TierGrid)?,
                self.part(graph, BankPart::TierStaged)?,
            )),
        };
        let (mut seed_grid, mut seed_staged) = (self.part(graph, BankPart::SeedGrid)?, self.part(graph, BankPart::SeedStaged)?);
        // Round 0 always launches (hit against cold is final only as the
        // request is served); its answer carries the count of records the
        // rounds convert, from which `routed_scratch_grids` writes every
        // round's park grid: a later round launches while it has records,
        // or the round before has and no sibling fill follows (the arrival
        // after the last round frees its records,
        // `ExpertBank::fill_followed`).
        let mut grids = self.part(graph, BankPart::ScratchGrids)?;
        let always = GraphConstant::i32(graph, &[i32::try_from(magnitude_kernels::constants::LAND_GROUPS).map_err(|_| "landing groups exceed i32")?, 1, 1])?;
        let grid = |grids: &WorkflowTensor, round: u64| grids.slice_leading(round, round + 1).reshape(&[3]);
        // The launch reads the conveyor as the host's: the host reads each
        // listed expert into its record before it appends its entry.
        let conveyor_gate = GateRecords::of(&conveyor);
        // The layer's map, which the host rewrites as it serves the request
        // (its hits final and the lent records' experts unmapped by round 0's
        // answer): each park is its writer, so the pass's map
        // (`routed_pass_map`) reads it only after the park, on every backend,
        // not only where launches keep submission order.
        let map = self.part(graph, BankPart::SlotMap)?;
        // The rounds repeat the topology: a trace is its rounds'.
        let window_rounds = numbers.window_rounds.word().ok_or("window rounds exceed i32")?;
        let rounds = numbers.window_rounds.pin();
        for round in 0..rounds {
            let mut list = {
                let round = u8::try_from(round).map_err(|_| "scratch rounds exceed u8")?;
                self.part(graph, BankPart::RoundList { round })?
            };
            let park = grid(grids.tensor(), round);
            let RecordTensors { gate, up, down } = scratch.as_mut();
            let mut gate = match gate {
                Some(gate) => gate.tensor().slice_leading(0, &numbers.layer_window),
                None => up.tensor().slice_leading(0, 0),
            };
            self.park(graph, ParkKind::Round(round));
            graph.enqueue(
                entries.round,
                &landing,
                routed_round::WorkflowArgs {
                    step: step.tensor().into(),
                    scratch: slots.tensor_mut().into(),
                    window: window.tensor().into(),
                    rounds: window_rounds,
                    round: i32::try_from(round).map_err(|_| "scratch rounds exceed i32")?.into(),
                    landing: list.tensor_mut().into(),
                    tiered: tiered.tensor().into(),
                    conveyor_gate: conveyor_gate.arg(),
                    conveyor_up: conveyor.up.tensor().into(),
                    conveyor_down: conveyor.down.tensor().into(),
                    gate: (&mut gate).into(),
                    up: up.tensor_mut().into(),
                    down: down.tensor_mut().into(),
                    seeds: seeds.tensor().into(),
                    map: map.tensor().into(),
                    grid: match round {
                        0 => always.port().tensor().into(),
                        _ => (&park).into(),
                    },
                },
            )?;
            if round == 0 {
                graph.enqueue(
                    entries.scratch_grids,
                    &[("FLIGHT", scratch_slots.clone()), ("LANES", scratch_lanes.clone()), ("RW", numbers.window_rounds.clone())],
                    routed_scratch_grids::WorkflowArgs {
                        step: step.tensor().into(),
                        scratch: slots.tensor().into(),
                        window: window.tensor().into(),
                        grids: grids.tensor_mut().into(),
                        round_records: numbers.round_records.word().ok_or("round records exceed i32")?,
                    },
                )?;
                // Beside a host tier, the tier copy: each expert of the step
                // the tier keeps whole, which round 0's answer lists, read
                // from its tier record in place into its scratch slot, on
                // the tier list's grid.
                if let Some((records, tier_grid, staged)) = tier.as_mut() {
                    let seeding = entries.seeding;
                    graph.enqueue(
                        seeding.list_grid,
                        &[("L", numbers.window_slots.clone())],
                        routed_list_grid::WorkflowArgs {
                            list: tiered.tensor().into(),
                            step: step.tensor().into(),
                            grid: tier_grid.tensor_mut().into(),
                            staged: staged.tensor_mut().into(),
                        },
                    )?;
                    let tensors = numbers.matrix.as_ref().zip(seeding.seed).zip(records.as_ref()).zip(scratch.as_mut());
                    for ((([n, k], copy), records), scratch) in tensors {
                        graph.enqueue(
                            copy,
                            &[
                                ("S", numbers.tier.clone()),
                                ("D", numbers.layer_window.clone()),
                                ("L", numbers.window_slots.clone()),
                                ("N", n.clone()),
                                ("K", k.clone()),
                            ],
                            routed_copy::WorkflowArgs {
                                source: records.tensor().into(),
                                list: staged.tensor().into(),
                                step: step.tensor().into(),
                                destination: scratch.tensor_mut().into(),
                                grid: tier_grid.tensor().into(),
                            },
                        )?;
                    }
                }
                // The seed list's grid: its host lists the group's last
                // step's seeds before acknowledging round 0, and no list of
                // another step launches the seed copy.
                graph.enqueue(
                    entries.seeding.list_grid,
                    &[("L", numbers.window_slots.clone())],
                    routed_list_grid::WorkflowArgs {
                        list: seeds.tensor().into(),
                        step: step.tensor().into(),
                        grid: seed_grid.tensor_mut().into(),
                        staged: seed_staged.tensor_mut().into(),
                    },
                )?;
            }
        }
        constants.push(always);
        // A prompt group's last step seeds the group's records, whoever
        // places, from the filled scratch: its listed experts copied into
        // their records on the seed list's grid, then each record
        // committed, which the host maps once the step has completed. A list
        // of another step, or a failed step, seeds nothing.
        let seeding = entries.seeding;
        let staged = &seed_staged;
        // The seeds' records from the first a seed takes
        // (`ExpertBankPlan::seeded_from`), which the seed list names them
        // from: the lender's lie past the lent records the conveyor and the
        // scratch fill, so the copy's destination is disjoint from its
        // source by construction. A lender with no record past them seeds
        // nothing: its last write the scratch's up records, which every
        // round converts into. Its records and commits, which only the seed
        // copy and commit read, are not made: a graph port no launch reads is
        // unbound.
        let first = &numbers.seeded_from;
        let seeded = numbers.records.checked_sub(first).ok_or("a bank group's lent records exceed its records")?;
        if seeded.is_zero() {
            return Ok(scratch.up.tensor().clone());
        }
        // The layer's group's records, its expert stacks, which the device's
        // seed copy writes.
        let records = {
            let (planned, scope) = (self.planned, self.scope);
            self.topology.record.try_map(|kind, _| {
                weight_port(graph, planned, WeightRole { scope, kind: kind.kind() }, WeightPart::Values, weights)
            })?
        };
        let commits = self.part(graph, BankPart::Commits)?;
        let tensors = numbers.matrix.as_ref().zip(seeding.seed).zip(scratch.as_ref()).zip(records.as_ref());
        for ((([n, k], seed), scratch), records) in tensors {
            let mut destination = records.tensor().slice_leading(first, &numbers.records);
            graph.enqueue(
                seed,
                &[
                    ("S", numbers.layer_window.clone()),
                    ("D", seeded.clone()),
                    ("L", numbers.window_slots.clone()),
                    ("N", n.clone()),
                    ("K", k.clone()),
                ],
                routed_copy::WorkflowArgs {
                    source: scratch.tensor().into(),
                    list: staged.tensor().into(),
                    step: step.tensor().into(),
                    destination: (&mut destination).into(),
                    grid: seed_grid.tensor().into(),
                },
            )?;
        }
        let mut committed = commits.tensor().slice_leading(first, &numbers.records);
        graph.enqueue(
            seeding.seed_commit,
            &[
                ("X", numbers.window_slots.clone()),
                ("BANK", seeded),
                ("FLIGHT", scratch_slots.clone()),
                ("LANES", scratch_lanes.clone()),
            ],
            routed_seed_commit::WorkflowArgs {
                list: seeds.tensor().into(),
                step: step.tensor().into(),
                scratch: slots.tensor().into(),
                window: window.tensor().into(),
                rounds: window_rounds,
                commits: (&mut committed).into(),
            },
        )?;
        Ok(commits.tensor().clone())
    }
}

/// A scratch-served step's fill of the streamed routed layer of `scope`, of
/// any form (`OnceStage::Fill`, `StreamedLayer::fill_scratch`).
#[allow(clippy::too_many_arguments)]
pub(crate) fn scratch_fill<'a, G: GraphDraft + 'a>(
    graph: &mut G,
    bank: LayerBank<'_, 'a, G>,
    form: &BankForm,
    numbers: Option<&BankNumbers>,
    planned: &(impl PlannedWeights + ?Sized),
    scope: WeightScope,
    weights: &mut Vec<(WeightPort, NativePort)>,
    constants: &mut Vec<GraphConstant>,
) -> Result<WorkflowTensor, GraphError> {
    let mut layer =
        StreamedLayer::of(planned, form, numbers, scope, bank)?.ok_or("a fill of a layer the load does not stream")?;
    layer.fill_scratch(graph, weights, constants)
}

/// What a run binds of a graph that sets its step word (`routed_step`): the
/// sequence it writes, and of a decode graph the guesses its front rows run
/// on (`routed_step`'s `guesses`, `[T, 2]` (guess, kind) rows at each front
/// row's exact row, T the graph's rows): the guesses of the chain made
/// behind the step before (`guess_chain`), which its token rows follow, or
/// the host's where it holds the guesses.
#[derive(Clone, Debug)]
pub(crate) struct StepInputs {
    pub sequence: NativePort,
    pub guesses: Option<NativePort>,
}

/// What a bank-served layer's route, landing launch and site bind, one port
/// each: the step (set here when the layer sets it, with the run's step
/// inputs), its group's map, its file table of the exchange, the landing
/// launch's grid the route writes, and where the bank has a host tier the
/// layer's tier map and its group's tier records (the streaming header's
/// requests in routed.seismic).
pub(crate) struct DecodePorts {
    pub step: NativePort,
    pub inputs: Option<StepInputs>,
    pub map: NativePort,
    pub tier: Option<TierPorts>,
    /// The layer's policy words (`BankPart::Policy`): while its POLICY_WHOLE
    /// is set its landing launch and site are not encoded and its route is the
    /// whole form; its lines are the route's assignment's.
    pub policy: NativePort,
    /// The layer's file table (`BankPart::FileTable`).
    pub file: NativePort,
    /// The layer's landing launch's grid (`BankPart::LandGrid`).
    pub land_grid: NativePort,
    /// The layer's CPU site's words and buffers, where it has a site.
    pub site: Option<SitePorts>,
}

impl DecodePorts {
    /// The route's `sited`: 1 where the layer has a CPU site (its graph
    /// binds the site's words), else 0, so its route assigns nothing to the
    /// CPU whatever its policy words hold.
    pub(crate) fn sited(&self) -> i32 {
        i32::from(self.site.is_some())
    }
}

/// A layer's site's ports (`BankPart::SiteTable`, `SiteRows`,
/// `SiteOutput`: streaming_words.seismic's `site`, `site_rows`,
/// `site_output`).
pub(crate) struct SitePorts {
    pub table: NativePort,
    pub rows: NativePort,
    pub output: NativePort,
}

impl SitePorts {
    pub(crate) fn into_combine(self) -> SiteCombinePorts {
        SiteCombinePorts { table: self.table, output: self.output }
    }
}

/// Minimal site ports for `routed_output_sited` / `routed_down_sited`:
/// binds only `SiteTable` and `SiteOutput` (no `SiteRows`).
pub(crate) struct SiteCombinePorts {
    pub table: NativePort,
    pub output: NativePort,
}

/// A bank-served layer's host tier ports: its tier map and its group's
/// tier records.
pub(crate) struct TierPorts {
    pub map: NativePort,
    pub records: RecordTensors<NativePort>,
}

/// Where a decode launch's expert entries read each choice's weights
/// (`decode_source`): of a bank-served layer the route's snapshot of the
/// step's sources (`XS` rows) and its group's host tier records (`TR` of
/// them; without a tier, views of no records of the bank's stacks); of a
/// resident layer none (`XS` 0: each expert from its stacks by index), and
/// views of no records of its stacks.
pub(crate) struct DecodeReads {
    pub sources: WorkflowTensorView,
    pub rows: Dim,
    pub tier: RecordTensors<WorkflowTensorView>,
    pub records: Dim,
    /// The layer's policy words; none of a resident layer, whose graph
    /// declares its zero words only where an entry reads them.
    pub policy: Option<WorkflowTensor>,
}

impl DecodeReads {
    /// The resident routed layer of `scope`'s of `rows` rows of `selected`
    /// choices over its stacks `stacks`.
    pub(crate) fn resident<G: GraphDraft>(
        graph: &mut G,
        rows: &Dim,
        selected: &Dim,
        stacks: RecordTensors<&WorkflowTensor>,
        constants: &mut Vec<GraphConstant>,
    ) -> Result<Self, GraphError> {
        let sources = GraphConstant::row_empty(graph, Element::i32(), rows, selected, constants)?.slice_leading(0, 0);
        Ok(Self {
            sources,
            rows: Dim::from(source_rows(false)),
            tier: stacks.map(|_, stack| stack.slice_leading(0, 0)),
            records: Dim::from(0),
            policy: None,
        })
    }

    /// The decode entries' dimensions of them: `XS` and `TR`.
    pub(crate) fn dimensions(&self) -> [(&'static str, Dim); 2] {
        [("XS", self.rows.clone()), ("TR", self.records.clone())]
    }
}

/// What a bank-served layer's route writes beside its outputs and its
/// later entries read: the step's sources (`routed.seismic`'s `resolve`).
pub(crate) struct RouteLocals {
    pub sources: NativePort,
}

impl RouteLocals {
    /// The step's sources snapshot, which every later decode entry of the
    /// layer reads.
    pub(crate) fn snapshot(&self) -> WorkflowTensorView {
        self.sources.tensor().slice_leading(0, magnitude_kernels::constants::SOURCE_ROWS)
    }
}

/// What a bank-served decode launch's route leaves its later entries: its
/// decode ports, the sources it wrote (`locals`), and the step's
/// routing record (`BankPart::Routing`), which the gate/up's late pass
/// reads.
pub(crate) struct RoutedStep {
    pub ports: DecodePorts,
    pub locals: RouteLocals,
    pub routing: NativePort,
}

/// What a decode launch's site request names of its step
/// ([`StreamedLayer::site_request`]): its choices' experts (`routes`) and
/// weights, the rows its experts project (`input`), and its rows.
pub(crate) struct StepChoices<'t> {
    pub routes: &'t WorkflowTensor,
    pub weights: &'t WorkflowTensor,
    pub input: &'t WorkflowTensor,
    pub rows: &'t Dim,
}

/// What a bank-served route binds of its layer's host tier: the tier map's
/// words (`TM`: the layer's experts, none without a tier) and the tensor, a
/// graph constant of no words without a tier.
pub(crate) struct RouteTier {
    pub experts: Dim,
    pub map: WorkflowTensor,
}

/// What a decode landing launch binds: the layer's decode ports, and the
/// sources its route wrote.
pub(crate) struct LandPorts<'f> {
    pub decode: &'f mut DecodePorts,
    pub sources: &'f mut WorkflowTensor,
    /// The stacks' ports, the bank's records where the bank places, which
    /// the misses the device lands are converted into.
    pub records: RecordTensors<&'f mut NativePort>,
}

/// Whether `load` streams the experts of the feed-forward of `scope`.
pub(crate) fn streams(load: &ModelLoadPlan, scope: WeightScope) -> bool {
    load.expert_bank().is_some_and(|bank| bank.feed_forward(scope).is_some())
}

/// Whether the streamed feed-forward of `scope` has a resident form
/// (`RunStage::Resident`): its bank was planned whole
/// (`ExpertBankPlan::resident`), so it starts whole.
pub(crate) fn resident_form(load: &ModelLoadPlan, scope: WeightScope) -> bool {
    streams(load, scope) && load.expert_bank().is_some_and(|bank| bank.resident)
}

impl<'p, 'a, G: GraphDraft + 'a, W: PlannedWeights + ?Sized> StreamedLayer<'p, 'a, G, W> {
    pub(crate) fn decode_ports(
        &mut self,
        graph: &mut G,
        rows: &Dim,
        constants: &mut Vec<GraphConstant>,
    ) -> Result<DecodePorts, GraphError> {
        let (step, inputs) = self.step(graph, Some(rows), constants)?;
        let tier = match self.numbers.tier.is_zero() {
            true => None,
            false => Some(TierPorts {
                map: self.part(graph, BankPart::TierMap)?,
                records: self.parts(graph, |kind| BankPart::Tier { kind })?,
            }),
        };
        Ok(DecodePorts {
            step,
            inputs,
            map: self.part(graph, BankPart::Map)?,
            tier,
            policy: self.part(graph, BankPart::Policy)?,
            file: self.part(graph, BankPart::FileTable)?,
            land_grid: self.part(graph, BankPart::LandGrid)?,
            site: match self.entries.site {
                Some(_) => Some(SitePorts {
                    table: self.part(graph, BankPart::SiteTable)?,
                    rows: self.part(graph, BankPart::SiteRows)?,
                    output: self.part(graph, BankPart::SiteOutput)?,
                }),
                None => None,
            },
        })
    }
}
