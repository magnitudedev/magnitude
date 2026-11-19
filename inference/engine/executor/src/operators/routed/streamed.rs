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
    BankBinding, BankPart, BankedFeedForward, ExpertBankPlan, HINT_LEADS, RecordTensors, RoutedTensor,
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
    BankPort, GraphPark, GraphTimeline, WeightPart, WeightPort, bank_port, relative_scope_of, weight, weight_port,
};
use crate::ModelLoadPlan;
use crate::programs::native_target_graph::PlannedWeights;
use magnitude_family_contracts::{WeightRole, WeightScope};
use magnitude_kernels::{
    routed_copy, routed_down_project, routed_down_sited, routed_gate_up, routed_gate_up_part,
    routed_hint_route, routed_hint_select, routed_land, routed_list_grid, routed_output_project,
    routed_output_sited, routed_pass_map, routed_round, routed_route_streamed, routed_scratch_grids,
    routed_scratch_request, routed_seed_commit, routed_select_streamed, routed_site_request, routed_step,
};
use seismic::{
    Dim, Element, NativeGraph, NativeGraphMetadata, NativePort, NativeTimelinePort, WorkflowTensor,
    WorkflowTensorRef, WorkflowTensorView,
};

/// A streamed layer's decode route and resolve by its routed form, plus its
/// ahead-of-step hint launches by `AHEAD` (`1..=HINT_LEADS`)
/// (`native::BankRouteKernels`).
pub(crate) enum BankRoute<'a, G: GraphDraft + 'a> {
    Fused {
        activation: Element,
        route: G::Ranged<'a, routed_route_streamed::Entry>,
        hints: [Option<G::Binding<'a, routed_hint_route::Entry>>; HINT_LEADS as usize],
    },
    General {
        activation: Element,
        select: G::Binding<'a, routed_select_streamed::Entry>,
        hints: [Option<G::Binding<'a, routed_hint_select::Entry>>; HINT_LEADS as usize],
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
    /// The CPU site's request and combine, where the layer has a site.
    pub site: Option<BankSiteEntries<'a, G>>,
    /// A gated layer's gate/up passes, its held choices' before the tier
    /// wait and the wait's after it (`native::ExpertBankKernels::gate_up`);
    /// none of up-only experts.
    pub gate_up: Option<G::Ranged<'a, routed_gate_up_part::Entry>>,
}

/// A streamed layer's CPU site entries (`native::BankSite`): its request
/// and its combine waiting for the CPU's answer, by its routed form.
pub(crate) struct BankSiteEntries<'a, G: GraphDraft + 'a> {
    pub request: G::Binding<'a, routed_site_request::Entry>,
    pub combine: BankCombineEntry<'a, G>,
}

/// A streamed layer's combine with the CPU's answer, by its routed form: its
/// projections (by row range), then their sum waiting for the answer
/// (`native::BankCombine`).
pub(crate) enum BankCombineEntry<'a, G: GraphDraft + 'a> {
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

impl<'a> From<&'a ExpertBankKernels> for BankGraphEntries<'a, NativeGraph> {
    fn from(kernels: &'a ExpertBankKernels) -> Self {
        Self {
            route: match &kernels.route {
                BankRouteKernels::Fused { activation, route, hints } => BankRoute::Fused {
                    activation: *activation,
                    route,
                    hints: hints.each_ref().map(Option::as_ref),
                },
                BankRouteKernels::General { activation, select, hints } => BankRoute::General {
                    activation: *activation,
                    select,
                    hints: hints.each_ref().map(Option::as_ref),
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
                combine: match &site.combine {
                    crate::native::BankCombine::Fused { project, sited } => BankCombineEntry::Fused { project, sited },
                    crate::native::BankCombine::General { project, sited } => BankCombineEntry::General { project, sited },
                },
            }),
            gate_up: kernels.gate_up.as_ref(),
        }
    }
}

/// The route's element assignments by routed form.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum CheckedRoute {
    Fused {
        route: [(&'static str, Element); 6],
        hints: [(&'static str, Element); 3],
    },
    General {
        select: [(&'static str, Element); 4],
        hints: [(&'static str, Element); 3],
    },
}

/// The bank entries' element assignments, from a layer's feed-forward
/// binding (either routed form) and its bank's.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct CheckedBankEntries {
    route: CheckedRoute,
    /// A landing launch's, decode's and a round's alike: each routed
    /// tensor's stored and resident elements.
    landing: [(&'static str, Element); 6],
    none: [(&'static str, Element); 0],
    /// The seed copy's.
    seed: RecordTensors<[(&'static str, Element); 1]>,
    /// The CPU site's request's and combine's, where the layer has a site
    /// (`native::BankSite::served`).
    site: Option<CheckedSite>,
    /// The gate/up passes', of gated experts.
    gate_up: Option<[(&'static str, Element); 3]>,
}

/// A CPU site's element assignments: its request's, and its combine's by
/// the layer's routed form.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct CheckedSite {
    request: [(&'static str, Element); 1],
    combine: CheckedCombine,
}

/// A combine's projections' and sum's element assignments.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum CheckedCombine {
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
            hints: [
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
                hints: [
                    ("NW", binding.norm),
                    ("RW", binding.router),
                    ("A", binding.activation),
                ],
            },
            BankedFeedForward::GeneralRouted(binding) => general(binding),
            BankedFeedForward::Parallel(binding) => general(binding.routed),
        };
        let site = crate::native::BankSite::served(slot, bank, backend)?.then(|| match slot {
            BankedFeedForward::Routed(binding) => CheckedSite {
                request: [("A", binding.activation)],
                combine: CheckedCombine::Fused(
                    [("EDW", binding.expert_down), ("SDW", binding.shared_down), ("A", binding.activation)],
                    [("A", binding.activation)],
                ),
            },
            BankedFeedForward::GeneralRouted(binding) | BankedFeedForward::Parallel(crate::ParallelBinding { routed: binding, .. }) => {
                CheckedSite {
                    request: [("A", binding.activation)],
                    combine: CheckedCombine::General(
                        [("EDW", binding.expert_down), ("A", binding.activation)],
                        [("A", binding.activation), ("R", super::graph::sum_element(&binding))],
                    ),
                }
            }
        });
        Ok(Self {
            route,
            landing,
            none: [],
            seed: bank.record.map(|_, tensor| [("U", tensor.resident)]),
            site,
            gate_up,
        })
    }

    pub(crate) fn entries(&self) -> BankGraphEntries<'_, NativeGraphMetadata> {
        BankGraphEntries {
            route: match &self.route {
                CheckedRoute::Fused { route, hints } => BankRoute::Fused {
                    activation: hints[2].1,
                    route: &route[..],
                    hints: [Some(&hints[..]); HINT_LEADS as usize],
                },
                CheckedRoute::General { select, hints } => BankRoute::General {
                    activation: hints[2].1,
                    select: &select[..],
                    hints: [Some(&hints[..]); HINT_LEADS as usize],
                },
            },
            land: &self.landing,
            step: &self.none,
            scratch_request: &self.none,
            round: &self.landing,
            scratch_grids: &self.none,
            pass_map: &self.none,
            seeding: BankSeedingEntries {
                seed: self.seed.as_ref().map(|_, seed| &seed[..]),
                seed_commit: &self.none[..],
                list_grid: &self.none[..],
            },
            site: self.site.as_ref().map(|site| BankSiteEntries {
                request: &site.request[..],
                combine: match &site.combine {
                    CheckedCombine::Fused(project, sited) => {
                        BankCombineEntry::Fused { project: &project[..], sited: &sited[..] }
                    }
                    CheckedCombine::General(project, sited) => {
                        BankCombineEntry::General { project: &project[..], sited: &sited[..] }
                    }
                },
            }),
            gate_up: self.gate_up.as_ref().map(|gate_up| &gate_up[..]),
        }
    }
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
/// point cannot build the launch without it.
pub(crate) fn land_launch<'a, G: GraphDraft + 'a>(
    graph: &mut G,
    entry: G::Binding<'a, routed_land::Entry>,
    dimensions: &[(&str, impl Clone + Into<Dim>)],
    args: routed_land::WorkflowArgs<'_>,
) -> Result<NativeTimelinePort<seismic::timeline::Parks>, GraphError> {
    let park = graph.timeline(seismic::timeline::Park);
    graph.enqueue(entry, dimensions, args)?;
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
    pub ahead: u64,
    pub leads: Vec<HintLead>,
    /// The layer whose policy words its hints bind, relative to its block.
    pub next_policy: Option<WeightScope>,
    pub placer: crate::planning::Placer,
    pub record: RecordTensors<Element>,
    pub stored: RecordTensors<Element>,
}

/// What a hint lead binds (`StreamedLayer::hints`): the layer's own hinted
/// weights, an earlier lead's, or the weights of the predicted layer of
/// `scope` it has (each hinted port's kind where that layer has the weight),
/// named relative to the layer's block (`relative_scope_of`).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) enum HintLead {
    Own,
    Earlier(usize),
    Layer { scope: WeightScope, kinds: Vec<Option<magnitude_family_contracts::WeightKind>> },
}

/// Every number a streamed layer's graph is built of
/// (`ExpertBankPlan::binding` and the bank's layout), in the bank plan's
/// names: `BANK`, `E`, the slot map's experts, `EN`, `K`, `T`, a round's
/// `L`, the widest layer's experts, `V`, `TR`, the scratch rounds,
/// `FLIGHT` and `P`, its tables' slots and widths, `RING`, `GM`, the bank's
/// choices (a decode landing list's entries), the first seeded record, and
/// each routed tensor's `[N, K]`.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct BankNumbers<D = Dim> {
    pub records: D,
    pub experts: D,
    pub layer_experts: D,
    pub next_experts: D,
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
    pub bank_choices: D,
    pub seeded_from: D,
    pub matrix: RecordTensors<[D; 2]>,
}

impl<D> Numbers<D> for BankNumbers<D> {
    type Of<E> = BankNumbers<E>;
    fn map<E>(self, f: &mut impl FnMut(D) -> E) -> BankNumbers<E> {
        BankNumbers {
            records: f(self.records),
            experts: f(self.experts),
            layer_experts: f(self.layer_experts),
            next_experts: f(self.next_experts),
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
            bank_choices: f(self.bank_choices),
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
    let leads = (0..crate::planning::HINT_LEADS as usize).map(|lead| plan.lead_layer(index, lead)).collect::<Vec<_>>();
    let leads = leads
        .iter()
        .enumerate()
        .map(|(lead, &predicted)| match leads[..lead].iter().position(|&bound| bound == predicted) {
            _ if predicted == index => HintLead::Own,
            Some(earlier) => HintLead::Earlier(earlier),
            None => {
                let predicted = &plan.layers()[predicted];
                HintLead::Layer {
                    scope: relative_scope_of(predicted.scope, scope),
                    kinds: predicted.hinted.iter().map(|hint| hint.element.map(|_| hint.kind)).collect(),
                }
            }
        })
        .collect();
    let (words, file, site, scratch) =
        (binding.words, binding.words.file(), binding.words.site(), binding.words.scratch());
    let topology = BankTopology {
        sited: plan.sited(index),
        sets_step: plan.sets_step(index),
        fronts: load.fronts(),
        ahead: plan.ahead(index),
        leads,
        next_policy: plan.hints(index, 0).map(|next| relative_scope_of(plan.layers()[next].scope, scope)),
        placer: binding.placer,
        record: binding.record.map(|_, tensor| tensor.resident),
        stored: binding.record.map(|_, tensor| tensor.source),
    };
    let numbers = BankNumbers {
        records: binding.records,
        experts: binding.experts,
        layer_experts: layer.experts,
        next_experts: binding.next_experts,
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
        bank_choices: plan.choices(),
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
            (BankForm::Streamed(topology), Some(numbers), LayerBank::Streamed(BankGraph { mut entries, ports, parks })) => {
                // The plan alone says which layers have a CPU site
                // (`ExpertBankPlan::sited`): a layer encodes the load's site
                // entries exactly there, and a sited layer needs them.
                entries.site = match (topology.sited, entries.site) {
                    (true, None) => return Err("a layer with a CPU site lacks its site entries".into()),
                    (sited, site) => site.filter(|_| sited),
                };
                Ok(Some(Self { planned, topology, numbers, scope, entries, ports, parks }))
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
        let numbers = self.numbers;
        let (tier, count) = match &ports.tier {
            Some(tier) => (
                tier.records.as_ref().map(|_, port| port.tensor().slice_leading(0, &numbers.tier)),
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
        Ok(DecodeReads {
            sources: locals.snapshot(),
            rows: Dim::from(source_rows(true)),
            tier,
            records: count,
            policy: Some(ports.policy.tensor().clone()),
        })
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
        shape: &DecodeShape,
        constants: &mut Vec<GraphConstant>,
    ) -> Result<(), GraphError> {
        let numbers = self.numbers;
        let LandPorts { decode: DecodePorts { step, policy, file, land_grid, .. }, sources, records } = ports;
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
        let dimensions = [
            ("M", shape.rows.clone()),
            ("K", numbers.selected.clone()),
            ("FW", numbers.file_width.clone()),
            ("FLIGHT", numbers.file_slots.clone()),
            ("L", numbers.bank_choices.clone()),
            ("V", from.clone()),
            ("GV", gated(&from)),
            ("S", into.clone()),
            ("GS", gated(&into)),
            ("F", features),
            ("H", hidden),
        ];
        let staging_gate = match &staging.gate {
            Some(gate) => gate.clone(),
            None => staging.up.slice_leading(0, 0),
        };
        let mut gate = match destination.gate.take() {
            Some(gate) => gate,
            None => destination.up.slice_leading(0, 0),
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
            },
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
                step: ports.step.tensor().into(),
                routes: routes.into(),
                sources: sources.into(),
                weights: weights.into(),
                normalized: normalized.into(),
                site: site.table.tensor_mut().into(),
                site_rows: site.rows.tensor_mut().into(),
            },
            &ports.policy,
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
        self.land(graph, LandPorts { decode: &mut step.ports, sources: &mut sources, records }, &shape, constants)?;
        let snapshot = step.locals.snapshot();
        self.site_request(graph, &mut step.ports, choices.routes, choices.weights, &snapshot, choices.input, &shape)
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

    /// The layer's combine with the CPU's answer and the dimensions of its
    /// site table (`FLIGHT`, `SW`), where the layer has a site: its
    /// projections, then its sum, which waits for the answer when a choice
    /// of the step is the CPU's.
    pub(crate) fn site_combine(&self) -> Option<(BankCombineEntry<'a, G>, [(&'static str, Dim); 2])> {
        let entries = self.entries.site?;
        let numbers = self.numbers;
        Some((entries.combine, [("FLIGHT", numbers.site_slots.clone()), ("SW", numbers.site_width.clone())]))
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
        // staged in device memory it reads.
        let mut tier = match numbers.tier.is_zero() {
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

/// The ports of a bank-served layer's ahead-of-step hint launch
/// (`routed_hint_route`, `routed_hint_select`), when the layer predicts ahead
/// (`ahead` in `1..=HINT_LEADS`): `AHEAD`; per lead, the weights its hint
/// launch reads of the layer it predicts there (`ExpertBankPlan::predicted`:
/// the next streamed layer, then the one after it) in its own place, in the
/// route's order (`BankLayer::hinted`); the next layer's policy words, whose
/// POLICY_RESOLVED skips the hint writes while set; and its rings. A lead past
/// `AHEAD` binds the next layer's weights again, which the hint launch zeros
/// without reading, so every lead's weights are of the experts `EN` names
/// (`ExpertBankPlan::lead_layer`).
pub(crate) struct HintPorts<const N: usize> {
    pub ahead: u64,
    pub weights: [[WorkflowTensor; N]; crate::planning::HINT_LEADS as usize],
    pub policy: WorkflowTensor,
    pub rings: NativePort,
}

impl<'a, G: GraphDraft + 'a, W: PlannedWeights + ?Sized> StreamedLayer<'_, 'a, G, W> {
    /// The hint ports of this layer when it predicts ahead (`ahead > 0`), or
    /// `None` on the last streamed layer, which launches no hint entry. A
    /// weight a predicted layer has is its own; one it lacks is this layer's
    /// port (`own`: an absent norm, a bias's zeros), as the plan found every
    /// streamed layer's hinted weights alike.
    pub(crate) fn hints<const N: usize>(
        &mut self,
        graph: &mut G,
        weights: &mut Vec<(WeightPort, NativePort)>,
        own: [&WorkflowTensor; N],
        policy: &WorkflowTensor,
    ) -> Result<Option<HintPorts<N>>, GraphError> {
        let topology = self.topology;
        let ahead = topology.ahead;
        if ahead == 0 {
            return Ok(None);
        }
        let mut hinted: [[WorkflowTensor; N]; crate::planning::HINT_LEADS as usize] =
            std::array::from_fn(|_| own.map(Clone::clone));
        let mut policy = policy.clone();
        for (lead, bound) in topology.leads.iter().enumerate() {
            match bound {
                HintLead::Own => {}
                // A layer an earlier lead binds is bound once.
                HintLead::Earlier(earlier) => hinted[lead] = hinted[*earlier].clone(),
                HintLead::Layer { scope, kinds } => {
                    for (port, kind) in hinted[lead].iter_mut().zip(kinds) {
                        if let Some(kind) = kind {
                            *port = weight(graph, self.planned, *scope, *kind, weights)?;
                        }
                    }
                }
            }
        }
        if let Some(next) = topology.next_policy {
            policy = bank_port(graph, self.planned, next, BankPart::Policy, self.ports)?.tensor().clone();
        }
        Ok(Some(HintPorts {
            ahead,
            weights: hinted,
            policy,
            rings: self.part(graph, BankPart::Rings)?,
        }))
    }
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

impl<'a, G: GraphDraft + 'a, W: PlannedWeights + ?Sized> StreamedLayer<'_, 'a, G, W> {
    /// The ports of a decode graph of `rows` rows.
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
