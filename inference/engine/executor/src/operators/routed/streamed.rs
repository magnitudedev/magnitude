//! What a streamed routed layer adds to its routed form's graph, fused or
//! general (`routed.seismic`'s streaming section):
//!
//! ```text
//! decode rows:  [step] → route + resolve + assign (each choice's source by
//!               the step-stamped map: a bank record, a host tier record
//!               the GPU reads in place or the CPU computes, or absent) →
//!               tier wait (spin on each absent choice's landing, source
//!               it there) → site request (the CPU's choices and the rows,
//!               across a link) → the resident gate/up, each choice read
//!               where its source names → the projections → the sum,
//!               waiting for the CPU's answer and adding it (a whole
//!               group's layer: the resident combine)
//! larger rows:  front: route → scratch request
//!               fill:  each round: wait, convert the staged → seed copy
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

use crate::native::{BankRouteKernels, ExpertBankKernels, LEADS};
use crate::planning::{BankBinding, BankPart, BankedFeedForward, ExpertBankPlan, RecordTensors, RoutedTensor};
use crate::programs::graph::{GraphError, draft::GraphDraft};
use crate::programs::graph::facts::Numbers;
use crate::programs::native_constants::GraphConstant;
use magnitude_kernels::constants::{COLUMNS_GRID, PARK_GRID, ROUND_GRIDS, ROWS_GRID};

/// The grid of a scratch round's convert of routed tensor `kind`'s records
/// (`routed_scratch_grids`): gate and up records hold F rows, down records
/// H.
pub(crate) fn record_grid(kind: RoutedTensor) -> u64 {
    match kind {
        RoutedTensor::Gate | RoutedTensor::Up => ROWS_GRID,
        RoutedTensor::Down => COLUMNS_GRID,
    }
}

/// A scratch round's convert threadgroups a row of records of `rows` rows
/// (`routed_scratch_grids`' `rows_groups` and `columns_groups`): a tile of
/// the declaration's rows each.
pub(crate) fn convert_groups(rows: u64) -> Result<i32, &'static str> {
    i32::try_from(rows.div_ceil(magnitude_kernels::constants::CONVERT_TILE_ROWS))
        .map_err(|_| "convert groups exceed i32")
}

/// What a host port of the gate's records of `ports` (staging or conveyor)
/// binds: the gate's, or of an up-only record, none of its up's (a view of
/// no records, `GT` 0), in the element its entries bind the gate's
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
    routed_convert, routed_copy, routed_down_project, routed_down_sited, routed_list_grid, routed_list_stage, routed_output_project, routed_pass_map,
    routed_output_sited, routed_route_streamed, routed_scratch_grids, routed_scratch_request, routed_scratch_wait,
    routed_seed_commit, routed_select_streamed, routed_site_request, routed_step, routed_tier_wait,
};
use seismic::{
    Dim, Element, NativeGraph, NativeGraphMetadata, NativePort, NativeTimelinePort, WorkflowTensor,
    WorkflowTensorRef, WorkflowTensorView,
};

/// A streamed layer's decode route and resolve by its routed form, each by
/// `AHEAD` ([`LEADS`]) a layer of its binding launches
/// (`native::BankRouteKernels`).
pub(crate) enum BankRoute<'a, G: GraphDraft + 'a> {
    Fused([Option<G::Binding<'a, routed_route_streamed::Entry>>; LEADS]),
    General([Option<G::Binding<'a, routed_select_streamed::Entry>>; LEADS]),
}

impl<'a, G: GraphDraft + 'a> Copy for BankRoute<'a, G> {}
impl<'a, G: GraphDraft + 'a> Clone for BankRoute<'a, G> {
    fn clone(&self) -> Self {
        *self
    }
}

/// A streamed layer's bank entries; `convert` and `seed` per routed tensor
/// of the record.
pub(crate) struct BankGraphEntries<'a, G: GraphDraft + 'a> {
    pub route: BankRoute<'a, G>,
    pub tier_wait: G::Binding<'a, routed_tier_wait::Entry>,
    pub step: G::Binding<'a, routed_step::Entry>,
    pub scratch_request: G::Binding<'a, routed_scratch_request::Entry>,
    pub scratch_wait: G::Binding<'a, routed_scratch_wait::Entry>,
    pub scratch_grids: G::Binding<'a, routed_scratch_grids::Entry>,
    pub list_stage: G::Binding<'a, routed_list_stage::Entry>,
    pub pass_map: G::Binding<'a, routed_pass_map::Entry>,
    pub convert: RecordTensors<G::Binding<'a, routed_convert::Entry>>,
    /// The device's seed copy, where it copies into the bank
    /// (`DevicePlacement`).
    pub seeding: Option<BankSeedingEntries<'a, G>>,
    /// The CPU site's request and combine, where the layer has a site.
    pub site: Option<BankSiteEntries<'a, G>>,
}

/// A streamed layer's CPU site entries (`native::BankSite`): its request
/// and its combine waiting for the CPU's answer, by its routed form.
pub(crate) struct BankSiteEntries<'a, G: GraphDraft + 'a> {
    pub request: G::Binding<'a, routed_site_request::Entry>,
    pub combine: BankCombineEntry<'a, G>,
}

/// A streamed layer's combine with the CPU's answer, by its routed form: its
/// projections, then their sum waiting for the answer
/// (`native::BankCombine`).
pub(crate) enum BankCombineEntry<'a, G: GraphDraft + 'a> {
    Fused {
        project: G::Binding<'a, routed_output_project::Entry>,
        sited: G::Binding<'a, routed_output_sited::Entry>,
    },
    General {
        project: G::Binding<'a, routed_down_project::Entry>,
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
                BankRouteKernels::Fused(route) => BankRoute::Fused(route.each_ref().map(Option::as_ref)),
                BankRouteKernels::General(select) => BankRoute::General(select.each_ref().map(Option::as_ref)),
            },
            tier_wait: &kernels.tier_wait,
            step: &kernels.step,
            scratch_request: &kernels.scratch_request,
            scratch_wait: &kernels.scratch_wait,
            scratch_grids: &kernels.scratch_grids,
            list_stage: &kernels.list_stage,
            pass_map: &kernels.pass_map,
            convert: kernels.convert.as_ref(),
            seeding: kernels.placement.as_ref().map(|placement| BankSeedingEntries {
                seed: placement.seed.as_ref(),
                seed_commit: &placement.seed_commit,
                list_grid: &placement.list_grid,
            }),
            site: kernels.site.as_ref().map(|site| BankSiteEntries {
                request: &site.request,
                combine: match &site.combine {
                    crate::native::BankCombine::Fused { project, sited } => BankCombineEntry::Fused { project, sited },
                    crate::native::BankCombine::General { project, sited } => BankCombineEntry::General { project, sited },
                },
            }),
        }
    }
}

/// The route's element assignments by routed form.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum CheckedRoute {
    Fused([(&'static str, Element); 5]),
    General([(&'static str, Element); 4]),
}

/// The bank entries' element assignments, from a layer's feed-forward
/// binding (either routed form) and its bank's.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct CheckedBankEntries {
    route: CheckedRoute,
    staging: [(&'static str, Element); 3],
    none: [(&'static str, Element); 0],
    /// The seed copy's, where the device copies into the bank.
    seed: Option<RecordTensors<[(&'static str, Element); 1]>>,
    convert: RecordTensors<[(&'static str, Element); 2]>,
    /// The CPU site's request's and combine's, where the layer has a site
    /// (`native::BankSite::served`).
    site: Option<CheckedSite>,
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
        // The conveyor a scratch round's park's host fills: an up-only
        // record's gate conveyor is empty (`GV` 0), in its up's stored
        // element.
        let staging = [
            ("SG", gate.map_or(up.source, |gate| gate.source)),
            ("SU", up.source),
            ("SD", down.source),
        ];
        let general = |binding: crate::GeneralRoutedBinding| {
            CheckedRoute::General([
                ("NW", binding.norm),
                ("RNW", binding.router_norm),
                ("RW", binding.router),
                ("A", binding.activation),
            ])
        };
        let route = match slot {
            BankedFeedForward::Routed(binding) => CheckedRoute::Fused([
                ("NW", binding.norm),
                ("RW", binding.router),
                ("SGW", binding.shared_gate),
                ("SUW", binding.shared_up),
                ("A", binding.activation),
            ]),
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
            staging,
            none: [],
            seed: (bank.placer == crate::planning::Placer::Tier).then(|| bank.record.map(|_, tensor| [("U", tensor.resident)])),
            convert: bank.record.map(|_, tensor| [("ST", tensor.source), ("U", tensor.resident)]),
            site,
        })
    }

    pub(crate) fn entries(&self) -> BankGraphEntries<'_, NativeGraphMetadata> {
        BankGraphEntries {
            route: match &self.route {
                CheckedRoute::Fused(route) => BankRoute::Fused([Some(&route[..]); LEADS]),
                CheckedRoute::General(select) => BankRoute::General([Some(&select[..]); LEADS]),
            },
            tier_wait: &self.none,
            step: &self.none,
            scratch_request: &self.none,
            scratch_wait: &self.staging,
            scratch_grids: &self.none,
            list_stage: &self.none,
            pass_map: &self.none,
            convert: self.convert.as_ref().map(|_, convert| &convert[..]),
            seeding: self.seed.as_ref().map(|seed| BankSeedingEntries {
                seed: seed.as_ref().map(|_, seed| &seed[..]),
                seed_commit: &self.none[..],
                list_grid: &self.none[..],
            }),
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
/// the lead its route predicts at (`AHEAD`), what each hint lead binds, the
/// layer whose policy words its hints bind, who places into its bank, and
/// its records' resident elements.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct BankTopology {
    pub sited: bool,
    pub sets_step: bool,
    pub ahead: u64,
    pub leads: Vec<HintLead>,
    /// The layer whose policy words its hints bind, relative to its block.
    pub next_policy: Option<WeightScope>,
    pub placer: crate::planning::Placer,
    pub record: RecordTensors<Element>,
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
/// choices and their landing list's words, a round list's words, the first
/// seeded record, the convert's row and column groups, and each routed
/// tensor's `[N, K]`.
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
    pub landing_list: D,
    pub list_words: D,
    pub seeded_from: D,
    pub rows_groups: D,
    pub columns_groups: D,
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
            landing_list: f(self.landing_list),
            list_words: f(self.list_words),
            seeded_from: f(self.seeded_from),
            rows_groups: f(self.rows_groups),
            columns_groups: f(self.columns_groups),
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
        ahead: plan.ahead(index),
        leads,
        next_policy: plan.hints(index, 0).map(|next| relative_scope_of(plan.layers()[next].scope, scope)),
        placer: binding.placer,
        record: binding.record.map(|_, tensor| tensor.resident),
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
        landing_list: magnitude_kernels::list_extent(plan.choices()),
        list_words: plan.list_words(),
        seeded_from: plan.seeded_from(layer.group),
        rows_groups: u64::try_from(convert_groups(binding.record.up.matrix[0])?).map_err(|_| "convert groups are negative")?,
        columns_groups: u64::try_from(convert_groups(binding.record.down.matrix[0])?)
            .map_err(|_| "convert groups are negative")?,
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
    /// layer sets it, with the sequence input the run writes.
    pub(crate) fn step(&mut self, graph: &mut G) -> Result<(NativePort, Option<NativePort>), GraphError> {
        let entries = self.entries;
        let mut step = self.part(graph, BankPart::Step)?;
        if !self.sets_step() {
            return Ok((step, None));
        }
        let sequence = graph.input_for(entries.step, "sequence", &[] as &[(&str, u64)])?;
        graph.enqueue(
            entries.step,
            &[] as &[(&str, u64)],
            routed_step::WorkflowArgs {
                step: step.tensor_mut().into(),
                sequence: sequence.tensor().into(),
            },
        )?;
        Ok((step, Some(sequence)))
    }

    /// The park node before this layer's call that waits for the host, of
    /// `kind`, which each run binds (`crate::experts::Parks`).
    fn park(&mut self, graph: &mut G, kind: ParkKind) {
        let port = graph.timeline(seismic::timeline::Park);
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

    /// The decode tier wait of the choices the route's file request names
    /// absent (`ports`), after its park: it waits on the step's file slot,
    /// whose answer names the record its host landed each expert into, and
    /// writes each into the step's sources where the route assigned the
    /// expert: the CPU's, or the GPU's in place (a bank record on unified
    /// memory). It dispatches the grid the route wrote: none when no choice
    /// is absent.
    ///
    /// Where the host places, the misses the device lands follow it: each
    /// routed tensor's `routed_convert` of the layer's landing list, from
    /// staging into the bank's records the stacks read, on the grids the
    /// host wrote before it answered: none where it landed them itself
    /// (`experts::landing`), and left out with the wait where nothing is
    /// absent. The wait declares the list host-written, which orders them
    /// after it.
    pub(crate) fn tier_wait(&mut self, graph: &mut G, ports: WaitPorts<'_>, shape: &DecodeShape) -> Result<(), GraphError> {
        let numbers = self.numbers;
        let WaitPorts { decode: DecodePorts { step, policy, file, .. }, sources, grid, records } = ports;
        let list = self.part(graph, BankPart::LandingList)?;
        let dimensions = [
            ("M", shape.rows.clone()),
            ("K", numbers.selected.clone()),
            ("FW", numbers.file_width.clone()),
            ("FLIGHT", numbers.file_slots.clone()),
            ("LW", numbers.landing_list.clone()),
        ];
        self.park(graph, ParkKind::TierWait);
        graph.enqueue(
            self.entries.tier_wait,
            &dimensions,
            routed_tier_wait::WorkflowArgs {
                step: step.tensor().into(),
                file: file.tensor_mut().into(),
                sources: (&mut *sources).into(),
                grid: grid.into(),
                policy: policy.tensor().into(),
                landing: list.tensor().into(),
            },
        )?;
        if self.topology.placer != crate::planning::Placer::Bank {
            return Ok(());
        }
        let staging = self.parts(graph, |kind| BankPart::Staging { kind })?;
        let grids = self.part(graph, BankPart::LandingGrids)?;
        let tensors = numbers.matrix.as_ref().zip(self.entries.convert).zip(staging.as_ref());
        for (kind, ((([n, k], convert), staging), destination)) in tensors.zip(records).into_tensors() {
            let converts = grids.tensor().slice_leading(0, 1).reshape(&[ROUND_GRIDS, 3]).slice_leading(record_grid(kind), record_grid(kind) + 1).reshape(&[3]);
            graph.enqueue_under(
                convert,
                &[
                    ("V", numbers.staging.clone()),
                    ("L", numbers.bank_choices.clone()),
                    ("S", numbers.records.clone()),
                    ("N", n.clone()),
                    ("K", k.clone()),
                ],
                routed_convert::WorkflowArgs {
                    conveyor: staging.tensor().into(),
                    list: list.tensor().into(),
                    step: step.tensor().into(),
                    destination: destination.tensor_mut().into(),
                    grid: (&converts).into(),
                },
                policy,
                magnitude_kernels::constants::POLICY_RESOLVED,
                seismic::LaunchEncoding::WhileClear,
            )?;
        }
        Ok(())
    }

    /// A decode launch's site request, where the layer has a site
    /// (`routed_site_request`): after its route and its tier wait, before
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
    /// of the layer's scratch table, of the run's step (`step`): the
    /// sequence input the run writes where the front sets it.
    pub(crate) fn request(
        &mut self,
        graph: &mut G,
        routes: &WorkflowTensor,
        rows: &Dim,
    ) -> Result<Option<NativePort>, GraphError> {
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
        let (step, sequence) = self.step(graph)?;
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
        Ok(sequence)
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
    ) -> Result<(WorkflowTensor, WorkflowTensor, RecordTensors<NativePort>, Dim, NativePort), GraphError> {
        let map = self.part(graph, BankPart::SlotMap)?;
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
        ))
    }

    /// A scratch-served step's fill of the scratch tier with the cold
    /// experts its front requested (`OnceStage::Fill`): each round's wait and
    /// convert of conveyed records, beside a host tier the copy of the
    /// experts it keeps whole after round 0, then a prompt group's seed copy,
    /// each reading the step word its front set or read. Every datum a wait's
    /// answer gives is read by these launches, after the wait in this graph,
    /// from host memory in place (`routed_scratch_wait`): no answer needs a
    /// submission to the device. Its merge set's pass
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
        // The layer's group's records, its expert stacks, which the device's
        // seed copy writes where it copies into the bank.
        let mut records = match entries.seeding {
            Some(_) => {
                let (planned, scope) = (self.planned, self.scope);
                Some(self.topology.record.try_map(|kind, _| {
                    weight_port(graph, planned, WeightRole { scope, kind: kind.kind() }, WeightPart::Values, weights)
                })?)
            }
            None => None,
        };
        let experts = &numbers.experts;
        let (scratch_slots, scratch_lanes) = (&numbers.scratch_slots, &numbers.scratch_lanes);
        let [features, hidden] = &numbers.matrix.up;
        // An up-only record's gate conveyor is empty (`GV` 0), a view of
        // its up's.
        let gate_conveyor = match self.topology.record.gate {
            Some(_) => numbers.conveyor.clone(),
            None => Dim::from(0),
        };
        // A round's wait's dimensions.
        let wait = [
            ("FLIGHT", scratch_slots.clone()),
            ("LANES", scratch_lanes.clone()),
            ("LW", numbers.list_words.clone()),
            ("V", numbers.conveyor.clone()),
            ("F", features.clone()),
            ("H", hidden.clone()),
            ("E", numbers.widest.clone()),
            ("EL", experts.clone()),
            ("GV", gate_conveyor),
        ];
        let step = self.part(graph, BankPart::Step)?;
        let mut slots = self.part(graph, BankPart::ScratchTable)?;
        let mut scratch = self.parts(graph, |kind| BankPart::Scratch { kind })?;
        let conveyor = self.parts(graph, |kind| BankPart::Conveyor { kind })?;
        let seeds = self.part(graph, BankPart::SeedList)?;
        let tiered = self.part(graph, BankPart::TierList)?;
        // Each round's convert list staged into device memory, which its
        // converts read.
        let mut staged_list = self.part(graph, BankPart::ConvertStaged)?;
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
        let mut seed_grid = entries
            .seeding
            .map(|_| Ok::<_, GraphError>((self.part(graph, BankPart::SeedGrid)?, self.part(graph, BankPart::SeedStaged)?)))
            .transpose()?;
        let mut commits = entries.seeding.map(|_| self.part(graph, BankPart::Commits)).transpose()?;
        // Round 0 always parks (hit against cold is final only as the request
        // is served); its park's return carries the count of records the
        // rounds convert, from which `routed_scratch_grids` writes every
        // round's grids: a round converts only the records it has, and a
        // later round parks while it has records, or the round before has
        // and no sibling fill follows (the arrival after the last round's
        // converts frees their records, `ExpertBank::fill_followed`).
        let mut grids = self.part(graph, BankPart::ScratchGrids)?;
        let always = GraphConstant::i32(graph, &[1, 1, 1])?;
        let grid = |grids: &WorkflowTensor, round: u64, grid: u64| {
            grids
                .slice_leading(round, round + 1)
                .reshape(&[ROUND_GRIDS, 3])
                .slice_leading(grid, grid + 1)
                .reshape(&[3])
        };
        // The park binds the conveyor as the host's: before answering it,
        // the host has read each listed expert into its record, which the
        // round's converts read in place.
        let conveyor_gate = GateRecords::of(&conveyor);
        // The layer's map, which the host rewrites as it serves the request
        // (its hits final and the lent records' experts unmapped by round 0's
        // answer): each park is its writer, so the pass's map
        // (`routed_pass_map`) reads it only after the park, on every backend,
        // not only where launches keep submission order.
        let map = self.part(graph, BankPart::SlotMap)?;
        // The rounds repeat the topology: a trace is its rounds'.
        let rounds = numbers.rounds.pin();
        for round in 0..rounds {
            let list = {
                let round = u8::try_from(round).map_err(|_| "scratch rounds exceed u8")?;
                self.part(graph, BankPart::ConvertList { round })?
            };
            let park = grid(grids.tensor(), round, PARK_GRID);
            self.park(graph, ParkKind::Round(round));
            graph.enqueue(
                entries.scratch_wait,
                &wait,
                routed_scratch_wait::WorkflowArgs {
                    step: step.tensor().into(),
                    scratch: slots.tensor_mut().into(),
                    round: i32::try_from(round).map_err(|_| "scratch rounds exceed i32")?.into(),
                    list: list.tensor().into(),
                    tiered: tiered.tensor().into(),
                    conveyor_gate: conveyor_gate.arg(),
                    conveyor_up: conveyor.up.tensor().into(),
                    conveyor_down: conveyor.down.tensor().into(),
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
                    &[("FLIGHT", scratch_slots.clone()), ("LANES", scratch_lanes.clone()), ("R", numbers.rounds.clone())],
                    routed_scratch_grids::WorkflowArgs {
                        step: step.tensor().into(),
                        scratch: slots.tensor().into(),
                        grids: grids.tensor_mut().into(),
                        round_records: numbers.round_records.word().ok_or("round records exceed i32")?,
                        rows_groups: numbers.rows_groups.word().ok_or("convert groups exceed i32")?,
                        columns_groups: numbers.columns_groups.word().ok_or("convert groups exceed i32")?,
                    },
                )?;
                // Beside a host tier, the tier copy: each expert of the step
                // the tier keeps whole, which round 0's answer lists, read
                // from its tier record in place into its scratch slot, on
                // the tier list's grid.
                if let Some((records, tier_grid, staged)) = tier.as_mut() {
                    let seeding = entries.seeding.ok_or("a host tier's bank seeds by the device")?;
                    graph.enqueue(
                        seeding.list_grid,
                        &[("L", numbers.widest.clone())],
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
                                ("D", experts.clone()),
                                ("L", numbers.widest.clone()),
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
                // The seed list's grid, where the device seeds: its host
                // lists the group's last step's seeds before acknowledging
                // round 0, and no list of another step launches the seed
                // copy.
                if let (Some(seeding), Some((seed_grid, staged))) = (entries.seeding, seed_grid.as_mut()) {
                    graph.enqueue(
                        seeding.list_grid,
                        &[("L", numbers.widest.clone())],
                        routed_list_grid::WorkflowArgs {
                            list: seeds.tensor().into(),
                            step: step.tensor().into(),
                            grid: seed_grid.tensor_mut().into(),
                            staged: staged.tensor_mut().into(),
                        },
                    )?;
                }
            }
            graph.enqueue(
                entries.list_stage,
                &[("L", numbers.round_records.clone())],
                routed_list_stage::WorkflowArgs {
                    list: list.tensor().into(),
                    step: step.tensor().into(),
                    staged: staged_list.tensor_mut().into(),
                },
            )?;
            let tensors = numbers.matrix.as_ref().zip(entries.convert).zip(conveyor.as_ref());
            for (kind, ((([n, k], convert), conveyor), scratch)) in tensors.zip(scratch.as_mut()).into_tensors() {
                let converts = grid(grids.tensor(), round, record_grid(kind));
                graph.enqueue(
                    convert,
                    &[
                        ("V", numbers.conveyor.clone()),
                        ("L", numbers.round_records.clone()),
                        ("S", experts.clone()),
                        ("N", n.clone()),
                        ("K", k.clone()),
                    ],
                    routed_convert::WorkflowArgs {
                        conveyor: conveyor.tensor().into(),
                        list: staged_list.tensor().into(),
                        step: step.tensor().into(),
                        destination: scratch.tensor_mut().into(),
                        grid: (&converts).into(),
                    },
                )?;
            }
        }
        constants.push(always);
        // Where the device copies into the bank, a prompt group's last step
        // seeds it from the filled tier: its listed experts copied into
        // their records on the seed list's grid, then each record
        // committed, which the host maps once the step has completed. A list
        // of another step, or a failed step, seeds nothing. Where the host
        // lands, nothing seeds.
        let (Some(seeding), Some(records), Some((seed_grid, staged)), Some(commits)) =
            (entries.seeding, records.as_mut(), seed_grid.as_ref(), commits.as_mut())
        else {
            // Its last write: the scratch tier's up records, which every
            // round converts into.
            return Ok(scratch.up.tensor().clone());
        };
        // The seeds' records from the first a seed takes
        // (`ExpertBankPlan::seeded_from`), which the seed list names them
        // from: beside a host tier the lender's lie past the lent records
        // the scratch fills, so the copy's destination is disjoint from its
        // source by construction. A lender with no record past them seeds
        // nothing.
        let first = &numbers.seeded_from;
        let seeded = numbers.records.checked_sub(first).ok_or("a bank group's lent records exceed its records")?;
        if seeded.is_zero() {
            return Ok(scratch.up.tensor().clone());
        }
        let tensors = numbers.matrix.as_ref().zip(seeding.seed).zip(scratch.as_ref()).zip(records.as_ref());
        for ((([n, k], seed), scratch), records) in tensors {
            let mut destination = records.tensor().slice_leading(first, &numbers.records);
            graph.enqueue(
                seed,
                &[
                    ("S", experts.clone()),
                    ("D", seeded.clone()),
                    ("L", numbers.widest.clone()),
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
                ("X", numbers.widest.clone()),
                ("BANK", seeded),
                ("FLIGHT", scratch_slots.clone()),
                ("LANES", scratch_lanes.clone()),
            ],
            routed_seed_commit::WorkflowArgs {
                list: seeds.tensor().into(),
                step: step.tensor().into(),
                scratch: slots.tensor().into(),
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

/// The ports of a bank-served layer's hints (`routed_route_streamed`,
/// `routed_select_streamed`): `AHEAD`; per lead, the weights its route reads
/// of the layer it predicts there (`ExpertBankPlan::predicted`: the next
/// streamed layer, then the one after it) in its own place, in the route's
/// order (`BankLayer::hinted`); the next layer's policy words, whose
/// POLICY_RESOLVED encodes the route without its hints while set (nothing
/// there can be absent, so nothing reads them); and its rings. A lead past
/// the route's `AHEAD` binds the next layer's weights again, which the route
/// does not read, so every lead's weights are of the experts the route's
/// `EN` names (`ExpertBankPlan::lead_layer`); the last streamed layer, which
/// hints nothing, binds its own weights and policy (its two launches
/// alike).
pub(crate) struct HintPorts<const N: usize> {
    pub ahead: u64,
    pub weights: [[WorkflowTensor; N]; crate::planning::HINT_LEADS as usize],
    pub policy: WorkflowTensor,
    pub rings: NativePort,
}

impl<'a, G: GraphDraft + 'a, W: PlannedWeights + ?Sized> StreamedLayer<'_, 'a, G, W> {
    /// The hint ports of this layer, whose own hinted weights, in its
    /// route's order (`operators::routed::hint_kinds`), and policy words are
    /// `own` and `policy`. A weight the next layer has is its own; one it
    /// lacks is this layer's port (an absent norm, a bias's zeros), as the
    /// plan found every streamed layer's hinted weights alike. The last
    /// layer's hints bind its own words, which are not read.
    pub(crate) fn hints<const N: usize>(
        &mut self,
        graph: &mut G,
        weights: &mut Vec<(WeightPort, NativePort)>,
        own: [&WorkflowTensor; N],
        policy: &WorkflowTensor,
    ) -> Result<HintPorts<N>, GraphError> {
        let topology = self.topology;
        let ahead = topology.ahead;
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
        Ok(HintPorts {
            ahead,
            weights: hinted,
            policy,
            rings: self.part(graph, BankPart::Rings)?,
        })
    }
}

/// What a bank-served layer's route, tier wait and site bind, one port
/// each: the step (set here when the layer sets it, with the run's
/// sequence input), its group's map, its file table of the exchange, and
/// where the bank has a host tier the layer's tier map and its group's
/// tier records (the streaming header's requests in routed.seismic).
pub(crate) struct DecodePorts {
    pub step: NativePort,
    pub sequence: Option<NativePort>,
    pub map: NativePort,
    pub tier: Option<TierPorts>,
    /// The layer's policy words (`BankPart::Policy`): while its POLICY_WHOLE
    /// is set its tier wait and site are not encoded and its route is the
    /// whole form; its lines are the route's assignment's.
    pub policy: NativePort,
    /// The layer's file table (`BankPart::FileTable`).
    pub file: NativePort,
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
/// later entries read: the step's sources and its grids, the tier wait's
/// (`routed.seismic`'s `resolve`).
pub(crate) struct RouteLocals {
    pub sources: NativePort,
    pub grids: NativePort,
}

impl RouteLocals {
    /// The grid `grid` of the route's grids (`WAIT_GRID`).
    pub(crate) fn grid(&self, grid: u64) -> WorkflowTensorView {
        self.grids.tensor().slice_leading(grid, grid + 1).reshape(&[3])
    }

    /// The step's sources snapshot, which every later decode entry of the
    /// layer reads.
    pub(crate) fn snapshot(&self) -> WorkflowTensorView {
        self.sources.tensor().slice_leading(0, magnitude_kernels::constants::SOURCE_ROWS)
    }
}

/// What a bank-served route binds of its layer's host tier: the tier map's
/// words (`TM`: the layer's experts, none without a tier) and the tensor, a
/// graph constant of no words without a tier.
pub(crate) struct RouteTier {
    pub experts: Dim,
    pub map: WorkflowTensor,
}

/// What a decode tier wait binds: the layer's decode ports, and the sources
/// its route wrote and the wait's grid of its grids.
pub(crate) struct WaitPorts<'f> {
    pub decode: &'f mut DecodePorts,
    pub sources: &'f mut WorkflowTensor,
    pub grid: &'f WorkflowTensorView,
    /// The stacks' ports, the bank's records where the host places, which
    /// the misses the device lands are converted into.
    pub records: RecordTensors<&'f mut NativePort>,
}

/// Whether `load` streams the experts of the feed-forward of `scope`.
pub(crate) fn streams(load: &ModelLoadPlan, scope: WeightScope) -> bool {
    load.expert_bank().is_some_and(|bank| bank.feed_forward(scope).is_some())
}

/// Whether the streamed feed-forward of `scope` has a resident form
/// (`RunStage::Resident`): its bank can hold every expert
/// (`ExpertBankPlan::reaches_whole`), so it can be whole.
pub(crate) fn resident_form(load: &ModelLoadPlan, scope: WeightScope) -> bool {
    streams(load, scope) && load.expert_bank().is_some_and(ExpertBankPlan::reaches_whole)
}

impl<'a, G: GraphDraft + 'a, W: PlannedWeights + ?Sized> StreamedLayer<'_, 'a, G, W> {
    pub(crate) fn decode_ports(&mut self, graph: &mut G) -> Result<DecodePorts, GraphError> {
        let (step, sequence) = self.step(graph)?;
        let tier = match self.numbers.tier.is_zero() {
            true => None,
            false => Some(TierPorts {
                map: self.part(graph, BankPart::TierMap)?,
                records: self.parts(graph, |kind| BankPart::Tier { kind })?,
            }),
        };
        Ok(DecodePorts {
            step,
            sequence,
            map: self.part(graph, BankPart::Map)?,
            tier,
            policy: self.part(graph, BankPart::Policy)?,
            file: self.part(graph, BankPart::FileTable)?,
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
