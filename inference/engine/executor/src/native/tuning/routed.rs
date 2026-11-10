//! Tuning cases of the routed (mixture-of-experts) entries. `routed_route`
//! serves every row class; the decode entries (`routed_expand`, or
//! `routed_route_shared` and `routed_gate_up` where the backend declares the
//! former, and `routed_output`) are tuned at the decode row points, the grouped
//! entries (`routed_group`, `routed_experts`,
//! `routed_combine`) at the grouped ones. Routing tables are synthetic:
//! every row selects distinct experts with the uneven expert loads of real
//! routing (`routes`), and the grouped tables are formed from them exactly as
//! `routed_group` forms them.
//!
//! Route and group write their tables through `&mut` parameters; each
//! argument set owns those tables as case state.

use super::cases::scale_extent;
use super::{
    CaseState, EntryTuning, ModelInputs, PointShape, TuningInputs, TuningLimits, row_points,
    served_row_points,
};
use crate::operators::routed::fused_graph::{DECODE_ROWS, TILE_ROWS, grouped_blocks};
use crate::planning::{BankBinding, HINT_LEADS, HintWeight};
use magnitude_family_contracts::{GluLimit, Operator, RouteNormalization, WeightKind, WeightScope};
use magnitude_kernels::constants::{
    DECODE_GRIDS, MAP_RECORD, MAP_WORDS, POLICY_PRICED, POLICY_WORDS, SOURCE_ROWS, STEP_LOW, STEP_WORDS,
};
use magnitude_kernels::{
    routed_combine, routed_expand, routed_experts, routed_gate_up, routed_group, routed_output,
    routed_output_project, routed_route, routed_route_shared, routed_route_streamed,
};
use seismic::{BuildError, Element, Tensor};

/// The routed geometry every case of one binding shares.
#[derive(Clone, Copy, Debug)]
pub(crate) struct RoutedShape {
    pub hidden: u64,
    pub experts: u64,
    pub selected: u64,
    pub features: u64,
    pub shared: u64,
}

/// The decode points: row counts the decode form serves.
pub(crate) fn decode_points(limits: TuningLimits) -> Vec<PointShape> {
    served_row_points(limits, |rows| rows <= DECODE_ROWS)
}

/// The grouped points: row counts past the decode form.
pub(crate) fn grouped_points(limits: TuningLimits) -> Vec<PointShape> {
    served_row_points(limits, |rows| rows > DECODE_ROWS)
}

/// Distinct experts per row with the uneven expert loads of real routing: the
/// expert of popularity rank r is (101 r) mod E (a permutation: 101 is prime
/// and no expert count is a multiple of it) with popularity 1 / sqrt(r + 1),
/// and each row draws its choices by popularity without replacement from a
/// fixed-seed generator. A grouped point then holds blocks of every live-row
/// count, from one row to full tiles, and experts with no rows.
pub(crate) fn routes(rows: u64, shape: RoutedShape) -> Vec<i32> {
    let experts = shape.experts as usize;
    let popularity = (0..experts)
        .map(|rank| 1.0 / ((rank + 1) as f64).sqrt())
        .collect::<Vec<_>>();
    let total = popularity.iter().sum::<f64>();
    let mut state = 0x9e37_79b9_7f4a_7c15_u64;
    let mut routes = Vec::with_capacity(rows as usize * shape.selected as usize);
    for _ in 0..rows {
        let mut taken = vec![false; experts];
        let mut remaining = total;
        for _ in 0..shape.selected {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let mut target = (state >> 11) as f64 / (1_u64 << 53) as f64 * remaining;
            // The last untaken expert absorbs the rounding of `remaining`.
            let mut rank = 0;
            for candidate in (0..experts).filter(|&candidate| !taken[candidate]) {
                rank = candidate;
                target -= popularity[candidate];
                if target < 0.0 {
                    break;
                }
            }
            taken[rank] = true;
            remaining -= popularity[rank];
            routes.push(((rank * 101) % experts) as i32);
        }
    }
    routes
}

/// The tables `routed_group` forms from `routes`: (order [B, T],
/// inverse [M, K], blocks [B]).
pub(crate) fn group(
    routes: &[i32],
    rows: u64,
    shape: RoutedShape,
) -> Result<(Vec<i32>, Vec<i32>, Vec<i32>), String> {
    let blocks = grouped_blocks(rows, shape.experts, shape.selected)?.fixed();
    let tile = TILE_ROWS as usize;
    let mut order = vec![-1; blocks as usize * tile];
    let mut inverse = vec![0; routes.len()];
    let mut table = vec![-1; blocks as usize];
    let mut block = 0usize;
    for expert in 0..shape.experts as i32 {
        let mut lane = 0usize;
        for (flat, route) in routes.iter().enumerate() {
            if *route != expert {
                continue;
            }
            table[block] = expert;
            order[block * tile + lane] = (flat / shape.selected as usize) as i32;
            inverse[flat] = (block * tile + lane) as i32;
            lane += 1;
            if lane == tile {
                lane = 0;
                block += 1;
            }
        }
        if lane > 0 {
            block += 1;
        }
    }
    Ok((order, inverse, table))
}

fn scores(
    inputs: &TuningInputs<'_, '_>,
    rows: u64,
    shape: RoutedShape,
    seed: u64,
) -> Result<Tensor, BuildError> {
    inputs.activation(Element::f32(), &[rows, shape.selected], seed)
}

/// `routed_route`: RMS prologue, router softmax, top-k selection and the
/// shared expert's coefficient. `SIMDGROUPS` reassociates the router's
/// reductions (arithmetic); its integer routes are still compared exactly,
/// so a configuration that flips a near-tie choice on the tuning rows is
/// not chosen.
#[derive(Clone)]
pub(crate) struct RoutedRouteTuning {
    pub norm: Element,
    pub router: Element,
    pub activation: Element,
    pub shape: RoutedShape,
    pub scopes: Vec<WeightScope>,
    pub epsilon: f32,
    /// Whether the route serves the decode rows too (the expand form), or
    /// only the grouped rows (the shared-route form).
    pub decode: bool,
}

pub(crate) struct RoutedRouteCase {
    residual: Tensor,
    norm: Tensor,
    router: Tensor,
    shared_router: Tensor,
    routes: CaseState,
    scores: CaseState,
    epsilon: f32,
    normalize: i32,
    /// The down weight's per-expert scales the scores fold in (ES).
    expert_scale: Tensor,
}

impl RoutedRouteTuning {
    fn elements(&self) -> routed_route::Elements {
        routed_route::Elements {
            NW: self.norm,
            RW: self.router,
            A: self.activation,
        }
    }

    /// Whether the model divides the selected scores by their sum.
    fn normalize(&self, inputs: &TuningInputs<'_, '_>) -> Result<bool, String> {
        match inputs.operator(&self.scopes)? {
            Operator::RoutedFfn(routed) => {
                Ok(routed.router.normalization == RouteNormalization::Sum)
            }
            other => Err(format!(
                "a {} layer has no routed feed-forward",
                other.name()
            )),
        }
    }
}

impl EntryTuning for RoutedRouteTuning {
    type Entry = routed_route::Entry;
    type Case = RoutedRouteCase;

    fn launches(&self) -> usize {
        self.scopes.len()
    }

    fn bindings(&self) -> String {
        format!(
            "NW={},RW={},A={}",
            self.norm.name(),
            self.router.name(),
            self.activation.name()
        )
    }

    fn statics(&self, inputs: &ModelInputs<'_>) -> Result<Vec<(&'static str, u64)>, String> {
        let shape = self.shape;
        let (parts, norm) = inputs.input_form(&self.scopes, shape.hidden)?;
        Ok(vec![
            ("H", shape.hidden),
            ("E", shape.experts),
            ("K", shape.selected),
            ("PT", parts),
            ("NM", norm),
            ("ES", scale_extent(inputs, &self.scopes, WeightKind::ExpertDown)?),
        ])
    }

    fn points(&self, limits: TuningLimits) -> Vec<PointShape> {
        if self.decode {
            row_points(limits)
        } else {
            grouped_points(limits)
        }
    }

    fn rotation(
        &self,
        inputs: &mut TuningInputs<'_, '_>,
        point: &PointShape,
    ) -> Result<Vec<Self::Case>, BuildError> {
        let (shape, rows) = (self.shape, point.rows);
        let normalize = i32::from(self.normalize(inputs)?);
        let (parts, norm) = inputs.input_form(&self.scopes, shape.hidden)?;
        TuningInputs::rotation_scopes(&self.scopes, point)
            .into_iter()
            .enumerate()
            .map(|(index, scope)| {
                let routes = inputs.scratch(Element::i32(), &[rows, shape.selected])?;
                let scores = inputs.scratch(Element::f32(), &[rows, shape.selected])?;
                Ok(RoutedRouteCase {
                    residual: inputs.activation(
                        Element::f32(),
                        &[rows, parts * shape.hidden],
                        index as u64 + 1,
                    )?,
                    norm: inputs.input_norm(scope, WeightKind::InputNorm, self.norm, norm)?,
                    router: inputs.weight(scope, WeightKind::Router)?,
                    shared_router: inputs.weight(scope, WeightKind::SharedRouter)?,
                    routes: inputs.state(routes, 0..rows)?,
                    scores: inputs.state(scores, 0..rows)?,
                    epsilon: self.epsilon,
                    normalize,
                    expert_scale: inputs
                        .unit_scale(inputs.scale_extent(scope, WeightKind::ExpertDown)?)?,
                })
            })
            .collect()
    }

    fn args<'a>(case: &'a mut Self::Case) -> routed_route::Args<'a> {
        routed_route::Args {
            residual: &case.residual,
            norm: &case.norm,
            router: &case.router,
            shared_router: &case.shared_router,
            routes: case.routes.tensor_mut(),
            scores: case.scores.tensor_mut(),
            eps: case.epsilon,
            normalize: case.normalize,
            expert_scale: &case.expert_scale,
        }
    }

    fn state(case: &Self::Case) -> Vec<(&'static str, &CaseState)> {
        vec![("routes", &case.routes), ("scores", &case.scores)]
    }

    generated_entry!(routed_route, this => this.elements());
}

/// `routed_route_shared`: `routed_route` with the shared expert's gate/up
/// (decode rows). SIMDGROUPS and ROWS never change bits; LANES reassociates
/// the shared expert's reductions (arithmetic). Routes are compared exactly,
/// as for `routed_route`.
#[derive(Clone)]
pub(crate) struct RoutedRouteSharedTuning {
    pub route: RoutedRouteTuning,
    pub shared_gate: Element,
    pub shared_up: Element,
}

pub(crate) struct RoutedRouteSharedCase {
    route: RoutedRouteCase,
    shared_gate: Tensor,
    shared_up: Tensor,
}

impl EntryTuning for RoutedRouteSharedTuning {
    type Entry = routed_route_shared::Entry;
    type Case = RoutedRouteSharedCase;

    fn launches(&self) -> usize {
        self.route.scopes.len()
    }

    fn bindings(&self) -> String {
        format!(
            "NW={},RW={},SGW={},SUW={},A={}",
            self.route.norm.name(),
            self.route.router.name(),
            self.shared_gate.name(),
            self.shared_up.name(),
            self.route.activation.name()
        )
    }

    fn statics(&self, inputs: &ModelInputs<'_>) -> Result<Vec<(&'static str, u64)>, String> {
        let shape = self.route.shape;
        let (parts, norm) = inputs.input_form(&self.route.scopes, shape.hidden)?;
        Ok(vec![
            ("H", shape.hidden),
            ("E", shape.experts),
            ("K", shape.selected),
            ("S", shape.shared),
            ("PT", parts),
            ("NM", norm),
            ("ES", scale_extent(inputs, &self.route.scopes, WeightKind::ExpertDown)?),
        ])
    }

    fn points(&self, limits: TuningLimits) -> Vec<PointShape> {
        decode_points(limits)
    }

    fn rotation(
        &self,
        inputs: &mut TuningInputs<'_, '_>,
        point: &PointShape,
    ) -> Result<Vec<Self::Case>, BuildError> {
        let scopes = TuningInputs::rotation_scopes(&self.route.scopes, point);
        self.route
            .rotation(inputs, point)?
            .into_iter()
            .zip(scopes)
            .map(|(route, scope)| {
                Ok(RoutedRouteSharedCase {
                    route,
                    shared_gate: inputs.weight(scope, WeightKind::SharedGate)?,
                    shared_up: inputs.weight(scope, WeightKind::SharedUp)?,
                })
            })
            .collect()
    }

    fn args<'a>(case: &'a mut Self::Case) -> routed_route_shared::Args<'a> {
        let route = &mut case.route;
        routed_route_shared::Args {
            residual: &route.residual,
            norm: &route.norm,
            router: &route.router,
            shared_router: &route.shared_router,
            shared_gate: &case.shared_gate,
            shared_up: &case.shared_up,
            routes: route.routes.tensor_mut(),
            scores: route.scores.tensor_mut(),
            eps: route.epsilon,
            normalize: route.normalize,
            expert_scale: &route.expert_scale,
        }
    }

    fn state(case: &Self::Case) -> Vec<(&'static str, &CaseState)> {
        RoutedRouteTuning::state(&case.route)
    }

    generated_entry!(routed_route_shared, this => routed_route_shared::Elements {
        NW: this.route.norm,
        RW: this.route.router,
        SGW: this.shared_gate,
        SUW: this.shared_up,
        A: this.route.activation,
    });
}

/// The layer each lead of a streamed layer's route binds
/// (`ExpertBankPlan::lead_layer`), each with what the route reads of it
/// (`BankLayer::hinted`); none for the last streamed layer, which hints
/// nothing.
pub(crate) fn streamed_hint<'p>(
    inputs: &ModelInputs<'p>,
    scope: WeightScope,
) -> Result<Vec<(WeightScope, &'p [HintWeight])>, String> {
    let bank = inputs
        .load
        .expert_bank()
        .ok_or("a streamed layer's load holds no bank")?;
    let (index, _) = bank
        .layer(scope)
        .ok_or_else(|| format!("{scope:?} is not a streamed layer"))?;
    // Each lead's layer (`ExpertBankPlan::lead_layer`); none where the
    // layer predicts none and binds its own.
    Ok((0..HINT_LEADS as usize)
        .map(|lead| bank.lead_layer(index, lead))
        .filter(|&layer| layer != index)
        .map(|layer| (bank.layers()[layer].scope, bank.layers()[layer].hinted.as_slice()))
        .collect())
}

/// The hint ports of kinds `kinds` at every lead a route binds: of each
/// lead, the layer `hinted` (`streamed_hint`) names there; of every lead,
/// its `own` where `hinted` names none.
pub(crate) fn lead_ports<const N: usize>(
    inputs: &mut TuningInputs<'_, '_>,
    hinted: &[(WeightScope, &[HintWeight])],
    kinds: [WeightKind; N],
    own: [&Tensor; N],
) -> Result<[[Tensor; N]; HINT_LEADS as usize], BuildError> {
    let mut leads = Vec::with_capacity(HINT_LEADS as usize);
    for lead in 0..HINT_LEADS as usize {
        let mut ports = Vec::with_capacity(N);
        for (&kind, own) in kinds.iter().zip(own) {
            ports.push(hint_port(inputs, hinted.get(lead).copied(), kind, own)?);
        }
        leads.push(<[Tensor; N]>::try_from(ports).unwrap_or_else(|_| unreachable!("N ports")));
    }
    Ok(<[[Tensor; N]; HINT_LEADS as usize]>::try_from(leads).unwrap_or_else(|_| unreachable!("a port set per lead")))
}

/// A hint port of the layer `hinted` predicts, of `kind`: that layer's own
/// weight where the route reads it (`HintWeight::element`), else `own`,
/// the hinting layer's port, as serving binds them
/// (`operators::routed::streamed::StreamedLayer::hints`).
pub(crate) fn hint_port(
    inputs: &mut TuningInputs<'_, '_>,
    hinted: Option<(WeightScope, &[HintWeight])>,
    kind: WeightKind,
    own: &Tensor,
) -> Result<Tensor, BuildError> {
    let next = hinted.filter(|(_, hints)| {
        hints
            .iter()
            .any(|hint| hint.kind == kind && hint.element.is_some())
    });
    match next {
        Some((scope, _)) => inputs.weight(scope, kind),
        None => Ok(own.clone()),
    }
}

/// The bank operands of a streamed layer's route or selection
/// (`routed_route_streamed`, `routed_select_streamed`) at one point of
/// `rows` rows, in the storage the bank holds each in
/// (`ExpertBankPlan::layout`: its words host-shared, its grids and the
/// step's sources the device's): a group map of the layer's experts alone
/// (`home` 0) that holds none of them, and where the bank has a host tier a
/// tier map that holds none either, so every choice is requested and, across
/// a link, assigned (the resolve's most work); policy words clear (no layer
/// whole), so the route resolves and, at an `AHEAD` above 0, hints the
/// layers ahead while the next lacks an expert (the launch with every piece of work); and the step's
/// sequence 1. The words it
/// writes are states.
pub(crate) struct BankOperands {
    pub map: Tensor,
    pub tier: Tensor,
    pub policy: Tensor,
    /// The route's `sited`: a site's where the bank has a host tier (a
    /// sited layer, the route's costlier form), none without one.
    pub sited: i32,
    pub step: Tensor,
    /// The file table (`BankPart::FileTable`), whose step's slot the route
    /// requests.
    pub file: CaseState,
    pub grids: CaseState,
    pub sources: CaseState,
    pub next_policy: Tensor,
    pub rings: CaseState,
    pub routing: CaseState,
}

impl BankOperands {
    pub fn new(
        inputs: &mut TuningInputs<'_, '_>,
        bank: BankBinding,
        ahead: u64,
        rows: u64,
    ) -> Result<Self, BuildError> {
        let step = words(inputs, &[STEP_WORDS])?;
        step.store_release(4 * STEP_LOW, 1)?;
        // Priced, over zero lines: a sited route's assignment runs.
        let policy = words(inputs, &[POLICY_WORDS])?;
        policy.store_release(4 * POLICY_PRICED, 1)?;
        let (ring, flight) = (bank.words.ring_words(), bank.words.flight);
        Ok(Self {
            map: unmapped(inputs, &[bank.experts, MAP_WORDS], MAP_WORDS, MAP_RECORD)?,
            tier: unmapped(inputs, &[tier_experts(bank)], 1, 0)?,
            policy,
            sited: i32::from(bank.tier > 0),
            step,
            file: written(inputs, words(inputs, &bank.words.file().extents())?)?,
            grids: written(inputs, inputs.scratch(Element::i32(), &[DECODE_GRIDS, 3])?)?,
            sources: written(inputs, inputs.scratch(Element::i32(), &[SOURCE_ROWS, rows, bank.selected])?)?,
            next_policy: words(inputs, &[POLICY_WORDS])?,
            rings: written(inputs, words(inputs, &[ahead, ring])?)?,
            routing: written(inputs, words(inputs, &[flight, bank.words.routing_words()])?)?,
        })
    }

    pub fn states(&self) -> [(&'static str, &CaseState); 5] {
        [
            ("file", &self.file),
            ("grids", &self.grids),
            ("sources", &self.sources),
            ("rings", &self.rings),
            ("routing", &self.routing),
        ]
    }
}

/// The experts a streamed layer's tier map names (`TM`): its experts where
/// the bank has a host tier, none without one.
pub(crate) fn tier_experts(bank: BankBinding) -> u64 {
    match bank.tier {
        0 => 0,
        _ => bank.experts,
    }
}

/// Zeroed i32 words of `extents`, host-shared as the bank holds them.
fn words(inputs: &TuningInputs<'_, '_>, extents: &[u64]) -> Result<Tensor, BuildError> {
    Ok(Tensor::shared(inputs.device, Element::i32(), extents)?)
}

/// Words of `extents` whose record word (`record` of each row of `width`
/// words) names none (-1): a map or tier map that holds nothing.
fn unmapped(inputs: &TuningInputs<'_, '_>, extents: &[u64], width: u64, record: u64) -> Result<Tensor, BuildError> {
    let map = words(inputs, extents)?;
    let rows = extents.iter().product::<u64>() / width;
    for row in 0..rows {
        map.store_release(4 * (row * width + record), u32::from_ne_bytes((-1_i32).to_ne_bytes()))?;
    }
    Ok(map)
}

/// `tensor` as a state the entry may write whole.
fn written(inputs: &TuningInputs<'_, '_>, tensor: Tensor) -> Result<CaseState, BuildError> {
    let rows = tensor.extents()[0];
    inputs.state(tensor, 0..rows)
}

/// `routed_route_streamed`: a streamed layer's decode route, its resident
/// route's (`routed_route_shared`) launch and parameters with the resolve
/// and, at `AHEAD` 1, the next layer's hints. It follows the resident
/// route's unit (`Specializer::tuned_following`): LANES (Metal) or KSPLIT
/// (CUDA), which set the shared expert's reduction order, are the resident
/// choice's, so its bits are the resident's; SIMDGROUPS and ROWS or TPW,
/// which change no bit, are searched for its own launch, which does more
/// than the resident's. Its points are the decode points over the layers
/// that launch it, its operands the resident's over them with the next
/// layer's hinted weights and a bank's ([`BankOperands`]).
#[derive(Clone)]
pub(crate) struct RoutedRouteStreamedTuning {
    /// The resident route it follows, over every layer of the binding: its
    /// static values are the streamed route's.
    pub resident: RoutedRouteSharedTuning,
    /// The streamed layers that launch it.
    pub layers: Vec<WeightScope>,
    pub bank: BankBinding,
    pub ahead: u64,
}

pub(crate) struct RoutedRouteStreamedCase {
    route: RoutedRouteSharedCase,
    /// Each lead's norm and router.
    leads: [[Tensor; 2]; HINT_LEADS as usize],
    bank: BankOperands,
}

impl RoutedRouteStreamedTuning {
    fn elements(&self) -> routed_route_streamed::Elements {
        let route = &self.resident.route;
        routed_route_streamed::Elements {
            NW: route.norm,
            RW: route.router,
            SGW: self.resident.shared_gate,
            SUW: self.resident.shared_up,
            A: route.activation,
        }
    }
}

impl EntryTuning for RoutedRouteStreamedTuning {
    type Entry = routed_route_streamed::Entry;
    type Case = RoutedRouteStreamedCase;

    fn launches(&self) -> usize {
        self.layers.len()
    }

    fn bindings(&self) -> String {
        self.resident.bindings()
    }

    fn statics(&self, inputs: &ModelInputs<'_>) -> Result<Vec<(&'static str, u64)>, String> {
        let mut statics = self.resident.statics(inputs)?;
        statics.extend(bank_statics(self.bank, self.ahead));
        Ok(statics)
    }

    fn points(&self, limits: TuningLimits) -> Vec<PointShape> {
        decode_points(limits)
    }

    fn rotation(
        &self,
        inputs: &mut TuningInputs<'_, '_>,
        point: &PointShape,
    ) -> Result<Vec<Self::Case>, BuildError> {
        let mut over = self.resident.clone();
        over.route.scopes = self.layers.clone();
        let scopes = TuningInputs::rotation_scopes(&self.layers, point);
        over.rotation(inputs, point)?
            .into_iter()
            .zip(scopes)
            .map(|(route, scope)| {
                let hinted = streamed_hint(inputs, scope)?;
                Ok(RoutedRouteStreamedCase {
                    leads: lead_ports(
                        inputs,
                        &hinted,
                        [WeightKind::InputNorm, WeightKind::Router],
                        [&route.route.norm, &route.route.router],
                    )?,
                    bank: BankOperands::new(inputs, self.bank, self.ahead, point.rows)?,
                    route,
                })
            })
            .collect()
    }

    fn args<'a>(case: &'a mut Self::Case) -> routed_route_streamed::Args<'a> {
        let RoutedRouteStreamedCase {
            route: shared,
            leads: [[next_norm, next_router], [after_norm, after_router]],
            bank,
        } = case;
        let route = &mut shared.route;
        routed_route_streamed::Args {
            residual: &route.residual,
            norm: &route.norm,
            router: &route.router,
            shared_router: &route.shared_router,
            shared_gate: &shared.shared_gate,
            shared_up: &shared.shared_up,
            routes: route.routes.tensor_mut(),
            scores: route.scores.tensor_mut(),
            eps: route.epsilon,
            normalize: route.normalize,
            expert_scale: &route.expert_scale,
            map: &bank.map,
            tier: &bank.tier,
            policy: &bank.policy,
            sited: bank.sited,
            step: &bank.step,
            file: bank.file.tensor_mut(),
            grids: bank.grids.tensor_mut(),
            sources: bank.sources.tensor_mut(),
            next_norm,
            next_router,
            after_norm,
            after_router,
            next_policy: &bank.next_policy,
            rings: bank.rings.tensor_mut(),
            routing: bank.routing.tensor_mut(),
        }
    }

    fn state(case: &Self::Case) -> Vec<(&'static str, &CaseState)> {
        let mut states = RoutedRouteSharedTuning::state(&case.route);
        states.extend(case.bank.states());
        states
    }

    generated_entry!(routed_route_streamed, this => this.elements());
}

/// The static values a streamed layer's route or selection adds to its
/// resident entry's: its bank's words at `ahead` and its tier map's experts
/// (`TM`, [`tier_experts`]).
pub(crate) fn bank_statics(bank: BankBinding, ahead: u64) -> [(&'static str, u64); 3] {
    [
        ("AHEAD", ahead),
        ("EN", bank.next_experts),
        ("TM", tier_experts(bank)),
    ]
}

/// `routed_group`: the grouped form's expert tiles, formed from the
/// routes. Its parameters are mappings.
#[derive(Clone)]
pub(crate) struct RoutedGroupTuning {
    pub shape: RoutedShape,
    /// Layers prepared with this specialization.
    pub layers: usize,
}

pub(crate) struct RoutedGroupCase {
    first: Tensor,
    rows: Tensor,
    routes: Tensor,
    counts: CaseState,
    order: CaseState,
    inverse: CaseState,
    blocks: CaseState,
    used: CaseState,
}

impl EntryTuning for RoutedGroupTuning {
    type Entry = routed_group::Entry;
    type Case = RoutedGroupCase;

    fn launches(&self) -> usize {
        self.layers
    }

    fn bindings(&self) -> String {
        "fixed".into()
    }

    fn statics(&self, _: &ModelInputs<'_>) -> Result<Vec<(&'static str, u64)>, String> {
        Ok(vec![("E", self.shape.experts), ("K", self.shape.selected)])
    }

    fn points(&self, limits: TuningLimits) -> Vec<PointShape> {
        grouped_points(limits)
    }

    /// One argument set: the tables are small and stay cache resident in a
    /// real step as well.
    fn rotation(
        &self,
        inputs: &mut TuningInputs<'_, '_>,
        point: &PointShape,
    ) -> Result<Vec<Self::Case>, BuildError> {
        let (shape, rows) = (self.shape, point.rows);
        let blocks = grouped_blocks(rows, shape.experts, shape.selected)?.fixed();
        let table = |extents: &[u64]| -> Result<CaseState, BuildError> {
            let tensor = inputs.scratch(Element::i32(), extents)?;
            inputs.state(tensor, 0..extents[0])
        };
        let (first, grouped) = grouped_rows(inputs, rows)?;
        Ok(vec![RoutedGroupCase {
            first,
            rows: grouped,
            routes: inputs.i32s(&[rows, shape.selected], &routes(rows, shape))?,
            counts: table(&[shape.experts])?,
            order: table(&[blocks, TILE_ROWS])?,
            inverse: table(&[rows, shape.selected])?,
            blocks: table(&[blocks])?,
            used: table(&[1])?,
        }])
    }

    fn args<'a>(case: &'a mut Self::Case) -> routed_group::Args<'a> {
        routed_group::Args {
            first: &case.first,
            rows: &case.rows,
            routes: &case.routes,
            counts: case.counts.tensor_mut(),
            order: case.order.tensor_mut(),
            inverse: case.inverse.tensor_mut(),
            blocks: case.blocks.tensor_mut(),
            used: case.used.tensor_mut(),
        }
    }

    fn state(case: &Self::Case) -> Vec<(&'static str, &CaseState)> {
        vec![
            ("counts", &case.counts),
            ("order", &case.order),
            ("inverse", &case.inverse),
            ("blocks", &case.blocks),
            ("used", &case.used),
        ]
    }

    generated_entry!(routed_group);
}

/// Unit per-expert gate and up scale operands of an expansion at their
/// planned extents (the gate's absent for up-only experts).
pub(super) struct ExpansionScales {
    pub(super) gate: Tensor,
    pub(super) up: Tensor,
}

impl ExpansionScales {
    pub(super) fn at(
        inputs: &TuningInputs<'_, '_>,
        scope: WeightScope,
        gated: bool,
    ) -> Result<Self, BuildError> {
        let gate = match gated {
            true => inputs.scale_extent(scope, WeightKind::ExpertGate)?,
            false => 0,
        };
        Ok(Self {
            gate: inputs.unit_scale(gate)?,
            up: inputs.unit_scale(inputs.scale_extent(scope, WeightKind::ExpertUp)?)?,
        })
    }

    /// The statics of the gate and up scales `scopes` share: GS (gated
    /// experts only) and US.
    pub(super) fn statics(
        inputs: &ModelInputs<'_>,
        scopes: &[WeightScope],
        gated: bool,
    ) -> Result<Vec<(&'static str, u64)>, String> {
        let up = ("US", scale_extent(inputs, scopes, WeightKind::ExpertUp)?);
        Ok(match gated {
            true => vec![("GS", scale_extent(inputs, scopes, WeightKind::ExpertGate)?), up],
            false => vec![up],
        })
    }
}

/// What a grouped expert pass reads beside its `order` and `blocks`: the
/// grouped rows, the blocks they fill (`table`'s experts), and the absent map
/// and step, so every expert is read from the stacks by index (which also
/// bind as its records).
pub(super) struct GroupedTables {
    pub(super) first: Tensor,
    pub(super) rows: Tensor,
    pub(super) used: Tensor,
    pub(super) map: Tensor,
    pub(super) step: Tensor,
}

impl GroupedTables {
    /// The tables of a pass over every one of `rows` rows, from row 0.
    pub(super) fn of(inputs: &TuningInputs<'_, '_>, rows: u64, table: &[i32]) -> Result<Self, BuildError> {
        let used = table.iter().filter(|expert| **expert >= 0).count();
        let used = i32::try_from(used).map_err(|_| String::from("grouped blocks exceed i32"))?;
        let (first, rows) = grouped_rows(inputs, rows)?;
        Ok(Self {
            first,
            rows,
            used: inputs.i32s(&[1], &[used])?,
            map: inputs.i32s(&[0, magnitude_kernels::constants::MAP_WORDS], &[])?,
            step: inputs.i32s(&[0], &[])?,
        })
    }
}

/// The `[1]` first row and row count of a grouped pass over every one of
/// `rows` rows, from row 0.
fn grouped_rows(inputs: &TuningInputs<'_, '_>, rows: u64) -> Result<(Tensor, Tensor), BuildError> {
    let rows = i32::try_from(rows).map_err(|_| String::from("grouped rows exceed i32"))?;
    Ok((inputs.i32s(&[1], &[0])?, inputs.i32s(&[1], &[rows])?))
}

/// `routed_expand`: the selected experts' and the shared expert's
/// gate/up with SiLU·mul (decode rows).
#[derive(Clone)]
pub(crate) struct RoutedExpandTuning {
    pub expert_gate: Element,
    pub expert_up: Element,
    pub shared_gate: Element,
    pub shared_up: Element,
    pub activation: Element,
    pub shape: RoutedShape,
    pub scopes: Vec<WeightScope>,
}

pub(crate) struct RoutedExpandCase {
    normalized: Tensor,
    routes: Tensor,
    expert_gate: Tensor,
    expert_up: Tensor,
    shared_gate: Tensor,
    shared_up: Tensor,
    /// The experts' gate and up scales.
    scales: ExpansionScales,
}

impl RoutedExpandTuning {
    fn elements(&self) -> routed_expand::Elements {
        routed_expand::Elements {
            EGW: self.expert_gate,
            EUW: self.expert_up,
            SGW: self.shared_gate,
            SUW: self.shared_up,
            A: self.activation,
        }
    }
}

impl EntryTuning for RoutedExpandTuning {
    type Entry = routed_expand::Entry;
    type Case = RoutedExpandCase;

    fn launches(&self) -> usize {
        self.scopes.len()
    }

    fn bindings(&self) -> String {
        format!(
            "EGW={},EUW={},SGW={},SUW={},A={}",
            self.expert_gate.name(),
            self.expert_up.name(),
            self.shared_gate.name(),
            self.shared_up.name(),
            self.activation.name()
        )
    }

    fn statics(&self, inputs: &ModelInputs<'_>) -> Result<Vec<(&'static str, u64)>, String> {
        let shape = self.shape;
        let mut statics = vec![
            ("H", shape.hidden),
            ("K", shape.selected),
            ("F", shape.features),
            ("S", shape.shared),
        ];
        statics.extend(ExpansionScales::statics(inputs, &self.scopes, true)?);
        Ok(statics)
    }

    fn points(&self, limits: TuningLimits) -> Vec<PointShape> {
        decode_points(limits)
    }

    fn rotation(
        &self,
        inputs: &mut TuningInputs<'_, '_>,
        point: &PointShape,
    ) -> Result<Vec<Self::Case>, BuildError> {
        let shape = self.shape;
        TuningInputs::rotation_scopes(&self.scopes, point)
            .into_iter()
            .enumerate()
            .map(|(index, scope)| {
                Ok(RoutedExpandCase {
                    normalized: inputs.activation(
                        self.activation,
                        &[point.rows, shape.hidden],
                        index as u64 + 1,
                    )?,
                    routes: inputs
                        .i32s(&[point.rows, shape.selected], &routes(point.rows, shape))?,
                    expert_gate: inputs.weight(scope, WeightKind::ExpertGate)?,
                    expert_up: inputs.weight(scope, WeightKind::ExpertUp)?,
                    shared_gate: inputs.weight(scope, WeightKind::SharedGate)?,
                    shared_up: inputs.weight(scope, WeightKind::SharedUp)?,
                    scales: ExpansionScales::at(inputs, scope, true)?,
                })
            })
            .collect()
    }

    fn args<'a>(case: &'a mut Self::Case) -> routed_expand::Args<'a> {
        routed_expand::Args {
            normalized: &case.normalized,
            routes: &case.routes,
            expert_gate: &case.expert_gate,
            expert_up: &case.expert_up,
            shared_gate: &case.shared_gate,
            shared_up: &case.shared_up,
            gate_scale: &case.scales.gate,
            up_scale: &case.scales.up,
        }
    }

    generated_entry!(routed_expand, this => this.elements());
}

/// A resident decode case's sources of `rows` rows (`decode_source`): none
/// (`XS` 0), each choice reading its expert's stack row by index.
pub(crate) fn no_sources(inputs: &TuningInputs<'_, '_>, rows: u64, selected: u64) -> Result<Tensor, BuildError> {
    inputs.i32s(&[0, rows, selected], &[])
}

/// A resident decode case's host tier records of `stack`'s tensor: a view
/// of none of its records (`TR` 0).
pub(crate) fn no_tier(stack: &Tensor) -> Result<Tensor, BuildError> {
    Ok(stack.slice_leading(0, 0)?)
}

/// `routed_gate_up` (SiLU) of the shared-route form: the selected experts'
/// gate/up (decode rows).
#[derive(Clone)]
pub(crate) struct RoutedChoicesTuning {
    pub expert_gate: Element,
    pub expert_up: Element,
    pub activation: Element,
    pub shape: RoutedShape,
    pub scopes: Vec<WeightScope>,
}

pub(crate) struct RoutedChoicesCase {
    normalized: Tensor,
    routes: Tensor,
    /// No sources: every expert read from the stacks by index.
    sources: Tensor,
    expert_gate: Tensor,
    expert_up: Tensor,
    tier_gate: Tensor,
    tier_up: Tensor,
    /// The experts' gate and up scales.
    scales: ExpansionScales,
    /// No clamp: the host port `[0, LB]` at LB = 0.
    glu_limit: Tensor,
}

impl EntryTuning for RoutedChoicesTuning {
    type Entry = routed_gate_up::Entry;
    type Case = RoutedChoicesCase;

    fn launches(&self) -> usize {
        self.scopes.len()
    }

    fn bindings(&self) -> String {
        format!(
            "EGW={},EUW={},A={}",
            self.expert_gate.name(),
            self.expert_up.name(),
            self.activation.name()
        )
    }

    fn statics(&self, inputs: &ModelInputs<'_>) -> Result<Vec<(&'static str, u64)>, String> {
        let shape = self.shape;
        let mut statics = vec![
            ("H", shape.hidden),
            ("K", shape.selected),
            ("F", shape.features),
            ("LB", GluLimit::word(None)),
        ];
        statics.extend(ExpansionScales::statics(inputs, &self.scopes, true)?);
        Ok(statics)
    }

    fn points(&self, limits: TuningLimits) -> Vec<PointShape> {
        decode_points(limits)
    }

    fn rotation(
        &self,
        inputs: &mut TuningInputs<'_, '_>,
        point: &PointShape,
    ) -> Result<Vec<Self::Case>, BuildError> {
        let shape = self.shape;
        TuningInputs::rotation_scopes(&self.scopes, point)
            .into_iter()
            .enumerate()
            .map(|(index, scope)| {
                let expert_gate = inputs.weight(scope, WeightKind::ExpertGate)?;
                let expert_up = inputs.weight(scope, WeightKind::ExpertUp)?;
                Ok(RoutedChoicesCase {
                    normalized: inputs.activation(
                        self.activation,
                        &[point.rows, shape.hidden],
                        index as u64 + 1,
                    )?,
                    routes: inputs
                        .i32s(&[point.rows, shape.selected], &routes(point.rows, shape))?,
                    sources: no_sources(inputs, point.rows, shape.selected)?,
                    tier_gate: no_tier(&expert_gate)?,
                    tier_up: no_tier(&expert_up)?,
                    expert_gate,
                    expert_up,
                    scales: ExpansionScales::at(inputs, scope, true)?,
                    glu_limit: inputs.i32s(&[0, GluLimit::word(None)], &[])?,
                })
            })
            .collect()
    }

    fn args<'a>(case: &'a mut Self::Case) -> routed_gate_up::Args<'a> {
        routed_gate_up::Args {
            normalized: &case.normalized,
            routes: &case.routes,
            sources: &case.sources,
            expert_gate: &case.expert_gate,
            expert_up: &case.expert_up,
            tier_gate: &case.tier_gate,
            tier_up: &case.tier_up,
            // SiLU, unclamped: the fused Qwen form's experts.
            activation: 0,
            gate_scale: &case.scales.gate,
            up_scale: &case.scales.up,
            glu_limit: &case.glu_limit,
        }
    }

    generated_entry!(routed_gate_up, this => routed_gate_up::Elements {
        EGW: this.expert_gate,
        EUW: this.expert_up,
        A: this.activation,
    });
}

/// `routed_output`: the selected experts' down projections in slot
/// order, the shared expert's, and the residual (decode rows).
#[derive(Clone)]
pub(crate) struct RoutedOutputTuning {
    pub expert_down: Element,
    pub shared_down: Element,
    pub activation: Element,
    pub shape: RoutedShape,
    pub scopes: Vec<WeightScope>,
}

pub(crate) struct RoutedOutputCase {
    residual: Tensor,
    inject: Tensor,
    expert_product: Tensor,
    shared_product: Tensor,
    routes: Tensor,
    /// No sources: every expert read from the stack by index.
    sources: Tensor,
    scores: Tensor,
    coefficient: Tensor,
    expert_down: Tensor,
    tier_down: Tensor,
    shared_down: Tensor,
}

impl RoutedOutputTuning {
    fn elements(&self) -> routed_output::Elements {
        routed_output::Elements {
            EDW: self.expert_down,
            SDW: self.shared_down,
            A: self.activation,
        }
    }
}

impl EntryTuning for RoutedOutputTuning {
    type Entry = routed_output::Entry;
    type Case = RoutedOutputCase;

    fn launches(&self) -> usize {
        self.scopes.len()
    }

    fn bindings(&self) -> String {
        format!(
            "EDW={},SDW={},A={}",
            self.expert_down.name(),
            self.shared_down.name(),
            self.activation.name()
        )
    }

    fn statics(&self, inputs: &ModelInputs<'_>) -> Result<Vec<(&'static str, u64)>, String> {
        let shape = self.shape;
        let (streams, inject) = inputs.join(&self.scopes)?;
        Ok(vec![
            ("H", shape.hidden),
            ("K", shape.selected),
            ("F", shape.features),
            ("S", shape.shared),
            ("SN", streams),
            ("IN", inject),
        ])
    }

    fn points(&self, limits: TuningLimits) -> Vec<PointShape> {
        decode_points(limits)
    }

    fn rotation(
        &self,
        inputs: &mut TuningInputs<'_, '_>,
        point: &PointShape,
    ) -> Result<Vec<Self::Case>, BuildError> {
        let (shape, rows) = (self.shape, point.rows);
        let (streams, inject) = inputs.join(&self.scopes)?;
        TuningInputs::rotation_scopes(&self.scopes, point)
            .into_iter()
            .enumerate()
            .map(|(index, scope)| {
                let seed = 8 * index as u64;
                let expert_down = inputs.weight(scope, WeightKind::ExpertDown)?;
                Ok(RoutedOutputCase {
                    residual: inputs.activation(
                        Element::f32(),
                        &[rows, streams * shape.hidden],
                        seed + 1,
                    )?,
                    inject: inputs.activation(Element::f32(), &[rows, inject], seed + 6)?,
                    expert_product: inputs.activation(
                        self.activation,
                        &[rows, shape.selected, shape.features],
                        seed + 2,
                    )?,
                    shared_product: inputs.activation(
                        self.activation,
                        &[rows, shape.shared],
                        seed + 3,
                    )?,
                    routes: inputs.i32s(&[rows, shape.selected], &routes(rows, shape))?,
                    sources: no_sources(inputs, rows, shape.selected)?,
                    scores: scores(inputs, rows, shape, seed + 4)?,
                    coefficient: inputs.activation(Element::f32(), &[rows], seed + 5)?,
                    tier_down: no_tier(&expert_down)?,
                    expert_down,
                    shared_down: inputs.weight(scope, WeightKind::SharedDown)?,
                })
            })
            .collect()
    }

    fn args<'a>(case: &'a mut Self::Case) -> routed_output::Args<'a> {
        routed_output::Args {
            residual: &case.residual,
            inject: &case.inject,
            expert_product: &case.expert_product,
            shared_product: &case.shared_product,
            routes: &case.routes,
            sources: &case.sources,
            scores: &case.scores,
            coefficient: &case.coefficient,
            expert_down: &case.expert_down,
            tier_down: &case.tier_down,
            shared_down: &case.shared_down,
        }
    }

    generated_entry!(routed_output, this => this.elements());
}

/// A streamed combine's projections' sources of `rows` rows over `routes`
/// (a decode case's): each choice sourced from the bank record of its
/// expert, none the CPU's, as the resolve writes them
/// (`magnitude_kernels::source_words`).
pub(crate) fn bank_sources(
    inputs: &TuningInputs<'_, '_>,
    rows: u64,
    selected: u64,
    routes: &[i32],
) -> Result<Tensor, BuildError> {
    let count = (rows * selected) as usize;
    inputs.i32s(&[SOURCE_ROWS, rows, selected], &magnitude_kernels::source_words(&routes[..count], &vec![-1; count]))
}

/// `routed_output_project`: the device's projections of a streamed layer's
/// fused combine, its resident output's (`routed_output`) launches and
/// parameters publishing the projections for the sum `routed_output_sited`
/// forms once the CPU site has answered. It follows the resident output's
/// unit (`Specializer::tuned_following`), so it keeps that output's
/// arithmetic choice and its bits, and is tuned in its launch shape over the
/// streamed layers.
#[derive(Clone)]
pub(crate) struct RoutedOutputProjectTuning {
    /// The resident output it follows, over every layer of the binding.
    pub resident: RoutedOutputTuning,
    /// The streamed layers that launch it.
    pub layers: Vec<WeightScope>,
}

impl EntryTuning for RoutedOutputProjectTuning {
    type Entry = routed_output_project::Entry;
    type Case = RoutedOutputCase;

    fn launches(&self) -> usize {
        self.layers.len()
    }

    fn bindings(&self) -> String {
        self.resident.bindings()
    }

    /// The resident output's statics but its join's (`SN`, `IN`), which the
    /// sum takes.
    fn statics(&self, inputs: &ModelInputs<'_>) -> Result<Vec<(&'static str, u64)>, String> {
        Ok(self.resident.statics(inputs)?.into_iter().filter(|(name, _)| !matches!(*name, "SN" | "IN")).collect())
    }

    fn points(&self, limits: TuningLimits) -> Vec<PointShape> {
        decode_points(limits)
    }

    fn rotation(
        &self,
        inputs: &mut TuningInputs<'_, '_>,
        point: &PointShape,
    ) -> Result<Vec<Self::Case>, BuildError> {
        let mut over = self.resident.clone();
        over.scopes = self.layers.clone();
        let shape = self.resident.shape;
        let table = routes(point.rows, shape);
        over.rotation(inputs, point)?
            .into_iter()
            .map(|mut output| {
                output.sources = bank_sources(inputs, point.rows, shape.selected, &table)?;
                Ok(output)
            })
            .collect()
    }

    fn args<'a>(output: &'a mut Self::Case) -> routed_output_project::Args<'a> {
        routed_output_project::Args {
            expert_product: &output.expert_product,
            shared_product: &output.shared_product,
            routes: &output.routes,
            sources: &output.sources,
            expert_down: &output.expert_down,
            tier_down: &output.tier_down,
            shared_down: &output.shared_down,
        }
    }

    generated_entry!(routed_output_project, this => routed_output_project::Elements {
        EDW: this.resident.expert_down,
        SDW: this.resident.shared_down,
        A: this.resident.activation,
    });
}

/// `routed_experts`: grouped expert tiles (grouped rows).
#[derive(Clone)]
pub(crate) struct RoutedExpertsTuning {
    pub expert_gate: Element,
    pub expert_up: Element,
    pub expert_down: Element,
    pub activation: Element,
    pub shape: RoutedShape,
    pub scopes: Vec<WeightScope>,
}

pub(crate) struct RoutedExpertsCase {
    /// The grouped rows, the blocks they fill, and the absent slot map: the
    /// experts read from the stacks by index.
    tables: GroupedTables,
    normalized: Tensor,
    order: Tensor,
    blocks: Tensor,
    expert_gate: Tensor,
    expert_up: Tensor,
    expert_down: Tensor,
    /// The experts' gate and up scales.
    scales: ExpansionScales,
    /// No clamp: the host port `[0, LB]` at LB = 0.
    glu_limit: Tensor,
}

impl RoutedExpertsTuning {
    fn elements(&self) -> routed_experts::Elements {
        routed_experts::Elements {
            EGW: self.expert_gate,
            EUW: self.expert_up,
            EDW: self.expert_down,
            A: self.activation,
        }
    }
}

impl EntryTuning for RoutedExpertsTuning {
    type Entry = routed_experts::Entry;
    type Case = RoutedExpertsCase;

    fn launches(&self) -> usize {
        self.scopes.len()
    }

    fn bindings(&self) -> String {
        format!(
            "EGW={},EUW={},EDW={},A={}",
            self.expert_gate.name(),
            self.expert_up.name(),
            self.expert_down.name(),
            self.activation.name()
        )
    }

    fn statics(&self, inputs: &ModelInputs<'_>) -> Result<Vec<(&'static str, u64)>, String> {
        let mut statics = vec![("H", self.shape.hidden), ("F", self.shape.features), ("LB", GluLimit::word(None))];
        statics.extend(ExpansionScales::statics(inputs, &self.scopes, true)?);
        Ok(statics)
    }

    fn points(&self, limits: TuningLimits) -> Vec<PointShape> {
        grouped_points(limits)
    }

    fn rotation(
        &self,
        inputs: &mut TuningInputs<'_, '_>,
        point: &PointShape,
    ) -> Result<Vec<Self::Case>, BuildError> {
        let (shape, rows) = (self.shape, point.rows);
        let (order, _, table) = group(&routes(rows, shape), rows, shape)?;
        let blocks = table.len() as u64;
        TuningInputs::rotation_scopes(&self.scopes, point)
            .into_iter()
            .enumerate()
            .map(|(index, scope)| {
                Ok(RoutedExpertsCase {
                    tables: GroupedTables::of(inputs, rows, &table)?,
                    normalized: inputs.activation(
                        self.activation,
                        &[rows, shape.hidden],
                        index as u64 + 1,
                    )?,
                    order: inputs.i32s(&[blocks, TILE_ROWS], &order)?,
                    blocks: inputs.i32s(&[blocks], &table)?,
                    expert_gate: inputs.weight(scope, WeightKind::ExpertGate)?,
                    expert_up: inputs.weight(scope, WeightKind::ExpertUp)?,
                    expert_down: inputs.weight(scope, WeightKind::ExpertDown)?,
                    scales: ExpansionScales::at(inputs, scope, true)?,
                    glu_limit: inputs.i32s(&[0, GluLimit::word(None)], &[])?,
                })
            })
            .collect()
    }

    fn args<'a>(case: &'a mut Self::Case) -> routed_experts::Args<'a> {
        routed_experts::Args {
            first: &case.tables.first,
            rows: &case.tables.rows,
            normalized: &case.normalized,
            order: &case.order,
            blocks: &case.blocks,
            used: &case.tables.used,
            expert_gate: &case.expert_gate,
            expert_up: &case.expert_up,
            expert_down: &case.expert_down,
            map: &case.tables.map,
            step: &case.tables.step,
            records_gate: &case.expert_gate,
            records_up: &case.expert_up,
            records_down: &case.expert_down,
            activation: 0,
            gate_scale: &case.scales.gate,
            up_scale: &case.scales.up,
            glu_limit: &case.glu_limit,
        }
    }

    generated_entry!(routed_experts, this => this.elements());
}

/// `routed_combine`: the shared expert over every row with the grouped
/// unpermute and the residual (grouped rows).
#[derive(Clone)]
pub(crate) struct RoutedCombineTuning {
    pub shared_gate: Element,
    pub shared_up: Element,
    pub shared_down: Element,
    pub activation: Element,
    pub shape: RoutedShape,
    pub scopes: Vec<WeightScope>,
}

pub(crate) struct RoutedCombineCase {
    residual: Tensor,
    inject: Tensor,
    expert_output: Tensor,
    inverse: Tensor,
    scores: Tensor,
    normalized: Tensor,
    coefficient: Tensor,
    shared_gate: Tensor,
    shared_up: Tensor,
    shared_down: Tensor,
}

impl RoutedCombineTuning {
    fn elements(&self) -> routed_combine::Elements {
        routed_combine::Elements {
            SGW: self.shared_gate,
            SUW: self.shared_up,
            SDW: self.shared_down,
            A: self.activation,
        }
    }
}

impl EntryTuning for RoutedCombineTuning {
    type Entry = routed_combine::Entry;
    type Case = RoutedCombineCase;

    fn launches(&self) -> usize {
        self.scopes.len()
    }

    fn bindings(&self) -> String {
        format!(
            "SGW={},SUW={},SDW={},A={}",
            self.shared_gate.name(),
            self.shared_up.name(),
            self.shared_down.name(),
            self.activation.name()
        )
    }

    fn statics(&self, inputs: &ModelInputs<'_>) -> Result<Vec<(&'static str, u64)>, String> {
        let shape = self.shape;
        let (streams, inject) = inputs.join(&self.scopes)?;
        Ok(vec![
            ("H", shape.hidden),
            ("K", shape.selected),
            ("S", shape.shared),
            ("SN", streams),
            ("IN", inject),
        ])
    }

    fn points(&self, limits: TuningLimits) -> Vec<PointShape> {
        grouped_points(limits)
    }

    fn rotation(
        &self,
        inputs: &mut TuningInputs<'_, '_>,
        point: &PointShape,
    ) -> Result<Vec<Self::Case>, BuildError> {
        let (shape, rows) = (self.shape, point.rows);
        let (_, inverse, table) = group(&routes(rows, shape), rows, shape)?;
        let blocks = table.len() as u64;
        let (streams, inject) = inputs.join(&self.scopes)?;
        TuningInputs::rotation_scopes(&self.scopes, point)
            .into_iter()
            .enumerate()
            .map(|(index, scope)| {
                let seed = 8 * index as u64;
                Ok(RoutedCombineCase {
                    residual: inputs.activation(
                        Element::f32(),
                        &[rows, streams * shape.hidden],
                        seed + 1,
                    )?,
                    inject: inputs.activation(Element::f32(), &[rows, inject], seed + 6)?,
                    expert_output: inputs.activation(
                        self.activation,
                        &[blocks, TILE_ROWS, shape.hidden],
                        seed + 2,
                    )?,
                    inverse: inputs.i32s(&[rows, shape.selected], &inverse)?,
                    scores: scores(inputs, rows, shape, seed + 3)?,
                    normalized: inputs.activation(
                        self.activation,
                        &[rows, shape.hidden],
                        seed + 4,
                    )?,
                    coefficient: inputs.activation(Element::f32(), &[rows], seed + 5)?,
                    shared_gate: inputs.weight(scope, WeightKind::SharedGate)?,
                    shared_up: inputs.weight(scope, WeightKind::SharedUp)?,
                    shared_down: inputs.weight(scope, WeightKind::SharedDown)?,
                })
            })
            .collect()
    }

    fn args<'a>(case: &'a mut Self::Case) -> routed_combine::Args<'a> {
        routed_combine::Args {
            residual: &case.residual,
            inject: &case.inject,
            expert_output: &case.expert_output,
            inverse: &case.inverse,
            scores: &case.scores,
            normalized: &case.normalized,
            coefficient: &case.coefficient,
            shared_gate: &case.shared_gate,
            shared_up: &case.shared_up,
            shared_down: &case.shared_down,
        }
    }

    generated_entry!(routed_combine, this => this.elements());
}
