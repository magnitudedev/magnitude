//! The serving heap: a loaded model's memory decisions settled on one price
//! ([`super::value`]). The expert bank's holdings (its slots, its host
//! tier's host slots, its stored tables' records) price each step by what
//! the decode step saves at the load's decode steps a second; a prompt
//! group's places by what its fewer groups save at its requests a second
//! ([`ExecutorDomain::places_seconds`], the plan's model on generic
//! measurements); the workload state's unborn room by what lending it to
//! the bank costs once a sequence grows into it
//! ([`ExecutorDomain::state_curve`]); the optional components by what
//! drafting and encoding save ([`super::holdings`]); the owner's cached
//! prefixes come in as [`OwnerHoldings`]. Each store settles its own
//! room ([`value::Supply`]): free room above its derived reserve, borrowed
//! room at its price, the device's hard limit, and what an ask owes.
//!
//! A settlement is applied by its owners: a bank holding resizes once to
//! its settled size ([`ExecutorDomain::grow_expert_bank`],
//! [`ExecutorDomain::shrink_expert_bank_to`]), the owner evicts the
//! prefixes and releases the optional components it released. The places
//! target and the state lent are the domain's, set as it settles: a group
//! takes its places as it opens (`grant_group`), their room reserved from
//! the bank's growth (`class_reserve`), and gives back between groups those
//! a settle released ([`ExecutorDomain::release_group_places`]).

use super::value::{self, Curve, Holding, Rate, Step, Supply};
use super::borrowing::Room;
use super::holdings::OptionalHolding;
use super::{BankCurve, BankHolding, BankStep, DomainError, ExecutorDomain, OptionalComponent, ProgramFamily, Reach, Reaches, SizeStep, StateBindings};
use crate::memory::HoldingClass;
use crate::planning::BankTier;
use crate::workload::{WorkloadMeter, WorkloadRates};
use std::time::{Duration, Instant};

/// The span the load's serving rates are read over: each count decays with
/// it (`WorkloadMeter`), so the rates follow the workload as it changes
/// over about as long as a memory decision is expected to stand, and an
/// exchange's gain, priced at those rates, is earned over it
/// (`Supply::horizon`).
const WORKLOAD_SPAN: Duration = Duration::from_secs(60);

/// What the owner holds that the heap prices beside the executor's own
/// holdings.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct OwnerHoldings {
    /// Cached prefixes it may evict, as their bytes and the serving seconds
    /// a second their hits save, the least valuable first
    /// (`PrefixCache::held_steps`).
    pub prefixes: Vec<(u64, f64)>,
}

/// A bank holding's growth a settle granted: from its size to `to`, the
/// bytes it backs, of them those borrowed beyond the free room.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BankGrant {
    pub holding: BankHolding,
    pub kind: BankTier,
    pub from: u64,
    pub to: u64,
    pub bytes: u64,
    pub borrowed: u64,
    /// The rate per byte of the steps it grants: what orders grants
    /// across stores.
    pub per_byte: f64,
}

/// A bank holding a settle released, to `to`, the rate per byte of the
/// steps it releases what orders releases.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BankRelease {
    pub holding: BankHolding,
    pub to: u64,
    pub per_byte: f64,
}

/// What one settle of the serving heap decided, for its owners to apply.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ServingSettlement {
    /// Cached prefixes to evict, the least valuable first.
    pub prefixes: usize,
    /// The idle optional components released.
    pub optional: Vec<OptionalComponent>,
    /// The bank's growths, the most valuable per byte first: one resize
    /// runs at a time, so the first begins and the rest settle again.
    pub grants: Vec<BankGrant>,
    /// The bank's releases, the least valuable per byte first, which go
    /// before any growth.
    pub releases: Vec<BankRelease>,
    /// Whether the places target fell below the places the groups hold:
    /// the class beyond it is released between groups
    /// ([`ExecutorDomain::release_group_places`]), before any bank resize.
    pub places_released: bool,
}

/// A curve's steps valued at `decode_steps_per_s`, each moving in what the
/// holding's last resize held the submitting thread (`move_seconds`), a
/// held one beside what its return costs once it is released (`returns`,
/// of the slots it releases).
fn priced(curve: &BankCurve, decode_steps_per_s: f64, move_seconds: f64, returns: &dyn Fn(u64, &SizeStep) -> f64) -> Curve<Holding> {
    let step = |size: &SizeStep, back: f64| Step {
        bytes: size.bytes,
        worth: Rate::new(size.saves.max(0.) * decode_steps_per_s),
        move_seconds: move_seconds + back,
    };
    // A held step releases from the size before it.
    let from = std::iter::once(curve.at).chain(curve.held.iter().map(|size| size.to));
    Curve {
        holding: Holding::Bank(curve.holding),
        // Only the one store's expert slots are held on borrowed room
        // (`BankRoom`).
        borrows: curve.holding == BankHolding::Experts && curve.kind == BankTier::Bank,
        lends: false,
        held: curve.held.iter().zip(from).map(|(size, from)| step(size, returns(from - size.to, size))).collect(),
        offered: curve.offered.iter().map(|size| step(size, 0.)).collect(),
    }
}

/// What a settle of the serving heap read that moves its outcome
/// ([`ExecutorDomain::settle_memory`]), and the room in each store it
/// stands in: a settle on the same holdings within that room settles the
/// same, so none runs ([`SettleStands::holds`]). The serving rates and the
/// decode's price move at every reading; their drift alone changes no step
/// a holding holds or offers, so it is no input (a request's arrival or end
/// is, `requests` and `idle`).
#[derive(Clone, Debug, PartialEq)]
struct SettleKey {
    resizes: super::ExpertResizes,
    places: usize,
    head: bool,
    vision: bool,
    idle: bool,
    measuring: bool,
    requests: u64,
    /// The owner's cached prefixes: how many, and their bytes.
    prefixes: (usize, u64),
    /// Each store's room (`MemoryPace::supply`): free and borrowable
    /// together, and what is owed.
    rooms: Vec<(seismic::StoreId, u64, u64)>,
}

/// The last settle's inputs and, in each store, how far its room may move
/// before the settle could take another step: up by the least step offered
/// that it did not grant, down by the least held step it did not release;
/// none where no step is.
pub(super) struct SettleStands {
    key: SettleKey,
    moves: StoreMoves,
}

/// Each store's room's moves a settle stands across ([`SettleStands`]):
/// up, down.
type StoreMoves = Vec<(seismic::StoreId, Option<u64>, Option<u64>)>;

impl SettleStands {
    /// Whether a settle at `key` settles as this one did: the same holdings,
    /// no more owed, and each store's room within the steps it stands
    /// between.
    fn holds(&self, key: &SettleKey) -> bool {
        let SettleKey { rooms, .. } = key;
        let same = SettleKey { rooms: self.key.rooms.clone(), ..key.clone() };
        same == self.key
            && rooms.len() == self.key.rooms.len()
            && rooms.iter().zip(&self.key.rooms).all(|(&(store, room, owed), &(was, before, owing))| {
                let (up, down) = self
                    .moves
                    .iter()
                    .find(|(of, _, _)| *of == store)
                    .map_or((None, None), |&(_, up, down)| (up, down));
                store == was
                    && owed <= owing
                    && up.is_none_or(|up| room < before.saturating_add(up))
                    && down.is_none_or(|down| room > before.saturating_sub(down))
            })
    }
}

/// The room a release settles into: none free, `owed` to cover.
fn owing(owed: u64) -> Supply {
    Supply { free: 0, owed, borrowable: 0, borrow_price: 0., ceiling: u64::MAX, horizon: Duration::ZERO }
}

impl<F: ProgramFamily> ExecutorDomain<F> {
    /// The load's serving rates now (`WorkloadMeter`): its meter begun at
    /// the serving prior once the load is priced ([`Self::serving_prior`]),
    /// banked or not; none before.
    pub fn serving_rates(&mut self) -> Result<Option<WorkloadRates>, DomainError> {
        let now = Instant::now();
        if self.workload.is_none() {
            self.workload = self.serving_prior()?.map(|prior| WorkloadMeter::new(prior, WORKLOAD_SPAN, now));
        }
        Ok(self.workload.as_ref().map(|meter| meter.rates(now)))
    }

    /// A round served: a decode step, or a prefill of `rows` rows taking
    /// `seconds`; counted once the meter runs.
    pub fn served_round(&mut self, decode: bool, rows: usize, seconds: f64) {
        let now = Instant::now();
        if let Some(meter) = self.workload.as_mut() {
            match decode {
                true => meter.decoded(now, 1),
                false => meter.prefilled(now, rows as u64, seconds),
            }
        }
    }

    /// A request arrived, its input encoded by the vision projector where
    /// `vision`; counted once the meter runs.
    pub fn served_request(&mut self, vision: bool) {
        self.requests += 1;
        if let Some(meter) = self.workload.as_mut() {
            meter.requested(Instant::now(), vision);
        }
    }

    /// The bank's curves at its sizes now (`ElasticBank::curves`), each
    /// with its heap curve at `decode_steps_per_s`: priced on the load's
    /// decode (`decode_price`), or, before its costs are measured, holding
    /// what saves nothing known and offering nothing. Each step moves in
    /// what the holding's last resize held the submitting thread; before
    /// one was timed, the bank's slots in what their shrink's moves take
    /// the device, a step moving at most a record of each group
    /// (`MoveLines`). None without a bank.
    fn bank_heap(&self, price: Option<&crate::assessment::DecodePrice>, decode_steps_per_s: f64, reach: Reaches) -> Result<Vec<(BankCurve, Curve<Holding>)>, DomainError> {
        let Some(bank) = self.family.expert_bank() else {
            return Ok(Vec::new());
        };
        let held = &self.resize_held.last;
        let measures = self.expert_measures();
        let moves = measures.and_then(|measures| measures.moves.measured());
        let returns = self.slot_returns(price)?;
        Ok(bank
            .curves(price, reach)?
            .into_iter()
            .map(|curve| {
                let slots = curve.holding == BankHolding::Experts && curve.kind == BankTier::Bank;
                let lined = slots.then(|| moves.map(|lines| lines.seconds(|_| 1))).flatten();
                let move_seconds = held.get(&curve.holding).copied().or(lined).unwrap_or(0.);
                let none = |_: u64, _: &SizeStep| 0.;
                let heap = priced(&curve, decode_steps_per_s, move_seconds, if slots { &returns } else { &none });
                (curve, heap)
            })
            .collect())
    }

    /// What a bank step's slots cost to come back once released
    /// ([`priced`]'s `returns`): the room a release makes is taken for a
    /// while (by the pressure it was owed to) and then given back,
    /// and the bank grows into it again, its released records read along
    /// the model store's line and landed at every streamed layer
    /// (`ExpertLines::filed`). The decode runs the steps that takes short
    /// of those slots, each losing the step's saving. Nothing known before
    /// the load's decode, its lines and its read line are priced.
    fn slot_returns<'a>(&'a self, price: Option<&'a crate::assessment::DecodePrice>) -> Result<Box<dyn Fn(u64, &SizeStep) -> f64 + 'a>, DomainError> {
        let measures = self.expert_measures();
        let lines = measures.and_then(|measures| measures.lines.measured());
        let read = measures.and_then(|measures| measures.read.measured());
        let step = match (price, self.held_slots()?) {
            (Some(price), Some(held)) => Some(price.step(held).total_seconds),
            _ => None,
        };
        Ok(Box::new(move |slots: u64, size: &SizeStep| match (lines, read, step) {
            (Some(lines), Some(read), Some(step)) if step > 0. => {
                let landing = lines.iter().map(|lines| lines.filed(read, slots)).sum::<f64>();
                landing / step * size.saves.max(0.)
            }
            _ => 0.,
        }))
    }

    /// What the process charges `store` now: its own claims there, read
    /// from the device's ledgers alone (`MemoryProbe::charged_in` without
    /// the probe's planned holdings, which a settle's room reads apart).
    fn store_charge(&self, store: seismic::StoreId) -> u64 {
        self.domain.device().ledgers().own_claims().into_iter().filter(|(of, _)| *of == store).map(|(_, bytes)| bytes).sum()
    }

    /// Whether the bank's tiers' storage lies in one store ([`Reaches::shared`]).
    fn tiers_shared(&self) -> bool {
        self.store_of(BankTier::Bank.storage()) == self.store_of(BankTier::Host.storage())
    }

    /// The stores the bank's storage kinds lie in, each with its kinds.
    fn bank_stores(&self) -> Vec<(seismic::StoreId, Vec<BankTier>)> {
        let mut stores: Vec<(seismic::StoreId, Vec<BankTier>)> = Vec::new();
        for tier in BankTier::ALL {
            let Some(store) = self.store_of(tier.storage()) else {
                continue;
            };
            match stores.iter_mut().find(|(of, _)| *of == store) {
                Some((_, tiers)) => tiers.push(tier),
                None => stores.push((store, vec![tier])),
            }
        }
        stores
    }

    /// The slots the bank and its host tier hold now, as the decode
    /// prices them.
    fn held_slots(&self) -> Result<Option<crate::assessment::HeldSlots>, DomainError> {
        let Some(bank) = self.family.expert_bank() else {
            return Ok(None);
        };
        let count = |value: u64| usize::try_from(value).map_err(|_| DomainError::invariant("bank slots exceed the host"));
        let host = bank.residency()?.host.map_or(0, |host| host.slots);
        Ok(Some(crate::assessment::HeldSlots { slots: count(bank.slots())?, host: count(host)? }))
    }

    /// The prompt's seconds at each place count from one to the most a
    /// group holds ([`Self::places_most`]), a whole-context prompt of that
    /// many chunks in groups of each count (`planning::grouped`): each
    /// group's submission (`LoadCosts::submission`), the rest of its chain
    /// queued behind its first launch; and where the plan banks its
    /// experts, their routed seconds by the plan's own model
    /// (`planning::routed_group_seconds`, at the plan's pass size and the
    /// workload prior's launch region), each pass's misses read along the
    /// model store's read line at the bank the plan settled
    /// (`StreamedLayers::pass_reads`): the expert loads one more place saves
    /// through reuse across a group's chunks, times the measured per-load
    /// cost. Generic machine figures and the plan's structure alone, never
    /// a reading or the load's own groups, so they change only with the
    /// load's measures: priced once per [`Self::measure_load`], each group
    /// size once and each pass size's reads once, and read at every settle
    /// after.
    ///
    /// None before the load is priced, or where a bank's pass, lines or
    /// read line are unmeasured: no place is worth anything known, and the
    /// places target stays the plan's.
    pub(super) fn places_seconds(&self) -> Result<Option<std::rc::Rc<[f64]>>, DomainError> {
        if let Some(prompt) = self.places_prompt.borrow().as_ref() {
            return Ok(Some(prompt.clone()));
        }
        let Some(costs) = self.load_costs() else {
            return Ok(None);
        };
        let submission = costs.submission;
        let most = self.places_most().get();
        let group: Box<dyn Fn(usize) -> f64 + '_> = match (self.expert_bank_plan(), self.family.expert_bank()) {
            (None, _) => Box::new(move |_| submission),
            (Some(plan), Some(bank)) => {
                let experts = self.expert_measures().ok_or_else(|| DomainError::invariant("a priced bank without its experts' measures"))?;
                let (Some(pass), Some(lines), Some(&read)) = (experts.pass.measured(), experts.lines.measured(), experts.read.measured()) else {
                    return Ok(None);
                };
                let layers = bank.streamed_layers(lines.clone());
                let count = |value: u64| usize::try_from(value).map_err(|_| DomainError::invariant("bank slots exceed the host"));
                let held = crate::assessment::HeldSlots { slots: count(plan.slots)?, host: count(plan.host_slots)? };
                let routed = plan.layers().iter().map(|layer| (layer.experts, plan.selected())).collect::<Vec<_>>();
                let (region, launches) = (self.prior()?.region_rows, plan.pass);
                // A pass's misses by its rows: at most one row count a pass
                // size, each read once.
                let reads = std::cell::RefCell::new(Vec::<(u64, f64)>::new());
                Box::new(move |steps| {
                    let pass_reads = |rows: u64| {
                        if let Some(&(_, seconds)) = reads.borrow().iter().find(|(of, _)| *of == rows) {
                            return seconds;
                        }
                        let seconds = layers.pass_reads(read, held, rows as f64);
                        reads.borrow_mut().push((rows, seconds));
                        seconds
                    };
                    submission + crate::planning::routed_group_seconds(pass, &routed, region, steps, launches, &pass_reads)
                })
            }
            (Some(_), None) => return Err(DomainError::invariant("a banked plan's load holds no bank")),
        };
        let sizes = (1..=most).map(|steps| group(steps)).collect::<Vec<_>>();
        let prompt: std::rc::Rc<[f64]> = (1..=most).map(|places| crate::planning::grouped(most, places, |steps| sizes[steps - 1])).collect();
        *self.places_prompt.borrow_mut() = Some(prompt.clone());
        Ok(Some(prompt))
    }

    /// A prompt group's places as the heap prices them
    /// ([`planning::places_curve`]): from one to the most a group holds,
    /// holding the target, each place more its class bytes against the
    /// prompt seconds it saves ([`Self::places_seconds`]) at `rates`'
    /// requests a second. None where a group holds one place at most, or
    /// before its seconds are priced.
    fn places_curve(&self, rates: &WorkloadRates) -> Result<Option<Curve<Holding>>, DomainError> {
        let most = self.places_most();
        if most == std::num::NonZeroUsize::MIN {
            return Ok(None);
        }
        let Some(prompt) = self.places_seconds()? else {
            return Ok(None);
        };
        Ok(Some(crate::planning::places_curve(
            self.places_target,
            most,
            &|places| self.class_ladder_bytes(places),
            &|places| prompt[places - 1],
            rates.requests_per_s,
        )))
    }

    /// The workload state's room no sequence has committed yet in device
    /// storage: the planned state's (`StateCapacityPlan::workload_state_bytes`)
    /// beyond the stores' committed slabs; none in another store.
    pub(super) fn unborn_state(&self) -> u64 {
        self.execution.resources().capacity().workload_state_bytes.saturating_sub(self.committed_state_bytes())
    }

    /// The unborn workload state's room as the heap prices it
    /// ([`Holding::State`], a lending curve): its unlent room held, its
    /// lent room offered back, in steps of one launch region's state. Each
    /// byte lent is worth keeping what lending it costs: a sequence grows
    /// into it once a request (every request of the prior reaches its
    /// planned context), when its claim is owed and the bank releases the
    /// slots on it and later takes them back, at the bank's own moves per
    /// byte (`experts`, its slots' held step: the shrink's move and the
    /// slots' return). None where the bank has no slot step to price that
    /// by (none whose moves are timed or measured), or the plan's state is
    /// none.
    fn state_curve(&self, rates: &WorkloadRates, experts: &Curve<Holding>) -> Result<Option<Curve<Holding>>, DomainError> {
        // A move not timed or measured yet is unknown, not free: lending
        // that costs nothing known lends nothing.
        let Some(slot) = experts.held.first().or(experts.offered.first()).filter(|slot| slot.move_seconds > 0.) else {
            return Ok(None);
        };
        let state = self.execution.resources().capacity().workload_state_bytes;
        let context = self.execution.resources().target_state().context_rows as u64;
        if state == 0 || context == 0 {
            return Ok(None);
        }
        let step = (state / context * self.prior()?.region_rows).max(1);
        let per_byte = rates.requests_per_s * slot.move_seconds / slot.bytes.max(1) as f64;
        let steps = |bytes: u64| -> Vec<Step> {
            (0..bytes.div_ceil(step))
                .map(|index| {
                    let bytes = step.min(bytes - index * step);
                    Step { bytes, worth: Rate::new(per_byte * bytes as f64), move_seconds: 0. }
                })
                .collect()
        };
        let unborn = self.unborn_state();
        Ok(Some(Curve {
            holding: Holding::State,
            borrows: false,
            lends: true,
            held: steps(unborn.saturating_sub(self.state_lent)),
            offered: steps(self.state_lent.min(unborn)),
        }))
    }

    /// The workload prior of the load's planned context and limits
    /// (`planning::Prior`).
    fn prior(&self) -> Result<crate::planning::Prior, DomainError> {
        let context = self.execution.resources().target_state().context_rows as u64;
        crate::planning::Prior::of(context, self.execution.policy().limits())
            .map_err(|error| DomainError::invariant(format!("the workload prior: {error}")))
    }

    /// The bound optional components as the heap prices them
    /// (`holdings::OptionalHolding`), each one held step where its worth is
    /// known at the measured serving `rates`. The head is worth what its
    /// drafts save the decode: before the load measured its acceptance, the
    /// generic prior from its declared depth (`holdings::prior_accepted`),
    /// its drafted step the decode step with the step of its verified rows
    /// in place of one row's, both in the form the bank decodes in now; and
    /// at least the load each request would pay again, its weights read
    /// along the model store's read line. Vision is worth that load each use
    /// would pay again. A component any of whose figures is not priced yet
    /// (the serving rates, the read line, the head's decode step and price)
    /// is unpriced: its worth is unknown, not its import seconds. None while
    /// a request is resident: the components are released only idle
    /// (`release_idle_optional_components`).
    fn optional_holdings(&self, rates: Option<&WorkloadRates>, price: Option<&crate::assessment::DecodePrice>) -> Result<Vec<OptionalHolding>, DomainError> {
        if !self.optional_idle() {
            return Ok(Vec::new());
        }
        let bytes = self.execution.resources().bytes();
        let read = self.expert_measures().and_then(|measures| measures.read.measured().copied());
        let mut holdings = Vec::new();
        if self.family.head_is_bound() {
            let depth = u32::from(self.execution.policy().method().widest());
            let accepted = super::holdings::prior_accepted(depth);
            let step = self.load_costs().map(|costs| &costs.step);
            let held = self.held_slots()?;
            holdings.push(match (rates, read, step, price, held) {
                (Some(rates), Some(read), Some(step), Some(price), Some(held)) => {
                    let compute = |rows: u32| {
                        step(rows, 0)
                            .map(|forms| price.compute(held, forms))
                            .map_err(|error| DomainError::invariant(format!("the decode step: {error}")))
                    };
                    let plain = price.step(held).total_seconds;
                    let drafting = super::holdings::Drafting {
                        tokens_per_s: rates.decode_steps_per_s * (1. + accepted),
                        plain_step_seconds: plain,
                        drafted_step_seconds: plain - compute(1)? + compute(1 + depth)?,
                        accepted_per_step: accepted,
                    };
                    let load = read.seconds(bytes.head_weights);
                    super::holdings::drafter(bytes.head_weights, true, drafting, rates.requests_per_s, load)
                }
                _ => OptionalHolding::unpriced(OptionalComponent::Drafter),
            });
        }
        if self.family.vision_is_bound() {
            holdings.push(match (rates, read) {
                (Some(rates), Some(read)) => {
                    let load = read.seconds(bytes.vision_weights);
                    super::holdings::vision(bytes.vision_weights, true, rates.vision_uses_per_s, load)
                }
                _ => OptionalHolding::unpriced(OptionalComponent::Vision),
            });
        }
        Ok(holdings)
    }

    /// The idle bound optional component the optional rung releases, and
    /// its rate per byte where its worth is known: the priced one worth
    /// least per byte, an unpriced one only once none is priced
    /// (`holdings::cheapest`); none with none bound and idle.
    pub fn lowest_optional(&mut self) -> Result<Option<(OptionalComponent, Option<f64>)>, DomainError> {
        let rates = self.serving_rates()?;
        let price = self.decode_price()?;
        Ok(super::holdings::cheapest(&self.optional_holdings(rates.as_ref(), price.as_ref())?))
    }

    /// The room a settle grows into in `store`, which `tiers` lie in
    /// (`MemoryPace::supply`): its free room what a growth's claim admits
    /// there beside the workload's unborn state, the measurement job's
    /// claim ([`Self::growth_room`]), and
    /// within the pace's headroom; its borrowable room the one store's
    /// bank's beyond that where it may borrow ([`Self::growth_rooms`]). In
    /// the Reclaim band, or blind, nothing is free. None before a probe of
    /// the store.
    fn growth_supply(&self, store: seismic::StoreId, tiers: &[BankTier]) -> Result<Option<Supply>, DomainError> {
        let Some(mut supply) = self.pace.borrow().supply(store, self.store_charge(store), WORKLOAD_SPAN) else {
            return Ok(None);
        };
        let tier = match tiers.contains(&BankTier::Bank) {
            true => BankTier::Bank,
            false => tiers[0],
        };
        let rooms = self.growth_rooms(tier);
        let bytes = match self.growth_room(tier, &rooms) {
            Ok(bytes) => bytes,
            Err(super::ClaimRefusal::Reclaim(_) | super::ClaimRefusal::Blind(_)) => vec![0; rooms.len()],
            Err(refusal) => return Err(DomainError::from(refusal)),
        };
        let room = |room: Room| rooms.iter().position(|&of| of == room).map_or(0, |at| bytes[at]);
        let committed = room(Room::Committed).min(supply.free);
        supply.borrowable = room(Room::Borrowed).saturating_sub(committed).min(supply.borrowable);
        supply.free = committed;
        Ok(Some(supply))
    }

    /// Settle the serving heap at a reading: in each store the bank lies
    /// in, its holdings' curves at the load's serving rates and, in device
    /// storage's store, a prompt group's places ([`Self::places_curve`]),
    /// the unborn state's room where the bank's slots may grow into it
    /// ([`Self::state_curve`]), the bound optional components and the
    /// owner's prefixes, into the store's room ([`Self::growth_supply`]).
    /// The places target and the state lent are set here; the rest is the
    /// owner's to apply. Nothing grows beside the measurement job's claim.
    ///
    /// It runs only when its inputs changed since the last
    /// ([`SettleStands::holds`]): settling the same holdings in the same
    /// room again decides the same, so a reading changes nothing where it
    /// moved neither. One whose decision the owner could not carry out
    /// settles again at the next reading ([`Self::settle_again`]).
    pub fn settle_memory(&mut self, owner: &OwnerHoldings) -> Result<ServingSettlement, DomainError> {
        let key = self.settle_key(owner);
        if self.settled.as_ref().is_some_and(|stands| stands.holds(&key)) {
            return Ok(ServingSettlement::default());
        }
        #[cfg(test)]
        self.settles.set(self.settles.get() + 1);
        let (settlement, moves) = self.settle_now(owner)?;
        self.settled = Some(SettleStands { key, moves });
        Ok(settlement)
    }

    /// The last settle's decision was not carried out (a resize deferred or
    /// refused): the next settles afresh.
    pub fn settle_again(&mut self) {
        self.settled = None;
    }

    /// What a settle reads now that moves its outcome ([`SettleKey`]).
    fn settle_key(&self, owner: &OwnerHoldings) -> SettleKey {
        let pace = self.pace.borrow();
        let mut rooms = self
            .bank_stores()
            .into_iter()
            .map(|(store, _)| store)
            .chain(self.store_of(seismic::Contract::DeviceOnly))
            .filter_map(|store| {
                pace.supply(store, self.store_charge(store), WORKLOAD_SPAN)
                    .map(|supply| (store, supply.free.saturating_add(supply.borrowable), supply.owed))
            })
            .collect::<Vec<_>>();
        rooms.sort_by_key(|&(store, _, _)| store);
        rooms.dedup_by_key(|&mut (store, _, _)| store);
        SettleKey {
            resizes: self.expert_resizes(),
            places: self.resources.group_places().get(),
            head: self.family.head_is_bound(),
            vision: self.family.vision_is_bound(),
            idle: self.optional_idle(),
            measuring: self.measurement_claim.is_some(),
            requests: self.requests,
            prefixes: (owner.prefixes.len(), owner.prefixes.iter().map(|&(bytes, _)| bytes).sum()),
            rooms,
        }
    }

    /// [`Self::settle_memory`] now, with how far each store's room may move
    /// before it would settle differently ([`SettleStands`]).
    fn settle_now(&mut self, owner: &OwnerHoldings) -> Result<(ServingSettlement, StoreMoves), DomainError> {
        // Before the load is priced every holding's worth is unknown, which
        // is no low worth: a settle then moves nothing.
        let Some(rates) = self.serving_rates()? else {
            return Ok((ServingSettlement::default(), Vec::new()));
        };
        let measuring = self.measurement_claim.is_some();
        // A sequence that grew into lent room took it back.
        self.state_lent = self.state_lent.min(self.unborn_state());
        let device = self.store_of(seismic::Contract::DeviceOnly);
        let mut device_curves = vec![super::holdings::prefixes(owner.prefixes.iter().copied())];
        if let Some(mut places) = self.places_curve(&rates)? {
            if measuring {
                places.offered.clear();
            }
            device_curves.push(places);
        }
        // A settle moves only the components whose worth is known.
        // The decode's price once a settle: every holding's worth is read on it.
        let price = self.decode_price()?;
        device_curves.extend(self.optional_holdings(Some(&rates), price.as_ref())?.into_iter().filter_map(OptionalHolding::curve));
        let mut stores = self.bank_stores();
        if let Some(device) = device.filter(|device| !stores.iter().any(|(store, _)| store == device)) {
            stores.push((device, Vec::new()));
        }
        let mut supplies = Vec::new();
        for (store, tiers) in stores {
            if let Some(supply) = self.growth_supply(store, &tiers)? {
                supplies.push((store, tiers, supply));
            }
        }
        // How far this settle can move the bank: releases cover what is owed
        // and make room for the other holdings' offers by exchange; grants
        // take the free and borrowable room and what the other holdings
        // would give up by exchange.
        let others = |steps: fn(&Curve<Holding>) -> &[Step]| -> u64 {
            device_curves.iter().flat_map(|curve| steps(curve)).map(|step| step.bytes).sum()
        };
        // The state's unlent room is lent only to the bank's slots in device
        // storage's store, only while they may grow and only once their
        // moves, which price the lending, are timed or measured
        // (`bank_heap`): it reaches as far as the bank's slots are priced
        // then, and no further.
        let moves_known = self.resize_held.last.contains_key(&BankHolding::Experts)
            || self.expert_measures().is_some_and(|measures| measures.moves.measured().is_some());
        let lends = !measuring
            && moves_known
            && device.is_some()
            && self.store_of(BankTier::Bank.storage()) == device
            && self.expert_bank_plan().zip(self.family.expert_bank()).is_some_and(|(plan, bank)| bank.slots() < plan.ceiling);
        let lendable = match lends {
            true => self.unborn_state().saturating_sub(self.state_lent),
            false => 0,
        };
        // Each in its tier's store: what is owed there and what the
        // device's other holdings offer, in device storage's store, with
        // the state's lendable room the bank's slots there may take.
        let reach = |tier: BankTier| {
            let store = self.store_of(tier.storage());
            let supply = supplies.iter().find(|(of, _, _)| Some(*of) == store).map(|(_, _, supply)| supply);
            let device = store.is_some() && store == device;
            let beside = |steps: fn(&Curve<Holding>) -> &[Step]| if device { others(steps) } else { 0 };
            let lent = match tier {
                BankTier::Bank => lendable,
                BankTier::Host => 0,
            };
            Reach {
                held: supply.map_or(0, |supply| supply.owed).saturating_add(beside(|curve| &curve.offered)),
                offered: match measuring {
                    true => 0,
                    false => supply
                        .map_or(0, |supply| supply.free.saturating_add(supply.borrowable))
                        .saturating_add(beside(|curve| &curve.held))
                        .saturating_add(lent),
                },
            }
        };
        let reach = Reaches { tiers: BankTier::ALL.map(reach), shared: self.tiers_shared() };
        let bank = self.bank_heap(price.as_ref(), rates.decode_steps_per_s, reach)?;
        let slots = bank.iter().find(|(curve, _)| curve.holding == BankHolding::Experts && curve.kind == BankTier::Bank);
        if let Some((_, experts)) = slots.filter(|_| lends) {
            device_curves.extend(self.state_curve(&rates, experts)?);
        }
        let mut settlement = ServingSettlement::default();
        let mut moves = Vec::new();
        let curve_of = |holding: Holding| device_curves.iter().find(|curve| curve.holding == holding);
        let step_bytes = |steps: &[Step]| steps.iter().map(|step| step.bytes).sum::<u64>();
        let (mut places, mut lent) = (self.places_target.get(), self.state_lent);
        for (store, tiers, supply) in supplies {
            let mut heap: Vec<Curve<Holding>> =
                bank.iter().filter(|(curve, _)| tiers.contains(&curve.kind)).map(|(_, heap)| heap.clone()).collect();
            if Some(store) == device {
                heap.extend(device_curves.iter().cloned());
            }
            let settled = value::settle(&heap, supply);
            let steps = |of: &[(Holding, usize)], holding: &Holding| of.iter().find(|(by, _)| by == holding).map_or(0, |&(_, steps)| steps);
            let least = |next: &dyn Fn(&Curve<Holding>) -> Option<u64>| heap.iter().filter_map(next).min();
            moves.push((
                store,
                least(&|curve| curve.offered.get(steps(&settled.granted, &curve.holding)).map(|step| step.bytes)),
                least(&|curve| curve.held.get(steps(&settled.released, &curve.holding)).map(|step| step.bytes)),
            ));
            let curve = |holding: &BankHolding| bank.iter().find(|(curve, _)| curve.holding == *holding);
            for (holding, steps) in &settled.released {
                match holding {
                    Holding::Prefixes => settlement.prefixes += steps,
                    Holding::Optional(component) => settlement.optional.push(*component),
                    Holding::Bank(holding) => {
                        let (curve, heap) = curve(holding).expect("a released bank holding has a curve");
                        let per_byte = value::per_byte(&heap.held[..*steps]);
                        settlement.releases.push(BankRelease { holding: *holding, to: curve.held[steps - 1].to, per_byte });
                    }
                    Holding::Places => places -= steps,
                    Holding::State => lent += step_bytes(&curve_of(Holding::State).expect("a released state has a curve").held[..*steps]),
                    Holding::PassRows => unreachable!("pass rows hold nothing a settle releases"),
                }
            }
            for (holding, steps) in &settled.granted {
                let holding = match holding {
                    Holding::Places => {
                        places += steps;
                        continue;
                    }
                    Holding::State => {
                        lent -= step_bytes(&curve_of(Holding::State).expect("a granted state has a curve").offered[..*steps]);
                        continue;
                    }
                    Holding::Bank(holding) => holding,
                    Holding::Prefixes | Holding::Optional(_) | Holding::PassRows => {
                        unreachable!("the owner's prefixes and the bound optional components offer nothing")
                    }
                };
                let (curve, heap) = curve(holding).expect("a granted bank holding has a curve");
                let bytes = curve.offered[..*steps].iter().map(|step| step.bytes).sum();
                // What of its bytes the grant borrowed (only a holding that
                // borrows does, [`Curve::borrows`]).
                let lent = settled.borrowed.iter().find(|(of, _)| *of == Holding::Bank(*holding)).map_or(0, |&(_, bytes)| bytes);
                settlement.grants.push(BankGrant {
                    holding: *holding,
                    kind: curve.kind,
                    from: curve.at,
                    to: curve.offered[steps - 1].to,
                    bytes,
                    borrowed: lent,
                    per_byte: value::per_byte(&heap.offered[..*steps]),
                });
            }
        }
        settlement.grants.sort_by(|a, b| b.per_byte.total_cmp(&a.per_byte));
        settlement.releases.sort_by(|a, b| a.per_byte.total_cmp(&b.per_byte));
        // The places target and the state lent are the domain's own: they
        // hold from here, the room they reserve or lend with them.
        let places = std::num::NonZeroUsize::new(places).expect("a settle releases no place below one");
        if places != self.places_target || lent != self.state_lent {
            tracing::info!(places = places.get(), lent, "prompt places and lent state settled");
        }
        self.places_target = places;
        self.state_lent = lent;
        settlement.places_released = self.resources.group_places() > places;
        Ok((settlement, moves))
    }

    /// The lowest rate per byte the bank holds at the load's serving rates:
    /// where the bank stands in the release order beside the owner's
    /// holdings. None without a step to release, or before the load is
    /// priced: what the bank saves is unknown then, not nothing, and the
    /// release ladder takes the bank only once no priced holding is left.
    pub fn bank_release_rate(&mut self) -> Result<Option<f64>, DomainError> {
        let Some(rates) = self.serving_rates()? else {
            return Ok(None);
        };
        Ok(self
            .bank_heap(self.decode_price()?.as_ref(), rates.decode_steps_per_s, Reaches { shared: self.tiers_shared(), ..Reaches::every(Reach { held: 1, offered: 0 }) })?
            .iter()
            .filter_map(|(_, heap)| heap.held.first().map(Step::per_byte))
            .min_by(f64::total_cmp))
    }

    /// The marginal state one retained prefix holds, what the heap prices
    /// each cached prefix's bytes at (`ResourcePlan::retained_entry_bytes`).
    pub fn retained_entry_bytes(&self) -> u64 {
        self.execution.resources().retained_entry_bytes()
    }

    /// The load's serving rates before it serves: the workload prior the
    /// plan priced its holdings with (`planning::Prior`), its prefill on
    /// the measured step's resident form, as the plan priced it, and its
    /// decode steps a banked load's the decode step the bank holds now, a
    /// load without a bank its one form's step at the prior's depth. A
    /// bank whose decode is not priced yet (its read line unmeasured) decodes
    /// no faster than the resident form: its rates are the prior's at that
    /// step until the load serves its own. None before the load's step is
    /// measured.
    pub fn serving_prior(&self) -> Result<Option<WorkloadRates>, DomainError> {
        let Some(step) = self.load_costs().map(|costs| &costs.step) else {
            return Ok(None);
        };
        let prior = self.prior()?;
        let resident = |rows: u32, depth: u32| {
            step(rows, depth)
                .map(|forms| forms.resident)
                .map_err(|error| DomainError::invariant(format!("the decode step: {error}")))
        };
        let priced = match self.family.expert_bank() {
            Some(_) => self.decode_price()?.zip(self.held_slots()?).map(|(price, held)| price.step(held).total_seconds),
            None => None,
        };
        let step_seconds = match priced {
            Some(seconds) => seconds,
            None => resident(1, prior.decode_depth)?,
        };
        prior.rates(step_seconds, resident).map(Some)
    }

    /// Begin `grant` (a settle's, [`Self::settle_memory`]): its bytes,
    /// which its reservation backs with fresh pages of its store, claimed
    /// from the heap first and held until the growth finishes
    /// ([`Self::advance_expert_bank`]); one resize of the holding to its
    /// settled size (`ElasticBank::grow`). A heap that refuses the claim
    /// (`Reclaim`, `Capacity`, `Blind`) grows nothing: the settle sized it
    /// within what the claim admitted, so a refusal is the machine's change
    /// between the two readings, which the next settle reads afresh.
    pub fn grow_expert_bank(&mut self, grant: &BankGrant, wake: crate::CompletionWake) -> Result<BankStep, DomainError> {
        let started = std::time::Instant::now();
        let Some(bank) = self.family.expert_bank() else {
            return Ok(BankStep::AtLimit);
        };
        let unborn = self.bank_unborn(grant.kind.storage());
        let mode = match grant.borrowed {
            0 => crate::platform::FitMode::Live,
            _ => crate::platform::FitMode::Borrow,
        };
        let planned = self.planned_unheld();
        let claim = self.memory.borrow_mut().claim_reserved(grant.kind, grant.bytes, unborn, HoldingClass::Model, mode, planned)?;
        match bank.grow(grant.holding, grant.to, wake) {
            Ok(BankStep::Begun(tier)) => {
                // The one store's bank's slots on borrowed room are its
                // reading's (`BankRoom`).
                let memory = self.memory.get_mut();
                if grant.holding == BankHolding::Experts && grant.kind == BankTier::Bank && memory.bank_room().is_some() {
                    memory.grow_bank(grant.borrowed);
                }
                self.bank_claim = Some(super::GrowthClaim { id: claim, kind: grant.kind, bytes: grant.bytes });
                self.resize_held.begin(tier, started);
                Ok(BankStep::Begun(tier))
            }
            other => {
                self.release_claim(claim);
                other
            }
        }
    }

    /// The release ladder's bank rung: one shrink ([`Self::shrink_expert_bank_to`])
    /// covering what the latest probe says each storage kind's domain is
    /// short until the next can take effect (`MemoryPace::shortfall`): the
    /// ladder probes before each release.
    pub fn shrink_expert_bank(&mut self, bindings: &mut StateBindings, wake: crate::CompletionWake) -> Result<BankStep, DomainError> {
        let deficits = BankTier::ALL.map(|kind| (kind, self.expert_reclaim_deficit(kind.storage())));
        let deficit = move |kind| deficits.iter().find(|(of, _)| *of == kind).map_or(0, |&(_, bytes)| bytes);
        match self.release_target(&deficit)? {
            Ok((holding, to)) => self.shrink_expert_bank_to(bindings, holding, to, wake),
            Err(step) => Ok(step),
        }
    }

    /// One shrink covering `deficit` bytes of device storage: what the
    /// measurement job's claim takes.
    pub fn shrink_expert_bank_by(&mut self, bindings: &mut StateBindings, deficit: u64, wake: crate::CompletionWake) -> Result<BankStep, DomainError> {
        let device = move |kind| if kind == BankTier::Bank { deficit } else { 0 };
        match self.release_target(&device)? {
            Ok((holding, to)) => self.shrink_expert_bank_to(bindings, holding, to, wake),
            Err(step) => Ok(step),
        }
    }

    /// The bank holding a release of `deficit` takes and the size it
    /// settles it at: each short storage kind's holdings settled to cover
    /// its deficit, the lowest rate per byte first; where none is short (a
    /// release asked for its own sake), the one step of every holding that
    /// saves least per byte. Of the holdings released, the one whose step
    /// saves least goes first; the next shrink settles the rest. A storage
    /// kind whose resize runs moves nothing until it finishes: with none
    /// released elsewhere, the rung waits on it ([`BankStep::Running`])
    /// rather than reading the size it is moving to as the floor
    /// ([`BankStep::AtLimit`]).
    fn release_target(&self, deficit: &dyn Fn(BankTier) -> u64) -> Result<Result<(BankHolding, u64), BankStep>, DomainError> {
        // The decode rate scales every bank step alike: the order is the
        // saving's own.
        let short = BankTier::ALL.into_iter().filter(|&kind| deficit(kind) > 0).collect::<Vec<_>>();
        let asked = match short.is_empty() {
            true => BankTier::ALL.to_vec(),
            false => short.clone(),
        };
        let (running, free): (Vec<BankTier>, Vec<BankTier>) = asked.into_iter().partition(|&kind| self.resize_held.runs_in(kind));
        let owed = short.iter().map(|&kind| deficit(kind)).max().unwrap_or(1);
        let bank = self.bank_heap(self.decode_price()?.as_ref(), 1., Reaches { shared: self.tiers_shared(), ..Reaches::every(Reach { held: owed, offered: 0 }) })?;
        let settles: Vec<(Vec<&(BankCurve, Curve<Holding>)>, u64)> = match short.is_empty() {
            true => vec![(bank.iter().filter(|(curve, _)| free.contains(&curve.kind)).collect(), 1)],
            false => free
                .iter()
                .map(|&kind| (bank.iter().filter(|(curve, _)| curve.kind == kind).collect(), deficit(kind)))
                .collect(),
        };
        let mut target: Option<(BankHolding, u64, f64)> = None;
        for (curves, owed) in settles {
            let heap = curves.iter().map(|(_, heap)| heap.clone()).collect::<Vec<_>>();
            for (holding, steps) in value::settle(&heap, owing(owed)).released {
                let Some((curve, heap)) = curves.iter().find(|(_, heap)| heap.holding == holding) else { continue };
                let rate = value::per_byte(&heap.held[..steps]);
                if target.is_none_or(|(_, _, lowest)| rate < lowest) {
                    target = Some((curve.holding, curve.held[steps - 1].to, rate));
                }
            }
        }
        Ok(match target {
            Some((holding, to, _)) => Ok((holding, to)),
            None if !running.is_empty() => Err(BankStep::Running),
            None => Err(BankStep::AtLimit),
        })
    }

    /// Begin one shrink of `holding` to `to`, the size a settle released it
    /// to (`ElasticBank::shrink`), the queued steps orphaned first, as they
    /// may name records it retires; its memory is released at later flight
    /// boundaries ([`Self::advance_expert_bank`]), `wake` completed as it
    /// progresses. The shrink's step records no graph: the decode launch's
    /// were recorded as its slots were made (`NativeGraphPool::grow`). A
    /// load without an elastic bank is at its limit. While a chain is under
    /// way ([`Self::chain_holder`]) nothing begins: its queued chunks are its
    /// holder's to claim, never orphaned, so the shrink is deferred
    /// ([`BankStep::Deferred`]) and its caller decides again once the chain
    /// ends.
    pub fn shrink_expert_bank_to(
        &mut self,
        bindings: &mut StateBindings,
        holding: BankHolding,
        to: u64,
        wake: crate::CompletionWake,
    ) -> Result<BankStep, DomainError> {
        let started = std::time::Instant::now();
        let Some(bank) = self.family.expert_bank() else {
            return Ok(BankStep::AtLimit);
        };
        let step = match self.chain_holder() {
            Some(_) => BankStep::Deferred,
            None => bank.shrink(holding, to, &mut || self.queue.orphan(self.measures.as_ref().and_then(|measures| measures.experts.as_ref()), bindings), wake)?,
        };
        match step {
            BankStep::Begun(tier) => {
                self.resize_held.begin(tier, started);
                self.shrink_step = true;
            }
            _ => self.resize_held.hold(started, super::HeldSource::Decision),
        }
        Ok(step)
    }

    /// Probe memory, settle with nothing of the owner's, and begin the
    /// most valuable growth: a test's one step of what the owner does at a
    /// reading.
    #[cfg(test)]
    pub(crate) fn grow_settled(&mut self, wake: crate::CompletionWake) -> Result<BankStep, DomainError> {
        self.probe_memory()?;
        let settled = self.settle_memory(&OwnerHoldings::default())?;
        // A test's step stands alone: the next settles afresh.
        self.settle_again();
        match settled.grants.first() {
            Some(grant) => self.grow_expert_bank(&grant.clone(), wake),
            None => Ok(BankStep::AtLimit),
        }
    }
}
