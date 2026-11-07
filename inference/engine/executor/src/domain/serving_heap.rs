//! The serving heap: a loaded model's memory decisions settled on one price
//! ([`super::value`]). The expert bank's holdings (its slots, its host
//! tier's host slots, its stored tables' records) price each step by what
//! the decode step saves at the load's decode steps a second; a prompt
//! group's places by what its fewer groups save at its requests a second
//! ([`ExecutorDomain::group_seconds`]), every load's alike;
//! the optional components by what drafting and encoding save
//! ([`super::holdings`]); the owner's cached prefixes come in as
//! [`OwnerHoldings`]. Each store settles its own
//! room ([`value::Supply`]): free room above its derived reserve, borrowed
//! room at its price, the device's hard limit, and what an ask owes.
//!
//! A settlement is applied by its owners: a bank holding resizes once to
//! its settled size ([`ExecutorDomain::grow_expert_bank`],
//! [`ExecutorDomain::shrink_expert_bank_to`]), a group's places are held as
//! it opens (`grant_group`), the owner evicts the prefixes and releases
//! the optional components it released.

use super::value::{self, Curve, Holding, Rate, Step, Supply};
use super::borrowing::Room;
use super::holdings::OptionalHolding;
use super::{BankCurve, BankHolding, BankStep, DomainError, ExecutorDomain, OptionalComponent, ProgramFamily, Reach, SizeStep, StateBindings};
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
    /// The places a prompt group holds: those held and those granted.
    pub places: usize,
    /// The bank's growths, the most valuable per byte first: one resize
    /// runs at a time, so the first begins and the rest settle again.
    pub grants: Vec<BankGrant>,
    /// The bank's releases, the least valuable per byte first, which go
    /// before any growth.
    pub releases: Vec<BankRelease>,
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
        held: curve.held.iter().zip(from).map(|(size, from)| step(size, returns(from - size.to, size))).collect(),
        offered: curve.offered.iter().map(|size| step(size, 0.)).collect(),
    }
}

/// What prices a prompt group's seconds ([`ExecutorDomain::group_seconds`]).
#[derive(Clone, Debug, PartialEq)]
pub(super) enum GroupSource {
    /// The load is not priced yet: nothing known.
    Unpriced,
    /// A load without a bank: each group's submission.
    Unbanked { submission: f64 },
    /// The load's measured group reads, at the bank's slots now.
    Measured { submission: f64, reads: crate::assessment::GroupReads, slots: crate::assessment::BankSlots },
    /// Before the load measured a group: the step model's passes, their
    /// misses at the bank's holding now.
    Modelled { submission: f64, held: Option<crate::assessment::HeldSlots> },
}

/// What a prompt group's places are priced at ([`ExecutorDomain::places_seconds`]):
/// the places held, the chunks they may grow to, the wait bound (its bits)
/// and the group's source.
#[derive(Clone, Debug, PartialEq)]
pub(super) struct PlacesKey {
    held: usize,
    chunks: usize,
    wait: Option<u64>,
    source: GroupSource,
}

/// The prompt's seconds at each place count from `key`'s held, as last
/// priced.
pub(super) struct PlacesPrompt {
    key: PlacesKey,
    prompt: std::rc::Rc<[f64]>,
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
    fn bank_heap(&self, decode_steps_per_s: f64, reach: Reach) -> Result<Vec<(BankCurve, Curve<Holding>)>, DomainError> {
        let Some(bank) = self.family.expert_bank() else {
            return Ok(Vec::new());
        };
        let price = self.decode_price()?;
        let held = &self.resize_held.last;
        let measures = self.expert_measures();
        let moves = measures.and_then(|measures| measures.moves.measured());
        let returns = self.slot_returns(price.as_ref())?;
        Ok(bank
            .curves(price.as_ref(), reach)?
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
    /// while (a prompt group's places for its group) and then given back,
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

    /// A prompt group's places from those held up to `chunks`, by the class
    /// bytes `bytes` (`bytes[p − held]`, [`Self::class_bytes`]): each place
    /// more worth the seconds it takes off the prompt's groups
    /// ([`Self::places_seconds`]) at `rates`' requests a second, none past a
    /// count whose group runs longer than the wait bound.
    fn places_curve(&self, chunks: usize, bytes: &[u64], rates: &WorkloadRates) -> Result<Curve<Holding>, DomainError> {
        let key = PlacesKey {
            held: self.resources.group_places().get(),
            chunks,
            wait: self.pace.borrow().wait_bound().map(f64::to_bits),
            source: self.group_source()?,
        };
        let held = key.held;
        let prompt = self.places_seconds(key)?;
        Ok(crate::planning::places_curve(
            held,
            held + prompt.len() - 1,
            &|places| bytes[places - held],
            &|places| prompt[places - held],
            rates.requests_per_s,
        ))
    }

    /// The prompt's seconds at each place count from those `key` holds to
    /// the most whose group runs within its wait bound
    /// ([`Self::group_seconds`]). They change only with what `key` names
    /// and the load's measures, so a settle at every reading prices them
    /// once per change, not once per reading (a priced group's seconds
    /// take each layer's union and merge, a count's seconds two groups).
    fn places_seconds(&self, key: PlacesKey) -> Result<std::rc::Rc<[f64]>, DomainError> {
        if let Some(memo) = self.places_prompt.borrow().as_ref().filter(|memo| memo.key == key) {
            return Ok(memo.prompt.clone());
        }
        let group = self.group_seconds(&key.source)?;
        // Each group size's seconds once: every count's prompt is of groups
        // of at most `key.chunks` chunks.
        let sizes = vec![std::cell::Cell::new(None); key.chunks + 1];
        let seconds = |chunks: usize, places: usize| {
            group.as_ref().map(|group| {
                crate::planning::grouped(chunks, places, |steps| {
                    let size = &sizes[steps];
                    size.get().unwrap_or_else(|| {
                        let seconds = group(steps);
                        size.set(Some(seconds));
                        seconds
                    })
                })
            })
        };
        let wait = key.wait.map(f64::from_bits);
        let most = (key.held + 1..=key.chunks)
            .take_while(|&places| !wait.zip(seconds(places, places)).is_some_and(|(wait, group)| group > wait))
            .last()
            .unwrap_or(key.held);
        let prompt: std::rc::Rc<[f64]> = (key.held..=most).map(|places| seconds(key.chunks, places).unwrap_or(0.)).collect();
        drop(group);
        *self.places_prompt.borrow_mut() = Some(PlacesPrompt { key, prompt: prompt.clone() });
        Ok(prompt)
    }

    /// What prices a prompt group's seconds now ([`GroupSource`]).
    fn group_source(&self) -> Result<GroupSource, DomainError> {
        let Some(costs) = self.load_costs() else {
            return Ok(GroupSource::Unpriced);
        };
        let submission = costs.submission;
        if let Some((reads, slots)) = self.family.expert_bank().map(|bank| bank.group_reads()).transpose()?.flatten() {
            return Ok(GroupSource::Measured { submission, reads, slots });
        }
        if self.expert_bank_plan().is_none() {
            return Ok(GroupSource::Unbanked { submission });
        }
        Ok(GroupSource::Modelled { submission, held: self.held_slots()? })
    }

    /// The seconds one group of `steps` prompt chunks takes, as far as its
    /// size changes them, priced by `source`: what a prompt group's places
    /// are worth (`planning::grouped`), and how long one group runs. Its
    /// submission (`LoadCosts::submission`), the rest of its chain queued
    /// behind its first launch; and where the plan banks its experts, their
    /// routed seconds: the load's measured group reads once it measured a
    /// group (`GroupReads::group_seconds`), before then the
    /// step model's passes by group on the executor's costs
    /// (`planning::routed_group_seconds`, at the plan's pass size and the
    /// workload prior's launch region, each pass's misses read at the bank
    /// as it holds now). Generic machine figures and the plan's structure
    /// alone until the load measures its own groups.
    ///
    /// None before the load is priced, or where the routed pass a bank's
    /// groups are priced by is unavailable: no place is worth anything
    /// known, so a group takes none. Never pending: every load serves on
    /// its executor's measured costs, the pass derived from them.
    fn group_seconds<'a>(&'a self, source: &'a GroupSource) -> Result<Option<Box<dyn Fn(usize) -> f64 + 'a>>, DomainError> {
        let pass = self.expert_measures().and_then(|experts| experts.pass.measured());
        Ok(match source {
            GroupSource::Unpriced => None,
            GroupSource::Unbanked { submission } => Some(Box::new(move |_| *submission)),
            GroupSource::Measured { submission, reads, slots } => {
                Some(Box::new(move |steps| submission + reads.group_seconds(*slots, steps, pass)))
            }
            GroupSource::Modelled { submission, held } => {
                let (Some(pass), Some(bank)) = (pass, self.expert_bank_plan()) else {
                    return Ok(None);
                };
                let routed = bank.layers().iter().map(|layer| (layer.experts, bank.selected())).collect::<Vec<_>>();
                let (region, launches) = (self.prior()?.region_rows, bank.pass);
                // Each pass's misses read at the bank as it holds now; none
                // known before its decode is priced.
                let price = self.decode_price()?;
                Some(Box::new(move |steps| {
                    let reads = |rows: u64| price.as_ref().zip(*held).map_or(0., |(price, held)| price.pass_reads(held, rows as f64));
                    submission + crate::planning::routed_group_seconds(pass, &routed, region, steps, launches, &reads)
                }))
            }
        })
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
    fn optional_holdings(&self, rates: Option<&WorkloadRates>) -> Result<Vec<OptionalHolding>, DomainError> {
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
            let price = self.decode_price()?;
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
        Ok(super::holdings::cheapest(&self.optional_holdings(rates.as_ref())?))
    }

    /// The room a settle grows into in `store`, which `tiers` lie in
    /// (`MemoryPace::supply`): its free room what a growth's claim admits
    /// there beside the workload's unborn state, the measurement job's
    /// claim ([`Self::growth_room`]), and
    /// within the pace's headroom; its borrowable room the one store's
    /// bank's beyond that where it may borrow ([`Self::growth_rooms`]). In
    /// the Reclaim band, or blind, nothing is free. None before a probe of
    /// the store.
    fn growth_supply(&self, store: seismic::StoreId, tiers: &[BankTier], horizon: Duration) -> Result<Option<Supply>, DomainError> {
        let Some(mut supply) = self.pace.borrow().supply(store, self.memory_probe().charged_in(store), horizon) else {
            return Ok(None);
        };
        let tier = match tiers.contains(&BankTier::Bank) {
            true => BankTier::Bank,
            false => tiers[0],
        };
        let rooms = self.growth_rooms(tier);
        let room = |room: Room| match rooms.contains(&room) {
            false => Ok(0),
            true => match self.growth_room(tier, room) {
                Ok((bytes, _)) => Ok(bytes),
                Err(super::ClaimRefusal::Reclaim(_) | super::ClaimRefusal::Blind(_)) => Ok(0),
                Err(refusal) => Err(DomainError::from(refusal)),
            },
        };
        let committed = room(Room::Committed)?.min(supply.free);
        supply.borrowable = room(Room::Borrowed)?.saturating_sub(committed).min(supply.borrowable);
        supply.free = committed;
        Ok(Some(supply))
    }

    /// Settle the serving heap at a reading: in each store the bank lies
    /// in, its holdings' curves at the load's serving rates and, in device
    /// storage's store, a prompt group's places up to the chain's capacity,
    /// the bound optional components, and the owner's prefixes, into the
    /// store's room
    /// ([`Self::growth_supply`]). Nothing grows beside the measurement
    /// job's claim.
    pub fn settle_memory(&mut self, owner: &OwnerHoldings) -> Result<ServingSettlement, DomainError> {
        let held = self.resources.group_places().get();
        let (_, bytes) = self.class_bytes(held, self.chain_capacity().get())?;
        self.settle_with(owner, &bytes, false)
    }

    /// The places a prompt group of `chunks` chunks holds as it opens
    /// (`grant_group`): those held, and those a settle with its places
    /// offered (`bytes[p − held]` their class's bytes) grants beside the
    /// bank's slots and the optional components, exchange included. The
    /// group holds its places for itself alone and gives them back as it
    /// ends, so an exchange for them is earned over one group's share of
    /// the requests (at most the workload span), against the moves of the
    /// slots it takes, their return included.
    pub(super) fn settle_places(&mut self, bytes: &[u64]) -> Result<usize, DomainError> {
        Ok(self.settle_with(&OwnerHoldings::default(), bytes, true)?.places)
    }

    /// [`Self::settle_memory`] with a group's places offered at the class
    /// bytes `bytes`, for the group opening where `group`.
    fn settle_with(&mut self, owner: &OwnerHoldings, bytes: &[u64], group: bool) -> Result<ServingSettlement, DomainError> {
        let held = self.resources.group_places().get();
        // Before the load is priced every holding's worth is unknown, which
        // is no low worth: a settle then moves nothing.
        let Some(rates) = self.serving_rates()? else {
            return Ok(ServingSettlement { places: held, ..ServingSettlement::default() });
        };
        let horizon = match group && rates.requests_per_s > 0. {
            true => WORKLOAD_SPAN.min(Duration::from_secs_f64(1. / rates.requests_per_s)),
            false => WORKLOAD_SPAN,
        };
        let measuring = self.measurement_claim.is_some();
        let mut places = self.places_curve(held + bytes.len() - 1, bytes, &rates)?;
        if measuring {
            places.offered.clear();
        }
        let device = self.store_of(seismic::Contract::DeviceOnly);
        let mut device_curves = vec![places, super::holdings::prefixes(owner.prefixes.iter().copied())];
        // A settle moves only the components whose worth is known.
        device_curves.extend(self.optional_holdings(Some(&rates))?.into_iter().filter_map(OptionalHolding::curve));
        let mut stores = self.bank_stores();
        if let Some(device) = device.filter(|device| !stores.iter().any(|(store, _)| store == device)) {
            stores.push((device, Vec::new()));
        }
        let mut supplies = Vec::new();
        for (store, tiers) in stores {
            if let Some(supply) = self.growth_supply(store, &tiers, horizon)? {
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
        let reach = Reach {
            held: supplies.iter().map(|(_, _, supply)| supply.owed).sum::<u64>().saturating_add(others(|curve| &curve.offered)),
            offered: match measuring {
                true => 0,
                false => supplies
                    .iter()
                    .map(|(_, _, supply)| supply.free.saturating_add(supply.borrowable))
                    .fold(others(|curve| &curve.held), u64::saturating_add),
            },
        };
        let bank = self.bank_heap(rates.decode_steps_per_s, reach)?;
        let mut settlement = ServingSettlement { places: held, ..ServingSettlement::default() };
        for (store, tiers, supply) in supplies {
            let mut heap: Vec<Curve<Holding>> =
                bank.iter().filter(|(curve, _)| tiers.contains(&curve.kind)).map(|(_, heap)| heap.clone()).collect();
            if Some(store) == device {
                heap.extend(device_curves.iter().cloned());
            }
            let settled = value::settle(&heap, supply);
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
                    Holding::Places | Holding::PassRows => unreachable!("a group's places and pass rows hold nothing a settle releases"),
                }
            }
            for (holding, steps) in &settled.granted {
                let holding = match holding {
                    Holding::Places => {
                        settlement.places = held + steps;
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
        Ok(settlement)
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
            .bank_heap(rates.decode_steps_per_s, Reach { held: 1, offered: 0 })?
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
        let unborn = self.unborn(grant.kind.storage());
        let mode = match grant.borrowed {
            0 => crate::platform::FitMode::Live,
            _ => crate::platform::FitMode::Borrow,
        };
        let claim = self.memory.borrow_mut().claim_reserved(grant.kind, grant.bytes, unborn, HoldingClass::Model, mode)?;
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
        let bank = self.bank_heap(1., Reach { held: owed, offered: 0 })?;
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
        self.probe_memory(std::time::Duration::from_millis(100))?;
        let settled = self.settle_memory(&OwnerHoldings::default())?;
        match settled.grants.first() {
            Some(grant) => self.grow_expert_bank(&grant.clone(), wake),
            None => Ok(BankStep::AtLimit),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn supply(free: u64) -> Supply {
        Supply { free, owed: 0, borrowable: 0, borrow_price: 0., ceiling: u64::MAX, horizon: Duration::from_secs(60) }
    }

    /// The bank's tail slots, a step of 50 bytes each, saving a 1 ms decode
    /// step a slot at 10 steps a second.
    fn bank() -> Curve<Holding> {
        let curve = BankCurve {
            holding: BankHolding::Experts,
            kind: BankTier::Bank,
            at: 4,
            held: vec![SizeStep { to: 3, bytes: 50, saves: 1e-3 }, SizeStep { to: 2, bytes: 50, saves: 1e-3 }],
            offered: Vec::new(),
        };
        priced(&curve, 10., 0.1, &|_, _| 0.)
    }

    /// A group's places, a place more of 100 bytes saving `seconds` of its
    /// prompt at one prompt a second.
    fn places(seconds: f64) -> Curve<Holding> {
        let step = Step { bytes: 100, worth: Rate::new(seconds), move_seconds: 0. };
        Curve { holding: Holding::Places, borrows: false, held: Vec::new(), offered: vec![step] }
    }

    /// A group opening with no free room takes the bank's lowest slots
    /// where its places save more than they do, both moves paid by the gain
    /// over the horizon; where they save less, the bank keeps its slots.
    #[test]
    fn a_group_opening_takes_the_banks_lowest_slots_when_its_places_are_worth_more() {
        let settled = value::settle(&[bank(), places(0.5)], supply(0));
        assert_eq!(settled.granted, vec![(Holding::Places, 1)]);
        assert_eq!(settled.released, vec![(Holding::Bank(BankHolding::Experts), 2)]);
        let settled = value::settle(&[bank(), places(0.005)], supply(0));
        assert_eq!((settled.granted, settled.released), (vec![], vec![]));
    }

    /// A load without a bank prices its places from the workload prior
    /// before it measured a group: the prior's rates on the load's own
    /// step, and each place more of a group of three chunks saving the one
    /// group's submission it removes (three groups to two, two to one) at
    /// the prior's requests a second. Before the load is priced it has no
    /// rates, and its settle moves nothing.
    #[test]
    fn a_dense_loads_places_are_priced_from_the_prior() {
        use super::super::fixtures::{fixture_of, limits, measure_load, tiny_definition, SUBMISSION};
        let (mut domain, mut bindings) =
            fixture_of(tiny_definition(), limits(2, 3, false), None, false, crate::platform::DeviceRequest::Automatic);
        let bytes = [0, 10, 20];
        assert_eq!(domain.serving_rates().unwrap(), None, "an unpriced load has no rates");
        let held = domain.group_places().get();
        assert_eq!(domain.settle_places(&bytes).unwrap(), held, "an unpriced load's settle moves nothing");
        measure_load(&mut domain, &mut bindings);
        let prior = domain.serving_prior().unwrap().expect("a priced load's prior");
        let step = domain.prior().unwrap().rates(0.015, |_, _| Ok::<_, ()>(0.015)).unwrap();
        assert_eq!(prior, step, "the prior's rates on the load's 15 ms step");
        let places = domain.places_curve(3, &bytes, &prior).unwrap();
        let saved = SUBMISSION * prior.requests_per_s;
        assert_eq!(places.offered.len(), 2);
        for step in &places.offered {
            assert!((step.worth.get() - saved).abs() <= 1e-12 * saved, "{step:?} against {saved}");
            assert_eq!(step.bytes, 10);
        }
    }
}
