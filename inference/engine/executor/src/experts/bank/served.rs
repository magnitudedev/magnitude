use super::{ExpertBank, HeldTable, count};
use crate::assessment::{DecodePrice, LiveDecode};
use crate::planning::Placer;
use crate::domain::{BankCurve, BankHolding, BankResize, BankResized, BankStep, DomainError, ElasticBank, ExpertResidency, Reach, ResizeTimes, SizeStep, TableResidency};
use crate::CompletionWake;
use std::time::{Duration, Instant};
use crate::native::BankMoves;
use std::sync::{Arc, Mutex};
use super::resize::{LandedShrink, Shrinking};

/// A load's expert bank as the domain resizes it (`ElasticBank`): the bank,
/// each group's record moves' kernels, and the stored tables it holds
/// (`ExpertBankPlan::tables`), the one owner of the memory of all of them.
pub(crate) struct ServedBank {
    bank: Arc<ExpertBank>,
    moves: Vec<BankMoves>,
    tables: Vec<HeldTable>,
    /// The resizes done when asked (a table's, a host tier's shrink), each
    /// until an advance reports it, in the order they were done.
    finished: Mutex<std::collections::VecDeque<BankResized>>,
}

/// A shrink of `holding` deferred to the next boundary: every cut from
/// `shallowest` to `deepest` slots falls below a record in flight.
fn in_flight(holding: BankHolding, (shallowest, deepest): (u64, u64)) -> BankStep {
    tracing::info!(?holding, shallowest, deepest, "a shrink waits a boundary: each of its cuts falls below a record in flight");
    BankStep::Deferred
}

/// A holding's sizes a step at a time from `at` toward `bound`, by
/// `charge` (what Seismic charges the holding at a size): each step to the
/// last size before its charge rises again, so a step backs storage and
/// takes every size that storage holds; the first step reaches past
/// `least` at once (an inclusive host tier's slots up to the bank's hold
/// copies of its experts and save nothing). Each step's size and the bytes
/// it backs beyond the last.
pub(super) fn sizes_up(at: u64, least: u64, bound: u64, charge: &dyn Fn(u64) -> Result<u64, DomainError>) -> Result<Vec<(u64, u64)>, DomainError> {
    let mut steps: Vec<(u64, u64)> = Vec::new();
    let mut last = charge(at)?;
    for to in at.saturating_add(1)..=bound {
        let charged = charge(to)?;
        match steps.last_mut() {
            Some(step) if charged == last || to <= least => {
                step.0 = to;
                step.1 += charged - last;
            }
            _ => steps.push((to, charged - last)),
        }
        last = charged;
    }
    Ok(steps)
}

impl ServedBank {
    pub fn new(
        bank: Arc<ExpertBank>,
        programs: &crate::AttestedPrograms,
        tables: Vec<HeldTable>,
    ) -> Result<Self, String> {
        let moves = programs.expert_bank_moves(&bank.plan)?;
        Ok(Self { bank, moves, tables, finished: Mutex::default() })
    }

    /// The slots the bank and its host tier hold now, as the decode prices
    /// them.
    fn held(&self) -> Result<crate::assessment::HeldSlots, DomainError> {
        Ok(crate::assessment::HeldSlots {
            slots: count(self.bank.slots())?,
            host: count(self.bank.host_slots())?,
        })
    }

    /// The reserved storage `holding` lies in (`ExpertBankPlan::holding_storage`).
    fn storage(&self, holding: BankHolding) -> Result<crate::planning::BankTier, DomainError> {
        self.bank.plan.holding_storage(holding).ok_or_else(|| DomainError::invariant(format!("{holding:?} of a bank without it")))
    }

    /// `holding`'s size now: its slots, host slots or records.
    fn size(&self, holding: BankHolding) -> u64 {
        match holding {
            BankHolding::Experts => self.bank.slots(),
            BankHolding::Host => self.bank.host_slots(),
            BankHolding::Table { table } => self.tables[table].table.records(),
        }
    }

    /// What `holding` is charged holding `at` (slots, host slots or
    /// records), as Seismic charges its reservations.
    fn holding_charge(&self, holding: BankHolding, at: u64) -> Result<u64, DomainError> {
        let counts = &self.bank.counts;
        match holding {
            BankHolding::Experts => Ok(counts.resident_charge(at)),
            BankHolding::Host => Ok(counts.host_charge(at)),
            BankHolding::Table { table } => self.tables[table].plan.charge(self.bank.plan.reservation(), at).map_err(DomainError::invariant),
        }
    }

    /// The shallowest shrink of `holding` from `from` releasing `bytes`
    /// ([`crate::planning::releasing`]), down to its floor.
    fn releasing(&self, holding: BankHolding, from: u64, bytes: u64) -> Result<Option<u64>, DomainError> {
        let floor = match holding {
            BankHolding::Experts => self.bank.counts.floor(),
            BankHolding::Host => 0,
            BankHolding::Table { table } => self.tables[table].plan.floor,
        };
        crate::planning::releasing(from, floor, bytes, |at| self.holding_charge(holding, at))
    }

    /// The sizes `holding` releases storage at below `at`, the nearest
    /// first, down to its floor ([`Self::releasing`]), each with the bytes
    /// it releases beyond the last.
    fn sizes_down(&self, holding: BankHolding, at: u64) -> Result<Vec<(u64, u64)>, DomainError> {
        let mut sizes = Vec::new();
        let mut from = at;
        while let Some(to) = self.releasing(holding, from, 0)? {
            if to >= from {
                return Err(DomainError::invariant(format!("{holding:?} releases nothing from {from} to {to}")));
            }
            sizes.push((to, self.holding_charge(holding, from)? - self.holding_charge(holding, to)?));
            from = to;
        }
        Ok(sizes)
    }

    /// `holding`'s curve at `at` in `kind` storage: its release steps
    /// ([`Self::sizes_down`]) and its growth steps ([`sizes_up`], from
    /// `least`, to `bound`), each as far as `reach` goes ([`Reach`]) and
    /// priced by `saved(smaller, larger)`, what holding the larger size
    /// saves a decode step.
    fn curve(
        &self,
        holding: BankHolding,
        at: u64,
        (least, bound): (u64, u64),
        reach: Reach,
        saved: &dyn Fn(u64, u64) -> Result<f64, DomainError>,
    ) -> Result<BankCurve, DomainError> {
        let kind = self.storage(holding)?;
        let priced = |sizes: Vec<(u64, u64)>, reach: u64, growing: bool| -> Result<Vec<SizeStep>, DomainError> {
            let (mut steps, mut from, mut moved) = (Vec::new(), at, 0u64);
            for (to, bytes) in sizes {
                if moved >= reach {
                    break;
                }
                let saves = match growing {
                    true => saved(from, to)?,
                    false => saved(to, from)?,
                };
                steps.push(SizeStep { to, bytes, saves });
                (from, moved) = (to, moved.saturating_add(bytes));
            }
            Ok(steps)
        };
        let held = priced(self.sizes_down(holding, at)?, reach.held, false)?;
        let offered = priced(sizes_up(at, least, bound, &|size| self.holding_charge(holding, size))?, reach.offered, true)?;
        Ok(BankCurve { holding, kind, at, held, offered })
    }

    /// Shrink the host tier now, between flights, to the deepest of `cuts`
    /// (its shallowest and deepest host slots) that keeps every record in
    /// flight, decided at `decided` and queued `queued`: the shrink the next
    /// advance reports, `wake` completed at once.
    fn shrink_host(&self, cuts: (u64, u64), decided: Instant, queued: Duration, wake: CompletionWake) -> Result<BankStep, DomainError> {
        let holding = BankHolding::Host;
        let LandedShrink { from, to: at, before, after } = match self.bank.shrink_host(cuts)? {
            Shrinking::Begun(shrunk) => shrunk,
            Shrinking::Running => return Ok(BankStep::Running),
            Shrinking::Deferred => return Ok(in_flight(holding, cuts)),
        };
        let done = Instant::now();
        let times = ResizeTimes {
            decided,
            queued,
            planned: done - decided - queued,
            trips: Default::default(),
            moves: Duration::ZERO,
            prepared: done,
            installed: done,
            released: done,
        };
        let released = before
            .checked_sub(after)
            .ok_or_else(|| DomainError::invariant("a shrink raised its records' charge"))?;
        let resized = BankResized::Shrunk { holding, from, to: at, released, times };
        self.finished.lock().expect("table resize outcome").push_back(resized);
        wake.complete();
        Ok(BankStep::Begun(self.storage(holding)?))
    }

    /// Whether a resize of the holdings in `tier` storage runs: of the
    /// bank's store, the experts' (until its advance finishes it) or a
    /// table's or landed shrink's (until an advance reports it); of host RAM,
    /// the host tier's growth or its backing to its planned slots
    /// ([`ExpertBank::back_host`]), or its landed shrink. Each store resizes
    /// on its own.
    fn running_in(&self, tier: crate::planning::BankTier) -> Result<bool, DomainError> {
        let finished = self
            .finished
            .lock()
            .expect("table resize outcome")
            .iter()
            .map(|resized| self.storage(resized.holding()))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(finished.contains(&tier)
            || match tier {
                crate::planning::BankTier::Bank => self.bank.resizing(),
                crate::planning::BankTier::Host => self.bank.host_growing() || self.bank.host_backing(),
            })
    }

    /// Whether a growth may begin: no resize the owner began runs in any
    /// store. The host tier's backing is the open's, which a growth runs
    /// beside (the tier offers no growth while it runs, `ElasticBank::curves`).
    fn growth_waits(&self) -> Result<bool, DomainError> {
        Ok(self.bank.resizing()
            || self.bank.host_growing()
            || !self.finished.lock().expect("table resize outcome").is_empty())
    }

    /// Resize table `index` to `records` records, now: the resize the next
    /// advance reports, `wake` completed at once.
    fn resize_table(
        &self,
        index: usize,
        records: u64,
        decided: Instant,
        queued: Duration,
        wake: CompletionWake,
    ) -> Result<BankStep, DomainError> {
        let held = &self.tables[index];
        let charged = || {
            held.table
                .charged_bytes()
                .ok_or_else(|| DomainError::invariant("a held table's slots are charged"))
        };
        let (before, from) = (charged()?, held.table.records());
        let outcome = held.table.resize(records).map_err(DomainError::invariant)?;
        let (after, to, done) = (charged()?, held.table.records(), Instant::now());
        let times = ResizeTimes {
            decided,
            queued,
            planned: done - decided - queued,
            trips: Default::default(),
            moves: Duration::ZERO,
            prepared: done,
            installed: done,
            released: done,
        };
        let holding = BankHolding::Table { table: index };
        let resized = match outcome {
            // A shrink needs no memory: its refusal breaks an invariant, as the bank's does.
            Err(refusal) if records < from => {
                return Err(DomainError::invariant(format!("a held table's storage refused a shrink: {refusal}")));
            }
            Err(refusal) => BankResized::GrowthRefused { holding, at: to, refusal },
            Ok(()) if after < before => BankResized::Shrunk { holding, from, to, released: before - after, times },
            Ok(()) => BankResized::Grown { holding, from, to, grown: after - before, refused: None, times },
        };
        self.finished.lock().expect("table resize outcome").push_back(resized);
        wake.complete();
        Ok(BankStep::Begun(self.storage(holding)?))
    }
}

impl ElasticBank for ServedBank {
    fn slots(&self) -> u64 {
        self.bank.slots()
    }

    #[cfg(test)]
    fn starve(&self, duration: std::time::Duration) -> bool {
        self.bank.starve(duration);
        true
    }

    /// Its tables resize on the domain's thread: only the bank's records
    /// rise off it.
    fn hold_rises(&self) -> std::sync::MutexGuard<'_, ()> {
        self.bank.hold_rises()
    }

    fn lines(&self) -> std::sync::Arc<[crate::assessment::BankLines]> {
        self.bank.lines()
    }

    fn price(&self, lines: Option<std::sync::Arc<[crate::assessment::ExpertLines]>>, pass: Option<crate::assessment::ExpertPass>) {
        self.bank.watcher.shared().price(lines, pass);
    }

    fn stop_growths(&self) {
        // Both stores: a claim's hold would leave either one's slots waiting.
        self.bank.stop_growth(crate::planning::BankTier::Bank);
        self.bank.stop_growth(crate::planning::BankTier::Host);
    }

    fn hold_traffic(&self) -> crate::domain::TrafficHold {
        self.bank.watcher.shared().hold_traffic()
    }

    fn traffic_quiet(&self) -> bool {
        self.bank.watcher.shared().traffic_quiet()
    }

    fn wait_traffic_quiet(&self) -> Result<(), DomainError> {
        self.bank.watcher.shared().wait_traffic_quiet().map_err(DomainError::invariant)
    }

    fn tier_copy(&self, costs: &crate::assessment::TierCopyCosts) -> Result<(), DomainError> {
        // The grid first: a copy is priced only by the blocks the device
        // launches it with.
        self.bank.tier_copy_blocks(costs.blocks).map_err(DomainError::invariant)?;
        self.bank.watcher.shared().price_copy(costs);
        Ok(())
    }

    fn device_convert(&self, lines: &[crate::assessment::RecordLine]) {
        self.bank.device_convert(lines);
    }

    #[cfg(test)]
    fn slow_host_landing(&self) {
        self.bank.watcher.shared().slow_host_landing();
    }

    #[cfg(test)]
    fn device_landed(&self) -> u64 {
        self.bank.watcher.shared().device_landed()
    }

    #[cfg(test)]
    fn resident_decodes(&self) -> u64 {
        self.bank.resident_decodes()
    }

    fn shrink(
        &self,
        holding: BankHolding,
        to: u64,
        between_flights: &mut dyn FnMut() -> Result<(), DomainError>,
        wake: CompletionWake,
    ) -> Result<BankStep, DomainError> {
        let kind = self.storage(holding)?;
        if holding == BankHolding::Host && (self.bank.host_backing() || self.bank.host_growing()) {
            tracing::info!("the host tier's growth stops at {} host slots for a shrink to {to}", self.bank.host_slots());
        }
        // A store's growth gives way to a shrink of its own memory, at its
        // next slot; a resize of it still running decides it again at a
        // later boundary. One of the other store's is no reason to wait.
        self.bank.stop_growth(kind);
        if self.running_in(kind)? {
            return Ok(BankStep::Running);
        }
        let from = self.size(holding);
        // The nearest size releasing storage, and the settled one past it:
        // at its floor, or settled at its size, the holding releases nothing.
        let Some(shallowest) = self.releasing(holding, from, 0)?.filter(|_| to < from) else {
            return Ok(BankStep::AtLimit);
        };
        let deepest = to.min(shallowest);
        let decided = Instant::now();
        // No step reads the records the shrink releases, nor runs while the
        // watcher plans it: the queue drained, a step still running defers it.
        between_flights()?;
        if !self.in_gap() {
            return Ok(BankStep::Deferred);
        }
        let queued = decided.elapsed();
        match (holding, self.bank.plan.placer()) {
            (BankHolding::Experts, Placer::Tier) => self.bank.shrink(&self.moves, deepest, decided, wake),
            (BankHolding::Experts, Placer::Bank) => {
                match self.bank.shrink_store(&self.moves, (shallowest, deepest), decided, queued, wake)? {
                    Shrinking::Begun(()) => Ok(BankStep::Begun(kind)),
                    Shrinking::Running => Ok(BankStep::Running),
                    Shrinking::Deferred => Ok(in_flight(holding, (shallowest, deepest))),
                }
            }
            (BankHolding::Host, _) => self.shrink_host((shallowest, deepest), decided, queued, wake),
            (BankHolding::Table { table: index }, _) => self.resize_table(index, deepest, decided, queued, wake),
        }
    }

    fn carried(&self) -> Result<(), DomainError> {
        self.bank.carried()
    }

    fn curves(&self, price: Option<&DecodePrice>, reach: Reach) -> Result<Vec<BankCurve>, DomainError> {
        let (plan, held) = (&self.bank.plan, self.held()?);
        // What holding `to` rather than `from` saves the decode step;
        // nothing known without a price. Adjacent steps share a size, whose
        // decode step is modelled once.
        let steps = std::cell::RefCell::new(Vec::<(crate::assessment::HeldSlots, f64)>::new());
        let step = |held: crate::assessment::HeldSlots, price: &DecodePrice| {
            if let Some(&(_, seconds)) = steps.borrow().iter().find(|(of, _)| *of == held) {
                return seconds;
            }
            let seconds = price.step(held).total_seconds;
            steps.borrow_mut().push((held, seconds));
            seconds
        };
        let saved = |from: crate::assessment::HeldSlots, to: crate::assessment::HeldSlots| {
            price.map_or(0., |price| step(from, price) - step(to, price))
        };
        type Saves<'a> = Box<dyn Fn(u64, u64) -> Result<f64, DomainError> + 'a>;
        let mut holdings: Vec<(BankHolding, u64, (u64, u64), Saves<'_>)> = Vec::new();
        // The experts' slots, up to where the decode's misses stop falling.
        let at = self.bank.slots();
        let bound = match price {
            Some(price) => price.saturation().map_or(plan.ceiling, |saturation| plan.ceiling.min(saturation)).max(at),
            None => at,
        };
        let slots = |slots: u64| -> Result<_, DomainError> { Ok(crate::assessment::HeldSlots { slots: count(slots)?, ..held }) };
        let experts = move |from: u64, to: u64| -> Result<f64, DomainError> { Ok(saved(slots(from)?, slots(to)?)) };
        holdings.push((BankHolding::Experts, at, (at, bound), Box::new(experts)));
        // The host tier, the inclusive copy of every expert, which every
        // miss reaches first: up to every expert, beside no backing of it
        // to its planned slots already on its way.
        if let Some((host, ceiling)) = self.bank.host_tier() {
            let bound = match (price, self.bank.host_backing()) {
                (Some(_), false) => ceiling.max(host),
                _ => host,
            };
            let tier = move |host: u64| -> Result<_, DomainError> { Ok(crate::assessment::HeldSlots { host: count(host)?, ..held }) };
            let saves = move |from: u64, to: u64| -> Result<f64, DomainError> { Ok(saved(tier(from)?, tier(to)?)) };
            holdings.push((BankHolding::Host, host, (host.max(at), bound), Box::new(saves)));
        }
        // Each stored table's records, up to where its curve's misses stop
        // falling; a size its curve does not price saves nothing known.
        let line = price.map(DecodePrice::read);
        for (index, table) in self.tables.iter().enumerate() {
            let (records, curve) = (table.table.records(), table.table.curve());
            let bound = match line {
                Some(_) => table.plan.records.get().min(curve.saturation()).max(records),
                None => records,
            };
            let saves = move |from: u64, to: u64| -> Result<f64, DomainError> {
                Ok(line
                    .and_then(|line| curve.step_seconds(from, line).zip(curve.step_seconds(to, line)))
                    .map_or(0., |(from, to)| from - to))
            };
            holdings.push((BankHolding::Table { table: index }, records, (records, bound), Box::new(saves)));
        }
        // What the holdings offer first, then what they hold as far as a
        // release reaches: what is owed and what any offer, the bank's own
        // among them, may take by exchange.
        let mut curves = holdings
            .iter()
            .map(|(holding, at, range, saves)| self.curve(*holding, *at, *range, Reach { held: 0, offered: reach.offered }, saves))
            .collect::<Result<Vec<_>, _>>()?;
        let offered = curves.iter().flat_map(|curve| &curve.offered).map(|step| step.bytes).fold(0, u64::saturating_add);
        let release = Reach { held: reach.held.saturating_add(offered), offered: 0 };
        for (curve, (holding, at, range, saves)) in curves.iter_mut().zip(&holdings) {
            curve.held = self.curve(*holding, *at, *range, release, saves)?.held;
        }
        Ok(curves)
    }

    fn live(&self) -> Option<LiveDecode> {
        self.bank.watcher.shared().live()
    }

    fn streamed_layers(&self, lines: Arc<[crate::assessment::ExpertLines]>) -> crate::assessment::StreamedLayers {
        let plan = &self.bank.plan;
        crate::assessment::StreamedLayers::new(plan.layers().iter().map(|layer| layer.experts).collect(), plan.selected(), lines)
    }

    fn grow(&self, holding: BankHolding, to: u64, wake: CompletionWake) -> Result<BankStep, DomainError> {
        if self.growth_waits()? {
            return Ok(BankStep::Running);
        }
        if to <= self.size(holding) {
            return Ok(BankStep::AtLimit);
        }
        match holding {
            BankHolding::Experts => self.bank.grow(to, wake),
            BankHolding::Table { table: index } => self.resize_table(index, to, Instant::now(), Duration::ZERO, wake),
            BankHolding::Host => self.bank.grow_host(to, wake),
        }
    }

    fn holding_storage(&self, holding: BankHolding) -> Result<crate::planning::BankTier, DomainError> {
        self.storage(holding)
    }

    /// The records' growth backs as it runs; a table grows on the domain's
    /// thread at once, all it grew backed while its outcome waits for the
    /// next advance to report it.
    fn growth_backed(&self, tier: crate::planning::BankTier) -> Result<u64, DomainError> {
        let mut backed = self.bank.growth_backed(tier)?;
        for resized in self.finished.lock().expect("table resize outcome").iter() {
            if let BankResized::Grown { holding: holding @ BankHolding::Table { .. }, grown, .. } = resized {
                if self.storage(*holding)? == tier {
                    backed += grown;
                }
            }
        }
        Ok(backed)
    }

    /// The records' resize advances at every boundary (a shrink's install
    /// is due at one), whatever else finished: one finished resize is
    /// reported a boundary, the others at the next.
    fn advance(&self, wake: CompletionWake) -> Result<Option<BankResized>, DomainError> {
        if let Some(finished) = self.bank.advance(wake)? {
            return Ok(Some(finished));
        }
        if let Some(finished) = self.bank.advance_host()? {
            return Ok(Some(finished));
        }
        Ok(self.finished.lock().expect("table resize outcome").pop_front())
    }

    fn running_begun(&self) -> Option<crate::domain::RunningBegun> {
        self.bank.take_running_begun()
    }

    fn in_gap(&self) -> bool {
        self.bank.watcher.shared().in_gap()
    }

    fn wait_tier_filled(&self) -> Result<(), DomainError> {
        match self.bank.host_tier() {
            Some(_) => self.bank.watcher.shared().wait_filled().map_err(DomainError::invariant),
            None => Ok(()),
        }
    }

    fn residency(&self) -> Result<ExpertResidency, DomainError> {
        let tables = self
            .tables
            .iter()
            .map(|held| {
                Ok(TableResidency {
                    records: held.table.records(),
                    floor: held.plan.floor,
                    ceiling: held.plan.records,
                    bytes: held
                        .table
                        .charged_bytes()
                        .ok_or_else(|| DomainError::invariant("a held table's slots are charged"))?,
                    last_resize: *held.last_resize.lock().expect("a table's last resize"),
                })
            })
            .collect::<Result<Vec<_>, DomainError>>()?;
        Ok(ExpertResidency { tables, ..self.bank.residency()? })
    }

    fn record_resize(&self, holding: BankHolding, resize: BankResize) {
        match holding {
            BankHolding::Table { table } => {
                let held = self.tables.get(table).expect("a resized table is one the bank holds");
                *held.last_resize.lock().expect("a table's last resize") = Some(resize);
            }
            BankHolding::Experts => self.bank.record_resize(super::BankSlots::Experts, resize),
            BankHolding::Host => self.bank.record_resize(super::BankSlots::Host, resize),
        }
    }
}
