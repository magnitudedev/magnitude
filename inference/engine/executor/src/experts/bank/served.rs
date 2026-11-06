use super::{ExpertBank, HeldTable, count, landed_bytes};
use crate::assessment::{BankSlots, DecodePrice, GroupReads, LiveDecode};
use crate::planning::Placer;
use crate::experts::watcher::MoveBudget;
use crate::domain::{BankHolding, BankResize, BankResized, BankStep, DomainError, ElasticBank, ExpertResidency, Growth, ResizeTimes, TableResidency};
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
    /// The rate the last shrink of the host tier released it, in bytes a
    /// second, while holding the submitting thread (the watcher's copies of
    /// the records it keeps, then the storage's release); none before one
    /// finished.
    landed_rate: Mutex<Option<f64>>,
}

/// A holding's next step (a growth's, or a shrink's last): the slots, host
/// slots or records it takes the holding from and leaves it at, the storage
/// kind and bytes it charges and what it saves a decode step, in seconds;
/// none before the load's decode (and a table's curve) prices it.
pub(super) struct Step {
    pub(super) holding: BankHolding,
    pub(super) kind: crate::planning::BankTier,
    pub(super) from: u64,
    pub(super) to: u64,
    pub(super) bytes: u64,
    pub(super) saves: Option<f64>,
}

/// A shrink of `holding` deferred to the next boundary: every cut from
/// `shallowest` to `deepest` slots falls below a record in flight.
fn in_flight(holding: BankHolding, (shallowest, deepest): (u64, u64)) -> BankStep {
    tracing::info!(?holding, shallowest, deepest, "a shrink waits a boundary: each of its cuts falls below a record in flight");
    BankStep::Deferred
}

/// The step a shrink takes, of `steps` (in holding order, the experts
/// first), toward each store's `deficit`: where any store is short, only a
/// step in a short store, none once those are at their floors (memory
/// another store gives back relieves none of it); else any. Of those, the
/// one that saves the decode least per byte. A holding not yet priced has
/// shown no saving and gives way first, the earlier on a tie (the experts
/// before a table, as a bank without tables shrinks).
pub(super) fn shrunk(steps: Vec<Step>, deficit: &dyn Fn(crate::planning::BankTier) -> u64) -> Option<Step> {
    let short = crate::planning::BankTier::ALL.into_iter().any(|kind| deficit(kind) > 0);
    let per_byte = |step: &Step| step.saves.map_or(0., |saves| saves / step.bytes as f64);
    steps
        .into_iter()
        .filter(|step| !short || deficit(step.kind) > 0)
        .min_by(|a, b| per_byte(a).total_cmp(&per_byte(b)))
}

/// The step a growth takes, of `steps`: the host tier's, which no price
/// decides (`ServedBank::growth_steps`); else of the holdings no resize has
/// measured what it holds the submitting thread (`measured` false), the
/// smallest step, which runs to measure it, priced or not; else the priced
/// one that saves the decode most per byte.
pub(super) fn grown(mut steps: Vec<(Step, u64)>, measured: &dyn Fn(BankHolding) -> bool) -> Option<(Step, u64)> {
    if let Some(tier) = steps.iter().position(|(step, _)| step.holding == BankHolding::Host) {
        return Some(steps.swap_remove(tier));
    }
    let (unmeasured, measured): (Vec<_>, Vec<_>) = steps.into_iter().partition(|(step, _)| !measured(step.holding));
    if let Some(step) = unmeasured.into_iter().min_by_key(|(step, _)| step.bytes) {
        return Some(step);
    }
    let per_byte = |step: &Step| step.saves.map(|saves| saves / step.bytes as f64);
    measured
        .into_iter()
        .filter_map(|step| per_byte(&step.0).map(|value| (value, step)))
        .max_by(|(a, _), (b, _)| a.total_cmp(b))
        .map(|(_, step)| step)
}

impl ServedBank {
    pub fn new(
        bank: Arc<ExpertBank>,
        programs: &crate::AttestedPrograms,
        tables: Vec<HeldTable>,
    ) -> Result<Self, String> {
        let moves = programs.expert_bank_moves(&bank.plan)?;
        Ok(Self { bank, moves, tables, finished: Mutex::default(), landed_rate: Mutex::new(None) })
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

    /// Each holding's nearest step that releases physical storage, down to
    /// its floor.
    fn shrink_steps(&self, price: Option<&DecodePrice>) -> Result<Vec<Step>, DomainError> {
        let mut steps = Vec::new();
        let held = self.held()?;
        let saving = |from, to| price.map(|price| price.saving(from, to));
        let slots = self.bank.slots();
        if let Some(to) = self.releasing(BankHolding::Experts, slots, 0)? {
            let below = crate::assessment::HeldSlots { slots: count(to)?, ..held };
            steps.push(Step {
                holding: BankHolding::Experts,
                kind: self.storage(BankHolding::Experts)?,
                from: slots,
                to,
                bytes: self.holding_charge(BankHolding::Experts, slots)? - self.holding_charge(BankHolding::Experts, to)?,
                saves: saving(below, held),
            });
        }
        let host = self.bank.host_slots();
        if host > 0 {
            if let Some(to) = self.releasing(BankHolding::Host, host, 0)? {
                let below = crate::assessment::HeldSlots { host: count(to)?, ..held };
                steps.push(Step {
                    holding: BankHolding::Host,
                    kind: self.storage(BankHolding::Host)?,
                    from: host,
                    to,
                    bytes: self.holding_charge(BankHolding::Host, host)? - self.holding_charge(BankHolding::Host, to)?,
                    saves: saving(below, held),
                });
            }
        }
        let line = price.map(DecodePrice::read);
        for (index, held) in self.tables.iter().enumerate() {
            let holding = BankHolding::Table { table: index };
            let records = held.table.records();
            let Some(to) = self.releasing(holding, records, 0)? else {
                continue;
            };
            let curve = held.table.curve();
            let saves = line.and_then(|line| {
                Some(curve.step_seconds(to, line)? - curve.step_seconds(records, line)?)
            });
            steps.push(Step {
                holding,
                kind: self.storage(holding)?,
                from: records,
                to,
                bytes: self.holding_charge(holding, records)? - self.holding_charge(holding, to)?,
                saves,
            });
        }
        Ok(steps)
    }

    /// Each holding in `kind` storage's growth step, to its target within
    /// `room` bytes, the most the store's claim admits (`growth_target`),
    /// with the bound a step with room enough reaches: the host tier's host
    /// slots up to every expert; the bank's slots up to where the priced
    /// misses stop falling (`DecodePrice::saturation`), a table's records up
    /// to where its curve's do (`TableCurve::saturation`), each at most its
    /// ceiling, and none of these before the load's decode is priced.
    fn growth_steps(
        &self,
        price: Option<&DecodePrice>,
        kind: crate::planning::BankTier,
        room: u64,
    ) -> Result<Vec<(Step, u64)>, DomainError> {
        let (plan, held) = (&self.bank.plan, self.held()?);
        let mut steps = Vec::new();
        // The host tier is the inclusive copy of every expert, which every
        // miss reaches first: it grows to every expert wherever its store
        // has room, priced by no record (a prompt group's cold experts are
        // read from it as much as decode's misses), beside no backing of it
        // already on its way.
        let tiered = plan.holding_storage(BankHolding::Host) == Some(kind);
        if let (true, Some((host, ceiling)), false) = (tiered, self.bank.host_tier(), self.bank.host_backing()) {
            let counts = &self.bank.counts;
            if let Some(to) = counts.host_growth(host, self.bank.slots(), room) {
                let step = Step {
                    holding: BankHolding::Host,
                    kind,
                    from: host,
                    to,
                    bytes: counts.host_charge(to) - counts.host_charge(host),
                    saves: None,
                };
                steps.push((step, ceiling.get()));
            }
        }
        let Some(price) = price else {
            return Ok(steps);
        };
        let target = |from, saturation, bytes: &dyn Fn(u64) -> Result<u64, String>| {
            crate::planning::growth_target(from, saturation, room, bytes).map_err(DomainError::invariant)
        };
        if kind == crate::planning::BankTier::Bank {
            let slots = self.bank.slots();
            let bytes = |to| Ok(self.bank.growth_bytes(to));
            let bound = price.saturation().map_or(plan.ceiling, |saturation| plan.ceiling.min(saturation));
            if let Some(to) = target(slots, bound, &bytes)? {
                let above = crate::assessment::HeldSlots { slots: count(to)?, ..held };
                let step = Step {
                    holding: BankHolding::Experts,
                    kind,
                    from: slots,
                    to,
                    bytes: self.bank.growth_bytes(to),
                    saves: Some(price.saving(held, above)),
                };
                steps.push((step, bound));
            }
            let line = price.read();
            for (index, table) in self.tables.iter().enumerate() {
                let (records, curve) = (table.table.records(), table.table.curve());
                let charge = |records| table.plan.charge(plan.reservation(), records);
                let bytes = |to| Ok(charge(to)? - charge(records)?);
                let bound = table.plan.records.get().min(curve.saturation());
                let Some(to) = target(records, bound, &bytes)? else {
                    continue;
                };
                let step = Step {
                    holding: BankHolding::Table { table: index },
                    kind,
                    from: records,
                    to,
                    bytes: bytes(to).map_err(DomainError::invariant)?,
                    saves: curve.step_seconds(records, line).zip(curve.step_seconds(to, line)).map(|(at, above)| at - above),
                };
                steps.push((step, bound));
            }
        }
        Ok(steps)
    }

    /// The step a shrink takes: of the holdings in storage short of memory
    /// by `deficit`, or of every holding where no storage is measured short
    /// (a release asked for its own sake), the one saving least per byte.
    fn shrinking(
        &self,
        price: Option<&DecodePrice>,
        deficit: &dyn Fn(crate::planning::BankTier) -> u64,
    ) -> Result<Option<Step>, DomainError> {
        Ok(shrunk(self.shrink_steps(price)?, deficit))
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
        *self.landed_rate.lock().expect("landed rate") = Some(released as f64 / times.planned.as_secs_f64());
        let resized = BankResized::Shrunk { holding, from, to: at, released, times };
        self.finished.lock().expect("table resize outcome").push_back(resized);
        wake.complete();
        Ok(BankStep::Begun(self.storage(holding)?))
    }

    /// The target of a shrink of the host tier from `from` host slots,
    /// toward `deepest`: within what one decode step may hold the submitting
    /// thread ([`landed_bytes`]). The owner's ladder resumes as each
    /// finishes, so a deficit is covered a step at a time.
    fn host_target(&self, from: u64, deepest: u64, price: Option<&DecodePrice>) -> Result<u64, DomainError> {
        let rate = *self.landed_rate.lock().expect("landed rate");
        let bytes = landed_bytes(self.step_seconds(price)?, rate);
        let paced = self.releasing(BankHolding::Host, from, bytes)?.expect("a selected shrink releases physical storage");
        Ok(paced.max(deepest))
    }

    /// The target of a shrink of the one store from `from` slots, between
    /// `shallowest`, the nearest that releases storage, and `deepest`: the
    /// deepest whose moves fit `budget` (one decode step's device time), at
    /// least `shallowest`; `shallowest` without a budget. A shrink moves at
    /// most every record it releases (a record moves only from past the new
    /// records), so the moves of the target fit by construction, whichever
    /// records the watcher keeps.
    fn store_target(&self, from: u64, (shallowest, deepest): (u64, u64), budget: Option<MoveBudget>) -> u64 {
        let Some(budget) = budget else {
            return shallowest;
        };
        let counts = &self.bank.counts;
        let released = |to: u64| move |group: usize| counts.records_at(group, from) - counts.records_at(group, to);
        (deepest..shallowest).rev().take_while(|&to| budget.fits(&released(to))).last().unwrap_or(shallowest)
    }

    /// The next growth step of the holdings in `kind` storage within `room`
    /// ([`ElasticBank::growth`]).
    fn growing(
        &self,
        price: Option<&DecodePrice>,
        measured: &dyn Fn(BankHolding) -> bool,
        kind: crate::planning::BankTier,
        room: u64,
    ) -> Result<Option<(Step, u64)>, DomainError> {
        Ok(grown(self.growth_steps(price, kind, room)?, measured))
    }

    /// What one shrink's moves may take: the decode step `price` prices at
    /// the slots held now, each moving group's launches priced on its
    /// `moves` line. None without either: one resize then takes the whole
    /// deficit.
    fn move_budget(
        &self,
        price: Option<&DecodePrice>,
        moves: Option<&crate::assessment::MoveLines>,
    ) -> Result<Option<MoveBudget>, DomainError> {
        Ok(moves.zip(self.step_seconds(price)?).map(|(lines, seconds)| MoveBudget { seconds, lines: lines.clone() }))
    }

    /// The decode step at the slots held now, as `price` prices it, in
    /// seconds; none without it.
    fn step_seconds(&self, price: Option<&DecodePrice>) -> Result<Option<f64>, DomainError> {
        let held = self.held()?;
        Ok(price.map(|price| price.step(held).total_seconds))
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
    /// beside (the tier's own growth waits for it, `Self::growth_steps`).
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

    fn growth_shortfall(&self) -> Result<u64, DomainError> {
        Ok(self.bank.growth_bytes(self.bank.plan.ceiling.max(self.bank.slots())))
    }

    fn whole(&self) -> bool {
        self.bank.slots() >= self.bank.plan.widest().get()
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
        price: Option<&DecodePrice>,
        moves: Option<&crate::assessment::MoveLines>,
        deficit: &dyn Fn(crate::planning::BankTier) -> u64,
        between_flights: &mut dyn FnMut() -> Result<(), DomainError>,
        wake: CompletionWake,
    ) -> Result<BankStep, DomainError> {
        use crate::planning::BankTier;
        // A store's growth gives way only to a shrink its own memory needs:
        // one that measures it short, or a release asked for its own sake,
        // where no store is short. Each gives way at its next slot.
        let unshort = BankTier::ALL.into_iter().all(|tier| deficit(tier) == 0);
        let pressed = |tier: BankTier| unshort || deficit(tier) > 0;
        if pressed(BankTier::Host) && (self.bank.host_backing() || self.bank.host_growing()) {
            let [device, host] = BankTier::ALL.map(|tier| deficit(tier));
            tracing::info!(
                "the host tier's growth stops at {} host slots for a shrink: host short {host} bytes, device short {device} bytes",
                self.bank.host_slots()
            );
        }
        for tier in BankTier::ALL.into_iter().filter(|&tier| pressed(tier)) {
            self.bank.stop_growth(tier);
        }
        // Which stores resize, read once for the decision: one that ends
        // while it prices is decided again at the next boundary.
        let mut running = [false; BankTier::ALL.len()];
        for tier in BankTier::ALL {
            running[tier.index()] = self.running_in(tier)?;
        }
        let mut pressed_tiers = BankTier::ALL.into_iter().filter(|&tier| pressed(tier));
        // A shrink takes a pressed store's holding, or none: where every
        // pressed store resizes, it is decided again at a later boundary
        // whatever the steps price, so none is priced.
        if pressed_tiers.clone().all(|tier| running[tier.index()]) {
            return Ok(BankStep::Running);
        }
        let Some(Step { holding, kind, from, to: shallowest, .. }) = self.shrinking(price, deficit)? else {
            // Every pressed holding is at its floor: a growth that gives way
            // there releases what it backed once its advance reports it.
            return Ok(match pressed_tiers.any(|tier| running[tier.index()]) {
                true => BankStep::Running,
                false => BankStep::AtLimit,
            });
        };
        // A resize of the shrink's own store runs: it is decided again at a
        // later boundary. One of the other store's is no reason to wait.
        if running[kind.index()] {
            return Ok(BankStep::Running);
        }
        // The shallowest target releasing the store's deficit, or the floor.
        let deepest = self.releasing(holding, from, deficit(kind))?
            .expect("a selected shrink releases physical storage");
        let decided = Instant::now();
        // No step reads the records the shrink releases, nor runs while the
        // watcher plans it: the queue drained, a step still running defers it.
        between_flights()?;
        if !self.in_gap() {
            return Ok(BankStep::Deferred);
        }
        let queued = decided.elapsed();
        match (holding, self.bank.plan.placer()) {
            (BankHolding::Experts, Placer::Tier) => {
                let budget = self.move_budget(price, moves)?;
                self.bank.shrink(&self.moves, (shallowest, deepest), budget, decided, wake)
            }
            (BankHolding::Experts, Placer::Bank) => {
                let target = self.store_target(from, (shallowest, deepest), self.move_budget(price, moves)?);
                match self.bank.shrink_store(&self.moves, (shallowest, target), decided, queued, wake)? {
                    Shrinking::Begun(()) => Ok(BankStep::Begun(kind)),
                    Shrinking::Running => Ok(BankStep::Running),
                    Shrinking::Deferred => Ok(in_flight(holding, (shallowest, target))),
                }
            }
            (BankHolding::Host, _) => {
                let target = self.host_target(from, deepest, price)?;
                self.shrink_host((shallowest, target), decided, queued, wake)
            }
            (BankHolding::Table { table: index }, _) => self.resize_table(index, deepest, decided, queued, wake),
        }
    }

    fn carried(&self) -> Result<(), DomainError> {
        self.bank.carried()
    }

    fn growth(
        &self,
        price: Option<&DecodePrice>,
        measured: &dyn Fn(BankHolding) -> bool,
        kind: crate::planning::BankTier,
        room: u64,
    ) -> Result<Option<Growth>, DomainError> {
        // A step is priced only on the decode (`Self::growth_steps`).
        let steps = price.map(|price| price.steps() as f64);
        Ok(self.growing(price, measured, kind, room)?.map(|(step, justified)| Growth {
            holding: step.holding,
            kind: step.kind,
            from: step.from,
            to: step.to,
            bytes: step.bytes,
            saving: step.saves.zip(steps).map(|(saves, steps)| saves * steps),
            justified,
        }))
    }

    fn group_reads(&self) -> Result<Option<(GroupReads, BankSlots)>, DomainError> {
        let Some(reads) = self.bank.watcher.shared().group_reads() else {
            return Ok(None);
        };
        let plan = &self.bank.plan;
        let bytes = plan.slot_bytes().map_err(DomainError::invariant)?;
        let host = self.bank.host_slots();
        Ok(Some((reads, BankSlots { now: self.bank.slots(), floor: self.bank.counts.floor(), bytes, host })))
    }

    fn live(&self) -> Option<LiveDecode> {
        self.bank.watcher.shared().live()
    }

    fn streamed_layers(&self, lines: Arc<[crate::assessment::ExpertLines]>) -> crate::assessment::StreamedLayers {
        let plan = &self.bank.plan;
        crate::assessment::StreamedLayers::new(plan.layers().iter().map(|layer| layer.experts).collect(), plan.selected(), lines)
    }

    fn grow(
        &self,
        price: Option<&DecodePrice>,
        measured: &dyn Fn(BankHolding) -> bool,
        kind: crate::planning::BankTier,
        room: u64,
        wake: CompletionWake,
    ) -> Result<BankStep, DomainError> {
        if self.growth_waits()? {
            return Ok(BankStep::Running);
        }
        match self.growing(price, measured, kind, room)?.map(|(step, _)| step) {
            None => Ok(BankStep::AtLimit),
            Some(Step { holding: BankHolding::Experts, to, .. }) => self.bank.grow(to, wake),
            Some(Step { holding: BankHolding::Table { table: index }, to, .. }) => {
                self.resize_table(index, to, Instant::now(), Duration::ZERO, wake)
            }
            Some(Step { holding: BankHolding::Host, to, .. }) => self.bank.grow_host(to, wake),
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
