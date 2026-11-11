use super::{ExpertBank, HeldTable};
use crate::assessment::{DecodePrice, LiveDecode};
use crate::planning::Placer;
use crate::domain::{BankHolding, BankResize, BankResized, BankStep, DomainError, ElasticBank, ExpertResidency, ResizeTimes, TableResidency};
use crate::experts::derived::{BankShape, HoldingTarget};
use crate::memory::ledger::{CacheKind, Claim, CommittedKind, DeviceLedgers, Recorded};
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
    /// Its committed claims, held for the load: a linked tier's landing
    /// rows and each holding's floor.
    _floors: Vec<Claim>,
}

/// A shrink of `holding` deferred to the next boundary: every cut from
/// `shallowest` to `deepest` slots falls below a record in flight.
fn in_flight(holding: BankHolding, (shallowest, deepest): (u64, u64)) -> BankStep {
    tracing::info!(?holding, shallowest, deepest, "a shrink waits a boundary: each of its cuts falls below a record in flight");
    BankStep::Deferred
}

impl ServedBank {
    /// The bank as the load serves it, its memory carved from what the
    /// load recorded allocating as it started (`recorded`, of `ledgers`):
    /// the target import allocated the bank whole, so a linked tier's
    /// landing rows (`ExpertBankPlan::landing`) and each holding's floor
    /// are committed of it, and what each holds above its floor is cache of
    /// it, never admitted again. A host tier backing to its load-time slots
    /// after the import (`ExpertBank::back_host`) holds its import's claim
    /// to them (`Recorded::backing`) as a growth: allocated as its slots
    /// back it, cache of what it backed once it ends.
    pub fn new(
        bank: Arc<ExpertBank>,
        programs: &crate::AttestedPrograms,
        tables: Vec<HeldTable>,
        ledgers: &DeviceLedgers,
        recorded: &mut Recorded,
    ) -> Result<Self, DomainError> {
        let moves = programs.expert_bank_moves(&bank.plan).map_err(DomainError::invariant)?;
        let served = Self {
            bank,
            moves,
            tables,
            finished: Mutex::default(),
            _floors: Vec::new(),
        };
        let mut floors = Vec::new();
        if let Some(need) = served.bank.plan.landing().map_err(DomainError::invariant)? {
            let ledger = ledgers.of(served.storage(BankHolding::Host)?);
            floors.push(recorded.carve(&ledger.store(), CommittedKind::Landing, ledger.bytes(&need)));
        }
        let landed = floors.iter().map(Claim::bytes).sum::<u64>();
        for target in served.targets(None)? {
            let floor = (target.charge)(target.shape.floor_slots());
            // A linked tier's floor holds its landing rows, carved above.
            let floor = match target.holding {
                BankHolding::Host => floor.saturating_sub(landed),
                _ => floor,
            };
            let store = ledgers.of(target.tier).store();
            floors.push(recorded.carve(&store, CommittedKind::BankFloor, floor));
            let from = (target.charge)(target.shape.floor_slots());
            let above = (target.charge)(target.at) - from;
            let backing = match target.holding {
                BankHolding::Host => recorded.backing(&store),
                _ => None,
            };
            match backing {
                // Its slots backed before the claim is held here are
                // recorded as it is; each later one as it backs.
                Some(backing) if served.bank.host_backing() => {
                    served.bank.cache.grow(BankHolding::Host, from, backing);
                    served.bank.cache.backed((target.charge)(served.bank.host_slots()));
                }
                Some(backing) => served.bank.cache.hold(BankHolding::Host, backing.into_backed_cache(above, CacheKind::Bank)),
                None => served.bank.cache.hold(target.holding, recorded.carve(&store, CommittedKind::Load, above).into_cache(CacheKind::Bank)),
            }

        }

        Ok(Self { _floors: floors, ..served })
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

    /// Shrink the host tier now, between flights, to the deepest of `cuts`
    /// (its shallowest and deepest host slots) that keeps every record in
    /// flight, decided at `decided` and queued `queued`: the shrink the next
    /// advance reports, `wake` completed at once.
    fn shrink_host(&self, cuts: (u64, u64), decided: Instant, queued: Duration, wake: CompletionWake) -> Result<BankStep, DomainError> {
        let holding = BankHolding::Host;
        let LandedShrink { unpinned, from, to: at, before, after } = match self.bank.shrink_host(cuts)? {
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
            unpinned,
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
    /// beside (the tier's target holds still while it runs, `ServedBank::targets`).
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
        let (before, from, unpinning) = (charged()?, held.table.records(), seismic::backing_times());
        let outcome = held.table.resize(records).map_err(DomainError::invariant)?;
        let (after, to, done) = (charged()?, held.table.records(), Instant::now());
        let unpinned = seismic::backing_times().since(unpinning).unpin;
        let times = ResizeTimes {
            decided,
            queued,
            planned: done - decided - queued,
            trips: Default::default(),
            moves: Duration::ZERO,
            prepared: done,
            installed: done,
            released: done,
            unpinned,
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

    /// Each holding as its target reads it, the experts first, then the
    /// host tier, then each table: within a store they share its room in
    /// that order. The experts' slots and the host tier's host slots hold
    /// at most their ceilings and where the decode's misses stop falling on
    /// `price` (the ceiling before the load is priced; the host tier holds
    /// still while its backing to its planned slots runs); a table's
    /// records where its curve's do.
    fn targets(&self, price: Option<&DecodePrice>) -> Result<Vec<HoldingTarget<'_>>, DomainError> {
        let plan = &self.bank.plan;
        let counts = &self.bank.counts;
        let block = plan.reservation().granule;
        let mut targets = vec![HoldingTarget {
            holding: BankHolding::Experts,
            tier: self.storage(BankHolding::Experts)?,
            at: self.bank.slots(),
            shape: BankShape::new(block, counts.floor(), plan.ceiling, price.and_then(DecodePrice::saturation).unwrap_or(plan.ceiling)),
            charge: Box::new(move |slots| counts.resident_charge(slots)),
        }];
        if let Some((host, ceiling)) = self.bank.host_tier() {
            let block = plan.tier_reservation().expect("invariant violated: a host tier beside a plan without one").granule;
            let bound = match self.bank.host_backing() {
                true => host,
                false => ceiling,
            };
            targets.push(HoldingTarget {
                holding: BankHolding::Host,
                tier: self.storage(BankHolding::Host)?,
                at: host,
                shape: BankShape::new(block, 0, ceiling, bound),
                charge: Box::new(move |slots| counts.host_charge(slots)),
            });
        }
        for (index, held) in self.tables.iter().enumerate() {
            let holding = BankHolding::Table { table: index };
            let reservation = plan.reservation();
            targets.push(HoldingTarget {
                holding,
                tier: self.storage(holding)?,
                at: held.table.records(),
                shape: BankShape::new(reservation.granule, held.plan.floor, held.plan.records.get(), held.table.curve().saturation()),
                charge: Box::new(move |records| {
                    held.plan.charge(reservation, records).expect("invariant violated: a table's plan charges every size it holds")
                }),
            });
        }
        Ok(targets)
    }

    #[cfg(test)]
    fn released_to(&self, bytes: u64) -> Result<Option<u64>, DomainError> {
        self.releasing(BankHolding::Experts, self.bank.slots(), bytes)
    }

    fn live(&self) -> Option<LiveDecode> {
        self.bank.watcher.shared().live()
    }

    fn streamed_layers(&self, lines: Arc<[crate::assessment::ExpertLines]>) -> crate::assessment::StreamedLayers {
        let plan = &self.bank.plan;
        crate::assessment::StreamedLayers::new(plan.layers().iter().map(|layer| layer.experts).collect(), plan.selected(), lines)
    }

    fn grow(&self, ledgers: &DeviceLedgers, holding: BankHolding, to: u64, wake: CompletionWake) -> Result<BankStep, DomainError> {
        if self.growth_waits()? {
            return Ok(BankStep::Running);
        }
        let at = self.size(holding);
        if to <= at {
            return Ok(BankStep::AtLimit);
        }
        // The growth's right to its memory first: none, and claims took the
        // room since the target was derived, so the next follow derives it
        // again.
        let from = self.holding_charge(holding, at)?;
        let bytes = self.holding_charge(holding, to)? - from;
        let Ok(claim) = ledgers.of(self.storage(holding)?).backing(bytes) else {
            return Ok(BankStep::Deferred);
        };
        let step = match holding {
            BankHolding::Experts => self.bank.grow(to, wake),
            BankHolding::Table { table: index } => self.resize_table(index, to, Instant::now(), Duration::ZERO, wake),
            BankHolding::Host => self.bank.grow_host(to, wake),
        }?;
        if let BankStep::Begun(_) = step {
            // Slots backed before the claim is held here are recorded as it
            // is; each later one as it backs (`grow_by_slots`).
            self.bank.cache.grow(holding, from, claim);
            self.bank.cache.backed(self.holding_charge(holding, self.size(holding))?);
        }
        Ok(step)
    }

    fn holding_storage(&self, holding: BankHolding) -> Result<crate::planning::BankTier, DomainError> {
        self.storage(holding)
    }

    /// The records' resize advances at every boundary (a shrink's install
    /// is due at one), whatever else finished: one finished resize is
    /// reported a boundary, the others at the next.
    fn advance(&self, wake: CompletionWake) -> Result<Option<BankResized>, DomainError> {
        let finished = match self.bank.advance(wake)? {
            Some(finished) => Some(finished),
            None => match self.bank.advance_host()? {
                Some((finished, true)) => Some(finished),
                // The host tier's backing at open settles its claim alone.
                Some((backed, false)) => {
                    self.bank.cache.settle(&backed);
                    self.finished.lock().expect("table resize outcome").pop_front()
                }
                None => self.finished.lock().expect("table resize outcome").pop_front(),
            },
        };
        if let Some(resized) = &finished {
            self.bank.cache.settle(resized);
        }
        Ok(finished)
    }

    fn running_begun(&self) -> Option<crate::domain::RunningBegun> {
        self.bank.take_running_begun()
    }

    fn in_gap(&self) -> bool {
        self.bank.watcher.shared().in_gap()
    }

    fn fill_epoch(&self) -> u64 {
        self.bank.watcher.shared().fill_epoch()
    }

    fn wait_tier_filled(&self) -> Result<(), DomainError> {
        self.bank.watcher.shared().wait_filled().map_err(DomainError::invariant)
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
