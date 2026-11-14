use super::{ExpertBank, HeldTable};
use crate::assessment::{DecodePrice, LiveDecode};
use crate::planning::BankTier;
use crate::domain::{BankHolding, BankResize, BankResized, BankStep, DomainError, ElasticBank, ExpertResidency, ResizeTimes, TableResidency};
use crate::experts::derived::{BankShape, HoldingTarget};
use crate::memory::ledger::{CacheKind, Claim, CommittedKind, DeviceLedgers, Recorded};
use crate::CompletionWake;
use std::time::{Duration, Instant};
use std::sync::{Arc, Mutex};

/// A load's expert bank as the domain resizes it (`ElasticBank`): the bank
/// and the stored tables it holds (`ExpertBankPlan::tables`), the one owner
/// of the memory of all of them.
pub(crate) struct ServedBank {
    bank: Arc<ExpertBank>,
    tables: Vec<HeldTable>,
    /// The resizes of tables done when asked, each until an advance
    /// reports it, in the order they were done.
    finished: Mutex<std::collections::VecDeque<BankResized>>,
    /// Its committed claims, held for the load: a linked tier's landing
    /// rows and each holding's floor.
    _floors: Vec<Claim>,
}

impl ServedBank {
    /// The bank as the load serves it, its memory carved from what the
    /// load recorded allocating as it started (`recorded`, of `ledgers`):
    /// the target import allocated the bank whole, so a linked tier's
    /// landing rows (`ExpertBankPlan::landing`) and each holding's floor
    /// are committed of it, and what each holds above its floor is cache of
    /// it, never admitted again. A host tier backing to its load-time slots
    /// from the bank's open (`ExpertBank::back_host`) holds its import's
    /// claim to them as a growth from then (`ExpertBank::open`): allocated
    /// as its slots back it, cache of what it backed once it ends.
    pub fn new(
        bank: Arc<ExpertBank>,
        tables: Vec<HeldTable>,
        ledgers: &DeviceLedgers,
        recorded: &mut Recorded,
    ) -> Result<Self, DomainError> {
        bank.cache.served_on(ledgers, bank.plan.host_slots);
        let served = Self {
            bank,
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
            // A host tier's slots are allocated in its claim as each backs,
            // from the bank's open (`BankCache::opened`), until an advance
            // settles it.
            match target.holding {
                BankHolding::Host if served.bank.plan.host_slots > 0 => {}
                holding => served.bank.cache.hold(holding, recorded.carve(&store, CommittedKind::Backing(crate::memory::ledger::PoolClass::Records), above).into_backed_cache(CacheKind::Bank)),
            }
        }

        Ok(Self { _floors: floors, ..served })
    }

    /// The pool `holding` lies in (`ExpertBankPlan::holding_storage`).
    fn storage(&self, holding: BankHolding) -> Result<BankTier, DomainError> {
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
    /// records), a granule of its pool a record.
    fn holding_charge(&self, holding: BankHolding, at: u64) -> Result<u64, DomainError> {
        let counts = &self.bank.counts;
        match holding {
            BankHolding::Experts => Ok(counts.resident_charge(at)),
            BankHolding::Host => Ok(counts.host_charge(at)),
            BankHolding::Table { .. } => crate::planning::BankTable::charge(self.bank.plan.block(), self.bank.plan.granule(), at).map_err(DomainError::invariant),
        }
    }

    /// The shallowest shrink of `holding` from `from` releasing `bytes`
    /// ([`crate::planning::releasing`]), down to its floor.
    fn releasing(&self, holding: BankHolding, from: u64, bytes: u64) -> Result<Option<u64>, DomainError> {
        crate::planning::releasing(from, self.floor(holding), bytes, |at| self.holding_charge(holding, at))
    }

    /// `holding`'s floor, in slots or records.
    fn floor(&self, holding: BankHolding) -> u64 {
        match holding {
            BankHolding::Experts => self.bank.floor(BankTier::Bank),
            BankHolding::Host => self.bank.floor(BankTier::Host),
            BankHolding::Table { table } => self.tables[table].plan.floor,
        }
    }

    /// The bytes `holding` holds above its floor at `at` slots or records:
    /// what its cache holds at most ([`super::cache::BankCache::fit`]).
    fn above_floor(&self, holding: BankHolding, at: u64) -> Result<u64, DomainError> {
        Ok(self.holding_charge(holding, at)?.saturating_sub(self.holding_charge(holding, self.floor(holding))?))
    }

    /// Whether a resize of the holdings in `tier` storage runs: of the
    /// bank's store, the experts' (until its advance finishes it) or a
    /// table's (until an advance reports it); of host RAM, the host tier's
    /// growth, its backing to its planned slots ([`ExpertBank::back_host`]),
    /// or its shrink until an advance reports it. Each store resizes on its
    /// own.
    fn running_in(&self, tier: BankTier) -> Result<bool, DomainError> {
        let finished = self
            .finished
            .lock()
            .expect("table resize outcome")
            .iter()
            .map(|resized| self.storage(resized.holding()))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(finished.contains(&tier)
            || match tier {
                BankTier::Bank => self.bank.resizing(),
                BankTier::Host => self.bank.host_growing() || self.bank.host_backing(),
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
        let charged = || held.table.charged_bytes();
        let (before, from) = (charged(), held.table.records());
        // The resize is one change: the pool backs or returns its slots'
        // granules, a growth's held by the claim its caller holds for it, and
        // a shrink's cache holds no more than the slots left.
        let holding = BankHolding::Table { table: index };
        let tier = self.storage(holding)?;
        let (outcome, after, returned) = self.bank.cache.changes().clone().change(|_| {
            let outcome = held.table.resize(records).map_err(DomainError::invariant)?;
            let after = charged();
            let mut returned = 0;
            if after < before {
                self.bank.cache.fit(holding, self.above_floor(holding, held.table.records())?);
                returned = self.bank.return_freed(tier, before - after)?;
            }
            Ok::<_, DomainError>((outcome, after, returned))
        })?;
        let (to, done) = (held.table.records(), Instant::now());
        let times = ResizeTimes {
            decided,
            queued,
            planned: (done - decided).saturating_sub(queued),
            trips: Default::default(),
            prepared: done,
            installed: done,
            released: done,
        };
        let resized = match outcome {
            // A shrink needs no memory: its refusal breaks an invariant, as the bank's does.
            Err(refusal) if records < from => {
                return Err(DomainError::invariant(format!("a held table's storage refused a shrink: {refusal}")));
            }
            Err(refusal) => BankResized::GrowthRefused { holding, at: to, refusal },
            Ok(()) if after < before => BankResized::Shrunk { holding, from, to, released: before - after, returned, times },
            Ok(()) => BankResized::Grown { holding, from, to, grown: after - before, refused: None, times },
        };
        self.finished.lock().expect("table resize outcome").push_back(resized);
        wake.complete();
        Ok(BankStep::Begun(tier))
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

    fn value(&self, value: crate::memory::ledger::Value) {
        self.bank.cache.value(value);
    }

    fn stop_growths(&self) {
        // Both stores: a claim's hold would leave either one's slots waiting.
        self.bank.stop_growth(BankTier::Bank);
        self.bank.stop_growth(BankTier::Host);
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
        // The bank's records shrink beside running steps, on every
        // placement: the watcher evicts none a step in flight writes or
        // reads in place, nothing moves, and their granules free behind the
        // last flight that read them. A table's resize runs between
        // flights: the queue drained, a step still running defers it.
        let step = match holding {
            BankHolding::Experts | BankHolding::Host => self.bank.shrink(kind, (shallowest, deepest), decided, Duration::ZERO)?,
            BankHolding::Table { table: index } => {
                between_flights()?;
                if !self.in_gap() {
                    return Ok(BankStep::Deferred);
                }
                return self.resize_table(index, deepest, decided, decided.elapsed(), wake);
            }
        };
        // Done at once: the next advance reports it.
        if let BankStep::Begun(_) = step {
            wake.complete();
        }
        Ok(step)
    }

    /// A shrink moves nothing, so no flight reads what a shrink's device
    /// work wrote: nothing to observe.
    fn carried(&self) -> Result<(), DomainError> {
        Ok(())
    }

    /// Each holding as its target reads it, the experts first, then the
    /// host tier, then each table: within a store they share its room in
    /// that order. The experts' slots hold at most their ceiling and where
    /// the decode's misses stop falling on `price` (the ceiling before the
    /// load is priced), a table's records where its curve's do, and the
    /// host tier's host slots its ceiling (held still while its backing to
    /// its planned slots runs). A curve bounds a growth, never what a
    /// holding holds ([`BankShape::curved`]).
    fn targets(&self, price: Option<&DecodePrice>) -> Result<Vec<HoldingTarget<'_>>, DomainError> {
        let plan = &self.bank.plan;
        let counts = &self.bank.counts;
        let block = plan.granule();
        let at = self.bank.slots();
        let mut targets = vec![HoldingTarget {
            holding: BankHolding::Experts,
            tier: self.storage(BankHolding::Experts)?,
            at,
            owed: self.bank.cache.owed(BankHolding::Experts, self.above_floor(BankHolding::Experts, at)?),
            shape: BankShape::curved(block, counts.floor(), plan.ceiling, price.and_then(DecodePrice::saturation), at),
            charge: Box::new(move |slots| counts.resident_charge(slots)),
        }];
        if let Some((host, ceiling)) = self.bank.host_tier() {
            let block = plan.host_granule().expect("invariant violated: a host tier beside a plan without its pool");
            let bound = match self.bank.host_backing() {
                true => host,
                false => ceiling,
            };
            targets.push(HoldingTarget {
                holding: BankHolding::Host,
                tier: self.storage(BankHolding::Host)?,
                at: host,
                owed: self.bank.cache.owed(BankHolding::Host, self.above_floor(BankHolding::Host, host)?),
                shape: BankShape::host(block, ceiling, bound, plan.whole_at(self.bank.slots())),
                charge: Box::new(move |slots| counts.host_charge(slots)),
            });
        }
        for (index, held) in self.tables.iter().enumerate() {
            let holding = BankHolding::Table { table: index };
            let (block, granule) = (plan.block(), plan.granule());
            let records = held.table.records();
            targets.push(HoldingTarget {
                holding,
                tier: self.storage(holding)?,
                at: records,
                owed: self.bank.cache.owed(holding, self.above_floor(holding, records)?),
                shape: BankShape::curved(granule, held.plan.floor, held.plan.records.get(), Some(held.table.curve().saturation()), records),
                charge: Box::new(move |records| {
                    crate::planning::BankTable::charge(block, granule, records).expect("invariant violated: a table's charge fits u64 at every size it holds")
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
        let Ok(claim) = ledgers.of(self.storage(holding)?).backing(crate::memory::ledger::PoolClass::Records, bytes) else {
            return Ok(BankStep::Deferred);
        };
        // Held before the growth begins: each slot's backing is charged to
        // it as one change with the slot (`grow_by_slots`), a table's with
        // its resize ([`Self::resize_table`]).
        self.bank.cache.grow(holding, claim);
        let step = match holding {
            BankHolding::Experts => self.bank.grow(to, wake),
            BankHolding::Table { table: index } => self.resize_table(index, to, Instant::now(), Duration::ZERO, wake),
            BankHolding::Host => self.bank.grow_host(to, wake),
        };
        if !matches!(step, Ok(BankStep::Begun(_))) {
            self.bank.cache.abandon(holding);
        }
        step
    }

    fn holding_storage(&self, holding: BankHolding) -> Result<BankTier, DomainError> {
        self.storage(holding)
    }

    /// The records' resize advances at every boundary (a shrink's release
    /// is due at one), whatever else finished: one finished resize is
    /// reported a boundary, the others at the next.
    fn advance(&self, _wake: CompletionWake) -> Result<Option<BankResized>, DomainError> {
        let finished = match self.bank.advance()? {
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

    fn cover(&self) -> crate::ExpertCover {
        self.bank.watcher.shared().cover()
    }

    fn wait_covered(&self) -> Result<(), DomainError> {
        self.bank.watcher.shared().wait_covered().map_err(DomainError::invariant)
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
                    bytes: held.table.charged_bytes(),
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
