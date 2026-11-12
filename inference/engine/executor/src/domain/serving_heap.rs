//! The serving heap: a loaded model's expert bank follows its derived
//! target ([`ExecutorDomain::follow_cache_targets`]), each holding's slots
//! the room its store's ledger leaves (`experts::derived`), and the load's
//! serving rates, which a cache holding's worth per byte is read at.

use super::holdings::OptionalHolding;
use super::{BankHolding, BankStep, DomainError, ExecutorDomain, OptionalComponent, ProgramFamily, StateBindings};
use crate::experts::derived::BankMove;
use crate::workload::{WorkloadMeter, WorkloadRates};
use std::time::{Duration, Instant};

/// The span the load's serving rates are read over: each count decays with
/// it (`WorkloadMeter`), so the rates follow the workload as it changes.
const WORKLOAD_SPAN: Duration = Duration::from_secs(60);

/// What following the bank's target did
/// ([`ExecutorDomain::follow_cache_targets`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Followed {
    /// The bank holds its target: nothing to resize.
    Holds,
    /// A claim took idle graph slots' cache in process: their release
    /// began, the bank followed at the next call.
    ReleasingSlots,
    /// A growth toward the target, and how far it got.
    Grew(BankStep),
    /// A shrink to the target, and how far it got.
    Shrank(BankStep),
}

impl<F: ProgramFamily> ExecutorDomain<F> {
    /// Follow the cache holdings' targets. Idle graph slots a claim took
    /// cache of in process are released first, as a claim taking cache
    /// expects its holder to evict it ([`Self::idle_slots_taken`]); then
    /// the bank's derived target: each of its holdings
    /// (`ElasticBank::targets`) derived from its store's ledger now
    /// (`experts::derived::derive`), the
    /// guarantee's in-process term the state the largest batch the load
    /// admits grows by ([`Self::in_process_growth`]), and the one resize
    /// toward them begun. Nothing is stored between calls: every call
    /// derives afresh, so the owner calls it at every reading and flight
    /// boundary, and a resize deferred or running is followed again at the
    /// next.
    pub fn follow_cache_targets(&mut self, bindings: &mut StateBindings, wake: crate::CompletionWake) -> Result<Followed, DomainError> {
        if self.idle_slots_taken() && !self.resources.releasing_slots() {
            // The slots' cache goes with their release, freed as it ends.
            if self.resources.release_idle_slots(std::mem::take(&mut self.idle_class), wake) {
                return Ok(Followed::ReleasingSlots);
            }
            // No slot idles any more: the cache named no memory. The bank
            // follows at the next call, which brings its own wake.
            return Ok(Followed::Holds);
        }

        let Some(bank) = self.family.expert_bank() else {
            return Ok(Followed::Holds);
        };
        match self.next_bank_move(bank)? {
            None => Ok(Followed::Holds),
            Some((holding, BankMove::Shrink { to })) => self.shrink_expert_bank_to(bindings, holding, to, wake).map(Followed::Shrank),
            Some((holding, BankMove::Grow { to })) => self.grow_expert_bank(holding, to, wake).map(Followed::Grew),
        }
    }

    /// The one resize toward `bank`'s derived targets
    /// (`experts::derived::next_move`), at the decode's price now.
    fn next_bank_move(&self, bank: &dyn super::ElasticBank) -> Result<Option<(BankHolding, BankMove)>, DomainError> {
        let price = self.decode_price()?;
        let holdings = bank.targets(price.as_ref())?;
        Ok(crate::experts::derived::next_move(&crate::experts::derived::derive(self.resources.ledgers(), &holdings, self.in_process_growth())))
    }

    /// The state the largest batch the load admits grows by at once: a
    /// prompt group's places, each a launch of rows
    /// (`LoadShape::state_bytes`, the pages the ledger sizes state rows by).
    pub(super) fn in_process_growth(&self) -> u64 {
        crate::experts::derived::in_process_growth(&self.execution, self.resources.most_places())
    }

    /// Begin one growth of `holding` to `to`, its derived target
    /// (`ElasticBank::grow`, which takes the growth's cache first).
    fn grow_expert_bank(&mut self, holding: BankHolding, to: u64, wake: crate::CompletionWake) -> Result<BankStep, DomainError> {
        let started = Instant::now();
        let bank = self.family.expert_bank().expect("invariant violated: a growth of a load without a bank");
        let step = bank.grow(self.resources.ledgers(), holding, to, wake)?;
        if let BankStep::Begun(tier) = step {
            self.resize_held.begin(tier, started);
        }
        Ok(step)
    }

    /// Begin one shrink of `holding` to `to`, its derived target
    /// (`ElasticBank::shrink`), the queued steps orphaned first, as they
    /// may name records it retires; its memory is released at later flight
    /// boundaries ([`Self::advance_expert_bank`]), `wake` completed as it
    /// progresses. The shrink's step records no graph: the decode launch's
    /// were recorded as its slots were made (`NativeGraphPool::grow`). A
    /// load without an elastic bank is at its limit. While a chain is under
    /// way ([`Self::chain_holder`]) nothing begins: its queued chunks are its
    /// holder's to claim, never orphaned, so the shrink is deferred
    /// ([`BankStep::Deferred`]) and its caller decides again once the chain
    /// ends.
    pub(crate) fn shrink_expert_bank_to(
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
                    super::holdings::drafter(bytes.head_weights, drafting, rates.requests_per_s, load)
                }
                _ => OptionalHolding::unpriced(OptionalComponent::Drafter),
            });
        }
        if self.family.vision_is_bound() {
            holdings.push(match (rates, read) {
                (Some(rates), Some(read)) => {
                    let load = read.seconds(bytes.vision_weights);
                    super::holdings::vision(bytes.vision_weights, rates.vision_uses_per_s, load)
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

    /// The marginal state one retained prefix holds, what each cached
    /// prefix's bytes are (`ResourcePlan::retained_entry_bytes`).
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

    /// A test's growth toward the bank's target, at a fresh probe: what the
    /// owner's follow begins where the target lies above the bank.
    #[cfg(test)]
    pub(crate) fn grow_settled(&mut self, wake: crate::CompletionWake) -> Result<BankStep, DomainError> {
        self.observe_memory()?;
        let bank = self.family.expert_bank().expect("a test grows a banked load");
        match self.next_bank_move(bank)? {
            Some((holding, BankMove::Grow { to })) => self.grow_expert_bank(holding, to, wake),
            _ => Ok(BankStep::AtLimit),
        }
    }

    /// A test's shrink of the experts' slots as a fall of their target by
    /// `bytes` makes it: to the shallowest size releasing them, down to the
    /// floor (`ElasticBank::released_to`), the bank answering as a shrink
    /// to its size where it is there: a resize still running is
    /// [`BankStep::Running`], not its destination read as the floor.
    #[cfg(test)]
    pub(crate) fn shrink_expert_bank_by(&mut self, bindings: &mut StateBindings, bytes: u64, wake: crate::CompletionWake) -> Result<BankStep, DomainError> {
        let bank = self.family.expert_bank().expect("a test shrinks a banked load");
        let to = bank.released_to(bytes)?.unwrap_or_else(|| bank.slots());
        self.shrink_expert_bank_to(bindings, BankHolding::Experts, to, wake)
    }
}

#[cfg(test)]
mod tests;
