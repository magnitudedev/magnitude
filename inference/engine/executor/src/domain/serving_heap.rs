//! The serving heap: a loaded model's cache on its stores' ledgers. Its
//! expert bank follows its derived target
//! ([`ExecutorDomain::follow_cache_targets`]), each holding's slots the
//! room its store's ledger leaves (`experts::derived`); every other kind
//! frees what the ledger took of it ([`ExecutorDomain::release_yielded`]).
//! Each kind holds its value on the ledger ([`ExecutorDomain::value_cache`]):
//! the seconds its holding saves per byte, counted over its uses since it
//! was retained and never decayed, the order the ledger takes cache in.

use super::{BankHolding, BankStep, DomainError, ExecutorDomain, OptionalComponent, ProgramFamily, StateBindings};
use crate::experts::derived::BankMove;
use crate::memory::ledger::{CacheKind, Cached, CommittedKind, Value};
use crate::workload::WorkloadRates;
use std::time::Instant;

/// What following the bank's target did
/// ([`ExecutorDomain::follow_cache_targets`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Followed {
    /// The bank holds its target: nothing to resize.
    Holds,
    /// A growth toward the target, and how far it got.
    Grew(BankStep),
    /// A shrink to the target, and how far it got.
    Shrank(BankStep),
}

/// What freeing the cache the ledger took came to
/// ([`ExecutorDomain::release_yielded`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Yielded {
    /// Memory was freed now: of it, `released` bytes of granules the state
    /// stores released, of which the state's pool returned `returned` bytes
    /// to the OS (`ExecutorDomain::yield_state`); both none where only
    /// other kinds were freed.
    Released { released: u64, returned: u64 },
    /// Memory frees off the submitting thread (idle graph slots releasing,
    /// or the expert bank's shrink the owner follows beside this): a later
    /// flight boundary finishes it.
    Pending,
    /// Nothing was taken that this load can free now.
    Nothing,
}

/// An optional component held as cache while no request is resident: its
/// weights' claims, which its loader keeps while it is in use, and its
/// binding's.
pub(super) struct IdleComponent {
    pub(super) component: OptionalComponent,
    weights: Vec<Cached>,
    binding: Vec<Cached>,
}

/// What the load served since it opened, counted and never decayed: the
/// uses each cache holding's value counts since it was retained
/// (`memory::ledger::Value::recompute`). A component's count starts again
/// as it loads again.
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct Served {
    /// Decode steps: the bank's slots above its floor serve each.
    decode_steps: u64,
    /// Requests since the drafter was imported: each would load it again.
    drafter: u64,
    /// Requests with images since the projector was imported.
    vision: u64,
}

impl IdleComponent {
    /// Its bytes on the ledger.
    fn bytes(&self) -> u64 {
        self.weights.iter().chain(&self.binding).map(Cached::bytes).sum()
    }

    /// Whether the ledger took any of it: its holder frees it whole.
    fn taken(&self) -> bool {
        self.weights.iter().chain(&self.binding).any(|cached| cached.releasing() > 0)
    }
}

impl<F: ProgramFamily> ExecutorDomain<F> {
    /// Follow the bank's derived target: each of its holdings
    /// (`ElasticBank::targets`) derived from its store's ledger now
    /// (`experts::derived::derive`), and the one resize toward them begun. Nothing is stored between calls: every call
    /// derives afresh, so the owner calls it at every reading and flight
    /// boundary, and a resize deferred or running is followed again at the
    /// next.
    pub fn follow_cache_targets(&mut self, bindings: &mut StateBindings, wake: crate::CompletionWake) -> Result<Followed, DomainError> {
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
    /// (`experts::derived::next_move`), at the decode's price now: the
    /// targets the domain reports (`Self::derive_bank`).
    fn next_bank_move(&self, bank: &dyn super::ElasticBank) -> Result<Option<(BankHolding, BankMove)>, DomainError> {
        self.derive_bank(bank, crate::experts::derived::next_move)
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
    /// progresses. Where decode graphs are recorded ahead (CUDA's), the
    /// shrink's step records no graph: the decode launch's were recorded as
    /// its slots were made (`NativeGraphPool::grow`); elsewhere it records
    /// its graphs at first use, as any new shape's. A
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
        let step = match self.unchained(bindings) {
            None => BankStep::Deferred,
            Some(mut unchained) => {
                bank.shrink(holding, to, &mut || self.queue.orphan(self.measures.as_ref().and_then(|measures| measures.experts.as_ref()), &mut unchained), wake)?
            }
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

    /// A round served: a decode step counts a use of the bank's slots.
    pub fn served_round(&mut self, decode: bool) {
        if decode {
            self.served.decode_steps += 1;
        }
    }

    /// A request arrived, its input encoded by the vision projector where
    /// `vision`: a use of the drafter, and of the projector with images.
    pub fn served_request(&mut self, vision: bool) {
        self.served.drafter += u64::from(self.head_loader.as_ref().is_some_and(|loader| loader.imported()));
        self.served.vision += u64::from(vision && self.vision_loader.as_ref().is_some_and(|loader| loader.imported()));
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

    /// Value each kind of the load's cache on its stores' ledgers
    /// (`Cached::value`), the seconds a holding saves per byte times its uses
    /// since it was retained (`Value::recompute`), no time decay:
    /// - the bank's slots above its floor: the decode step they save, at
    ///   the decode's price, over the decode steps served (`ElasticBank::value`);
    /// - the retained prefixes: the prefill of the tokens the entry worth
    ///   least saves, at the prefill rate the load's step prices, over its
    ///   hits, `prefix` (its holder's figures: saved tokens and hits);
    /// - an idle drafter or projector: the read of its weights a use would
    ///   pay again, along the model store's read line, over its uses.
    /// A kind whose figures are not priced yet keeps the ledger's value of a
    /// new entry.
    pub fn value_cache(&mut self, prefix: Option<(u64, u64)>) -> Result<(), DomainError> {
        let price = self.decode_price()?;
        if let (Some(bank), Some(price), Some(held), Some(plan)) = (self.family.expert_bank(), price.as_ref(), self.held_slots()?, self.expert_bank_plan()) {
            let floor = usize::try_from(plan.floor()).map_err(|_| DomainError::invariant("bank slots exceed the host"))?;
            let saved = price.step(crate::assessment::HeldSlots { slots: floor.min(held.slots), ..held }).total_seconds - price.step(held).total_seconds;
            // A bank at its floor holds no cache to value.
            if let Some(bytes) = std::num::NonZeroU64::new(self.resources.ledgers().device().estimate().cache_of(CacheKind::Bank)) {
                bank.value(Value::recompute(saved.max(0.0), self.served.decode_steps, bytes));
            }
        }
        if let (Some(rates), Some((tokens, hits)), Some(bytes)) = (self.serving_prior()?, prefix, std::num::NonZeroU64::new(self.retained_entry_bytes())) {
            let seconds = tokens as f64 / rates.prefill_tokens_per_s;
            self.state_slabs.value_prefix(Value::recompute(seconds, hits, bytes));
        }
        if let Some(read) = self.expert_measures().and_then(|measures| measures.read.measured().copied()) {
            let served = self.served;
            for idle in &mut self.idle_components {
                let uses = match idle.component {
                    OptionalComponent::Drafter => served.drafter,
                    OptionalComponent::Vision => served.vision,
                };
                let Some(bytes) = std::num::NonZeroU64::new(idle.bytes()) else { continue };
                let value = Value::recompute(read.seconds(bytes.get()), uses, bytes);
                idle.weights.iter().chain(&idle.binding).for_each(|cached| cached.value(value));
            }
        }
        Ok(())
    }

    /// Free what the ledger took of the load's cache other than the bank's
    /// (which follows its target) and the retained prefixes' (which their
    /// holder evicts, [`Self::prefix_owed`]): the spare slabs' empty slabs,
    /// the idle graph slots off this thread (`wake` completed once they
    /// are), an idle optional component whole. An optional component idles
    /// as cache first where no request is resident. Between flights only:
    /// the queued steps are orphaned where slabs leave the stores, so spare
    /// slabs wait while a chain is under way ([`Self::chain_holder`]).
    pub fn release_yielded(&mut self, bindings: &mut StateBindings, wake: crate::CompletionWake) -> Result<Yielded, DomainError> {
        self.idle_optional();
        let mut freed = false;
        if self.idle_components.iter().any(IdleComponent::taken) {
            let (taken, kept): (Vec<_>, Vec<_>) = std::mem::take(&mut self.idle_components).into_iter().partition(IdleComponent::taken);
            self.idle_components = kept;
            for idle in taken {
                self.unload_component(idle.component, idle);
            }
            freed = true;
        }
        let (mut released, mut returned) = (0, 0);
        if let Some(mut unchained) = self.unchained(bindings).filter(|_| self.state_slabs.spare_owed() > 0) {
            // The stores release the slabs beyond the live requests' next
            // steps, moving no row, and the pool returns what they freed.
            let state = self.yield_state(&mut unchained)?;
            (released, returned) = (state.released, state.returned);
            freed |= released + returned > 0;
        }
        if freed {
            return Ok(Yielded::Released { released, returned });
        }
        if self.resources.releasing_slots() {
            return Ok(Yielded::Pending);
        }
        if self.idle_slots_taken() {
            // The slots' cache goes with their release, freed as it ends;
            // with no slot idling any more, their cache is let go at once.
            return Ok(match self.resources.release_idle_slots(std::mem::take(&mut self.idle_class), wake) {
                true => Yielded::Pending,
                false => Yielded::Released { released: 0, returned: 0 },
            });
        }
        Ok(Yielded::Nothing)
    }

    /// What the ledger took of the retained prefixes' cache that the slabs
    /// still hold: their holder evicts entries, worth least first, while
    /// any is owed, settling the slabs after each ([`Self::settle_state`]).
    pub fn prefix_owed(&self) -> u64 {
        self.state_slabs.prefix_owed()
    }

    /// The device's ledger settled with the slabs the stores hold now: what
    /// an evicted prefix's rows gave back is freed.
    pub fn settle_state(&mut self) {
        self.change_state(|_| ());
    }

    /// Where no request is resident, hold the bound optional components as
    /// cache ([`CacheKind::Drafter`], [`CacheKind::Vision`]): their
    /// weights' and binding's claims turned in place, so the ledger may
    /// take them, worth least first, until a request opens again
    /// ([`Self::activate_components`]).
    pub(super) fn idle_optional(&mut self) {
        if !self.optional_idle() {
            return;
        }
        let cache = |claims: Vec<crate::memory::ledger::Claim>, kind: CacheKind| claims.into_iter().map(|claim| claim.into_cache(kind)).collect::<Vec<_>>();
        if let Some(loader) = self.head_loader.as_ref().filter(|loader| loader.imported()) {
            let weights = cache(loader.take_claims(), CacheKind::Drafter);
            let binding = cache(std::mem::take(&mut self.head_binding), CacheKind::Drafter);
            if !weights.is_empty() || !binding.is_empty() {
                self.idle_components.push(IdleComponent { component: OptionalComponent::Drafter, weights, binding });
            }
        }
        if let Some(loader) = self.vision_loader.as_ref().filter(|loader| loader.imported()) {
            let weights = cache(loader.take_claims(), CacheKind::Vision);
            let binding = cache(std::mem::take(&mut self.vision_binding), CacheKind::Vision);
            if !weights.is_empty() || !binding.is_empty() {
                self.idle_components.push(IdleComponent { component: OptionalComponent::Vision, weights, binding });
            }
        }
    }

    /// As a request opens, commit the idle optional components again
    /// (`Cached::into_claim`), each whole where the ledger took none of it;
    /// one it took any of is unloaded whole, and its next use loads it
    /// again.
    pub(super) fn activate_components(&mut self) {
        for idle in std::mem::take(&mut self.idle_components) {
            let IdleComponent { component, weights, binding } = idle;
            let commit = |cached: Vec<Cached>| {
                cached.into_iter().map(|cached| cached.into_claim(CommittedKind::Component)).collect::<Vec<_>>()
            };
            let (weights, binding) = (commit(weights), commit(binding));
            if weights.iter().chain(&binding).all(Result::is_ok) {
                let claims = |claims: Vec<Result<crate::memory::ledger::Claim, Cached>>| claims.into_iter().flatten().collect::<Vec<_>>();
                let (weights, binding) = (claims(weights), claims(binding));
                match component {
                    OptionalComponent::Drafter => {
                        self.head_loader.as_ref().expect("an idle drafter has its loader").restore_claims(weights);
                        self.head_binding = binding;
                    }
                    OptionalComponent::Vision => {
                        self.vision_loader.as_ref().expect("an idle vision encoder has its loader").restore_claims(weights);
                        self.vision_binding = binding;
                    }
                }
                continue;
            }
            // What was committed again is let go with the rest: the
            // component is unloaded whole.
            self.unload_component(component, (weights, binding));
        }
    }

    /// Unload `component` whole: unbound, its imports released, and `held`,
    /// what the ledger holds of it, let go, as one change. Its uses count
    /// again from its next import.
    pub(super) fn unload_component<H>(&mut self, component: OptionalComponent, held: H) {
        match component {
            OptionalComponent::Drafter => self.served.drafter = 0,
            OptionalComponent::Vision => self.served.vision = 0,
        }
        let changes = self.resources.ledgers().changes().clone();
        changes.change(|_| {
            self.unbind(component);
            match component {
                OptionalComponent::Drafter => self.head_loader.as_ref().map(|loader| loader.unload()),
                OptionalComponent::Vision => self.vision_loader.as_ref().map(|loader| loader.unload()),
            };
            drop(held);
        });
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
        self.observe_memory();
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
