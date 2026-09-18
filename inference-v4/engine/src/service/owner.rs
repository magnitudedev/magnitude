//! Serialized execution ownership. A host event loop calls `step` on admission,
//! output credit, cancellation, and the reserved completion notification path.
use super::policy::{
    order_victims, CapacityEpoch, Limits, Operation, Phase, RequestId, Scheduler, Selection, Victim,
};
use crate::{
    generation::{
        FinishReason, Generation, OutputToken, Proposal, Readiness, WaitReason, WorkKind,
    },
    models::sequence::Advance,
};
use std::{collections::BTreeMap, sync::Arc};

pub struct Work {
    pub request: RequestId,
    pub proposal: Proposal,
    pub mask: Option<Arc<[u32]>>,
    /// Prepare a fresh sequence before this row; failed batch preparation must
    /// undo restoration so it leaves the request evicted.
    pub restore: bool,
}
/// Completion covers forward execution, deferred control transfers, and sampling.
/// A result may be read only after completion, including failed submissions.
pub trait Completion {
    fn is_complete(&self) -> bool;
    fn result(&mut self) -> Result<(), String>;
    /// Register the one reserved host wake after all completion obligations.
    /// Invoke it immediately if already complete. This must not poll or block.
    fn notify(&mut self, wake: super::worker::CompletionWake);
}
pub struct Submitted {
    pub completion: Box<dyn Completion>,
    /// Same order and count as the prepared work; each row accepts independently.
    pub rows: Vec<Box<dyn Advance>>,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PrepareError {
    Request { request: RequestId, message: String },
    Capacity { required: u64, available: u64 },
    Fatal(String),
}
/// Implementations own live numerical sequences and use automatic Seismic
/// selection. Failed preparation must unwind unpublished resources and retain
/// any submitted native uses. Capacity prices are bytes, not tensor geometry.
/// Numerical snapshots never leave the execution owner.
pub struct NumericalCheckpoint<T> {
    pub position: usize,
    pub state: T,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct CheckpointId(u64);
struct Checkpoint<T> {
    generation: Generation,
    numerical: T,
    error: Option<String>,
    protected_until: Option<usize>,
}
pub trait Executor {
    type Checkpoint;
    fn checkpoint(
        &self,
        request: RequestId,
    ) -> Result<NumericalCheckpoint<Self::Checkpoint>, String>;
    /// Install a fork atomically; on error leave no request resources behind.
    fn open_checkpoint(
        &mut self,
        request: RequestId,
        checkpoint: &Self::Checkpoint,
    ) -> Result<(), String>;
    fn open(&mut self, request: RequestId) -> Result<(), String>;
    /// Pending encoder/source work is prefill service, independent of token
    /// advancement. Text-only executors have no such work.
    fn input_pending(&self, _request: RequestId) -> Result<bool, String> {
        Ok(false)
    }
    /// Rows publish one completed input-preparation unit on commit. They must
    /// return no selected token, and abort/drop must discard unpublished output.
    fn prepare_inputs(&mut self, _requests: &[RequestId]) -> Result<Submitted, PrepareError> {
        Err(PrepareError::Fatal(
            "executor does not implement input preparation".into(),
        ))
    }
    fn input_preparation_identity(&self, _requests: &[RequestId]) -> Result<String, String> {
        Err("executor does not implement input preparation identity".into())
    }
    fn prepare(&mut self, work: &[Work]) -> Result<Submitted, PrepareError>;
    /// Opaque identity of the resources/physical capacity classes required by
    /// this work, including readout and state-dependent geometry. Equal keys
    /// under unchanged capacity mean repeating a failed preparation cannot help.
    fn preparation_identity(&self, work: &[Work]) -> Result<String, String>;
    /// Release idle resources and unclaimed temporary capacity. Return the
    /// actual decrease in charged bytes; zero cannot justify a repeated shape.
    fn reclaim_idle(&mut self) -> Result<u64, String>;
    /// Exclusive bytes released by closing this entire set, accounting for aliases.
    fn reclaimable(&self, requests: &[RequestId]) -> Result<u64, String>;
    /// Atomically evict the set. A zero return or error must preserve sequences;
    /// success returns the actual charged-byte decrease and keeps handles reopenable.
    fn evict(&mut self, requests: &[RequestId]) -> Result<u64, String>;
    fn close(&mut self, request: RequestId) -> Result<(), String>;
}
/// Optional typed host-source admission. Layout interpretation is read-only;
/// opening validates prompt/layout correspondence and installs a source atomically.
/// Live device allocation belongs to negotiated preparation, not admission.
pub trait InputExecutor<I>: Executor {
    fn input_layout(
        &self,
        source: &I,
        tokens: &[crate::inputs::TokenId],
    ) -> Result<crate::inputs::InputLayout, String>;
    fn open_input(
        &mut self,
        request: RequestId,
        source: I,
        tokens: &[crate::inputs::TokenId],
        layout: &crate::inputs::InputLayout,
    ) -> Result<(), String>;
}
struct Record {
    generation: Generation,
    waiting_since: u64,
    service_ns: u64,
    error: Option<String>,
    capacity: Option<(CapacityEpoch, u64, u64)>,
    preemption_debt: u32,
    protected_until: Option<usize>,
    retire_on_completion: bool,
}
impl Record {
    fn ready(&self, allowance: usize) -> Result<Readiness, String> {
        if self.generation.is_resident() {
            self.generation.ready(allowance)
        } else {
            self.generation.ready_for_recovery(allowance)
        }
    }
}
struct Flight {
    selection: Selection,
    started: u64,
    completion: Box<dyn Completion>,
    abandoned: Vec<Box<dyn Advance>>,
    input_rows: Vec<(RequestId, Box<dyn Advance>)>,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Status {
    Runnable,
    AwaitingCompletion,
    OutputBlocked,
    Preempted,
    CapacityBlocked { required: u64, available: u64 },
    Terminal(FinishReason),
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Step {
    Submitted,
    Reconciled,
    Waiting,
    Idle,
    Progress,
}
pub struct Owner<E: Executor> {
    checkpoints: BTreeMap<CheckpointId, Checkpoint<E::Checkpoint>>,
    executor: E,
    scheduler: Scheduler,
    records: BTreeMap<RequestId, Record>,
    flight: Option<Flight>,
    epoch: CapacityEpoch,
    next_id: u64,
    now: u64,
    fatal: Option<String>,
}
impl<E: Executor> Owner<E> {
    pub fn new(executor: E, limits: Limits) -> Result<Self, String> {
        Ok(Self {
            checkpoints: BTreeMap::new(),
            executor,
            scheduler: Scheduler::new(limits)?,
            records: BTreeMap::new(),
            flight: None,
            epoch: CapacityEpoch::default(),
            next_id: 1,
            now: 0,
            fatal: None,
        })
    }
    fn time(&mut self, now: u64) -> Result<(), String> {
        if now < self.now {
            return Err("service clock moved backwards".into());
        }
        self.now = now;
        Ok(())
    }
    pub fn admit(&mut self, generation: Generation, now: u64) -> Result<RequestId, String> {
        self.admit_with(generation, now, |executor, id, _| executor.open(id))
    }
    pub fn input_layout<I>(
        &self,
        source: &I,
        tokens: &[crate::inputs::TokenId],
    ) -> Result<crate::inputs::InputLayout, String>
    where
        E: InputExecutor<I>,
    {
        self.executor.input_layout(source, tokens)
    }
    pub fn admit_input<I>(
        &mut self,
        generation: Generation,
        source: I,
        now: u64,
    ) -> Result<RequestId, String>
    where
        E: InputExecutor<I>,
    {
        self.admit_with(generation, now, |executor, id, generation| {
            executor.open_input(id, source, generation.prompt(), generation.layout())
        })
    }
    fn admit_with(
        &mut self,
        generation: Generation,
        now: u64,
        open: impl FnOnce(&mut E, RequestId, &Generation) -> Result<(), String>,
    ) -> Result<RequestId, String> {
        self.time(now)?;
        if let Some(error) = &self.fatal {
            return Err(error.clone());
        }
        if self.records.len() >= self.scheduler.limits().max_requests {
            return Err("service request limit reached".into());
        }
        if generation.awaiting_completion() {
            return Err("cannot admit an already submitted generation".into());
        }
        if generation.processed() != 0
            || generation.usage().completion_tokens != 0
            || !generation.is_resident()
        {
            return Err("continued generation requires checkpoint admission".into());
        }
        let id = RequestId(self.next_id);
        self.next_id = self
            .next_id
            .checked_add(1)
            .ok_or("service identity exhausted")?;
        open(&mut self.executor, id, &generation)?;
        self.records.insert(
            id,
            Record {
                generation,
                waiting_since: now,
                service_ns: 0,
                error: None,
                capacity: None,
                preemption_debt: 0,
                protected_until: None,
                retire_on_completion: false,
            },
        );
        self.epoch.advance()?;
        Ok(id)
    }
    pub fn checkpoint(&mut self, request: RequestId) -> Result<CheckpointId, String> {
        if let Some(error) = &self.fatal {
            return Err(error.clone());
        }
        if self.checkpoints.len() >= self.scheduler.limits().max_requests {
            return Err("service checkpoint limit reached".into());
        }
        let record = self.records.get(&request).ok_or("unknown request")?;
        if record.retire_on_completion {
            return Err("request is being released".into());
        }
        if self
            .flight
            .as_ref()
            .is_some_and(|flight| flight.selection.requests().contains(&request))
        {
            return Err("cannot checkpoint submitted input or numerical work".into());
        }
        let numerical = self.executor.checkpoint(request)?;
        let generation = record.generation.fork_at(numerical.position)?;
        static NEXT_CHECKPOINT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
        let id = CheckpointId(
            NEXT_CHECKPOINT
                .fetch_update(
                    std::sync::atomic::Ordering::Relaxed,
                    std::sync::atomic::Ordering::Relaxed,
                    |id| id.checked_add(1),
                )
                .map_err(|_| "checkpoint identity exhausted")?,
        );
        self.checkpoints.insert(
            id,
            Checkpoint {
                generation,
                numerical: numerical.state,
                error: record.error.clone(),
                protected_until: record.protected_until,
            },
        );
        Ok(id)
    }
    pub fn fork_checkpoint(
        &mut self,
        checkpoint: CheckpointId,
        now: u64,
    ) -> Result<RequestId, String> {
        self.time(now)?;
        if let Some(error) = &self.fatal {
            return Err(error.clone());
        }
        if self.records.len() >= self.scheduler.limits().max_requests {
            return Err("service request limit reached".into());
        }
        let checkpoint = self
            .checkpoints
            .get(&checkpoint)
            .ok_or("unknown checkpoint")?;
        let generation = checkpoint
            .generation
            .fork_at(checkpoint.generation.processed())?;
        let id = RequestId(self.next_id);
        self.next_id = self
            .next_id
            .checked_add(1)
            .ok_or("service identity exhausted")?;
        self.executor.open_checkpoint(id, &checkpoint.numerical)?;
        self.records.insert(
            id,
            Record {
                generation,
                waiting_since: now,
                service_ns: 0,
                error: checkpoint.error.clone(),
                capacity: None,
                preemption_debt: 0,
                protected_until: checkpoint.protected_until,
                retire_on_completion: false,
            },
        );
        self.epoch.advance()?;
        Ok(id)
    }
    pub fn release_checkpoint(&mut self, checkpoint: CheckpointId) -> Result<(), String> {
        if self.checkpoints.remove(&checkpoint).is_some() {
            self.epoch.advance()?;
        }
        Ok(())
    }
    pub fn error(&self, id: RequestId) -> Option<&str> {
        self.records.get(&id).and_then(|r| r.error.as_deref())
    }
    pub fn output_len(&self, id: RequestId) -> Result<usize, String> {
        Ok(self
            .records
            .get(&id)
            .ok_or("unknown request")?
            .generation
            .output_len())
    }
    pub fn usage(&self, id: RequestId) -> Result<crate::generation::Usage, String> {
        Ok(self
            .records
            .get(&id)
            .ok_or("unknown request")?
            .generation
            .usage())
    }
    pub fn fatal_error(&self) -> Option<&str> {
        self.fatal.as_deref()
    }
    pub fn completed_service_ns(&self) -> u128 {
        self.scheduler.completed_service_ns()
    }
    pub fn status(&self, id: RequestId) -> Result<Status, String> {
        let r = self.records.get(&id).ok_or("unknown request")?;
        if r.generation.awaiting_completion()
            || self
                .flight
                .as_ref()
                .is_some_and(|flight| flight.selection.requests().contains(&id))
        {
            return Ok(Status::AwaitingCompletion);
        }
        if let Some(reason) = r.generation.finish_reason() {
            return Ok(Status::Terminal(reason));
        }
        if let Some((epoch, required, available)) = r.capacity {
            if !self.epoch.changed_since(epoch) {
                return Ok(Status::CapacityBlocked {
                    required,
                    available,
                });
            }
        }
        Ok(match r.generation.ready(1)? {
            Readiness::Ready(_) => Status::Runnable,
            Readiness::Wait(WaitReason::Output) => Status::OutputBlocked,
            Readiness::Wait(WaitReason::Residency) => Status::Preempted,
            _ => return Err("inconsistent generation status".into()),
        })
    }
    pub fn take(&mut self, id: RequestId, count: usize) -> Result<Vec<OutputToken>, String> {
        let output = self
            .records
            .get_mut(&id)
            .ok_or("unknown request")?
            .generation
            .take(count)?;
        if !output.is_empty() {
            self.epoch.advance()?;
        }
        Ok(output)
    }
    pub fn cancel(&mut self, id: RequestId, discard_output: bool) -> Result<(), String> {
        let record = self.records.get_mut(&id).ok_or("unknown request")?;
        let changed = record.generation.finish_reason().is_none()
            || (discard_output && record.generation.output_len() > 0);
        record.generation.cancel();
        if discard_output {
            record.generation.discard_output();
        }
        if changed {
            self.epoch.advance()?;
        }
        Ok(())
    }
    /// Terminal output must be drained or explicitly discarded before retirement.
    pub fn retire(&mut self, id: RequestId) -> Result<(), String> {
        if self
            .flight
            .as_ref()
            .is_some_and(|flight| flight.selection.requests().contains(&id))
        {
            return Err("retirement must wait for shared batch reconciliation".into());
        }
        let record = self.records.get(&id).ok_or("unknown request")?;
        if record.generation.awaiting_completion()
            || record.generation.finish_reason().is_none()
            || record.generation.output_len() != 0
        {
            return Err("retirement requires terminal reconciled work and drained output".into());
        }
        self.executor.close(id)?;
        self.records.remove(&id);
        self.epoch.advance()
    }
    /// Caller disconnect: discard output and close after shared completion.
    /// Repeated release is harmless, including a late preparation cleanup.
    pub fn release(&mut self, id: RequestId) -> Result<(), String> {
        if !self.records.contains_key(&id) {
            return Ok(());
        }
        self.cancel(id, true)?;
        if self
            .flight
            .as_ref()
            .is_some_and(|flight| flight.selection.requests().contains(&id))
        {
            self.records.get_mut(&id).unwrap().retire_on_completion = true;
            Ok(())
        } else {
            self.retire(id)
        }
    }
    fn fail_all(&mut self, error: String) {
        self.fatal.get_or_insert_with(|| error.clone());
        for record in self.records.values_mut() {
            if record.generation.finish_reason().is_none()
                || record.generation.awaiting_completion()
            {
                record.generation.fail();
                record.error.get_or_insert_with(|| error.clone());
            }
        }
    }
    fn fail_request(&mut self, id: RequestId, error: String) {
        if let Some(record) = self.records.get_mut(&id) {
            record.generation.fail();
            record.error = Some(error);
            record.capacity = None;
        }
    }
    pub fn step(&mut self, now: u64) -> Result<Step, String> {
        self.time(now)?;
        if let Some(flight) = &self.flight {
            if !flight.completion.is_complete() {
                return Ok(Step::Waiting);
            }
            let rows_ready = flight.selection.requests().iter().all(|id| {
                self.records.get(id).is_some_and(|r| {
                    !r.generation.awaiting_completion() || r.generation.completion_ready()
                })
            }) && flight.abandoned.iter().all(|row| row.is_complete())
                && flight.input_rows.iter().all(|(_, row)| row.is_complete());
            if !rows_ready {
                self.fail_all("batch completed before all numerical rows completed".into());
                return Ok(Step::Waiting);
            }
            let mut flight = self.flight.take().unwrap();
            if let Err(error) = flight.completion.result() {
                self.fail_all(error);
            }
            for (id, mut row) in flight.input_rows.drain(..) {
                if self.records[&id].generation.finish_reason().is_some() {
                    continue;
                }
                let result = row.selected().and_then(|token| {
                    if token.is_some() {
                        return Err("input preparation returned a generated token".into());
                    }
                    row.commit()
                });
                if let Err(error) = result {
                    self.fail_request(id, error);
                }
            }
            let elapsed = now - flight.started;
            for id in flight.selection.requests() {
                let record = self.records.get_mut(id).unwrap();
                if record.generation.awaiting_completion() {
                    if let Err(error) = record.generation.reconcile() {
                        record.error = Some(error);
                    }
                }
                // Attribution is shared batch service, not individual device time.
                record.service_ns = record.service_ns.saturating_add(elapsed);
                record.waiting_since = now;
                if record
                    .protected_until
                    .is_some_and(|boundary| record.generation.processed() > boundary)
                {
                    record.protected_until = None;
                }
            }
            self.scheduler.completed(flight.selection, elapsed);
            self.epoch.advance()?;
            let released: Vec<_> = self
                .records
                .iter()
                .filter_map(|(&id, record)| record.retire_on_completion.then_some(id))
                .collect();
            for id in released {
                self.retire(id)?;
            }
            return Ok(Step::Reconciled);
        }
        if self.fatal.is_some() {
            return Ok(Step::Idle);
        }
        let mut candidates = Vec::new();
        let mut input_pending = std::collections::BTreeSet::new();
        for (&id, record) in &mut self.records {
            if record
                .capacity
                .is_some_and(|(epoch, _, _)| !self.epoch.changed_since(epoch))
            {
                continue;
            }
            record.capacity = None;
            match record.ready(1) {
                Ok(Readiness::Ready(proposal)) => {
                    let pending = match self.executor.input_pending(id) {
                        Ok(pending) => pending,
                        Err(error) => {
                            record.generation.fail();
                            record.error = Some(error);
                            continue;
                        }
                    };
                    if pending {
                        input_pending.insert(id);
                    }
                    candidates.push(Operation {
                        identity: id,
                        phase: if !pending && proposal.kind() == WorkKind::Decode {
                            Phase::Decode
                        } else {
                            Phase::Prefill
                        },
                        active: record.service_ns > 0,
                        resident: record.generation.is_resident(),
                        waiting_since_ns: record.waiting_since,
                        service_ns: record.service_ns,
                        preemption_debt: record.preemption_debt,
                    });
                }
                Ok(Readiness::Wait(_)) => {}
                Err(error) => {
                    record.generation.fail();
                    record.error = Some(error);
                }
            }
        }
        let Some(mut selection) = self.scheduler.select(&candidates, now)? else {
            return Ok(Step::Idle);
        };
        // Batch only a homogeneous preparation class, preserving scheduler
        // priority. Trailing members remain eligible for the next selection.
        let preparing_input = input_pending.contains(&selection.requests()[0]);
        let members = selection
            .requests()
            .iter()
            .take_while(|id| input_pending.contains(id) == preparing_input)
            .count();
        selection.truncate(members)?;
        let allowance = match selection.phase() {
            Phase::Prefill => self.scheduler.limits().prefill_tokens,
            Phase::Decode => self.scheduler.limits().decode_tokens,
        };
        let mut work = Vec::new();
        for &id in selection.requests() {
            let record = &self.records[&id];
            let prepared = (|| {
                let Readiness::Ready(proposal) = record.ready(allowance)? else {
                    return Err("selected generation is no longer ready".into());
                };
                let mask = if !preparing_input && proposal.needs_sample() {
                    record.generation.selection_mask()?
                } else {
                    None
                };
                Ok(Work {
                    request: id,
                    proposal,
                    mask,
                    restore: !record.generation.is_resident(),
                })
            })();
            match prepared {
                Ok(row) => work.push(row),
                Err(error) => {
                    self.fail_request(id, error);
                    self.epoch.advance()?;
                    return Ok(Step::Progress);
                }
            }
        }
        match self.prepare_with_capacity(&mut selection, &mut work, allowance, preparing_input) {
            Ok(submitted) => {
                let mut flight = Flight {
                    selection,
                    started: now,
                    completion: submitted.completion,
                    abandoned: Vec::new(),
                    input_rows: Vec::new(),
                };
                if submitted.rows.len() != work.len() {
                    flight.abandoned = submitted.rows;
                    self.fail_all("executor returned the wrong number of numerical rows".into());
                } else if preparing_input {
                    flight.input_rows = work
                        .into_iter()
                        .zip(submitted.rows)
                        .map(|(work, row)| (work.request, row))
                        .collect();
                } else {
                    for (work, row) in work.into_iter().zip(submitted.rows) {
                        if work.restore {
                            if let Err(error) = self
                                .records
                                .get_mut(&work.request)
                                .unwrap()
                                .generation
                                .restored()
                            {
                                self.fail_all(error);
                            }
                        }
                        if let Err(error) = self
                            .records
                            .get_mut(&work.request)
                            .unwrap()
                            .generation
                            .attach(work.proposal, row)
                        {
                            self.fail_all(error);
                        }
                    }
                }
                self.flight = Some(flight);
                Ok(Step::Submitted)
            }
            Err(PrepareError::Request { request, message }) => {
                if !selection.requests().contains(&request) {
                    self.fail_all("executor reported an error for an unselected request".into());
                } else {
                    self.fail_request(request, message);
                }
                self.epoch.advance()?;
                Ok(Step::Progress)
            }
            Err(PrepareError::Fatal(error)) => {
                self.fail_all(error);
                self.epoch.advance()?;
                Ok(Step::Progress)
            }
            Err(PrepareError::Capacity {
                required,
                available,
            }) => {
                let can_change = self.records.iter().any(|(id, record)| {
                    !selection.requests().contains(id)
                        && (record.generation.finish_reason().is_some()
                            || record.generation.output_len() > 0
                            || record.capacity.is_none())
                });
                if can_change {
                    for id in selection.requests() {
                        self.records.get_mut(id).unwrap().capacity =
                            Some((self.epoch, required, available));
                    }
                    Ok(Step::Waiting)
                } else {
                    for &id in selection.requests() {
                        self.fail_request(id,format!("insufficient capacity: required {required} bytes, available {available} bytes"));
                    }
                    self.epoch.advance()?;
                    Ok(Step::Progress)
                }
            }
        }
    }

    fn prepare_with_capacity(
        &mut self,
        selection: &mut Selection,
        work: &mut Vec<Work>,
        mut allowance: usize,
        preparing_input: bool,
    ) -> Result<Submitted, PrepareError> {
        let mut reclaimed = false;
        let mut failed = BTreeMap::<String, PrepareError>::new();
        loop {
            let requests: Vec<_> = work.iter().map(|row| row.request).collect();
            let identity = if preparing_input {
                self.executor.input_preparation_identity(&requests)
            } else {
                self.executor.preparation_identity(work)
            }
            .map_err(PrepareError::Fatal)?;
            if identity.is_empty() {
                return Err(PrepareError::Fatal(
                    "executor returned an empty preparation identity".into(),
                ));
            }
            let result = if let Some(error) = failed.get(&identity) {
                Err(error.clone())
            } else {
                if preparing_input {
                    self.executor.prepare_inputs(&requests)
                } else {
                    self.executor.prepare(work)
                }
            };
            let capacity = match result {
                Err(error @ PrepareError::Capacity { .. }) => error,
                result => return result,
            };
            failed.insert(identity, capacity.clone());
            if !reclaimed {
                reclaimed = true;
                let released = self.executor.reclaim_idle().map_err(PrepareError::Fatal)?;
                if released > 0 {
                    failed.clear();
                    self.epoch.advance().map_err(PrepareError::Fatal)?;
                    continue;
                }
            }
            if work.len() > 1 {
                work.pop();
                selection
                    .truncate(work.len())
                    .map_err(PrepareError::Fatal)?;
                continue;
            }
            let previous = work[0].proposal.tokens().len();
            let request = work[0].request;
            let mut smaller = None;
            while !preparing_input && allowance > 1 {
                allowance = (allowance / 2).max(1);
                let record = &self.records[&request];
                let generation = &record.generation;
                let readiness = record
                    .ready(allowance)
                    .map_err(|message| PrepareError::Request { request, message })?;
                let Readiness::Ready(proposal) = readiness else {
                    return Err(PrepareError::Fatal(
                        "capacity negotiation lost runnable work".into(),
                    ));
                };
                // An indivisible span may ignore a smaller soft allowance. Do
                // not submit the same physical row geometry again in that case.
                if proposal.tokens().len() >= previous {
                    continue;
                }
                let mask = if proposal.needs_sample() {
                    generation
                        .selection_mask()
                        .map_err(|message| PrepareError::Request { request, message })?
                } else {
                    None
                };
                smaller = Some(Work {
                    request,
                    proposal,
                    mask,
                    restore: !generation.is_resident(),
                });
                break;
            }
            if let Some(smaller) = smaller {
                work[0] = smaller;
                continue;
            }
            if self.evict_victims(selection.requests())? {
                failed.clear();
                continue;
            }
            return Err(capacity);
        }
    }
    fn evict_victims(&mut self, selected: &[RequestId]) -> Result<bool, PrepareError> {
        let mut victims = Vec::new();
        for (&id, record) in &self.records {
            if selected.contains(&id)
                || !record.generation.is_resident()
                || record.generation.awaiting_completion()
                || record.generation.finish_reason().is_some()
                || record.protected_until.is_some()
            {
                continue;
            }
            victims.push(Victim {
                identity: id,
                output_blocked: matches!(
                    record.generation.ready(1),
                    Ok(Readiness::Wait(WaitReason::Output))
                ),
                preemption_debt: record.preemption_debt,
                exclusive_bytes: self
                    .executor
                    .reclaimable(&[id])
                    .map_err(PrepareError::Fatal)?,
                replay_tokens: record.generation.accepted_position() as u64,
                service_ns: record.service_ns,
            });
        }
        order_victims(&mut victims);
        let mut set = Vec::new();
        for victim in victims {
            set.push(victim.identity);
            if self
                .executor
                .reclaimable(&set)
                .map_err(PrepareError::Fatal)?
                == 0
            {
                continue;
            }
            let released = self.executor.evict(&set).map_err(PrepareError::Fatal)?;
            if released == 0 {
                continue;
            }
            for id in set {
                let record = self.records.get_mut(&id).unwrap();
                record.protected_until = Some(record.generation.accepted_position());
                record.generation.evicted().map_err(PrepareError::Fatal)?;
                record.preemption_debt = record.preemption_debt.saturating_add(1);
                record.capacity = None;
            }
            self.epoch.advance().map_err(PrepareError::Fatal)?;
            return Ok(true);
        }
        Ok(false)
    }
}

impl<E: Executor> super::worker::Driven for Owner<E> {
    fn failure(&self) -> Option<&str> {
        self.fatal_error()
    }
    fn advance(
        &mut self,
        now: u64,
        wake: super::worker::CompletionWake,
    ) -> Result<super::worker::Drive, String> {
        use super::worker::Drive;
        match self.step(now)? {
            Step::Submitted => {
                self.flight
                    .as_mut()
                    .expect("submitted flight")
                    .completion
                    .notify(wake);
                Ok(Drive::AwaitingCompletion)
            }
            Step::Reconciled | Step::Progress => Ok(Drive::Progress),
            Step::Idle => Ok(Drive::Idle),
            Step::Waiting if self.flight.is_none() => Ok(Drive::Idle),
            Step::Waiting => Err("completion notification preceded numerical completion".into()),
        }
    }
    fn failed(&mut self, error: &str) {
        self.fail_all(error.into());
    }
    fn shutdown(&mut self) -> Result<bool, String> {
        self.checkpoints.clear();
        let ids: Vec<_> = self.records.keys().copied().collect();
        for &id in &ids {
            self.cancel(id, true)?;
        }
        if self.flight.is_some() {
            return Ok(false);
        }
        for id in ids {
            self.retire(id)?;
        }
        Ok(true)
    }
}
