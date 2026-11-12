//! Target submission, completion, and row construction.

use super::queue::Behind;
use super::*;
use crate::programs::SubmittedTarget;
use std::sync::atomic::{AtomicBool, Ordering};

static TARGET_PREFILL_LAUNCH_TRACED: AtomicBool = AtomicBool::new(false);
static TARGET_VERIFY_LAUNCH_TRACED: AtomicBool = AtomicBool::new(false);

/// Host timing of one finished target step. Device execution overlaps the
/// encode span; the selection gap is time the device may idle between steps.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TargetHostTiming {
    /// From entry into `submit_target` to the step's first commit.
    pub launch: Duration,
    /// From the step's first to its last command-buffer commit.
    pub encode: Duration,
    /// From the previous selection read to this step's first commit, when a
    /// selection was read since the previous step was submitted.
    pub selection_to_commit: Option<Duration>,
    /// From entry into `finish_target` to its return.
    pub finish: Duration,
}

impl<F: ProgramFamily> ExecutorDomain<F> {
    pub(super) fn trace_target_submission(
        &self,
        operations: &[Operation],
    ) -> Option<seismic::SubmissionTrace> {
        let trace_position = std::env::var("MAGNITUDE_TRACE_TARGET_PREFILL_MIN_POSITION")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(0);
        let trace_prefill = std::env::var_os("MAGNITUDE_TRACE_TARGET_PREFILL_LAUNCHES").is_some()
            && operations.iter().any(|operation| {
                matches!(operation, Operation::Forward { position, kind: WorkKind::Prefill, .. }
                    if *position >= trace_position)
            });
        let verify_position = std::env::var("MAGNITUDE_TRACE_TARGET_VERIFY_MIN_POSITION")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(0);
        let trace_verify = std::env::var_os("MAGNITUDE_TRACE_TARGET_VERIFY_LAUNCHES").is_some()
            && operations.iter().any(|operation| {
                matches!(operation, Operation::Forward {
                    position,
                    kind: WorkKind::Verify | WorkKind::Decode,
                    ..
                } if *position >= verify_position)
            });
        let traced = if trace_prefill && !TARGET_PREFILL_LAUNCH_TRACED.swap(true, Ordering::Relaxed)
        {
            Some(&TARGET_PREFILL_LAUNCH_TRACED)
        } else if trace_verify && !TARGET_VERIFY_LAUNCH_TRACED.swap(true, Ordering::Relaxed) {
            Some(&TARGET_VERIFY_LAUNCH_TRACED)
        } else {
            None
        };
        if let Some(traced) = traced {
            match self.domain.device().trace_submissions(
                if std::env::var("MAGNITUDE_TRACE_TARGET_DETAIL").as_deref() == Ok("submissions") {
                    seismic::TraceDetail::Submissions
                } else {
                    seismic::TraceDetail::Launches
                },
            ) {
                Ok(trace) => Some(trace),
                Err(error) => {
                    traced.store(false, Ordering::Relaxed);
                    eprintln!("target launch attribution unavailable: {error}");
                    None
                }
            }
        } else {
            None
        }
    }

    /// Submit one target group with state advances already reserved, or
    /// claim the queued step it equals. The steps that run behind it
    /// (`behind`) queue in the domain (`queue`); the flight holds the
    /// bindings until `finish_target` returns them.
    pub fn submit_target(
        &mut self,
        bindings: StateBindings,
        operations: &[Operation],
        reservation: TargetGraphReservation,
    ) -> Result<TargetFlight<F>, SubmitFailure> {
        let work = match reservation {
            TargetGraphReservation::Claim(slots) => self
                .claim_queued(operations, slots)
                .map_err(SubmitFailure::Failed)?,
            TargetGraphReservation::Launch(reservation) => {
                match self.launch_target(operations, reservation) {
                    Ok(work) => work,
                    Err(failure) => return Err(failure.with(bindings)),
                }
            }
        };
        // The step is on the device: any failure from here drops it. Its
        // tokens, now known whether launched or claimed, follow the
        // sequences' accepted ones.
        self.feed_tokens(operations)
            .map_err(|error| SubmitFailure::Failed(DomainError::invariant(error)))?;
        self.queue_successor(&work, operations)
            .map_err(SubmitFailure::Failed)?;
        // This launch was the round's verification: the next round's
        // drafter, if it proposes, marks its request again, as the next
        // round's method does a request whose round after prepares.
        for operation in operations {
            self.proposed.remove(&operation.request());
            self.publishing.remove(&operation.request());
        }
        Ok(TargetFlight { work, bindings })
    }

    /// Whether `request`'s next target launch is a round drafting nothing
    /// whose next round prepares to draft from a separate draft's fused
    /// taps: its decode continuation then publishes its feature, so the
    /// preparing round is that queued decode, discarding nothing
    /// (`queue::continuations`). Set before each round's launch.
    pub fn publish_behind(&mut self, request: RequestId, publishes: bool) {
        match publishes {
            true => self.publishing.insert(request),
            false => self.publishing.remove(&request),
        };
    }

    /// Launch the group's target step as the first of a chain: alone, or with
    /// the rest of the chain it opens (`behind`), queued behind it. Then each
    /// primed chunk's drafter entry behind its own step. A refused chain
    /// restores the step's accepted states and drafter entry.
    fn launch_target(
        &mut self,
        operations: &[Operation],
        mut reservation: TargetLaunchReservation,
    ) -> Result<TargetWork<F>, Unsubmitted> {
        let priming = reservation.priming.take();
        let abort = |domain: &mut Self, priming: Option<PrimingReservation>| {
            if let (Some(priming), [operation]) = (priming, operations) {
                domain
                    .head
                    .insert(operation.request(), priming.advance.abort());
            }
        };
        let (first, requests, spare) = match self.build_target_launch(operations, reservation) {
            Ok(built) => built,
            Err(failure) => {
                abort(self, priming);
                return Err(failure);
            }
        };
        let rest = match self.behind(operations) {
            Behind::Chain { chunks } => {
                self.build_chain(operations, &first, priming.as_ref(), chunks)
            }
            Behind::Nothing | Behind::Continuation { .. } | Behind::NextChunk { .. } => Ok((Vec::new(), Vec::new())),
        };
        let (launches, pending) = match rest {
            Ok(rest) => rest,
            Err(error) => {
                let (core, _) = first.into_submission_parts();
                let (_, advances, _, _) = core.into_parts();
                self.restore_advances(requests, advances);
                abort(self, priming);
                return Err(Unsubmitted::Failed(error));
            }
        };
        let mut chain = Chain::one(first);
        for launch in launches {
            chain.push(launch);
        }
        let launch_trace = self.trace_target_submission(operations);
        let started = Instant::now();
        let epoch = self.launch_epoch();
        let decodes = operations.iter().all(|operation| matches!(operation, Operation::Forward { kind: WorkKind::Decode, .. }));
        // A bank shrink's step is the first decode launch after it began; it
        // records no graph only where decode graphs are recorded ahead
        // (CUDA's): elsewhere it records its graphs at first use, as any
        // new shape's.
        let shrink_step = (decodes && std::mem::take(&mut self.shrink_step) && self.family.records_decode_ahead()).then(|| self.family.graph_replays().recorded);
        let submitted = self.submit_chain(chain, spare);
        if let Some(before) = shrink_step {
            assert_eq!(
                self.family.graph_replays().recorded,
                before,
                "a bank shrink's step records no graph where decode graphs are recorded ahead: every graph its decode launch replays was kept as its slots were made"
            );
        }
        let (submission, queued) = match submitted {
            Ok(submissions) => submissions.split_first(),
            Err(crate::programs::ChainRefusal { error, chain }) => {
                // The rest of the chain follows the first launch's advances
                // and returns its rows and banks as it drops.
                let (core, _) = chain.split_first().0.into_submission_parts();
                let (_, advances, _, _) = core.into_parts();
                self.restore_advances(requests, advances);
                abort(self, priming);
                return Err(Unsubmitted::Failed(error.into()));
            }
        };
        let flight_requests = requests.iter().map(|(request, ..)| *request).collect::<Vec<_>>();
        let work = TargetWork {
            runnable: self.service_start(&flight_requests, started),
            requests,
            submission,
            priming: None,
            launch_trace,
            started,
            epoch,
            previous_selection: self.selection_read.take(),
            id: self.flight_id(),
            continuation: None,
        };
        let work = self.attach_priming(work, operations, priming)?;
        if !pending.is_empty() {
            self.queue_chain(
                &work,
                pending.into_iter().zip(queued),
                started,
                epoch,
            )
            .map_err(Unsubmitted::Failed)?;
        }
        Ok(work)
    }

    /// The group's drafter entry behind its launched step, when its prompt
    /// chunk primes one.
    fn attach_priming(
        &mut self,
        work: TargetWork<F>,
        operations: &[Operation],
        priming: Option<PrimingReservation>,
    ) -> Result<TargetWork<F>, Unsubmitted> {
        let Some(PrimingReservation {
            advance,
            graph_workspace,
            graph_output,
        }) = priming
        else {
            return Ok(work);
        };
        let [
            Operation::Forward {
                request,
                prime: Some(prime),
                ..
            },
        ] = operations
        else {
            return Err(Unsubmitted::Failed(DomainError::invariant(
                "a drafter entry was reserved for an unprimed group",
            )));
        };
        self.launch_priming(
            work,
            *request,
            prime,
            TentativeAdvance::Accepted(advance),
            self.input
                .get(request)
                .and_then(|input| input.boundary.clone()),
            graph_workspace,
            graph_output,
        )
        .map_err(Unsubmitted::Failed)
    }

    /// Draft `prime` behind `work` (whose lone slot is its prompt chunk),
    /// conditioned by the step's feature output on the device, in the
    /// drafter's form: a block drafter's rows inject, a chained head's rows
    /// run its entry pass over the head history. `work` is on the device, so
    /// every error is fatal.
    pub(super) fn launch_priming(
        &mut self,
        mut work: TargetWork<F>,
        request: RequestId,
        prime: &Priming,
        advance: TentativeAdvance,
        previous: Option<FeatureRef>,
        graph_workspace: NativeGraphWorkspaceLease,
        graph_output: NativeGraphOutputLease,
    ) -> Result<TargetWork<F>, DomainError> {
        let features = work
            .submission
            .output()
            .readout
            .as_ref()
            .map(|readout| readout.features.clone())
            .ok_or_else(|| DomainError::invariant("a primed prompt chunk publishes no features"))?;
        let store = self
            .head_store
            .clone()
            .ok_or_else(|| DomainError::invariant("a drafter entry without a drafter store"))?;
        let vocabulary = self.definition.decoder.vocabulary as usize;
        let class_limits = self.head_class_limits()?;
        let batch = match self.execution.policy().method().draft_form() {
            Some(crate::DraftForm::Block) => self
                .block_slot(&prime.tokens, prime.position, &[], &advance, 0)
                .and_then(|slot| {
                    crate::batching::ValidatedHeadBatch::from_block_slots(
                        &[slot],
                        vocabulary,
                        class_limits,
                    )
                    .map_err(|error| error.to_string())
                }),
            Some(crate::DraftForm::Chained) => self
                .head_slot(request, &prime.tokens, prime.position, &[], &advance, 0)
                .and_then(|slot| {
                    crate::batching::ValidatedHeadBatch::from_slots(
                        &[slot],
                        vocabulary,
                        class_limits,
                    )
                    .map_err(|error| error.to_string())
                }),
            None => Err("a drafter entry without a drafting method".into()),
        }
        .map_err(DomainError::invariant)?;
        // Entry rows every window drops before the first draft skip the
        // windowed layers (see `submit_head`).
        let windowed = super::head::skippable_window(&store)
            .is_none_or(|window| prime.position + prime.tokens.len() + window > prime.draft_from);
        let inputs = HeadLaunchInputs::new(
            batch,
            vec![advance],
            HeadConditioning::Features {
                current: features,
                previous: prime.boundary.then_some(previous).flatten(),
            },
            windowed,
            graph_workspace,
            graph_output,
        );
        let launch = match ValidatedHeadLaunch::new(
            inputs,
            &store,
            self.domain.id(),
            self.head_conditioning_bytes(),
        ) {
            Ok(launch) => launch,
            Err((_, error)) => return Err(DomainError::Invariant(error)),
        };
        let submission = self
            .family
            .submit_head(launch)
            .map_err(|(error, _)| DomainError::from(error))?;
        work.priming = Some(PrimingFlight {
            request,
            submission,
            continuation: None,
        });
        Ok(work)
    }

    /// The group's target launch over its reserved advances and leases,
    /// with each operation's request, rows, conditioning, kind and
    /// committed rows, and the spare output slot of the chain it heads. A
    /// refused launch restores the accepted states.
    #[allow(clippy::type_complexity)]
    fn build_target_launch(
        &mut self,
        operations: &[Operation],
        reservation: TargetLaunchReservation,
    ) -> Result<
        (
            ValidatedTargetLaunch,
            Vec<(RequestId, usize, Option<crate::ConditioningRef>, WorkKind, usize)>,
            NativeGraphOutputLease,
        ),
        Unsubmitted,
    > {
        let TargetLaunchReservation {
            advances,
            graph_workspace,
            graph_output,
            spare_output,
            readout_workspace,
            readout_output,
            priming: _,
            boundary,
        } = reservation;
        let advances = advances
            .into_iter()
            .map(TentativeAdvance::Accepted)
            .collect::<Vec<_>>();
        // Reservation checked the group and moved its states into the
        // advances: a mismatch now is a broken domain invariant.
        let broken = |detail: String| Unsubmitted::Failed(DomainError::invariant(detail));
        if operations.len() != advances.len() {
            return Err(broken(
                "target reservation advance count differs from operations".into(),
            ));
        }
        if operations.is_empty() {
            return Err(Unsubmitted::Refused("target group is empty".into()));
        }
        let mut seen = BTreeSet::new();
        let mut rows = 0usize;
        let mut demand = crate::batching::Demand::NONE;
        let mut segments = 1usize;
        for (operation, advance) in operations.iter().zip(&advances) {
            operation
                .validate()
                .map_err(|error| broken(error.to_string()))?;
            let Operation::Forward {
                request,
                position,
                tokens,
                conditioning,
                demand: next,
                ..
            } = operation
            else {
                return Err(broken(
                    "target group contains another operation kind".into(),
                ));
            };
            if !seen.insert(*request) {
                return Err(broken("target group repeats a request".into()));
            }
            if advance.position() != *position || advance.rows() != tokens.len() {
                return Err(broken("target position differs from accepted state".into()));
            }
            if conditioning.as_ref().is_some_and(|lease| {
                lease.domain() != self.domain.id() || lease.allocation().rows() != tokens.len()
            }) {
                return Err(broken(
                    "target conditioning differs from physical rows or resource domain".into(),
                ));
            }
            rows = rows
                .checked_add(tokens.len())
                .ok_or_else(|| broken("target row count overflow".into()))?;
            demand |= *next;
            segments = segments.max(reserved_segments(
                &self.target_store,
                advance.span_count(),
                tokens.len(),
            ));
        }
        let class_limits = self.target_class_limits();
        let class = crate::LaunchClass::covering(rows, segments, demand, class_limits)
            .map_err(|error| broken(error.to_string()))?;
        let preceding = self.target_preceding(operations, &[]).map_err(broken)?;
        let mut metadata = Vec::with_capacity(operations.len());
        for operation in operations {
            let request = operation.request();
            let Operation::Forward {
                conditioning,
                kind,
                committed,
                ..
            } = operation
            else {
                unreachable!()
            };
            metadata.push((
                request,
                operation.row_count(),
                conditioning.clone(),
                *kind,
                *committed,
            ));
        }
        let prepared = operations
            .iter()
            .zip(&advances)
            .map(|(operation, advance)| self.target_slot(operation, advance))
            .collect::<Result<Vec<_>, _>>();
        let prepared = match prepared {
            Ok(prepared) => prepared,
            Err(error) => {
                self.restore_advances(metadata, advances);
                return Err(Unsubmitted::Refused(error.into()));
            }
        };
        let (slots, conditioning_slices): (Vec<_>, Vec<_>) = prepared.into_iter().unzip();
        let batch = match ValidatedTargetBatch::covering(
            &slots,
            self.definition.decoder.vocabulary as usize,
            class_limits,
            segments,
        ) {
            Ok(batch) if batch.class() == class => batch,
            Ok(_) => {
                self.restore_advances(metadata, advances);
                return Err(broken(
                    "target packed class changed after reservation".into(),
                ));
            }
            Err(error) => {
                self.restore_advances(metadata, advances);
                return Err(Unsubmitted::Refused(error.to_string().into()));
            }
        };
        let conditioning = metadata
            .iter()
            .map(|(_, _, lease, _, _)| lease.clone())
            .collect();
        let work = crate::operation::LaunchWork::of(metadata.iter().map(|&(_, _, _, kind, _)| kind));
        let inputs = TargetLaunchInputs::new(
            batch,
            work,
            TargetTokens::Host,
            advances,
            conditioning,
            conditioning_slices,
            graph_workspace,
            graph_output,
            readout_workspace,
            readout_output,
        )
        .with_boundary(boundary)
        .with_preceding(preceding);
        match ValidatedTargetLaunch::new(
            inputs,
            &self.target_store,
            self.domain.id(),
            self.definition.decoder.hidden as usize,
            self.definition.head.as_ref().map_or(self.definition.decoder.hidden, |head| head.conditioning_width(&self.definition.decoder)) as usize,
        ) {
            Ok(launch) => Ok((launch, metadata, spare_output)),
            Err((inputs, error)) => {
                let (_, advances, _, _) = inputs.into_parts();
                self.restore_advances(metadata, advances);
                Err(Unsubmitted::Failed(DomainError::Invariant(error)))
            }
        }
    }

    /// Completion exposes immutable outcomes while every owned advance stays
    /// suspended. The caller can stage grammar and method changes from these
    /// views before choosing physical prefixes. Returns the bindings the
    /// flight held, with the steps queued behind it; an error consumes them, as
    /// the domain cannot continue.
    /// Per operation of a target group, the tokens its sequence holds before
    /// the operation's first row, as many as the model's row rules read
    /// (`TargetLaunchCore::preceding`); none of a model without one. They
    /// are the sequence's fed tokens followed by those of its operations in
    /// `ahead`, the launches of its chain queued before the group, which
    /// feed theirs only once the chain is on the device.
    pub(super) fn target_preceding(
        &self,
        operations: &[Operation],
        ahead: &[Operation],
    ) -> Result<Vec<Vec<u64>>, String> {
        let context = crate::operators::row_rule_context(&self.definition.decoder);
        if context == 0 {
            return Ok(Vec::new());
        }
        operations
            .iter()
            .map(|operation| {
                let Operation::Forward {
                    request, position, ..
                } = operation
                else {
                    return Err("target group contains another operation kind".to_owned());
                };
                let ahead = ahead
                    .iter()
                    .filter(|ahead| ahead.request() == *request)
                    .map(|ahead| match ahead {
                        Operation::Forward {
                            position, tokens, ..
                        } => Ok((*position, tokens)),
                        _ => Err("a chain holds another operation kind".to_owned()),
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let mut end = ahead.first().map_or(*position, |(start, _)| *start);
                let mut tokens = self.preceding_tokens(*request, end, context)?.to_vec();
                for (at, fed) in ahead {
                    if at != end {
                        return Err("a chain's launches do not follow each other".into());
                    }
                    tokens.extend_from_slice(fed);
                    end += fed.len();
                }
                if end != *position {
                    return Err("a chained launch does not follow the launches before it".into());
                }
                Ok(tokens[tokens.len().saturating_sub(context)..]
                    .iter()
                    .map(|token| u64::from(token.0))
                    .collect())
            })
            .collect()
    }

    pub fn finish_target(
        &mut self,
        flight: TargetFlight<F>,
    ) -> Result<(Vec<PendingOperationOutcome>, StateBindings), DomainError> {
        let TargetFlight { work, bindings } = flight;
        let pending = self.finish_target_work(work)?;
        Ok((pending, bindings))
    }

    fn finish_target_work(
        &mut self,
        flight: TargetWork<F>,
    ) -> Result<Vec<PendingOperationOutcome>, DomainError> {
        let completed = match flight.submission.finish() {
            Ok(completed) => completed,
            Err(error) => {
                return Err(DomainError::Device(error));
            }
        };
        // The expert bank's moves the device ran before this flight, which
        // it read: their failure is this flight's.
        if let Some(bank) = self.family.expert_bank() {
            bank.carried()?;
        }
        // The drafter entry follows its chunk on the device; its advance
        // commits with the chunk. A claimed queued step's entry attaches to the
        // accepted drafter state; an unclaimed one's rows are discarded.
        let mut primed = flight
            .priming
            .map(|priming| self.finish_priming_work(priming))
            .transpose()?
            .flatten();
        if let Some(trace) = &flight.launch_trace {
            let submissions = trace
                .collect()
                .map_err(|error| DomainError::Input(error.to_string()))?;
            let mut entries = BTreeMap::<String, (usize, f64)>::new();
            let mut geometry =
                BTreeMap::<(String, usize, [u64; 3], [u64; 3], u64), (usize, usize, f64)>::new();
            eprintln!("target trace scope: {} submissions", submissions.len());
            for submission in submissions {
                for launch in submission.launches {
                    if let Some(physical) = launch.geometry {
                        let entry = geometry
                            .entry((
                                launch.entry.clone(),
                                launch.launch,
                                physical.workgroups,
                                physical.threads_per_workgroup,
                                physical.shared_bytes,
                            ))
                            .or_default();
                        entry.0 += 1;
                        entry.1 = entry.1.max(launch.occurrence + 1);
                        if let Some((start, end)) = launch.device {
                            entry.2 += (end - start) * 1_000.0;
                        }
                    }
                    if let Some((start, end)) = launch.device {
                        let label = if launch.launch == 0 {
                            launch.entry
                        } else {
                            format!("{}#{}", launch.entry, launch.launch)
                        };
                        let entry = entries.entry(label).or_default();
                        entry.0 += 1;
                        entry.1 += (end - start) * 1_000.0;
                    }
                }
            }
            let mut entries = entries.into_iter().collect::<Vec<_>>();
            entries.sort_by(|a, b| b.1.1.total_cmp(&a.1.1));
            for ((entry, launch, groups, threads, shared), (count, occurrences, gpu_ms)) in geometry
            {
                eprintln!(
                    "target geometry entry={entry} launch={launch} groups={groups:?} threads={threads:?} shared_bytes={shared} count={count} max_occurrences={occurrences} gpu_ms={gpu_ms:.3}"
                );
            }
            eprintln!("target timed launch attribution (trace scope):");
            for (entry, (launches, ms)) in entries {
                eprintln!("  {entry}: {ms:.3} ms across {launches} launches");
            }
        }
        let finish_started = Instant::now();
        let physical_duration = finish_started.saturating_duration_since(flight.runnable);
        let timed = self.timed(flight.epoch);
        if std::env::var_os("MAGNITUDE_TRACE_FLIGHTS").is_some() {
            eprintln!(
                "flight target duration_ms={:.3} requests={:?}",
                physical_duration.as_secs_f64() * 1000.0,
                flight
                    .requests
                    .iter()
                    .map(|(request, rows, _, kind, committed)| { (request, rows, kind, committed) })
                    .collect::<Vec<_>>()
            );
        }
        let (core, output) = completed.into_parts();
        let batch = core.batch();
        if batch.actual_slots() != flight.requests.len() {
            return Err("completed target slot count differs from submitted requests".into());
        }
        // A sequence whose weights (stored-table rows, landed rows or a
        // streamed layer's experts) could not be read in this step read
        // others: it is poisoned, and this outcome and every later one of
        // it fail. The step's other sequences are unaffected. A step the
        // expert bank interrupted (a late beat of its load's pulse) poisons
        // nothing: every request of it is interrupted, to be re-run (below).
        let crate::programs::native_target::CompletedTargetOutput {
            readout: output,
            commits,
            boundary,
            failures,
            experts,
        } = output.complete(batch, super::recording(self.expert_measures(), core.work())).map_err(DomainError::Invariant)?;
        let interrupted = match experts {
            crate::programs::native_target::ExpertStepEnd::Ran => None,
            crate::programs::native_target::ExpertStepEnd::Interrupted(reason) => Some(reason),
        };
        for ((request, ..), failure) in flight.requests.iter().zip(failures) {
            if let Some(error) = failure {
                self.poisoned.entry(*request).or_insert(error);
            }
        }
        let upload = batch.upload();
        if let Some(rows) = std::env::var_os("MAGNITUDE_TRACE_TARGET_BLOCK_ROWS") {
            self.trace_block_rows(&rows.to_string_lossy(), batch.actual_rows())
                .map_err(DomainError::invariant)?;
        }
        let selected = if upload.select_rows.is_empty() {
            Vec::new()
        } else {
            let tensor = output
                .as_ref()
                .and_then(|result| result.selected.as_ref())
                .ok_or("completed target has no selection tensor")?
                .slice_leading(0, upload.select_rows.len() as u64)
                .map_err(|error| error.to_string())?;
            let bytes = tensor.tensor().read_to_host().map_err(|error| {
                DomainError::Device(crate::DeviceError::Transfer(error.to_string()))
            })?;
            self.selection_read = Some(Instant::now());
            decode_selected(&bytes)?
        };
        let mut physical = vec![RowResult::default(); batch.actual_rows()];
        for (projected_index, &output_index) in output
            .as_ref()
            .map(|result| result.projected_output_rows.as_slice())
            .unwrap_or_default()
            .iter()
            .enumerate()
        {
            let row = *upload
                .out_rows
                .get(output_index)
                .ok_or("projected output row exceeds output rows")?;
            let row = usize::try_from(row).map_err(|_| "negative target output row")?;
            let result = physical
                .get_mut(row)
                .ok_or("target output row exceeds actual batch")?;
            let demand = crate::batching::Demand::from_bits(upload.demand[row])
                .ok_or("target output demand is invalid")?;
            if demand.contains(crate::batching::Demand::LOGITS) {
                let view = output
                    .as_ref()
                    .and_then(|result| result.logits.as_ref())
                    .ok_or("target logits output is absent")?
                    .slice_leading(projected_index as u64, projected_index as u64 + 1)
                    .map_err(|error| error.to_string())?;
                result.logits = Some(
                    self.domain
                        .publish_graph_logits(view)
                        .map_err(|error| error.to_string())?,
                );
            }
        }
        for (index, &output_index) in upload.select_rows.iter().enumerate() {
            let output_index = usize::try_from(output_index)
                .map_err(|_| "negative target selection output index")?;
            let row = *upload
                .out_rows
                .get(output_index)
                .ok_or("target selection output index exceeds output rows")?;
            let row = usize::try_from(row).map_err(|_| "negative target selection row")?;
            let result = physical
                .get_mut(row)
                .ok_or("target selection row exceeds actual batch")?;
            result.selected = Some(
                *selected
                    .get(index)
                    .ok_or("target selection count differs")?,
            );
        }
        let mut request_start = 0usize;
        for (_, request_rows, _, _, _) in &flight.requests {
            let request_end = request_start
                .checked_add(*request_rows)
                .ok_or("request row range overflows")?;
            let feature_outputs = upload
                .out_rows
                .iter()
                .enumerate()
                .filter_map(|(output_index, row)| {
                    let row = usize::try_from(*row).ok()?;
                    let demand = crate::batching::Demand::from_bits(*upload.demand.get(row)?)?;
                    (row >= request_start
                        && row < request_end
                        && demand.contains(crate::batching::Demand::FEATURES))
                    .then_some((output_index, row))
                })
                .collect::<Vec<_>>();
            if let Some(&(first, _)) = feature_outputs.first() {
                if feature_outputs
                    .iter()
                    .enumerate()
                    .any(|(offset, (index, _))| *index != first + offset)
                {
                    return Err("request feature rows are not contiguous in physical output".into());
                }
                let view = output
                    .as_ref()
                    .map(|result| &result.features)
                    .ok_or("target feature output is absent")?
                    .slice_leading(first as u64, (first + feature_outputs.len()) as u64)
                    .map_err(|error| error.to_string())?;
                let aggregate = self
                    .domain
                    .publish_graph_features(view)
                    .map_err(|error| error.to_string())?;
                let aggregate = match &boundary {
                    Some(boundary) => aggregate
                        .with_boundary(boundary.clone())
                        .map_err(|error| error.to_string())?,
                    None => aggregate,
                };
                for (_, row) in feature_outputs {
                    physical[row].features = Some(aggregate.clone());
                }
            }
            request_start = request_end;
        }
        let (_, advances, _, _) = core.into_parts();
        let mut pending = Vec::with_capacity(flight.requests.len());
        let mut offset = 0usize;
        let mut continuation = flight.continuation;
        // A flight the rest of a chain is queued behind opens that chain.
        let opens_chain = self.queue.opens_chain(flight.id);
        // The chain an interrupted chunk releases, once every outcome is in.
        let mut released_chain = None;
        for (index, ((request, rows, _, kind, committed_rows), advance)) in
            flight.requests.into_iter().zip(advances).enumerate()
        {
            let end = offset
                .checked_add(rows)
                .ok_or("target request row count overflow")?;
            let result = physical
                .get(offset..end)
                .ok_or("target request row slice exceeds batch")?
                .to_vec();
            offset = end;
            let source = continuation.as_mut().map(|sources| sources[index].take());
            let advance = match (advance, source) {
                (TentativeAdvance::Accepted(advance), None) if opens_chain => {
                    self.open_chain(request, advance)?
                }
                (TentativeAdvance::Accepted(advance), None) => Advance::Step(advance),
                // A claimed chunk of the chain its holder's state is in.
                (TentativeAdvance::Chained(chunk), Some(None)) => {
                    self.join_chain(request, chunk)?
                }
                (TentativeAdvance::Successor(successor), Some(Some(state))) => {
                    match successor.attach(state.into_state()) {
                        Ok(advance) => Advance::Step(advance),
                        Err((state, error)) => {
                            self.target.insert(request, state);
                            return Err(DomainError::invariant(format!(
                                "claimed queued slot no longer follows its state: {error}"
                            )));
                        }
                    }
                }
                // Nobody claimed this slot: its rows are discarded.
                (TentativeAdvance::Successor(_), Some(None)) => continue,
                _ => {
                    return Err(DomainError::invariant(
                        "target flight advances differ from its continuation",
                    ));
                }
            };
            let mut primed_here = primed.take_if(|(primed, _)| *primed == request).map(|(_, advance)| advance);
            let (outcome, advance) = match (self.poisoned.get(&request), &interrupted) {
                (Some(error), _) if matches!(advance, Advance::Step(_)) => {
                    (Outcome::WeightReadFailed(error.clone()), Some(advance))
                }
                // A chunk of a chain that read weights it could not: its
                // holder fails and claims no more of the chain, so the chain
                // is released whole now, its holder's state with it, as an
                // interrupted chain's is. Held on, the next request's work
                // would orphan a chain under way.
                (Some(error), _) => {
                    let outcome = Outcome::WeightReadFailed(error.clone());
                    released_chain = Some(self.release_chunk_chain(request, advance, primed_here.take()));
                    (outcome, None)
                }
                (None, None) => (Outcome::Forward { rows: result }, Some(advance)),
                // Its accepted state returns as its outcome is aborted: it is
                // re-run from there.
                (None, Some(reason)) if matches!(advance, Advance::Step(_)) => {
                    (Outcome::Interrupted { reason: reason.clone(), resident: true }, Some(advance))
                }
                // A chunk of a chain: the chain held what its earlier chunks
                // computed, which their rounds already reconciled, and holds
                // this chunk's now. It is released whole, and its holder's
                // state with it: the request re-prefills, as an evicted one.
                (None, Some(reason)) => {
                    released_chain = Some(self.release_chunk_chain(request, advance, primed_here.take()));
                    (Outcome::Interrupted { reason: reason.clone(), resident: false }, None)
                }
            };
            pending.push(PendingOperationOutcome {
                request,
                outcome,
                advance,
                primed: primed_here,
                rows,
                committed_rows,
                kind,
                physical_duration,
                timed,
                image: None,
            });
        }
        if primed.is_some() {
            return Err(DomainError::invariant(
                "a drafter entry's chunk is absent from its flight",
            ));
        }
        match released_chain {
            Some(Released::Held) => self.release_chain()?,
            Some(Released::Whole(holder)) => {
                self.release_state(&[holder]).map_err(DomainError::invariant)?;
            }
            None => {}
        }
        let finished = Instant::now();
        let timing = TargetHostTiming {
            launch: commits.first.saturating_duration_since(flight.started),
            encode: commits.last.saturating_duration_since(commits.first),
            selection_to_commit: flight
                .previous_selection
                .map(|read| commits.first.saturating_duration_since(read)),
            finish: finished.saturating_duration_since(finish_started),
        };
        if self.trace_host_steps {
            eprintln!(
                "target host step rows={} launch_us={} encode_us={} selection_to_commit_us={} finish_us={}",
                physical.len(),
                timing.launch.as_micros(),
                timing.encode.as_micros(),
                timing
                    .selection_to_commit
                    .map_or_else(|| "-".to_owned(), |gap| gap.as_micros().to_string()),
                timing.finish.as_micros(),
            );
        }
        // Finish already times result reads/materialization. Include that
        // work in completed service without another timer or device wait.
        for item in &mut pending {
            item.physical_duration += timing.finish;
        }
        // The queued step's service starts where this flight's ends, so the
        // finish is charged once: a backend whose reads queue behind the
        // queued step waits in it for that step's device work.
        self.predecessor_selected(flight.id, &selected, finished);
        self.target_timing = Some(timing);
        self.begin_host_turn(pending.iter().map(|item| item.request));
        Ok(pending)
    }

    /// Release whole the chain `request`'s chunk was of, `advance` (a chunk's, never a step's),
    /// its holder's state with it, and the drafter state `primed` beside the chunk: the request
    /// claims no more of the chain. What is released, once every outcome is in.
    fn release_chunk_chain(&mut self, request: RequestId, advance: Advance, primed: Option<OwnedStateAdvance>) -> Released {
        let released = match advance {
            Advance::Chain(chain) => {
                self.target.insert(request, chain.abort());
                Released::Whole(request)
            }
            Advance::Chained => Released::Held,
            Advance::Step(_) => unreachable!("a step's advance is of no chain"),
        };
        if let Some(primed) = primed {
            self.head.insert(request, primed.abort());
        }
        released
    }

    fn finish_priming_work(
        &mut self,
        priming: PrimingFlight<F::HeadSubmission>,
    ) -> Result<Option<(RequestId, OwnedStateAdvance)>, DomainError> {
        let completed = priming.submission.finish().map_err(DomainError::Device)?;
        let (core, _) = completed.into_parts();
        let (_, advances, _) = core.into_parts();
        let [advance]: [TentativeAdvance; 1] = advances
            .try_into()
            .map_err(|_| DomainError::invariant("a drafter entry has other than one slot"))?;
        match (advance, priming.continuation) {
            (TentativeAdvance::Accepted(advance), None) => Ok(Some((priming.request, advance))),
            (TentativeAdvance::Successor(successor), Some(Some(state))) => {
                match successor.attach(state.into_state()) {
                    Ok(advance) => Ok(Some((priming.request, advance))),
                    Err((state, error)) => {
                        self.head.insert(priming.request, state);
                        return Err(DomainError::invariant(format!(
                            "claimed drafter entry no longer follows its state: {error}"
                        )));
                    }
                }
            }
            (TentativeAdvance::Successor(_), Some(None)) => Ok(None),
            _ => {
                return Err(DomainError::invariant(
                    "drafter entry advance differs from its continuation",
                ));
            }
        }
    }

    /// Finish a target chunk for publication without returning the binding
    /// right while its final drafter entry still uses it. The ordinary
    /// finish remains useful to callers that consume the whole chunk.
    pub fn finish_target_for_publication(
        &mut self,
        mut flight: TargetFlight<F>,
        final_prompt_chunk: bool,
    ) -> Result<(Vec<PendingOperationOutcome>, TargetContinuation<F>), DomainError> {
        let priming = final_prompt_chunk
            .then(|| flight.detach_priming())
            .flatten();
        let (outcomes, bindings) = self.finish_target(flight)?;
        let continuation = match priming {
            None => TargetContinuation::Ready(bindings),
            Some((work, target_end)) => TargetContinuation::Priming(PrimingCompletionFlight {
                work,
                target_end,
                bindings,
                started: Instant::now(),
            }),
        };
        Ok((outcomes, continuation))
    }

    pub fn finish_priming(
        &mut self,
        flight: PrimingCompletionFlight<F>,
    ) -> Result<(RequestId, Duration, StateBindings), DomainError> {
        let request = flight.work.request;
        if let Some((_, advance)) = self.finish_priming_work(flight.work)? {
            if self.head.contains_key(&request) {
                return Err(DomainError::invariant(
                    "deferred entry has another head owner",
                ));
            }
            // The entry commits with its chunk: in the request's accepted
            // state, or in the chain under way that holds it.
            let accepted = self.target_end(request) == Some(flight.target_end);
            let state = if accepted {
                let rows = advance.rows();
                let (OwnedAdvanceResolution::Aborted(state)
                | OwnedAdvanceResolution::Committed(state)) =
                    advance.commit(rows).map_err(|(state, error)| {
                        self.head.insert(request, state);
                        DomainError::from(error)
                    })?;
                state
            } else {
                advance.abort()
            };
            self.head.insert(request, state);
        }
        Ok((request, flight.started.elapsed(), flight.bindings))
    }

    fn restore_advances(
        &mut self,
        metadata: Vec<(
            RequestId,
            usize,
            Option<crate::ConditioningRef>,
            WorkKind,
            usize,
        )>,
        advances: Vec<TentativeAdvance>,
    ) {
        for ((request, _, _, _, _), advance) in metadata.into_iter().zip(advances) {
            match advance {
                TentativeAdvance::Accepted(advance) => {
                    self.target.insert(request, advance.abort());
                }
                // A successor's or chained chunk's source state stays with
                // its predecessor; dropping it releases its rows and bank.
                TentativeAdvance::Successor(successor) => drop(successor),
                TentativeAdvance::Chained(chunk) => drop(chunk),
            }
        }
    }

    /// Per history read of the target store (`StateStore::history_reads`),
    /// whether its layers attend media rows bidirectionally
    /// (`operators::attention::admit_media`).
    fn bidirectional_reads(&self) -> Result<Vec<bool>, String> {
        let store_reads = self.target_store.history_reads();
        let mut reads = vec![false; store_reads.len()];
        for (index, sublayer) in self.definition.decoder.sublayers() {
            let magnitude_family_contracts::Operator::Attention(attention) = &sublayer.op else {
                continue;
            };
            if attention.media_rows == magnitude_family_contracts::MediaRowAttention::Bidirectional
            {
                let read = store_reads
                    .read(magnitude_state::LayerRef::Target(index.block))
                    .ok_or("an attention layer has no history read")?;
                reads[read] = true;
            }
        }
        Ok(reads)
    }

    /// The launch audit's stored values (`MAGNITUDE_TRACE_TARGET_BLOCK_ROWS`,
    /// a comma list of physical rows): once a target launch of `launch_rows`
    /// rows completes, every Block domain plane's values at those rows (the
    /// pooled rows and index keys as stored, after their rounding), so two
    /// runs' stored codes compare row for row.
    fn trace_block_rows(&self, rows: &str, launch_rows: usize) -> Result<(), String> {
        let wanted = rows
            .split(',')
            .map(|row| row.trim().parse::<usize>().map_err(|_| format!("a traced block row {row:?} is no row")))
            .collect::<Result<Vec<_>, _>>()?;
        let planes = self.target_store.history_planes().map_err(|error| error.to_string())?;
        for plane in planes {
            if !matches!(
                self.target_store.history_domain_kind(plane.domain),
                magnitude_state::HistoryDomainKind::Block { .. }
            ) {
                continue;
            }
            let dtype = self.target_store.history_components(plane.domain)[plane.component_index]
                .planes()
                .iter()
                .find(|described| described.name == plane.name && described.vector == plane.vector)
                .map(|described| described.dtype)
                .ok_or("a history plane has no descriptor")?;
            let bytes = plane.buffer.read_to_host().map_err(|error| error.to_string())?;
            for &row in &wanted {
                let Some(stored) = bytes.get(row * plane.row_bytes..(row + 1) * plane.row_bytes) else {
                    continue;
                };
                let values = match dtype {
                    seismic::DType::BF16 => stored
                        .chunks_exact(2)
                        .map(|value| f32::from_bits(u32::from(u16::from_le_bytes([value[0], value[1]])) << 16).to_string())
                        .collect::<Vec<_>>(),
                    seismic::DType::F32 => stored
                        .chunks_exact(4)
                        .map(|value| f32::from_le_bytes([value[0], value[1], value[2], value[3]]).to_string())
                        .collect(),
                    _ => stored.iter().map(|byte| format!("{byte:02x}")).collect(),
                };
                eprintln!(
                    "target block row launch_rows={launch_rows} layer={:?} plane={:?} row={row} values={}",
                    plane.layer,
                    plane.name,
                    values.join(",")
                );
            }
        }
        Ok(())
    }

    pub(super) fn target_slot(
        &self,
        operation: &Operation,
        advance: &TentativeAdvance,
    ) -> Result<(Slot, Vec<crate::ConditioningSlice>), String> {
        let Operation::Forward {
            request,
            tokens,
            position,
            kind,
            ..
        } = operation
        else {
            return Err("non-forward target operation".into());
        };
        let (coordinates, slices) = self.input_rows(*request, *position, tokens)?;
        let binding = advance.bindings();
        // A media row's span end; an advance never splits a span. The
        // history reads whose layers attend media bidirectionally read every
        // fresh row of the span.
        let bidirectional = self.bidirectional_reads()?;
        let span_end = |index: usize| {
            slices
                .iter()
                .map(|slice| (slice.destination, slice.destination + slice.source.count))
                .find(|(start, end)| (*start..*end).contains(&index))
                .map(|(_, end)| end)
        };
        let appends = SlotAppends::of(&self.target_store, advance);
        let mut rows = Vec::with_capacity(tokens.len());
        let audit_history = matches!(kind, WorkKind::Verify | WorkKind::Decode)
            && std::env::var_os("MAGNITUDE_TRACE_TARGET_VERIFY_LAUNCHES").is_some()
            && *position
                >= std::env::var("MAGNITUDE_TRACE_TARGET_VERIFY_MIN_POSITION")
                    .ok()
                    .and_then(|value| value.parse::<usize>().ok())
                    .unwrap_or(0)
            && !TARGET_VERIFY_LAUNCH_TRACED.load(Ordering::Relaxed);
        for (index, token) in tokens.iter().enumerate() {
            let coordinates = *coordinates
                .get(index)
                .ok_or("prepared input coordinate count differs from tokens")?;
            let histories = row_histories(
                &self.target_store,
                advance,
                &appends,
                index,
                span_end(index),
                &bidirectional,
            )?;
            if audit_history && index < 4 {
                for domain in self.target_store.history_domains() {
                    let history = &histories[domain.0];
                    let visible_rows: i64 = history
                        .visible
                        .iter()
                        .map(|range| i64::from(range[1]) - i64::from(range[0]))
                        .sum();
                    eprintln!(
                        "target history position={} row={index} domain={} kind={:?} visible_rows={visible_rows} spans={} first_span={:?} last_span={:?} fresh_start={} slab_rows={}",
                        position + index,
                        domain.0,
                        self.target_store.history_domain_kind(domain),
                        history.visible.len(),
                        history.visible.first(),
                        history.visible.last(),
                        history.fresh_start,
                        self.target_store.history_slab_rows(domain)
                    );
                    if index == 0 {
                        for component in self.target_store.history_components(domain) {
                            eprintln!(
                                "target history component layer={:?} heads={} codec={:?} planes={:?}",
                                component.layer,
                                component.heads,
                                component.codec,
                                component.planes()
                            );
                        }
                    }
                }
            }
            rows.push(Row {
                token: i32::try_from(token.0).map_err(|_| "token exceeds i32")?,
                coordinates,
                histories,
                demand: row_demand(operation, index),
                select: selection(operation, index),
            });
        }
        Ok((
            Slot {
                rows,
                bank: i32::try_from(binding.previous_bank).map_err(|_| "bank exceeds i32")?,
                previous_tape: i32::try_from(binding.previous_tape)
                    .map_err(|_| "tape rows exceed i32")?,
                following_bank: i32::try_from(binding.following_bank)
                    .map_err(|_| "successor bank exceeds i32")?,
                stop: i32::try_from(binding.stop).map_err(|_| "committed rows exceed i32")?,
            },
            slices,
        ))
    }
}

/// The histories row `index` of `advance` reads and writes, one per history
/// read of `store` (`StateStore::history_reads`). A stored domain's: its
/// accepted rows from the query's window start, the advance's fresh rows
/// from the same start (through the row, or, for a media row of a read in
/// `bidirectional`, through its media span's end `span_end`), and the row's
/// destination. A Shared source domain's (`shared_read_history`) follow.
/// `appends` is the advance's (`SlotAppends::of`), formed once for its rows.
pub(super) fn row_histories(
    store: &StateStore,
    advance: &TentativeAdvance,
    appends: &SlotAppends,
    index: usize,
    span_end: Option<usize>,
    bidirectional: &[bool],
) -> Result<Vec<RowHistory>, String> {
    let binding = advance.bindings();
    let end = |read: usize| span_end.filter(|_| bidirectional[read]);
    let stored = store.history_domains().map(|domain| {
        if store.is_produced_domain(domain) {
            let appended = appends.causal[domain.0]
                .as_ref()
                .ok_or("an indexer domain has no appended rows")?;
            return indexer_history(store, advance, appended, domain, index);
        }
        let (mut visible, mut fresh_start) = accepted_history(store, advance, domain, index)?;
        // Policy comes from this target row's declared causal/media behavior,
        // never a window size or the shape of a drafter's block.
        if let Some(appended) = appends.causal[domain.0]
            .as_ref()
            .filter(|_| end(domain.0).is_none())
        {
            visible.extend(appended.causal_prior(fresh_start, index)?);
            fresh_start = index;
        }
        let destination = binding.destinations[domain.0]
            .get(index)
            .map_or(Ok(-1), |value| {
                i32::try_from(*value).map_err(|_| "target destination exceeds i32")
            })?;
        Ok(RowHistory {
            visible,
            fresh_start: i32::try_from(fresh_start).map_err(|_| "fresh start exceeds i32")?,
            bidirectional_end: end(domain.0)
                .map(|end| i32::try_from(end).map_err(|_| "media span end exceeds i32"))
                .transpose()?,
            destination,
        })
    });
    let stored_reads = store.history_domains().count();
    let shared = appends
        .shared
        .iter()
        .enumerate()
        .map(|(position, (domain, appended))| {
            shared_read_history(
                store,
                advance,
                *domain,
                appended,
                index,
                end(stored_reads + position),
            )
        });
    stored.chain(shared).collect()
}

/// What row `index` of `advance` reads of an indexer's `domain`
/// (`indexer_select`), every row as stored history, this launch's earlier
/// rows at their destinations: in its raw keys' Window domain the rows of
/// the `rows` positions before it, and its own destination; in its block
/// keys' Block domain every block complete through it (`advance_blocks`:
/// the one it completes too, which its launch writes before any row reads
/// it), and the destination of the block it completes (or -1). Its fresh
/// span is the row alone.
/// `appended` is the advance's runs in `domain` (`AppendedRuns::of`).
pub(crate) fn indexer_history(
    store: &StateStore,
    advance: &TentativeAdvance,
    appended: &AppendedRuns,
    domain: magnitude_state::HistoryDomainId,
    index: usize,
) -> Result<RowHistory, String> {
    let start = advance.position();
    let position = start
        .checked_add(index)
        .ok_or("target row position overflows")?;
    let destinations = &advance.bindings().destinations[domain.0];
    let (from, prior, destination) = match store.history_domain_kind(domain) {
        magnitude_state::HistoryDomainKind::Window { rows } => {
            let from = position.saturating_sub(rows);
            (from, from.saturating_sub(start)..index, destinations.get(index))
        }
        magnitude_state::HistoryDomainKind::Block { rate } => {
            let (read, completes) = advance_blocks(rate, start, position);
            (
                0,
                0..read,
                completes.and_then(|block| destinations.get(block)),
            )
        }
        kind => return Err(format!("an indexer in a {kind:?} history domain")),
    };
    let mut visible = advance
        .visible_ranges(domain, from)
        .into_iter()
        .map(|(start, count)| {
            Ok([
                i32::try_from(start).map_err(|_| "history start exceeds i32")?,
                i32::try_from(start.checked_add(count).ok_or("history end overflow")?)
                    .map_err(|_| "history end exceeds i32")?,
            ])
        })
        .collect::<Result<Vec<_>, String>>()?;
    visible.extend(appended.causal_prior(prior.start, prior.end)?);
    Ok(RowHistory {
        visible,
        fresh_start: i32::try_from(index).map_err(|_| "fresh start exceeds i32")?,
        bidirectional_end: None,
        destination: destination.map_or(Ok(-1), |value| {
            i32::try_from(*value).map_err(|_| "indexer destination exceeds i32")
        })?,
    })
}

/// What a row at `position` of an advance from `start` makes of a
/// Block(rate) domain: how many of the advance's appended blocks it reads,
/// the first that many, every block complete through the row (the
/// model's `(position + 1) / rate` visible blocks less the accepted ones),
/// and the block it completes, the last it reads, if any.
fn advance_blocks(rate: usize, start: usize, position: usize) -> (usize, Option<usize>) {
    let read = (position + 1) / rate - start / rate;
    (read, ((position + 1) % rate == 0).then(|| read - 1))
}

/// Target domains whose rows read their launch's earlier rows as stored
/// history even while those rows belong to the same launch: a compressed
/// history's, whose prior causal rows must use their stored representation,
/// and a selected read's, whose indexer selects among stored rows and whose
/// rows read only themselves fresh.
pub(crate) fn cached_causal_domain(
    store: &StateStore,
    domain: magnitude_state::HistoryDomainId,
) -> bool {
    store.history_components(domain).iter().any(|component| {
        component.causal_rows_use_cache()
            || matches!(component.layer, magnitude_state::LayerRef::Target(block)
                if store.index_key_domains(block, 0).is_some())
    })
}

/// The rows one advance appends to one stored history domain, in slot order,
/// as runs of consecutive rows within one slab: what `coalesce_within_slabs`
/// makes of the destinations taken one row at a time. Formed once per slot;
/// what a row reads of them is a clip of the runs, never a pass over the
/// destinations.
pub(crate) struct AppendedRuns {
    runs: Vec<AppendedRun>,
    /// The appended rows as ascending intervals `[first, last]`, no two of
    /// them overlapping or adjacent.
    intervals: Vec<[usize; 2]>,
    rows: usize,
    slab_rows: usize,
}

/// `rows` consecutive history rows from `first`, appended by the slot rows
/// from `offset`.
struct AppendedRun {
    offset: usize,
    first: usize,
    rows: usize,
}

/// The refusals of one reader of an advance's appended rows.
struct AppendedRefusals {
    outside: &'static str,
    exceeds: &'static str,
    overflows: &'static str,
}

impl AppendedRuns {
    pub(crate) fn new(destinations: &[usize], slab_rows: usize) -> Self {
        let mut runs: Vec<AppendedRun> = Vec::new();
        for (offset, &row) in destinations.iter().enumerate() {
            match runs.last_mut() {
                Some(run)
                    if run.first.checked_add(run.rows) == Some(row)
                        && run.first / slab_rows == row / slab_rows =>
                {
                    run.rows += 1
                }
                _ => runs.push(AppendedRun {
                    offset,
                    first: row,
                    rows: 1,
                }),
            }
        }
        let mut sorted = runs
            .iter()
            .map(|run| [run.first, run.first + (run.rows - 1)])
            .collect::<Vec<_>>();
        sorted.sort_unstable();
        let mut intervals: Vec<[usize; 2]> = Vec::with_capacity(sorted.len());
        for [first, last] in sorted {
            match intervals.last_mut() {
                Some(previous) if first <= previous[1].saturating_add(1) => {
                    previous[1] = previous[1].max(last)
                }
                _ => intervals.push([first, last]),
            }
        }
        Self {
            runs,
            intervals,
            rows: destinations.len(),
            slab_rows,
        }
    }

    /// The runs of the rows `advance` appends to `domain`.
    pub(crate) fn of(
        store: &StateStore,
        advance: &TentativeAdvance,
        domain: magnitude_state::HistoryDomainId,
    ) -> Self {
        Self::new(
            &advance.bindings().destinations[domain.0],
            store.history_slab_rows(domain),
        )
    }

    /// The runs of the slot rows `start..end`, each as its first history row
    /// and row count: the destinations of those slot rows coalesced within
    /// slabs. `None` when the slot rows are not rows of the advance.
    fn between(&self, start: usize, end: usize) -> Option<Vec<[usize; 2]>> {
        if start > end || end > self.rows {
            return None;
        }
        if start == end {
            return Some(Vec::new());
        }
        let from = self
            .runs
            .partition_point(|run| run.offset + run.rows <= start);
        Some(
            self.runs[from..]
                .iter()
                .take_while(|run| run.offset < end)
                .map(|run| {
                    let low = start.max(run.offset);
                    let high = end.min(run.offset + run.rows);
                    [run.first + (low - run.offset), high - low]
                })
                .collect(),
        )
    }

    /// `between` as packed `[start, end)` history ranges.
    fn ranges(
        &self,
        start: usize,
        end: usize,
        refusals: AppendedRefusals,
    ) -> Result<Vec<[i32; 2]>, String> {
        let ranges = self
            .between(start, end)
            .ok_or(refusals.outside)?
            .into_iter()
            .map(|[first, rows]| {
                let start = i32::try_from(first).map_err(|_| refusals.exceeds)?;
                let end = i32::try_from(rows)
                    .ok()
                    .and_then(|rows| start.checked_add(rows))
                    .ok_or(refusals.overflows)?;
                Ok([start, end])
            })
            .collect::<Result<Vec<_>, String>>()?;
        u32::try_from(self.slab_rows).map_err(|_| "slab rows exceed u32")?;
        Ok(ranges)
    }

    /// The history ranges a compressed causal row at slot row `end` reads of
    /// its own launch: the rows appended by the slot rows from `start` (its
    /// first fresh row) up to itself.
    pub(crate) fn causal_prior(&self, start: usize, end: usize) -> Result<Vec<[i32; 2]>, String> {
        self.ranges(
            start,
            end,
            AppendedRefusals {
                outside: "causal prior rows exceed destinations",
                exceeds: "causal prior destination exceeds i32",
                overflows: "causal prior range overflows",
            },
        )
    }

    /// The history ranges a Shared reader reads of the rows its source
    /// appends by the slot rows `start..end`. An advance that appends nothing
    /// to the domain has no batch rows in it.
    fn shared_appended(&self, start: usize, end: usize) -> Result<Vec<[i32; 2]>, String> {
        let (start, end) = if self.rows == 0 { (0, 0) } else { (start, end) };
        self.ranges(
            start,
            end,
            AppendedRefusals {
                outside: "shared history row exceeds the advance's destinations",
                exceeds: "history row exceeds i32",
                overflows: "history row exceeds i32",
            },
        )
    }

    /// Whether the advance appends every history row of `[start, end)` (an
    /// empty range has none to append).
    pub(crate) fn covers(&self, start: i32, end: i32) -> bool {
        if start >= end {
            return true;
        }
        let (Ok(first), Ok(last)) = (usize::try_from(start), usize::try_from(end - 1)) else {
            return false;
        };
        let after = self
            .intervals
            .partition_point(|interval| interval[0] <= first);
        after > 0 && last <= self.intervals[after - 1][1]
    }
}

/// The appended runs the rows of one advance read, formed once for its slot:
/// of each compressed causal stored domain (`cached_causal_domain`) and each
/// produced domain (`StateStore::is_produced_domain`), by domain, and of each
/// Shared source domain, in `HistoryReads::shared_sources` order.
pub(super) struct SlotAppends {
    causal: Vec<Option<AppendedRuns>>,
    shared: Vec<(magnitude_state::HistoryDomainId, AppendedRuns)>,
}

impl SlotAppends {
    pub(super) fn of(store: &StateStore, advance: &TentativeAdvance) -> Self {
        Self {
            // An indexer reads its launch's earlier rows as stored history
            // too (`indexer_history`).
            causal: store
                .history_domains()
                .map(|domain| {
                    (cached_causal_domain(store, domain) || store.is_produced_domain(domain))
                        .then(|| AppendedRuns::of(store, advance, domain))
                })
                .collect(),
            shared: store
                .history_reads()
                .shared_sources()
                .iter()
                .map(|&domain| (domain, AppendedRuns::of(store, advance, domain)))
                .collect(),
        }
    }
}

/// The history ranges a launch reserves per row for an advance of `rows`
/// rows over a state of `spans` accepted ranges: those ranges, and for a
/// store with Shared reads (`shared_read_history`) the ranges of the rows
/// the advance appends, at most one per slab they touch. A row reads those
/// of a cached-causal or produced domain when an earlier row of its launch
/// appended them, and a Block domain's row the blocks the advance completed
/// through it, the one it completes even alone (`advance_blocks`).
pub(super) fn reserved_segments(store: &StateStore, spans: usize, rows: usize) -> usize {
    // A Shared reader also sees the rows its source appends in the advance:
    // the rest of the source's last page, then whole pages (see
    // `magnitude_state::history_geometry`).
    spans
        + store
            .history_reads()
            .shared_sources()
            .iter()
            .copied()
            .chain(store.history_domains().filter(|&domain| {
                matches!(
                    store.history_domain_kind(domain),
                    magnitude_state::HistoryDomainKind::Block { .. }
                ) || (rows > 1
                    && (cached_causal_domain(store, domain) || store.is_produced_domain(domain)))
            }))
            .map(|domain| {
                let page = store.history_page_rows(domain);
                match store.history_domain_kind(domain) {
                    // An advance of `rows` rows completes at most
                    // rows.div_ceil(rate) blocks, appended as one run of
                    // block rows: the rest of a page, then whole pages.
                    magnitude_state::HistoryDomainKind::Block { rate } => {
                        match rows.div_ceil(rate) {
                            0 => 0,
                            blocks => 1 + (blocks - 1).div_ceil(page),
                        }
                    }
                    _ => rows.div_ceil(page) + 1,
                }
            })
            .max()
            .unwrap_or(0)
}

/// The accepted-history ranges row `index` of `advance` reads in stored
/// `domain` (from the query's window start), and the offset in the slot of
/// its first fresh row. The window start never exceeds the row's own
/// position.
pub(crate) fn accepted_history(
    store: &StateStore,
    advance: &TentativeAdvance,
    domain: magnitude_state::HistoryDomainId,
    index: usize,
) -> Result<(Vec<[i32; 2]>, usize), String> {
    let position = advance
        .position()
        .checked_add(index)
        .ok_or("target row position overflows")?;
    let from = store.history_domain_kind(domain).visible_from(position);
    let visible = advance
        .visible_ranges(domain, from)
        .into_iter()
        .map(|(start, count)| {
            Ok([
                i32::try_from(start).map_err(|_| "history start exceeds i32")?,
                i32::try_from(start.checked_add(count).ok_or("history end overflow")?)
                    .map_err(|_| "history end exceeds i32")?,
            ])
        })
        .collect::<Result<Vec<_>, String>>()?;
    Ok((visible, from.saturating_sub(advance.position())))
}

/// The history a Shared layer reads for row `index` of `advance` from its
/// source's stored `domain`. It appends nothing and projects no fresh keys,
/// so it reads the source's accepted rows and, as history, the rows the
/// source appended for the advance from the window start through the row
/// (through its media span's end `bidirectional_end` for a row attending its
/// span bidirectionally), each range within one slab; of a Block source,
/// the blocks the advance completed through the row (`advance_blocks`). Its
/// fresh span and destination are unused. `appended` is the advance's runs
/// in `domain`.
fn shared_read_history(
    store: &StateStore,
    advance: &TentativeAdvance,
    domain: magnitude_state::HistoryDomainId,
    appended: &AppendedRuns,
    index: usize,
    bidirectional_end: Option<usize>,
) -> Result<RowHistory, String> {
    let (mut visible, fresh_start) = accepted_history(store, advance, domain, index)?;
    let (start, end) = match store.history_domain_kind(domain) {
        magnitude_state::HistoryDomainKind::Block { rate } => {
            if bidirectional_end.is_some() {
                return Err(
                    "a Shared read of pooled rows attends no media span bidirectionally".into(),
                );
            }
            let position = advance
                .position()
                .checked_add(index)
                .ok_or("target row position overflows")?;
            (0, advance_blocks(rate, advance.position(), position).0)
        }
        _ => (fresh_start, bidirectional_end.unwrap_or(index + 1)),
    };
    visible.extend(appended.shared_appended(start, end)?);
    Ok(RowHistory {
        visible,
        fresh_start: i32::try_from(index).map_err(|_| "fresh start exceeds i32")?,
        bidirectional_end: None,
        destination: -1,
    })
}

fn row_demand(operation: &Operation, row: usize) -> crate::batching::Demand {
    let Operation::Forward {
        tokens,
        kind,
        demand,
        ..
    } = operation
    else {
        return crate::batching::Demand::NONE;
    };
    match kind {
        WorkKind::Decode | WorkKind::Verify => *demand,
        WorkKind::Prefill | WorkKind::Replay if row + 1 == tokens.len() => *demand,
        WorkKind::Prefill | WorkKind::Replay => {
            *demand & (crate::batching::Demand::FEATURES | crate::batching::Demand::TAPS)
        }
    }
}

fn selection(operation: &Operation, row: usize) -> Option<Select> {
    operation.selection_for_row(row).map(select_row)
}

/// The packed selection controls of one selection spec.
pub(super) fn select_row(spec: &crate::SelectSpec) -> Select {
    Select {
        draw: Draw {
            kind: match spec.sampling {
                crate::Sampling::Greedy => DrawKind::Greedy,
                crate::Sampling::Categorical => DrawKind::Categorical,
            },
            seed: spec.seed,
            position: spec.position as u64,
            domain: spec.domain,
        },
        mask: spec.mask.clone(),
        shaping: RowShaping {
            temperature: spec.shaping.temperature,
            top_p: spec.shaping.top_p,
            top_k: spec.shaping.top_k,
            min_p: spec.shaping.min_p,
            repetition_penalty: spec.shaping.repetition_penalty,
            presence_penalty: spec.shaping.presence_penalty,
            frequency_penalty: spec.shaping.frequency_penalty,
            flags: 0,
        },
        history: spec
            .history
            .as_ref()
            .map_or_else(Vec::new, |value| value.to_vec()),
    }
}

#[cfg(test)]
mod causal_cache_tests {
    use super::*;
    use magnitude_state::{
        BankCapacity, ComponentDescriptor, HistoryDomainLayout, HistoryDomainPlan, KvCodec,
        LayerRef, OwnedStateAdvance, RowDemand,
    };

    #[test]
    fn compressed_causal_rows_read_prepared_prior_but_media_keeps_dense_block() {
        let catalog = seismic::Devices::discover(&[]).expect("devices are discovered");
        let device = crate::test_support::open_backend(&catalog, seismic::BackendName::Cpu).expect("the CPU backend opens");
        let device = Rc::new(device);
        for codec in [KvCodec::Dense, KvCodec::AffineK8V4] {
            let (store, mut bindings) = StateStore::new(
                Rc::clone(&device),
                16,
                16,
                vec![HistoryDomainPlan {
                    layout: HistoryDomainLayout::Token {
                        components: vec![
                            ComponentDescriptor::new(
                                LayerRef::Target(0),
                                codec.spec(seismic::DType::F32, 64, 64),
                                1,
                            )
                            .unwrap(),
                        ],
                    },
                    logical_rows: 16,
                }],
                vec![],
                BankCapacity {
                    active: 1,
                    in_flight: 1,
                    retained: 0,
                },
            )
            .unwrap();
            let domain = store.sole_history_domain().unwrap();
            bindings
                .provision(&[RowDemand { domain, rows: 3 }], 0)
                .unwrap();
            let advance = TentativeAdvance::Accepted(
                OwnedStateAdvance::begin(store.create().unwrap(), 3)
                    .ok()
                    .unwrap(),
            );
            let appends = SlotAppends::of(&store, &advance);
            let causal = row_histories(&store, &advance, &appends, 1, None, &[true]).unwrap();
            let media = row_histories(&store, &advance, &appends, 1, Some(3), &[true]).unwrap();
            assert_eq!(media[0].fresh_start, 0);
            assert_eq!(media[0].bidirectional_end, Some(3));
            assert!(media[0].visible.is_empty());
            if codec == KvCodec::AffineK8V4 {
                let prior = advance.bindings().destinations[domain.0][0] as i32;
                assert_eq!(causal[0].visible, vec![[prior, prior + 1]]);
                assert_eq!(causal[0].fresh_start, 1);
            } else {
                assert!(causal[0].visible.is_empty());
                assert_eq!(causal[0].fresh_start, 0);
            }
            assert_eq!(causal[0].bidirectional_end, None);
            let first = row_histories(&store, &advance, &appends, 0, None, &[false]).unwrap();
            assert!(first[0].visible.is_empty());
            assert_eq!(first[0].fresh_start, 0);
        }
    }
}

#[cfg(test)]
mod appended_runs_tests {
    use super::super::draft::coalesce_within_slabs;
    use super::AppendedRuns;

    /// Destination lists with slab crossings, gaps, returns to earlier rows
    /// and a repeated row, each with the slab sizes it is read under.
    fn cases() -> Vec<(Vec<usize>, Vec<usize>)> {
        let mut generated = Vec::new();
        let mut state = 0x9e37_79b9_u32;
        let mut next = |bound: u32| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (state >> 16) % bound
        };
        for _ in 0..24 {
            let mut row = next(40) as usize;
            let mut destinations = Vec::new();
            for _ in 0..next(48) {
                destinations.push(row);
                row = match next(8) {
                    0 => row + 2 + next(9) as usize,
                    1 => row.saturating_sub(next(12) as usize),
                    _ => row + 1,
                };
            }
            generated.push(destinations);
        }
        let mut cases = vec![
            (Vec::new(), vec![1, 4]),
            (vec![7], vec![1, 4, 8]),
            ((0..40).collect(), vec![1, 4, 16, 64]),
            ((13..77).collect(), vec![8, 16, 32]),
            (vec![3, 4, 5, 9, 10, 16, 17, 18, 2, 3, 4], vec![1, 4, 8]),
            (vec![5, 5, 6, 7, 7, 8], vec![4, 16]),
        ];
        cases.extend(
            generated
                .into_iter()
                .map(|destinations| (destinations, vec![1, 3, 4, 16, 512])),
        );
        cases
    }

    #[test]
    fn clipped_runs_equal_the_coalesced_destinations_of_every_slot_range() {
        for (destinations, slabs) in cases() {
            for slab_rows in slabs {
                let runs = AppendedRuns::new(&destinations, slab_rows);
                for start in 0..=destinations.len() {
                    for end in start..=destinations.len() {
                        let single = destinations[start..end]
                            .iter()
                            .map(|&row| [row as i32, row as i32 + 1])
                            .collect::<Vec<_>>();
                        let expected = coalesce_within_slabs(&single, slab_rows as u32);
                        assert_eq!(
                            runs.causal_prior(start, end).unwrap(),
                            expected,
                            "{destinations:?} slab {slab_rows} rows {start}..{end}"
                        );
                        assert_eq!(runs.shared_appended(start, end).unwrap(), expected);
                    }
                }
            }
        }
    }

    #[test]
    fn slot_rows_outside_the_advance_are_refused_as_before() {
        let runs = AppendedRuns::new(&[4, 5, 6], 4);
        assert_eq!(
            runs.causal_prior(0, 4).unwrap_err(),
            "causal prior rows exceed destinations"
        );
        assert_eq!(
            runs.causal_prior(2, 1).unwrap_err(),
            "causal prior rows exceed destinations"
        );
        assert_eq!(
            runs.shared_appended(1, 4).unwrap_err(),
            "shared history row exceeds the advance's destinations"
        );
        // An advance that appends nothing to a Shared source has no batch
        // rows in it, whatever the row asks for.
        let empty = AppendedRuns::new(&[], 4);
        assert!(empty.shared_appended(2, 7).unwrap().is_empty());
        assert_eq!(
            empty.causal_prior(0, 1).unwrap_err(),
            "causal prior rows exceed destinations"
        );
        let wide = AppendedRuns::new(&[i32::MAX as usize - 1, i32::MAX as usize], 1 << 20);
        assert_eq!(
            wide.causal_prior(0, 2).unwrap_err(),
            "causal prior range overflows"
        );
        let beyond = AppendedRuns::new(&[i32::MAX as usize + 1], 1 << 20);
        assert_eq!(
            beyond.causal_prior(0, 1).unwrap_err(),
            "causal prior destination exceeds i32"
        );
        assert_eq!(
            beyond.shared_appended(0, 1).unwrap_err(),
            "history row exceeds i32"
        );
    }

    #[test]
    fn covered_ranges_are_exactly_those_whose_every_row_is_appended() {
        for (destinations, slabs) in cases() {
            let runs = AppendedRuns::new(&destinations, slabs[0]);
            let limit = destinations.iter().max().map_or(4, |row| *row as i32 + 4);
            for start in -2..limit {
                for end in -2..limit {
                    let every_row_appended = (start..end).all(|row| {
                        usize::try_from(row).is_ok_and(|row| destinations.contains(&row))
                    });
                    assert_eq!(
                        runs.covers(start, end),
                        every_row_appended,
                        "{destinations:?} range {start}..{end}"
                    );
                }
            }
        }
    }
}

/// The chain an interrupted chunk releases, with its holder's state: one
/// still held behind queued chunks (`release_chain`), or one its last chunk
/// completed, whose state is the holder's again.
enum Released {
    Held,
    Whole(RequestId),
}
