//! Target submission, completion, and row construction.

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
    /// claim the queued lookahead step it equals. A continuable group queues
    /// its own lookahead behind it, in the bindings the flight holds until
    /// `finish_target` returns them.
    pub fn submit_target(
        &mut self,
        mut bindings: StateBindings<F>,
        operations: &[Operation],
        reservation: TargetGraphReservation,
    ) -> Result<TargetFlight<F>, SubmitFailure<F>> {
        let work = match reservation {
            TargetGraphReservation::Claim(slots) => self
                .claim_lookahead(&mut bindings, operations, slots)
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
        self.queue_lookahead(&mut bindings, &work, operations)
            .map_err(SubmitFailure::Failed)?;
        Ok(TargetFlight { work, bindings })
    }

    /// Launch the group's target step, then a primed prompt chunk's drafter
    /// entry behind it.
    fn launch_target(
        &mut self,
        operations: &[Operation],
        mut reservation: TargetLaunchReservation,
    ) -> Result<TargetWork<F>, Unsubmitted> {
        let priming = reservation.priming.take();
        // Submitted or not, the launch's readout values are Seismic's charge
        // once its graph launch returns.
        let readout_claim = reservation.readout_claim.take();
        let launched = self.launch_target_graph(operations, reservation);
        drop(readout_claim);
        let work = match launched {
            Ok(work) => work,
            Err(failure) => {
                if let (Some(priming), [operation]) = (priming, operations) {
                    self.head
                        .insert(operation.request(), priming.advance.abort());
                }
                return Err(failure);
            }
        };
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

    fn launch_target_graph(
        &mut self,
        operations: &[Operation],
        reservation: TargetLaunchReservation,
    ) -> Result<TargetWork<F>, Unsubmitted> {
        let TargetLaunchReservation {
            advances,
            graph_workspace,
            graph_outputs,
            readout_workspace,
            readout_output,
            priming: _,
            boundary,
            barriers,
            readout_claim: _,
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
        for (operation, advance) in operations.iter().zip(&advances) {
            operation
                .validate()
                .map_err(|error| broken(error.to_string()))?;
            let Operation::Forward {
                request,
                position,
                tokens,
                conditioning,
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
        }
        let group = operations
            .iter()
            .zip(&advances)
            .map(|(operation, advance)| (operation, advance as &dyn AcceptedHistory))
            .collect::<Vec<_>>();
        let class = self.target_class(&group).map_err(broken)?;
        let class_limits = self.target_class_limits();
        let preceding = self.target_preceding(operations).map_err(broken)?;
        let visibility = self.launch_visibility(operations).map_err(broken)?;
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
        // Each operation's slots, one per launch chunk: one launch computes
        // every operation whole, or a lone operation in chunks.
        let prepared = operations
            .iter()
            .zip(&advances)
            .map(|(operation, advance)| {
                let chunks = self.target_chunks(operation.row_count());
                let count = chunks.len();
                chunks
                    .into_iter()
                    .enumerate()
                    .map(|(chunk, rows)| {
                        self.target_slot(operation, advance, rows.clone(), (chunk, count))
                            .map(|(slot, slices)| (slot, slices, rows))
                    })
                    .collect::<Result<Vec<_>, _>>()
            })
            .collect::<Result<Vec<_>, _>>();
        let prepared = match prepared {
            Ok(prepared) => prepared,
            Err(error) => {
                self.restore_advances(metadata, advances);
                return Err(Unsubmitted::Refused(error.into()));
            }
        };
        let vocabulary = self.definition.decoder.vocabulary as usize;
        // Every batch packs in its class rows (`class`'s, or a layer-major
        // chunk's own), and `class`'s segments, which cover at most the span
        // limit.
        let segments = class.segments().min(class_limits.segments);
        let batch = |slots: &[Slot], rows: usize| {
            match ValidatedTargetBatch::covering(slots, vocabulary, class_limits, rows, segments) {
                Ok(batch)
                    if batch.class().rows() == rows
                        && batch.class().segments() == class.segments() =>
                {
                    Ok(batch)
                }
                Ok(_) => Err(broken("target packed class changed after reservation".into())),
                Err(error) => Err(Unsubmitted::Refused(error.to_string().into())),
            }
        };
        let layered = matches!(prepared.as_slice(), [chunks] if chunks.len() > 1);
        let (rows, conditioning_slices) = match layered {
            true => {
                // Every chunk reads its rows' part of the advance's slices
                // (`ConditioningSlice::within`), in its own class.
                let slices = prepared[0][0].1.clone();
                let built = prepared[0]
                    .iter()
                    .map(|(slot, _, rows)| {
                        self.chunk_class_rows(&operations[0], rows)
                            .map_err(broken)
                            .and_then(|class_rows| batch(std::slice::from_ref(slot), class_rows))
                            .map(|batch| crate::LaunchChunk {
                                batch,
                                rows: rows.clone(),
                            })
                    })
                    .collect::<Result<Vec<_>, _>>();
                match built {
                    Ok(chunks) => (
                        crate::LaunchRows::LayerMajor(crate::LayerMajor { chunks, barriers }),
                        vec![slices],
                    ),
                    Err(error) => {
                        self.restore_advances(metadata, advances);
                        return Err(error);
                    }
                }
            }
            false => {
                let (slots, slices): (Vec<_>, Vec<_>) = prepared
                    .into_iter()
                    .flatten()
                    .map(|(slot, slices, _)| (slot, slices))
                    .unzip();
                match batch(&slots, class.rows()) {
                    Ok(batch) if batch.class() == class => (crate::LaunchRows::Batch(batch), slices),
                    Ok(_) => {
                        self.restore_advances(metadata, advances);
                        return Err(broken(
                            "target packed class changed after reservation".into(),
                        ));
                    }
                    Err(error) => {
                        self.restore_advances(metadata, advances);
                        return Err(error);
                    }
                }
            }
        };
        let conditioning = metadata
            .iter()
            .map(|(_, _, lease, _, _)| lease.clone())
            .collect();
        let inputs = TargetLaunchInputs::new(
            rows,
            TargetTokens::Host,
            advances,
            conditioning,
            conditioning_slices,
            graph_workspace,
            graph_outputs,
            readout_workspace,
            readout_output,
        )
        .with_boundary(boundary)
        .with_preceding(preceding)
        .with_visibility(visibility)
        .with_stores(
            operations
                .iter()
                .map(|operation| {
                    self.retained_rows(operation.request())
                        .0
                        .iter()
                        .map(|store| store.features().clone())
                        .collect()
                })
                .collect(),
        );
        let launch = match ValidatedTargetLaunch::new(
            inputs,
            &self.target_store,
            self.domain.id(),
            self.definition.decoder.hidden as usize,
            self.definition.head.as_ref().map_or(self.definition.decoder.hidden, |head| head.conditioning_width(&self.definition.decoder)) as usize,
        ) {
            Ok(launch) => launch,
            Err((inputs, error)) => {
                let (_, advances, _, _) = inputs.into_parts();
                self.restore_advances(metadata, advances);
                return Err(Unsubmitted::Failed(DomainError::Invariant(error)));
            }
        };
        let launch_trace = self.trace_target_submission(operations);
        let started = Instant::now();
        let recorded_graphs = self.family.recorded_graphs();
        let submitted = self.family.submit_target(launch);
        let submission = match submitted {
            Ok(submission) => submission,
            Err((error, launch)) => {
                let (core, _) = launch.into_submission_parts();
                let (_, advances, _, _) = core.into_parts();
                self.restore_advances(metadata, advances);
                return Err(Unsubmitted::Failed(error.into()));
            }
        };
        let requests = metadata.iter().map(|(request, ..)| *request).collect::<Vec<_>>();
        let runnable = self.service_start(&requests, started);
        Ok(TargetWork {
            requests: metadata,
            submission,
            priming: None,
            runnable,
            launch_trace,
            started,
            recorded_graphs,
            previous_selection: self.selection_read.take(),
            id: self.flight_id(),
            continuation: None,
        })
    }

    /// Completion exposes immutable outcomes while every owned advance stays
    /// suspended. The caller can stage grammar and method changes from these
    /// views before choosing physical prefixes. Returns the bindings the
    /// flight held, with the lookahead it queued; an error consumes them, as
    /// the domain cannot continue.
    /// Per operation of a target group, the accepted tokens its sequence
    /// holds before the operation's first row, as many as the model's row
    /// rules read (`TargetLaunchCore::preceding`); none of a model without
    /// one.
    pub(super) fn target_preceding(&self, operations: &[Operation]) -> Result<Vec<Vec<u64>>, String> {
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
                Ok(self
                    .preceding_tokens(*request, *position, context)?
                    .iter()
                    .map(|token| u64::from(token.0))
                    .collect())
            })
            .collect()
    }

    pub fn finish_target(
        &mut self,
        flight: TargetFlight<F>,
    ) -> Result<(Vec<PendingOperationOutcome>, StateBindings<F>), DomainError> {
        let TargetFlight { work, mut bindings } = flight;
        let pending = self.finish_target_work(&mut bindings, work)?;
        Ok((pending, bindings))
    }

    fn finish_target_work(
        &mut self,
        bindings: &mut StateBindings<F>,
        flight: TargetWork<F>,
    ) -> Result<Vec<PendingOperationOutcome>, DomainError> {
        let completed = match flight.submission.finish() {
            Ok(completed) => completed,
            Err(error) => {
                return Err(DomainError::Device(error));
            }
        };
        // The drafter entry follows its chunk on the device; its advance
        // commits with the chunk. A claimed lookahead's entry attaches to the
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
        let replayed = self.family.recorded_graphs() == flight.recorded_graphs;
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
        let (core, mut output) = completed.into_parts();
        // A sequence whose stored-table rows could not be read in this step
        // read other rows: it is poisoned, and this outcome and every later
        // one of it fail. The step's other sequences are unaffected. A
        // landed read that could not land its rows fails every sequence.
        let landing = output.complete_landed_reads();
        for (slot, (request, ..)) in flight.requests.iter().enumerate() {
            if let Some(error) = output.table_read_failure(slot).or_else(|| landing.clone()) {
                self.poisoned.entry(*request).or_insert(error);
            }
        }
        output.retire_stored_reads();
        let crate::TargetOutput {
            readout: output,
            readout_values,
            commits,
            boundary,
            landed_reads: _,
            stored_reads: _,
        } = output;
        // A layer-major launch's rows read nothing but readout values
        // (`LayerMajor`); a lone batch's rows read any output.
        let launch = core.rows();
        let mut physical = vec![RowResult::default(); launch.actual_rows()];
        let mut selected = Vec::new();
        let slots = match launch {
            crate::LaunchRows::Batch(batch) => batch.actual_slots(),
            crate::LaunchRows::LayerMajor(_) => 1,
        };
        if slots != flight.requests.len() {
            return Err("completed target slot count differs from submitted requests".into());
        }
        if let crate::LaunchRows::Batch(batch) = launch {
            let upload = batch.upload();
            if !upload.select_rows.is_empty() {
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
                selected = decode_selected(&bytes)?;
            }
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
        }
        // Each batch's readouts' values, by their index in its readouts: a
        // logits readout's tokens' columns of its group's projection onto
        // their union, a head readout's row of the head's products, a scores
        // readout's scores of its runs of keys in order; its rows are the
        // launch's from the batch's first.
        for ((batch, offset), readout_values) in launch.batches().into_iter().zip(readout_values) {
            let upload = batch.upload();
            let mut values = vec![None::<Vec<f32>>; upload.readouts.len()];
            for group in readout_values.logits {
                let projected = host_f32(&group.values)?;
                let width = *group
                    .values
                    .extents()
                    .get(1)
                    .ok_or("readout values are not a matrix")? as usize;
                for (output, (readout, tokens)) in group.rows.iter().enumerate() {
                    let row_values = tokens
                        .iter()
                        .map(|token| {
                            let column = group
                                .columns
                                .binary_search(token)
                                .map_err(|_| "a readout token is absent from its projection")?;
                            projected
                                .get(output * width + column)
                                .copied()
                                .ok_or("readout values are shorter than their rows")
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    *values.get_mut(*readout).ok_or("readout index exceeds the batch's")? =
                        Some(row_values);
                }
            }
            if let Some(head) = readout_values.head {
                let products = host_f32(&head.values)?;
                let width = *head
                    .values
                    .extents()
                    .get(1)
                    .ok_or("readout head values are not a matrix")? as usize;
                for (output, readout) in head.readouts.iter().enumerate() {
                    let row = products
                        .get(output * width..(output + 1) * width)
                        .ok_or("readout head values are shorter than their rows")?;
                    *values.get_mut(*readout).ok_or("readout index exceeds the batch's")? =
                        Some(row.iter().map(|value| head.element.publish(*value)).collect());
                }
            }
            for scores in readout_values.scores {
                let mut keys = Vec::new();
                for run in &scores.runs {
                    keys.extend(host_f32(run)?);
                }
                *values
                    .get_mut(scores.readout)
                    .ok_or("readout index exceeds the batch's")? = Some(keys);
            }
            // A retention returns nothing; every other readout its values.
            for ((&output, readout), values) in upload
                .readout_rows
                .iter()
                .zip(upload.readouts)
                .zip(values)
            {
                if matches!(readout, crate::batching::RowReadout::Retain { .. }) {
                    continue;
                }
                let row = usize::try_from(output)
                    .ok()
                    .and_then(|output| upload.out_rows.get(output))
                    .and_then(|row| usize::try_from(*row).ok())
                    .ok_or("readout output row exceeds output rows")?;
                physical
                    .get_mut(offset + row)
                    .ok_or("readout row exceeds actual batch")?
                    .readouts
                    .push(values.ok_or("a readout returned no values")?);
            }
        }
        let (_, advances, _, _) = core.into_parts();
        let mut pending = Vec::with_capacity(flight.requests.len());
        let mut offset = 0usize;
        let mut continuation = flight.continuation;
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
                (TentativeAdvance::Accepted(advance), None) => advance,
                (TentativeAdvance::Successor(successor), Some(Some(state))) => {
                    match successor.attach(state.into_state()) {
                        Ok(advance) => advance,
                        Err((state, error)) => {
                            self.target.insert(request, state);
                            return Err(DomainError::invariant(format!(
                                "claimed lookahead slot no longer follows its state: {error}"
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
            let outcome = match self.poisoned.get(&request) {
                Some(error) => Outcome::TableReadFailed(error.clone()),
                None => Outcome::Forward { rows: result },
            };
            pending.push(PendingOperationOutcome {
                request,
                outcome,
                advance: Some(advance),
                primed: primed
                    .take_if(|(primed, _)| *primed == request)
                    .map(|(_, advance)| advance),
                rows,
                committed_rows,
                kind,
                physical_duration,
                replayed,
                image: None,
            });
        }
        if primed.is_some() {
            return Err(DomainError::invariant(
                "a drafter entry's chunk is absent from its flight",
            ));
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
        bindings.predecessor_selected(flight.id, &selected, finished);
        self.target_timing = Some(timing);
        self.begin_host_turn(pending.iter().map(|item| item.request));
        Ok(pending)
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
    ) -> Result<(RequestId, Duration, StateBindings<F>), DomainError> {
        let request = flight.work.request;
        if let Some((_, advance)) = self.finish_priming_work(flight.work)? {
            if self.head.contains_key(&request) {
                return Err(DomainError::invariant(
                    "deferred entry has another head owner",
                ));
            }
            let accepted = self
                .target
                .get(&request)
                .is_some_and(|state| state.position() == flight.target_end);
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
                // A successor's source state stays with its predecessor;
                // dropping it releases its rows and bank.
                TentativeAdvance::Successor(successor) => drop(successor),
            }
        }
    }

    /// How each history read of the target store reads (`ReadForms`): its
    /// layers attend media rows bidirectionally (`operators::attention::admit_media`),
    /// and it is global.
    pub(super) fn read_forms(&self) -> Result<ReadForms, String> {
        let store = &self.target_store;
        let mut bidirectional = vec![false; store.history_reads()];
        for (index, sublayer) in self.definition.decoder.sublayers() {
            let magnitude_family_contracts::Operator::Attention(attention) = &sublayer.op else {
                continue;
            };
            if attention.media_rows == magnitude_family_contracts::MediaRowAttention::Bidirectional
            {
                let read = store
                    .history_read(magnitude_state::LayerRef::Target(index.block))
                    .ok_or("an attention layer has no history read")?;
                bidirectional[read] = true;
            }
        }
        let global = (0..store.history_reads())
            .map(|read| global_read(store, read))
            .collect();
        Ok(ReadForms {
            bidirectional,
            global,
        })
    }

    /// The history ranges a launch computing rows `computed` of `operation`
    /// (`target_chunks`) reserves per row over a state of `spans` accepted
    /// ranges (`reserved_segments`); in a chunk after the first, also the
    /// destinations of the rows earlier chunks computed that a causal row
    /// reads, at most one range per page they touch and one more; and for a
    /// blocked input's rows what `blocked_reads` presents on each global
    /// read: the accepted ranges holding the positions it reads (the state's
    /// `accepted(domain, from)`, its ranges from position `from`), and the
    /// destinations of its advance's rows it reads outside its fresh span,
    /// at most one range per page each run of them touches and one more.
    pub(super) fn target_segments(
        &self,
        operation: &Operation,
        computed: &std::ops::Range<usize>,
        spans: usize,
        accepted: &dyn Fn(magnitude_state::HistoryDomainId, usize) -> Vec<(usize, usize)>,
    ) -> Result<usize, String> {
        let Operation::Forward {
            request,
            position,
            tokens,
            ..
        } = operation
        else {
            return Err("non-forward target operation".into());
        };
        let store = &self.target_store;
        let visibility = &self.installed_input(*request)?.visibility;
        let forms = self.read_forms()?;
        let blocked = !visibility.is_causal();
        let prior = store
            .history_domains()
            .filter(|&domain| {
                computed.start > 0
                    && !cached_causal_domain(store, domain)
                    && !indexer_domain(store, domain)
                    && !(blocked && forms.global[domain.0])
            })
            .map(|domain| {
                let rows = match store.history_domain_kind(domain) {
                    magnitude_state::HistoryDomainKind::Window { rows } => rows.min(computed.start),
                    _ => computed.start,
                };
                rows.div_ceil(store.history_page_rows(domain)) + 1
            })
            .max()
            .unwrap_or(0);
        let reserved = reserved_segments(store, spans, tokens.len()) + prior;
        if !blocked {
            return Ok(reserved);
        }
        let reads_of = store
            .history_domains()
            .chain(store.shared_source_domains())
            .enumerate()
            .filter(|(read, _)| forms.global[*read])
            .map(|(read, domain)| (domain, fresh_span(store, read)))
            .collect::<Vec<_>>();
        let start = *position;
        let mut held = BTreeMap::new();
        let mut segments = reserved;
        for row in start + computed.start..start + computed.end {
            let (reads, block) = visibility
                .reads(row)
                .zip(visibility.block(row))
                .ok_or("a blocked row lies outside its input")?;
            for &(domain, span) in &reads_of {
                let key = (
                    domain.0,
                    reads.iter().map(|read| (read.start, read.end)).collect::<Vec<_>>(),
                );
                let history = match held.get(&key) {
                    Some(&history) => history,
                    None => {
                        let history = reads
                            .iter()
                            .filter(|read| read.start < start)
                            .map(|read| {
                                let mut remaining = read.end.min(start) - read.start;
                                accepted(domain, read.start)
                                    .into_iter()
                                    .take_while(|(_, count)| {
                                        let taken = remaining > 0;
                                        remaining = remaining.saturating_sub(*count);
                                        taken
                                    })
                                    .count()
                            })
                            .sum::<usize>();
                        held.insert(key, history);
                        history
                    }
                };
                let page_rows = store.history_page_rows(domain);
                let index = row - start;
                let launch = reads
                    .iter()
                    .filter(|read| read.end > start)
                    .map(|read| read.start.max(start) - start..read.end - start)
                    .map(|run| {
                        let own = if run.contains(&index) {
                            within(
                                fresh_rows(span, &run, index, block.start - start..block.end - start),
                                computed,
                            )
                        } else {
                            index..index
                        };
                        [run.start..own.start.min(run.end), own.end.max(run.start)..run.end]
                            .into_iter()
                            .filter(|part| !part.is_empty())
                            .map(|part| part.len().div_ceil(page_rows) + 1)
                            .sum::<usize>()
                    })
                    .sum::<usize>();
                segments = segments.max(history + launch);
            }
        }
        Ok(segments)
    }

    /// The fewest rows of the class of a launch computing `operation`'s rows
    /// `computed`. A blocked row reading rows of its launch apart
    /// (`Visibility::reads_apart`) reads some of them as stored history,
    /// so its attention must publish every row before any row attends: a
    /// prefill class's attention always does; over compressed histories
    /// (`cached_causal_domain`), whose rows read their launch's rows outside
    /// their block as stored, a decode class's does too past one row.
    pub(super) fn launch_floor(
        &self,
        operation: &Operation,
        computed: &std::ops::Range<usize>,
    ) -> Result<usize, String> {
        let Operation::Forward { request, position, .. } = operation else {
            return Err("non-forward target operation".into());
        };
        let visibility = &self.installed_input(*request)?.visibility;
        if !visibility.reads_apart(position + computed.start..position + computed.end) {
            return Ok(1);
        }
        let store = &self.target_store;
        let dense = store
            .history_domains()
            .any(|domain| global_read(store, domain.0) && !cached_causal_domain(store, domain));
        Ok(if dense {
            crate::operators::attention::graph::DECODE_ROWS as usize + 1
        } else {
            2
        })
    }

    /// The launch class of a target group over each operation's accepted
    /// state: it covers every launch chunk's rows (`target_chunks`), the
    /// history ranges they read (`target_segments`) and the rows their
    /// attention needs (`launch_floor`). An operation of several chunks is a
    /// committed prompt chunk alone in its launch, reading nothing but
    /// readout values, in a load that computes it (`computes_advance`).
    pub(super) fn target_class(
        &self,
        group: &[(&Operation, &dyn AcceptedHistory)],
    ) -> Result<crate::LaunchClass, String> {
        let mut rows = 0usize;
        let mut floor = 1usize;
        let mut segments = 1usize;
        let mut demand = crate::batching::Demand::NONE;
        for &(operation, state) in group {
            let chunks = self.target_chunks(operation.row_count());
            if let [_, _, ..] = chunks.as_slice() {
                let Operation::Forward {
                    kind: WorkKind::Prefill,
                    conditioning: None,
                    select,
                    prime: None,
                    demand,
                    committed,
                    tokens,
                    ..
                } = operation
                else {
                    return Err(
                        "an advance computed in chunks is an unconditioned prompt chunk".into(),
                    );
                };
                if group.len() != 1
                    || !select.is_empty()
                    || *committed != tokens.len()
                    || *demand != *demand & crate::batching::Demand::READOUT
                    || self.computes_advance(tokens.len()).is_err()
                {
                    return Err(
                        "an advance computed in chunks is a committed prompt chunk alone in its \
                         launch, reading only readout values, of a load that computes it"
                            .into(),
                    );
                }
            }
            let widest = chunks.iter().map(|chunk| chunk.len()).max().unwrap_or(0);
            rows = rows.checked_add(widest).ok_or("target row count overflow")?;
            for chunk in &chunks {
                segments = segments.max(self.target_segments(
                    operation,
                    chunk,
                    state.span_count(),
                    &|domain, from| state.visible_ranges(domain, from),
                )?);
                floor = floor.max(self.launch_floor(operation, chunk)?);
            }
            demand |= self.launch_demand(operation);
        }
        crate::LaunchClass::covering(rows.max(floor), segments, demand, self.target_class_limits())
            .map_err(|error| error.to_string())
    }

    /// The rows of an advance of `rows` rows each chunk of its launch
    /// computes: the whole advance when a launch class holds it, otherwise
    /// consecutive chunks of the largest launch's rows, the last holding
    /// the rest, computed layer-major (`LayerMajor`). Chunks are cut where
    /// full launches of the same rows would be, so recurrent layers
    /// partition their rows alike.
    pub(super) fn target_chunks(&self, rows: usize) -> Vec<std::ops::Range<usize>> {
        let largest = self
            .target_class_limits()
            .rows
            .min(crate::batching::MAX_CLASS_ROWS);
        (0..rows.div_ceil(largest).max(1))
            .map(|chunk| chunk * largest..((chunk + 1) * largest).min(rows))
            .collect()
    }

    /// The class rows chunk `computed` of `operation`'s layer-major advance
    /// launches in: those of its own rows, the rows its attention needs
    /// (`launch_floor`) and the least a layer-major chunk takes
    /// (`ProgramFamily::layer_major_rows`), so every chunk fires the
    /// launches the first does. The first chunk is the widest, in the
    /// launch's class (`target_class`); a narrower last one computes no
    /// more than that floor's padding.
    pub(super) fn chunk_class_rows(
        &self,
        operation: &Operation,
        computed: &std::ops::Range<usize>,
    ) -> Result<usize, String> {
        let rows = computed
            .len()
            .max(self.launch_floor(operation, computed)?)
            .max(self.family.layer_major_rows());
        crate::batching::row_class(rows).ok_or_else(|| "a launch chunk has no row class".into())
    }

    /// Whether this load computes an advance of `rows` rows of a prepared
    /// prompt, else what an advance wider than one launch needs: in one
    /// launch, or layer-major in chunks, which needs a model computing
    /// layer-major (`computes_layer_major`).
    pub fn computes_advance(&self, rows: usize) -> Result<(), &'static str> {
        if self.target_chunks(rows).len() == 1 {
            return Ok(());
        }
        crate::planning::computes_layer_major(&self.definition)
    }

    /// The blocks whose history every chunk of a layer-major advance of
    /// `request` publishes before any chunk's rows attend there
    /// (`Barriers`): for a blocked input, whose rows read later chunks'
    /// rows on global attention, the blocks attending a global stored
    /// history; none for a causal one.
    pub(super) fn layer_barriers(&self, request: RequestId) -> Result<Vec<usize>, String> {
        if self.installed_input(request)?.visibility.is_causal() {
            return Ok(Vec::new());
        }
        let store = &self.target_store;
        let mut barriers = std::collections::BTreeSet::new();
        for (index, sublayer) in self.definition.decoder.sublayers() {
            if !matches!(sublayer.op, magnitude_family_contracts::Operator::Attention(_)) {
                continue;
            }
            let read = store
                .history_read(magnitude_state::LayerRef::Target(index.block))
                .ok_or("an attention layer has no history read")?;
            if read < store.history_domains().count() && global_read(store, read) {
                barriers.insert(index.block as usize);
            }
        }
        Ok(barriers.into_iter().collect())
    }

    /// The f32 elements a layer-major launch of `chunks` chunks in `class`
    /// holds between stretches (`Barriers::held`): every chunk's class rows
    /// of the widest input of a barrier block.
    pub(super) fn held_elements(
        &self,
        barriers: &[usize],
        chunks: usize,
        class: crate::LaunchClass,
    ) -> Result<u64, String> {
        let width = barriers
            .iter()
            .map(|&block| {
                crate::programs::native_target_graph::block_input_width(
                    &self.definition.decoder,
                    block,
                )
            })
            .try_fold(0, |widest, width| width.map(|width| widest.max(width)))?;
        (chunks as u64)
            .checked_mul(class.rows() as u64)
            .and_then(|rows| rows.checked_mul(width))
            .ok_or_else(|| "held rows overflow".into())
    }

    /// Per operation of a target launch, what its rows read on global
    /// attention layers (`TargetLaunchInputs::with_visibility`).
    pub(super) fn launch_visibility(
        &self,
        operations: &[Operation],
    ) -> Result<Vec<magnitude_readout::Visibility>, String> {
        operations
            .iter()
            .map(|operation| Ok(self.installed_input(operation.request())?.visibility.clone()))
            .collect()
    }

    /// Whether inputs may attend in blocks (`install_visible_input`), else
    /// what that needs: a model computing layer-major
    /// (`computes_layer_major`), which every block wider than one launch
    /// needs and whose load prepares what a blocked advance reads.
    pub fn reads_blocks(&self) -> Result<(), &'static str> {
        crate::planning::computes_layer_major(&self.definition)
    }

    /// The slot of `operation`'s rows `computed` (`target_chunks`), chunk
    /// `chunk` of `chunks`, over `advance`.
    pub(super) fn target_slot(
        &self,
        operation: &Operation,
        advance: &TentativeAdvance,
        computed: std::ops::Range<usize>,
        (chunk, chunks): (usize, usize),
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
        let retained = self.retained_rows(*request);
        let version = chunk_version(&advance.bindings(), chunks, chunk, computed.len())?;
        // A media row's span end; an advance never splits a span. The
        // history reads whose layers attend media bidirectionally read every
        // fresh row of the span; a model computing an advance in chunks reads
        // media causally (`computes_layer_major`), so its chunks may split
        // one. A blocked input's rows read their blocks on
        // the global reads; an advance never splits a block's reads either
        // (`input_rows`).
        let forms = self.read_forms()?;
        let span_end = |index: usize| {
            slices
                .iter()
                .map(|slice| (slice.destination, slice.destination + slice.source.count))
                .find(|(start, end)| (*start..*end).contains(&index))
                .map(|(_, end)| end)
        };
        let visibility = &self.installed_input(*request)?.visibility;
        let blocked = !visibility.is_causal();
        let appends = SlotAppends::of(&self.target_store, advance, &forms, blocked, &computed);
        let mut rows = Vec::with_capacity(computed.len());
        let audit_history = matches!(kind, WorkKind::Verify | WorkKind::Decode)
            && std::env::var_os("MAGNITUDE_TRACE_TARGET_VERIFY_LAUNCHES").is_some()
            && *position
                >= std::env::var("MAGNITUDE_TRACE_TARGET_VERIFY_MIN_POSITION")
                    .ok()
                    .and_then(|value| value.parse::<usize>().ok())
                    .unwrap_or(0)
            && !TARGET_VERIFY_LAUNCH_TRACED.load(Ordering::Relaxed);
        for (index, token) in tokens.iter().enumerate().take(computed.end).skip(computed.start) {
            let coordinates = *coordinates
                .get(index)
                .ok_or("prepared input coordinate count differs from tokens")?;
            let reads = blocked
                .then(|| {
                    visibility
                        .reads(position + index)
                        .zip(visibility.block(position + index))
                        .ok_or("a blocked row lies outside its input")
                })
                .transpose()?;
            let histories = row_histories(
                &self.target_store,
                advance,
                &appends,
                index,
                &computed,
                RowReach {
                    span_end: span_end(index),
                    blocks: reads.as_ref().map(|(reads, block)| BlockedRow { reads, block }),
                },
                &forms,
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
            let readouts = super::retention::row_readouts(operation, index, &retained)?;
            rows.push(Row {
                token: i32::try_from(token.0).map_err(|_| "token exceeds i32")?,
                coordinates,
                histories,
                demand: row_demand(operation, index)
                    | if readouts.is_empty() {
                        crate::batching::Demand::NONE
                    } else {
                        crate::batching::Demand::READOUT
                    },
                select: selection(operation, index),
                readouts,
            });
        }
        Ok((
            Slot {
                rows,
                bank: version.bank,
                previous_tape: version.tape,
                following_bank: version.following,
                stop: version.stop,
            },
            slices,
        ))
    }
}

/// The recurrent version one chunk of an advance reads and the bank it
/// publishes to. Of an advance computed in several chunks
/// (`ExecutorDomain::target_chunks`), the first chunk reads the accepted
/// version and each later one the bank the one before published; the last
/// publishes to the advance's successor bank and the others alternately to
/// its interim bank and its successor, so no bank is written while it is
/// read. Every chunk publishes after all of its `rows`. A lone chunk is the
/// advance's own version.
pub(crate) struct ChunkVersion {
    pub bank: i32,
    pub tape: i32,
    pub following: i32,
    pub stop: i32,
}

/// The version chunk `chunk` of `chunks` (`rows` rows) of the advance
/// bound by `binding` reads and publishes (`ChunkVersion`).
pub(crate) fn chunk_version(
    binding: &magnitude_state::OwnedAdvanceBindings<'_>,
    chunks: usize,
    chunk: usize,
    rows: usize,
) -> Result<ChunkVersion, String> {
    let word = |value: usize| i32::try_from(value).map_err(|_| "a recurrent version exceeds i32");
    if chunks == 1 {
        return Ok(ChunkVersion {
            bank: word(binding.previous_bank)?,
            tape: word(binding.previous_tape)?,
            following: word(binding.following_bank)?,
            stop: word(binding.stop)?,
        });
    }
    let interim = binding
        .interim_bank
        .ok_or("an advance computed in chunks holds no interim bank")?;
    // The bank chunk `chunk` reads, published by the chunk before it.
    let published = |chunk: usize| {
        if (chunks - chunk) % 2 == 0 {
            binding.following_bank
        } else {
            interim
        }
    };
    Ok(ChunkVersion {
        bank: word(if chunk == 0 {
            binding.previous_bank
        } else {
            published(chunk)
        })?,
        tape: word(if chunk == 0 { binding.previous_tape } else { 0 })?,
        following: word(published(chunk + 1))?,
        stop: word(rows)?,
    })
}

/// The accepted state a target launch reads of each of its operations: the
/// request's state when its group is reserved, its advance's when launched.
pub(super) trait AcceptedHistory {
    /// Its accepted history ranges.
    fn span_count(&self) -> usize;
    /// The accepted rows of `domain` at positions `from` and later.
    fn visible_ranges(
        &self,
        domain: magnitude_state::HistoryDomainId,
        from: usize,
    ) -> Vec<(usize, usize)>;
}

impl AcceptedHistory for SequenceState {
    fn span_count(&self) -> usize {
        SequenceState::span_count(self)
    }
    fn visible_ranges(
        &self,
        domain: magnitude_state::HistoryDomainId,
        from: usize,
    ) -> Vec<(usize, usize)> {
        SequenceState::visible_ranges(self, domain, from)
    }
}

impl AcceptedHistory for TentativeAdvance {
    fn span_count(&self) -> usize {
        TentativeAdvance::span_count(self)
    }
    fn visible_ranges(
        &self,
        domain: magnitude_state::HistoryDomainId,
        from: usize,
    ) -> Vec<(usize, usize)> {
        TentativeAdvance::visible_ranges(self, domain, from)
    }
}

/// Whether history read `read` of `store` (`StateStore::history_reads`) is
/// global: its domain, or a Shared read's source domain, holds every token
/// (`HistoryDomainKind::Token`) and no indexer selects among it. A blocked
/// input's rows read their blocks there.
pub(crate) fn global_read(store: &StateStore, read: usize) -> bool {
    let stored = store.history_domains().count();
    let domain = if read < stored {
        magnitude_state::HistoryDomainId(read)
    } else {
        match store.shared_source_domains().get(read - stored) {
            Some(domain) => *domain,
            None => return false,
        }
    };
    store.history_domain_kind(domain) == magnitude_state::HistoryDomainKind::Token
        && !indexer_domain(store, domain)
}

/// What one row of an advance reads beyond the causal rule: through its
/// media span's end `span_end` on the history reads that attend media
/// bidirectionally, or, for a row of a blocked input, the positions `blocks`
/// on the global history reads.
#[derive(Clone, Copy)]
pub(super) struct RowReach<'a> {
    pub span_end: Option<usize>,
    pub blocks: Option<BlockedRow<'a>>,
}

/// What a row of a blocked input reads on global reads: the positions
/// `reads` (`Visibility::reads`), among them its own block's, `block`.
#[derive(Clone, Copy)]
pub(crate) struct BlockedRow<'a> {
    pub reads: &'a [std::ops::Range<usize>],
    pub block: &'a std::ops::Range<usize>,
}

/// The histories row `index` of `advance` reads and writes, one per history
/// read of `store` (`StateStore::history_reads`), in a launch computing the
/// advance's rows `computed` (`ExecutorDomain::target_chunks`). A stored
/// domain's: its
/// accepted rows from the query's window start, the advance's rows from the
/// same start (through the row, or, for a media row of a read in
/// `forms.bidirectional`, through its media span's end), and the row's
/// destination; a blocked row's on a global read (`blocked_reads`). A
/// Shared source domain's (`shared_read_history`) follow. `appends` is the
/// advance's (`SlotAppends::of`), formed once for its rows. Fresh rows are
/// offsets in the launch's slot, the rows `computed`.
pub(super) fn row_histories(
    store: &StateStore,
    advance: &TentativeAdvance,
    appends: &SlotAppends,
    index: usize,
    computed: &std::ops::Range<usize>,
    reach: RowReach<'_>,
    forms: &ReadForms,
) -> Result<Vec<RowHistory>, String> {
    let binding = advance.bindings();
    let end = |read: usize| reach.span_end.filter(|_| forms.bidirectional[read]);
    let blocks = |read: usize| reach.blocks.filter(|_| forms.global[read]);
    // The fresh span `[start, end)` of advance rows, as slot offsets.
    let slot = |start: usize, end: Option<usize>| -> Result<(i32, Option<i32>), String> {
        if start < computed.start || end.is_some_and(|end| end > computed.end) {
            return Err("a row's fresh rows lie outside its launch".into());
        }
        Ok((
            i32::try_from(start - computed.start).map_err(|_| "fresh start exceeds i32")?,
            end.map(|end| i32::try_from(end - computed.start).map_err(|_| "fresh end exceeds i32"))
                .transpose()?,
        ))
    };
    let stored = store.history_domains().map(|domain| {
        if indexer_domain(store, domain) {
            let appended = appends.stored[domain.0]
                .as_ref()
                .ok_or("an indexer domain has no appended rows")?;
            let history = indexer_history(store, advance, appended, domain, index)?;
            return Ok(RowHistory {
                fresh_start: slot(index, None)?.0,
                ..history
            });
        }
        let destination = binding.destinations[domain.0]
            .get(index)
            .map_or(Ok(-1), |value| {
                i32::try_from(*value).map_err(|_| "target destination exceeds i32")
            })?;
        if let Some(row) = blocks(domain.0) {
            let appended = appends.stored[domain.0]
                .as_ref()
                .ok_or("a blocked global domain has no appended rows")?;
            let (visible, own) = blocked_reads(
                advance,
                domain,
                appended,
                row,
                index,
                computed,
                fresh_span(store, domain.0),
            )?;
            let (fresh_start, bidirectional_end) = slot(own.start, Some(own.end))?;
            return Ok(RowHistory {
                visible,
                fresh_start,
                bidirectional_end,
                destination,
            });
        }
        let (mut visible, mut fresh_start) = accepted_history(store, advance, domain, index)?;
        // Policy comes from this target row's declared causal/media behavior,
        // never a window size or the shape of a drafter's block. Rows of the
        // advance an earlier chunk computed are read at their destinations.
        if cached_causal_domain(store, domain) && end(domain.0).is_none() {
            let appended = appends.stored[domain.0]
                .as_ref()
                .ok_or("a compressed causal domain has no appended rows")?;
            visible.extend(appended.causal_prior(fresh_start, index)?);
            fresh_start = index;
        } else if fresh_start < computed.start {
            let appended = appends.stored[domain.0]
                .as_ref()
                .ok_or("a chunked domain has no appended rows")?;
            visible.extend(appended.causal_prior(fresh_start, computed.start)?);
            fresh_start = computed.start;
        }
        let (fresh_start, bidirectional_end) = slot(fresh_start, end(domain.0))?;
        Ok(RowHistory {
            visible,
            fresh_start,
            bidirectional_end,
            destination,
        })
    });
    let stored_reads = store.history_domains().count();
    let shared = appends
        .shared
        .iter()
        .enumerate()
        .map(|(position, (domain, appended))| {
            let read = stored_reads + position;
            let history = match blocks(read) {
                Some(row) => RowHistory {
                    visible: blocked_reads(
                        advance,
                        *domain,
                        appended,
                        row,
                        index,
                        computed,
                        fresh_span(store, read),
                    )?
                    .0,
                    fresh_start: 0,
                    bidirectional_end: None,
                    destination: -1,
                },
                None => shared_read_history(store, advance, *domain, appended, index, end(read))?,
            };
            Ok(RowHistory {
                fresh_start: slot(index, None)?.0,
                ..history
            })
        });
    stored.chain(shared).collect()
}

/// Per history read of a target store (`StateStore::history_reads`): whether
/// its layers attend media rows bidirectionally, and whether it is global
/// (its domain holds every token, `HistoryDomainKind::Token`), where a
/// blocked input's rows read their blocks.
pub(super) struct ReadForms {
    pub bidirectional: Vec<bool>,
    pub global: Vec<bool>,
}

/// Which of its launch's rows a blocked row reads fresh on one global read,
/// the rest of them as stored history.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum FreshSpan {
    /// The run of positions it reads around itself: a dense history stores
    /// a key exactly as a fresh read sees it, so the widest run reads alike
    /// with the fewest ranges.
    Run,
    /// Its own block's: a compressed history's stored rows read otherwise
    /// than fresh ones (`cached_causal_domain`), so a row reads every row
    /// outside its block as stored, however its launches are cut.
    Block,
    /// None: a Shared reader projects no fresh keys.
    Nothing,
}

/// The slot rows a blocked row at slot row `index` reads fresh under `span`,
/// of the slot rows `run` it reads around itself; `block` is its block's,
/// which an advance holds whole.
fn fresh_rows(
    span: FreshSpan,
    run: &std::ops::Range<usize>,
    index: usize,
    block: std::ops::Range<usize>,
) -> std::ops::Range<usize> {
    match span {
        FreshSpan::Run => run.clone(),
        FreshSpan::Block => block,
        FreshSpan::Nothing => index..index,
    }
}

/// The rows of `rows` within `computed`, the rows a launch computes; empty
/// when none is.
fn within(
    rows: std::ops::Range<usize>,
    computed: &std::ops::Range<usize>,
) -> std::ops::Range<usize> {
    let start = rows.start.max(computed.start);
    start..rows.end.min(computed.end).max(start)
}

/// How a blocked row reads its launch on history read `read` of `store`.
pub(crate) fn fresh_span(store: &StateStore, read: usize) -> FreshSpan {
    if read >= store.history_domains().count() {
        FreshSpan::Nothing
    } else if cached_causal_domain(store, magnitude_state::HistoryDomainId(read)) {
        FreshSpan::Block
    } else {
        FreshSpan::Run
    }
}

/// What row `index` of `advance` reads in the global `domain` as a blocked
/// `row`, in a launch computing the advance's rows `computed`: the visible
/// ranges of the positions it reads, accepted history and, of the advance,
/// the destinations of the rows outside its fresh span (`span`, within
/// `computed`); and that span, as advance rows. `appended` is the advance's
/// runs in `domain`. Every position must precede the advance's end.
pub(crate) fn blocked_reads(
    advance: &TentativeAdvance,
    domain: magnitude_state::HistoryDomainId,
    appended: &AppendedRuns,
    row: BlockedRow<'_>,
    index: usize,
    computed: &std::ops::Range<usize>,
    span: FreshSpan,
) -> Result<(Vec<[i32; 2]>, std::ops::Range<usize>), String> {
    let reads = row.reads;
    let start = advance.position();
    let mut visible = Vec::new();
    for read in reads {
        let accepted = read.start..read.end.min(start);
        if !accepted.is_empty() {
            visible.extend(accepted_between(advance, domain, accepted)?);
        }
    }
    let fresh = reads
        .iter()
        .filter(|read| read.end > start)
        .map(|read| read.start.max(start) - start..read.end - start)
        .collect::<Vec<_>>();
    let run = fresh
        .iter()
        .find(|run| run.contains(&index))
        .ok_or("a blocked row does not read itself")?;
    let own = within(
        fresh_rows(span, run, index, row.block.start - start..row.block.end - start),
        computed,
    );
    for run in &fresh {
        for part in [run.start..own.start.min(run.end), own.end.max(run.start)..run.end] {
            if !part.is_empty() {
                visible.extend(appended.launch_reads(part.start, part.end)?);
            }
        }
    }
    Ok((visible, own))
}

/// The accepted-history ranges of stored `domain` holding positions
/// `positions` of `advance`'s sequence, each within one slab.
fn accepted_between(
    advance: &TentativeAdvance,
    domain: magnitude_state::HistoryDomainId,
    positions: std::ops::Range<usize>,
) -> Result<Vec<[i32; 2]>, String> {
    if positions.start < advance.history_start(domain) {
        return Err("a blocked row reads trimmed history".into());
    }
    let mut remaining = positions.len();
    let mut ranges = Vec::new();
    for (first, count) in advance.visible_ranges(domain, positions.start) {
        if remaining == 0 {
            break;
        }
        let rows = count.min(remaining);
        remaining -= rows;
        ranges.push([
            i32::try_from(first).map_err(|_| "history start exceeds i32")?,
            i32::try_from(first.checked_add(rows).ok_or("history end overflow")?)
                .map_err(|_| "history end exceeds i32")?,
        ]);
    }
    if remaining > 0 {
        return Err("a blocked row reads history its sequence does not hold".into());
    }
    Ok(ranges)
}

/// Whether `domain` holds a selected read's indexer keys
/// (`ComponentDescriptor::attended`): no attention reads it, its indexer
/// does (`indexer_history`).
pub(crate) fn indexer_domain(store: &StateStore, domain: magnitude_state::HistoryDomainId) -> bool {
    !store
        .history_components(domain)
        .iter()
        .all(magnitude_state::ComponentDescriptor::attended)
}

/// What row `index` of `advance` reads of an indexer's `domain`
/// (`indexer_select`), every row as stored history, this launch's earlier
/// rows at their destinations: in its raw keys' Window domain the rows of
/// the `rows` positions before it, and its own destination; in its block
/// keys' Block domain every block complete before it, and the destination
/// of the block it completes (or -1). Its fresh span is the row alone.
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
            // The blocks the advance completed before the row, then the one
            // it completes.
            let before = position / rate - start / rate;
            let completes = (position + 1) % rate == 0;
            (
                0,
                0..before,
                completes.then(|| destinations.get(before)).flatten(),
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
                if store.indexer_domains(block).is_some())
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

    /// The history ranges a blocked row reads of the rows its launch
    /// appends by the slot rows `start..end`.
    fn launch_reads(&self, start: usize, end: usize) -> Result<Vec<[i32; 2]>, String> {
        self.ranges(
            start,
            end,
            AppendedRefusals {
                outside: "blocked rows exceed the advance's destinations",
                exceeds: "blocked row destination exceeds i32",
                overflows: "blocked row range overflows",
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
/// of each compressed causal stored domain (`cached_causal_domain`), each
/// indexer domain (`indexer_domain`), for a blocked advance each global
/// domain, and for a chunk after an advance's first every domain, by domain,
/// and of each Shared source domain, in `shared_source_domains` order.
pub(super) struct SlotAppends {
    stored: Vec<Option<AppendedRuns>>,
    shared: Vec<(magnitude_state::HistoryDomainId, AppendedRuns)>,
}

impl SlotAppends {
    pub(super) fn of(
        store: &StateStore,
        advance: &TentativeAdvance,
        forms: &ReadForms,
        blocked: bool,
        computed: &std::ops::Range<usize>,
    ) -> Self {
        Self {
            // An indexer reads its launch's earlier rows as stored history
            // too (`indexer_history`), a blocked row the rows of its launch
            // outside its fresh span (`blocked_reads`), and a chunk's rows
            // the rows earlier chunks computed (`row_histories`).
            stored: store
                .history_domains()
                .map(|domain| {
                    (cached_causal_domain(store, domain)
                        || indexer_domain(store, domain)
                        || (blocked && forms.global[domain.0])
                        || computed.start > 0)
                        .then(|| AppendedRuns::of(store, advance, domain))
                })
                .collect(),
            shared: store
                .shared_source_domains()
                .into_iter()
                .map(|domain| (domain, AppendedRuns::of(store, advance, domain)))
                .collect(),
        }
    }
}

/// The history ranges a launch reserves per row for an advance of `rows`
/// rows over a state of `spans` accepted ranges: those ranges, and for a
/// store with Shared reads (`shared_read_history`) the ranges of the rows
/// the advance appends, at most one per slab they touch.
pub(super) fn reserved_segments(store: &StateStore, spans: usize, rows: usize) -> usize {
    // A Shared reader also sees the rows its source appends in the advance:
    // the rest of the source's last page, then whole pages (see
    // `magnitude_state::history_geometry`).
    spans
        + store
            .shared_source_domains()
            .into_iter()
            .chain(store.history_domains().filter(|&domain| {
                rows > 1 && (cached_causal_domain(store, domain) || indexer_domain(store, domain))
            }))
            .map(|domain| rows.div_ceil(store.history_page_rows(domain)) + 1)
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
/// span bidirectionally), each range within one slab. Its fresh span and
/// destination are unused. `appended` is the advance's runs in `domain`.
fn shared_read_history(
    store: &StateStore,
    advance: &TentativeAdvance,
    domain: magnitude_state::HistoryDomainId,
    appended: &AppendedRuns,
    index: usize,
    bidirectional_end: Option<usize>,
) -> Result<RowHistory, String> {
    let (mut visible, fresh_start) = accepted_history(store, advance, domain, index)?;
    visible.extend(appended.shared_appended(fresh_start, bidirectional_end.unwrap_or(index + 1))?);
    Ok(RowHistory {
        visible,
        fresh_start: i32::try_from(index).map_err(|_| "fresh start exceeds i32")?,
        bidirectional_end: None,
        destination: -1,
    })
}

fn row_demand(operation: &Operation, row: usize) -> crate::batching::Demand {
    use crate::batching::Demand;
    let Operation::Forward {
        tokens,
        kind,
        demand,
        ..
    } = operation
    else {
        return Demand::NONE;
    };
    // The readout demand is the readout rows' alone (`row_readouts`).
    let outputs = Demand::LOGITS | Demand::FEATURES | Demand::SELECT | Demand::TAPS;
    match kind {
        WorkKind::Decode | WorkKind::Verify => *demand & outputs,
        WorkKind::Prefill | WorkKind::Replay if row + 1 == tokens.len() => *demand & outputs,
        WorkKind::Prefill | WorkKind::Replay => *demand & (Demand::FEATURES | Demand::TAPS),
    }
}

/// F32 values of a device tensor, read to the host.
fn host_f32(tensor: &seismic::Tensor) -> Result<Vec<f32>, DomainError> {
    let bytes = tensor.read_to_host().map_err(|error| {
        DomainError::Device(crate::DeviceError::Transfer(error.to_string()))
    })?;
    Ok(bytes
        .chunks_exact(4)
        .map(|word| f32::from_le_bytes([word[0], word[1], word[2], word[3]]))
        .collect())
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
        let Ok(catalog) = seismic::Devices::discover(&[]) else {
            return;
        };
        let Ok(device) = catalog.open_backend(seismic::BackendName::Cpu) else {
            return;
        };
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
            let forms = |bidirectional: bool| ReadForms {
                bidirectional: vec![bidirectional],
                global: vec![true],
            };
            let whole = 0..3;
            let appends = SlotAppends::of(&store, &advance, &forms(true), false, &whole);
            let causal = row_histories(
                &store,
                &advance,
                &appends,
                1,
                &whole,
                RowReach { span_end: None, blocks: None },
                &forms(true),
            )
            .unwrap();
            let media = row_histories(
                &store,
                &advance,
                &appends,
                1,
                &whole,
                RowReach {
                    span_end: Some(3),
                    blocks: None,
                },
                &forms(true),
            )
            .unwrap();
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
            let first = row_histories(
                &store,
                &advance,
                &appends,
                0,
                &whole,
                RowReach { span_end: None, blocks: None },
                &forms(false),
            )
            .unwrap();
            assert!(first[0].visible.is_empty());
            assert_eq!(first[0].fresh_start, 0);
            // A chunk after the first reads the rows before it at their
            // destinations, fresh only its own.
            let chunk = 1..3;
            let appends = SlotAppends::of(&store, &advance, &forms(false), false, &chunk);
            let second = row_histories(
                &store,
                &advance,
                &appends,
                2,
                &chunk,
                RowReach { span_end: None, blocks: None },
                &forms(false),
            )
            .unwrap();
            let destinations = &advance.bindings().destinations[domain.0];
            let read = second[0]
                .visible
                .iter()
                .flat_map(|range| range[0] as usize..range[1] as usize)
                .collect::<Vec<_>>();
            let expected = if codec == KvCodec::AffineK8V4 {
                (destinations[..2].to_vec(), 1)
            } else {
                (destinations[..1].to_vec(), 0)
            };
            assert_eq!((read, second[0].fresh_start), expected);
        }
    }
}

#[cfg(test)]
mod blocked_history_tests {
    use super::*;
    use magnitude_readout::{Block, ReadoutSequence, Visibility};
    use magnitude_state::{
        BankCapacity, ComponentDescriptor, HistoryDomainLayout, HistoryDomainPlan, KvCodec,
        LayerRef, OwnedAdvanceResolution, OwnedStateAdvance, RowDemand,
    };
    use std::collections::BTreeSet;

    fn visibility(rows: usize, blocks: Vec<Block>) -> Visibility {
        let tokens = (0..rows as u32).map(magnitude_artifacts::TokenId).collect::<Vec<_>>();
        let input = magnitude_family_contracts::PreparedModelInput::from_text_coordinates(
            magnitude_family_contracts::TokenPlan::new(
                tokens,
                magnitude_family_contracts::InputLayout::new(rows, vec![]).unwrap(),
            )
            .unwrap(),
            (0..rows).map(|row| [row as i32; 3]).collect(),
        )
        .unwrap();
        ReadoutSequence::new(input, blocks, Vec::new())
            .unwrap()
            .visibility()
    }

    /// Every history row a row's tables read: its visible ranges, then the
    /// destinations of its fresh slot rows, each read once.
    fn read_rows(history: &RowHistory, destinations: &[usize], offset: usize) -> Vec<usize> {
        let mut rows = history
            .visible
            .iter()
            .flat_map(|range| range[0] as usize..range[1] as usize)
            .collect::<Vec<_>>();
        let end = history.bidirectional_end.map_or(offset + 1, |end| end as usize);
        rows.extend((history.fresh_start as usize..end).map(|row| destinations[row]));
        rows
    }

    /// The history rows of a blocked row's tables are exactly those of the
    /// positions its blocks read, accepted or of its advance, for dense and
    /// compressed histories, whether its advance is one launch or chunks: a
    /// continuation block over an accepted trunk (`bidirectional_trunk`),
    /// and a trunk packed with blocks that each read only the trunk and
    /// themselves (`packed_set`).
    #[test]
    fn blocked_rows_read_exactly_the_positions_of_their_blocks() {
        let Ok(catalog) = seismic::Devices::discover(&[]) else {
            return;
        };
        let Ok(device) = catalog.open_backend(seismic::BackendName::Cpu) else {
            return;
        };
        let device = Rc::new(device);
        let layouts = [
            (
                4,
                visibility(
                    7,
                    vec![Block::new(0..4, vec![0]), Block::new(4..7, vec![0, 1])],
                ),
            ),
            (
                0,
                visibility(
                    9,
                    vec![
                        Block::new(0..3, vec![0, 1, 2, 3]),
                        Block::new(3..5, vec![0, 1]),
                        Block::new(5..6, vec![0, 2]),
                        Block::new(6..9, vec![0, 3]),
                    ],
                ),
            ),
            (
                3,
                visibility(
                    9,
                    vec![
                        Block::new(0..3, vec![0]),
                        Block::new(3..5, vec![1]),
                        Block::new(5..9, vec![0, 2]),
                    ],
                ),
            ),
        ];
        for codec in [KvCodec::Dense, KvCodec::AffineK8V4] {
            for (accepted, visibility) in &layouts {
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
                    .provision(&[RowDemand { domain, rows: 12 }], 0)
                    .unwrap();
                let mut state = store.create().unwrap();
                if *accepted > 0 {
                    let advance = OwnedStateAdvance::begin(state, *accepted).ok().unwrap();
                    let (OwnedAdvanceResolution::Committed(committed)
                    | OwnedAdvanceResolution::Aborted(committed)) =
                        advance.commit(*accepted).ok().unwrap();
                    state = committed;
                }
                let held = state
                    .visible_ranges(domain, 0)
                    .into_iter()
                    .flat_map(|(start, count)| start..start + count)
                    .collect::<Vec<_>>();
                assert_eq!(held.len(), *accepted);
                let rows = (*accepted..)
                    .take_while(|&row| visibility.reads(row).is_some())
                    .count();
                let advance =
                    TentativeAdvance::Accepted(OwnedStateAdvance::begin(state, rows).ok().unwrap());
                let forms = ReadForms {
                    bidirectional: vec![false],
                    global: vec![true],
                };
                let destinations = advance.bindings().destinations[domain.0].clone();
                // The advance in one launch, and in two chunks.
                let launches = [vec![0..rows], vec![0..rows / 2, rows / 2..rows]];
                for computed in launches.into_iter().flatten() {
                    let appends = SlotAppends::of(&store, &advance, &forms, true, &computed);
                    for index in computed.clone() {
                        let reads = visibility.reads(accepted + index).unwrap();
                        let block = visibility.block(accepted + index).unwrap();
                        let histories = row_histories(
                            &store,
                            &advance,
                            &appends,
                            index,
                            &computed,
                            RowReach {
                                span_end: None,
                                blocks: Some(BlockedRow {
                                    reads: &reads,
                                    block: &block,
                                }),
                            },
                            &forms,
                        )
                        .unwrap();
                        let read = read_rows(
                            &histories[0],
                            &destinations[computed.clone()],
                            index - computed.start,
                        );
                        let expected = reads
                            .iter()
                            .flat_map(|read| read.clone())
                            .map(|position| match position.checked_sub(*accepted) {
                                None => held[position],
                                Some(offset) => destinations[offset],
                            })
                            .collect::<BTreeSet<_>>();
                        assert_eq!(read.len(), expected.len(), "{codec:?} row {index} reads a row twice");
                        assert_eq!(read.into_iter().collect::<BTreeSet<_>>(), expected);
                        // A compressed history reads its own block fresh and
                        // every other row as stored; a dense one its whole run;
                        // each within the rows its launch computes.
                        let fresh = histories[0].fresh_start as usize + computed.start
                            ..histories[0].bidirectional_end.unwrap() as usize + computed.start;
                        let own = within(block.start - accepted..block.end - accepted, &computed);
                        if codec == KvCodec::AffineK8V4 {
                            assert_eq!(fresh, own);
                        } else {
                            assert!(fresh.start <= own.start && own.end <= fresh.end);
                            assert!(computed.start <= fresh.start && fresh.end <= computed.end);
                        }
                    }
                }
                drop(advance);
            }
        }
    }

    /// A row before a boundary reads nothing after it; a packed trunk
    /// reads every block, so no position inside the pack is a boundary.
    #[test]
    fn boundaries_follow_what_rows_read() {
        let packed = visibility(
            9,
            vec![
                Block::new(0..3, vec![0, 1, 2]),
                Block::new(3..5, vec![0, 1]),
                Block::new(5..9, vec![0, 2]),
            ],
        );
        assert_eq!(
            (0..=9).filter(|&position| packed.boundary(position)).collect::<Vec<_>>(),
            vec![0, 9]
        );
        assert_eq!(packed.reads(6), Some(vec![0..3, 5..9]));
        assert_eq!(packed.reads(1), Some(vec![0..9]));
        let trunk = visibility(
            7,
            vec![Block::new(0..4, vec![0]), Block::new(4..7, vec![0, 1])],
        );
        assert_eq!(
            (0..=7).filter(|&position| trunk.boundary(position)).collect::<Vec<_>>(),
            vec![0, 4, 7]
        );
        assert_eq!(trunk.boundary_from(5), Some(7));
        assert_eq!(trunk.prefix(5), trunk.prefix(4));
        assert_ne!(trunk.prefix(4), Visibility::causal());
        assert_eq!(Visibility::causal().reads(3), Some(vec![0..4]));
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
