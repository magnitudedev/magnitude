//! head lifecycle for the executor domain.

use super::*;
use crate::batching::{HeadPasses, HeadSlot};
use crate::DraftForm;
use std::sync::atomic::{AtomicBool, Ordering};

static HEAD_LAUNCH_TRACED: AtomicBool = AtomicBool::new(false);

impl<F: ProgramFamily> ExecutorDomain<F> {
    /// Bytes of one head conditioning row: the target's normalized output
    /// feature in the activation representation.
    pub(super) fn head_conditioning_bytes(&self) -> usize {
        self.definition.decoder.hidden as usize * self.definition.decoder.activation_dtype.bytes()
    }

    /// Submit one batch of head transactions. Each commits its entry rows to
    /// head state when it completes; its chained proposal rows are scratch.
    /// The flight holds `bindings` until `finish_head` returns them.
    pub fn submit_head(
        &mut self,
        bindings: StateBindings<F>,
        operations: &[Operation],
        graph_workspace: NativeGraphWorkspaceLease,
        graph_output: NativeGraphOutputLease,
        advances: Vec<OwnedStateAdvance>,
    ) -> Result<HeadFlight<F>, SubmitFailure<F>> {
        if operations.is_empty() && advances.is_empty() {
            return Err(SubmitFailure::Refused(
                "head group is empty".into(),
                bindings,
            ));
        }
        // Reservation checked the group and moved its states into the
        // advances: a mismatch now is a broken domain invariant.
        let broken =
            |detail: String| SubmitFailure::Failed(DomainError::invariant(detail));
        let store = self
            .head_store
            .clone()
            .ok_or_else(|| broken("head advances without a head store".into()))?;
        if operations.len() != advances.len() {
            return Err(broken(
                "head reservation advance count differs from operations".into(),
            ));
        }
        let advances = advances
            .into_iter()
            .map(TentativeAdvance::Accepted)
            .collect::<Vec<_>>();
        let mut seen = BTreeSet::new();
        let mut steps = 0usize;
        for (operation, advance) in operations.iter().zip(&advances) {
            operation
                .validate()
                .map_err(|error| broken(error.to_string()))?;
            let Operation::Head {
                request,
                position,
                proposals,
                ..
            } = operation
            else {
                return Err(broken("head group contains another operation kind".into()));
            };
            if !seen.insert(*request) {
                return Err(broken("head request is repeated".into()));
            }
            if advance.position() != *position || advance.rows() != operation.row_count() {
                return Err(broken(
                    "head position or rows differ from accepted state".into(),
                ));
            }
            steps = steps.max(proposals.len());
        }
        if std::env::var_os("MAGNITUDE_TRACE_DRAFT").is_some() {
            for operation in operations {
                if let Operation::Head {
                    request,
                    tokens,
                    position,
                    proposals,
                    ..
                } = operation
                {
                    eprintln!(
                        "draft transaction request={request:?} position={position} proposals={} entry={:?}",
                        proposals.len(),
                        tokens.iter().map(|token| token.0).collect::<Vec<_>>()
                    );
                }
            }
        }
        let metadata = operations
            .iter()
            .map(|operation| match operation {
                Operation::Head {
                    request,
                    tokens,
                    proposals,
                    ..
                } => (*request, tokens.len(), proposals.len()),
                _ => unreachable!("validated head group"),
            })
            .collect::<Vec<_>>();
        let form = match &operations[0] {
            Operation::Head { form, .. } => *form,
            _ => unreachable!("validated head group"),
        };
        if operations.iter().any(
            |operation| !matches!(operation, Operation::Head { form: other, .. } if *other == form),
        ) {
            self.restore_head_advances(&metadata, advances);
            return Err(SubmitFailure::Refused(
                "head group mixes draft forms".into(),
                bindings,
            ));
        }
        // A drafting block drafts the load's proposals; a request takes its
        // leading ones.
        let steps = match form {
            DraftForm::Block if steps > 0 => self.execution.policy().method().draft_rows(),
            DraftForm::Chained | DraftForm::Block => steps,
        };
        let vocabulary = self.definition.decoder.vocabulary as usize;
        let class_limits = self.head_class_limits().map_err(SubmitFailure::Failed)?;
        let batch = match form {
            DraftForm::Chained => operations
                .iter()
                .zip(&advances)
                .map(|(operation, advance)| self.head_slot(operation, advance, steps))
                .collect::<Result<Vec<_>, _>>()
                .and_then(|slots| {
                    crate::batching::ValidatedHeadBatch::from_slots(
                        &slots,
                        vocabulary,
                        class_limits,
                    )
                    .map_err(|error| error.to_string())
                }),
            DraftForm::Block => operations
                .iter()
                .zip(&advances)
                .map(|(operation, advance)| self.draft_slot(operation, advance, steps))
                .collect::<Result<Vec<_>, _>>()
                .and_then(|slots| {
                    crate::batching::ValidatedHeadBatch::from_block_slots(
                        &slots,
                        vocabulary,
                        class_limits,
                    )
                    .map_err(|error| error.to_string())
                }),
        };
        let batch = match batch {
            Ok(batch) => batch,
            Err(error) => {
                self.restore_head_advances(&metadata, advances);
                return Err(SubmitFailure::Refused(error.into(), bindings));
            }
        };
        let conditioning = HeadConditioning::Rows(
            operations
                .iter()
                .map(|operation| match operation {
                    Operation::Head { conditioning, .. } => conditioning.clone(),
                    _ => unreachable!("validated head group"),
                })
                .collect(),
        );
        // A block drafter's first draft anchors at `draft_from` and its
        // windowed layers read history only from `draft_from + 1 - window`:
        // priming rows before that every window drops unread.
        let windowed = !(form == DraftForm::Block && steps == 0)
            || skippable_window(&store).is_none_or(|window| {
                operations.iter().any(|operation| match operation {
                    Operation::Head {
                        phase: crate::HeadPhase::Priming { draft_from },
                        tokens,
                        position,
                        ..
                    } => position + tokens.len() + window > *draft_from,
                    _ => true,
                })
            });
        let inputs = HeadLaunchInputs::new(
            batch,
            advances,
            conditioning,
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
            Err((inputs, error)) => {
                let (_, advances, _, _, _) = inputs.into_parts();
                self.restore_head_advances(&metadata, advances);
                return Err(SubmitFailure::Failed(DomainError::Invariant(error)));
            }
        };
        // A single proposing flight supplies kernel attribution without
        // making every served step pay launch-detail tracing overhead.
        let trace_position = std::env::var("MAGNITUDE_TRACE_HEAD_MIN_POSITION")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(0);
        let launch_trace = if steps > 0
            && std::env::var_os("MAGNITUDE_TRACE_HEAD_LAUNCHES").is_some()
            && operations.iter().any(|operation| {
                matches!(operation, Operation::Head { position, .. } if *position >= trace_position)
            })
            && !HEAD_LAUNCH_TRACED.swap(true, Ordering::Relaxed)
        {
            match self
                .domain
                .device()
                .trace_submissions(seismic::TraceDetail::Launches)
            {
                Ok(trace) => Some(trace),
                Err(error) => {
                    HEAD_LAUNCH_TRACED.store(false, Ordering::Relaxed);
                    eprintln!("head launch attribution unavailable: {error}");
                    None
                }
            }
        } else {
            None
        };
        let started = Instant::now();
        let submission = match self.family.submit_head(launch) {
            Ok(submission) => submission,
            Err((error, launch)) => {
                let (core, _, _) = launch.into_submission_parts();
                let (_, advances, _) = core.into_parts();
                self.restore_head_advances(&metadata, advances);
                return Err(SubmitFailure::Failed(error.into()));
            }
        };
        Ok(HeadFlight {
            requests: metadata,
            steps,
            submission,
            started,
            launch_trace,
            bindings,
        })
    }

    /// Return launches' accepted head states; a successor's tentative rows
    /// return to the store when it drops.
    pub(super) fn restore_head_advances(
        &mut self,
        metadata: &[(RequestId, usize, usize)],
        advances: Vec<TentativeAdvance>,
    ) {
        for ((request, _, _), advance) in metadata.iter().zip(advances) {
            if let TentativeAdvance::Accepted(advance) = advance {
                self.head.insert(*request, advance.abort());
            }
        }
    }

    /// The head rows of one transaction padded to `steps` selections: entry
    /// rows at the accepted head position, then one chained row per further
    /// step. Chained rows past the request's own proposals append nowhere and
    /// select at the request's last proposal position; their selections are
    /// discarded.
    fn head_slot(
        &self,
        operation: &Operation,
        advance: &TentativeAdvance,
        steps: usize,
    ) -> Result<HeadSlot, String> {
        let Operation::Head {
            request,
            tokens,
            position,
            proposals,
            ..
        } = operation
        else {
            return Err("non-head operation".into());
        };
        let binding = advance.bindings();
        let i32_of = |value: usize, what: &str| {
            i32::try_from(value).map_err(|_| format!("head {what} exceeds i32"))
        };
        // The draft head's store has one Token domain.
        let domain = self
            .head_store
            .as_ref()
            .ok_or("a head operation without a head store")?
            .sole_history_domain()
            .map_err(|error| error.to_string())?;
        let history = advance
            .history_ranges(domain)
            .into_iter()
            .map(|(start, count)| {
                Ok([
                    i32_of(start, "history start")?,
                    i32_of(
                        start
                            .checked_add(count)
                            .ok_or("head history end overflow")?,
                        "history end",
                    )?,
                ])
            })
            .collect::<Result<Vec<[i32; 2]>, String>>()?;
        let destination = |row: usize| -> Result<i32, String> {
            binding.destinations[domain.0]
                .get(row)
                .map_or(Ok(-1), |value| i32_of(*value, "destination"))
        };
        // A head row at head position p pairs the token after target row p
        // with that row's feature, so it takes target row p's coordinates.
        let chained = steps.saturating_sub(1);
        let coordinates = self.input_coordinates(*request, *position, tokens.len() + chained)?;
        let row = |index: usize, token: i32, visible: Vec<[i32; 2]>| -> Result<Row, String> {
            Ok(Row {
                token,
                coordinates: *coordinates
                    .get(index)
                    .ok_or("prepared input coordinate count differs from head rows")?,
                histories: vec![RowHistory {
                    visible,
                    fresh_start: 0,
                    bidirectional_end: None,
                    destination: destination(index)?,
                }],
                demand: crate::batching::Demand::NONE,
                select: None,
            })
        };
        let entry = tokens
            .iter()
            .enumerate()
            .map(|(index, token)| {
                row(
                    index,
                    i32::try_from(token.0).map_err(|_| "head token exceeds i32")?,
                    history.clone(),
                )
            })
            .collect::<Result<Vec<_>, String>>()?;
        // Chained row j sees the history, the entry rows, and the chained
        // rows before it; the entry rows were appended by the entry pass.
        let mut visible = history.clone();
        for index in 0..tokens.len() {
            let appended = destination(index)?;
            visible.push([appended, appended + 1]);
        }
        let mut chain = Vec::with_capacity(chained);
        for step in 1..steps {
            let index = tokens.len() + step - 1;
            chain.push(row(index, 0, coalesce(&visible))?);
            let appended = destination(index)?;
            if appended >= 0 {
                visible.push([appended, appended + 1]);
            }
        }
        let last = proposals.last();
        let selections = (0..steps)
            .map(|step| {
                let spec = proposals
                    .get(step)
                    .or(last)
                    .ok_or("a causal head in a drafting batch has no selection")?;
                Ok(super::target::select_row(spec))
            })
            .collect::<Result<Vec<_>, String>>()?;
        Ok(HeadSlot {
            entry: Slot {
                rows: entry,
                bank: i32_of(binding.previous_bank, "bank")?,
                previous_tape: i32_of(binding.previous_tape, "tape rows")?,
                following_bank: i32_of(binding.following_bank, "successor bank")?,
                stop: i32_of(binding.stop, "committed rows")?,
            },
            chain,
            proposals: selections,
        })
    }

    /// Complete a head batch: read every drafted selection once, then hold
    /// each transaction for its owner's reconciliation. Returns the bindings
    /// the flight held; an error consumes them, as the domain cannot
    /// continue.
    pub fn finish_head(
        &mut self,
        flight: HeadFlight<F>,
    ) -> Result<(Vec<PendingOperationOutcome>, StateBindings<F>), DomainError> {
        let HeadFlight {
            requests,
            steps,
            submission,
            started,
            launch_trace,
            bindings,
        } = flight;
        let completed = submission.finish().map_err(DomainError::Device)?;
        if let Some(trace) = &launch_trace {
            let submissions = trace
                .collect()
                .map_err(|error| DomainError::Input(error.to_string()))?;
            let mut entries = BTreeMap::<String, (usize, f64)>::new();
            let detail = std::env::var_os("MAGNITUDE_TRACE_HEAD_DETAIL").is_some();
            let mut attention_layer = 0;
            for submission in submissions {
                for launch in submission.launches {
                    if let Some((start, end)) = launch.device {
                        if detail && launch.entry == "attention_prefill" && launch.launch == 1 {
                            eprintln!(
                                "head attention layer={} attend_ms={:.3}",
                                attention_layer,
                                (end - start) * 1_000.0
                            );
                            attention_layer += 1;
                        }
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
            entries.sort_by(|a, b| b.1 .1.total_cmp(&a.1 .1));
            eprintln!("head launch attribution (one proposing flight):");
            for (entry, (launches, ms)) in entries {
                eprintln!("  {entry}: {ms:.3} ms across {launches} launches");
            }
        }
        let duration = started.elapsed();
        if std::env::var_os("MAGNITUDE_TRACE_FLIGHTS").is_some() {
            eprintln!(
                "flight head duration_ms={:.3} requests={:?}",
                duration.as_secs_f64() * 1000.0,
                requests
            );
        }
        let (core, selections) = completed.into_parts();
        let slots = core.batch().actual_slots();
        if slots != requests.len() {
            return Err(DomainError::invariant(
                "head slot count differs from request count",
            ));
        }
        let selected = match selections {
            Some(selections) => {
                decode_selected(&selections.tensor().read_to_host().map_err(|error| {
                    DomainError::Device(crate::DeviceError::Transfer(error.to_string()))
                })?)
                .map_err(DomainError::invariant)?
            }
            None => Vec::new(),
        };
        let slot_class = selected.len().checked_div(steps).unwrap_or(0);
        if steps != 0 && (slot_class < slots || selected.len() != steps * slot_class) {
            return Err(DomainError::invariant(
                "head selections differ from the batch's steps and slots",
            ));
        }
        let passes = core.batch().passes();
        if std::env::var_os("MAGNITUDE_TRACE_DRAFT").is_some() {
            for (slot, (request, _, proposals)) in requests.iter().enumerate() {
                eprintln!(
                    "draft proposals request={request:?} {:?}",
                    (0..*proposals)
                        .map(|step| {
                            let selected = selected[step * slot_class + slot];
                            (selected.token.0, selected.status)
                        })
                        .collect::<Vec<_>>()
                );
            }
        }
        let (_, advances, _) = core.into_parts();
        let advances = advances
            .into_iter()
            .map(|advance| match advance {
                TentativeAdvance::Accepted(advance) => Ok(advance),
                TentativeAdvance::Successor(_) => Err(DomainError::invariant(
                    "a head transaction follows an in-flight advance",
                )),
            })
            .collect::<Result<Vec<_>, _>>()?;
        let pending = requests
            .into_iter()
            .zip(advances)
            .enumerate()
            .map(
                |(slot, ((request, rows, proposals), advance))| PendingOperationOutcome {
                    #[cfg(any(test, feature = "experimental-pipeline-cuda"))]
                    pipeline_prefix: None,
                    request,
                    outcome: Outcome::Head {
                        proposals: (0..proposals)
                            .map(|step| selected[step * slot_class + slot])
                            .collect(),
                    },
                    advance: Some(advance),
                    primed: None,
                    rows: advance_rows(passes, rows, proposals),
                    committed_rows: rows,
                    kind: WorkKind::Decode,
                    physical_duration: duration,
                    image: None,
                },
            )
            .collect();
        Ok((pending, bindings))
    }
}

/// The widest window of a drafter store whose injection can skip its
/// windowed layers: one that also keeps a history no window drops (with
/// every layer windowed, skipping them leaves nothing to inject).
pub(super) fn skippable_window(store: &StateStore) -> Option<usize> {
    let mut widest = None;
    let mut full = false;
    for domain in store.history_domains() {
        match store.history_domain_kind(domain) {
            magnitude_state::HistoryDomainKind::Window { rows } => {
                widest = widest.max(Some(rows))
            }
            _ => full = true,
        }
    }
    widest.filter(|_| full)
}

/// Rows a head transaction advances its state by: the entry rows, and a
/// chained head's speculative rows (block rows append nowhere).
fn advance_rows(passes: HeadPasses, entry: usize, proposals: usize) -> usize {
    match passes {
        HeadPasses::Chained => entry + proposals.saturating_sub(1),
        HeadPasses::Block => entry,
    }
}

/// Ascending visible spans with adjacent spans merged.
fn coalesce(spans: &[[i32; 2]]) -> Vec<[i32; 2]> {
    let mut merged: Vec<[i32; 2]> = Vec::with_capacity(spans.len());
    for &span in spans {
        match merged.last_mut() {
            Some(previous) if previous[1] == span[0] => previous[1] = span[1],
            _ => merged.push(span),
        }
    }
    merged
}
