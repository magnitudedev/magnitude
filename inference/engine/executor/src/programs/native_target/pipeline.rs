//! Complete stage execution reuses ordinary state binding and readout. All
//! submissions are observed before any output lease can return to its pool.
use super::*;
use crate::pipeline::ExecutableStage;

pub(crate) enum StageResult {
    Boundary {
        rows: u64,
        bytes: Vec<u8>,
        elapsed: std::time::Duration,
    },
    Readout(Option<TargetReadoutGraphResult>),
}

fn run_completed(
    ready: seismic::ReadyNativeGraphRun<'_>,
    pending: &mut Option<NativeGraphOutputs>,
    commits: &mut Option<CommitSpan>,
) -> Result<(), SubmitError> {
    let (outputs, completion) = ready.submit().map_err(device)?;
    *pending = Some(outputs);
    let now = Instant::now();
    match commits {
        Some(span) => span.last = now,
        None => {
            *commits = Some(CommitSpan {
                first: now,
                last: now,
            })
        }
    }
    completion.wait().map_err(device)
}

impl ExecutableStage {
    /// The caller retains both checked launches and their tentative advances.
    /// This performs physical work only; it cannot publish logical acceptance.
    pub(crate) fn execute_stage(
        &self,
        launch: &mut ValidatedTargetLaunch,
        input: Option<&Tensor>,
        commits: &mut Option<CommitSpan>,
    ) -> Result<StageResult, SubmitError> {
        if launch.domain() != &self.domain {
            return Err(invalid("stage launch belongs to another resource domain"));
        }
        let (core, workspace, outputs, readout_workspace, readout_output) =
            launch.submission_parts_mut();
        if !Rc::ptr_eq(core.store(), &self.store)
            || !core.store().belongs_to_device(&self.device)
            || !matches!(core.tokens(), TargetTokens::Host)
            || !matches!(
                core.advances(),
                [magnitude_state::TentativeAdvance::Accepted(_)]
            )
            || core.conditioning().iter().any(Option::is_some)
            || core.conditioning_slices().iter().any(|s| !s.is_empty())
            || core.batch().actual_slots() != 1
            || core.batch().class().segments() != 1
        {
            return Err(invalid(
                "stage requires its local store and exclusive launch class",
            ));
        }
        let batch = core.batch().upload();
        let rows = batch.class.rows() as u64;
        let view = self.assignment.view();
        let selected = self
            .graphs
            .blocks
            .iter()
            .filter(|block| block.prepared.rows == rows)
            .collect::<Vec<_>>();
        if selected.len() != view.blocks().len()
            || selected
                .iter()
                .zip(view.layers())
                .any(|(block, (local, global, _))| {
                    (block.prepared.local, block.prepared.global) != (local, global)
                })
        {
            return Err(invalid(
                "stage graphs do not exactly cover assigned original blocks",
            ));
        }
        let mut pending: [Option<NativeGraphOutputs>; 2] = [None, None];
        let outcome = (|| {
            let state = RowState(core);
            let controls = GraphControls::new(&batch)?;
            let mut hidden;
            let mut current;
            if self.assignment.owns_entry() {
                if input.is_some() || !matches!(state.tokens(), TargetTokens::Host) {
                    return Err(invalid(
                        "entry stage requires uploaded token IDs, not foreign activation",
                    ));
                }
                let (_, entry, bound) = self
                    .graphs
                    .entries
                    .iter()
                    .find(|(r, _, _)| *r == rows)
                    .ok_or_else(|| invalid("entry row class was not prepared"))?;
                let mut active = workspace.slot_mut().activate(&entry.plan).map_err(device)?;
                active
                    .write_input(&entry.tokens, &token_rows(batch.tokens))
                    .map_err(device)?;
                let ready = active
                    .attach(
                        bound.bindings(),
                        outputs[0]
                            .activate(&entry.plan)
                            .map_err(SubmitError::Invariant)?,
                    )
                    .map_err(device)?;
                run_completed(ready, &mut pending[0], commits)?;
                hidden = pending[0]
                    .as_ref()
                    .and_then(|o| o.exported(&entry.hidden))
                    .ok_or_else(|| invalid("completed entry did not export hidden rows"))?;
                current = Some(0);
            } else {
                let input =
                    input.ok_or_else(|| invalid("suffix has no local handoff activation"))?;
                if !input.device().same_device(&self.device)
                    || input.element() != Element::f32()
                    || input.extents() != [rows, view.decoder().hidden]
                {
                    return Err(invalid(
                        "handoff device, representation or geometry differs",
                    ));
                }
                hidden = input.clone();
                current = None;
            }
            let mut component = 0;
            for block in selected {
                let graph = &block.prepared.graph;
                let next = current.map_or(0, |previous| 1 - previous);
                let mut active = workspace.slot_mut().activate(&graph.plan).map_err(device)?;
                let mut bindings = block.bound.bindings();
                bindings.set(&graph.hidden, &hidden).map_err(device)?;
                bind_block_state(
                    &state,
                    &controls,
                    block.prepared.global.kv_layer(),
                    component,
                    &graph.state,
                    &graph.controls,
                    &mut active,
                    &mut bindings,
                )?;
                let ready = active
                    .attach(
                        bindings,
                        outputs[next]
                            .activate(&graph.plan)
                            .map_err(SubmitError::Invariant)?,
                    )
                    .map_err(device)?;
                run_completed(ready, &mut pending[next], commits)?;
                // Completion precedes both consuming and recycling producer storage.
                drop(hidden);
                if let Some(previous) = current {
                    outputs[previous]
                        .recycle(
                            pending[previous]
                                .take()
                                .ok_or_else(|| invalid("stage producer output is absent"))?,
                        )
                        .map_err(SubmitError::Invariant)?;
                }
                hidden = pending[next]
                    .as_ref()
                    .and_then(|o| o.exported(&graph.output))
                    .ok_or_else(|| invalid("completed block did not export hidden rows"))?;
                current = Some(next);
                if let BlockStatePorts::Recurrent(ports) = &graph.state {
                    component += ports.len();
                }
            }
            if !hidden.device().same_device(&self.device)
                || hidden.element() != Element::f32()
                || hidden.extents() != [rows, view.decoder().hidden]
            {
                return Err(invalid(
                    "completed activation does not match stage boundary contract",
                ));
            }
            if !self.assignment.owns_readout() {
                let began = Instant::now();
                let bytes = hidden.read_to_host().map_err(device)?;
                return Ok(StageResult::Boundary {
                    rows,
                    bytes,
                    elapsed: began.elapsed(),
                });
            }
            let graphs = self
                .graphs
                .readout
                .as_ref()
                .ok_or_else(|| invalid("final stage readout family is absent"))?;
            let output = readout_output
                .take()
                .ok_or_else(|| invalid("final stage readout output lease is absent"))?;
            let mut submitter = StepSubmitter::new(&self.device);
            let result = queue_readout(
                graphs,
                |_| Err(invalid("pipeline excludes draft taps")),
                &batch,
                &hidden,
                readout_workspace,
                output,
                &mut submitter,
            );
            let flushed = submitter.flush();
            // Observe already-submitted work even when queueing or flush failed.
            let waited = wait_all(std::mem::take(&mut submitter.completions));
            let readout = result?;
            flushed?;
            waited?;
            if let Some(span) = commits {
                span.last = Instant::now();
            }
            Ok(StageResult::Readout(readout))
        })();
        // The closure has dropped every hidden tensor view. Runs were observed
        // even on failure, so completed output arenas can now return safely.
        let mut cleanup = Ok(());
        for (lease, pending) in outputs.iter_mut().zip(pending) {
            if let Some(output) = pending {
                cleanup = cleanup.and(lease.recycle(output).map_err(SubmitError::Invariant));
            }
        }
        match outcome {
            Err(error) => Err(error),
            Ok(result) => {
                cleanup?;
                Ok(result)
            }
        }
    }
}
