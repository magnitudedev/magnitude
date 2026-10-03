//! Paired reservation/submission within the ordinary target lifecycle.
use super::*;
use crate::pipeline::ValidatedPipelineLaunch;
use magnitude_state::PipelineSequenceState;

pub(super) struct PrefixLaunchReservation {
    request: RequestId,
    advance: OwnedStateAdvance,
    graph_workspace: NativeGraphWorkspaceLease,
    graph_outputs: [NativeGraphOutputLease; 2],
    readout_workspace: NativeGraphWorkspaceLease,
    readout_output: NativeGraphOutputLease,
}

impl<F: ProgramFamily> ExecutorDomain<F> {
    pub(super) fn pipeline_requirements(
        &self,
        operations: &[Operation],
    ) -> Result<Option<Vec<RowDemand>>, DomainError> {
        let Some(prefix) = &self.pipeline_owner else {
            return Ok(None);
        };
        let operation = plain_forward(operations)?;
        let Operation::Forward {
            request,
            position,
            tokens,
            ..
        } = operation
        else {
            unreachable!()
        };
        if self.head_store.is_some() || self.execution.policy().limits().lookahead {
            return Err(DomainError::Input(
                "pipeline head/lookahead profile is unsupported".into(),
            ));
        }
        let a = self
            .pipeline_prefix
            .get(request)
            .ok_or_else(|| DomainError::Input("prefix request is not idle".into()))?;
        let b = self
            .target
            .get(request)
            .ok_or_else(|| DomainError::Input("suffix request is not idle".into()))?;
        if a.position() != *position || b.position() != *position {
            return Err(DomainError::Input(
                "paired accepted positions differ from requested position".into(),
            ));
        }
        for (state, store) in [(a, prefix.store.as_ref()), (b, self.target_store.as_ref())] {
            if super::target::reserved_segments(store, state.span_count(), tokens.len()) > 1 {
                return Err(DomainError::Input(
                    "explicit pipeline supports one history segment".into(),
                ));
            }
        }
        Ok(Some(a.demands(tokens.len())))
    }

    pub(super) fn can_reserve_pipeline(&self, demands: &[RowDemand]) -> Result<(), CapacityError> {
        let prefix = self.pipeline_owner.as_ref().expect("paired requirements");
        for (resource, required, available) in [
            (
                ResourceKind::Workspace,
                2,
                prefix.pool.available_workspace(),
            ),
            (ResourceKind::Output, 3, prefix.pool.available_output()),
            (
                ResourceKind::RecurrentBanks,
                1,
                prefix.store.available_banks(),
            ),
            (
                ResourceKind::Workspace,
                1,
                self.resources.target_readout_graph().available_workspace(),
            ),
            (
                ResourceKind::Output,
                1,
                self.resources.target_readout_graph().available_output(),
            ),
        ] {
            if available < required {
                return Err(CapacityError {
                    resource,
                    required: required as u64,
                    available: available as u64,
                });
            }
        }
        holds(&prefix.store, demands)
    }

    pub(super) fn reserve_pipeline(
        &mut self,
        bindings: &mut StateBindings<F>,
        operations: &[Operation],
    ) -> Result<DomainReservation, DomainError> {
        self.probe_memory()?;
        let requirements = self.requirements(bindings, operations)?;
        self.provision(bindings, operations)?;
        self.can_reserve(&requirements)
            .map_err(DomainError::Capacity)?;
        let prefix = self.pipeline_owner.as_ref().expect("paired reservation");
        let graph = self.resources.target_graph();
        let readout = self.resources.target_readout_graph();
        // Every fallible lease acquisition precedes moving either source.
        let graph_workspace = graph.acquire_workspace().map_err(DomainError::Capacity)?;
        let graph_outputs = [
            graph.acquire_output().map_err(DomainError::Capacity)?,
            graph.acquire_output().map_err(DomainError::Capacity)?,
        ];
        let readout_workspace = readout.acquire_workspace().map_err(DomainError::Capacity)?;
        let readout_output = readout.acquire_output().map_err(DomainError::Capacity)?;
        let prefix_workspace = prefix
            .pool
            .acquire_workspace()
            .map_err(DomainError::Capacity)?;
        let prefix_outputs = [
            prefix
                .pool
                .acquire_output()
                .map_err(DomainError::Capacity)?,
            prefix
                .pool
                .acquire_output()
                .map_err(DomainError::Capacity)?,
        ];
        let prefix_readout_workspace = prefix
            .pool
            .acquire_workspace()
            .map_err(DomainError::Capacity)?;
        let prefix_readout_output = prefix
            .pool
            .acquire_output()
            .map_err(DomainError::Capacity)?;
        let request = operations[0].request();
        let sources = vec![
            self.pipeline_prefix
                .remove(&request)
                .expect("prefix preflight"),
            self.target.remove(&request).expect("suffix preflight"),
        ];
        let source = PipelineSequenceState::new(sources).map_err(|failure| {
            let (sources, error) = *failure;
            self.restore_pipeline_sources(request, sources);
            DomainError::from(error)
        })?;
        let transaction = source.begin(operations[0].row_count()).map_err(|failure| {
            let (source, error) = *failure;
            self.restore_pipeline_sources(request, source.into_stages());
            DomainError::from(error)
        })?;
        let [a, b]: [OwnedStateAdvance; 2] = transaction
            .into_stages()
            .try_into()
            .unwrap_or_else(|_| unreachable!("two stage sources"));
        Ok(DomainReservation {
            requirements,
            resources: ReservedResources::Target(TargetGraphReservation::Launch(
                TargetLaunchReservation {
                    advances: vec![b],
                    graph_workspace,
                    graph_outputs,
                    readout_workspace,
                    readout_output,
                    priming: None,
                    pipeline: Some(PrefixLaunchReservation {
                        request,
                        advance: a,
                        graph_workspace: prefix_workspace,
                        graph_outputs: prefix_outputs,
                        readout_workspace: prefix_readout_workspace,
                        readout_output: prefix_readout_output,
                    }),
                },
            )),
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn pipeline_stage_launch(
        &self,
        operation: &Operation,
        advance: OwnedStateAdvance,
        store: &Rc<StateStore>,
        domain: &ResourceDomainId,
        graph_workspace: NativeGraphWorkspaceLease,
        graph_outputs: [NativeGraphOutputLease; 2],
        readout_workspace: NativeGraphWorkspaceLease,
        readout_output: NativeGraphOutputLease,
        publish: bool,
    ) -> Result<ValidatedTargetLaunch, (SequenceState, DomainError)> {
        let advance = TentativeAdvance::Accepted(advance);
        let preparation = (|| {
            // Plain text has no bidirectional media reads. Common input/row
            // construction remains the ordinary implementation, with local state.
            let (mut slot, slices) = self.target_slot_for_store(
                operation,
                &advance,
                store,
                &vec![false; store.history_reads()],
            )?;
            if !slices.is_empty() {
                return Err("pipeline conditioning is unsupported".into());
            }
            if !publish {
                for row in &mut slot.rows {
                    row.demand = Demand::NONE;
                    row.select = None;
                }
            }
            ValidatedTargetBatch::covering(
                &[slot],
                self.definition.decoder.vocabulary as usize,
                self.target_class_limits(),
                1,
            )
            .map_err(|e| e.to_string())
        })();
        let batch = match preparation {
            Ok(batch) => batch,
            Err(error) => {
                let TentativeAdvance::Accepted(advance) = advance else {
                    unreachable!()
                };
                return Err((advance.abort(), DomainError::Input(error)));
            }
        };
        let inputs = TargetLaunchInputs::new(
            batch,
            TargetTokens::Host,
            vec![advance],
            vec![None],
            vec![vec![]],
            graph_workspace,
            graph_outputs,
            readout_workspace,
            readout_output,
        );
        ValidatedTargetLaunch::new(
            inputs,
            store,
            domain,
            self.definition.decoder.hidden as usize,
        )
        .map_err(|(inputs, error)| {
            let (_, mut advances, _, _) = inputs.into_parts();
            let TentativeAdvance::Accepted(advance) = advances.pop().expect("one advance") else {
                unreachable!()
            };
            (advance.abort(), DomainError::Invariant(error))
        })
    }

    pub(super) fn launch_pipeline_target(
        &mut self,
        operations: &[Operation],
        reservation: TargetLaunchReservation,
    ) -> Result<TargetWork<F>, Unsubmitted> {
        let TargetLaunchReservation {
            mut advances,
            graph_workspace,
            graph_outputs,
            readout_workspace,
            readout_output,
            priming: _,
            pipeline,
        } = reservation;
        let PrefixLaunchReservation {
            request,
            advance: a,
            graph_workspace: aw,
            graph_outputs: ao,
            readout_workspace: arw,
            readout_output: aro,
        } = pipeline.expect("paired reservation");
        let b = advances.pop().expect("one suffix advance");
        let operation = match plain_forward(operations) {
            Ok(operation)
                if operation.request() == request
                    && operation.row_count() == a.rows()
                    && operation.row_count() == b.rows()
                    && matches!(operation, Operation::Forward { position, .. }
                    if *position == a.position() && *position == b.position()) =>
            {
                operation
            }
            _ => {
                self.pipeline_prefix.insert(request, a.abort());
                self.target.insert(request, b.abort());
                return Err(Unsubmitted::Refused(DomainError::Input(
                    "pipeline submission differs from its reserved request or row extent".into(),
                )));
            }
        };
        let prefix = self.pipeline_owner.as_ref().expect("paired owner");
        let first = match self.pipeline_stage_launch(
            operation,
            a,
            &prefix.store,
            prefix.pool.domain(),
            aw,
            ao,
            arw,
            aro,
            false,
        ) {
            Ok(first) => first,
            Err((a, error)) => {
                self.pipeline_prefix.insert(request, a);
                self.target.insert(request, b.abort());
                return Err(Unsubmitted::Refused(error));
            }
        };
        let second = match self.pipeline_stage_launch(
            operation,
            b,
            &self.target_store,
            self.domain.id(),
            graph_workspace,
            graph_outputs,
            readout_workspace,
            readout_output,
            true,
        ) {
            Ok(second) => second,
            Err((b, error)) => {
                self.pipeline_prefix.insert(request, abort_launch(first));
                self.target.insert(request, b);
                return Err(Unsubmitted::Refused(error));
            }
        };
        let prefix_device = prefix.memory.borrow().device().clone();
        let launch =
            ValidatedPipelineLaunch::new([first, second], [&prefix_device, self.domain.device()])
                .map_err(|failure| {
                let ([a, b], error) = *failure;
                self.pipeline_prefix.insert(request, abort_launch(a));
                self.target.insert(request, abort_launch(b));
                Unsubmitted::Refused(DomainError::Invariant(error))
            })?;
        let started = Instant::now();
        // This is the same qualified executor. From here a failure discards the
        // request; releasing reservations does not roll back completed GPU writes.
        let (submission, prefix) = self
            .family
            .submit_pipeline_target(launch)
            .map_err(|error| Unsubmitted::Failed(DomainError::Submit(error)))?;
        let Operation::Forward {
            kind, committed, ..
        } = operation
        else {
            unreachable!()
        };
        Ok(TargetWork {
            requests: vec![(request, operation.row_count(), None, *kind, *committed)],
            pipeline_prefix: Some(prefix),
            submission,
            priming: None,
            launch_trace: None,
            started,
            runnable: started,
            previous_selection: self.selection_read.take(),
            id: self.flight_id(),
            continuation: None,
        })
    }
}
fn abort_launch(launch: ValidatedTargetLaunch) -> SequenceState {
    let (core, _) = launch.into_submission_parts();
    let (_, mut advances, _, _) = core.into_parts();
    let TentativeAdvance::Accepted(advance) = advances.pop().expect("one advance") else {
        unreachable!()
    };
    advance.abort()
}

fn plain_forward(operations: &[Operation]) -> Result<&Operation, DomainError> {
    let [Operation::Forward {
        tokens,
        committed,
        conditioning,
        prime,
        kind,
        ..
    }] = operations
    else {
        return Err(DomainError::Input(
            "explicit pipeline requires one plain forward".into(),
        ));
    };
    if !matches!(tokens.len(), 1 | 2)
        || *committed != tokens.len()
        || conditioning.is_some()
        || prime.is_some()
        || *kind == WorkKind::Verify
    {
        return Err(DomainError::Input(
            "unsupported explicit pipeline forward profile".into(),
        ));
    }
    operations[0]
        .validate()
        .map_err(|error| DomainError::Input(error.to_string()))?;
    Ok(&operations[0])
}

#[cfg(test)]
mod tests {
    use super::*;
    fn forward(rows: usize) -> Operation {
        Operation::Forward {
            request: RequestId(1),
            kind: WorkKind::Prefill,
            tokens: vec![TokenId(1); rows],
            position: 0,
            conditioning: None,
            demand: Demand::NONE,
            select: vec![],
            committed: rows,
            prime: None,
        }
    }
    #[test]
    fn plain_profile_accepts_one_or_two_whole_rows_only() {
        for rows in [1, 2] {
            assert!(plain_forward(&[forward(rows)]).is_ok());
        }
        for rows in [0, 3, 8] {
            assert!(plain_forward(&[forward(rows)]).is_err());
        }
        assert!(plain_forward(&[]).is_err());
        assert!(plain_forward(&[forward(1), forward(1)]).is_err());
    }
    #[test]
    fn plain_profile_refuses_speculation_verification_and_priming() {
        for change in 0..3 {
            let mut operation = forward(2);
            let Operation::Forward {
                committed,
                kind,
                prime,
                ..
            } = &mut operation
            else {
                unreachable!()
            };
            match change {
                0 => *committed = 1,
                1 => *kind = WorkKind::Verify,
                _ => {
                    *prime = Some(Priming {
                        tokens: vec![TokenId(1)],
                        position: 0,
                        draft_from: 1,
                    })
                }
            }
            assert!(plain_forward(&[operation]).is_err());
        }
    }
}
