//! Source call construction: body enumeration at a checked call, the
//! selected child's construction, and its splice into the caller.
use super::*;

/// Owned call construction progress. Child bodies and physical payloads belong
/// to this lexical call; no borrowed parent builder survives a pause.
pub(crate) struct CallConstruction<B: seismic_native_target::TargetFamily> {
    call: NodeId,
    location: crate::candidate_domain::CallLocation,
    arguments: Vec<ValueBinding>,
    bodies: Vec<CallBody>,
    selected: Option<usize>,
    begun: bool,
    child: Option<ConstructedCandidate<B>>,
}

struct CallBody {
    candidate: seismic_lang::entry::Candidate,
    selection: crate::candidate_domain::BodySelection,
    initialization: Result<(), seismic_lang::initialization::InitializationFailure>,
}

impl<B: seismic_native_target::TargetFamily> CallConstruction<B> {
    pub(crate) fn location(&self) -> &crate::candidate_domain::CallLocation {
        &self.location
    }
    pub(crate) fn choice(&self) -> Option<crate::candidate_domain::BodyChoice> {
        self.selected
            .is_none()
            .then(|| crate::candidate_domain::BodyChoice {
                path: crate::candidate_domain::CallPath(vec![self.location.clone()]),
                alternatives: self
                    .bodies
                    .iter()
                    .map(|body| body.selection.clone())
                    .collect(),
            })
    }
    pub(crate) fn select(&mut self, selection: crate::candidate_domain::BodySelection) {
        assert!(self.selected.is_none(), "call body was already selected");
        self.selected = Some(
            self.bodies
                .iter()
                .position(|body| body.selection == selection)
                .expect("selected body is absent from this checked call"),
        );
    }
    pub(crate) fn initialization_pending(
        &self,
    ) -> Option<seismic_lang::initialization::InitializationFailure> {
        self.bodies[self.selected.expect("call body must be selected")]
            .initialization
            .as_ref()
            .err()
            .cloned()
    }
}

fn source_node_path(function: &SemanticFunction, target: NodeId) -> Vec<u32> {
    fn find(
        function: &SemanticFunction,
        region: seismic_lang::ids::RegionId,
        target: NodeId,
        path: &mut Vec<u32>,
    ) -> bool {
        for (ordinal, (id, node)) in function.nodes(region).enumerate() {
            path.push(u32::try_from(ordinal).expect("source node ordinal exceeds u32"));
            if id == target {
                return true;
            }
            let children = match node.view() {
                SemanticNodeView::If {
                    then, otherwise, ..
                } => vec![then, otherwise],
                SemanticNodeView::Loop { body, .. } => vec![body],
                _ => Vec::new(),
            };
            for (arm, child) in children.into_iter().enumerate() {
                path.push(arm as u32);
                if find(function, child, target, path) {
                    return true;
                }
                path.pop();
            }
            path.pop();
        }
        false
    }
    let mut path = Vec::new();
    assert!(
        find(function, function.root(), target, &mut path),
        "source call is absent from its function"
    );
    path
}

impl<'a, B: seismic_native_target::TargetFamily> ImplementationBuilder<'a, B> {
    pub(crate) fn begin_call(
        &mut self,
        call: NodeId,
        arguments: &[ValueBinding],
        selections: &crate::portable::BindingSelections,
        contents: &crate::portable::initialization::StorageContents,
        initialized_arguments: &[crate::portable::initialization::CallArgument<'_>],
        binders: &[seismic_lang::expr::SymbolId],
    ) -> CallConstruction<B> {
        self.inner.begin_call(
            call,
            arguments,
            selections,
            contents,
            initialized_arguments,
            binders,
        )
    }

    pub(crate) fn begin_call_child(
        &mut self,
        progress: &mut CallConstruction<B>,
        selections: &crate::portable::BindingSelections,
        contents: &crate::portable::initialization::StorageContents,
        binders: &[seismic_lang::expr::SymbolId],
    ) -> Option<crate::portable::construction::SourceConstruction<B>> {
        self.inner
            .begin_call_child(progress, selections, contents, binders)
    }

    pub(crate) fn complete_call_child(
        &mut self,
        progress: &mut CallConstruction<B>,
        child: ConstructedCandidate<B>,
    ) {
        self.inner.complete_call_child(progress, child)
    }

    pub(crate) fn finish_call(
        &mut self,
        progress: CallConstruction<B>,
        selections: &crate::portable::BindingSelections,
        contents: &mut crate::portable::initialization::StorageContents,
        initialized_arguments: &[crate::portable::initialization::CallArgument<'_>],
        binders: &[seismic_lang::expr::SymbolId],
    ) -> Vec<(SemanticValueId, ValueBinding)> {
        self.inner.finish_call(
            progress,
            selections,
            contents,
            initialized_arguments,
            binders,
        )
    }
}

impl<'a, B: seismic_native_target::TargetFamily> internals::Builder<'a, B> {
    pub(super) fn begin_call(
        &mut self,
        call: NodeId,
        arguments: &[ValueBinding],
        selections: &crate::portable::BindingSelections,
        contents: &crate::portable::initialization::StorageContents,
        initialized_arguments: &[crate::portable::initialization::CallArgument<'_>],
        binders: &[seismic_lang::expr::SymbolId],
    ) -> CallConstruction<B> {
        let (family_id, call_inputs, call_outputs) = match self.function.node(call).view() {
            SemanticNodeView::Call {
                family,
                inputs,
                outputs,
            } => (family, inputs, outputs),
            _ => panic!("call construction requires a semantic Call node"),
        };
        assert_eq!(
            arguments.len(),
            call_inputs.len(),
            "call argument arity differs"
        );
        let family = self.program.family(family_id);
        let reference = family.reference().candidate();
        let reference_function = self.program.function(reference.function);
        let mut bodies = Vec::new();
        for candidate in std::iter::once(reference).chain(family.alternatives()) {
            let backend_matches = match candidate.kind {
                CandidateKind::Portable => true,
                CandidateKind::Lowering { backend } | CandidateKind::Helper { backend } => {
                    backend == B::NAME
                }
            };
            if !backend_matches
                || candidate
                    .requires
                    .iter()
                    .any(|capability| !self.target.supports_capability(*capability))
            {
                continue;
            }
            let function = self.program.function(candidate.function);
            // The checker established the reference's obligations over source
            // facts. Construction splices any body, the reference included,
            // only where the reached contents establish them too; where they
            // do not, the construction stays a typed Initialization pending
            // instead of splicing a call it cannot account for.
            let initialization = contents.applicable(
                &mut crate::portable::initialization_context(self.arena, selections, binders),
                function.initialization(),
                reference_function.initialization(),
                initialized_arguments,
            );
            let contract = FunctionContract::derive(function);
            assert_eq!(
                contract.parameters.len(),
                arguments.len(),
                "child parameter arity differs"
            );
            assert_eq!(
                contract.results.len(),
                call_outputs.len(),
                "child result arity differs"
            );
            let mode = match candidate.kind {
                CandidateKind::Portable => crate::portable::SemanticMode::Portable,
                _ => crate::portable::SemanticMode::AuthoredBackend,
            };
            bodies.push(CallBody {
                candidate: candidate.clone(),
                selection: crate::candidate_domain::BodySelection::new(
                    self.program,
                    function,
                    mode,
                ),
                initialization: initialization.clone(),
            });
            if candidate.kind == CandidateKind::Portable {
                bodies.push(CallBody {
                    candidate: candidate.clone(),
                    selection: crate::candidate_domain::BodySelection::new(
                        self.program,
                        function,
                        crate::portable::SemanticMode::PortableParallel,
                    ),
                    initialization,
                });
            }
        }
        assert!(
            !bodies.is_empty(),
            "checked call has no source-compatible body"
        );
        let selected = (bodies.len() == 1).then_some(0);
        let occurrence = self.state.call_occurrences.entry(call).or_default();
        let location = crate::candidate_domain::CallLocation {
            body: self.function.stable(),
            source_definition: self.function.source_definition(),
            node: source_node_path(self.function, call),
            occurrence: *occurrence,
        };
        *occurrence += 1;
        CallConstruction {
            call,
            location,
            arguments: arguments.to_vec(),
            bodies,
            selected,
            begun: false,
            child: None,
        }
    }

    pub(super) fn begin_call_child(
        &mut self,
        progress: &mut CallConstruction<B>,
        selections: &crate::portable::BindingSelections,
        contents: &crate::portable::initialization::StorageContents,
        binders: &[seismic_lang::expr::SymbolId],
    ) -> Option<crate::portable::construction::SourceConstruction<B>> {
        if progress.begun {
            return None;
        }
        assert!(
            progress.initialization_pending().is_none(),
            "unresolved initialization must remain a suspended construction"
        );
        let body = &progress.bodies[progress
            .selected
            .expect("select a child before constructing it")];
        progress.begun = true;
        let site = CallSite::Spliced {
            call: progress.call,
            arguments: &progress.arguments,
            bindings: &self.state.bindings,
            contents,
            binders,
            storage: self.state.construction.storage(),
            selections,
        };
        let builder = ImplementationBuilder::new(
            self.arena,
            self.program,
            self.program.function(body.candidate.function),
            self.target,
            self.registry,
            self.constants,
            self.precision,
            body.candidate.applicability,
            site,
            None,
        );
        let (child, _) = crate::portable::construction::SourceConstruction::begin(
            builder,
            body.selection.mode(),
        );
        Some(child)
    }

    pub(super) fn complete_call_child(
        &mut self,
        progress: &mut CallConstruction<B>,
        child: ConstructedCandidate<B>,
    ) {
        assert!(
            progress.child.replace(child).is_none(),
            "selected child completed twice"
        );
    }

    pub(super) fn finish_call(
        &mut self,
        progress: CallConstruction<B>,
        selections: &crate::portable::BindingSelections,
        contents: &mut crate::portable::initialization::StorageContents,
        initialized_arguments: &[crate::portable::initialization::CallArgument<'_>],
        binders: &[seismic_lang::expr::SymbolId],
    ) -> Vec<(SemanticValueId, ValueBinding)> {
        let CallConstruction {
            call,
            arguments: parameter_bindings,
            child,
            ..
        } = progress;
        let (family_id, call_outputs) = match self.function.node(call).view() {
            SemanticNodeView::Call {
                family, outputs, ..
            } => (family, outputs),
            _ => unreachable!("call progress has a checked call owner"),
        };
        let reference_function = self
            .program
            .function(self.program.family(family_id).reference().function());
        let child = child.expect("selected call child has not completed");
        contents
            .call(
                &mut crate::portable::initialization_context(self.arena, selections, binders),
                reference_function.initialization(),
                initialized_arguments,
            )
            .unwrap_or_else(|error| {
                panic!("call body admitted by its reached contents lost its initialized input: {error:?}")
            });
        let parts = child.into_parts();
        let child_ir = parts
            .executable
            .into_ir()
            .into_importable()
            .unwrap_or_else(|error| {
                panic!("spliced child produced a root-only executable: {error:?}")
            });
        let imported =
            self.state
                .construction
                .import(self.arena, self.state.schedule_region, child_ir, &[]);
        let results = parts.bindings.import_into(
            &mut self.state.bindings,
            &parameter_bindings,
            &imported,
            self.state.construction.storage(),
            contents,
        );
        assert_eq!(
            results.len(),
            call_outputs.len(),
            "child complete result arity differs"
        );
        self.state.choices.extend(parts.choices);
        self.state
            .numerical_children
            .selected_child(&parts.numerical_applicability);
        self.state.callees.push(parts.provenance.root);
        self.state.callees.extend(parts.provenance.callees);
        self.state.constraints.push(parts.semantic_coverage.node());
        self.state.constraints.push(parts.hard_constraints);
        call_outputs.iter().copied().zip(results).collect()
    }
}
