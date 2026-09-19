use super::*;
use crate::reduction::structured::{ElementRetention, Merge, Reduction, StateRetention, StepOperand, StepState};

impl Builder<'_> {
    pub(super) fn coupled_reduction(&mut self, mut operation: Reduction, occurrence: &OccurrenceId, guard: &Guard) -> Result<RegionId, String> {
        let captured = operation.capture_inputs(&mut self.family.template.vars);
        let mut children = Vec::new();
        for (index, statement) in captured.into_iter().enumerate() {
            let origin = child_origin(occurrence, "reduction.capture", index, super::operation(&statement));
            children.push(self.leaf(statement, &origin, guard)?);
        }
        let state = operation.state.iter().map(|value| value.ty.clone()).collect::<Vec<_>>();
        let merge_source = operation.merge.source().ok_or("source reduction merge callback is absent")?.clone();
        let (merge, merge_body) = self.callback(&merge_source, &state, &state, &state, occurrence, guard, CallbackRole::Merge)?;
        operation.implementation = Some(merge);
        let mut callbacks = vec![ReductionCallback { role: CallbackRole::Merge, body: merge_body }];
        if let Some(step_source) = operation.step.as_ref().map(|step| step.call.source().cloned()) {
            let step_source = step_source.ok_or("source reduction step callback is absent")?;
            let leaves = operation.inputs.iter().map(|input| crate::reduction::structured::slice(input, operation.axis,
                &crate::reduction::structured::integer(0, operation.span), operation.span).ty).collect::<Vec<_>>();
            let (step, body) = self.callback(&step_source, &state, &leaves, &state, occurrence, guard, CallbackRole::Step)?;
            operation.step.as_mut().unwrap().implementation = Some(step);
            callbacks.push(ReductionCallback { role: CallbackRole::Step, body });
        }
        let open_step = callbacks.iter().find(|callback| callback.role == CallbackRole::Step
            && self.closed_region(callback.body).is_none()).map(|callback| callback.body);
        let implementation = operation.step.as_ref().and_then(|step| step.implementation.clone());
        let reduction = self.reduction(operation, occurrence, guard)?;
        if let Some(body) = open_step {
            self.callback_permissions(reduction, body, implementation.as_ref().ok_or("step callback bindings are absent")?, occurrence, guard)?;
        }
        let callback_effects = callbacks.iter().map(|callback| self.family.regions[callback.body.0].effects.clone()).collect::<Vec<_>>();
        let retained = &mut self.family.regions[reduction.0];
        for effect in callback_effects { merge_effects(&mut retained.effects, &effect); }
        let RegionKind::Reduction(operation) = &mut retained.kind else { unreachable!() };
        operation.callbacks = callbacks;
        children.push(reduction);
        Ok(self.sequence(children, occurrence, guard))
    }

    fn callback(&mut self, source: &Expr, left: &[Ty], right: &[Ty], output: &[Ty], occurrence: &OccurrenceId, guard: &Guard,
        role: CallbackRole) -> Result<(Merge, RegionId), String> {
        let mut call = source.clone();
        let ExprKind::Call { callee, args, elem_args, .. } = &mut call.kind else { return Err("source callback is not a construct/function call".into()); };
        let definition = self.program.functions.iter().find(|definition| &definition.name == callee).ok_or("missing callback definition")?;
        let parameters = left.iter().chain(right).chain(output).map(|ty| {
            let id = self.family.template.vars.len();
            self.family.template.vars.push(Var { name: format!("callback_{id}"), ty: ty.clone(), span: source.span, kind: VarKind::Local });
            Expr { kind: ExprKind::Var(id), ty: ty.clone(), sym: None, span: source.span }
        }).collect::<Vec<_>>();
        for (name, element) in definition.elem_params.iter().zip(elem_args.iter_mut()) {
            for ((_, formal), actual) in definition.params.iter().zip(&parameters) {
                if matches!(formal.shaped().map(|shape| &shape.elem), Some(Elem::Param(parameter)) if parameter == name) {
                    *element = actual.ty.shaped().ok_or("generic callback operand is not shaped")?.elem.clone();
                    break;
                }
            }
        }
        *args = parameters.clone();
        let origin = child_origin(occurrence, "reduction.callback", usize::from(role == CallbackRole::Step), format!("{role:?}:{callee}"));
        let body = self.call(&call, &origin, guard)?;
        Ok((Merge { left: parameters[..left.len()].to_vec(), right: parameters[left.len()..left.len() + right.len()].to_vec(),
            output: parameters[left.len() + right.len()..].to_vec(),
            // Ownership never reasons from an empty placeholder. The retained
            // graph replaces this effect union before any actual execution.
            body: self.closed_region(body).unwrap_or_else(|| self.analysis_region(body)) }, body))
    }

    fn callback_permissions(&mut self, reduction: RegionId, body: RegionId, implementation: &Merge,
        occurrence: &OccurrenceId, guard: &Guard) -> Result<(), String> {
        let RegionKind::Reduction(retained) = &self.family.regions[reduction.0].kind else { unreachable!() };
        let mut operands = retained.operands.clone();
        let mut state = retained.state.clone();
        for (input, parameter) in implementation.right.iter().enumerate() {
            if !parameter.ty.shaped().is_some_and(|shape| matches!(shape.elem, Elem::Dtype(_)))
                || operands.iter().any(|(existing, _)| *existing == input) { continue; }
            let decision = self.decision(occurrence, guard, Decision { kind: DecisionKind::FoldOperand { input, ty: parameter.ty.clone() },
                alternatives: vec![Alternative::StepOperand(StepOperand::Private), Alternative::StepOperand(StepOperand::View)].into() }, input)?;
            self.callback_operand_permissions(body, implementation, input, &decision)?;
            operands.push((input, decision));
        }
        {
            let decision = match state.take() { Some(decision) => decision, None => self.decision(occurrence, guard, Decision { kind: DecisionKind::FoldState {
                fields: implementation.left.iter().map(|value| value.ty.clone()).collect(),
            }, alternatives: vec![Alternative::StepState(StepState::Separate), Alternative::StepState(StepState::Retained)].into() }, 0)? };
            let mut proof = CallbackRetention { builder: self, invalid: Vec::new(), unresolved: false };
            let result = StateRetention::new(implementation).and_then(|state| proof.state(body, state));
            if proof.unresolved {
                self.obligation(occurrence, &guard.with(decision.clone(), 1), DecisionClass::FoldState,
                    "callback alternatives have different intermediate state-publication facts");
            } else if result.as_ref().is_some_and(StateRetention::complete) {
                let invalid = std::mem::take(&mut proof.invalid);
                for active in invalid { self.family.requirements.push(Requirement { guard: active.with(decision.clone(), 1), nonnegative: Sym::constant(-1) }); }
            } else {
                self.family.requirements.push(Requirement { guard: guard.with(decision.clone(), 1), nonnegative: Sym::constant(-1) });
            }
            state = Some(decision);
        }
        let RegionKind::Reduction(retained) = &mut self.family.regions[reduction.0].kind else { unreachable!() };
        retained.operands = operands; retained.state = state;
        Ok(())
    }

    fn callback_operand_permissions(&mut self, body: RegionId, implementation: &Merge, input: usize, decision: &DecisionId) -> Result<(), String> {
        let mut union = implementation.clone(); union.body = self.analysis_region(body);
        if union.can_view_operand(input) { return Ok(()); }
        let retained = self.family.regions[body.0].clone();
        if self.closed_region(body).is_some() {
            self.family.requirements.push(Requirement { guard: retained.guard.with(decision.clone(), 1), nonnegative: Sym::constant(-1) });
            return Ok(());
        }
        match retained.kind {
            RegionKind::Choice { arms, .. } => for arm in arms { self.callback_operand_permissions(arm, implementation, input, decision)?; },
            RegionKind::Sequence(children) if children.len() == 1 => self.callback_operand_permissions(children[0], implementation, input, decision)?,
            _ => self.obligation(&retained.occurrence, &retained.guard.with(decision.clone(), 1), DecisionClass::FoldOperand,
                "callback operand aliases cross separately guarded regions"),
        }
        Ok(())
    }

    /// Obtain exact computation only when there is no implementation choice in
    /// this region. This is a semantic fact used by ownership proofs, not a
    /// selected baseline from an unresolved choice domain.
    pub(super) fn closed_region(&self, id: RegionId) -> Option<Vec<Stmt>> {
        match &self.family.regions[id.0].kind {
            RegionKind::Statement(statement) => Some(vec![statement.clone()]),
            RegionKind::Sequence(children) => {
                let mut statements = Vec::new(); for &child in children { statements.extend(self.closed_region(child)?); } Some(statements)
            },
            RegionKind::Repeated { header, body, .. } => {
                let mut header = header.clone(); let statements = self.closed_region(*body)?;
                match &mut header.kind {
                    StmtKind::Parallel { body, .. } | StmtKind::Owned { body, .. } | StmtKind::Range { body, .. } | StmtKind::Lanes { body, .. } => *body = statements,
                    _ => return None,
                } Some(vec![header])
            },
            RegionKind::Conditional { header, then, els } => {
                let mut header = header.clone(); let yes = self.closed_region(*then)?; let no = self.closed_region(*els)?;
                let StmtKind::If { then, els, .. } = &mut header.kind else { return None };
                *then = yes; *els = no; Some(vec![header])
            },
            RegionKind::Choice { arms, .. } if arms.len() == 1 => self.closed_region(arms[0]),
            RegionKind::Choice { .. } | RegionKind::Stream { .. } | RegionKind::Reduction(_) | RegionKind::Replicated { .. } => None,
        }
    }
}

struct CallbackRetention<'a, 'program> {
    builder: &'a Builder<'program>,
    invalid: Vec<Guard>,
    unresolved: bool,
}

impl CallbackRetention<'_, '_> {
    fn reject<T>(&mut self, id: RegionId) -> Option<T> {
        self.invalid.push(self.builder.family.regions[id.0].guard.clone()); None
    }

    fn state(&mut self, id: RegionId, mut state: StateRetention) -> Option<StateRetention> {
        match self.builder.family.regions[id.0].kind.clone() {
            RegionKind::Sequence(children) => {
                for child in children { state = self.state(child, state)?; }
                Some(state)
            },
            RegionKind::Choice { arms, .. } => {
                let mut joined: Option<StateRetention> = None;
                for arm in arms {
                    if let Some(next) = self.state(arm, state.clone()) {
                        if let Some(joined) = &mut joined { if !joined.merge(&next) { self.unresolved = true; } }
                        else { joined = Some(next); }
                    }
                }
                joined
            },
            RegionKind::Repeated { header: Stmt { kind: StmtKind::Owned { vars, tile, .. }, .. }, body, .. } => {
                let Some(element) = state.begin(&vars, &tile) else { return self.reject(id); };
                let element = self.element(body, element)?;
                if state.finish(&element) { Some(state) } else { self.reject(id) }
            },
            RegionKind::Statement(Stmt { kind: StmtKind::Owned { vars, tile, body }, .. }) => {
                let Some(mut element) = state.begin(&vars, &tile) else { return self.reject(id); };
                for statement in &body { if !element.statement(statement) { return self.reject(id); } }
                if state.finish(&element) { Some(state) } else { self.reject(id) }
            },
            _ => self.reject(id),
        }
    }

    fn element(&mut self, id: RegionId, mut element: ElementRetention) -> Option<ElementRetention> {
        match self.builder.family.regions[id.0].kind.clone() {
            RegionKind::Sequence(children) => {
                for child in children { element = self.element(child, element)?; }
                Some(element)
            },
            RegionKind::Choice { arms, .. } => {
                let mut joined: Option<ElementRetention> = None;
                for arm in arms {
                    if let Some(next) = self.element(arm, element.clone()) {
                        if let Some(joined) = &mut joined { if !joined.merge(&next) { self.unresolved = true; } }
                        else { joined = Some(next); }
                    }
                }
                joined
            },
            RegionKind::Statement(statement) => if element.statement(&statement) { Some(element) } else {
                if element.conditional_read(&statement) { self.unresolved = true; }
                self.reject(id)
            },
            // The concrete retention contract admits straight-line element
            // updates. More complex control remains an explicit coverage gap
            // when its legality depends on retained implementation choices.
            RegionKind::Conditional { .. } | RegionKind::Repeated { .. } | RegionKind::Replicated { .. }
            | RegionKind::Stream { .. } | RegionKind::Reduction(_) => self.reject(id),
        }
    }
}
