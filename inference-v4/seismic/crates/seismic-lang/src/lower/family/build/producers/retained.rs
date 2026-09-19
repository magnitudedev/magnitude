use super::*;

impl Builder<'_> {
    /// The lifetime proof sees every possible read and write. This projection is
    /// never executable: concatenating choice arms is a conservative effect
    /// union, not a choice of an implementation or an enumeration of paths.
    pub(in super::super) fn analysis_region(&self, id: RegionId) -> Vec<Stmt> {
        match &self.family.regions[id.0].kind {
            RegionKind::Statement(statement) => vec![statement.clone()],
            RegionKind::Sequence(children) =>
                children.iter().flat_map(|&child| self.analysis_region(child)).collect(),
            RegionKind::Choice { arms, .. } => {
                // Keep branch-local definitions below a control boundary. A
                // lifetime union must not promote a conditional allocation to
                // an unconditional reaching producer in the enclosing block.
                let span = crate::span::Span::default();
                vec![Stmt { id: None, span, kind: StmtKind::If {
                    cond: Expr { kind: ExprKind::Bool(true), ty: Ty::Scalar(crate::types::DType::Bool), sym: None, span },
                    then: arms.iter().flat_map(|&arm| self.analysis_region(arm)).collect(), els: Vec::new(),
                } }]
            },
            RegionKind::Repeated { header, body, .. } | RegionKind::Stream { header, body, .. } => {
                let mut header = header.clone();
                match &mut header.kind {
                    StmtKind::Parallel { body: statements, .. } | StmtKind::Owned { body: statements, .. }
                    | StmtKind::Range { body: statements, .. } | StmtKind::Lanes { body: statements, .. }
                    | StmtKind::LoadLoop { body: statements, .. } => *statements = self.analysis_region(*body),
                    _ => unreachable!("retained repetition has a control header"),
                }
                vec![header]
            },
            RegionKind::Replicated { index, count, body } => vec![Stmt { id: None,
                span: self.family.template.vars[*index].span, kind: StmtKind::Range {
                    var: *index, lo: Sym::constant(0), hi: count.clone(), body: self.analysis_region(*body),
                } }],
            RegionKind::Conditional { header, then, els } => {
                let mut header = header.clone();
                let StmtKind::If { then: yes, els: no, .. } = &mut header.kind else { unreachable!() };
                *yes = self.analysis_region(*then); *no = self.analysis_region(*els);
                vec![header]
            },
            RegionKind::Reduction(reduction) => {
                let mut operation = reduction.operation.clone();
                for callback in &reduction.callbacks {
                    let implementation = match callback.role {
                        CallbackRole::Merge => operation.implementation.as_mut(),
                        CallbackRole::Step => operation.step.as_mut().and_then(|step| step.implementation.as_mut()),
                    };
                    if let Some(implementation) = implementation { implementation.body = self.analysis_region(callback.body); }
                }
                vec![Stmt { id: None, span: operation.span, kind: StmtKind::Reduction(Box::new(operation)) }]
            },
        }
    }

    pub(in super::super) fn retained_producer_family(&mut self, children: Vec<RegionId>, occurrence: &OccurrenceId,
        guard: &Guard) -> Result<RegionId, String> {
        let analysis = children.iter().flat_map(|&child| self.analysis_region(child)).collect::<Vec<_>>();
        let producers = self.producer_decisions(&analysis, occurrence, guard)?;
        let mut rewritten = Vec::with_capacity(children.len());
        for child in children {
            rewritten.push(if producers.is_empty() { child } else { self.producer_region(child, &producers, &Guard::default())? });
        }
        Ok(self.sequence(rewritten, occurrence, guard))
    }

    /// Preserve source occurrence identities, callback bindings and original
    /// decisions. New regions allow a shared child to keep its original guard
    /// when different enclosing paths need different producer bindings.
    fn producer_region(&mut self, id: RegionId, producers: &Producers, inherited: &Guard) -> Result<RegionId, String> {
        let original = self.family.regions[id.0].clone();
        let occurrence = &original.occurrence;
        let mut guard = original.guard.clone();
        for choice in &inherited.choices { if !guard.choices.contains(choice) { guard.choices.push(choice.clone()); } }
        for predicate in &inherited.predicates { if !guard.predicates.contains(predicate) { guard.predicates.push(predicate.clone()); } }
        for clause in &inherited.one_of { if !guard.one_of.contains(clause) { guard.one_of.push(clause.clone()); } }
        let definition = match &original.kind {
            RegionKind::Statement(statement) | RegionKind::Repeated { header: statement, .. } => producer_definition(statement, producers).cloned(),
            _ => None,
        };
        if let Some((decision, ..)) = definition {
            let mut dependencies = producers.clone(); dependencies.retain(|_, (candidate, ..)| candidate != &decision);
            let stored = self.push(occurrence, &guard, original.kind.clone(), original.effects.clone());
            let stored = self.producer_region(stored, &dependencies, &guard.with(decision.clone(), 0))?;
            let removed = self.sequence(Vec::new(), occurrence, &guard.with(decision.clone(), 1));
            let summary = self.family.regions[stored.0].effects.clone();
            return Ok(self.push(occurrence, &guard, RegionKind::Choice { decision, arms: vec![stored, removed] }, summary));
        }
        let mut prefix = Vec::new();
        let replacement = match original.kind {
            RegionKind::Sequence(children) => {
                let children = children.into_iter().map(|child| self.producer_region(child, producers, &guard)).collect::<Result<_, _>>()?;
                self.sequence(children, occurrence, &guard)
            },
            RegionKind::Choice { decision, arms } => {
                let mut rewritten = Vec::with_capacity(arms.len()); let mut summary = Effects::default();
                for arm in arms {
                    let arm = self.producer_region(arm, producers, &guard)?;
                    merge_effects(&mut summary, &self.family.regions[arm.0].effects); rewritten.push(arm);
                }
                self.push(occurrence, &guard, RegionKind::Choice { decision, arms: rewritten }, summary)
            },
            RegionKind::Statement(statement) => {
                // Replace this exact leaf's blanket projection obligation with
                // the local projection proof (or its precise guarded gap).
                if let StmtKind::Assign { value, .. } = &statement.kind {
                    if producer_root(value).is_some_and(|variable| producers.contains_key(&variable)) {
                        self.family.obligations.retain(|obligation| !(obligation.occurrence == *occurrence
                            && obligation.class == DecisionClass::Producer
                            && obligation.reason == "bounded producer projection needs retained reaching-view, recomputation and publication lifetime regions"));
                    }
                }
                self.producer_statement(&statement, producers, occurrence, &guard)?
            },
            RegionKind::Repeated { mut header, body, repetition } => {
                if let StmtKind::Owned { tile, .. } = &mut header.kind {
                    self.producer_geometry(tile, producers, occurrence, &guard, &mut prefix)?;
                }
                let body = self.producer_region(body, producers, &guard)?;
                self.repeated(header, body, repetition.order, repetition.indices, repetition.counts, occurrence, &guard)?
            },
            RegionKind::Replicated { index, count, body } => {
                let body = self.producer_region(body, producers, &guard)?;
                let summary = self.family.regions[body.0].effects.clone();
                self.push(occurrence, &guard, RegionKind::Replicated { index, count, body }, summary)
            },
            RegionKind::Conditional { mut header, then, els } => {
                if let StmtKind::If { cond, .. } = &mut header.kind {
                    self.producer_expression(cond, producers, occurrence, &guard, &mut prefix)?;
                }
                let then = self.producer_region(then, producers, &guard)?;
                let els = self.producer_region(els, producers, &guard)?;
                let mut summary = effects(&header, self.family.template.vars.len());
                merge_effects(&mut summary, &self.family.regions[then.0].effects);
                merge_effects(&mut summary, &self.family.regions[els.0].effects);
                self.push(occurrence, &guard, RegionKind::Conditional { header, then, els }, summary)
            },
            RegionKind::Stream { mut header, body, capacity, geometry } => {
                let StmtKind::LoadLoop { domain, views, .. } = &mut header.kind else { unreachable!() };
                self.producer_geometry(&mut domain.view, producers, occurrence, &guard, &mut prefix)?;
                for view in views { self.producer_expression(view, producers, occurrence, &guard, &mut prefix)?; }
                let body = self.producer_region(body, producers, &guard)?;
                let mut summary = effects(&header, self.family.template.vars.len());
                merge_effects(&mut summary, &self.family.regions[body.0].effects);
                self.push(occurrence, &guard, RegionKind::Stream { header, body, capacity, geometry }, summary)
            },
            RegionKind::Reduction(mut reduction) => {
                // A reduction's decomposition owns views of the same source
                // operands. Bind once at its boundary and remap those views,
                // preserving the original trees, piece domains and callbacks.
                let mut rename = HashMap::new();
                for (&variable, _) in producers {
                    if !reduction.operation.operands().any(|value| crate::effects::expressions(value,
                        &|value| matches!(value.kind, ExprKind::Var(candidate) if candidate == variable))) { continue; }
                    let var = self.family.template.vars[variable].clone();
                    let mut value = Expr { kind: ExprKind::Var(variable), ty: var.ty, sym: None, span: var.span };
                    self.producer_expression(&mut value, producers, occurrence, &guard, &mut prefix)?;
                    if let ExprKind::Var(target) = value.kind { rename.insert(variable, target); }
                }
                remap_operation(&mut reduction.operation, &rename);
                if let Some(decomposition) = &mut reduction.decomposition {
                    for piece in [&mut decomposition.full, &mut decomposition.tail] {
                        for statement in &mut piece.setup { crate::composition::remap(statement, &rename, &[]); }
                        remap_operation(&mut piece.operation, &rename);
                    }
                }
                if let Some(decomposition) = &mut reduction.dynamic_decomposition {
                    crate::composition::remap(&mut decomposition.header, &rename, &[]);
                    remap_operation(&mut decomposition.operation, &rename);
                }
                for callback in &mut reduction.callbacks { callback.body = self.producer_region(callback.body, producers, &guard)?; }
                let statement = Stmt { id: None, span: reduction.operation.span, kind: StmtKind::Reduction(Box::new(reduction.operation.clone())) };
                let mut summary = effects(&statement, self.family.template.vars.len());
                for callback in &reduction.callbacks { merge_effects(&mut summary, &self.family.regions[callback.body.0].effects); }
                self.push(occurrence, &guard, RegionKind::Reduction(reduction), summary)
            },
        };
        let replacement = if prefix.is_empty() { replacement } else {
            prefix.push(replacement); self.sequence(prefix, occurrence, &guard)
        };
        Ok(replacement)
    }
}

fn remap_operation(operation: &mut crate::reduction::structured::Reduction, rename: &HashMap<VarId, VarId>) {
    let mut statement = Stmt { id: None, span: operation.span, kind: StmtKind::Reduction(Box::new(operation.clone())) };
    crate::composition::remap(&mut statement, rename, &[]);
    let StmtKind::Reduction(remapped) = statement.kind else { unreachable!() }; *operation = *remapped;
}
