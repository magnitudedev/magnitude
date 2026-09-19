use super::*;
mod retained;

type Producers = BTreeMap<VarId, (DecisionId, Vec<VarId>, Expr)>;

impl Builder<'_> {
    pub(super) fn producer_family(&mut self, body: &[Stmt], occurrence: &OccurrenceId, guard: &Guard) -> Result<Option<RegionId>, String> {
        let producers = self.producer_decisions(body, occurrence, guard)?;
        if producers.is_empty() { return Ok(None); }
        self.producer_statements(body, &producers, occurrence, guard).map(Some)
    }

    fn producer_decisions(&mut self, body: &[Stmt], occurrence: &OccurrenceId, guard: &Guard) -> Result<Producers, String> {
        let candidates = lower::pure_producer_candidates(body, &self.family.template.vars).into_iter().collect::<BTreeMap<_, _>>();
        let mut producers = Producers::new();
        for (variable, (indices, value)) in candidates {
            if self.family.decisions.iter().any(|decision| matches!(decision.domain.kind,
                DecisionKind::Producer { variable: existing, .. } if existing == variable)
                && decision.guard.choices.iter().all(|choice| guard.choices.contains(choice))
                && decision.guard.predicates.iter().all(|predicate| guard.predicates.contains(predicate))
                && decision.guard.one_of.iter().all(|clause| guard.one_of.contains(clause))) { continue; }
            let origin = child_origin(occurrence, "producer", variable, format!("{}:{:?}", self.family.template.vars[variable].name, crate::normalize::value_identity(&value)));
            let decision = self.decision(&origin, guard, Decision { kind: DecisionKind::Producer { variable,
                name: self.family.template.vars[variable].name.clone(), ty: self.family.template.vars[variable].ty.clone() },
                alternatives: vec![Alternative::Materialize, Alternative::Recompute].into() }, 0)?;
            producers.insert(variable, (decision, indices, value));
        }
        Ok(producers)
    }

    fn producer_statements(&mut self, body: &[Stmt], producers: &Producers, occurrence: &OccurrenceId, guard: &Guard) -> Result<RegionId, String> {
        let mut children = Vec::new();
        for (index, statement) in body.iter().enumerate() {
            let origin = child_origin(occurrence, "producer.operation", index, operation(statement));
            let definition = producer_definition(statement, producers);
            if let Some((decision, _, _)) = definition {
                let retained = guard.with(decision.clone(), 0);
                let mut dependencies = producers.clone();
                dependencies.retain(|_, (id, ..)| id != decision);
                let stored = self.producer_statement(statement, &dependencies, &origin, &retained)?;
                let removed = self.sequence(Vec::new(), &origin, &guard.with(decision.clone(), 1));
                let summary = self.family.regions[stored.0].effects.clone();
                children.push(self.push(&origin, guard, RegionKind::Choice { decision: decision.clone(), arms: vec![stored, removed] }, summary));
            } else {
                children.push(self.producer_statement(statement, producers, &origin, guard)?);
            }
        }
        Ok(self.sequence(children, occurrence, guard))
    }

    fn producer_statement(&mut self, statement: &Stmt, producers: &Producers, occurrence: &OccurrenceId, guard: &Guard) -> Result<RegionId, String> {
        let mut header = statement.clone();
        let mut prefix = Vec::new();
        let kind = match &mut header.kind {
            StmtKind::Assign { target, value, .. } => {
                self.producer_address(target, producers, occurrence, guard, &mut prefix)?;
                self.producer_expression(value, producers, occurrence, guard, &mut prefix)?;
                None
            },
            StmtKind::Expr(value) => { self.producer_expression(value, producers, occurrence, guard, &mut prefix)?; None },
            StmtKind::If { cond, then, els } => {
                self.producer_expression(cond, producers, occurrence, guard, &mut prefix)?;
                let yes = self.producer_statements(then, producers, &child_origin(occurrence, "then", 0, "producer reads".into()), guard)?;
                let no = self.producer_statements(els, producers, &child_origin(occurrence, "else", 0, "producer reads".into()), guard)?;
                then.clear(); els.clear();
                Some((yes, Some(no), RepeatOrder::Serial, Vec::new(), Vec::new()))
            },
            StmtKind::Parallel { vars, extents, body } => {
                let child = self.producer_statements(body, producers, occurrence, guard)?;
                body.clear(); Some((child, None, RepeatOrder::Parallel, vars.clone(), extents.clone()))
            },
            StmtKind::Range { var, lo, hi, body } => {
                let child = self.producer_statements(body, producers, occurrence, guard)?;
                body.clear(); Some((child, None, RepeatOrder::Serial, vec![*var], vec![hi.sub(lo)]))
            },
            StmtKind::Lanes { var, extent, body, .. } => {
                let child = self.producer_statements(body, producers, occurrence, guard)?;
                body.clear(); Some((child, None, RepeatOrder::Parallel, vec![*var], vec![extent.clone()]))
            },
            StmtKind::Owned { vars, tile, body } => {
                self.producer_geometry(tile, producers, occurrence, guard, &mut prefix)?;
                let child = self.producer_statements(body, producers, occurrence, guard)?;
                body.clear(); Some((child, None, RepeatOrder::Parallel, vars.clone(), tile.ty.shaped().map(|shape| shape.shape.clone()).unwrap_or_default()))
            },
            StmtKind::LoadLoop { .. } | StmtKind::Reduction(_) => {
                None
            },
        };
        let main = match kind {
            Some((then, Some(els), ..)) => {
                let mut summary = effects(&header, self.family.template.vars.len());
                merge_effects(&mut summary, &self.family.regions[then.0].effects);
                merge_effects(&mut summary, &self.family.regions[els.0].effects);
                self.push(occurrence, guard, RegionKind::Conditional { header, then, els }, summary)
            },
            Some((body, None, order, indices, counts)) => self.repeated(header, body, order, indices, counts, occurrence, guard)?,
            None => {
                let summary = effects(&header, self.family.template.vars.len());
                let snapshot = match &header.kind {
                    StmtKind::Assign { target: Expr { kind: ExprKind::Var(variable), .. }, value, .. } => match &value.kind {
                        ExprKind::Builtin { name: Builtin::Load, args } => args.first().map(|source| (*variable, source.clone())),
                        ExprKind::Load { view, .. } => Some((*variable, (**view).clone())),
                        _ => None,
                    },
                    _ => None,
                };
                let id = self.push(occurrence, guard, RegionKind::Statement(header), summary);
                if let Some((variable, source)) = snapshot {
                    self.family.snapshots.push(Snapshot { producer: id, variable, source, guard: guard.clone(), consumers: Vec::new() });
                }
                id
            },
        };
        prefix.push(main);
        Ok(self.sequence(prefix, occurrence, guard))
    }

    fn producer_address(&mut self, target: &mut Expr, producers: &Producers, occurrence: &OccurrenceId, guard: &Guard, prefix: &mut Vec<RegionId>) -> Result<(), String> {
        // The storage named by an lvalue is never a read of its old value.
        // Only address operands may consume another producer.
        match &mut target.kind {
            ExprKind::Index { base, indices } => {
                self.producer_address(base, producers, occurrence, guard, prefix)?;
                for index in indices { match index {
                    Index::Point(value) => self.producer_expression(value, producers, occurrence, guard, prefix)?,
                    Index::Slice { start, end } => for value in start.iter_mut().chain(end) {
                        self.producer_expression(value, producers, occurrence, guard, prefix)?;
                    },
                } }
            },
            ExprKind::Transpose(base) | ExprKind::Accessor { base, .. } | ExprKind::Lanes { base, .. } =>
                self.producer_address(base, producers, occurrence, guard, prefix)?,
            _ => {},
        }
        Ok(())
    }

    fn producer_expression(&mut self, expression: &mut Expr, producers: &Producers, occurrence: &OccurrenceId, guard: &Guard, prefix: &mut Vec<RegionId>) -> Result<(), String> {
        // Each original read becomes one value binding with local arms. Other
        // producer decisions appear only in those operands that actually use
        // them; there is no product of complete downstream statements.
        if let ExprKind::Builtin { name: Builtin::Extent, args } = &mut expression.kind {
            if let Some((view, axes)) = args.split_first_mut() {
                self.producer_geometry(view, producers, occurrence, guard, prefix)?;
                for axis in axes { self.producer_expression(axis, producers, occurrence, guard, prefix)?; }
            }
            return Ok(());
        }
        if matches!(expression.ty, Ty::Tile(_)) {
            if let Some(variable) = producer_root(expression).filter(|variable| producers.contains_key(variable)) {
                let (decision, coordinates, value) = &producers[&variable];
                let original = expression.clone();
                let target_id = self.family.template.vars.len();
                self.family.template.vars.push(Var { name: format!("producer_view_{target_id}"),
                    ty: original.ty.clone(), span: original.span, kind: VarKind::Local });
                let target = Expr { kind: ExprKind::Var(target_id), ty: original.ty.clone(), sym: None, span: original.span };
                let origin = child_origin(occurrence, "producer.view", target_id, variable.to_string());
                let mut remaining = producers.clone(); remaining.remove(&variable);
                let materialize = Stmt { id: None, span: original.span, kind: StmtKind::Assign {
                    target: target.clone(), op: AssignOp::Assign, value: original.clone(),
                } };
                let stored = self.producer_statements(std::slice::from_ref(&materialize), &remaining, &origin, &guard.with(decision.clone(), 0))?;
                let active = guard.with(decision.clone(), 1);
                let checkpoint = self.family.template.vars.len();
                let projected = crate::composition::family::project_producer(&original, &target, variable, coordinates,
                    value, &mut self.family.template.vars)?;
                let projected = match projected {
                    Some(projected) => projected,
                    None => {
                        self.family.template.vars.truncate(checkpoint);
                        self.obligation(&origin, &active, DecisionClass::Producer,
                            format!("producer {variable} has a view whose coordinate geometry cannot be projected"));
                        vec![materialize]
                    },
                };
                let recomputed = self.producer_statements(&projected, &remaining, &origin, &active)?;
                let mut summary = self.family.regions[stored.0].effects.clone();
                merge_effects(&mut summary, &self.family.regions[recomputed.0].effects);
                prefix.push(self.push(&origin, guard, RegionKind::Choice { decision: decision.clone(), arms: vec![stored, recomputed] }, summary));
                *expression = target;
                return Ok(());
            }
        }
        if let ExprKind::Index { base, indices } = &expression.kind {
            if let ExprKind::Var(variable) = base.kind {
                if let Some((decision, coordinates, value)) = producers.get(&variable) {
                    if indices.iter().all(|index| matches!(index, Index::Point(_))) {
                        let original = expression.clone();
                        let mut recomputed = original.clone();
                        lower::rewrite_reads(&mut recomputed, &HashMap::from([(variable, (coordinates.clone(), value.clone()))]), &self.family.template.vars);
                        let target_id = self.family.template.vars.len();
                        self.family.template.vars.push(Var { name: format!("producer_read_{target_id}"), ty: expression.ty.clone(), span: expression.span, kind: VarKind::Local });
                        let target = Expr { kind: ExprKind::Var(target_id), ty: expression.ty.clone(), sym: None, span: expression.span };
                        let origin = child_origin(occurrence, "producer.read", target_id, format!("{}", variable));
                        let mut arms = Vec::new(); let mut summary = Effects::default();
                        for (ordinal, mut read) in [original, recomputed].into_iter().enumerate() {
                            let active = guard.with(decision.clone(), ordinal);
                            let mut prerequisites = Vec::new();
                            // Never rewrite the retained read into itself. Its
                            // coordinate operands can still depend on producers.
                            let mut remaining = producers.clone(); remaining.remove(&variable);
                            self.producer_expression(&mut read, &remaining, &origin, &active, &mut prerequisites)?;
                            let statement = Stmt { id: None, span: read.span, kind: StmtKind::Assign { target: target.clone(), op: AssignOp::Assign, value: read } };
                            let effects = effects(&statement, self.family.template.vars.len());
                            prerequisites.push(self.push(&origin, &active, RegionKind::Statement(statement), effects));
                            let arm = self.sequence(prerequisites, &origin, &active);
                            merge_effects(&mut summary, &self.family.regions[arm.0].effects);
                            arms.push(arm);
                        }
                        prefix.push(self.push(&origin, guard, RegionKind::Choice { decision: decision.clone(), arms }, summary));
                        *expression = target;
                        return Ok(());
                    }
                }
            }
        }
        match &mut expression.kind {
            ExprKind::Index { base, indices } => {
                self.producer_expression(base, producers, occurrence, guard, prefix)?;
                for index in indices { match index {
                    Index::Point(value) => self.producer_expression(value, producers, occurrence, guard, prefix)?,
                    Index::Slice { start, end } => for value in start.iter_mut().chain(end) { self.producer_expression(value, producers, occurrence, guard, prefix)?; },
                } }
            },
            ExprKind::Load { view: value, .. } | ExprKind::Transpose(value) | ExprKind::Unary { expr: value, .. }
                | ExprKind::Cast { expr: value, .. } | ExprKind::Accessor { base: value, .. } | ExprKind::Lanes { base: value, .. } =>
                self.producer_expression(value, producers, occurrence, guard, prefix)?,
            ExprKind::Binary { op, lhs, rhs } if matches!(op, crate::ast::BinaryOp::And | crate::ast::BinaryOp::Or) => {
                self.producer_expression(lhs, producers, occurrence, guard, prefix)?;
                let mut right_prefix = Vec::new();
                self.producer_expression(rhs, producers, occurrence, guard, &mut right_prefix)?;
                if !right_prefix.is_empty() {
                    // A producer read in a short-circuit operand remains inside
                    // that operand's runtime branch, including its bounds checks.
                    let id = self.family.template.vars.len();
                    let ty = expression.ty.clone(); let span = expression.span;
                    self.family.template.vars.push(Var { name: format!("producer_condition_{id}"), ty: ty.clone(), span, kind: VarKind::Local });
                    let target = Expr { kind: ExprKind::Var(id), ty, sym: None, span };
                    let origin = child_origin(occurrence, "producer.lazy", id, op.text().into());
                    let assign = Stmt { id: None, span, kind: StmtKind::Assign { target: target.clone(), op: AssignOp::Assign, value: (**lhs).clone() } };
                    let summary = effects(&assign, self.family.template.vars.len());
                    prefix.push(self.push(&origin, guard, RegionKind::Statement(assign), summary));
                    let assign = Stmt { id: None, span, kind: StmtKind::Assign { target: target.clone(), op: AssignOp::Assign, value: (**rhs).clone() } };
                    let summary = effects(&assign, self.family.template.vars.len());
                    right_prefix.push(self.push(&origin, guard, RegionKind::Statement(assign), summary));
                    let then = self.sequence(right_prefix, &origin, guard);
                    let els = self.sequence(Vec::new(), &origin, guard);
                    let cond = if *op == crate::ast::BinaryOp::And { target.clone() } else {
                        Expr { kind: ExprKind::Unary { op: crate::ast::UnaryOp::Not, expr: Box::new(target.clone()) }, ..target.clone() }
                    };
                    let header = Stmt { id: None, span, kind: StmtKind::If { cond, then: Vec::new(), els: Vec::new() } };
                    let mut summary = effects(&header, self.family.template.vars.len());
                    merge_effects(&mut summary, &self.family.regions[then.0].effects);
                    prefix.push(self.push(&origin, guard, RegionKind::Conditional { header, then, els }, summary));
                    *expression = target;
                }
            },
            ExprKind::Binary { lhs, rhs, .. } => {
                self.producer_expression(lhs, producers, occurrence, guard, prefix)?;
                self.producer_expression(rhs, producers, occurrence, guard, prefix)?;
            },
            ExprKind::Builtin { args, .. } | ExprKind::Call { args, .. } | ExprKind::Intrinsic { args, .. } | ExprKind::Tuple(args) => {
                for argument in args { self.producer_expression(argument, producers, occurrence, guard, prefix)?; }
            },
            ExprKind::Int(_) | ExprKind::Float(_) | ExprKind::Bool(_) | ExprKind::Var(_)
                | ExprKind::ShapeParam(_) | ExprKind::TileAlloc { .. } => {},
        }
        Ok(())
    }

    fn producer_geometry(&mut self, expression: &mut Expr, producers: &Producers, occurrence: &OccurrenceId,
        guard: &Guard, prefix: &mut Vec<RegionId>) -> Result<(), String> {
        if let Some(variable) = producer_root(expression).filter(|variable| producers.contains_key(variable)) {
            let source = self.family.template.vars[variable].clone();
            let shape = source.ty.shaped().ok_or("producer geometry has no shape")?.clone();
            let id = self.family.template.vars.len();
            self.family.template.vars.push(Var { name: format!("producer_geometry_{id}"), ..source.clone() });
            let target = Expr { kind: ExprKind::Var(id), ty: source.ty.clone(), sym: None, span: source.span };
            let statement = Stmt { id: None, span: source.span, kind: StmtKind::Assign {
                target: target.clone(), op: AssignOp::Assign,
                value: Expr { kind: ExprKind::TileAlloc { shape: shape.shape, dtype: shape.elem },
                    ty: source.ty, sym: None, span: source.span },
            } };
            let summary = effects(&statement, self.family.template.vars.len());
            prefix.push(self.push(occurrence, guard, RegionKind::Statement(statement), summary));
            *expression = lower::subst_vars(expression, &HashMap::from([(variable, target)]), &HashMap::new());
        }
        self.producer_expression(expression, producers, occurrence, guard, prefix)
    }
}

fn producer_root(value: &Expr) -> Option<VarId> {
    match &value.kind {
        ExprKind::Var(variable) => Some(*variable),
        ExprKind::Index { base, .. } | ExprKind::Transpose(base) => producer_root(base),
        ExprKind::Builtin { name: Builtin::Reshape, args } => args.first().and_then(producer_root),
        _ => None,
    }
}

fn producer_definition<'a>(statement: &Stmt, producers: &'a Producers) -> Option<&'a (DecisionId, Vec<VarId>, Expr)> {
    match &statement.kind {
        StmtKind::Assign { target: Expr { kind: ExprKind::Var(variable), .. },
            value: Expr { kind: ExprKind::TileAlloc { .. }, .. }, .. } => producers.get(variable),
        StmtKind::Owned { vars, tile: Expr { kind: ExprKind::Var(variable), .. }, .. } =>
            producers.get(variable).filter(|(_, coordinates, _)| coordinates == vars),
        _ => None,
    }
}
