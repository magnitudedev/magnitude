use super::*;

impl Builder<'_> {
    pub(super) fn compile_parameter_expression(&self, expression: &Sym) -> bool {
        expression.params().iter().all(|name| self.family.decisions.iter().any(|decision|
            matches!(&decision.numeric, Some(NumericParameter { atom: Atom::Param(parameter), .. }) if parameter == name)))
    }

    pub(super) fn predicate(&self, guard: &Guard, expression: Sym) -> Result<Guard, String> {
        let mut parameters = Vec::new();
        for name in expression.params() {
            let definition = self.family.decisions.iter().find(|decision|
                matches!(&decision.numeric, Some(NumericParameter { atom: Atom::Param(parameter), .. }) if *parameter == name))
                .ok_or_else(|| format!("numeric presence references a runtime value `{name}`"))?;
            parameters.push((name, definition.id.clone(), definition.domain.alternatives.clone()));
        }
        let mut guard = guard.clone();
        guard.predicates.push(Predicate { nonnegative: expression, parameters });
        Ok(guard)
    }

    pub(super) fn numeric_bound(&self, expression: &Sym) -> Option<i64> {
        expression.eval_interval(&|name| {
            self.family.decisions.iter().find_map(|decision| {
                let parameter = decision.numeric.as_ref()?;
                if parameter.atom != Atom::Param(name.to_owned()) { return None; }
                let numeric = decision.domain.alternatives.numeric()?;
                let bounds = numeric.runs().try_fold(None, |bounds: Option<(i64, i64)>, run| {
                    let last = run.get(run.ordinal.checked_add(run.count)?.checked_sub(1)?)?;
                    let range = (run.first.min(last), run.first.max(last));
                    Some(Some(match bounds { None => range, Some((lo, hi)) => (lo.min(range.0), hi.max(range.1)) }))
                }).flatten();
                bounds
            })
        }).map(|(_, upper)| upper)
    }

    pub(super) fn partition_capacities(&self, extent: &Sym, maximum: i64) -> Result<Alternatives, String> {
        match self.options.piece {
            Some(piece) if extent.as_constant().is_none() && self.compile_parameter_expression(extent) =>
                Alternatives::stream_capacities(piece.min(maximum)),
            Some(piece) => Ok(vec![Alternative::StreamCapacity(piece.min(maximum))].into()),
            None => Alternatives::stream_capacities(maximum),
        }
    }

    pub(super) fn constrain_partition_capacity(&mut self, extent: &Sym, capacity: &Sym, guard: &Guard) -> Result<(), String> {
        if !self.compile_parameter_expression(extent) { return Ok(()); }
        self.family.requirements.push(Requirement { guard: guard.clone(), nonnegative: extent.sub(capacity) });
        if let Some(piece) = self.options.piece.filter(|_| extent.as_constant().is_none()) {
            // An explicit piece is a maximum, including when an enclosing
            // partition makes this extent smaller. Retain min(piece, extent)
            // as a relation over the original choices instead of rejecting the
            // smaller upstream assignment.
            let piece = Sym::constant(piece);
            let whole = self.predicate(guard, piece.sub(extent).sub(&Sym::constant(1)))?;
            self.family.requirements.push(Requirement { guard: whole, nonnegative: capacity.sub(extent) });
            let bounded = self.predicate(guard, extent.sub(&piece))?;
            self.family.requirements.push(Requirement { guard: bounded, nonnegative: capacity.sub(&piece) });
        }
        Ok(())
    }

    pub(super) fn partition_call(&mut self, call: &Expr, definition: &Function, occurrence: &OccurrenceId, guard: &Guard) -> Result<Option<RegionId>, String> {
        let ExprKind::Call { callee, shape_args, elem_args, args } = &call.kind else { unreachable!() };
        // Expand transparent source definitions for semantic dependence facts.
        // Portable expansion retains stream/reduction contracts and rejects any
        // implementation-choice callback, so this cannot hide a selected path.
        let (variables, semantic) = lower::portable_body(self.program, definition)?;
        let domains = lower::decomposition::domains(definition, &variables, &semantic);
        for domain in domains {
            let key = (callee.clone(), domain.parameter.clone());
            if self.partitioning.contains(&key) { continue; }
            let axis = definition.shape_params.iter().position(|name| name == &domain.parameter).ok_or("missing contraction shape parameter")?;
            let extent = shape_args[axis].clone();
            let maximum = extent.as_constant().or_else(|| self.numeric_bound(&extent))
                .or_else(|| self.bounds.get(&extent).map(|(bound, _)| *bound));
            let Some(maximum) = maximum.filter(|bound| *bound > 0) else { continue; };
            let roots = |partitioned: bool| args.iter().zip(&domain.axes).filter(|(_, axis)| axis.is_some() == partitioned)
                .filter_map(|(argument, _)| {
                    let mut root = argument;
                    while let ExprKind::Index { base, .. } | ExprKind::Transpose(base) = &root.kind { root = base; }
                    if let ExprKind::Var(id) = root.kind { Some(id) } else { None }
                }).collect::<HashSet<_>>();
            if !roots(true).is_disjoint(&roots(false)) { continue; }
            let piece = Atom::Param(format!("family#callpiece#{}", self.family.decisions.len()));
            let id = self.decision(occurrence, guard, Decision { kind: DecisionKind::Stream { piece: piece.clone(), extent: extent.clone(), maximum },
                alternatives: self.partition_capacities(&extent, maximum)? }, axis)?;
            let parameter = self.family.decisions.last().unwrap().numeric.clone().unwrap();
            let capacity = Sym::atom(parameter.atom.clone());
            self.constrain_partition_capacity(&extent, &capacity, guard)?;
            let geometry = StreamGeometry { parameter, extent: extent.clone(), capacity: capacity.clone(),
                complete_pieces: extent.quot(&capacity), tail_extent: extent.rem(&capacity),
                visits: extent.add(&capacity).sub(&Sym::constant(1)).quot(&capacity), piece: piece.clone() };
            self.partitioning.insert(key.clone());
            let result = if self.compile_parameter_expression(&extent) {
                let index = self.family.template.vars.len();
                let atom = Atom::Param(format!("family#calloffset#{index}"));
                self.family.template.vars.push(Var { name: format!("call_offset_{index}"), ty: Ty::Scalar(crate::types::DType::I32),
                    span: call.span, kind: VarKind::Index(atom.clone()) });
                let full_origin = child_origin(occurrence, "contraction.full", axis, domain.parameter.clone());
                let full = self.call_piece(call, &domain.axes, axis, Sym::atom(atom).mul(&capacity), capacity.clone(), &full_origin, guard)?;
                let header = Stmt { id: None, span: call.span, kind: StmtKind::Range { var: index, lo: Sym::constant(0), hi: geometry.complete_pieces.clone(), body: Vec::new() } };
                let full = self.repeated(header, full, RepeatOrder::Serial, vec![index], vec![geometry.complete_pieces.clone()], &full_origin, guard)?;
                let tail_origin = child_origin(occurrence, "contraction.tail", axis, domain.parameter.clone());
                let tail_guard = self.predicate(guard, geometry.tail_extent.sub(&Sym::constant(1)))?;
                let tail = self.call_piece(call, &domain.axes, axis, geometry.tail_start(), geometry.tail_extent.clone(), &tail_origin, &tail_guard)?;
                let empty = self.sequence(Vec::new(), &tail_origin, guard);
                let condition = numeric_condition(geometry.tail_extent.clone(), crate::ast::BinaryOp::Gt, Sym::constant(0), call.span);
                let header = Stmt { id: None, span: call.span, kind: StmtKind::If { cond: condition, then: Vec::new(), els: Vec::new() } };
                let summary = self.family.regions[tail.0].effects.clone();
                let tail = self.push(&tail_origin, guard, RegionKind::Conditional { header, then: tail, els: empty }, summary);
                self.sequence(vec![full, tail], occurrence, guard)
            } else {
                let domain_source = self.bounds.get(&extent).map(|(_, domain)| domain.clone())
                    .ok_or("dynamic contraction has no captured extent source")?;
                let mut piece_args = args.clone(); let mut bindings = Vec::new(); let mut views = Vec::new(); let mut axes = Vec::new();
                for (argument, axis) in piece_args.iter_mut().zip(&domain.axes) {
                    let Some(axis) = axis else { continue; };
                    let mut shape = argument.ty.shaped().ok_or("contraction argument is not shaped")?.clone();
                    shape.shape[*axis] = Sym::atom(piece.clone());
                    let ty = Ty::Tile(shape); let variable = self.family.template.vars.len();
                    self.family.template.vars.push(Var { name: format!("call_piece_{variable}"), ty: ty.clone(), span: call.span, kind: VarKind::Local });
                    views.push(argument.clone()); axes.push(*axis); bindings.push(variable);
                    *argument = Expr { kind: ExprKind::Var(variable), ty, sym: None, span: call.span };
                }
                let mut shapes = shape_args.clone(); shapes[axis] = Sym::atom(piece.clone());
                let piece_call = Expr { kind: ExprKind::Call { callee: callee.clone(), shape_args: shapes, elem_args: elem_args.clone(), args: piece_args }, ..call.clone() };
                let body_origin = child_origin(occurrence, "contraction.dynamic", axis, domain.parameter.clone());
                let body = self.call(&piece_call, &body_origin, guard)?;
                let header = Stmt { id: None, span: call.span, kind: StmtKind::LoadLoop { domain: domain_source, offset: None,
                    modes: None, vars: bindings, views, axes, piece, capacity: None, body: Vec::new() } };
                let summary = self.family.regions[body.0].effects.clone();
                self.push(occurrence, guard, RegionKind::Stream { header, body, capacity: id, geometry }, summary)
            };
            self.partitioning.remove(&key);
            return Ok(Some(result));
        }
        Ok(None)
    }

    fn call_piece(&mut self, call: &Expr, axes: &[Option<usize>], dimension: usize, start: Sym, count: Sym, occurrence: &OccurrenceId, guard: &Guard) -> Result<RegionId, String> {
        let ExprKind::Call { callee, shape_args, elem_args, args } = &call.kind else { unreachable!() };
        let mut children = Vec::new(); let mut arguments = Vec::new();
        for (position, (argument, axis)) in args.iter().zip(axes).enumerate() {
            if let Some(axis) = axis {
                let (statement, value) = lower::decomposition::slice_input(argument, *axis, start.clone(), count.clone(), &mut self.family.template.vars)?;
                let origin = child_origin(occurrence, "slice", position, operation(&statement));
                children.push(self.leaf(statement, &origin, guard)?); arguments.push(value);
            } else { arguments.push(argument.clone()); }
        }
        let mut shapes = shape_args.clone(); shapes[dimension] = count;
        let piece = Expr { kind: ExprKind::Call { callee: callee.clone(), shape_args: shapes, elem_args: elem_args.clone(), args: arguments }, ..call.clone() };
        children.push(self.call(&piece, occurrence, guard)?);
        Ok(self.sequence(children, occurrence, guard))
    }
}

pub(super) fn numeric_condition(left: Sym, operator: crate::ast::BinaryOp, right: Sym, span: crate::span::Span) -> Expr {
    let expression = |value: Sym| Expr { kind: ExprKind::ShapeParam(value.to_string()), ty: Ty::Scalar(crate::types::DType::I32), sym: Some(value), span };
    Expr { kind: ExprKind::Binary { op: operator, lhs: Box::new(expression(left)), rhs: Box::new(expression(right)) }, ty: Ty::Scalar(crate::types::DType::Bool), sym: None, span }
}
