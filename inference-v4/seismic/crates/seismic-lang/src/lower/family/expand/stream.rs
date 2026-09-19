use super::*;

impl Expander {
    pub(super) fn stream(&mut self, header: &Stmt, body: RegionId, geometry: &StreamGeometry, site: &Site) -> Result<RegionId, String> {
        let StmtKind::LoadLoop { domain, views, vars, axes, offset, modes, .. } = &header.kind else { return Err("stream family has an invalid typed header".into()); };
        if vars.len() != views.len() || vars.len() != axes.len() { return Err("stream transfer bindings disagree".into()); }
        // Capturing the complete view expressions once preserves endpoint
        // evaluation independently of how many complete pieces are visited.
        let mut prefix = Vec::new(); let mut captured = Vec::new();
        for view in views {
            let target = self.ordinary(site).local(view.ty.clone());
            prefix.push(stmt(StmtKind::Assign { target: target.clone(), op: AssignOp::Assign, value: view.clone() }, site.span)); captured.push(target);
        }
        if !views.iter().any(|view| crate::normalize::value_identity(view) == crate::normalize::value_identity(&domain.view)) {
            let target = self.ordinary(site).local(domain.view.ty.clone());
            prefix.insert(0, stmt(StmtKind::Assign { target, op: AssignOp::Assign, value: domain.view.clone() }, site.span));
        }
        let mut result = vec![self.statements(prefix, site)];
        let group = self.ordinary(site).index();
        let start = group.sym.as_ref().unwrap().mul(&geometry.capacity);
        let full = self.stream_piece(body, vars, &captured, axes, *offset, modes.as_deref(), &geometry.piece, start, geometry.capacity.clone(), site)?;
        result.push(self.range(&group, Sym::constant(0), geometry.complete_pieces.clone(), full, site));
        let tail_start = geometry.tail_start();
        if let Ok(tail_site) = self.predicate(site, geometry.tail_extent.sub(&Sym::constant(1))) {
            result.push(self.stream_piece(body, vars, &captured, axes, *offset, modes.as_deref(), &geometry.piece, tail_start, geometry.tail_extent.clone(), &tail_site)?);
        } else {
            let tail = self.stream_piece(body, vars, &captured, axes, *offset, modes.as_deref(), &geometry.piece, tail_start, geometry.tail_extent.clone(), site)?;
            let empty = self.sequence(Vec::new(), site);
            result.push(self.conditional(condition(geometry.tail_extent.clone(), BinaryOp::Gt, Sym::constant(0), site.span), tail, empty, site));
        }
        Ok(self.sequence(result, site))
    }
    fn stream_piece(&mut self, body: RegionId, bindings: &[VarId], views: &[Expr], axes: &[usize], offset: Option<VarId>, modes: Option<&[LoadMode]>, piece: &Atom,
        start: Sym, count: Sym, site: &Site) -> Result<RegionId, String> {
        let Atom::Param(piece_name) = piece else { return Err("stream piece has no named symbolic identity".into()); };
        let shapes = HashMap::from([(piece_name.clone(), count.clone())]);
        let mut rename = HashMap::new(); let existing = self.family.template.vars.len();
        for id in 0..existing {
            let variable = &self.family.template.vars[id];
            let ty = crate::lower::subst_ty(&variable.ty, &shapes);
            if ty != variable.ty || bindings.contains(&id) {
                let mut variable = variable.clone(); variable.ty = ty;
                let next = self.family.template.vars.len(); variable.name = format!("{}_piece_{next}", variable.name);
                self.family.template.vars.push(variable); rename.insert(id, next);
            }
        }
        let mut setup = Vec::new();
        if let Some(offset) = offset {
            setup.push(stmt(StmtKind::Assign { target: variable(offset, &self.family.template.vars), op: AssignOp::Assign, value: symbol(start.clone(), site.span) }, site.span));
        }
        for (position, ((binding, source), &axis)) in bindings.iter().zip(views).zip(axes).enumerate() {
            let target = variable(rename[binding], &self.family.template.vars);
            let shape = source.ty.shaped().ok_or("stream transfer source has no shape")?;
            let mut indices = vec![Index::Slice { start: None, end: None }; shape.shape.len()];
            indices[axis] = Index::Slice { start: Some(symbol(start.clone(), site.span)), end: Some(symbol(start.add(&count), site.span)) };
            let mut shape = shape.clone(); shape.shape[axis] = count.clone();
            let view_ty = match source.ty { Ty::Tensor(_) => Ty::Tensor(shape), Ty::Frag(_) => Ty::Frag(shape), _ => Ty::Tile(shape) };
            let view = Expr { kind: ExprKind::Index { base: Box::new(source.clone()), indices }, ty: view_ty, sym: None, span: site.span };
            let kind = if let Some(mode) = modes.and_then(|modes| modes.get(position)) { ExprKind::Load { view: Box::new(view), mode: *mode } }
                else { ExprKind::Builtin { name: Builtin::Load, args: vec![view] } };
            let value = Expr { kind, ty: target.ty.clone(), sym: None, span: site.span };
            setup.push(stmt(StmtKind::Assign { target, op: AssignOp::Assign, value }, site.span));
        }
        let setup = self.statements(setup, site);
        let body = self.remap_region(body, &rename, &[(piece.clone(), count)], site)?;
        Ok(self.sequence(vec![setup, body], site))
    }
}
