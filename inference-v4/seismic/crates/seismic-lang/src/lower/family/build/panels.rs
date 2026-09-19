use super::*;

impl Builder<'_> {
    pub(super) fn matrix_panel(&mut self, repeated: RegionId, occurrence: &OccurrenceId, guard: &Guard) -> Result<RegionId, String> {
        let Some(statements) = self.closed_region(repeated) else { return Ok(repeated); };
        let [statement] = statements.as_slice() else { return Ok(repeated); };
        let atom = Atom::Param(format!("family#numeric#{}", self.family.decisions.len()));
        let width = Sym::atom(atom);
        let Some((domain, implementation)) = crate::composition::matrix_panel_family(statement, &mut self.family.template.vars, width.clone())? else { return Ok(repeated); };
        self.decision(occurrence, guard, domain, 0)?;
        let active = self.predicate(guard, width.sub(&Sym::constant(2)))?;
        let origin = child_origin(occurrence, "matrix.panel", 0, operation(statement));
        let summary = effects(&implementation, self.family.template.vars.len());
        let panel = self.push(&origin, &active, RegionKind::Statement(implementation), summary);
        let header = Stmt { id: None, span: statement.span, kind: StmtKind::If {
            cond: partition::numeric_condition(width, crate::ast::BinaryOp::Gt, Sym::constant(1), statement.span), then: Vec::new(), els: Vec::new() } };
        let mut summary = self.family.regions[repeated.0].effects.clone();
        merge_effects(&mut summary, &self.family.regions[panel.0].effects);
        Ok(self.push(occurrence, guard, RegionKind::Conditional { header, then: panel, els: repeated }, summary))
    }
}
