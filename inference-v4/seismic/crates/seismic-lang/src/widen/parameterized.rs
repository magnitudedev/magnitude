//! Numeric bounded widening with static copy identity. A retained copy region
//! is compile-time replication; it is never replaced by an ordinary Range.
use super::*;
use std::collections::{BTreeSet, HashMap};

/// A variable family is addressed by its original binding and copy ordinal.
/// All dependent statements in a scope share this address space, preserving
/// producer/consumer identity across separated replicated statements.
#[derive(Clone, Debug)]
pub struct Copies {
    pub variable: VarId,
    pub count: Sym,
}
#[derive(Clone, Copy, Debug)]
pub enum Guard {
    WidthAtLeast(u64),
    HasTail,
}
pub struct Union {
    pub body: Vec<Stmt>,
    pub aliases: Vec<(VarId, VarId)>,
}
/// Instantiate one original static-copy occurrence without re-running widening
/// or its dependency partition. Carried bindings deliberately retain identity
/// across consecutive copies; this is also the source family's copy semantics.
pub fn replicate(body: &[Stmt], index: VarId, atom: &Atom, ordinal: i64, span: crate::span::Span) -> Vec<Stmt> {
    let value = Expr { kind: ExprKind::Int(ordinal), ty: Ty::Scalar(crate::types::DType::I32),
        sym: Some(Sym::constant(ordinal)), span };
    let mut body = body.to_vec();
    super::replace_index(&mut body, index, atom, &value);
    body
}
pub fn copy_bindings(body: &[Stmt], vars: &mut Vec<Var>) -> Union {
    let (body, copies) = super::copy_bindings(body, vars);
    Union { body, aliases: copies.into_iter().collect() }
}
/// Attach the original phase handoff to a copied local implementation using
/// the same binding aliases as its retained source body.
pub fn remap_bindings(body: &[Stmt], aliases: &[(VarId, VarId)]) -> Vec<Stmt> {
    let aliases = aliases.iter().copied().collect::<HashMap<_, _>>();
    let mut body = body.to_vec();
    for statement in &mut body { crate::composition::remap(statement, &aliases, &[]); }
    body
}
#[derive(Clone, Debug)]
pub enum Region {
    Shared(Stmt),
    Replicated { statement: Stmt, count: Sym, bindings: Vec<VarId> },
    SharedScope { header: Stmt, body: Vec<Region> },
}
#[derive(Clone, Debug)]
pub struct Geometry {
    pub width: Sym,
    pub extent: Sym,
    pub complete: Sym,
    pub tail: Sym,
    pub visits: Sym,
}
#[derive(Clone, Debug)]
pub struct Family {
    pub geometry: Geometry,
    pub full: Vec<Region>,
    /// A partial item executes whole original occurrences under per-copy
    /// validity guards; its shared-producer behavior is not the full-item form.
    pub tail: Vec<Stmt>,
    pub copies: Vec<Copies>,
    original: Vec<Stmt>,
    inner: VarId,
    atom: Atom,
    base: Expr,
}
impl Family {
    pub fn new(body: &[Stmt], inner: VarId, atom: &Atom, width: Sym, extent: Sym,
        vars: &[Var], base: &Expr) -> Result<Self, String> {
        if inner >= vars.len() || base.sym.is_none() { return Err("retained widening needs a bound symbolic index".into()); }
        if width.as_constant().is_some_and(|value| value <= 0)
            || extent.as_constant().is_some_and(|value| value < 0) {
            return Err("retained widening requires positive width and nonnegative extent".into());
        }
        let mut copied = BTreeSet::new();
        let full = retain(body, inner, atom, &width, vars, &mut copied);
        let complete = extent.quot(&width);
        let tail = extent.rem(&width);
        let visits = extent.add(&width).sub(&Sym::constant(1)).quot(&width);
        Ok(Self {
            geometry: Geometry { width: width.clone(), extent, complete, tail, visits },
            full, tail: body.to_vec(), copies: copied.into_iter().map(|variable| Copies { variable, count: width.clone() }).collect(),
            original: body.to_vec(), inner, atom: atom.clone(), base: base.clone(),
        })
    }
    /// Deterministic instantiation of retained definitions. This method does
    /// not run dependency analysis or discover implementation alternatives.
    pub fn instantiate(&self, parameters: &dyn Fn(&str) -> Option<i64>, vars: &mut Vec<Var>) -> Result<Vec<Stmt>, String> {
        let width = self.geometry.width.eval(parameters).ok_or("missing retained widening width")?;
        let extent = self.geometry.extent.eval(parameters).ok_or("missing retained widening extent")?;
        if width <= 0 || extent < 0 || extent > i64::from(i32::MAX) || width > i64::from(i32::MAX) {
            return Err("retained widening assignment exceeds its index domain".into());
        }
        // The original compiler skips widening for width one. Preserve that
        // exact variable/scope topology rather than constructing fresh copies.
        if width == 1 { return Ok(self.original.clone()); }
        let count = usize::try_from(width).map_err(|_| "widening width exceeds host indexing")?;
        let mut copies = vec![HashMap::new(); count];
        let full = self.instantiate_regions(&self.full, width, vars, &mut copies)?;
        if extent % width == 0 { return Ok(full); }
        let mut tail = Vec::new();
        for offset in 0..width {
            let value = super::offset(&self.base, offset);
            let mut copies = HashMap::new();
            let occurrence = self.tail.iter().map(|statement|
                super::substitute(statement, self.inner, &self.atom, &value, vars, &mut copies)).collect();
            tail.push(self.guarded(extent - offset, occurrence, Vec::new()));
        }
        Ok(vec![self.guarded(extent - width + 1, full, tail)])
    }
    /// One bounded occurrence union for a target that retains conditional
    /// regions. `guard` attaches original parameter predicates; it never chooses
    /// a width. Independent widened bindings retain one copy per ordinal.
    pub fn union(&self, maximum_width: usize, vars: &mut Vec<Var>,
        guard: &mut dyn FnMut(Guard, Vec<Stmt>, Vec<Stmt>, &mut Vec<Var>) -> Result<Vec<Stmt>, String>) -> Result<Union, String> {
        if maximum_width == 0 { return Err("empty widening occurrence domain".into()); }
        let mut copies = vec![HashMap::new(); maximum_width];
        let full = self.union_regions(&self.full, vars, &mut copies, guard)?;
        let mut aliases = copies.iter().flat_map(|copies| copies.iter().map(|(&source,&copy)| (source,copy))).collect::<Vec<_>>();
        let mut tail = Vec::new();
        let extent = self.geometry.extent.clone();
        for ordinal in 0..maximum_width {
            let offset = i64::try_from(ordinal).map_err(|_| "widening occurrence exceeds integer")?;
            let value = super::offset(&self.base, offset);
            let mut copies = HashMap::new();
            let occurrence = self.tail.iter().map(|statement|
                super::substitute(statement, self.inner, &self.atom, &value, vars, &mut copies)).collect();
            aliases.extend(copies.into_iter());
            let runtime = self.guarded_symbol(extent.sub(&Sym::constant(offset)), occurrence, Vec::new());
            tail.extend(guard(Guard::WidthAtLeast(ordinal as u64 + 1), vec![runtime], Vec::new(), vars)?);
        }
        use crate::{ast::BinaryOp, types::DType};
        let end = self.geometry.extent.sub(&self.geometry.width).add(&Sym::constant(1));
        let limit = Expr { kind: ExprKind::ShapeParam("widening_full_limit".into()), ty: Ty::Scalar(DType::I32), sym: Some(end), span: self.base.span };
        let condition = Expr { kind: ExprKind::Binary { op: BinaryOp::Lt, lhs: Box::new(self.base.clone()), rhs: Box::new(limit) }, ty: Ty::Scalar(DType::Bool), sym: None, span: self.base.span };
        let bounded = Stmt { id: None, span: self.base.span, kind: StmtKind::If { cond: condition, then: full.clone(), els: tail } };
        let widened = guard(Guard::HasTail, vec![bounded], full, vars)?;
        let body = guard(Guard::WidthAtLeast(2), widened, self.original.clone(), vars)?;
        Ok(Union { body, aliases })
    }
    fn union_regions(&self, regions: &[Region], vars: &mut Vec<Var>, copies: &mut [HashMap<VarId,VarId>],
        guard: &mut dyn FnMut(Guard, Vec<Stmt>, Vec<Stmt>, &mut Vec<Var>) -> Result<Vec<Stmt>, String>) -> Result<Vec<Stmt>, String> {
        let mut output = Vec::new();
        for region in regions {
            match region {
                Region::Shared(statement) => output.push(statement.clone()),
                Region::Replicated { statement, .. } => for (ordinal, copies) in copies.iter_mut().enumerate() {
                    let value = super::offset(&self.base, ordinal as i64);
                    let copy = super::substitute(statement, self.inner, &self.atom, &value, vars, copies);
                    output.extend(guard(Guard::WidthAtLeast(ordinal as u64 + 1), vec![copy], Vec::new(), vars)?);
                },
                Region::SharedScope { header, body } => {
                    let mut statement = header.clone();
                    let selected = self.union_regions(body, vars, copies, guard)?;
                    match &mut statement.kind {
                        StmtKind::Range { body, .. } | StmtKind::LoadLoop { body, .. } => *body = selected,
                        _ => return Err("invalid retained widening scope".into()),
                    }
                    output.push(statement);
                },
            }
        }
        Ok(output)
    }
    fn guarded(&self, end: i64, then: Vec<Stmt>, els: Vec<Stmt>) -> Stmt {
        self.guarded_symbol(Sym::constant(end), then, els)
    }
    fn guarded_symbol(&self, end: Sym, then: Vec<Stmt>, els: Vec<Stmt>) -> Stmt {
        use crate::{ast::BinaryOp, types::DType};
        let kind = end.as_constant().map(ExprKind::Int).unwrap_or_else(|| ExprKind::ShapeParam("widening_tail_limit".into()));
        let literal = Expr { kind, ty: Ty::Scalar(DType::I32), sym: Some(end), span: self.base.span };
        let condition = Expr { kind: ExprKind::Binary { op: BinaryOp::Lt, lhs: Box::new(self.base.clone()), rhs: Box::new(literal) },
            ty: Ty::Scalar(DType::Bool), sym: None, span: self.base.span };
        Stmt { id: None, kind: StmtKind::If { cond: condition, then, els }, span: self.base.span }
    }
    fn instantiate_regions(&self, regions: &[Region], width: i64, vars: &mut Vec<Var>, copies: &mut [HashMap<VarId, VarId>]) -> Result<Vec<Stmt>, String> {
        let mut output = Vec::new();
        for region in regions {
            match region {
                Region::Shared(statement) => output.push(statement.clone()),
                Region::Replicated { statement, .. } => {
                    for (ordinal, copies) in copies.iter_mut().enumerate() {
                        let value = super::offset(&self.base, ordinal as i64);
                        output.push(super::substitute(statement, self.inner, &self.atom, &value, vars, copies));
                    }
                },
                Region::SharedScope { header, body } => {
                    let mut statement = header.clone();
                    let selected = self.instantiate_regions(body, width, vars, copies)?;
                    match &mut statement.kind {
                        StmtKind::Range { body, .. } | StmtKind::LoadLoop { body, .. } => *body = selected,
                        _ => return Err("invalid retained widening scope".into()),
                    }
                    output.push(statement);
                },
            }
        }
        Ok(output)
    }
}
fn retain(body: &[Stmt], inner: VarId, atom: &Atom, width: &Sym, vars: &[Var], copied: &mut BTreeSet<VarId>) -> Vec<Region> {
    // The partition is structural; numeric width changes occurrence counts,
    // not which original dependencies make a statement copy-local.
    let plan = super::plan_at(body, inner, atom, 1);
    let shallow = HashSet::from([inner]);
    body.iter().enumerate().map(|(index, statement)| {
        if plan.shared.contains(&index) { return Region::Shared(statement.clone()); }
        let shared_body = match &statement.kind {
            StmtKind::LoadLoop { domain, views, body, .. }
                if !std::iter::once(&domain.view).chain(views).any(|view| super::expr_depends_shallow(view, &shallow, atom)) => Some(body),
            StmtKind::Range { lo, hi, body, .. } if !lo.atoms().contains(atom) && !hi.atoms().contains(atom) => Some(body),
            _ => None,
        };
        if let Some(body) = shared_body {
            let retained = retain(body, inner, atom, width, vars, copied);
            let mut header = statement.clone(); header.id = None;
            match &mut header.kind { StmtKind::LoadLoop { body, .. } | StmtKind::Range { body, .. } => body.clear(), _ => unreachable!() }
            return Region::SharedScope { header, body: retained };
        }
        let mut bindings = HashSet::new();
        crate::rewrite::value_writes(statement, vars, &mut bindings);
        super::bound_vars(statement, &mut bindings);
        bindings.remove(&inner);
        let mut bindings = bindings.into_iter().collect::<Vec<_>>(); bindings.sort_unstable();
        copied.extend(bindings.iter().copied());
        Region::Replicated { statement: statement.clone(), count: width.clone(), bindings }
    }).collect()
}
