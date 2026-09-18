//! Lowering by inlining: for a target backend and concrete shapes, replace every
//! construct call by the selected `lower` block (or the construct's portable body)
//! until only primitives and intrinsics remain.

use crate::ir::*;
use crate::lowered_ir::*;
use crate::program::Program;
use crate::sym::{Atom, Sym};
use crate::types::{Elem, Shaped, Ty};
use std::collections::HashMap;
pub mod alternatives;

/// Sizes the model will close; explicit for now.
#[derive(Clone, Debug, Default)]
pub struct Options {
    /// Explicit legal piece capacity. `None` uses the whole axis, bounded by its
    /// backing view for runtime extents. This is a baseline, not a performance choice.
    pub piece: Option<i64>,
}

pub fn lower(program: &Program, name: &str, backend: &str, shapes: &HashMap<String, i64>) -> Result<LoweredIr, String> {
    lower_with(program, name, backend, shapes, &Options::default())
}

pub fn lower_with(program: &Program, name: &str, backend: &str, shapes: &HashMap<String, i64>, opts: &Options) -> Result<LoweredIr, String> {
    lower_specialized(program, name, backend, shapes, &HashMap::new(), opts)
}

/// Bind entry element parameters as well as shapes before choosing lowerings.
pub fn lower_specialized(program: &Program, name: &str, backend: &str, shapes: &HashMap<String, i64>, elements: &HashMap<String, Elem>, opts: &Options) -> Result<LoweredIr, String> {
    lower_selected(program, name, backend, shapes, elements, opts, &mut |decision| {
        // Deterministic diagnostic baseline only. It makes no optimality claim.
        decision.alternatives.first().cloned().ok_or_else(||
            format!("empty decision domain on `{backend}`: {:?}", decision.kind))
    })
}

/// Expand only choices made by the caller, after checking their applicability.
/// Every construct decision is exposed, including singleton domains and portable
/// alternatives. An invalid choice fails; there is no silent preferred fallback.
#[allow(clippy::too_many_arguments)]
pub fn lower_selected(
    program: &Program, name: &str, backend: &str,
    shapes: &HashMap<String, i64>, elements: &HashMap<String, Elem>, opts: &Options,
    select: &mut dyn FnMut(&Decision) -> Result<Alternative, String>,
) -> Result<LoweredIr, String> {
    if opts.piece.is_some_and(|capacity| capacity <= 0) {
        return Err("stream piece capacity must be positive".into());
    }
    let f = program.functions.iter().find(|f| f.name == name).ok_or_else(|| format!("no function `{name}`"))?;
    let mut decisions = Vec::new();
    let mut recording = |domain: &Decision| {
        let selected = select(domain)?;
        if !domain.alternatives.contains(&selected) {
            return Err(format!("selected alternative {selected:?} is not applicable to {:?}", domain.kind));
        }
        decisions.push(DecisionRecord { domain: domain.clone(), selected: selected.clone() });
        Ok(selected)
    };
    let mut ctx = Inliner { program, backend, select: &mut recording, selections: Vec::new(), counter: 0, opts: opts.clone(), piece_values: HashMap::new(), elements: elements.clone() };
    let env: HashMap<String, Sym> = shapes.iter().map(|(k, v)| (k.clone(), Sym::constant(*v))).collect();
    for p in &f.shape_params {
        if !shapes.contains_key(p) {
            return Err(format!("shape parameter `{p}` of `{name}` is not bound"));
        }
    }
    crate::program::validate_element_bindings(f, elements)?;
    let mut vars: Vec<Var> = f.vars.iter().map(|v| Var { ty: subst_elem_ty(&subst_ty(&v.ty, &env), elements), ..v.clone() }).collect();
    let mut body = ctx.inline_block(&f.body, &env, &HashMap::new(), &mut vars, &mut HashMap::new(), 0)?;
    select_producers(&mut body, &vars, ctx.select)?;
    let params = f.params.iter().map(|(n, t)| (n.clone(), subst_elem_ty(&subst_ty(t, &env), elements))).collect();
    let index_params = f.index_params.iter().map(|(name,bound)| (name.clone(), subst_sym(bound,&env,&HashMap::new()))).collect();
    Ok(LoweredIr { name: name.to_string(), backend: backend.to_string(), params, index_params, vars, body, shapes: shapes.clone(), selections: ctx.selections, decisions })
}

/// A slice cannot exceed its parent axis. Follow that structural bound rather
/// than assigning a performance-dependent default to a runtime-sized view.
fn view_axis_capacity(view: &Expr, axis: usize) -> Result<i64, String> {
    let shaped = view.ty.shaped().ok_or("stream requires a shaped view")?;
    let extent = shaped.shape.get(axis).ok_or("invalid stream axis")?;
    if let Some(n) = extent.as_constant() {
        return if n >= 0 { Ok(n) } else { Err("negative stream extent".into()) };
    }
    match &view.kind {
        ExprKind::Index { base, indices } => {
            let rank = base.ty.shaped().ok_or("indexed stream requires a shaped parent")?.shape.len();
            let parent_axis = (0..rank)
                .filter(|i| !matches!(indices.get(*i), Some(Index::Point(_))))
                .nth(axis).ok_or("invalid indexed stream axis")?;
            view_axis_capacity(base, parent_axis)
        }
        ExprKind::Transpose(base) => {
            let rank = shaped.shape.len();
            view_axis_capacity(base, rank - 1 - axis)
        }
        _ => Err(format!("stream axis `{extent}` has no proven static capacity")),
    }
}

struct Inliner<'a> {
    program: &'a Program,
    backend: &'a str,
    select: &'a mut dyn FnMut(&Decision) -> Result<Alternative, String>,
    selections: Vec<Selection>,
    counter: usize,
    opts: Options,
    elements: HashMap<String, Elem>,
    /// Piece atoms over static extents: the concrete extents their pieces take (the capacity
    /// and the tail), so residuals can be decided exactly.
    piece_values: HashMap<String, Vec<i64>>,
}

/// Maps a callee's variable ids to expressions in the caller (parameters bound to arguments,
/// locals renamed into the caller's table).
type VarMap = HashMap<VarId, Expr>;

impl<'a> Inliner<'a> {
    fn inline_block(&mut self, stmts: &[Stmt], env: &HashMap<String, Sym>, vmap: &VarMap, vars: &mut Vec<Var>, atom_map: &mut HashMap<String, Atom>, depth: usize) -> Result<Vec<Stmt>, String> {
        if depth > 32 {
            return Err("lowering recursion too deep".into());
        }
        let mut out = Vec::new();
        for s in stmts {
            out.extend(self.inline_stmt(s, env, vmap, vars, atom_map, depth)?);
        }
        Ok(out)
    }

    fn inline_stmt(&mut self, s: &Stmt, env: &HashMap<String, Sym>, vmap: &VarMap, vars: &mut Vec<Var>, atom_map: &mut HashMap<String, Atom>, depth: usize) -> Result<Vec<Stmt>, String> {
        let span = s.span;
        let kind = match &s.kind {
            StmtKind::Parallel { vars: vs, extents, body } => StmtKind::Parallel {
                vars: vs.iter().map(|v| remap_var(*v, vmap)).collect(),
                extents: extents.iter().map(|e| subst_sym(e, env, atom_map)).collect(),
                body: self.inline_block(body, env, vmap, vars, atom_map, depth)?,
            },
            StmtKind::LoadLoop { vars: vs, views, axis, piece, body, .. } => {
                let views: Vec<Expr> = views.iter().map(|v| self.inline_expr(v, env, vmap, vars, atom_map)).collect::<Result<_, _>>()?;
                // This version streams the whole axis as one piece; the piece extent is the axis extent.
                let extent = match &views[0].ty {
                    Ty::Tensor(s) => s.shape[*axis].clone(),
                    Ty::Tuple(items) => match &items[0] {
                        Ty::Tensor(s) => s.shape[*axis].clone(),
                        _ => unreachable!(),
                    },
                    other => return Err(format!("load loop over {other}")),
                };
                let Atom::Param(pname) = piece else { unreachable!() };
                let mut inner_env = env.clone();
                let capacity = match (extent.as_constant(), self.opts.piece) {
                    (Some(e), Some(c)) if e > c => {
                        // Static extent chunked: pieces of `c` and a tail of `e % c`.
                        let mut values = vec![c];
                        if e % c != 0 {
                            values.push(e % c);
                        }
                        self.piece_values.insert(pname.clone(), values);
                        Some(c)
                    }
                    (Some(_), _) => {
                        inner_env.insert(pname.clone(), extent);
                        None
                    }
                    // Dynamic extent: static pieces of a chosen capacity; the atom stays symbolic.
                    (None, explicit) => {
                        let bound = views.iter().map(|view| view_axis_capacity(view, *axis))
                            .collect::<Result<Vec<_>, _>>()?.into_iter().min()
                            .ok_or("stream requires a view")?;
                        // Even an empty backing axis needs a positive loop step; the
                        // runtime domain remains empty and performs no reads.
                        Some(explicit.unwrap_or(bound.max(1)))
                    },
                };
                let new_vars: Vec<VarId> = vs.iter().map(|v| remap_var(*v, vmap)).collect();
                for v in &new_vars {
                    vars[*v].ty = subst_ty(&vars[*v].ty, &inner_env);
                }
                StmtKind::LoadLoop {
                    modes: None,
                    vars: new_vars,
                    views,
                    axis: *axis,
                    piece: remap_atom(piece, atom_map),
                    capacity,
                    body: self.inline_block(body, &inner_env, vmap, vars, atom_map, depth)?,
                }
            }
            StmtKind::Owned { vars: vs, tile, body } => StmtKind::Owned {
                vars: vs.iter().map(|v| remap_var(*v, vmap)).collect(),
                tile: self.inline_expr(tile, env, vmap, vars, atom_map)?,
                body: self.inline_block(body, env, vmap, vars, atom_map, depth)?,
            },
            StmtKind::Range { var, lo, hi, body } => StmtKind::Range {
                var: remap_var(*var, vmap),
                lo: subst_sym(lo, env, atom_map),
                hi: subst_sym(hi, env, atom_map),
                body: self.inline_block(body, env, vmap, vars, atom_map, depth)?,
            },
            StmtKind::Lanes { var, extent, width, body } => StmtKind::Lanes {
                var: remap_var(*var, vmap),
                extent: subst_sym(extent, env, atom_map),
                width: *width,
                body: self.inline_block(body, env, vmap, vars, atom_map, depth)?,
            },
            StmtKind::If { cond, then, els } => StmtKind::If {
                cond: self.inline_expr(cond, env, vmap, vars, atom_map)?,
                then: self.inline_block(then, env, vmap, vars, atom_map, depth)?,
                els: self.inline_block(els, env, vmap, vars, atom_map, depth)?,
            },
            StmtKind::Assign { target, op, value } => StmtKind::Assign {
                target: self.inline_expr(target, env, vmap, vars, atom_map)?,
                op: *op,
                value: self.inline_expr(value, env, vmap, vars, atom_map)?,
            },
            StmtKind::Expr(e) => {
                if let ExprKind::Call { callee, shape_args, elem_args, args } = &e.kind {
                    let elem_args = elem_args.iter().map(|e| subst_elem(e, &self.elements)).collect::<Vec<_>>();
                    return self.inline_call(callee, shape_args, &elem_args, args, env, vmap, vars, atom_map, depth);
                }
                StmtKind::Expr(self.inline_expr(e, env, vmap, vars, atom_map)?)
            }
        };
        Ok(vec![Stmt { id: None, kind, span }])
    }

    #[allow(clippy::too_many_arguments)]
    fn inline_call(&mut self, callee: &str, shape_args: &[Sym], elem_args: &[Elem], args: &[Expr], env: &HashMap<String, Sym>, vmap: &VarMap, vars: &mut Vec<Var>, atom_map: &mut HashMap<String, Atom>, depth: usize) -> Result<Vec<Stmt>, String> {
        let f = self.program.functions.iter().find(|f| f.name == callee).ok_or_else(|| format!("no function `{callee}`"))?;
        // Concrete shape arguments in the caller's environment.
        // Shape arguments may stay symbolic when they carry a piece extent; a block whose
        // residual depends on such an argument is then not applicable and the portable body is used.
        let mut inner_env: HashMap<String, Sym> = HashMap::new();
        let mut concrete = Vec::new();
        for (p, s) in f.shape_params.iter().zip(shape_args) {
            let v = subst_sym(s, env, atom_map);
            concrete.push(v.as_constant().unwrap_or(-1));
            inner_env.insert(p.clone(), v);
        }
        let inlined_args: Vec<Expr> = args.iter().map(|a| self.inline_expr(a, env, vmap, vars, atom_map)).collect::<Result<_, _>>()?;
        // Choose a body.
        let (body, body_vars, choice) = if f.is_construct {
            let blocks: Vec<&Lowering> = self.program.lowerings.iter().filter(|l| l.construct == callee && l.backend == self.backend).collect();
            let applicable = |l: &Lowering| -> bool {
                let elems_ok = l.elem_bindings.iter().all(|(p, e)| {
                    let idx = f.elem_params.iter().position(|x| x == p);
                    idx.map(|i| &elem_args[i] == e).unwrap_or(false)
                });
                // A residual over a chunked piece must hold for every extent the piece takes.
                let residual_ok = l.residual.iter().all(|r| {
                    let mut assignments: Vec<HashMap<String, i64>> = vec![HashMap::new()];
                    for p in r.params() {
                        let Some(s) = inner_env.get(&p) else { return false };
                        let values: Vec<i64> = match s.as_constant() {
                            Some(v) => vec![v],
                            None => match single_piece(s) {
                                Some(atom) => match self.piece_values.get(&atom) {
                                    Some(vs) => vs.clone(),
                                    None => return false,
                                },
                                None => return false,
                            },
                        };
                        let mut next = Vec::new();
                        for a in &assignments {
                            for v in &values {
                                let mut b = a.clone();
                                b.insert(p.clone(), *v);
                                next.push(b);
                            }
                        }
                        assignments = next;
                    }
                    assignments.iter().all(|a| r.eval(&|p| a.get(p).copied()).map(|v| v >= 0).unwrap_or(false))
                });
                elems_ok && residual_ok
            };
            let mut alternatives = Vec::new();
            let mut portable = false;
            for (i, block) in blocks.iter().enumerate() {
                if block.body.is_empty() && block.residual.is_empty() && block.elem_bindings.is_empty() {
                    portable = true;
                } else if applicable(block) {
                    alternatives.push(Alternative::Body(Choice::Block(i)));
                }
            }
            if portable {
                alternatives.push(Alternative::Body(Choice::Portable));
            }
            let decision = Decision {
                kind: DecisionKind::Construct {
                    name: callee.to_string(),
                    shape_args: f.shape_params.iter().map(|p| inner_env[p].clone()).collect(),
                    element_args: elem_args.to_vec(),
                },
                alternatives,
            };
            if decision.alternatives.is_empty() {
                return Err(format!("no lowering of `{callee}` on `{}` applies to shapes {:?} and elements {:?}", self.backend, concrete, elem_args));
            }
            let choice = (self.select)(&decision)?;
            if !decision.alternatives.contains(&choice) {
                return Err(format!("selected lowering {choice:?} is not applicable to `{callee}`; legal alternatives: {:?}", decision.alternatives));
            }
            match choice {
                Alternative::Body(Choice::Block(i)) => (blocks[i].body.clone(), blocks[i].vars.clone(), Choice::Block(i)),
                Alternative::Body(Choice::Portable) => (f.body.clone(), f.vars.clone(), Choice::Portable),
                _ => unreachable!("validated construct domain"),
            }
        } else {
            (f.body.clone(), f.vars.clone(), Choice::Portable)
        };
        self.selections.push(Selection { construct: callee.to_string(), shape_args: concrete, choice });
        // Element substitution for the callee's generic element types.
        let mut elem_env: HashMap<String, Elem> = HashMap::new();
        for (p, e) in f.elem_params.iter().zip(elem_args) {
            elem_env.insert(p.clone(), e.clone());
        }
        // Build the variable map: parameters bind to argument expressions; locals are appended.
        let mut inner_map: VarMap = HashMap::new();
        let mut inner_atoms: HashMap<String, Atom> = HashMap::new();
        for (id, v) in body_vars.iter().enumerate() {
            match &v.kind {
                VarKind::Param(i) => {
                    inner_map.insert(id, inlined_args[*i].clone());
                }
                VarKind::Local => {
                    let new_id = vars.len();
                    let ty = subst_elem_ty(&subst_ty(&v.ty, &inner_env), &elem_env);
                    vars.push(Var { name: format!("{}_{}", v.name, self.fresh()), ty: ty.clone(), span: v.span, kind: VarKind::Local });
                    inner_map.insert(id, Expr { kind: ExprKind::Var(new_id), ty, sym: None, span: v.span });
                }
                VarKind::Index(atom) => {
                    let new_id = vars.len();
                    let Atom::Param(name) = atom else { unreachable!() };
                    let new_atom = Atom::Param(format!("{name}_{}", self.fresh()));
                    inner_atoms.insert(name.clone(), new_atom.clone());
                    vars.push(Var { name: format!("{}_{}", v.name, self.fresh()), ty: v.ty.clone(), span: v.span, kind: VarKind::Index(new_atom.clone()) });
                    inner_map.insert(id, Expr { kind: ExprKind::Var(new_id), ty: v.ty.clone(), sym: Some(Sym::atom(new_atom)), span: v.span });
                }
            }
        }
        let caller_elements = std::mem::replace(&mut self.elements, elem_env);
        let result = self.inline_block(&body, &inner_env, &inner_map, vars, &mut inner_atoms, depth + 1);
        self.elements = caller_elements;
        result
    }

    fn fresh(&mut self) -> usize {
        self.counter += 1;
        self.counter
    }

    fn inline_expr(&mut self, e: &Expr, env: &HashMap<String, Sym>, vmap: &VarMap, vars: &mut Vec<Var>, atom_map: &mut HashMap<String, Atom>) -> Result<Expr, String> {
        let ty = subst_elem_ty(&subst_ty(&e.ty, env), &self.elements);
        let sym = e.sym.as_ref().map(|s| subst_sym(s, env, atom_map));
        let span = e.span;
        let mut sub = |x: &Expr, this: &mut Self| this.inline_expr(x, env, vmap, vars, atom_map);
        let kind = match &e.kind {
            ExprKind::Load { .. } => return Err("selected execution loads cannot appear in lowering definitions".into()),
            ExprKind::Var(id) => {
                if let Some(bound) = vmap.get(id) {
                    let mut b = bound.clone();
                    // A remapped index variable carries its renamed atom.
                    if let Some(s) = &sym {
                        if matches!(b.kind, ExprKind::Var(_)) && b.sym.is_some() {
                            b.sym = Some(s.clone());
                        }
                    }
                    return Ok(b);
                }
                ExprKind::Var(*id)
            }
            ExprKind::ShapeParam(p) => match sym.clone().and_then(|s| s.as_constant()) {
                Some(v) => ExprKind::Int(v),
                // A piece extent stays symbolic; the printer reads it from `sym`.
                None => ExprKind::ShapeParam(p.clone()),
            },
            ExprKind::Int(v) => ExprKind::Int(*v),
            ExprKind::Float(v) => ExprKind::Float(*v),
            ExprKind::Bool(b) => ExprKind::Bool(*b),
            ExprKind::TileAlloc { shape, dtype } => ExprKind::TileAlloc { shape: shape.iter().map(|s| subst_sym(s, env, atom_map)).collect(), dtype: match subst_elem(dtype, &self.elements) {
                Elem::Repr(_) => return Err("local tile allocation requires a dense dtype".into()),
                dtype => dtype,
            } },
            ExprKind::Index { base, indices } => ExprKind::Index {
                base: Box::new(sub(base, self)?),
                indices: indices
                    .iter()
                    .map(|i| {
                        Ok(match i {
                            Index::Point(p) => Index::Point(self.inline_expr(p, env, vmap, vars, atom_map)?),
                            Index::Slice { start, end } => Index::Slice {
                                start: start.as_ref().map(|x| self.inline_expr(x, env, vmap, vars, atom_map)).transpose()?,
                                end: end.as_ref().map(|x| self.inline_expr(x, env, vmap, vars, atom_map)).transpose()?,
                            },
                        })
                    })
                    .collect::<Result<_, String>>()?,
            },
            ExprKind::Transpose(inner) => ExprKind::Transpose(Box::new(sub(inner, self)?)),
            ExprKind::Accessor { base, name } => ExprKind::Accessor { base: Box::new(sub(base, self)?), name: name.clone() },
            ExprKind::Lanes { base, extent } => ExprKind::Lanes { base: Box::new(sub(base, self)?), extent: subst_sym(extent, env, atom_map) },
            ExprKind::Builtin { name, args } => {
                let args=args.iter().map(|a|self.inline_expr(a,env,vmap,vars,atom_map)).collect::<Result<Vec<_>,_>>()?;
                if *name == Builtin::Reshape && matches!(args[0].ty.shaped().map(|s| &s.elem), Some(Elem::Repr(_))) {
                    return Err("reshape currently requires dense storage".into());
                }
                if *name == Builtin::Store {
                    let source = args[0].ty.shaped().ok_or("store source has no shape")?;
                    let target = args[1].ty.shaped().ok_or("store target has no shape")?;
                    if !matches!((&source.elem, &target.elem), (Elem::Dtype(a), Elem::Dtype(b)) if a == b || (a.is_float() && b.is_float())) {
                        return Err(format!("store specialization cannot publish {} into {}", source.elem, target.elem));
                    }
                }
                if *name==Builtin::Reduce && matches!(args.get(2).map(|e|&e.kind),Some(ExprKind::Int(3))) {
                    let axis=args[1].sym.as_ref().and_then(Sym::as_constant).and_then(|n|usize::try_from(n).ok()).ok_or("argmax axis unresolved")?;
                    let extent=args[0].ty.shaped().and_then(|s|s.shape.get(axis)).and_then(Sym::as_constant).ok_or("argmax runtime nonempty-domain validation is not implemented")?;
                    if extent<=0{return Err("argmax requires a nonempty axis after shape specialization".into());}
                }
                ExprKind::Builtin{name:*name,args}
            },
            ExprKind::Call { .. } => return Err("a call in expression position cannot be inlined; calls are statements".into()),
            ExprKind::Intrinsic { op: name, args } => ExprKind::Intrinsic { op: *name, args: args.iter().map(|a| self.inline_expr(a, env, vmap, vars, atom_map)).collect::<Result<_, _>>()? },
            ExprKind::Unary { op, expr } => ExprKind::Unary { op: *op, expr: Box::new(sub(expr, self)?) },
            ExprKind::Binary { op, lhs, rhs } => ExprKind::Binary { op: *op, lhs: Box::new(sub(lhs, self)?), rhs: Box::new(sub(rhs, self)?) },
            ExprKind::Cast { dtype, expr } => ExprKind::Cast { dtype: *dtype, expr: Box::new(sub(expr, self)?) },
            ExprKind::Tuple(items) => ExprKind::Tuple(items.iter().map(|a| self.inline_expr(a, env, vmap, vars, atom_map)).collect::<Result<_, _>>()?),
        };
        Ok(Expr { kind, ty, sym, span })
    }
}

fn remap_var(v: VarId, vmap: &VarMap) -> VarId {
    match vmap.get(&v) {
        Some(Expr { kind: ExprKind::Var(id), .. }) => *id,
        Some(_) => panic!("loop variable bound to a non-variable"),
        None => v,
    }
}

fn remap_atom(a: &Atom, atom_map: &HashMap<String, Atom>) -> Atom {
    match a {
        Atom::Param(p) => atom_map.get(p).cloned().unwrap_or_else(|| a.clone()),
        _ => a.clone(),
    }
}

/// Substitute shape parameters by their bindings and rename loop atoms.
/// The piece atom name when a symbol is exactly one piece atom.
fn single_piece(s: &Sym) -> Option<String> {
    let params = s.params();
    if params.len() == 1 && *s == Sym::param(&params[0]) && params[0].contains('#') {
        Some(params[0].clone())
    } else {
        None
    }
}

pub fn subst_sym(s: &Sym, env: &HashMap<String, Sym>, atom_map: &HashMap<String, Atom>) -> Sym {
    let mut out = s.clone();
    for a in s.atoms() {
        match &a {
            Atom::Param(p) => {
                if let Some(v) = env.get(p) {
                    out = out.subst(&a, v);
                } else if let Some(n) = atom_map.get(p) {
                    out = out.subst(&a, &Sym::atom(n.clone()));
                }
            }
            Atom::Quot(n, d) => {
                let r = Sym::atom(Atom::Quot(Box::new(subst_sym(n, env, atom_map)), Box::new(subst_sym(d, env, atom_map))));
                let r = simplify_div(&r);
                out = out.subst(&a, &r);
            }
            Atom::Rem(n, d) => {
                let r = Sym::atom(Atom::Rem(Box::new(subst_sym(n, env, atom_map)), Box::new(subst_sym(d, env, atom_map))));
                let r = simplify_div(&r);
                out = out.subst(&a, &r);
            }
        }
    }
    out
}

/// Re-normalize a quotient or remainder atom whose parts may now be constants.
fn simplify_div(s: &Sym) -> Sym {
    if let [atom] = s.atoms().as_slice() {
        if *s == Sym::atom(atom.clone()) {
            match atom {
                Atom::Quot(n, d) => return n.quot(d),
                Atom::Rem(n, d) => return n.rem(d),
                _ => {}
            }
        }
    }
    s.clone()
}

pub fn subst_ty(t: &Ty, env: &HashMap<String, Sym>) -> Ty {
    let empty = HashMap::new();
    let f = |s: &Shaped| Shaped { shape: s.shape.iter().map(|d| subst_sym(d, env, &empty)).collect(), elem: s.elem.clone(), packed_axis: s.packed_axis };
    match t {
        Ty::Tensor(s) => Ty::Tensor(f(s)),
        Ty::Tile(s) => Ty::Tile(f(s)),
        Ty::Frag(s) => Ty::Frag(f(s)),
        Ty::Tuple(items) => Ty::Tuple(items.iter().map(|i| subst_ty(i, env)).collect()),
        other => other.clone(),
    }
}

pub(crate) fn subst_elem(elem: &Elem, elems: &HashMap<String, Elem>) -> Elem {
    match elem {
        Elem::Param(p) => elems.get(p).cloned().unwrap_or_else(|| elem.clone()),
        other => other.clone(),
    }
}

fn subst_elem_ty(t: &Ty, elems: &HashMap<String, Elem>) -> Ty {
    let f = |s: &Shaped| {
        let elem = subst_elem(&s.elem, elems);
        let packed_axis = match (&elem, s.packed_axis) {
            (Elem::Repr(_), None) => Some(s.shape.len().saturating_sub(1)),
            (_, a) => a,
        };
        Shaped { shape: s.shape.clone(), elem, packed_axis }
    };
    match t {
        Ty::Tensor(s) => Ty::Tensor(f(s)),
        Ty::Tile(s) => Ty::Tile(f(s)),
        Ty::Frag(s) => Ty::Frag(f(s)),
        Ty::Tuple(items) => Ty::Tuple(items.iter().map(|i| subst_elem_ty(i, elems)).collect()),
        other => other.clone(),
    }
}

/// Select legal producer materialization after expansion is complete. A producer
/// is offered once; nested expansion cannot override a previous retention decision.
fn select_producers(
    block: &mut Vec<Stmt>, vars: &[Var],
    select: &mut dyn FnMut(&Decision) -> Result<Alternative, String>,
) -> Result<(), String> {
    for statement in block.iter_mut() {
        match &mut statement.kind {
            StmtKind::Parallel { body, .. } | StmtKind::Range { body, .. }
            | StmtKind::Owned { body, .. } | StmtKind::Lanes { body, .. }
            | StmtKind::LoadLoop { body, .. } => select_producers(body, vars, select)?,
            StmtKind::If { then: then_body, els: else_body, .. } => {
                select_producers(then_body, vars, select)?;
                select_producers(else_body, vars, select)?;
            }
            _ => {}
        }
    }
    // Only a tile allocated in this block can be removed here. A write to a
    // loop-carried or enclosing tile escapes this block even without a local read.
    let allocated: std::collections::HashSet<VarId> = block.iter().filter_map(|s| match &s.kind {
        StmtKind::Assign { target, value, .. } if matches!(value.kind, ExprKind::TileAlloc { .. }) => match target.kind { ExprKind::Var(v) => Some(v), _ => None },
        _ => None,
    }).collect();
    let mut writes = Writes::default();
    for statement in block.iter() { written_vars(statement, &mut writes); }
    // Producers at this level: `for i.. in owned(a): a[i..] = value`.
    let mut producers: HashMap<VarId, (Vec<VarId>, Expr)> = HashMap::new();
    for (position, s) in block.iter().enumerate() {
        if let StmtKind::Owned { vars: ivs, tile, body } = &s.kind {
            let ExprKind::Var(a) = tile.kind else { continue };
            if !matches!(vars[a].kind, VarKind::Local) || !allocated.contains(&a) || writes.variables.get(&a) != Some(&2) {
                continue;
            }
            let [only] = body.as_slice() else { continue };
            let StmtKind::Assign { target, op: crate::ast::AssignOp::Assign, value } = &only.kind else { continue };
            let ExprKind::Index { base, indices } = &target.kind else { continue };
            if !matches!(base.kind, ExprKind::Var(b) if b == a) {
                continue;
            }
            let plain = indices.iter().zip(ivs).all(|(ix, v)| matches!(ix, Index::Point(p) if matches!(p.kind, ExprKind::Var(x) if x == *v)));
            if !plain || indices.len() != ivs.len() || mentions_var(value, a) {
                continue;
            }
            // Recomputing a value later is legal only while all its source values
            // remain unchanged. Unknown memory effects conservatively prevent motion.
            let mut dependencies = std::collections::HashSet::new();
            walk_expr(value, &mut |e| { if let ExprKind::Var(v) = e.kind { if !ivs.contains(&v) { dependencies.insert(v); } } });
            let mut future_writes = Writes::default();
            for later in &block[position + 1..] { written_vars(later, &mut future_writes); }
            if future_writes.unknown || (future_writes.tensors && dependencies.iter().any(|v| matches!(vars[*v].ty, Ty::Tensor(_)))) || dependencies.iter().any(|v| future_writes.variables.contains_key(v)) { continue; }
            producers.insert(a, (ivs.clone(), value.clone()));
        }
    }
    if producers.is_empty() {
        return Ok(());
    }
    // Keep only producers whose every other use is an element read.
    let mut other_uses: HashMap<VarId, usize> = HashMap::new();
    for s in block.iter() {
        count_non_element_uses(s, &producers, &mut other_uses);
    }
    producers.retain(|a, _| other_uses.get(a).copied().unwrap_or(0) == 0);
    if producers.is_empty() {
        return Ok(());
    }
    // Sort by the source variable identity; HashMap iteration must never affect
    // replay, candidate identity, or coverage of independent decisions.
    let mut candidates: Vec<_> = producers.keys().copied().collect();
    candidates.sort_unstable();
    for variable in candidates {
        let decision = Decision {
            kind: DecisionKind::Producer {
                variable, name: vars[variable].name.clone(), ty: vars[variable].ty.clone(),
            },
            alternatives: vec![Alternative::Materialize, Alternative::Recompute],
        };
        match select(&decision)? {
            Alternative::Materialize => { producers.remove(&variable); }
            Alternative::Recompute => {}
            other => return Err(format!("invalid producer alternative {other:?}")),
        }
    }
    // Drop their definitions and rewrite the reads.
    block.retain(|s| match &s.kind {
        StmtKind::Owned { tile, .. } => !matches!(tile.kind, ExprKind::Var(a) if producers.contains_key(&a)),
        StmtKind::Assign { target, value, .. } => !(matches!(target.kind, ExprKind::Var(a) if producers.contains_key(&a)) && matches!(value.kind, ExprKind::TileAlloc { .. })),
        _ => true,
    });
    for s in block.iter_mut() {
        rewrite_reads_stmt(s, &producers, vars);
    }
    Ok(())
}

#[derive(Default)]
struct Writes {
    variables: HashMap<VarId, usize>,
    tensors: bool,
    unknown: bool,
}
/// Syntactic effects across nested control flow. Materialized tile values cannot
/// alias tensor storage, while tensor reads require an alias proof to cross stores.
fn written_vars(statement: &Stmt, writes: &mut Writes) {
    fn expression(e: &Expr, writes: &mut Writes) {
        walk_expr(e, &mut |e| {
            if matches!(e.kind, ExprKind::Intrinsic { .. } | ExprKind::Call { .. } | ExprKind::Builtin { name: Builtin::Atomic, .. }) { writes.unknown = true; }
            if matches!(e.kind, ExprKind::Builtin { name: Builtin::Store, .. }) { writes.tensors = true; }
        });
    }
    match &statement.kind {
        StmtKind::Assign { target, value, .. } => {
            expression(value, writes);
            let mut root = target;
            while let ExprKind::Index { base, .. } | ExprKind::Transpose(base) = &root.kind { root = base; }
            if let ExprKind::Var(v) = root.kind { *writes.variables.entry(v).or_default() += 1; }
        }
        StmtKind::Expr(e) => expression(e, writes),
        StmtKind::Parallel { body, .. } | StmtKind::Range { body, .. } | StmtKind::Lanes { body, .. } => {
            for child in body { written_vars(child, writes); }
        }
        StmtKind::Owned { tile, body, .. } => {
            expression(tile, writes);
            for child in body { written_vars(child, writes); }
        }
        StmtKind::LoadLoop { views, body, .. } => {
            for view in views { expression(view, writes); }
            for child in body { written_vars(child, writes); }
        }
        StmtKind::If { cond, then, els } => {
            expression(cond, writes);
            for child in then.iter().chain(els) { written_vars(child, writes); }
        }
    }
}

fn mentions_var(e: &Expr, a: VarId) -> bool {
    let mut found = false;
    walk_expr(e, &mut |x| {
        if matches!(x.kind, ExprKind::Var(v) if v == a) {
            found = true;
        }
    });
    found
}

fn walk_expr(e: &Expr, f: &mut dyn FnMut(&Expr)) {
    f(e);
    match &e.kind {
        ExprKind::Index { base, indices } => {
            walk_expr(base, f);
            for i in indices {
                match i {
                    Index::Point(p) => walk_expr(p, f),
                    Index::Slice { start, end } => {
                        if let Some(x) = start { walk_expr(x, f) }
                        if let Some(x) = end { walk_expr(x, f) }
                    }
                }
            }
        }
        ExprKind::Transpose(x) | ExprKind::Accessor { base: x, .. } | ExprKind::Lanes { base: x, .. } | ExprKind::Unary { expr: x, .. } | ExprKind::Cast { expr: x, .. } => walk_expr(x, f),
        ExprKind::Builtin { args, .. } | ExprKind::Intrinsic { args, .. } | ExprKind::Call { args, .. } | ExprKind::Tuple(args) => {
            for a in args {
                walk_expr(a, f);
            }
        }
        ExprKind::Binary { lhs, rhs, .. } => {
            walk_expr(lhs, f);
            walk_expr(rhs, f);
        }
        _ => {}
    }
}

/// Uses of a producer tile other than all-point element reads, its own definition excluded.
fn count_non_element_uses(s: &Stmt, producers: &HashMap<VarId, (Vec<VarId>, Expr)>, out: &mut HashMap<VarId, usize>) {
    let expr_uses = |e: &Expr, out: &mut HashMap<VarId, usize>| {
        // Walk manually so element reads of producers are not descended into as bare vars.
        fn go(e: &Expr, producers: &HashMap<VarId, (Vec<VarId>, Expr)>, out: &mut HashMap<VarId, usize>) {
            match &e.kind {
                ExprKind::Var(v) => {
                    if producers.contains_key(v) {
                        *out.entry(*v).or_insert(0) += 1;
                    }
                }
                ExprKind::Index { base, indices } => {
                    let all_points = indices.iter().all(|i| matches!(i, Index::Point(_)));
                    match &base.kind {
                        ExprKind::Var(v) if producers.contains_key(v) && all_points => {}
                        _ => go(base, producers, out),
                    }
                    for i in indices {
                        match i {
                            Index::Point(p) => go(p, producers, out),
                            Index::Slice { start, end } => {
                                if let Some(x) = start { go(x, producers, out) }
                                if let Some(x) = end { go(x, producers, out) }
                            }
                        }
                    }
                }
                ExprKind::Transpose(x) | ExprKind::Accessor { base: x, .. } | ExprKind::Lanes { base: x, .. } | ExprKind::Unary { expr: x, .. } | ExprKind::Cast { expr: x, .. } => go(x, producers, out),
                ExprKind::Builtin { args, .. } | ExprKind::Intrinsic { args, .. } | ExprKind::Call { args, .. } | ExprKind::Tuple(args) => {
                    for a in args {
                        go(a, producers, out);
                    }
                }
                ExprKind::Binary { lhs, rhs, .. } => {
                    go(lhs, producers, out);
                    go(rhs, producers, out);
                }
                _ => {}
            }
        }
        go(e, producers, out);
    };
    match &s.kind {
        StmtKind::Owned { tile, body, .. } => {
            // The producer's own loop is its definition; any other owned loop over it is a use.
            let own = matches!(tile.kind, ExprKind::Var(a) if producers.contains_key(&a)) && body.len() == 1 && matches!(&body[0].kind, StmtKind::Assign { target, .. } if matches!(&target.kind, ExprKind::Index { base, .. } if matches!(base.kind, ExprKind::Var(b) if matches!(tile.kind, ExprKind::Var(a) if a == b))));
            if !own {
                expr_uses(tile, out);
                for b in body {
                    count_non_element_uses(b, producers, out);
                }
            }
        }
        StmtKind::Assign { target, value, .. } => {
            let alloc = matches!(target.kind, ExprKind::Var(a) if producers.contains_key(&a)) && matches!(value.kind, ExprKind::TileAlloc { .. });
            if !alloc {
                // An element write elsewhere is a use that forbids inlining.
                if let ExprKind::Index { base, .. } = &target.kind {
                    if let ExprKind::Var(a) = base.kind {
                        if producers.contains_key(&a) {
                            *out.entry(a).or_insert(0) += 1;
                        }
                    }
                } else {
                    expr_uses(target, out);
                }
                expr_uses(value, out);
            }
        }
        StmtKind::Expr(e) => expr_uses(e, out),
        StmtKind::Parallel { body, .. } | StmtKind::Range { body, .. } | StmtKind::Lanes { body, .. } => {
            for b in body {
                count_non_element_uses(b, producers, out);
            }
        }
        StmtKind::LoadLoop { views, body, .. } => {
            for v in views {
                expr_uses(v, out);
            }
            for b in body {
                count_non_element_uses(b, producers, out);
            }
        }
        StmtKind::If { cond, then, els } => {
            expr_uses(cond, out);
            for b in then.iter().chain(els.iter()) {
                count_non_element_uses(b, producers, out);
            }
        }
    }
}

fn rewrite_reads_stmt(s: &mut Stmt, producers: &HashMap<VarId, (Vec<VarId>, Expr)>, vars: &[Var]) {
    match &mut s.kind {
        StmtKind::Owned { tile, body, .. } => {
            rewrite_reads(tile, producers, vars);
            for b in body {
                rewrite_reads_stmt(b, producers, vars);
            }
        }
        StmtKind::Assign { target, value, .. } => {
            rewrite_reads(target, producers, vars);
            rewrite_reads(value, producers, vars);
        }
        StmtKind::Expr(e) => rewrite_reads(e, producers, vars),
        StmtKind::Parallel { body, .. } | StmtKind::Range { body, .. } | StmtKind::Lanes { body, .. } => {
            for b in body {
                rewrite_reads_stmt(b, producers, vars);
            }
        }
        StmtKind::LoadLoop { views, body, .. } => {
            for v in views {
                rewrite_reads(v, producers, vars);
            }
            for b in body {
                rewrite_reads_stmt(b, producers, vars);
            }
        }
        StmtKind::If { cond, then, els } => {
            rewrite_reads(cond, producers, vars);
            for b in then.iter_mut().chain(els.iter_mut()) {
                rewrite_reads_stmt(b, producers, vars);
            }
        }
    }
}

fn rewrite_reads(e: &mut Expr, producers: &HashMap<VarId, (Vec<VarId>, Expr)>, vars: &[Var]) {
    if let ExprKind::Index { base, indices } = &e.kind {
        if let ExprKind::Var(a) = base.kind {
            if let Some((ivs, value)) = producers.get(&a) {
                if indices.iter().all(|i| matches!(i, Index::Point(_))) {
                    let points: Vec<Expr> = indices.iter().map(|i| match i { Index::Point(p) => { let mut p = p.clone(); rewrite_reads(&mut p, producers, vars); p } _ => unreachable!() }).collect();
                    let mut map: HashMap<VarId, Expr> = HashMap::new();
                    let mut atoms: HashMap<String, Option<Sym>> = HashMap::new();
                    for (v, p) in ivs.iter().zip(points) {
                        if let VarKind::Index(Atom::Param(name)) = &vars[*v].kind {
                            atoms.insert(name.clone(), p.sym.clone());
                        }
                        map.insert(*v, p);
                    }
                    let mut replaced = subst_vars(value, &map, &atoms);
                    rewrite_reads(&mut replaced, producers, vars);
                    // Preserve the eliminated tile's publication precision, even
                    // when its producer computed a wider intermediate expression.
                    if let Ty::Tile(tile) = &vars[a].ty {
                        if let Elem::Dtype(dtype) = tile.elem {
                            replaced = Expr { kind: ExprKind::Cast { dtype, expr: Box::new(replaced) }, ty: Ty::Scalar(dtype), sym: None, span: e.span };
                        }
                    }
                    if let Ty::Scalar(dtype) = e.ty {
                        if replaced.ty != e.ty {
                            replaced = Expr { kind: ExprKind::Cast { dtype, expr: Box::new(replaced) }, ty: e.ty.clone(), sym: None, span: e.span };
                        }
                    }
                    *e = replaced;
                    return;
                }
            }
        }
    }
    match &mut e.kind {
        ExprKind::Index { base, indices } => {
            rewrite_reads(base, producers, vars);
            for i in indices {
                match i {
                    Index::Point(p) => rewrite_reads(p, producers, vars),
                    Index::Slice { start, end } => {
                        if let Some(x) = start { rewrite_reads(x, producers, vars) }
                        if let Some(x) = end { rewrite_reads(x, producers, vars) }
                    }
                }
            }
        }
        ExprKind::Transpose(x) | ExprKind::Accessor { base: x, .. } | ExprKind::Lanes { base: x, .. } | ExprKind::Unary { expr: x, .. } | ExprKind::Cast { expr: x, .. } => rewrite_reads(x, producers, vars),
        ExprKind::Builtin { args, .. } | ExprKind::Intrinsic { args, .. } | ExprKind::Call { args, .. } | ExprKind::Tuple(args) => {
            for a in args {
                rewrite_reads(a, producers, vars);
            }
        }
        ExprKind::Binary { lhs, rhs, .. } => {
            rewrite_reads(lhs, producers, vars);
            rewrite_reads(rhs, producers, vars);
        }
        _ => {}
    }
}

/// Substitute loop-index variables by expressions, in kinds and in symbolic values.
fn subst_vars(e: &Expr, map: &HashMap<VarId, Expr>, atoms: &HashMap<String, Option<Sym>>) -> Expr {
    if let ExprKind::Var(v) = e.kind {
        if let Some(r) = map.get(&v) {
            return r.clone();
        }
    }
    let sym = e.sym.as_ref().and_then(|s| {
        let mut out = s.clone();
        for (name, value) in atoms {
            let atom = Atom::Param(name.clone());
            if out.atoms().contains(&atom) {
                match value {
                    Some(v) => out = out.subst(&atom, v),
                    None => return None,
                }
            }
        }
        Some(out)
    });
    let sub = |x: &Expr| subst_vars(x, map, atoms);
    let kind = match &e.kind {
        ExprKind::Index { base, indices } => ExprKind::Index {
            base: Box::new(sub(base)),
            indices: indices
                .iter()
                .map(|i| match i {
                    Index::Point(p) => Index::Point(sub(p)),
                    Index::Slice { start, end } => Index::Slice { start: start.as_ref().map(sub), end: end.as_ref().map(sub) },
                })
                .collect(),
        },
        ExprKind::Transpose(x) => ExprKind::Transpose(Box::new(sub(x))),
        ExprKind::Accessor { base, name } => ExprKind::Accessor { base: Box::new(sub(base)), name: name.clone() },
        ExprKind::Lanes { base, extent } => ExprKind::Lanes { base: Box::new(sub(base)), extent: extent.clone() },
        ExprKind::Builtin { name, args } => ExprKind::Builtin { name: *name, args: args.iter().map(sub).collect() },
        ExprKind::Intrinsic { op: name, args } => ExprKind::Intrinsic { op: *name, args: args.iter().map(sub).collect() },
        ExprKind::Call { callee, shape_args, elem_args, args } => ExprKind::Call { callee: callee.clone(), shape_args: shape_args.clone(), elem_args: elem_args.clone(), args: args.iter().map(sub).collect() },
        ExprKind::Tuple(items) => ExprKind::Tuple(items.iter().map(sub).collect()),
        ExprKind::Unary { op, expr } => ExprKind::Unary { op: *op, expr: Box::new(sub(expr)) },
        ExprKind::Cast { dtype, expr } => ExprKind::Cast { dtype: *dtype, expr: Box::new(sub(expr)) },
        ExprKind::Binary { op, lhs, rhs } => ExprKind::Binary { op: *op, lhs: Box::new(sub(lhs)), rhs: Box::new(sub(rhs)) },
        other => other.clone(),
    };
    Expr { kind, ty: e.ty.clone(), sym, span: e.span }
}
