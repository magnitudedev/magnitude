//! Composition from common-IR iteration domains, typed access maps and effects.
mod producers;
mod regions;
mod ranges;
mod panels;
pub(crate) use panels::family as matrix_panel_family;
mod reductions;
pub(crate) mod family;
pub(crate) use reductions::read_only as parameter_read_only;
mod representations;
pub(crate) use representations::{select as select_representations, decode_packet_segment, decode_segment_parameterized, prepare_packet_coefficients, prepare_packet_words};
pub(crate) use representations::{packet_aligned, packet_supported};
use crate::{
    ast::AssignOp,
    ir::*,
    lowered_ir::{Alternative, Decision, DecisionKind, LoweredIr},
    span::Span,
    sym::{Atom, Sym},
    types::{DType, Elem, Shaped, Ty},
};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

/// Invocation-owned, nonescaping roots. Admission must establish that their
/// allocations are independent of all other arguments. Stores still round.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Ownership {
    pub intermediates: BTreeSet<String>,
}
impl Ownership {
    pub fn validate(&self, f: &LoweredIr) -> Result<(), String> {
        for name in &self.intermediates {
            if !f.params.iter().any(|(n, t)| {
                n == name && matches!(t,Ty::Tensor(s) if matches!(s.elem,Elem::Dtype(_)))
            }) {
                return Err(format!(
                    "composition intermediate {name} must name a dense tensor parameter"
                ));
            }
        }
        Ok(())
    }
    fn roots(&self, f: &LoweredIr) -> BTreeSet<VarId> {
        f.vars
            .iter()
            .enumerate()
            .filter_map(|(v, d)| match d.kind {
                VarKind::Param(p) if self.intermediates.contains(&f.params[p].0) => Some(v),
                _ => None,
            })
            .collect()
    }
}

pub(crate) fn project_producers(f:&mut LoweredIr,program:&crate::program::Program,select:&mut dyn FnMut(&Decision)->Result<Alternative,String>,project_call:&mut regions::ProjectCall<'_>)->Result<(),String>{
    producers::select(f,program,select,project_call)
}

pub(crate) fn select(
    f: &mut LoweredIr,
    program: &crate::program::Program,
    ownership: &Ownership,
    select: &mut dyn FnMut(&Decision) -> Result<Alternative, String>,
    project_call: &mut regions::ProjectCall<'_>,
) -> Result<(), String> {
    ownership.validate(f)?;
    producers::select(f, program,select,project_call)?;
    let private = ownership.roots(f);
    let mut i = 0;
    while i < f.body.len() {
        let mut j = i + 1;
        while j < f.body.len() {
            let (left_domain, right_domain) = match (&f.body[i].kind, &f.body[j].kind) {
                (StmtKind::Parallel { extents: a, .. }, StmtKind::Parallel { extents: b, .. }) => {
                    (a.clone(), b.clone())
                }
                _ => {
                    j += 1;
                    continue;
                }
            };
            let mut variants = vec![(false, f.clone())];
            if left_domain.len() == right_domain.len() + 1
                && left_domain[..right_domain.len()] == right_domain
            {
                let mut phase = f.clone();
                phase.body = vec![f.body[j].clone()];
                if let Ok(p) = crate::partition::pointwise(&phase, 1) {
                    if matches!(&p.function.body[0].kind,StmtKind::Parallel{extents,..} if extents==&left_domain)
                    {
                        let mut refined = f.clone();
                        refined.vars = p.function.vars;
                        refined.body[j] = p.function.body[0].clone();
                        variants.push((true, refined));
                    }
                }
            }
            let mut alternatives = vec![Alternative::Separate];
            for (refine, candidate) in &variants {
                let StmtKind::Parallel { extents: b, .. } = &candidate.body[j].kind else {
                    unreachable!()
                };
                let shared = left_domain
                    .iter()
                    .zip(b)
                    .take_while(|(a, b)| a == b)
                    .count();
                for n in 0..=shared {
                    if join_parallel(candidate, i, j, n, &private).is_some() {
                        alternatives.push(Alternative::ParallelFusion {
                            shared_axes: n,
                            refine_consumer: *refine,
                        });
                    }
                }
            }
            if alternatives.len() == 1 {
                j += 1;
                continue;
            }
            let d = Decision {
                kind: DecisionKind::ParallelFusion {
                    boundary: i,
                    other: j,
                    left_domain,
                    right_domain,
                },
                alternatives: alternatives.into(),
            };
            match select(&d)? {
                Alternative::Separate => j += 1,
                Alternative::ParallelFusion {
                    shared_axes,
                    refine_consumer,
                } => {
                    let mut candidate = variants
                        .into_iter()
                        .find(|(refine, _)| *refine == refine_consumer)
                        .ok_or("invalid composition refinement")?
                        .1;
                    let (phase, atoms) = join_parallel(&candidate, i, j, shared_axes, &private)
                        .ok_or("selected composition lost legality")?;
                    candidate.body[i] = phase;
                    candidate.body.remove(j);
                    for v in &mut candidate.vars {
                        map_ty(&mut v.ty, &atoms)
                    }
                    *f = candidate;
                }
                _ => return Err("invalid composition assignment".into()),
            }
        }
        i += 1;
    }
    forward(&mut f.body, &mut f.vars, &private, select)?;
    let effects = Accesses::of(&f.body);
    if !effects.unknown {
        let reads = effects
            .accesses
            .iter()
            .filter(|a| !a.write)
            .map(|a| a.view.root)
            .collect::<BTreeSet<_>>();
        remove_dead_publications(&mut f.body, &private.difference(&reads).copied().collect());
    }
    select_streams(&mut f.body, &mut f.vars, select)?;
    ranges::select(&mut f.body, &mut f.vars, select)?;
    share_loads(&mut f.body, select)?;
    reductions::select(&mut f.body, &f.vars, select)?;
    producers::share(f, select)?;
    panels::select(&mut f.body, &mut f.vars, select)?;
    Ok(())
}
fn join_parallel(
    f: &LoweredIr,
    i: usize,
    j: usize,
    prefix: usize,
    private: &BTreeSet<VarId>,
) -> Option<(Stmt, Vec<(Atom, Sym)>)> {
    let (
        StmtKind::Parallel {
            vars: a,
            extents: ae,
            body: ab,
        },
        StmtKind::Parallel {
            vars: b,
            extents: be,
            body: bb,
        },
    ) = (&f.body[i].kind, &f.body[j].kind)
    else {
        return None;
    };
    if prefix > a.len().min(b.len()) || ae[..prefix] != be[..prefix] {
        return None;
    }
    for crossed in &f.body[i + 1..j] {
        if !commute(&f.body[j], crossed, private) {
            return None;
        }
    }
    let (aw, bw, ar, br) = (written(ab), written(bb), used(ab), used(bb));
    if aw.iter().any(|v| br.contains(v) && !private.contains(v))
        || bw.iter().any(|v| ar.contains(v) && !private.contains(v))
    {
        return None;
    }
    let rename = b[..prefix]
        .iter()
        .copied()
        .zip(a[..prefix].iter().copied())
        .collect::<HashMap<_, _>>();
    let atoms = index_substitutions(&rename, &f.vars);
    let mut right = serial_suffix(bb.clone(), &b[prefix..], &be[prefix..], f.body[j].span);
    for s in &mut right {
        remap(s, &rename, &atoms)
    }
    let mut body = serial_suffix(ab.clone(), &a[prefix..], &ae[prefix..], f.body[i].span);
    body.extend(right);
    let effects = Accesses::of(&body);
    if effects.unknown
        || effects
            .accesses
            .iter()
            .any(|x| x.write && !private.contains(&x.view.root))
    {
        return None;
    }
    let writes = effects
        .accesses
        .iter()
        .filter(|a| a.write)
        .map(|a| a.view.root)
        .collect::<BTreeSet<_>>();
    for access in effects
        .accesses
        .iter()
        .filter(|a| writes.contains(&a.view.root))
    {
        for (&index, extent) in a[..prefix].iter().zip(&ae[..prefix]) {
            let VarKind::Index(atom) = &f.vars[index].kind else {
                return None;
            };
            if !access
                .view
                .axes
                .iter()
                .zip(&access.view.root_shape)
                .any(|((start, n), root_n)| {
                    start == &Sym::atom(atom.clone())
                        && n.as_constant() == Some(1)
                        && root_n == extent
                })
            {
                return None;
            }
        }
    }
    Some((
        Stmt {
            id: None,
            span: f.body[i].span.to(f.body[j].span),
            kind: StmtKind::Parallel {
                vars: a[..prefix].to_vec(),
                extents: ae[..prefix].to_vec(),
                body,
            },
        },
        atoms,
    ))
}
fn serial_suffix(mut body: Vec<Stmt>, vars: &[VarId], extents: &[Sym], span: Span) -> Vec<Stmt> {
    for (&var, extent) in vars.iter().zip(extents).rev() {
        body = vec![Stmt {
            id: None,
            span,
            kind: StmtKind::Range {
                var,
                lo: Sym::constant(0),
                hi: extent.clone(),
                body,
            },
        }];
    }
    body
}
fn commute(a: &Stmt, b: &Stmt, private: &BTreeSet<VarId>) -> bool {
    let ea = Accesses::of(std::slice::from_ref(a));
    let eb = Accesses::of(std::slice::from_ref(b));
    if ea.unknown || eb.unknown {
        return false;
    }
    if ea
        .accesses
        .iter()
        .chain(&eb.accesses)
        .any(|x| x.write && !private.contains(&x.view.root))
    {
        return false;
    }
    for x in &ea.accesses {
        for y in &eb.accesses {
            if x.view.root == y.view.root && (x.write || y.write) {
                return false;
            }
        }
    }
    let (aw, bw, ar, br) = (
        written(std::slice::from_ref(a)),
        written(std::slice::from_ref(b)),
        used(std::slice::from_ref(a)),
        used(std::slice::from_ref(b)),
    );
    !aw.iter().any(|v| br.contains(v) || bw.contains(v)) && !bw.iter().any(|v| ar.contains(v))
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct View {
    pub(crate) root: VarId,
    pub(crate) root_shape: Vec<Sym>,
    pub(crate) axes: Vec<(Sym, Sym)>,
    pub(crate) visible: Vec<usize>,
}
impl View {
    pub(crate) fn of(e: &Expr) -> Option<Self> {
        match &e.kind {
            ExprKind::Var(root) if matches!(e.ty, Ty::Tensor(_) | Ty::Tile(_)) => {
                let s = e.ty.shaped()?;
                Some(Self {
                    root: *root,
                    root_shape: s.shape.clone(),
                    axes: s
                        .shape
                        .iter()
                        .map(|n| (Sym::constant(0), n.clone()))
                        .collect(),
                    visible: (0..s.shape.len()).collect(),
                })
            }
            ExprKind::Index { base, indices } => {
                let mut v = Self::of(base)?;
                if indices.len() > v.visible.len() {
                    return None;
                }
                let old = v.visible.clone();
                v.visible.clear();
                for (axis, index) in old.iter().copied().zip(indices) {
                    let (start, extent) = v.axes[axis].clone();
                    match index {
                        Index::Point(p) => {
                            v.axes[axis] = (start.add(p.sym.as_ref()?), Sym::constant(1))
                        }
                        Index::Slice { start: lo, end: hi } => {
                            let lo = match lo {
                                Some(x) => x.sym.clone()?,
                                None => Sym::constant(0),
                            };
                            let hi = match hi {
                                Some(x) => x.sym.clone()?,
                                None => extent,
                            };
                            v.axes[axis] = (start.add(&lo), hi.sub(&lo));
                            v.visible.push(axis);
                        }
                    }
                }
                v.visible.extend_from_slice(&old[indices.len()..]);
                Some(v)
            }
            ExprKind::Transpose(base) => {
                let mut v = Self::of(base)?;
                v.visible.reverse();
                Some(v)
            }
            ExprKind::Builtin {
                name: Builtin::Reshape,
                args,
            } => {
                let mut v = Self::of(args.first()?)?;
                if !v.contiguous() {
                    return None;
                }
                let shape = &e.ty.shaped()?.shape;
                let physical = v
                    .visible
                    .iter()
                    .filter(|&&a| v.axes[a].1.as_constant() != Some(1))
                    .copied()
                    .collect::<Vec<_>>();
                if shape.len() != physical.len()
                    || *shape
                        != physical
                            .iter()
                            .map(|&a| v.axes[a].1.clone())
                            .collect::<Vec<_>>()
                {
                    return None;
                }
                v.visible = physical;
                Some(v)
            }
            _ => None,
        }
    }
    fn contiguous(&self) -> bool {
        // After the first non-singleton selected axis all remaining physical
        // axes must be complete. Dropped point axes still participate here.
        let mut varying = false;
        for ((start, n), root_n) in self.axes.iter().zip(&self.root_shape) {
            if varying && (!start.is_zero() || n != root_n) {
                return false;
            }
            if n.as_constant() != Some(1) {
                varying = true;
            }
        }
        self.visible.windows(2).all(|w| w[0] < w[1])
    }
    fn interval(&self) -> Option<(VarId, Sym, Sym)> {
        if !self.contiguous() {
            return None;
        }
        let (mut offset, mut stride, mut count) =
            (Sym::constant(0), Sym::constant(1), Sym::constant(1));
        for ((start, n), size) in self.axes.iter().zip(&self.root_shape).rev() {
            offset = offset.add(&start.mul(&stride));
            stride = stride.mul(size);
            count = count.mul(n);
        }
        Some((self.root, offset, count))
    }
}
pub(crate) struct Access {
    pub(crate) view: View,
    pub(crate) write: bool,
}
#[derive(Default)]
pub(crate) struct Accesses {
    pub(crate) accesses: Vec<Access>,
    pub(crate) unknown: bool,
}
impl Accesses {
    pub(crate) fn of(body: &[Stmt]) -> Self {
        let mut a = Self::default();
        for s in body {
            a.stmt(s)
        }
        a
    }
    /// Shape evaluation observes view construction and its scalar endpoints,
    /// without reading the elements addressed by the view itself.
    fn metadata(&mut self, e: &Expr) {
        match &e.kind {
            ExprKind::Var(_) | ExprKind::TileAlloc { .. } if e.ty.shaped().is_some() => {}
            ExprKind::Index { base, indices } => {
                self.metadata(base);
                for index in indices { match index {
                    Index::Point(point) => self.expr(point, false),
                    Index::Slice { start, end } => {
                        for endpoint in start.iter().chain(end) { self.expr(endpoint, false); }
                    }
                }}
            }
            ExprKind::Transpose(base) | ExprKind::Load { view: base, .. } => self.metadata(base),
            ExprKind::Builtin { name: Builtin::Load | Builtin::Reshape, args } if !args.is_empty() => {
                self.metadata(&args[0]);
                for dimension in &args[1..] { self.expr(dimension, false); }
            }
            _ => self.unknown = true,
        }
    }
    fn expr(&mut self, e: &Expr, write: bool) {
        fn tensor_root(e: &Expr) -> bool {
            match &e.kind {
                ExprKind::Var(_) => matches!(e.ty, Ty::Tensor(_)),
                ExprKind::Index { base, .. }
                | ExprKind::Transpose(base)
                | ExprKind::Accessor { base, .. } => tensor_root(base),
                _ => matches!(e.ty, Ty::Tensor(_)),
            }
        }
        if tensor_root(e) {
            // View geometry also describes local tile values. External memory
            // effects only concern tensors; local value dependencies are tracked
            // by the same statement reads/writes used for composition legality.
            if let Some(view) = View::of(e) {
                self.accesses.push(Access { view, write });
            } else {
                self.unknown = true;
            }
            return;
        }
        match &e.kind {
            ExprKind::Builtin { name: Builtin::Extent, args } if args.len() == 2 => {
                self.metadata(&args[0]);
                self.expr(&args[1], false);
            }
            ExprKind::Call { .. } => self.unknown = true,
            ExprKind::Intrinsic { op, args } => {
                if op.writes_tensor_memory() {
                    self.unknown = true
                }
                for a in args {
                    self.expr(a, false)
                }
            }
            ExprKind::Builtin {
                name: Builtin::Store,
                args,
            } if args.len() == 2 => {
                self.expr(&args[0], false);
                self.expr(&args[1], true)
            }
            ExprKind::Builtin {
                name: Builtin::Atomic,
                ..
            } => self.unknown = true,
            _ => children(e, &mut |x| self.expr(x, false)),
        }
    }
    fn stmt(&mut self, s: &Stmt) {
        match &s.kind {
            StmtKind::Reduction(r) => {
                for e in r.operands() {
                    self.expr(e, false)
                }
                for e in &r.state {
                    self.expr(e, true)
                }
                for body in r.bodies() {
                    for s in body {
                        self.stmt(s)
                    }
                }
            }
            StmtKind::Assign { target, value, .. } => {
                self.expr(target, true);
                self.expr(value, false)
            }
            StmtKind::Expr(e) => self.expr(e, false),
            StmtKind::Parallel { body, .. }
            | StmtKind::Range { body, .. }
            | StmtKind::Lanes { body, .. } => {
                for s in body {
                    self.stmt(s)
                }
            }
            StmtKind::Owned { tile, body, .. } => {
                self.expr(tile, false);
                for s in body {
                    self.stmt(s)
                }
            }
            StmtKind::LoadLoop { domain, views, body, .. } => {
                self.expr(&domain.view,false);
                for e in views {
                    self.expr(e, false)
                }
                for s in body {
                    self.stmt(s)
                }
            }
            StmtKind::If { cond, then, els } => {
                self.expr(cond, false);
                for s in then.iter().chain(els) {
                    self.stmt(s)
                }
            }
        }
    }
}
fn remove_dead_publications(body: &mut Vec<Stmt>, dead: &BTreeSet<VarId>) {
    body.retain(|s|!matches!(&s.kind,StmtKind::Expr(Expr{kind:ExprKind::Builtin{name:Builtin::Store,args},..}) if args.len()==2&&View::of(&args[1]).is_some_and(|v|dead.contains(&v.root))));
    for s in body {
        nested_mut(s, &mut |b| remove_dead_publications(b, dead))
    }
}

fn select_streams(
    body: &mut Vec<Stmt>,
    vars: &mut Vec<Var>,
    select: &mut dyn FnMut(&Decision) -> Result<Alternative, String>,
) -> Result<(), String> {
    for s in body.iter_mut() {
        let mut error = None;
        nested_mut(s, &mut |b| {
            if error.is_none() {
                error = select_streams(b, vars, select).err()
            }
        });
        if let Some(e) = error {
            return Err(e);
        }
    }
    let mut i = 0;
    while i < body.len() {
        if !matches!(
            body[i].kind,
            StmtKind::LoadLoop { .. } | StmtKind::Range { .. }
        ) {
            i += 1;
            continue;
        }
        let Some(j) = (i + 1..body.len()).find(|&j| {
            matches!(
                body[j].kind,
                StmtKind::LoadLoop { .. } | StmtKind::Range { .. }
            )
        }) else {
            break;
        };
        if let Some((extent, replacement, atoms)) =
            join_streams(body, vars, i, j).or_else(|| join_ranges(body, vars, i, j))
        {
            let d = Decision {
                kind: DecisionKind::StreamFusion {
                    first: i,
                    second: j,
                    extent,
                },
                alternatives: vec![Alternative::Separate, Alternative::Fuse].into(),
            };
            match select(&d)? {
                Alternative::Separate => i = j,
                Alternative::Fuse => {
                    body.splice(i..=j, replacement);
                    for v in vars.iter_mut() {
                        map_ty(&mut v.ty, &atoms)
                    }
                }
                _ => return Err("invalid stream composition assignment".into()),
            }
        } else {
            i = j
        }
    }
    Ok(())
}
fn join_ranges(
    body: &[Stmt],
    vars: &[Var],
    i: usize,
    j: usize,
) -> Option<(Sym, Vec<Stmt>, Vec<(Atom, Sym)>)> {
    let (
        StmtKind::Range {
            var: a,
            lo: al,
            hi: ah,
            body: ab,
        },
        StmtKind::Range {
            var: b,
            lo: bl,
            hi: bh,
            body: bb,
        },
    ) = (&body[i].kind, &body[j].kind)
    else {
        return None;
    };
    if al != bl || ah != bh {
        return None;
    }
    let (ar, aw, br, bw) = (used(ab), written(ab), used(bb), written(bb));
    if aw.iter().any(|v| br.contains(v) || bw.contains(v)) || bw.iter().any(|v| ar.contains(v)) {
        return None;
    }
    if Accesses::of(ab).unknown
        || Accesses::of(bb).unknown
        || ab.iter().chain(bb).any(crate::effects::tensor_effect)
    {
        return None;
    }
    let mut before = Vec::new();
    let mut after = Vec::new();
    for s in &body[i + 1..j] {
        let (sr, sw) = (
            used(std::slice::from_ref(s)),
            written(std::slice::from_ref(s)),
        );
        if sw.iter().any(|v| br.contains(v)) {
            if crate::effects::tensor_effect(s)
                || sr.iter().any(|v| aw.contains(v))
                || sw.iter().any(|v| ar.contains(v) || aw.contains(v))
                || after.iter().any(|p| !independent(p, s))
            {
                return None;
            }
            before.push(s.clone());
        } else {
            if crate::effects::tensor_effect(s)
                || sr.iter().any(|v| bw.contains(v))
                || sw.iter().any(|v| br.contains(v) || bw.contains(v))
            {
                return None;
            }
            after.push(s.clone());
        }
    }
    let rename = HashMap::from([(*b, *a)]);
    let atoms = index_substitutions(&rename, vars);
    let mut right = bb.clone();
    for s in &mut right {
        remap(s, &rename, &atoms);
    }
    let mut merged = ab.clone();
    merged.extend(right);
    before.push(Stmt {
        id: None,
        span: body[i].span,
        kind: StmtKind::Range {
            var: *a,
            lo: al.clone(),
            hi: ah.clone(),
            body: merged,
        },
    });
    before.extend(after);
    Some((ah.sub(al), before, atoms))
}
fn share_loads(
    body: &mut Vec<Stmt>,
    select: &mut dyn FnMut(&Decision) -> Result<Alternative, String>,
) -> Result<(), String> {
    for s in body.iter_mut() {
        let mut error = None;
        nested_mut(s, &mut |b| {
            if error.is_none() {
                error = share_loads(b, select).err()
            }
        });
        if let Some(e) = error {
            return Err(e);
        }
    }
    let mut i = 0;
    while i < body.len() {
        let (a, view) = match &body[i].kind {
            StmtKind::Assign {
                target,
                value:
                    Expr {
                        kind:
                            ExprKind::Builtin {
                                name: Builtin::Load,
                                args,
                            },
                        ..
                    },
                op: AssignOp::Assign,
            } => match target.kind {
                ExprKind::Var(v) => (v, args[0].clone()),
                _ => {
                    i += 1;
                    continue;
                }
            },
            _ => {
                i += 1;
                continue;
            }
        };
        if body[i + 1..]
            .iter()
            .any(|s| crate::effects::tile_mutated(s, a))
        {
            i += 1;
            continue;
        }
        let mut j = i + 1;
        while j < body.len() {
            if crate::effects::tensor_effect(&body[j])
                || written(std::slice::from_ref(&body[j])).iter().any(|v|mentions(&view,*v)) {
                break;
            }
            let candidate = match &body[j].kind {
                StmtKind::Assign {
                    target,
                    value:
                        Expr {
                            kind:
                                ExprKind::Builtin {
                                    name: Builtin::Load,
                                    args,
                                },
                            ..
                        },
                    op: AssignOp::Assign,
                } if same_expr(&view, &args[0]) => match target.kind {
                    ExprKind::Var(v) => Some(v),
                    _ => None,
                },
                _ => None,
            };
            if let Some(b) = candidate {
                if !body[j + 1..]
                    .iter()
                    .any(|s| crate::effects::tile_mutated(s, b))
                {
                    let d = Decision {
                        kind: DecisionKind::Intermediate {
                            variable: b,
                            publication: j,
                        },
                        alternatives: vec![Alternative::Materialize, Alternative::RetainLocal]
                            .into(),
                    };
                    match select(&d)? {
                        Alternative::Materialize => {}
                        Alternative::RetainLocal => {
                            let rename = HashMap::from([(b, a)]);
                            for s in &mut body[j + 1..] {
                                remap(s, &rename, &[]);
                            }
                            body.remove(j);
                            continue;
                        }
                        _ => return Err("invalid shared producer choice".into()),
                    }
                }
            }
            j += 1;
        }
        i += 1;
    }
    Ok(())
}
fn join_streams(body: &[Stmt],vars:&[Var], i: usize, j: usize) -> Option<(Sym, Vec<Stmt>, Vec<(Atom, Sym)>)> {
    let (
        StmtKind::LoadLoop {
            domain: ad,
            offset:ao,
            vars: a,
            views: av,
            axes: aa,
            piece: ap,
            capacity: ac,
            body: ab,
            modes: None,
        },
        StmtKind::LoadLoop {
            domain: bd,
            offset:bo,
            vars: b,
            views: bv,
            axes: ba,
            piece: bp,
            capacity: bc,
            body: bb,
            modes: None,
        },
    ) = (&body[i].kind, &body[j].kind)
    else {
        return None;
    };
    let extent = ad.view.ty.shaped()?.shape.get(ad.axis)?.clone();
    if ac != bc || bd.view.ty.shaped()?.shape.get(bd.axis)? != &extent {
        return None;
    }
    let (ar, aw, br, bw) = (used(ab), written(ab), used(bb), written(bb));
    if aw.iter().any(|v| br.contains(v) || bw.contains(v)) || bw.iter().any(|v| ar.contains(v)) {
        return None;
    }
    if Accesses::of(ab).unknown
        || Accesses::of(bb).unknown
        || ab.iter().chain(bb).any(crate::effects::tensor_effect)
    {
        return None;
    }
    let mut before = Vec::new();
    let mut after = Vec::new();
    for s in &body[i + 1..j] {
        let sr = used(std::slice::from_ref(s));
        let sw = written(std::slice::from_ref(s));
        let needed = sw
            .iter()
            .any(|v| br.contains(v) || mentions(&bd.view,*v) || bv.iter().any(|e| mentions(e, *v)));
        if needed {
            if crate::effects::tensor_effect(s)
                || sr.iter().any(|v| aw.contains(v))
                || sw.iter().any(|v| ar.contains(v) || aw.contains(v))
                || after.iter().any(|p| !independent(p, s))
            {
                return None;
            }
            before.push(s.clone());
        } else {
            if sr.iter().any(|v| bw.contains(v))
                || sw.iter().any(|v| br.contains(v) || bw.contains(v))
                || crate::effects::tensor_effect(s)
            {
                return None;
            }
            after.push(s.clone());
        }
    }
    let mut atoms = vec![(bp.clone(), Sym::atom(ap.clone()))];
    let mut views = av.clone();
    let mut bindings = a.clone();
    let mut axes = aa.clone();
    let mut rename = HashMap::new();
    if let (Some(a),Some(b))=(ao,bo){rename.insert(*b,*a);atoms.extend(index_substitutions(&rename,vars));}
    for ((&v, view),axis) in b.iter().zip(bv).zip(ba) {
        let mut e = view.clone();
        map_expr(&mut e, &HashMap::new(), &atoms);
        if let Some(n) = views.iter().enumerate().position(|(n,x)| axes[n]==*axis && same_expr(x, &e)) {
            if ab
                .iter()
                .any(|s| crate::effects::tile_mutated(s, bindings[n]))
                || bb.iter().any(|s| crate::effects::tile_mutated(s, v))
            {
                return None;
            }
            rename.insert(v, bindings[n]);
        } else {
            bindings.push(v);
            views.push(e);
            axes.push(*axis);
        }
    }
    let mut merged = ab.clone();
    let mut right = bb.clone();
    for s in &mut right {
        remap(s, &rename, &atoms)
    }
    merged.extend(right);
    let mut loop_ = body[i].clone();
    loop_.kind = StmtKind::LoadLoop {
        domain:ad.clone(),
        offset:ao.or(*bo),
        vars: bindings,
        views,
        axes,
        piece: ap.clone(),
        capacity: *ac,
        body: merged,
        modes: None,
    };
    before.push(loop_);
    before.extend(after);
    Some((extent, before, atoms))
}
fn independent(a: &Stmt, b: &Stmt) -> bool {
    if crate::effects::tensor_effect(a) || crate::effects::tensor_effect(b) {
        return false;
    }
    let (ar, br, aw, bw) = (
        used(std::slice::from_ref(a)),
        used(std::slice::from_ref(b)),
        written(std::slice::from_ref(a)),
        written(std::slice::from_ref(b)),
    );
    !aw.iter().any(|v| br.contains(v) || bw.contains(v)) && !bw.iter().any(|v| ar.contains(v))
}
fn written(body: &[Stmt]) -> HashSet<VarId> {
    let mut v = HashSet::new();
    for s in body {
        crate::rewrite::writes(s, &mut v)
    }
    v
}
fn used(body: &[Stmt]) -> HashSet<VarId> {
    let mut v = HashSet::new();
    for s in body {
        visit_stmt(s, &mut |e| {
            walk(e, &mut |e| {
                if let ExprKind::Var(i) = e.kind {
                    v.insert(i);
                }
            })
        });
    }
    v
}
fn mentions(e: &Expr, v: VarId) -> bool {
    let mut found = false;
    walk(e, &mut |e| {
        found |= matches!(e.kind,ExprKind::Var(x) if x==v)
    });
    found
}
fn same_expr(a: &Expr, b: &Expr) -> bool {
    if let (Some(a), Some(b)) = (View::of(a), View::of(b)) {
        return a == b;
    }
    let (mut a, mut b) = (a.clone(), b.clone());
    erase_spans(&mut a);
    erase_spans(&mut b);
    a == b
}
fn erase_spans(e: &mut Expr) {
    e.span = Span::default();
    children_mut(e, &mut erase_spans);
}

/// Reaching stores define distinct versions. The local snapshot retains the
/// original rounding; every consumer receives an independent value copy.
fn forward(
    body: &mut Vec<Stmt>,
    vars: &mut Vec<Var>,
    private: &BTreeSet<VarId>,
    select: &mut dyn FnMut(&Decision) -> Result<Alternative, String>,
) -> Result<(), String> {
    for s in body.iter_mut() {
        let mut error = None;
        nested_mut(s, &mut |b| {
            if error.is_none() {
                error = forward(b, vars, private, select).err()
            }
        });
        if let Some(error) = error {
            return Err(error);
        }
    }
    let mut reaching: BTreeMap<(VarId, Sym, Sym), usize> = BTreeMap::new();
    let mut consumers: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for (i, s) in body.iter().enumerate() {
        if let StmtKind::Assign {
            op: AssignOp::Assign,
            value,
            ..
        } = &s.kind
        {
            if let ExprKind::Builtin {
                name: Builtin::Load,
                args,
            } = &value.kind
            {
                if let Some(key) = args.first().and_then(View::of).and_then(|v| v.interval()) {
                    if let Some(&store) = reaching.get(&key) {
                        consumers.entry(store).or_default().push(i);
                    }
                }
            }
        }
        let effects = Accesses::of(std::slice::from_ref(s));
        if effects.unknown {
            reaching.clear();
            continue;
        }
        for root in effects
            .accesses
            .iter()
            .filter(|a| a.write)
            .map(|a| a.view.root)
        {
            reaching.retain(|key, _| key.0 != root);
        }
        if let StmtKind::Expr(Expr {
            kind:
                ExprKind::Builtin {
                    name: Builtin::Store,
                    args,
                },
            ..
        }) = &s.kind
        {
            if args.len() == 2 && matches!(args[0].ty, Ty::Tile(_)) {
                if let Some(key) = View::of(&args[1]).and_then(|v| v.interval()) {
                    if private.contains(&key.0) {
                        reaching.insert(key, i);
                    }
                }
            }
        }
    }
    let mut replacements = BTreeMap::new();
    for (at, uses) in consumers {
        let StmtKind::Expr(Expr {
            kind: ExprKind::Builtin { args, .. },
            ..
        }) = &body[at].kind
        else {
            unreachable!()
        };
        let (source, target) = (&args[0], &args[1]);
        let Some(Elem::Dtype(dtype)) = target.ty.shaped().map(|s| &s.elem) else {
            continue;
        };
        let source_shape = source.ty.shaped().unwrap().shape.clone();
        if source_shape.iter().any(|n| n.as_constant().is_none())
            || uses.iter().any(|&c| match &body[c].kind {
                StmtKind::Assign { target, .. } => target
                    .ty
                    .shaped()
                    .is_none_or(|s| s.shape.iter().any(|n| n.as_constant().is_none())),
                _ => true,
            })
        {
            continue;
        }
        let root = View::of(target).ok_or("missing intermediate root")?.root;
        let decision = Decision {
            kind: DecisionKind::Intermediate {
                variable: root,
                publication: at,
            },
            alternatives: vec![Alternative::Materialize, Alternative::RetainLocal].into(),
        };
        match select(&decision)? {
            Alternative::Materialize => continue,
            Alternative::RetainLocal => {}
            _ => return Err("invalid intermediate retention assignment".into()),
        }
        let snapshot = fresh(
            vars,
            "snapshot",
            Ty::Tile(Shaped::new(source_shape, Elem::Dtype(*dtype))),
            body[at].span,
        );
        let snap = variable(snapshot, vars);
        let mut define = copy(source, &snap, vars);
        define.push(body[at].clone());
        replacements.insert(at, define);
        for c in uses {
            let StmtKind::Assign { target, .. } = &body[c].kind else {
                unreachable!()
            };
            replacements.insert(c, copy(&snap, target, vars));
        }
    }
    if !replacements.is_empty() {
        *body = body
            .iter()
            .enumerate()
            .flat_map(|(i, s)| replacements.remove(&i).unwrap_or_else(|| vec![s.clone()]))
            .collect();
    }
    Ok(())
}
fn copy(source: &Expr, target: &Expr, vars: &mut Vec<Var>) -> Vec<Stmt> {
    let shape = target.ty.shaped().unwrap().shape.clone();
    let elem = target.ty.shaped().unwrap().elem.clone();
    let Elem::Dtype(dtype) = elem else {
        unreachable!()
    };
    let span = target.span;
    let allocation = Stmt {
        id: None,
        span,
        kind: StmtKind::Assign {
            target: target.clone(),
            op: AssignOp::Assign,
            value: Expr {
                kind: ExprKind::TileAlloc {
                    shape: shape.clone(),
                    dtype: elem,
                },
                ty: target.ty.clone(),
                sym: None,
                span,
            },
        },
    };
    let indices = shape
        .iter()
        .map(|_| {
            let v = vars.len();
            vars.push(Var {
                name: "coordinate".into(),
                ty: Ty::Scalar(DType::I32),
                span,
                kind: VarKind::Index(Atom::Param(format!("composition#{v}"))),
            });
            v
        })
        .collect::<Vec<_>>();
    let mut linear = Sym::constant(0);
    for (&v, n) in indices.iter().zip(&shape) {
        let VarKind::Index(a) = &vars[v].kind else {
            unreachable!()
        };
        linear = linear.mul(n).add(&Sym::atom(a.clone()));
    }
    let mut remainder = linear;
    let mut source_indices = Vec::new();
    for n in source.ty.shaped().unwrap().shape.iter().rev() {
        source_indices.push(Index::Point(symbol(remainder.rem(n), span)));
        remainder = remainder.quot(n);
    }
    source_indices.reverse();
    let value = Expr {
        kind: ExprKind::Index {
            base: Box::new(source.clone()),
            indices: source_indices,
        },
        ty: Ty::Scalar(source.ty.shaped().unwrap().elem.read_dtype().unwrap()),
        sym: None,
        span,
    };
    let dst = Expr {
        kind: ExprKind::Index {
            base: Box::new(target.clone()),
            indices: indices
                .iter()
                .map(|&v| Index::Point(variable(v, vars)))
                .collect(),
        },
        ty: Ty::Scalar(dtype),
        sym: None,
        span,
    };
    vec![
        allocation,
        Stmt {
            id: None,
            span,
            kind: StmtKind::Owned {
                vars: indices,
                tile: target.clone(),
                body: vec![Stmt {
                    id: None,
                    span,
                    kind: StmtKind::Assign {
                        target: dst,
                        op: AssignOp::Assign,
                        value: Expr {
                            kind: ExprKind::Cast {
                                dtype,
                                expr: Box::new(value),
                            },
                            ty: Ty::Scalar(dtype),
                            sym: None,
                            span,
                        },
                    },
                }],
            },
        },
    ]
}
fn fresh(vars: &mut Vec<Var>, name: &str, ty: Ty, span: Span) -> VarId {
    let v = vars.len();
    vars.push(Var {
        name: format!("{name}#{v}"),
        ty,
        span,
        kind: VarKind::Local,
    });
    v
}
fn variable(v: VarId, vars: &[Var]) -> Expr {
    Expr {
        kind: ExprKind::Var(v),
        ty: vars[v].ty.clone(),
        sym: match &vars[v].kind {
            VarKind::Index(a) => Some(Sym::atom(a.clone())),
            _ => None,
        },
        span: vars[v].span,
    }
}
fn symbol(sym: Sym, span: Span) -> Expr {
    Expr {
        kind: ExprKind::ShapeParam(sym.to_string()),
        ty: Ty::Scalar(DType::I32),
        sym: Some(sym),
        span,
    }
}

fn index_substitutions(rename: &HashMap<VarId, VarId>, vars: &[Var]) -> Vec<(Atom, Sym)> {
    rename
        .iter()
        .filter_map(|(a, b)| match (&vars[*a].kind, &vars[*b].kind) {
            (VarKind::Index(a), VarKind::Index(b)) => Some((a.clone(), Sym::atom(b.clone()))),
            _ => None,
        })
        .collect()
}
fn map_sym(s: &mut Sym, atoms: &[(Atom, Sym)]) {
    let parameters = atoms.iter().filter_map(|(atom, value)| match atom {
        Atom::Param(name) => Some((name.clone(), value.clone())), _ => None,
    }).collect::<HashMap<_, _>>();
    *s = crate::lower::subst_sym(s, &parameters, &HashMap::new());
    for (a, v) in atoms {
        if !matches!(a, Atom::Param(_)) { *s = s.subst(a, v); }
    }
}
fn map_ty(t: &mut Ty, atoms: &[(Atom, Sym)]) {
    match t {
        Ty::Tensor(s) | Ty::Tile(s) | Ty::Frag(s) => {
            for n in &mut s.shape {
                map_sym(n, atoms)
            }
        }
        Ty::Tuple(t) => {
            for t in t {
                map_ty(t, atoms)
            }
        }
        _ => {}
    }
}
fn map_expr(e: &mut Expr, rename: &HashMap<VarId, VarId>, atoms: &[(Atom, Sym)]) {
    map_ty(&mut e.ty, atoms);
    if let Some(s) = &mut e.sym {
        map_sym(s, atoms)
    }
    match &mut e.kind {
        ExprKind::Var(v) => {
            if let Some(n) = rename.get(v) {
                *v = *n
            }
        }
        ExprKind::TileAlloc { shape, .. } => {
            for n in shape {
                map_sym(n, atoms)
            }
        }
        ExprKind::Lanes { extent, .. } => map_sym(extent, atoms),
        ExprKind::Call { shape_args, .. } => {
            for n in shape_args {
                map_sym(n, atoms)
            }
        }
        _ => {}
    }
    children_mut(e, &mut |e| map_expr(e, rename, atoms));
}
pub(crate) fn remap(s: &mut Stmt, rename: &HashMap<VarId, VarId>, atoms: &[(Atom, Sym)]) {
    if let StmtKind::Reduction(r) = &mut s.kind {
        if let Some(call) = r.merge.source_mut() { map_expr(call, rename, atoms); }
        if let Some(step) = &mut r.step {
            if let Some(call) = step.call.source_mut() { map_expr(call, rename, atoms); }
        }
        for m in r.implementations_mut() {
            for e in m.left.iter_mut().chain(&mut m.right).chain(&mut m.output) {
                map_expr(e, rename, atoms)
            }
        }
    }
    direct_exprs_mut(s, &mut |e| map_expr(e, rename, atoms));
    match &mut s.kind {
        StmtKind::Range { var, lo, hi, .. } => {
            if let Some(mapped) = rename.get(var) { *var = *mapped; }
            map_sym(lo, atoms);
            map_sym(hi, atoms)
        }
        StmtKind::Parallel { vars, extents, .. } => {
            for variable in vars { if let Some(mapped) = rename.get(variable) { *variable = *mapped; } }
            for n in extents {
                map_sym(n, atoms)
            }
        }
        StmtKind::Lanes { var, extent, .. } => { if let Some(mapped) = rename.get(var) { *var = *mapped; } map_sym(extent, atoms); },
        StmtKind::Owned { vars, .. } | StmtKind::LoadLoop { vars, .. } => {
            for variable in vars { if let Some(mapped) = rename.get(variable) { *variable = *mapped; } }
            if let StmtKind::LoadLoop { offset: Some(offset), .. } = &mut s.kind {
                if let Some(mapped) = rename.get(offset) { *offset = *mapped; }
            }
        },
        _ => {}
    }
    nested_mut(s, &mut |b| {
        for s in b {
            remap(s, rename, atoms)
        }
    });
}
fn children(e: &Expr, f: &mut impl FnMut(&Expr)) {
    match &e.kind {
        ExprKind::Index { base, indices } => {
            f(base);
            for i in indices {
                match i {
                    Index::Point(e) => f(e),
                    Index::Slice { start, end } => {
                        for e in start.iter().chain(end) {
                            f(e)
                        }
                    }
                }
            }
        }
        ExprKind::Load { view: e, .. }
        | ExprKind::Transpose(e)
        | ExprKind::Accessor { base: e, .. }
        | ExprKind::Lanes { base: e, .. }
        | ExprKind::Unary { expr: e, .. }
        | ExprKind::Cast { expr: e, .. } => f(e),
        ExprKind::Builtin { args, .. }
        | ExprKind::Call { args, .. }
        | ExprKind::Intrinsic { args, .. }
        | ExprKind::Tuple(args) => {
            for e in args {
                f(e)
            }
        }
        ExprKind::Binary { lhs, rhs, .. } => {
            f(lhs);
            f(rhs)
        }
        _ => {}
    }
}
fn children_mut(e: &mut Expr, f: &mut impl FnMut(&mut Expr)) {
    match &mut e.kind {
        ExprKind::Index { base, indices } => {
            f(base);
            for i in indices {
                match i {
                    Index::Point(e) => f(e),
                    Index::Slice { start, end } => {
                        for e in start.iter_mut().chain(end) {
                            f(e)
                        }
                    }
                }
            }
        }
        ExprKind::Load { view: e, .. }
        | ExprKind::Transpose(e)
        | ExprKind::Accessor { base: e, .. }
        | ExprKind::Lanes { base: e, .. }
        | ExprKind::Unary { expr: e, .. }
        | ExprKind::Cast { expr: e, .. } => f(e),
        ExprKind::Builtin { args, .. }
        | ExprKind::Call { args, .. }
        | ExprKind::Intrinsic { args, .. }
        | ExprKind::Tuple(args) => {
            for e in args {
                f(e)
            }
        }
        ExprKind::Binary { lhs, rhs, .. } => {
            f(lhs);
            f(rhs)
        }
        _ => {}
    }
}
fn walk(e: &Expr, f: &mut impl FnMut(&Expr)) {
    f(e);
    children(e, &mut |e| walk(e, f));
}
fn visit_stmt(s: &Stmt, f: &mut impl FnMut(&Expr)) {
    match &s.kind {
        StmtKind::Reduction(r) => {
            for e in r.operands() {
                f(e)
            }
            for body in r.bodies() {
                for s in body {
                    visit_stmt(s, f)
                }
            }
        }
        StmtKind::Assign { target, value, .. } => {
            f(target);
            f(value)
        }
        StmtKind::Expr(e) => f(e),
        StmtKind::Owned { tile, .. } => f(tile),
        StmtKind::LoadLoop { domain, views, .. } => {
            f(&domain.view);
            for e in views {
                f(e)
            }
        }
        StmtKind::If { cond, .. } => f(cond),
        _ => {}
    }
    match &s.kind {
        StmtKind::Parallel { body, .. }
        | StmtKind::Owned { body, .. }
        | StmtKind::Range { body, .. }
        | StmtKind::Lanes { body, .. }
        | StmtKind::LoadLoop { body, .. } => {
            for s in body {
                visit_stmt(s, f)
            }
        }
        StmtKind::If { then, els, .. } => {
            for s in then.iter().chain(els) {
                visit_stmt(s, f)
            }
        }
        _ => {}
    }
}
fn direct_exprs_mut(s: &mut Stmt, f: &mut impl FnMut(&mut Expr)) {
    match &mut s.kind {
        StmtKind::Reduction(r) => {
            for e in r.operands_mut() {
                f(e)
            }
        }
        StmtKind::Assign { target, value, .. } => {
            f(target);
            f(value)
        }
        StmtKind::Expr(e) => f(e),
        StmtKind::Owned { tile, .. } => f(tile),
        StmtKind::LoadLoop { domain, views, .. } => {
            f(&mut domain.view);
            for e in views {
                f(e)
            }
        }
        StmtKind::If { cond, .. } => f(cond),
        _ => {}
    }
}
fn nested_mut(s: &mut Stmt, f: &mut impl FnMut(&mut Vec<Stmt>)) {
    match &mut s.kind {
        StmtKind::Reduction(r) => {
            for m in r.implementations_mut() {
                f(&mut m.body)
            }
        }
        StmtKind::Parallel { body, .. }
        | StmtKind::Owned { body, .. }
        | StmtKind::Range { body, .. }
        | StmtKind::Lanes { body, .. }
        | StmtKind::LoadLoop { body, .. } => f(body),
        StmtKind::If { then, els, .. } => {
            f(then);
            f(els)
        }
        _ => {}
    }
}

/// Substitute immutable local values through the existing typed expression and
/// statement walkers. The replacement contains no binder identities to rename.
pub(crate) fn substitute_values(body: &mut [Stmt], values: &HashMap<VarId, Expr>) {
    if values.is_empty() { return; }
    for statement in body {
        direct_exprs_mut(statement, &mut |e| *e = crate::lower::subst_vars(e, values, &HashMap::new()));
        if let StmtKind::Reduction(r) = &mut statement.kind {
            if let Some(call) = r.merge.source_mut() { *call = crate::lower::subst_vars(call, values, &HashMap::new()); }
            if let Some(step) = &mut r.step {
                if let Some(call) = step.call.source_mut() { *call = crate::lower::subst_vars(call, values, &HashMap::new()); }
            }
        }
        nested_mut(statement, &mut |body| substitute_values(body, values));
    }
}
