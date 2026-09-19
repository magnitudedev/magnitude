//! Regions under the selected widths: root parallel regions are launches, every other
//! traversal is an ordered loop over pieces inside its owner; region results are local
//! tiles with leading piece axes; merges are the canonical adjacent-pair recurrence.
use super::context::*;
use super::stmt::Item;
use crate::exec::ir::{self, Expr, ExprKind, Stmt, StmtKind};
use crate::exec::types::{Shaped, Ty};
use crate::span::Span;
use crate::sir;
use crate::syntax::ast::RegionMode;
use crate::types as st;
use crate::sym::{Atom, Sym};
use crate::types::Elem;

/// Partition of one binder for the current visit of the enclosing scope.
#[derive(Clone, Debug)]
struct Geometry {
    slice: st::SliceId,
    var: sir::VarId,
    lo: Sym,
    width: i64,
    /// Piece count; symbolic only for width one over a runtime extent.
    count: Sym,
    site: Option<crate::family::SiteId>,
}

impl Geometry {
    fn same(&self, other: &Geometry) -> bool {
        self.lo == other.lo && self.width == other.width && self.count == other.count
    }
}

/// Index variables of one traversal.
struct Traversal {
    vars: Vec<ir::VarId>,
    atoms: Vec<Atom>,
    ordinals: Vec<Expr>,
}

impl<'a> Instantiation<'a> {
    fn geometry(&mut self, f: &mut Frame<'a>, region: &'a sir::Region, source: Option<&ResultValue>) -> Result<Vec<Geometry>, String> {
        let definition: &'a sir::Definition = f.definition;
        let name = &definition.name;
        if let Some(result) = source {
            if result.binders.len() != region.binders.len() {
                return Err(format!("region#{} of `{name}` rebinds {} binders of a {}-binder result", region.id.0, region.binders.len(), result.binders.len()));
            }
        }
        let mut out = Vec::with_capacity(region.binders.len());
        for (k, var) in region.binders.iter().enumerate() {
            let sir::VarKind::Slice(slice) = f.declared(*var)?.kind else {
                return Err(format!("binder `{}` of region#{} in `{name}` is not a slice", f.name(*var), region.id.0));
            };
            if let Some(result) = source {
                let b = &result.binders[k];
                out.push(Geometry { slice, var: *var, lo: b.lo.clone(), width: b.width, count: Sym::constant(b.count), site: b.site });
                continue;
            }
            let decl = f.body.slices.get(slice.0 as usize).ok_or_else(|| format!("slice#{} is outside the body of `{name}`", slice.0))?;
            let (lo, extent) = match &decl.parent {
                sir::SliceParent::Domain { lo, hi } => (self.resolve(f, lo)?, self.resolve(f, &hi.sub(lo))?),
                sir::SliceParent::Refine(parent) => {
                    let parent = f.slice(*parent)?;
                    (parent.lo.clone(), Sym::constant(parent.capacity))
                }
                sir::SliceParent::Rebind(_) => return Err(format!("binder `{}` of region#{} in `{name}` rebinds a result its region does not traverse", f.name(*var), region.id.0)),
            };
            let (site, parts, value) = self.site(f, region.id, slice)?;
            let (width, count) = match extent.as_constant() {
                Some(extent) if extent < 0 => return Err(format!("domain of `{}` in `{name}` has the negative extent {extent}", f.name(*var))),
                Some(extent) => {
                    let width = width_of(extent, parts, value).map_err(|e| format!("`{}` in `{name}`: {e}", f.name(*var)))?;
                    (width, Sym::constant(if width == 0 { 0 } else { extent / width }))
                }
                // A runtime extent has no tail only under width one.
                None if !parts && value == 1 => (1, extent),
                None => return Err(format!("tail pieces of the runtime extent `{extent}` of `{}` in `{name}` are not supported by instantiation yet", f.name(*var))),
            };
            out.push(Geometry { slice, var: *var, lo, width, count, site: Some(site) });
        }
        Ok(out)
    }

    fn traversal(&mut self, f: &Frame<'a>, geometry: &[Geometry], span: Span) -> Traversal {
        let mut t = Traversal { vars: Vec::new(), atoms: Vec::new(), ordinals: Vec::new() };
        for g in geometry {
            let (id, atom, e) = self.index(f.name(g.var), span);
            t.vars.push(id);
            t.atoms.push(atom);
            t.ordinals.push(e);
        }
        t
    }

    /// Bind every binder to its piece of the traversal; returns the displaced live sites.
    fn bind_slices(&mut self, f: &mut Frame<'a>, geometry: &[Geometry], t: &Traversal) -> Vec<(crate::family::SiteId, Option<Slice>)> {
        let mut displaced = Vec::new();
        for (g, atom) in geometry.iter().zip(&t.atoms) {
            // The only piece of a one-piece traversal is piece zero.
            let ordinal = if g.count.as_constant() == Some(1) { Sym::constant(0) } else { Sym::atom(atom.clone()) };
            let lo = g.lo.add(&ordinal.scale(g.width));
            let slice = Slice { hi: lo.add(&Sym::constant(g.width)), lo, capacity: g.width, ordinal };
            if let Some(site) = g.site {
                displaced.push((site, self.live_sites.insert(site, slice.clone())));
            }
            f.slices.insert(g.slice, slice.clone());
            f.vars[g.var] = Some(Value::Slice(slice));
        }
        displaced
    }

    fn restore_sites(&mut self, displaced: Vec<(crate::family::SiteId, Option<Slice>)>) {
        for (site, previous) in displaced.into_iter().rev() {
            match previous {
                Some(slice) => self.live_sites.insert(site, slice),
                None => self.live_sites.remove(&site),
            };
        }
    }

    /// Ordered loops over pieces, first binder outermost (lexicographic visit order).
    fn loops(geometry: &[Geometry], t: &Traversal, mut body: Vec<Stmt>, span: Span) -> Vec<Stmt> {
        for (g, var) in geometry.iter().zip(&t.vars).rev() {
            body = vec![stmt(StmtKind::Range { var: *var, lo: Sym::constant(0), hi: g.count.clone(), body }, span)];
        }
        body
    }

    fn launch(&self, f: &Frame<'a>, geometry: &[Geometry], t: &Traversal, body: Vec<Stmt>, span: Span) -> Result<Stmt, String> {
        if let Some(g) = geometry.iter().find(|g| g.count.as_constant().is_none()) {
            return Err(format!("root parallel binder `{}` of `{}` has the runtime piece count `{}`; a launch needs a static work domain", f.name(g.var), f.definition.name, g.count));
        }
        Ok(stmt(StmtKind::Parallel { vars: t.vars.clone(), extents: geometry.iter().map(|g| g.count.clone()).collect(), body }, span))
    }

    pub fn region(&mut self, f: &mut Frame<'a>, region: &'a sir::Region, root: bool, out: &mut Vec<Stmt>) -> Result<Value<'a>, String> {
        let span = region.body.first().map_or(Span::default(), |s| s.span);
        let source = match &region.source {
            sir::RegionSource::Domains => None,
            sir::RegionSource::Results(e) => match self.value(f, e, out)? {
                Value::Result(result) => Some(result),
                _ => return Err(format!("region#{} of `{}` traverses a value that is not a region result", region.id.0, f.definition.name)),
            },
        };
        let geometry = self.geometry(f, region, source.as_ref())?;
        let launch = root && region.mode == RegionMode::Parallel;
        if launch && (region.merge.is_some() || region.result.is_some()) {
            return Err(format!(
                "region#{} of `{}` is a root-level parallel region whose {} cross launches; the execution IR has no storage that parallel work items of one launch write and a later launch reads",
                region.id.0,
                f.definition.name,
                if region.merge.is_some() { "merge partials" } else { "region results" }
            ));
        }
        if region.mode == RegionMode::Pipeline && (region.merge.is_some() || region.result.is_some()) {
            return Err(format!("pipeline region#{} of `{}` is used as a result expression", region.id.0, f.definition.name));
        }
        let t = self.traversal(f, &geometry, span);
        let displaced = self.bind_slices(f, &geometry, &t);
        let result = self.region_body(f, region, &geometry, &t, launch, span, out);
        self.restore_sites(displaced);
        result
    }

    #[allow(clippy::too_many_arguments)]
    fn region_body(&mut self, f: &mut Frame<'a>, region: &'a sir::Region, geometry: &[Geometry], t: &Traversal, launch: bool, span: Span, out: &mut Vec<Stmt>) -> Result<Value<'a>, String> {
        let definition: &'a sir::Definition = f.definition;
        let name = &definition.name;
        let counts = || -> Result<Vec<i64>, String> {
            geometry.iter().map(|g| g.count.as_constant().ok_or_else(|| format!("region#{} of `{name}` keeps one value per piece of a runtime extent, which needs static storage", region.id.0))).collect()
        };
        if let Some(merge) = &region.merge {
            let counts = counts()?;
            let parts: i64 = counts.iter().product();
            if parts == 0 {
                // An empty domain has no visits: the merged value is the identity.
                return self.value(f, &merge.identity, out);
            }
            // Partials in lexicographic piece order behind one flat leading axis.
            let member = self.member_storage(f, &merge.identity.ty, &[parts], span, out)?;
            let mut flat = Sym::constant(0);
            for (atom, count) in t.atoms.iter().zip(&counts) {
                flat = flat.scale(*count).add(&Sym::atom(atom.clone()));
            }
            f.yields.push(YieldSink::Result { member, pieces: vec![symbol(flat, span)], loops: f.loops });
            let body = self.block(f, &region.body, false);
            let Some(YieldSink::Result { member, .. }) = f.yields.pop() else {
                return Err(format!("region#{} of `{name}` lost its yield boundary", region.id.0));
            };
            out.extend(Self::loops(geometry, t, body?, span));
            return self.combine(f, merge, member, parts, span, out);
        }
        if let Some(st::Ty::Result(ty)) = &region.result {
            let counts = counts()?;
            let member = self.member_storage(f, &ty.member, &counts, span, out)?;
            f.yields.push(YieldSink::Result { member, pieces: t.ordinals.clone(), loops: f.loops });
            let body = self.block(f, &region.body, false);
            let Some(YieldSink::Result { member, .. }) = f.yields.pop() else {
                return Err(format!("region#{} of `{name}` lost its yield boundary", region.id.0));
            };
            out.extend(Self::loops(geometry, t, body?, span));
            let binders = geometry.iter().zip(&counts).map(|(g, count)| Binder { lo: g.lo.clone(), width: g.width, count: *count, site: g.site }).collect();
            return Ok(Value::Result(ResultValue { binders, atoms: t.atoms.clone(), member }));
        }
        if region.result.is_some() {
            return Err(format!("region#{} of `{name}` has a result type that is not a region result", region.id.0));
        }
        let owners = !launch && self.inner_owners(f, region, geometry);
        // The owner body of a launch admits inner owner regions; an inner owner body does not.
        let enclosing = match (launch, owners) {
            (true, _) => std::mem::replace(&mut self.launch_owner, Some(LaunchOwner { candidate: f.candidate, binders: geometry.iter().map(|g| g.slice).collect() })),
            (false, true) => self.launch_owner.take(),
            (false, false) => self.launch_owner.clone(),
        };
        let body = self.block(f, &region.body, false);
        self.launch_owner = enclosing;
        let body = body?;
        if launch {
            out.push(self.launch(f, geometry, t, body, span)?);
        } else if owners {
            // Inner owners of the enclosing launch piece: independent visits with static counts.
            out.push(stmt(StmtKind::Parallel { vars: t.vars.clone(), extents: geometry.iter().map(|g| g.count.clone()).collect(), body }, span));
        } else {
            out.extend(Self::loops(geometry, t, body, span));
        }
        Ok(Value::Void)
    }

    /// Whether `region` is an inner owner region of the launch being instantiated: a
    /// statement-position `parallel` region without merge or result, in the owner body of a
    /// root `parallel` launch of the same candidate (outside element loops and branches),
    /// every binder refining a distinct binder of that launch with a static piece count, on
    /// a target whose mapping gives inner owners participants of their own
    /// (`owner_regions`). Every other nested region is ordered loops over its pieces.
    fn inner_owners(&self, f: &Frame<'a>, region: &sir::Region, geometry: &[Geometry]) -> bool {
        let Some(owner) = self.launch_owner.as_ref().filter(|owner| owner.candidate == f.candidate) else { return false };
        super::owner_regions(&self.family.target) && super::inner_owner_region(f.body, &owner.binders, region) && geometry.iter().all(|g| g.count.as_constant().is_some())
    }

    /// Root parallel regions of one selected interval with identical binder geometry:
    /// one launch running every body in authored order per work item.
    pub fn fused_regions(&mut self, f: &mut Frame<'a>, members: &[Item<'a>], root: bool, out: &mut Vec<Stmt>) -> Result<(), String> {
        let mut regions = Vec::new();
        for member in members {
            match &member.stmt.kind {
                sir::StmtKind::Bind { pattern: sir::Pattern::Var(v), value } if f.folded.contains(v) => f.vars[*v] = Some(Value::Deferred(value)),
                sir::StmtKind::Region(r) if root && r.mode == RegionMode::Parallel && r.merge.is_none() && r.result.is_none() && r.source == sir::RegionSource::Domains => regions.push(r),
                _ => return Err(format!("a fused region interval of `{}` groups statements other than root-level parallel statement regions over domains", f.definition.name)),
            }
        }
        let Some(first) = regions.first() else { return Ok(()) };
        let span = first.body.first().map_or(Span::default(), |s| s.span);
        let shared = self.geometry(f, first, None)?;
        let t = self.traversal(f, &shared, span);
        let mut body = Vec::new();
        for region in &regions {
            let geometry = self.geometry(f, region, None)?;
            if geometry.len() != shared.len() || geometry.iter().zip(&shared).any(|(a, b)| !a.same(b)) {
                return Err(format!("the fused regions #{} and #{} of `{}` have different binder geometry under the selected widths", first.id.0, region.id.0, f.definition.name));
            }
            let displaced = self.bind_slices(f, &geometry, &t);
            let stmts = self.block(f, &region.body, false);
            self.restore_sites(displaced);
            body.extend(stmts?);
        }
        out.push(self.launch(f, &shared, &t, body, span)?);
        Ok(())
    }

    // ---- region results ----

    /// Storage of one member schema behind the given leading piece axes.
    fn member_storage(&mut self, f: &Frame<'a>, ty: &st::Ty, leading: &[i64], span: Span, out: &mut Vec<Stmt>) -> Result<Member, String> {
        let axes = || leading.iter().map(|n| Sym::constant(*n));
        let allocate = |this: &mut Self, shaped: Shaped, out: &mut Vec<Stmt>| {
            let tile = this.allocate("results", shaped, span, out);
            if let ExprKind::Var(id) = tile.kind {
                this.temporaries.remove(&id);
                this.mutable.insert(id);
            }
            tile
        };
        match ty {
            st::Ty::Scalar(_) | st::Ty::Index(_) => {
                let dtype = self.dtype(f, ty)?;
                Ok(Member::Scalar(allocate(self, Shaped::new(axes().collect(), Elem::Dtype(dtype)), out)))
            }
            st::Ty::Tile(shaped) => {
                let shaped = self.shaped(f, shaped)?;
                if !matches!(shaped.elem, Elem::Dtype(_)) {
                    return Err(format!("a region result of `{}` yields packed tiles, which local storage cannot hold", f.definition.name));
                }
                Ok(Member::Tile(allocate(self, Shaped::new(axes().chain(shaped.shape).collect(), shaped.elem), out)))
            }
            st::Ty::Tuple(items) => Ok(Member::Tuple(items.iter().map(|t| self.member_storage(f, t, leading, span, out)).collect::<Result<_, _>>()?)),
            st::Ty::Result(inner) => {
                // Nested results: the inner producer's piece axes follow the outer ones.
                let mut binders = Vec::with_capacity(inner.binders.len());
                let mut all = leading.to_vec();
                for slice in &inner.binders {
                    let (width, count) = self.static_geometry(f, *slice)?;
                    all.push(count);
                    binders.push(Binder { lo: Sym::constant(0), width, count, site: None });
                }
                let member = self.member_storage(f, &inner.member, &all, span, out)?;
                Ok(Member::Result(Box::new(ResultValue { binders, atoms: Vec::new(), member })))
            }
            other => Err(format!("a region result of `{}` yields `{other}`, which has no surviving local storage", f.definition.name)),
        }
    }

    /// Write one yielded value at the given pieces.
    pub fn store_member(&mut self, member: &mut Member, pieces: &[Expr], value: Value<'a>, out: &mut Vec<Stmt>) -> Result<(), String> {
        match (member, value) {
            (Member::Scalar(store), Value::Scalar(e)) => {
                let place = points(store.clone(), pieces)?;
                let Ty::Scalar(dtype) = place.ty else { return Err("region-result storage lost its piece axes".into()) };
                out.push(assign(place, conform(e, dtype)));
                Ok(())
            }
            (Member::Tile(store), Value::Shaped(e)) => self.store_tile(store, pieces, e, out),
            (Member::Tuple(members), Value::Tuple(values)) if members.len() == values.len() => {
                members.iter_mut().zip(values).try_for_each(|(m, v)| self.store_member(m, pieces, v, out))
            }
            (Member::Result(slot), Value::Result(inner)) => {
                if slot.binders.len() != inner.binders.len() || slot.binders.iter().zip(&inner.binders).any(|(a, b)| a.width != b.width || a.count != b.count) {
                    return Err("a yielded nested region result differs from the partition its storage was allocated for".into());
                }
                // The storage takes the geometry of the producer that filled it.
                slot.binders = inner.binders.clone();
                slot.atoms = inner.atoms.clone();
                self.store_nested(&slot.member, pieces, &inner.member, out)
            }
            _ => Err("a yielded value differs from the member schema of its region result".into()),
        }
    }

    fn store_tile(&mut self, store: &Expr, pieces: &[Expr], source: Expr, out: &mut Vec<Stmt>) -> Result<(), String> {
        let source = if matches!(source.ty, Ty::Tensor(_)) { self.load(source, out)? } else { source };
        if let ExprKind::Var(id) = source.kind {
            self.temporaries.remove(&id);
        }
        let span = source.span;
        let (vars, at) = self.coordinates(tile_shape(&source)?.shape.len(), span);
        let mut place = pieces.to_vec();
        place.extend(at.iter().cloned());
        let body = vec![assign(points(store.clone(), &place)?, points(source.clone(), &at)?)];
        out.push(stmt(StmtKind::Owned { vars, tile: source, body }, span));
        Ok(())
    }

    fn store_nested(&mut self, slot: &Member, pieces: &[Expr], inner: &Member, out: &mut Vec<Stmt>) -> Result<(), String> {
        match (slot, inner) {
            (Member::Scalar(store), Member::Scalar(source)) | (Member::Tile(store), Member::Tile(source)) => self.store_tile(store, pieces, source.clone(), out),
            (Member::Tuple(slots), Member::Tuple(inners)) if slots.len() == inners.len() => slots.iter().zip(inners).try_for_each(|(s, i)| self.store_nested(s, pieces, i, out)),
            (Member::Result(slot), Member::Result(inner)) => self.store_nested(&slot.member, pieces, &inner.member, out),
            _ => Err("a yielded nested region result differs from the member schema of its storage".into()),
        }
    }

    /// The member at the given pieces. Nested geometry is re-expressed in the consumer's
    /// piece ordinals.
    fn select(member: &Member, pieces: &[Expr], rebase: &dyn Fn(&Sym) -> Result<Sym, String>) -> Result<Value<'a>, String> {
        Ok(match member {
            Member::Scalar(store) => Value::Scalar(points(store.clone(), pieces)?),
            Member::Tile(store) => Value::Shaped(points(store.clone(), pieces)?),
            Member::Tuple(members) => Value::Tuple(members.iter().map(|m| Self::select(m, pieces, rebase)).collect::<Result<_, _>>()?),
            Member::Result(inner) => Value::Result(ResultValue {
                binders: inner.binders.iter().map(|b| Ok(Binder { lo: rebase(&b.lo)?, ..b.clone() })).collect::<Result<_, String>>()?,
                atoms: inner.atoms.clone(),
                member: Self::view(&inner.member, pieces)?,
            }),
        })
    }

    fn view(member: &Member, pieces: &[Expr]) -> Result<Member, String> {
        Ok(match member {
            Member::Scalar(store) => Member::Scalar(points(store.clone(), pieces)?),
            Member::Tile(store) => Member::Tile(points(store.clone(), pieces)?),
            Member::Tuple(members) => Member::Tuple(members.iter().map(|m| Self::view(m, pieces)).collect::<Result<_, _>>()?),
            Member::Result(inner) => Member::Result(Box::new(ResultValue { binders: inner.binders.clone(), atoms: inner.atoms.clone(), member: Self::view(&inner.member, pieces)? })),
        })
    }

    /// `results[p]`: the member of the current visit of the rebound traversal.
    pub fn member(&mut self, f: &mut Frame<'a>, result: &'a sir::Expr, slices: &[st::SliceId], out: &mut Vec<Stmt>) -> Result<Value<'a>, String> {
        let Value::Result(result) = self.value(f, result, out)? else {
            return Err(format!("member selection in `{}` is not over a region result", f.definition.name));
        };
        if slices.len() != result.binders.len() {
            return Err(format!("member selection in `{}` uses {} slices for a {}-binder result", f.definition.name, slices.len(), result.binders.len()));
        }
        let ordinals = slices.iter().map(|s| f.slice(*s).map(|s| s.ordinal.clone())).collect::<Result<Vec<_>, _>>()?;
        let pieces: Vec<Expr> = ordinals.iter().map(|o| symbol(o.clone(), Span::default())).collect();
        let atoms = result.atoms.clone();
        let rebase = move |sym: &Sym| substitute(sym, &|atom| Ok(atoms.iter().position(|a| a == atom).map(|k| ordinals[k].clone())));
        Self::select(&result.member, &pieces, &rebase)
    }

    // ---- merge ----

    fn read_partial(&mut self, member: &Member, at: &Expr, out: &mut Vec<Stmt>) -> Result<Value<'a>, String> {
        Ok(match member {
            // Scalars are read before the level overwrites their slot.
            Member::Scalar(store) => {
                let value = points(store.clone(), std::slice::from_ref(at))?;
                let local = self.local("partial", value.ty.clone(), value.span);
                out.push(assign(local.clone(), value));
                Value::Scalar(local)
            }
            Member::Tile(store) => Value::Shaped(points(store.clone(), std::slice::from_ref(at))?),
            Member::Tuple(members) => Value::Tuple(members.iter().map(|m| self.read_partial(m, at, out)).collect::<Result<_, _>>()?),
            Member::Result(_) => return Err("a merge over nested region results has no execution form".into()),
        })
    }

    /// Values fully evaluated into storage independent of the partial stores.
    fn detach(&mut self, value: Value<'a>, out: &mut Vec<Stmt>) -> Result<Value<'a>, String> {
        Ok(match value {
            Value::Scalar(e) if !matches!(e.kind, ExprKind::Var(_)) => {
                let local = self.local("merged", e.ty.clone(), e.span);
                out.push(assign(local.clone(), e));
                Value::Scalar(local)
            }
            Value::Shaped(e) if !matches!(e.kind, ExprKind::Var(_)) => Value::Shaped(self.copy(e, out)?),
            Value::Tuple(items) => Value::Tuple(items.into_iter().map(|v| self.detach(v, out)).collect::<Result<_, _>>()?),
            other => other,
        })
    }

    /// Adjacent pairs level by level over `parts` partials, odd value forwarded; the
    /// merged value ends in slot zero.
    fn combine(&mut self, f: &mut Frame<'a>, merge: &'a sir::Merge, mut member: Member, parts: i64, span: Span, out: &mut Vec<Stmt>) -> Result<Value<'a>, String> {
        let mut live = parts;
        while live > 1 {
            let pairs = live / 2;
            let (var, atom, pair) = self.index("pair", span);
            let ordinal = Sym::atom(atom);
            let mut body = Vec::new();
            let left = self.read_partial(&member, &symbol(ordinal.scale(2), span), &mut body)?;
            let right = self.read_partial(&member, &symbol(ordinal.scale(2).add(&Sym::constant(1)), span), &mut body)?;
            self.bind_operand(f, &merge.left, left)?;
            self.bind_operand(f, &merge.right, right)?;
            f.yields.push(YieldSink::Slots(Slots { conditional: f.conditional, loops: f.loops + 1, ..Slots::default() }));
            f.loops += 1;
            let merged = self.block(f, &merge.body, false);
            f.loops -= 1;
            let Some(YieldSink::Slots(slots)) = f.yields.pop() else {
                return Err(format!("merge of `{}` lost its yield boundary", f.definition.name));
            };
            let merged = merged?;
            body.extend(slots.prologue);
            body.extend(merged);
            let mut values = slots.locals.or(slots.direct).ok_or_else(|| format!("merge body of `{}` yields no value", f.definition.name))?;
            let value = if values.len() == 1 { values.swap_remove(0) } else { Value::Tuple(values) };
            let value = self.detach(value, &mut body)?;
            self.store_member(&mut member, std::slice::from_ref(&pair), value, &mut body)?;
            out.push(stmt(StmtKind::Range { var, lo: Sym::constant(0), hi: Sym::constant(pairs), body }, span));
            if live % 2 == 1 {
                let forwarded = self.read_partial(&member, &int(live - 1, span), out)?;
                self.store_member(&mut member, &[int(pairs, span)], forwarded, out)?;
            }
            live = pairs + live % 2;
        }
        self.read_partial(&member, &int(0, span), out)
    }

    fn bind_operand(&mut self, f: &mut Frame<'a>, pattern: &sir::Pattern, value: Value<'a>) -> Result<(), String> {
        match (pattern, value) {
            (sir::Pattern::Var(v), value) => {
                f.vars[*v] = Some(value);
                Ok(())
            }
            (sir::Pattern::Tuple(patterns), Value::Tuple(values)) if patterns.len() == values.len() => patterns.iter().zip(values).try_for_each(|(p, v)| self.bind_operand(f, p, v)),
            _ => Err(format!("merge operand pattern in `{}` does not match the partial schema", f.definition.name)),
        }
    }
}
