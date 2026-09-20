//! The target-neutral structural walk of candidate template bodies (`bind_structure`).
//!
//! One walk per candidate partitions its root block into launches and derives its account:
//!   root `parallel`            -> one launch dispatched over pieces
//!   root `ordered`/`pipeline`  -> one single-piece launch with serial windows
//!   nested region              -> serial loops in its owner
//!   `pipeline`                 -> synchronous, same participant, ring depth 1
//!   `merge`                    -> canonical recurrence in the owner
//!   region results             -> storage inside one owner
//!   root stages                -> successive launches
//!   invocation-scope serial statements -> one single-participant launch per run
//! Anything else is `UnsupportedMapping`. Nothing here chooses among alternatives.
//!
//! What differs between targets is stated by the backend's `Accounting`: the classes of work
//! it prices, how element work is shared inside a piece, what it records about tile storage,
//! and which intrinsics it maps. The walk itself, name and bound resolution, multiplicity,
//! distinct-view analysis and the launch partition are shared.
use super::quantity::{Quantity, SymbolicExtent};
use super::SelectionError;
use seismic_lang::family::{CandidateRef, Family, ScopeStep, Sequence, SiteId, SiteKind, Unit};
use seismic_lang::intrinsics::Operation;
use seismic_lang::sir::{
    Block, Body, CallId, Definition, Expr, ExprKind, Index, Pattern, Program, Region, RegionSource,
    SliceParent, Stmt, StmtKind, VarId,
};
use seismic_lang::sym::Sym;
use seismic_lang::syntax::ast::AssignOp;
use seismic_lang::sir::RegionMode;
use seismic_lang::types::{Elem, Extent, RegionId, Shaped, SliceId, Ty};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

/// A tile the walk met, offered to the backend's ledger.
pub struct TileEvent<'e, 'a> {
    pub bound: &'e Bound<'a>,
    /// The bound variable of a tile binding; `None` for a `load` of external storage in
    /// expression position.
    pub variable: Option<VarId>,
    pub shaped: &'e Shaped,
    /// The tile is a snapshot of external storage (the value is `load(view)` of a tensor).
    pub snapshot: bool,
    pub pieces: &'e Quantity,
    pub launch: Option<(CandidateRef, usize)>,
    /// Where the tile lives relative to the inner owner regions of its launch (always
    /// `Piece` for a backend without `INNER_OWNERS`).
    pub owner: TileOwner,
}

/// The owner of a tile in a launch whose mapping gives inner owner regions participants of
/// their own.
#[derive(Clone, Debug)]
pub enum TileOwner {
    /// The launch has no inner owner region: the tile belongs to the launch piece.
    Piece,
    /// The outer owner of a launch with inner owner regions: one tile per launch piece,
    /// shared by its inner owners.
    Outer,
    /// An inner owner: one tile per inner owner, this many per launch piece.
    Inner(Quantity),
}

/// What a backend's mapping accounts for while the shared walk traverses a candidate body.
/// Every method is a deterministic rule of that mapping; none chooses among alternatives.
pub trait Accounting: Sized + 'static {
    /// Backend name in diagnostics.
    const NAME: &'static str;
    /// Tiles live in the same memory as tensors: element accesses and conversions of local
    /// tiles are memory traffic like those of external storage.
    const TILES_ARE_MEMORY_TRAFFIC: bool = false;
    /// `atomic` updates have a mapping.
    const ATOMIC: bool = true;
    /// The mapping gives the inner owner regions of a launch participants of their own (see
    /// `seismic_lang::instantiate::owner_regions`): the pieces that run concurrently are then
    /// the inner owners of every launch piece. Otherwise a nested region is serial loops.
    const INNER_OWNERS: bool = false;
    /// Work totals of one execution scope, each term already scaled by multiplicity.
    type Work: Clone + std::fmt::Debug + Default + Send + Sync;
    /// Storage and participation facts of one candidate that limits and estimates read.
    type Ledger: Clone + std::fmt::Debug + Default;

    /// Branch join: each class charged at the maximum of its arms.
    fn maximum(into: &mut Self::Work, then: Self::Work, els: Self::Work);
    fn quantities(work: &Self::Work) -> Vec<&Quantity>;
    fn push_operations(work: &mut Self::Work, scaled: Quantity);
    /// One serial window visit of an `ordered` or `pipeline` region (or any region).
    fn push_visit(work: &mut Self::Work, mode: RegionMode, scaled: Quantity);
    fn push_traffic(work: &mut Self::Work, external: bool, scaled: Quantity);
    /// A position in the element operations of `work`, for `carried_update`.
    fn operations_mark(_work: &Self::Work) -> usize {
        0
    }
    /// The element operations charged since `mark` are `trips` trips of a loop whose body holds
    /// a loop-carried scalar update (a statement whose value reads its own target at a place
    /// no binder of that loop selects). A backend on which such a trip is latency-bound states
    /// its floor here.
    fn carried_update(_work: &mut Self::Work, _mark: usize, _trips: Quantity) {}
    /// One completion among every participant of a launch piece with inner owner regions: the
    /// publication of a tile of the outer owner, or the completion of an inner owner region.
    /// `scaled` counts one per inner owner that takes part (`INNER_OWNERS` only).
    fn push_owner_completion(_work: &mut Self::Work, _scaled: Quantity) {}
    /// Element operations on the accounted path of elementwise work over `elements`.
    fn share(elements: Quantity) -> Quantity {
        elements
    }
    /// Trip count of `owned` coordinates over `axes` of `shaped`, `count` elements in all.
    fn coordinates_trip(
        _bound: &Bound<'_>,
        _shaped: &Shaped,
        _axes: &[usize],
        count: Quantity,
    ) -> Quantity {
        count
    }
    /// Element operations of consuming one element of packed `shaped`; `None` when dense.
    fn packed_element(bound: &Bound<'_>, shaped: &Shaped) -> Option<Quantity>;
    /// Element operations of one reduction of `value` along `axis` into `result`.
    fn reduction(
        bound: &Bound<'_>,
        value: &Expr,
        axis: usize,
        unordered: bool,
        result: &Ty,
    ) -> Quantity;
    fn tile(ledger: &mut Self::Ledger, event: TileEvent<'_, '_>);
    /// An intrinsic whose value is coordinate arithmetic (never charged).
    /// An `owned` loop over every axis of a tile, with its coordinate binders, before its body
    /// is walked: a backend whose storage rule depends on how element loops read other tiles
    /// records it in its ledger.
    fn owned_loop(_walker: &mut Walker<'_, '_, Self>, _coordinates: &[VarId], _body: &Block) {}

    fn induction_intrinsic(_op: &Operation) -> bool {
        false
    }
    /// One argument of a call. The default walks it like any expression.
    fn call_argument(
        walker: &mut Walker<'_, '_, Self>,
        _call: CallId,
        _ordinal: usize,
        argument: &Expr,
    ) -> Result<(), SelectionError> {
        walker.expr(argument)
    }
    /// A target intrinsic whose arguments were already walked.
    fn intrinsic(
        walker: &mut Walker<'_, '_, Self>,
        op: &Operation,
        args: &[Expr],
    ) -> Result<(), SelectionError>;
    /// Work a callee's body scope inherits from its call in the parent.
    fn inherited(_parent: &Self::Ledger, _call: CallId) -> Self::Work {
        Self::Work::default()
    }
}

/// Where a statement executes: multiplicity of its block and the pieces of its launch.
#[derive(Clone, Debug)]
pub struct Scope {
    pub multiplicity: Vec<Quantity>,
    pub pieces: Quantity,
    /// Invocation scope of a root-context candidate: regions here are launches.
    pub invocation: bool,
    /// The root launch statements here execute in: its owning candidate and launch ordinal.
    /// `None` for invocation-scope serial statements.
    pub launch: Option<(CandidateRef, usize)>,
    /// Inner owners per launch piece when the statements execute inside an inner owner
    /// region of their launch (`Accounting::INNER_OWNERS`).
    pub owners: Option<Quantity>,
}

#[derive(Clone, Debug)]
pub struct Context {
    pub scope: Scope,
    /// The candidate and its ancestors, entry first.
    pub guards: Vec<CandidateRef>,
}

#[derive(Clone, Debug)]
pub struct Launch<A: Accounting> {
    pub region: RegionId,
    pub mode: RegionMode,
    pub merge: bool,
    /// `(site, extent)` of each binder that is a width site.
    pub binders: Vec<(SiteId, u64)>,
    pub binder_count: usize,
    /// Pieces that run concurrently: the launch pieces, times the inner owners of each when
    /// the mapping gives inner owner regions participants of their own.
    pub pieces: Quantity,
    /// The launch pieces: visits of the launch's own binders.
    pub groups: Quantity,
    /// Inner owners per launch piece of every inner owner region of the launch, in authored
    /// order (empty without `Accounting::INNER_OWNERS`).
    pub owners: Vec<Quantity>,
    pub work: A::Work,
}

#[derive(Clone, Debug)]
pub struct Account<A: Accounting> {
    pub context: Context,
    pub launches: Vec<Launch<A>>,
    /// Work outside root regions (the whole body for a nested candidate).
    pub rest: A::Work,
    /// Runs of invocation-scope serial statements: one single-participant launch each.
    pub serial_launches: u64,
    pub calls: BTreeMap<CallId, Scope>,
    pub regions: BTreeMap<RegionId, Scope>,
    pub ledger: A::Ledger,
}

/// Name resolution of one candidate body against the family: slices to sites, shape
/// parameters to workload constants or caller slice widths.
pub struct Bound<'a> {
    pub definition: &'a Definition,
    pub body: &'a Body,
    allow_numerical_effects: bool,
    constants: Arc<BTreeMap<String, i64>>,
    /// Static `[lower, upper]` range of every non-constant name an extent may mention: shape
    /// parameters bound to caller slices (exact site widths), dynamic shape parameters,
    /// `@dyn#n` range lengths and index variables (`name#var`, see `check::var_atom`).
    ranges: Arc<BTreeMap<String, (Quantity, Quantity)>>,
    /// Names of `ranges` whose value is fixed by the witness (caller slice widths). Every
    /// other ranged name is a runtime value known only by its enclosure.
    selected: BTreeSet<String>,
    elems: &'a BTreeMap<String, Elem>,
    slices: BTreeMap<SliceId, (SiteId, u64, bool)>,
    /// Width sites of this body that refine another of its width sites (`Family::refinements`).
    refined: BTreeMap<SiteId, SiteId>,
}

/// Visits of a refinement per visit of the binder it refines: `refined width / width`.
fn refinement_visits(refined: SiteId, site: SiteId) -> Quantity {
    let apply = |values: &[u64]| match values {
        [outer, inner] if *inner > 0 => Ok(outer.div_ceil(*inner)),
        _ => Err("rule `refinement_visits` reads a refined width and a positive width".to_string()),
    };
    Quantity::Rule(
        super::quantity::Rule {
            name: "refinement_visits",
            apply: Arc::new(apply),
        },
        vec![Quantity::Site(refined), Quantity::Site(site)],
    )
}

fn site_extent(family: &Family, id: SiteId) -> Result<u64, SelectionError> {
    let site = family.sites.get(id.0 as usize).ok_or_else(|| {
        SelectionError::Reconstruction(format!("site {} is absent from the family", id.0))
    })?;
    u64::try_from(site.extent)
        .ok()
        .filter(|&e| e > 0)
        .ok_or_else(|| {
            SelectionError::AnalysisUnavailable(format!(
                "site {} has no positive static extent",
                id.0
            ))
        })
}

impl<'a> Bound<'a> {
    pub fn new(
        program: &'a Program,
        family: &'a Family,
        candidate: CandidateRef,
        parent: Option<&Bound<'a>>,
    ) -> Result<Self, SelectionError> {
        let selected = family.candidate(candidate);
        let template = family.template(selected.template);
        let definition = program.definition(template.definition);
        let body = &definition.body;
        let mut ranges = BTreeMap::new();
        for (name, reference) in &selected.structural {
            let id = reference.0;
            let extent = site_extent(family, id)?;
            let width = match family.sites[id.0 as usize].kind {
                SiteKind::Width { .. } => Quantity::Site(id),
                SiteKind::Parts { .. } => Quantity::Pieces { extent, site: id },
            };
            ranges.insert(name.clone(), (width.clone(), width));
        }
        let mut slices = BTreeMap::new();
        for &id in &selected.sites {
            let extent = site_extent(family, id)?;
            match family.sites[id.0 as usize].kind {
                SiteKind::Width { slice, .. } => slices.insert(slice, (id, extent, false)),
                SiteKind::Parts { slice, .. } => slices.insert(slice, (id, extent, true)),
            };
        }
        // Runtime-valued extents are charged at their static upper bound.
        // A dynamic shape parameter: the caller's extent expression, bounded in the caller.
        for (name, sym) in family.dynamic_args(program, candidate) {
            let caller = parent.ok_or_else(|| {
                SelectionError::Reconstruction(format!(
                    "`{}`: dynamic parameter `{name}` without a caller",
                    definition.name
                ))
            })?;
            ranges.insert(
                name,
                (caller.bounded(&sym, true), caller.bounded(&sym, false)),
            );
        }
        if let Some(name) = template
            .dynamic
            .iter()
            .find(|name| !ranges.contains_key(*name))
        {
            return Err(SelectionError::AnalysisUnavailable(format!(
                "`{}`: dynamic shape parameter `{name}` has no caller extent to bound it by",
                definition.name
            )));
        }
        let selected_names = selected
            .structural
            .iter()
            .map(|(name, _)| name.clone())
            .collect();
        let refined = family
            .refinements
            .iter()
            .filter(|(inner, _)| selected.sites.contains(inner))
            .copied()
            .collect();
        let mut bound = Bound {
            definition,
            body,
            allow_numerical_effects: family.allow_numerical_effects,
            constants: Arc::new(template.shapes.clone()),
            ranges: Arc::new(ranges),
            selected: selected_names,
            elems: &template.elems,
            slices,
            refined,
        };
        // Runtime names of the body, each with the inclusive range it is bounded by:
        //   `@dyn#n` (length of a runtime-bounded range view)  [0, extent of the selected axis]
        //   range-loop binder                                   [lo, hi - 1]
        //   slice-member binder                                 [lo, hi - 1] of the slice's domain
        //   owned/axis coordinate                               [0, axis extent - 1]
        //   any `index[N]` variable                             [0, N - 1]
        // Resolved to a fixpoint, since a bound may mention other runtime names.
        enum End<'e> {
            Zero,
            Sym(Sym),
            Axis(&'e Extent, bool),
        }
        let mut pending: Vec<(String, End, End)> = Vec::new();
        let atom = |var: VarId| body.vars.get(var).map(|v| format!("{}#{var}", v.name));
        let one = Sym::constant(1);
        fn loops<'e>(block: &'e Block, visit: &mut dyn FnMut(&'e Stmt)) {
            for s in block {
                visit(s);
                match &s.kind {
                    StmtKind::Region(r) => {
                        loops(&r.body, visit);
                        if let Some(m) = &r.merge {
                            loops(&m.body, visit);
                        }
                    }
                    StmtKind::Stages(stages) => stages.iter().for_each(|st| loops(&st.body, visit)),
                    StmtKind::Range { body, .. }
                    | StmtKind::Coordinates { body, .. }
                    | StmtKind::Members { body, .. } => loops(body, visit),
                    StmtKind::If { then, els, .. } => {
                        loops(then, visit);
                        loops(els, visit);
                    }
                    _ => {}
                }
            }
        }
        let mut statement = |s: &'a Stmt| match &s.kind {
            StmtKind::Range { var, lo, hi, .. } => {
                if let (Some(name), Some(lo), Some(hi)) = (atom(*var), &lo.sym, &hi.sym) {
                    pending.push((name, End::Sym(lo.clone()), End::Sym(hi.sub(&one))));
                }
            }
            StmtKind::Members { var, slice, .. } => {
                let mut current = *slice;
                for _ in 0..=body.slices.len() {
                    match body.slices.get(current.0 as usize).map(|d| &d.parent) {
                        Some(SliceParent::Domain { lo, hi }) => {
                            if let Some(name) = atom(*var) {
                                pending.push((name, End::Sym(lo.clone()), End::Sym(hi.sub(&one))));
                            }
                            break;
                        }
                        Some(SliceParent::Refine(parent) | SliceParent::Rebind(parent)) => {
                            current = *parent
                        }
                        None => break,
                    }
                }
            }
            StmtKind::Coordinates { vars, of, axes, .. } => {
                for (var, axis) in vars.iter().zip(axes) {
                    if let (Some(name), Some(extent)) =
                        (atom(*var), of.ty.shaped().and_then(|t| t.axes.get(*axis)))
                    {
                        pending.push((name, End::Zero, End::Axis(extent, true)));
                    }
                }
            }
            _ => {}
        };
        loops(&body.block, &mut statement);
        // Regions and loops in expression position.
        each_expr(&body.block, &mut |e| {
            if let ExprKind::Region(r) = &e.kind {
                loops(&r.body, &mut statement);
            }
        });
        each_expr(&body.block, &mut |e| {
            let ExprKind::Index { base, indices } = &e.kind else {
                return;
            };
            let (Some(from), Some(to)) = (base.ty.shaped(), e.ty.shaped()) else {
                return;
            };
            let kept: Vec<usize> = (0..from.axes.len())
                .filter(|&a| !matches!(indices.get(a), Some(Index::Point(_) | Index::Coord(_))))
                .collect();
            if kept.len() != to.axes.len() {
                return;
            }
            for (position, &axis) in kept.iter().enumerate() {
                if let (Some(Index::Range { .. }), Extent::Semantic(sym)) =
                    (indices.get(axis), &to.axes[position])
                {
                    let name = sym
                        .params()
                        .into_iter()
                        .find(|p| p.starts_with('@') && *sym == Sym::param(p));
                    if let Some(name) = name {
                        pending.push((name, End::Zero, End::Axis(&from.axes[axis], false)));
                    }
                }
            }
        });
        for (var, declared) in body.vars.iter().enumerate() {
            if let Ty::Index(n) = &declared.ty {
                pending.push((
                    format!("{}#{var}", declared.name),
                    End::Zero,
                    End::Sym(n.sub(&one)),
                ));
            }
        }
        // The first bound stated for a name stands (a loop's own bounds before its index type).
        let mut seen = BTreeSet::new();
        pending.retain(|(name, ..)| seen.insert(name.clone()));
        loop {
            let known = |name: &String| {
                bound.constants.contains_key(name) || bound.ranges.contains_key(name)
            };
            let ready = |end: &End| match end {
                End::Zero | End::Axis(Extent::Structural(_), _) => true,
                End::Sym(sym) | End::Axis(Extent::Semantic(sym), _) => {
                    sym.params().iter().all(known)
                }
            };
            let Some(position) = pending
                .iter()
                .position(|(_, lo, hi)| ready(lo) && ready(hi))
            else {
                break;
            };
            let (name, lo, hi) = pending.remove(position);
            let quantity = |end: End, lower: bool| match end {
                End::Zero => Quantity::Constant(0),
                End::Sym(sym) => bound.bounded(&sym, lower),
                End::Axis(extent, false) => bound.axis(extent),
                End::Axis(extent, true) => Quantity::Predecessor(Box::new(bound.axis(extent))),
            };
            let range = (quantity(lo, true), quantity(hi, false));
            let mut resolved = (*bound.ranges).clone();
            resolved.insert(name, range);
            bound.ranges = Arc::new(resolved);
        }
        // A runtime range length with no static bound is reported here; an unbounded index
        // variable only when an extent mentions it.
        if let Some((name, _, End::Axis(extent, _))) =
            pending.iter().find(|(name, ..)| name.starts_with('@'))
        {
            return Err(SelectionError::AnalysisUnavailable(format!("`{}`: runtime range extent `{name}` selects from an axis of extent `{extent}` with no static upper bound", definition.name)));
        }
        Ok(bound)
    }

    pub fn allow_numerical_effects(&self) -> bool {
        self.allow_numerical_effects
    }

    fn site(&self, slice: SliceId) -> Option<(SiteId, u64, bool)> {
        let mut current = slice;
        for _ in 0..=self.body.slices.len() {
            if let Some(found) = self.slices.get(&current) {
                return Some(*found);
            }
            match self.body.slices.get(current.0 as usize)?.parent {
                SliceParent::Rebind(parent) => current = parent,
                _ => return None,
            }
        }
        None
    }

    /// The width site `slice` refines, when `slice` is a refinement of a binder of this body.
    pub fn refined_site(&self, slice: SliceId) -> Option<SiteId> {
        self.site(slice)
            .and_then(|(site, _, _)| self.refined.get(&site).copied())
    }

    /// Pieces of the refinement `slice` per piece of the binder it refines.
    pub fn refinement_pieces(&self, slice: SliceId) -> Option<Quantity> {
        let (site, _, _) = self.site(slice)?;
        Some(refinement_visits(*self.refined.get(&site)?, site))
    }

    pub fn width_site(&self, slice: SliceId) -> Option<(SiteId, u64)> {
        self.site(slice).filter(|s| !s.2).map(|s| (s.0, s.1))
    }

    fn missing(&self, slice: SliceId) -> Quantity {
        Quantity::Unknown(format!(
            "slice#{} of `{}` has no numerical site",
            slice.0, self.definition.name
        ))
    }

    pub fn width(&self, slice: SliceId) -> Quantity {
        match self.site(slice) {
            Some((site, _, false)) => Quantity::Site(site),
            Some((site, extent, true)) => Quantity::Pieces { extent, site },
            None => self.missing(slice),
        }
    }

    pub fn visits(&self, slice: SliceId) -> Quantity {
        match self.site(slice) {
            Some((site, extent, false)) => Quantity::Pieces { extent, site },
            Some((site, _, true)) => Quantity::Site(site),
            None => self.missing(slice),
        }
    }

    /// The static upper bound of `sym` (its value when nothing in it is runtime).
    pub fn symbolic(&self, sym: &Sym) -> Quantity {
        self.bounded(sym, false)
    }

    pub fn bounded(&self, sym: &Sym, lower: bool) -> Quantity {
        match sym.as_constant() {
            Some(c) => Quantity::Constant(u64::try_from(c).unwrap_or(0)),
            None => Quantity::Symbolic(Arc::new(SymbolicExtent {
                sym: sym.clone(),
                lower,
                constants: self.constants.clone(),
                ranges: self.ranges.clone(),
            })),
        }
    }

    /// Whether the extent is fixed at selection time (workload constants and site widths).
    /// Whether an extent is static under the selected geometry (not a runtime length).
    pub fn fixed(&self, extent: &Extent) -> bool {
        match extent {
            Extent::Structural(_) => true,
            Extent::Semantic(sym) => sym
                .params()
                .iter()
                .all(|name| self.constants.contains_key(name) || self.selected.contains(name)),
        }
    }

    pub fn axis(&self, extent: &Extent) -> Quantity {
        match extent {
            Extent::Semantic(sym) => self.symbolic(sym),
            Extent::Structural(slice) => self.width(*slice),
        }
    }

    pub fn elements(&self, ty: &Ty) -> Quantity {
        match ty.shaped() {
            Some(shaped) => self.elements_of(shaped),
            None => Quantity::one(),
        }
    }

    pub fn elements_of(&self, shaped: &Shaped) -> Quantity {
        Quantity::product(shaped.axes.iter().map(|a| self.axis(a)))
    }

    /// Stored bits per value. Packed representations round their fractional rate up.
    pub fn bits(&self, elem: &Elem) -> Quantity {
        match elem {
            Elem::Dtype(d) => Quantity::Constant(u64::from(d.bytes()) * 8),
            Elem::Repr(name) => match seismic_lang::repr::lookup(name) {
                Some(repr) => Quantity::Constant(repr.bits_per_value().ceil() as u64),
                None => Quantity::Unknown(format!("representation `{name}` is not registered")),
            },
            Elem::Param(name) => match self.elems.get(name) {
                Some(Elem::Param(_)) | None => Quantity::Unknown(format!(
                    "element parameter `{name}` of `{}` is unbound",
                    self.definition.name
                )),
                Some(bound) => self.bits(bound),
            },
        }
    }

    /// Stored bits of `elements` values of `elem`, at a packed representation's exact
    /// fractional rate (in sixteenths of a bit per value).
    pub fn stored_bits(&self, elements: Quantity, elem: &Elem) -> Quantity {
        let resolved = match elem {
            Elem::Param(name) => self.elems.get(name).unwrap_or(elem),
            other => other,
        };
        match resolved {
            Elem::Repr(name) => match seismic_lang::repr::lookup(name) {
                Some(repr) => Quantity::Quotient(
                    Box::new(Quantity::product([
                        elements,
                        Quantity::Constant((repr.bits_per_value() * 16.0).ceil() as u64),
                    ])),
                    16,
                ),
                None => Quantity::Unknown(format!("representation `{name}` is not registered")),
            },
            other => Quantity::product([elements, self.bits(other)]),
        }
    }

    /// The dense dtype `elem` resolves to, if it is one.
    pub fn dtype(&self, elem: &Elem) -> Option<seismic_lang::types::DType> {
        match elem {
            Elem::Dtype(d) => Some(*d),
            Elem::Param(name) => match self.elems.get(name) {
                Some(Elem::Dtype(d)) => Some(*d),
                _ => None,
            },
            Elem::Repr(_) => None,
        }
    }

    /// The coefficient group width of `elem` when it resolves to a packed representation.
    pub fn packed_group(&self, elem: &Elem) -> Option<u64> {
        match elem {
            Elem::Repr(name) => seismic_lang::repr::lookup(name).map(|repr| u64::from(repr.group)),
            Elem::Param(name) => match self.elems.get(name) {
                Some(bound @ Elem::Repr(_)) => self.packed_group(bound),
                _ => None,
            },
            Elem::Dtype(_) => None,
        }
    }

    /// The concrete element `elem` resolves to under this candidate's element bindings.
    pub fn element<'e>(&'e self, elem: &'e Elem) -> Option<&'e Elem> {
        match elem {
            Elem::Param(name) => self
                .elems
                .get(name)
                .filter(|bound| !matches!(bound, Elem::Param(_))),
            concrete => Some(concrete),
        }
    }

    pub fn tile_bits(&self, shaped: &Shaped) -> Quantity {
        Quantity::product(
            shaped
                .axes
                .iter()
                .map(|a| self.axis(a))
                .chain([self.bits(&shaped.elem)]),
        )
    }

    /// Trip count of a loop statement. Runtime ranges are charged at their static bound.
    pub fn trip<A: Accounting>(&self, statement: &Stmt) -> Option<Quantity> {
        match &statement.kind {
            StmtKind::Range { var, lo, hi, .. } => {
                let bound = match self.body.vars.get(*var).map(|v| &v.ty) {
                    Some(Ty::Index(n)) => self.symbolic(n),
                    _ => Quantity::Unknown(format!(
                        "range in `{}` has no static bound",
                        self.definition.name
                    )),
                };
                Some(match (&lo.sym, &hi.sym) {
                    (Some(lo), Some(hi)) => Quantity::Bounded {
                        exact: Box::new(self.symbolic(&hi.sub(lo))),
                        bound: Box::new(bound),
                    },
                    _ => bound,
                })
            }
            StmtKind::Coordinates { of, axes, .. } => {
                let shaped = of.ty.shaped()?;
                let count = Quantity::product(
                    axes.iter()
                        .filter_map(|&a| shaped.axes.get(a))
                        .map(|a| self.axis(a)),
                );
                Some(A::coordinates_trip(self, shaped, axes, count))
            }
            StmtKind::Members { slice, .. } => Some(self.width(*slice)),
            _ => None,
        }
    }
}

pub fn tile_root(expr: &Expr) -> Option<VarId> {
    match &expr.kind {
        ExprKind::Var(v) => Some(*v),
        ExprKind::Index { base, .. }
        | ExprKind::Transpose(base)
        | ExprKind::Reshape { base, .. }
        | ExprKind::Load(base) => tile_root(base),
        _ => None,
    }
}

pub fn external(ty: &Ty) -> bool {
    matches!(ty, Ty::Tensor(_) | Ty::View(_))
}

fn each_expr_of_region<'a>(r: &'a Region, f: &mut dyn FnMut(&'a Expr)) {
    if let RegionSource::Results(e) = &r.source {
        each_expr_in(e, f);
    }
    each_expr(&r.body, f);
    if let Some(merge) = &r.merge {
        each_expr_in(&merge.identity, f);
        each_expr(&merge.body, f);
    }
}

/// Every expression nested in `e`, itself included, in authored order.
pub fn each_expr_in<'a>(e: &'a Expr, f: &mut dyn FnMut(&'a Expr)) {
    let (expr, region) = (each_expr_in, each_expr_of_region);
    {
        f(e);
        match &e.kind {
            ExprKind::Int(_)
            | ExprKind::Float(_)
            | ExprKind::Bool(_)
            | ExprKind::Var(_)
            | ExprKind::ShapeParam(_)
            | ExprKind::TileAlloc
            | ExprKind::CoordOf(_) => {}
            ExprKind::Tuple(items)
            | ExprKind::Math { args: items, .. }
            | ExprKind::Call { args: items, .. }
            | ExprKind::Intrinsic { args: items, .. } => items.iter().for_each(|i| expr(i, f)),
            ExprKind::Range { lo, hi } => {
                expr(lo, f);
                expr(hi, f);
            }
            ExprKind::Field { base, .. }
            | ExprKind::Filled { like: base, .. }
            | ExprKind::Member { result: base, .. }
            | ExprKind::Transpose(base)
            | ExprKind::Reshape { base, .. }
            | ExprKind::Load(base)
            | ExprKind::Decode(base)
            | ExprKind::Cast { expr: base, .. }
            | ExprKind::Unary { expr: base, .. }
            | ExprKind::Reduce { value: base, .. }
            | ExprKind::ExtentOf { base, .. }
            | ExprKind::Accessor { base, .. }
            | ExprKind::Geometry { base, .. } => expr(base, f),
            ExprKind::Index { base, indices } => {
                expr(base, f);
                for index in indices {
                    match index {
                        Index::Point(p) => expr(p, f),
                        Index::Range { start, end } => {
                            [start, end].into_iter().flatten().for_each(|b| expr(b, f))
                        }
                        Index::Coord(_) | Index::Slice(_) => {}
                    }
                }
            }
            ExprKind::Binary { lhs, rhs, .. }
            | ExprKind::Atomic {
                place: lhs,
                value: rhs,
                ..
            } => {
                expr(lhs, f);
                expr(rhs, f);
            }
            ExprKind::Select { cond, then, els } => {
                expr(cond, f);
                expr(then, f);
                expr(els, f);
            }
            ExprKind::Region(r) => region(r, f),
        }
    }
}

/// Every expression of `block`, nested bodies included, in authored order.
pub fn each_expr<'a>(block: &'a [Stmt], f: &mut dyn FnMut(&'a Expr)) {
    let (expr, region) = (each_expr_in, each_expr_of_region);
    for s in block {
        match &s.kind {
            StmtKind::Bind { value, .. } => expr(value, f),
            StmtKind::Assign { target, value, .. } => {
                expr(target, f);
                expr(value, f);
            }
            StmtKind::Region(r) => region(r, f),
            StmtKind::Stages(stages) => stages.iter().for_each(|st| each_expr(&st.body, f)),
            StmtKind::Range { lo, hi, body, .. } => {
                expr(lo, f);
                expr(hi, f);
                each_expr(body, f);
            }
            StmtKind::Coordinates { of, body, .. } => {
                expr(of, f);
                each_expr(body, f);
            }
            StmtKind::Members { body, .. } => each_expr(body, f),
            StmtKind::If { cond, then, els } => {
                expr(cond, f);
                each_expr(then, f);
                each_expr(els, f);
            }
            StmtKind::Publish { value, destination } => {
                expr(value, f);
                expr(destination, f);
            }
            StmtKind::Yield(items) | StmtKind::Return(items) => {
                items.iter().for_each(|i| expr(i, f))
            }
            StmtKind::Expr(e) => expr(e, f),
        }
    }
}

fn find_region(block: &Block, id: RegionId) -> Option<&Region> {
    fn in_region(r: &Region, id: RegionId) -> Option<&Region> {
        if r.id == id {
            return Some(r);
        }
        find_region(&r.body, id).or_else(|| r.merge.as_ref().and_then(|m| find_region(&m.body, id)))
    }
    for s in block {
        let nested = match &s.kind {
            StmtKind::Region(r) => in_region(r, id),
            StmtKind::Stages(stages) => stages.iter().find_map(|st| find_region(&st.body, id)),
            StmtKind::Range { body, .. }
            | StmtKind::Coordinates { body, .. }
            | StmtKind::Members { body, .. } => find_region(body, id),
            StmtKind::If { then, els, .. } => {
                find_region(then, id).or_else(|| find_region(els, id))
            }
            _ => None,
        };
        if nested.is_some() {
            return nested;
        }
    }
    // Regions in expression position, and statement regions inside their bodies.
    let mut found = None;
    each_expr(block, &mut |e| {
        if let (ExprKind::Region(r), None) = (&e.kind, found) {
            found = in_region(r, id);
        }
    });
    found
}

/// Inner owners per launch piece of every inner owner region in the owner body `block` of a
/// launch with binders `launch`, in authored order. Mirrors the positions instantiation maps:
/// the owner body itself and the bodies of its other regions and stages, never loops or
/// branches (`seismic_lang::instantiate::inner_owner_region`).
fn inner_owner_regions(
    bound: &Bound<'_>,
    launch: &[SliceId],
    block: &Block,
    out: &mut Vec<Quantity>,
) {
    fn region(bound: &Bound<'_>, launch: &[SliceId], r: &Region, out: &mut Vec<Quantity>) {
        if seismic_lang::instantiate::inner_owner_region(bound.body, launch, r) {
            let binders = bound
                .body
                .regions
                .get(r.id.0 as usize)
                .map(|declared| declared.binders.as_slice())
                .unwrap_or(&[]);
            out.push(Quantity::product(
                binders.iter().filter_map(|b| bound.refinement_pieces(*b)),
            ));
        } else {
            inner_owner_regions(bound, launch, &r.body, out);
        }
    }
    for s in block {
        match &s.kind {
            StmtKind::Region(r) => region(bound, launch, r, out),
            StmtKind::Stages(stages) => stages
                .iter()
                .for_each(|stage| inner_owner_regions(bound, launch, &stage.body, out)),
            StmtKind::Bind { value, .. } | StmtKind::Expr(value) => {
                if let ExprKind::Region(r) = &value.kind {
                    region(bound, launch, r, out);
                }
            }
            _ => {}
        }
    }
}

pub struct Walker<'a, 'b, A: Accounting> {
    pub bound: &'b Bound<'a>,
    candidate: CandidateRef,
    account: Account<A>,
    /// Work of the scope being walked: the current root launch, or the rest of the body.
    pub work: A::Work,
    multiplicity: Vec<Quantity>,
    pieces: Quantity,
    launch: Option<(CandidateRef, usize)>,
    root: bool,
    depth: usize,
    host_loops: usize,
    /// Immutable integer bindings whose value is induction arithmetic (see `induction`).
    inductions: BTreeSet<VarId>,
    /// The factors of `multiplicity`, each with the binder that repeats it (aligned with
    /// `multiplicity`: one entry per region binder and per loop).
    repetitions: Vec<(Repeats, Quantity)>,
    /// Loops and branches entered since the body began: scalar control of the current owner.
    control: usize,
    /// Per enclosing loop: its body holds a loop-carried scalar update.
    carried: Vec<bool>,
    /// The root `parallel` launch being walked: its binders, the control depth of its owner
    /// body, and the inner owners per piece of each of its inner owner regions.
    owner_body: Option<(Vec<SliceId>, usize, Vec<Quantity>)>,
    /// Inner owners per launch piece while an inner owner region is walked (also inherited
    /// by a callee walked inside one).
    owners: Option<Quantity>,
}

/// What repeats one factor of a statement's multiplicity.
enum Repeats {
    /// A repetition of the calling scope: nothing here names its binder.
    Inherited,
    /// Visits of one region binder.
    Slice(SliceId),
    /// Trips of one loop, by its binders.
    Vars(Vec<VarId>),
}

impl<A: Accounting> Walker<'_, '_, A> {
    /// Integer arithmetic over loop binders, constants, shape parameters, selected geometry
    /// and the participant index: coordinate and address computation. The native compiler
    /// strength-reduces it into the loop it indexes, so it is not charged as lane work; an
    /// integer operation on loaded data (code words, routing indices) is charged.
    fn induction(&self, e: &Expr) -> bool {
        use seismic_lang::sir::VarKind;
        if !matches!(e.ty, Ty::Scalar(d) if d.is_int())
            && !matches!(e.ty, Ty::Index(_) | Ty::Coord(_))
        {
            return false;
        }
        match &e.kind {
            ExprKind::Int(_)
            | ExprKind::ShapeParam(_)
            | ExprKind::CoordOf(_)
            | ExprKind::Geometry { .. }
            | ExprKind::ExtentOf { .. } => true,
            ExprKind::Var(v) => {
                self.inductions.contains(v)
                    || matches!(
                        self.bound.body.vars.get(*v).map(|var| &var.kind),
                        Some(VarKind::RangeIndex | VarKind::Coordinate | VarKind::SliceMember(_))
                    )
            }
            ExprKind::Cast { expr, .. } | ExprKind::Unary { expr, .. } => self.induction(expr),
            ExprKind::Binary { lhs, rhs, .. } => self.induction(lhs) && self.induction(rhs),
            ExprKind::Intrinsic { op, .. } => A::induction_intrinsic(op),
            _ => false,
        }
    }

    pub fn unsupported<T>(&self, construct: &str) -> Result<T, SelectionError> {
        Err(SelectionError::UnsupportedMapping(format!(
            "`{}`: {construct} has no {} mapping",
            self.bound.definition.name,
            A::NAME
        )))
    }

    fn invocation(&self) -> bool {
        self.root && self.depth == 0
    }

    fn scope(&self) -> Scope {
        Scope {
            multiplicity: self.multiplicity.clone(),
            pieces: self.pieces.clone(),
            invocation: self.invocation() && self.host_loops == 0,
            launch: self.launch,
            owners: self.owners.clone(),
        }
    }

    /// The owner of a tile declared by the statement being walked.
    fn tile_owner(&self) -> TileOwner {
        match (&self.owners, &self.owner_body) {
            (Some(owners), _) => TileOwner::Inner(owners.clone()),
            (None, Some((_, _, regions))) if !regions.is_empty() => TileOwner::Outer,
            _ => TileOwner::Piece,
        }
    }

    /// `q` repeated by the multiplicity of the statement being walked.
    pub fn scaled(&self, q: Quantity) -> Quantity {
        Quantity::product(self.multiplicity.iter().cloned().chain([q]))
    }

    pub fn ledger(&mut self) -> &mut A::Ledger {
        &mut self.account.ledger
    }

    pub fn ops(&mut self, q: Quantity) {
        let q = self.scaled(q);
        A::push_operations(&mut self.work, q);
    }

    fn elementwise(&mut self, ty: &Ty) {
        let n = self.bound.elements(ty);
        self.ops(A::share(n));
    }

    pub fn traffic(&mut self, external: bool, bits: Quantity) {
        let q = self.scaled(bits);
        A::push_traffic(&mut self.work, external, q);
    }

    /// Offer a tile to the backend's ledger, in the launch being walked.
    pub fn tile(&mut self, variable: Option<VarId>, shaped: &Shaped, snapshot: bool) {
        let owner = self.tile_owner();
        // A tile of the outer owner is published to the inner owners of its launch piece.
        if let (TileOwner::Outer, Some(_), false, Some((_, _, regions))) =
            (&owner, variable, snapshot, &self.owner_body)
        {
            if let Some(per_piece) = regions.first() {
                let completions = self.scaled(per_piece.clone());
                A::push_owner_completion(&mut self.work, completions);
            }
        }
        let event = TileEvent {
            bound: self.bound,
            variable,
            shaped,
            snapshot,
            pieces: &self.pieces,
            launch: self.launch,
            owner,
        };
        A::tile(&mut self.account.ledger, event);
    }

    fn block(&mut self, block: &Block) -> Result<(), SelectionError> {
        let mut run = false;
        for s in block {
            if self.invocation() && self.host_loops == 0 {
                let mut boundary = matches!(s.kind, StmtKind::Region(_) | StmtKind::Stages(_));
                each_expr(std::slice::from_ref(s), &mut |e| {
                    boundary |= matches!(e.kind, ExprKind::Region(_) | ExprKind::Call { .. })
                });
                if boundary {
                    run = false;
                } else if !matches!(s.kind, StmtKind::Yield(_) | StmtKind::Return(_)) && !run {
                    self.account.serial_launches += 1;
                    run = true;
                }
            }
            self.stmt(s)?;
        }
        Ok(())
    }

    fn looped(&mut self, statement: &Stmt, body: &Block) -> Result<(), SelectionError> {
        let trip = self
            .bound
            .trip::<A>(statement)
            .unwrap_or_else(|| Quantity::Unknown("loop without a trip count".into()));
        let host = self.invocation();
        self.host_loops += usize::from(host);
        let binders = match &statement.kind {
            StmtKind::Range { var, .. } | StmtKind::Members { var, .. } => vec![*var],
            StmtKind::Coordinates { vars, .. } => vars.clone(),
            _ => Vec::new(),
        };
        self.repetitions
            .push((Repeats::Vars(binders), trip.clone()));
        self.multiplicity.push(trip);
        self.control += 1;
        let mark = A::operations_mark(&self.work);
        self.carried.push(false);
        self.block(body)?;
        if self.carried.pop() == Some(true) {
            let trips = self.scaled(Quantity::one());
            A::carried_update(&mut self.work, mark, trips);
        }
        self.control -= 1;
        self.multiplicity.pop();
        self.repetitions.pop();
        self.host_loops -= usize::from(host);
        Ok(())
    }

    fn stmt(&mut self, s: &Stmt) -> Result<(), SelectionError> {
        match &s.kind {
            StmtKind::Bind { pattern, value } => {
                if let (Pattern::Var(v), true) = (pattern, self.induction(value)) {
                    if matches!(
                        self.bound.body.vars.get(*v).map(|var| &var.kind),
                        Some(seismic_lang::sir::VarKind::Value)
                    ) {
                        self.inductions.insert(*v);
                    }
                }
                self.expr(value)?;
                if let (Pattern::Var(v), Ty::Tile(shaped)) = (pattern, &value.ty) {
                    let snapshot =
                        matches!(&value.kind, ExprKind::Load(view) if external(&view.ty));
                    self.tile(Some(*v), shaped, snapshot);
                }
            }
            StmtKind::Assign { target, op, value } => {
                self.expr(value)?;
                self.place(target)?;
                if *op != AssignOp::Assign {
                    self.elementwise(&target.ty);
                }
                // A scalar update inside a loop whose value reads its own target is a
                // loop-carried chain: each trip of that loop waits for the previous one.
                // The place is the same on every trip: no binder of the innermost loop selects it.
                let innermost: &[VarId] = match self.repetitions.last() {
                    Some((Repeats::Vars(binders), _)) => binders,
                    _ => &[],
                };
                let mut same_place = !innermost.is_empty();
                each_expr_in(target, &mut |e| match &e.kind {
                    ExprKind::Var(v) => same_place &= !innermost.contains(v),
                    ExprKind::Index { indices, .. } => {
                        same_place &= !indices
                            .iter()
                            .any(|index| matches!(index, Index::Coord(v) if innermost.contains(v)))
                    }
                    _ => {}
                });
                if let (Some(variable), true, true) =
                    (tile_root(target), same_place, target.ty.shaped().is_none())
                {
                    let mut carried = *op != AssignOp::Assign;
                    each_expr_in(value, &mut |e| {
                        carried |= matches!(e.kind, ExprKind::Var(v) if v == variable)
                    });
                    if let (true, Some(innermost)) = (carried, self.carried.last_mut()) {
                        *innermost = true;
                    }
                }
            }
            StmtKind::Region(r) => self.region(r)?,
            StmtKind::Stages(stages) => {
                for stage in stages {
                    self.block(&stage.body)?;
                }
            }
            StmtKind::Range { lo, hi, body, .. } => {
                self.expr(lo)?;
                self.expr(hi)?;
                self.looped(s, body)?;
            }
            StmtKind::Coordinates {
                vars,
                of,
                axes,
                body,
            } => {
                self.expr(of)?;
                if of
                    .ty
                    .shaped()
                    .is_some_and(|shaped| axes.len() == shaped.rank())
                {
                    A::owned_loop(self, vars, body);
                }
                self.looped(s, body)?;
            }
            StmtKind::Members { body, .. } => self.looped(s, body)?,
            StmtKind::If { cond, then, els } => {
                self.expr(cond)?;
                let outer = std::mem::take(&mut self.work);
                self.control += 1;
                self.block(then)?;
                let taken = std::mem::take(&mut self.work);
                self.block(els)?;
                self.control -= 1;
                let other = std::mem::replace(&mut self.work, outer);
                A::maximum(&mut self.work, taken, other);
            }
            StmtKind::Publish { value, destination } => {
                self.expr(value)?;
                self.place(destination)?;
                self.elementwise(&value.ty);
                if let Some(shaped) = destination.ty.shaped() {
                    let bits = Quantity::product([
                        self.bound.elements(&value.ty),
                        self.bound.bits(&shaped.elem),
                    ]);
                    self.traffic(true, bits);
                }
            }
            StmtKind::Yield(items) | StmtKind::Return(items) => {
                for item in items {
                    self.expr(item)?;
                }
            }
            StmtKind::Expr(e) => self.expr(e)?,
        }
        Ok(())
    }

    /// An assignment or publication target: index expressions execute; a scalar element of
    /// external storage is a device write.
    fn place(&mut self, target: &Expr) -> Result<(), SelectionError> {
        if let ExprKind::Index { base, indices } = &target.kind {
            self.indices(indices)?;
            if target.ty.shaped().is_none() {
                if let (true, Some(shaped)) = (
                    external(&base.ty) || A::TILES_ARE_MEMORY_TRAFFIC,
                    base.ty.shaped(),
                ) {
                    let bits = self.bound.bits(&shaped.elem);
                    self.traffic(external(&base.ty), bits);
                }
            }
            return self.place(base);
        }
        Ok(())
    }

    fn indices(&mut self, indices: &[Index]) -> Result<(), SelectionError> {
        for index in indices {
            match index {
                Index::Point(p) => self.expr(p)?,
                Index::Range { start, end } => {
                    for bound in [start, end].into_iter().flatten() {
                        self.expr(bound)?;
                    }
                }
                Index::Coord(_) | Index::Slice(_) => {}
            }
        }
        Ok(())
    }

    /// How often distinct bytes of external `view` are read here: the product of the enclosing
    /// repetitions whose binder the view expression mentions. A view that mentions any other
    /// non-parameter variable, or a repetition inherited from a caller, counts in full.
    pub fn distinct_reads(&self, view: &Expr) -> Quantity {
        use seismic_lang::sir::VarKind;
        let (mut slices, mut vars, mut opaque) = (BTreeSet::new(), BTreeSet::new(), false);
        each_expr_in(view, &mut |e| match &e.kind {
            ExprKind::Var(v) => match self.bound.body.vars.get(*v).map(|var| &var.kind) {
                Some(VarKind::Param(_)) => {}
                Some(VarKind::Slice(slice)) => {
                    slices.insert(*slice);
                }
                Some(VarKind::SliceMember(slice)) => {
                    // A member coordinate differs between the visits of its slice.
                    slices.insert(*slice);
                    vars.insert(*v);
                }
                Some(VarKind::RangeIndex | VarKind::Coordinate) => {
                    vars.insert(*v);
                }
                _ => opaque = true,
            },
            ExprKind::Index { indices, .. } => {
                for index in indices {
                    match index {
                        Index::Slice(slice) => {
                            slices.insert(*slice);
                        }
                        Index::Coord(v) => {
                            vars.insert(*v);
                        }
                        Index::Point(_) | Index::Range { .. } => {}
                    }
                }
            }
            _ => {}
        });
        let rebound = |slice: &SliceId| {
            let mut current = *slice;
            for _ in 0..=self.bound.body.slices.len() {
                if slices.contains(&current) {
                    return true;
                }
                match self
                    .bound
                    .body
                    .slices
                    .get(current.0 as usize)
                    .map(|d| &d.parent)
                {
                    Some(SliceParent::Rebind(parent) | SliceParent::Refine(parent)) => {
                        current = *parent
                    }
                    _ => return false,
                }
            }
            false
        };
        Quantity::product(
            self.repetitions
                .iter()
                .filter(|(tag, _)| match tag {
                    _ if opaque => true,
                    Repeats::Inherited => true,
                    Repeats::Slice(slice) => {
                        slices.contains(slice)
                            || rebound(slice)
                            || slices.iter().any(|s| self.refines(*s, *slice))
                    }
                    Repeats::Vars(binders) => binders.iter().any(|b| vars.contains(b)),
                })
                .map(|(_, q)| q.clone()),
        )
    }

    /// Whether `slice` is a refinement or rebinding (transitively) of `ancestor`.
    fn refines(&self, slice: SliceId, ancestor: SliceId) -> bool {
        let mut current = slice;
        for _ in 0..=self.bound.body.slices.len() {
            if current == ancestor {
                return true;
            }
            match self
                .bound
                .body
                .slices
                .get(current.0 as usize)
                .map(|d| &d.parent)
            {
                Some(SliceParent::Rebind(parent) | SliceParent::Refine(parent)) => {
                    current = *parent
                }
                _ => return false,
            }
        }
        false
    }

    /// Read-and-convert of a whole operand into a tile. Bytes of external storage cross the
    /// device bus once per distinct view (`distinct_reads`). Further visits of the same view
    /// are served by the device's caches and cost only the lane operations that consume them
    /// (charged above); a repeated view too large for those caches is underpriced by this rule.
    fn transfer(&mut self, result: &Ty, operand: &Expr) {
        self.elementwise(result);
        if let (true, Some(shaped)) = (A::TILES_ARE_MEMORY_TRAFFIC, result.shaped()) {
            let written = self.bound.tile_bits(shaped);
            self.traffic(false, written);
        }
        if let Some(shaped) = operand.ty.shaped() {
            let bits = self
                .bound
                .stored_bits(self.bound.elements(result), &shaped.elem);
            if external(&operand.ty) {
                let distinct = Quantity::product([self.distinct_reads(operand), bits]);
                A::push_traffic(&mut self.work, true, distinct);
                return;
            }
            self.traffic(false, bits);
        }
    }

    pub fn expr(&mut self, e: &Expr) -> Result<(), SelectionError> {
        if self.induction(e) {
            return Ok(());
        }
        match &e.kind {
            ExprKind::Int(_)
            | ExprKind::Float(_)
            | ExprKind::Bool(_)
            | ExprKind::Var(_)
            | ExprKind::ShapeParam(_)
            | ExprKind::TileAlloc
            | ExprKind::CoordOf(_) => {}
            ExprKind::Tuple(items) => {
                for item in items {
                    self.expr(item)?;
                }
            }
            ExprKind::Range { lo, hi } => {
                self.expr(lo)?;
                self.expr(hi)?;
            }
            ExprKind::Field { base, .. }
            | ExprKind::Member { result: base, .. }
            | ExprKind::Transpose(base)
            | ExprKind::Reshape { base, .. }
            | ExprKind::ExtentOf { base, .. }
            | ExprKind::Accessor { base, .. }
            | ExprKind::Geometry { base, .. } => self.expr(base)?,
            ExprKind::Filled { like, .. } => {
                self.expr(like)?;
                self.elementwise(&e.ty);
            }
            ExprKind::Index { base, indices } => {
                self.expr(base)?;
                self.indices(indices)?;
                if e.ty.shaped().is_none() {
                    if let Some(shaped) = base.ty.shaped() {
                        if let Some(decode) = A::packed_element(self.bound, shaped) {
                            self.ops(decode);
                        }
                        if external(&base.ty) || A::TILES_ARE_MEMORY_TRAFFIC {
                            let bits = self.bound.bits(&shaped.elem);
                            self.traffic(external(&base.ty), bits);
                        }
                    }
                }
            }
            ExprKind::Load(view) | ExprKind::Decode(view) => {
                self.expr(view)?;
                if let (ExprKind::Load(_), true, Some(shaped)) =
                    (&e.kind, external(&view.ty), e.ty.shaped())
                {
                    self.tile(None, shaped, true);
                }
                self.transfer(&e.ty, view);
            }
            ExprKind::Cast { expr: operand, .. } => {
                self.expr(operand)?;
                if e.ty.shaped().is_some() {
                    self.transfer(&e.ty, operand)
                } else {
                    self.ops(Quantity::one())
                }
            }
            ExprKind::Unary { expr: operand, .. } => {
                self.expr(operand)?;
                self.elementwise(&e.ty);
            }
            ExprKind::Binary { lhs, rhs, .. } => {
                self.expr(lhs)?;
                self.expr(rhs)?;
                self.elementwise(&e.ty);
            }
            ExprKind::Math { args, .. } => {
                for arg in args {
                    self.expr(arg)?;
                }
                self.elementwise(&e.ty);
            }
            ExprKind::Select { cond, then, els } => {
                self.expr(cond)?;
                self.expr(then)?;
                self.expr(els)?;
                self.elementwise(&e.ty);
            }
            ExprKind::Reduce {
                value,
                axis,
                unordered,
                ..
            } => {
                self.expr(value)?;
                let folded = A::reduction(self.bound, value, *axis, *unordered, &e.ty);
                self.ops(folded);
            }
            ExprKind::Call { call, args } => {
                // Calls are logical composition, not native launches.  Preserve the
                // enclosing loop multiplicity and non-invocation scope for the callee;
                // instantiation inlines the selected body into that scope.  Rejecting
                // this shape made ordinary helpers (for example a reduction used by
                // each row of a normalization) impossible to express.
                self.account.calls.insert(*call, self.scope());
                for (ordinal, arg) in args.iter().enumerate() {
                    A::call_argument(self, *call, ordinal, arg)?;
                }
            }
            ExprKind::Region(r) => self.region(r)?,
            ExprKind::Intrinsic { op, args } => {
                for arg in args {
                    self.expr(arg)?;
                }
                A::intrinsic(self, op, args)?;
            }
            ExprKind::Atomic { place, value, .. } => {
                if !A::ATOMIC {
                    return self.unsupported("an atomic update");
                }
                self.expr(value)?;
                self.place(place)?;
                self.ops(Quantity::one());
            }
        }
        Ok(())
    }

    fn region(&mut self, r: &Region) -> Result<(), SelectionError> {
        let launch = self.invocation();
        if launch {
            if self.host_loops > 0 {
                return self.unsupported("a region inside an invocation-scope loop");
            }
            if r.result.is_some() && r.merge.is_none() {
                return self.unsupported("a region result crossing launches");
            }
            if matches!(r.source, RegionSource::Results(_)) {
                return self.unsupported("a root traversal of region results");
            }
        }
        if let RegionSource::Results(source) = &r.source {
            self.expr(source)?;
        }
        let bound = self.bound;
        let Some(declared) = bound.body.regions.get(r.id.0 as usize) else {
            return Err(SelectionError::Reconstruction(format!(
                "`{}`: region#{} is undeclared",
                bound.definition.name, r.id.0
            )));
        };
        let visits = Quantity::product(declared.binders.iter().map(|b| bound.visits(*b)));
        let outer = launch.then(|| std::mem::take(&mut self.work));
        if launch {
            self.launch = Some((self.candidate, self.account.launches.len()));
        }
        let enclosing_owner_body = self.owner_body.clone();
        if launch && r.mode == RegionMode::Parallel {
            self.pieces = visits.clone();
            // The inner owner regions of this launch, known before its owner body is walked:
            // they decide where the owner's tiles live.
            let mut regions = Vec::new();
            if A::INNER_OWNERS {
                inner_owner_regions(bound, &declared.binders, &r.body, &mut regions);
            }
            self.owner_body = Some((declared.binders.clone(), self.control, regions));
        }
        // An inner owner region of the launch being walked (`instantiate::inner_owner_region`
        // plus its position: the owner body, outside loops, branches and inner owner regions).
        let owners = match &self.owner_body {
            Some((binders, control, _))
                if A::INNER_OWNERS
                    && !launch
                    && self.owners.is_none()
                    && *control == self.control
                    && seismic_lang::instantiate::inner_owner_region(bound.body, binders, r) =>
            {
                Some(Quantity::product(
                    declared
                        .binders
                        .iter()
                        .filter_map(|b| bound.refinement_pieces(*b)),
                ))
            }
            _ => None,
        };
        if let Some(per_piece) = &owners {
            // Every inner owner of the launch piece meets at the region's completion.
            let completions = self.scaled(per_piece.clone());
            A::push_owner_completion(&mut self.work, completions);
        }
        // A refinement partitions the visits of the binder it refines: inside it that binder
        // repeats nothing, and the refinement's visits are those of the whole refined domain.
        let mut partitioned = Vec::new();
        for binder in &declared.binders {
            let refined = bound.refined_site(*binder).and_then(|site| {
                self.repetitions.iter().rposition(|(tag, _)| matches!(tag, Repeats::Slice(slice) if bound.width_site(*slice).is_some_and(|(id, _)| id == site)))
            });
            let visits = match (refined, bound.refinement_pieces(*binder)) {
                (Some(position), _) => {
                    let factor =
                        std::mem::replace(&mut self.multiplicity[position], Quantity::one());
                    let repetition =
                        std::mem::replace(&mut self.repetitions[position].1, Quantity::one());
                    partitioned.push((position, factor, repetition));
                    bound.visits(*binder)
                }
                // The refined binder repeats outside this body's view: pieces per refined piece.
                (None, Some(per_piece)) => per_piece,
                (None, None) => bound.visits(*binder),
            };
            self.multiplicity.push(visits.clone());
            self.repetitions.push((Repeats::Slice(*binder), visits));
        }
        let enclosing = owners.as_ref().map(|per_piece| {
            let pieces = Quantity::product([self.pieces.clone(), per_piece.clone()]);
            (
                std::mem::replace(&mut self.pieces, pieces),
                self.owners.replace(per_piece.clone()),
            )
        });
        self.depth += 1;
        let visit = self.scaled(Quantity::one());
        A::push_visit(&mut self.work, r.mode, visit);
        self.account.regions.insert(r.id, self.scope());
        self.block(&r.body)?;
        if let Some(merge) = &r.merge {
            self.expr(&merge.identity)?;
            self.block(&merge.body)?;
        }
        self.depth -= 1;
        if let Some((pieces, owners)) = enclosing {
            self.pieces = pieces;
            self.owners = owners;
        }
        self.multiplicity
            .truncate(self.multiplicity.len() - declared.binders.len());
        self.repetitions
            .truncate(self.repetitions.len() - declared.binders.len());
        for (position, factor, repetition) in partitioned.into_iter().rev() {
            self.multiplicity[position] = factor;
            self.repetitions[position].1 = repetition;
        }
        if let Some(outer) = outer {
            let work = std::mem::replace(&mut self.work, outer);
            let groups = std::mem::replace(&mut self.pieces, Quantity::one());
            let owners = match std::mem::replace(&mut self.owner_body, enclosing_owner_body) {
                Some((_, _, regions)) if r.mode == RegionMode::Parallel => regions,
                _ => Vec::new(),
            };
            let pieces = match owners.first() {
                Some(per_piece) => Quantity::product([groups.clone(), per_piece.clone()]),
                None => groups.clone(),
            };
            self.launch = None;
            self.account.launches.push(Launch {
                region: r.id,
                mode: r.mode,
                merge: r.merge.is_some(),
                binders: declared
                    .binders
                    .iter()
                    .filter_map(|b| bound.width_site(*b))
                    .collect(),
                binder_count: declared.binders.len(),
                pieces,
                groups,
                owners,
                work,
            });
        }
        Ok(())
    }
}

/// Accounts of every candidate of the family, parents before children.
pub struct Analysis<'a, A: Accounting> {
    pub family: &'a Family,
    pub bounds: BTreeMap<CandidateRef, Bound<'a>>,
    pub accounts: BTreeMap<CandidateRef, Account<A>>,
}

impl<'a, A: Accounting> Analysis<'a, A> {
    pub fn new(
        program: &'a Program,
        family: &'a Family,
        target: &str,
    ) -> Result<Self, SelectionError> {
        let mut pending: Vec<CandidateRef> = family
            .occurrences
            .iter()
            .flat_map(|o| {
                (0..o.candidates.len() as u32).map(move |candidate| CandidateRef {
                    occurrence: o.id,
                    candidate,
                })
            })
            .collect();
        let mut analysis = Analysis {
            family,
            bounds: BTreeMap::new(),
            accounts: BTreeMap::new(),
        };
        while !pending.is_empty() {
            let before = pending.len();
            let mut later = Vec::new();
            for candidate in pending {
                let occurrence = family.occurrence(candidate.occurrence);
                let mut borrowed = A::Work::default();
                let context = match (occurrence.parent, occurrence.call) {
                    (None, _) => Context {
                        scope: Scope {
                            multiplicity: Vec::new(),
                            pieces: Quantity::one(),
                            invocation: true,
                            launch: None,
                            owners: None,
                        },
                        guards: vec![candidate],
                    },
                    (Some(parent), Some(call)) => {
                        let Some(account) = analysis.accounts.get(&parent) else {
                            later.push(candidate);
                            continue;
                        };
                        let scope = account.calls.get(&call).cloned().ok_or_else(|| {
                            SelectionError::Reconstruction(format!(
                                "occurrence {} names call#{} absent from its parent body",
                                occurrence.id.0, call.0
                            ))
                        })?;
                        borrowed = A::inherited(&account.ledger, call);
                        Context {
                            scope,
                            guards: account
                                .context
                                .guards
                                .iter()
                                .copied()
                                .chain([candidate])
                                .collect(),
                        }
                    }
                    (Some(_), None) => {
                        return Err(SelectionError::Reconstruction(format!(
                            "occurrence {} has a parent but no call",
                            occurrence.id.0
                        )))
                    }
                };
                let bound = Bound::new(
                    program,
                    family,
                    candidate,
                    occurrence.parent.and_then(|p| analysis.bounds.get(&p)),
                )?;
                if let Some(required) = bound.definition.kind.target().filter(|t| *t != target) {
                    return Err(SelectionError::UnsupportedMapping(format!(
                        "`{}`: primitives of target `{required}` have no {} mapping",
                        bound.definition.name,
                        A::NAME
                    )));
                }
                let mut walker: Walker<'_, '_, A> = Walker {
                    bound: &bound,
                    candidate,
                    account: Account {
                        context: context.clone(),
                        launches: Vec::new(),
                        rest: A::Work::default(),
                        serial_launches: 0,
                        calls: BTreeMap::new(),
                        regions: BTreeMap::new(),
                        ledger: A::Ledger::default(),
                    },
                    work: borrowed,
                    multiplicity: context.scope.multiplicity.clone(),
                    pieces: context.scope.pieces.clone(),
                    launch: context.scope.launch,
                    root: context.scope.invocation,
                    depth: 0,
                    host_loops: 0,
                    inductions: BTreeSet::new(),
                    repetitions: context
                        .scope
                        .multiplicity
                        .iter()
                        .map(|q| (Repeats::Inherited, q.clone()))
                        .collect(),
                    control: 0,
                    carried: Vec::new(),
                    owner_body: None,
                    owners: context.scope.owners.clone(),
                };
                walker.block(&bound.body.block)?;
                let mut account = walker.account;
                account.rest = walker.work;
                analysis.accounts.insert(candidate, account);
                analysis.bounds.insert(candidate, bound);
            }
            if later.len() == before {
                return Err(SelectionError::Reconstruction(
                    "occurrence parents form a cycle".into(),
                ));
            }
            pending = later;
        }
        Ok(analysis)
    }

    /// The block of `sequence` inside its owner's body and the scope it executes in.
    pub fn sequence_block(&self, sequence: &Sequence) -> Result<(&'a Block, Scope), String> {
        let (bound, account) = self
            .bounds
            .get(&sequence.owner)
            .zip(self.accounts.get(&sequence.owner))
            .ok_or("sequence owner has no account")?;
        let mut block = &bound.body.block;
        let mut scope = account.context.scope.clone();
        for step in &sequence.scope {
            let lost = || {
                format!(
                    "sequence {} of `{}` names a scope step absent from the body",
                    sequence.id.0, bound.definition.name
                )
            };
            match step {
                ScopeStep::Region(id) => {
                    block = &find_region(&bound.body.block, *id).ok_or_else(lost)?.body;
                    scope = account.regions.get(id).cloned().ok_or_else(lost)?;
                }
                ScopeStep::Stage(ordinal) => {
                    let stage = block
                        .iter()
                        .filter_map(|s| match &s.kind {
                            StmtKind::Stages(stages) => Some(stages),
                            _ => None,
                        })
                        .flatten()
                        .nth(*ordinal);
                    block = &stage.ok_or_else(lost)?.body;
                }
                ScopeStep::Then(ordinal) | ScopeStep::Else(ordinal) => {
                    match block.get(*ordinal).map(|s| &s.kind) {
                        Some(StmtKind::If { then, els, .. }) => {
                            block = if matches!(step, ScopeStep::Then(_)) {
                                then
                            } else {
                                els
                            }
                        }
                        _ => return Err(lost()),
                    }
                }
                ScopeStep::Loop(ordinal) => {
                    let statement = block.get(*ordinal).ok_or_else(lost)?;
                    let (StmtKind::Range { body, .. }
                    | StmtKind::Coordinates { body, .. }
                    | StmtKind::Members { body, .. }) = &statement.kind
                    else {
                        return Err(lost());
                    };
                    scope
                        .multiplicity
                        .push(bound.trip::<A>(statement).ok_or_else(lost)?);
                    scope.invocation = false;
                    block = body;
                }
            }
        }
        Ok((block, scope))
    }

    /// Output tile of an elementwise unit: `(variable, stored bits)`.
    pub fn unit_output(
        &self,
        owner: CandidateRef,
        block: &Block,
        unit: &Unit,
    ) -> Result<(Option<VarId>, Quantity), String> {
        let bound = self.bounds.get(&owner).ok_or("unit owner has no account")?;
        let statement = unit
            .statements
            .end
            .checked_sub(1)
            .and_then(|last| block.get(last))
            .ok_or("unit covers no statement of its block")?;
        let (variable, ty) = match &statement.kind {
            StmtKind::Bind {
                pattern: Pattern::Var(v),
                value,
            } => (Some(*v), &value.ty),
            StmtKind::Assign { target, .. } => (tile_root(target), &target.ty),
            _ => {
                return Err(format!(
                    "elementwise unit of `{}` has no tile-valued producer",
                    bound.definition.name
                ))
            }
        };
        match ty {
            Ty::Tile(shaped) => Ok((variable, bound.tile_bits(shaped))),
            other => Err(format!(
                "elementwise unit of `{}` produces {other}, not a tile",
                bound.definition.name
            )),
        }
    }
}

/// Whether `variable` is referenced by `statements`.
pub fn referenced(statements: &[Stmt], variable: VarId) -> bool {
    let mut found = false;
    each_expr(statements, &mut |e| {
        found |= matches!(e.kind, ExprKind::Var(v) if v == variable)
    });
    found
}
