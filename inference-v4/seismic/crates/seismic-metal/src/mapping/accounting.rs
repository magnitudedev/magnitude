//! Metal's accounting over the shared structural walk (`seismic_compiler::selection::structure`):
//! the work classes the estimate prices, the lane-share and reduction rules that mirror
//! `realize`, the tile ledger the memory limits read, and the intrinsic table.
//!
//!   root `parallel`            -> one launch dispatched over pieces (one subgroup per piece)
//!   root `ordered`/`pipeline`  -> one single-piece launch with serial windows
//!   `pipeline`                 -> synchronous same-participant, ring depth 1 (plan R8)
//!   invocation-scope serial statements -> one single-thread launch per run (plan R7)
use crate::execution::SUBGROUP;
use seismic_compiler::selection::quantity::{Quantity, Rule};
use seismic_compiler::selection::structure::{each_expr, external, tile_root, Accounting, Bound, TileEvent, TileOwner, Walker};
use seismic_compiler::selection::SelectionError;
use seismic_lang::family::CandidateRef;
use seismic_lang::intrinsics::Operation;
use seismic_lang::sir::{Block, CallId, Expr, ExprKind, Index, VarId};
use seismic_lang::syntax::ast::RegionMode;
use seismic_lang::types::{DType, Elem, Shaped, Ty};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

/// Element operations one `simd_shuffle` step of an ordered fold over a lane-distributed tile
/// stands for: a 2560-element fold measured 0.1 ms (39 ns per element) on Apple M4 Max.
pub(crate) const FOLD_SHUFFLE_OPS: u64 = 24;

/// Element operations one subgroup collective (`simd_sum`, `simd_max`, `simd_min`) stands for:
/// a loop of `simd_sum` measured 24.6 ns per collective on Apple M4 Max.
pub(crate) const COLLECTIVE_OPS: u64 = 16;

/// Element operations one step of a loop-carried scalar update stands for at least. On Apple
/// M4 Max a standalone serial FMA chain runs 12.8 ns per dependent step, and the emitted
/// ordered contraction over staged f32 operands (128 x 9216 x 2560, every lane busy) 16 ns
/// per step: 10 operations at the lane rate.
pub(crate) const CARRIED_STEP_OPS: u64 = 10;

/// Element operations of decoding one packed element at its consumption. Code: word load,
/// shift, mask, integer conversion. Coefficients: two loads with their conversions.
const PACKED_CODE_OPS: u64 = 4;
const PACKED_COEFFICIENT_OPS: u64 = 4;

fn rule(name: &'static str, arguments: Vec<Quantity>, apply: impl Fn(&[u64]) -> Result<u64, String> + Send + Sync + 'static) -> Quantity {
    Quantity::Rule(Rule { name, apply: Arc::new(apply) }, arguments)
}

fn arity(name: &str) -> String {
    format!("rule `{name}` was given the wrong number of quantities")
}

/// Critical-path share of elementwise work over `n` elements inside one piece: tiles of at
/// least one subgroup of elements are lane-distributed (`ceil(n / 32)`), smaller ones
/// replicated (`n`). Mirrors the storage rule in `realize`.
pub(crate) fn lane_share(elements: Quantity) -> Quantity {
    rule("lane_share", vec![elements], |values| match values {
        [n] => Ok(if *n >= SUBGROUP as u64 { n.div_ceil(SUBGROUP as u64) } else { *n }),
        _ => Err(arity("lane_share")),
    })
}

/// Critical-path element operations of one reduction of `elements` values into `outputs`
/// results (`inner` elements follow the reduced axis), under the storage and reduction
/// rules of `realize`: an input below one subgroup of elements is replicated and folded whole
/// by every lane; a distributed input is folded lane-locally when the outputs exceed one
/// subgroup and `inner` is a whole number of subgroups; else, when the contract permits
/// reassociation, each lane folds its share and one subgroup collective per output
/// combines the partials; else every element reaches the folding lane through a shuffle.
fn fold(elements: Quantity, outputs: Quantity, inner: Quantity, reassociable: bool) -> Quantity {
    rule("fold", vec![elements, outputs, inner], move |values| {
        let [n, outputs, inner] = values else { return Err(arity("fold")) };
        let (n, outputs, inner, lanes) = (*n, *outputs, *inner, SUBGROUP as u64);
        let overflow = || "derived quantity overflows u64".to_string();
        if n < lanes {
            Ok(n)
        } else if outputs > lanes && inner > 0 && inner % lanes == 0 {
            Ok(n.div_ceil(lanes))
        } else if reassociable {
            outputs.checked_mul(COLLECTIVE_OPS).and_then(|c| c.checked_add(n.div_ceil(lanes))).ok_or_else(overflow)
        } else {
            n.checked_mul(FOLD_SHUFFLE_OPS).ok_or_else(overflow)
        }
    })
}

/// Work totals of one execution scope, each a sum of terms already scaled by multiplicity.
#[derive(Clone, Debug, Default)]
pub(crate) struct Work {
    /// Element operations on the critical path of a piece (see `lane_share`).
    pub lane_ops: Vec<Quantity>,
    /// Ordered/pipeline window visits.
    pub visits: Vec<Quantity>,
    /// Cooperative 8x8 matrix multiply-accumulates of the piece's subgroup.
    pub matrix_multiplies: Vec<Quantity>,
    /// Matrix loads and stores against threadgroup memory (each with its barrier).
    pub matrix_transfers: Vec<Quantity>,
    /// Bits read from or written to device buffers.
    pub device_bits: Vec<Quantity>,
    /// Bits moved through tile storage (threadgroup or thread-private).
    pub local_bits: Vec<Quantity>,
    /// Matrix loads of an operand that is resident where the atom reads it: a parameter tile
    /// staged by the caller or external storage, with no staging write and barrier of its own.
    pub resident_loads: Vec<Quantity>,
    /// Threadgroup barriers, one count per SIMD group that waits in it: publications of the
    /// outer owner's tiles and completions of inner owner regions.
    pub owner_completions: Vec<Quantity>,
}

impl Work {
    pub fn quantities(&self) -> impl Iterator<Item = &Quantity> {
        self.lane_ops.iter().chain(&self.visits).chain(&self.matrix_multiplies).chain(&self.matrix_transfers).chain(&self.device_bits).chain(&self.local_bits).chain(&self.resident_loads).chain(&self.owner_completions)
    }
}

#[derive(Clone, Debug)]
pub(crate) struct Tile {
    /// Declared storage in bits at the native element type (threadgroup operand placement).
    pub bits: Quantity,
    /// Declared bits per lane under private placement, at the widened local element type.
    pub private_bits: Quantity,
    /// Declared bits per lane when every lane holds the whole tile.
    pub replicated_bits: Quantity,
    /// A snapshot of external storage: the load rule borrows it wherever the borrow proof
    /// holds, and a borrowed snapshot declares no array.
    pub snapshot: bool,
    pub launch: Option<(CandidateRef, usize)>,
    pub pieces: Quantity,
    /// Whose tile it is in a launch with inner owner regions (structure rule): the outer
    /// owner's tiles are threadgroup-placed once per threadgroup; an inner owner's tile exists
    /// once per SIMD group of the threadgroup.
    pub owner: TileOwner,
}

#[derive(Clone, Debug, Default)]
pub(crate) struct Ledger {
    /// Device and re-read traffic of borrowed external snapshots passed to each call, already
    /// scaled by the call's multiplicity; charged in the selected callee's body scope.
    pub call_traffic: BTreeMap<CallId, Work>,
    /// `(call, argument ordinal, root variable)` of tile arguments.
    pub call_arguments: Vec<(CallId, usize, VarId)>,
    pub tiles: BTreeMap<VarId, Tile>,
    /// Roots of tiles used as matrix-intrinsic operands: threadgroup-placed by the storage rule.
    pub operands: BTreeSet<VarId>,
    /// Roots of tiles of at least one subgroup of elements that an owned loop reads at foreign
    /// coordinates (`owned_loop`): threadgroup-placed by the storage rule unless they are
    /// snapshots of external storage, which the load rule is taken to borrow.
    pub cross_reads: BTreeSet<VarId>,
}

impl Ledger {
    /// Roots the storage rule places in threadgroup memory. A parameter is included: its
    /// placement is decided at the caller's tile.
    pub fn placed(&self) -> BTreeSet<VarId> {
        let borrowed = |v: &VarId| self.tiles.get(v).is_some_and(|tile| tile.snapshot);
        // Every tile of the outer owner of a launch with inner owner regions is staged once
        // per threadgroup for its SIMD groups.
        let outer = self.tiles.iter().filter(|(_, tile)| matches!(tile.owner, TileOwner::Outer) && !tile.snapshot).map(|(v, _)| *v);
        self.operands.iter().copied().chain(self.cross_reads.iter().copied().filter(|v| !borrowed(v))).chain(outer).collect()
    }

    /// Threadgroup bits of a placed tile per threadgroup: a matrix operand keeps its native
    /// element type, any other threadgroup array holds the widened local type; an inner
    /// owner's tile exists once per SIMD group of the threadgroup.
    pub fn placed_bits(&self, root: VarId) -> Option<Quantity> {
        let tile = self.tiles.get(&root)?;
        let bits = if self.operands.contains(&root) { tile.bits.clone() } else { tile.replicated_bits.clone() };
        Some(match &tile.owner {
            TileOwner::Inner(per_group) => Quantity::product([bits, per_group.clone()]),
            TileOwner::Piece | TileOwner::Outer => bits,
        })
    }
}

/// Bits per value as held in a private array: half-width floats (dense elements and
/// packed coefficient planes) are widened (`dispatch::local_storage_dtype`).
fn held_bits(bound: &Bound<'_>, elem: &Elem) -> Quantity {
    let widened = |d: DType| u64::from(seismic_realization::dispatch::local_storage_dtype(d, false).bytes()) * 8;
    match bound.element(elem) {
        Some(Elem::Dtype(d)) => Quantity::Constant(widened(*d)),
        Some(Elem::Repr(name)) => match seismic_lang::repr::lookup(name) {
            Some(repr) => {
                let native = repr.bits_per_value();
                let planes: f64 = repr.planes().iter().map(|p| {
                    let ratio = widened(p.dtype()) as f64 / (u64::from(p.dtype().bytes()) * 8) as f64;
                    if matches!(p.dtype(), DType::BF16 | DType::F16) { p.entry_bits() as f64 * p.fields as f64 / p.group as f64 * (ratio - 1.0) } else { 0.0 }
                }).sum();
                Quantity::Constant((native + planes).ceil() as u64)
            }
            None => Quantity::Unknown(format!("representation `{name}` is not registered")),
        },
        Some(Elem::Param(_)) | None => Quantity::Unknown(format!("element parameter `{elem}` of `{}` is unbound", bound.definition.name)),
    }
}

pub(crate) struct MetalAccounting;

impl Accounting for MetalAccounting {
    const NAME: &'static str = "Metal";
    /// An inner owner region of a launch is one SIMD group per visit inside the threadgroup of
    /// the launch piece (`seismic_lang::instantiate::owner_regions`).
    const INNER_OWNERS: bool = true;
    type Work = Work;
    type Ledger = Ledger;

    fn maximum(into: &mut Work, then: Work, els: Work) {
        let join = |into: &mut Vec<Quantity>, a: Vec<Quantity>, b: Vec<Quantity>| {
            if !(a.is_empty() && b.is_empty()) {
                into.push(Quantity::Max(vec![Quantity::Sum(a), Quantity::Sum(b)]));
            }
        };
        join(&mut into.lane_ops, then.lane_ops, els.lane_ops);
        join(&mut into.visits, then.visits, els.visits);
        join(&mut into.matrix_multiplies, then.matrix_multiplies, els.matrix_multiplies);
        join(&mut into.matrix_transfers, then.matrix_transfers, els.matrix_transfers);
        join(&mut into.device_bits, then.device_bits, els.device_bits);
        join(&mut into.local_bits, then.local_bits, els.local_bits);
        join(&mut into.resident_loads, then.resident_loads, els.resident_loads);
        join(&mut into.owner_completions, then.owner_completions, els.owner_completions);
    }

    fn quantities(work: &Work) -> Vec<&Quantity> {
        work.quantities().collect()
    }

    fn push_operations(work: &mut Work, scaled: Quantity) {
        work.lane_ops.push(scaled);
    }

    /// Only ordered and pipeline windows pay loop and window bookkeeping.
    fn push_visit(work: &mut Work, mode: RegionMode, scaled: Quantity) {
        if mode != RegionMode::Parallel {
            work.visits.push(scaled);
        }
    }

    fn push_traffic(work: &mut Work, external: bool, scaled: Quantity) {
        if external { work.device_bits.push(scaled) } else { work.local_bits.push(scaled) }
    }

    fn push_owner_completion(work: &mut Work, scaled: Quantity) {
        work.owner_completions.push(scaled);
    }

    fn operations_mark(work: &Work) -> usize {
        work.lane_ops.len()
    }

    /// A loop-carried scalar update is latency-bound on a lane: each step waits for the
    /// previous one, however few element operations it counts.
    fn carried_update(work: &mut Work, mark: usize, steps: Quantity) {
        let counted: Vec<Quantity> = work.lane_ops.drain(mark.min(work.lane_ops.len())..).collect();
        work.lane_ops.push(Quantity::Max(vec![Quantity::Sum(counted), Quantity::product([steps, Quantity::Constant(CARRIED_STEP_OPS)])]));
    }

    fn share(elements: Quantity) -> Quantity {
        lane_share(elements)
    }

    /// `owned` over every axis of a tile is divided among the lanes of the piece: the storage
    /// rule makes every tile of at least one subgroup of elements cooperative (lane-distributed,
    /// or threadgroup-shared when other lanes read it), partitioned by its capacity, so a
    /// runtime extent is shared like a fixed one (charged at its static bound).
    fn coordinates_trip(_bound: &Bound<'_>, shaped: &Shaped, axes: &[usize], count: Quantity) -> Quantity {
        if axes.len() == shaped.rank() { lane_share(count) } else { count }
    }

    /// The coefficients of a value whose packed axis lies within one group are invariant over
    /// the whole value; the native compiler hoists their loads (the emitted group coordinate
    /// is exact).
    fn packed_element(bound: &Bound<'_>, shaped: &Shaped) -> Option<Quantity> {
        let group = bound.packed_group(&shaped.elem)?;
        let span = shaped.axes.get(shaped.packed_axis.unwrap_or(shaped.rank().saturating_sub(1))).map(|extent| bound.axis(extent));
        Some(match span {
            Some(span) => Quantity::exceeds(span, group, PACKED_CODE_OPS + PACKED_COEFFICIENT_OPS, PACKED_CODE_OPS),
            None => Quantity::Constant(PACKED_CODE_OPS + PACKED_COEFFICIENT_OPS),
        })
    }

    /// Ordered fold: every element is on the critical path. The operand of a fold is a tile;
    /// when that tile is lane-distributed (the storage rule `lane_share` mirrors), each
    /// element reaches the folding lane through one subgroup shuffle.
    fn reduction(bound: &Bound<'_>, value: &Expr, axis: usize, unordered: bool, result: &Ty) -> Quantity {
        let n = bound.elements(&value.ty);
        match value.ty.shaped().filter(|shaped| shaped.axes.iter().all(|a| bound.fixed(a))) {
            // A runtime-extent tile is never lane-distributed: every lane folds it whole.
            None => n,
            Some(shaped) => {
                // The collective is qualified for 32-bit elements only (`ReductionDomain`).
                let wide = matches!(bound.dtype(&shaped.elem), Some(DType::F32 | DType::I32 | DType::U32));
                let inner = Quantity::product(shaped.axes.iter().skip(axis + 1).map(|a| bound.axis(a)));
                fold(n, bound.elements(result), inner, unordered && wide)
            }
        }
    }

    /// Private placement (storage rule in `realize`): packed planes and tiles of fewer than
    /// one subgroup of elements are replicated per lane, larger tiles distributed.
    fn tile(ledger: &mut Ledger, event: TileEvent<'_, '_>) {
        let Some(variable) = event.variable else { return };
        let (bound, shaped) = (event.bound, event.shaped);
        let held = held_bits(bound, &shaped.elem);
        let elements = bound.elements_of(shaped);
        let replicated_bits = Quantity::product([elements.clone(), held.clone()]);
        let private_bits = if matches!(shaped.elem, Elem::Repr(_)) { replicated_bits.clone() } else { Quantity::product([lane_share(elements), held]) };
        // A launch with inner owner regions has one launch piece per threadgroup (its grid
        // limit is stated over the launch pieces), so no further pieces share the threadgroup.
        let pieces = match event.owner {
            TileOwner::Piece => event.pieces.clone(),
            TileOwner::Outer | TileOwner::Inner(_) => Quantity::one(),
        };
        ledger.tiles.insert(variable, Tile { bits: bound.tile_bits(shaped), private_bits, replicated_bits, snapshot: event.snapshot, pieces, launch: event.launch, owner: event.owner });
    }

    /// Storage rule: a tile of at least one subgroup of elements whose elements an owned loop
    /// reads at coordinates other than its own is threadgroup-placed and written cooperatively.
    /// A size that depends on selected geometry counts as placed (the limit stays conservative).
    fn owned_loop(walker: &mut Walker<'_, '_, Self>, coordinates: &[VarId], body: &Block) {
        let mut placed = Vec::new();
        each_expr(body, &mut |e| {
            let ExprKind::Index { base, indices } = &e.kind else { return };
            let (Some(root), Ty::Tile(_)) = (tile_root(base), &base.ty) else { return };
            let own = indices.len() == coordinates.len() && indices.iter().zip(coordinates).all(|(index, coordinate)| matches!(index, Index::Coord(v) if v == coordinate));
            if !own {
                placed.push(root);
            }
        });
        for root in placed {
            let elements = walker.bound.body.vars.get(root).map(|var| walker.bound.elements(&var.ty));
            if elements.is_some_and(|n| n.eval(&|_| None).map_or(true, |n| n >= SUBGROUP as u64)) {
                walker.ledger().cross_reads.insert(root);
            }
        }
    }

    fn induction_intrinsic(op: &Operation) -> bool {
        matches!(op, Operation::LaneIndex)
    }

    fn call_argument(walker: &mut Walker<'_, '_, Self>, call: CallId, ordinal: usize, arg: &Expr) -> Result<(), SelectionError> {
        if let (Some(root), true) = (tile_root(arg), matches!(arg.ty, Ty::Tile(_))) {
            walker.ledger().call_arguments.push((call, ordinal, root));
        }
        let (device, local, ops) = (walker.work.device_bits.len(), walker.work.local_bits.len(), walker.work.lane_ops.len());
        // A snapshot of a view of an immutable tile passed straight to the callee is borrowed
        // too (the borrow proof holds for a value nothing can write): only the index
        // arithmetic of the view executes.
        if let ExprKind::Load(view) = &arg.kind {
            let immutable = tile_root(view).is_some_and(|root| matches!(walker.bound.body.vars.get(root).map(|v| &v.kind), Some(seismic_lang::sir::VarKind::Value)));
            if !external(&view.ty) && immutable {
                return walker.expr(view);
            }
        }
        walker.expr(arg)?;
        // A snapshot of external storage passed straight to the callee is borrowed (load
        // rule): nothing is copied, and its bytes cross the bus while the callee consumes
        // them, so the traffic belongs to the callee's span, where it overlaps that compute.
        if let ExprKind::Load(view) = &arg.kind {
            if external(&view.ty) {
                let device_bits: Vec<Quantity> = walker.work.device_bits.drain(device..).collect();
                let local_bits: Vec<Quantity> = walker.work.local_bits.drain(local..).collect();
                // Keep the index arithmetic of the view; drop the copy's element operations.
                walker.work.lane_ops.truncate(ops);
                let traffic = walker.ledger().call_traffic.entry(call).or_default();
                traffic.device_bits.extend(device_bits);
                traffic.local_bits.extend(local_bits);
                walker.expr(view)?;
            }
        }
        Ok(())
    }

    fn intrinsic(walker: &mut Walker<'_, '_, Self>, op: &Operation, args: &[Expr]) -> Result<(), SelectionError> {
        match op {
            Operation::MatrixLoad | Operation::MatrixLoadTranspose | Operation::MatrixStore => {
                let root = args.get(1).and_then(tile_root);
                if let Some(root) = root {
                    walker.ledger().operands.insert(root);
                }
                // A load from a parameter reads an operand that is resident already (staged by
                // the caller, or external storage): no staging write and barrier precede it in
                // this body. Every other transfer is a store, or a load of a tile this body
                // stages with its barrier.
                let resident = !matches!(op, Operation::MatrixStore) && root.is_some_and(|root| matches!(walker.bound.body.vars.get(root).map(|v| &v.kind), Some(seismic_lang::sir::VarKind::Param(_))));
                let transfers = walker.scaled(Quantity::one());
                if resident { walker.work.resident_loads.push(transfers) } else { walker.work.matrix_transfers.push(transfers) }
                walker.traffic(false, Quantity::Constant(64 * 32));
            }
            Operation::MatrixMultiplyAccumulate => {
                let multiplies = walker.scaled(Quantity::one());
                walker.work.matrix_multiplies.push(multiplies);
            }
            Operation::SimdSum | Operation::SimdMax | Operation::SimdMin => walker.ops(Quantity::Constant(COLLECTIVE_OPS)),
            Operation::ShuffleIndex => walker.ops(Quantity::Constant(FOLD_SHUFFLE_OPS)),
            Operation::LaneIndex | Operation::Matrix => {}
        }
        Ok(())
    }

    fn inherited(parent: &Ledger, call: CallId) -> Work {
        parent.call_traffic.get(&call).cloned().unwrap_or_default()
    }
}
