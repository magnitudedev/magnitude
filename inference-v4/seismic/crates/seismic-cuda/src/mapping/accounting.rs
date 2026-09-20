//! CUDA's accounting over the shared structural walk, as the scalar PTX realization executes
//! it: one thread owns a piece and runs its work serially; every tile and materialized
//! snapshot is one aligned bump allocation of that thread's scratch in device global memory,
//! so tile accesses are memory traffic like tensor accesses. The intrinsic table is lane
//! index, shuffle and sum; there is no matrix coverage.
use seismic_compiler::selection::SelectionError;
use seismic_compiler::selection::quantity::Quantity;
use seismic_compiler::selection::structure::{Accounting, Bound, TileEvent, Walker, external};
use seismic_lang::family::CandidateRef;
use seismic_lang::intrinsics::Operation;
use seismic_lang::sir::{CallId, Expr, ExprKind};
use seismic_lang::sir::RegionMode;
use seismic_lang::types::{Shaped, Ty};

/// Every tile and every materialized snapshot is one bump allocation of the per-thread
/// scratch, aligned to eight bytes (`seismic-compiler` scalar realization).
pub(crate) const SCRATCH_ALIGNMENT_BITS: u64 = 64;

/// Element operations one warp collective (`simd_sum`: five butterfly shuffles with their
/// additions and one broadcast shuffle, see `ptx::prepare`) stands for. Counted from the
/// emitted PTX expansion; its native cost is unmeasured.
pub(crate) const COLLECTIVE_OPS: u64 = 16;

/// Element operations one `shuffle_index` exchange stands for (three emitted PTX
/// instructions). Unmeasured.
pub(crate) const SHUFFLE_OPS: u64 = 3;

/// Element operations of decoding one packed element at its consumption. Code: word load,
/// shift, mask, integer conversion. Coefficients: two loads with their conversions.
const PACKED_CODE_OPS: u64 = 4;
const PACKED_COEFFICIENT_OPS: u64 = 4;

/// Work totals of one execution scope, each a sum of terms already scaled by multiplicity.
#[derive(Clone, Debug, Default)]
pub(crate) struct Work {
    /// Element operations of the scope, all serial inside the one thread that owns a piece.
    pub ops: Vec<Quantity>,
    /// Ordered/pipeline window visits.
    pub visits: Vec<Quantity>,
    /// Bits read from or written to device global memory: tensors and scratch tiles alike
    /// (the scalar realization keeps every tile in invocation-owned global memory).
    pub memory_bits: Vec<Quantity>,
}

impl Work {
    pub fn quantities(&self) -> impl Iterator<Item = &Quantity> {
        self.ops.iter().chain(&self.visits).chain(&self.memory_bits)
    }
}

#[derive(Clone, Debug)]
pub(crate) struct Tile {
    /// Scratch bits of one thread for this allocation: the whole tile at its native element
    /// type, rounded up to the scratch alignment. A snapshot of external storage is counted
    /// as materialized (the load rule borrows it only where the borrow proof holds, which
    /// is known after instantiation; `realize` rechecks the exact scratch).
    pub scratch_bits: Quantity,
    pub launch: Option<(CandidateRef, usize)>,
    pub pieces: Quantity,
}

#[derive(Clone, Debug, Default)]
pub(crate) struct Ledger {
    /// Tile bindings and snapshots of external storage, in authored order.
    pub tiles: Vec<Tile>,
    /// The body names a participant intrinsic: the whole entry then runs one warp per piece.
    pub participants: bool,
}

pub(crate) struct CudaAccounting;

impl Accounting for CudaAccounting {
    const NAME: &'static str = "CUDA";
    const TILES_ARE_MEMORY_TRAFFIC: bool = true;
    type Work = Work;
    type Ledger = Ledger;

    fn maximum(into: &mut Work, then: Work, els: Work) {
        let join = |into: &mut Vec<Quantity>, a: Vec<Quantity>, b: Vec<Quantity>| {
            if !(a.is_empty() && b.is_empty()) {
                into.push(Quantity::Max(vec![Quantity::Sum(a), Quantity::Sum(b)]));
            }
        };
        join(&mut into.ops, then.ops, els.ops);
        join(&mut into.visits, then.visits, els.visits);
        join(&mut into.memory_bits, then.memory_bits, els.memory_bits);
    }

    fn quantities(work: &Work) -> Vec<&Quantity> {
        work.quantities().collect()
    }

    fn push_operations(work: &mut Work, scaled: Quantity) {
        work.ops.push(scaled);
    }

    fn push_visit(work: &mut Work, mode: RegionMode, scaled: Quantity) {
        if mode != RegionMode::Parallel {
            work.visits.push(scaled);
        }
    }

    fn push_traffic(work: &mut Work, _external: bool, scaled: Quantity) {
        work.memory_bits.push(scaled);
    }

    fn packed_element(bound: &Bound<'_>, shaped: &Shaped) -> Option<Quantity> {
        let group = bound.packed_group(&shaped.elem)?;
        let span = shaped
            .axes
            .get(
                shaped
                    .packed_axis
                    .unwrap_or(shaped.rank().saturating_sub(1)),
            )
            .map(|extent| bound.axis(extent));
        Some(match span {
            Some(span) => Quantity::exceeds(
                span,
                group,
                PACKED_CODE_OPS + PACKED_COEFFICIENT_OPS,
                PACKED_CODE_OPS,
            ),
            None => Quantity::Constant(PACKED_CODE_OPS + PACKED_COEFFICIENT_OPS),
        })
    }

    /// The owner thread folds every element in authored ascending order, whether or not the
    /// contract permits reassociation.
    fn reduction(
        bound: &Bound<'_>,
        value: &Expr,
        _axis: usize,
        _unordered: bool,
        _result: &Ty,
    ) -> Quantity {
        bound.elements(&value.ty)
    }

    /// A tile binding is one scratch allocation. A snapshot of external storage is recorded
    /// where its `load` is walked, and only a dense one: a packed snapshot has no scratch
    /// form in the scalar realization and reserves nothing.
    fn tile(ledger: &mut Ledger, event: TileEvent<'_, '_>) {
        let reserved = match event.variable {
            Some(_) => !event.snapshot,
            None => event.bound.dtype(&event.shaped.elem).is_some(),
        };
        if reserved {
            let scratch_bits = Quantity::Aligned(
                Box::new(event.bound.tile_bits(event.shaped)),
                SCRATCH_ALIGNMENT_BITS,
            );
            ledger.tiles.push(Tile {
                scratch_bits,
                launch: event.launch,
                pieces: event.pieces.clone(),
            });
        }
    }

    /// The lane index is coordinate arithmetic (never charged, as on Metal).
    fn induction_intrinsic(op: &Operation) -> bool {
        matches!(op, Operation::LaneIndex)
    }

    /// A snapshot of external storage handed straight to a call: the load rule borrows it, so
    /// no copy is charged; its stored bits are read once per distinct view.
    fn call_argument(
        walker: &mut Walker<'_, '_, Self>,
        _call: CallId,
        _ordinal: usize,
        arg: &Expr,
    ) -> Result<(), SelectionError> {
        match &arg.kind {
            ExprKind::Load(view) if external(&view.ty) => {
                walker.expr(view)?;
                if let Some(shaped) = arg.ty.shaped() {
                    walker.tile(None, shaped, true);
                }
                if let Some(shaped) = view.ty.shaped() {
                    let bits = walker
                        .bound
                        .stored_bits(walker.bound.elements(&arg.ty), &shaped.elem);
                    let distinct = Quantity::product([walker.distinct_reads(view), bits]);
                    walker.work.memory_bits.push(distinct);
                }
                Ok(())
            }
            _ => walker.expr(arg),
        }
    }

    fn intrinsic(
        walker: &mut Walker<'_, '_, Self>,
        op: &Operation,
        _args: &[Expr],
    ) -> Result<(), SelectionError> {
        match op {
            Operation::SimdSum => {
                walker.ledger().participants = true;
                walker.ops(Quantity::Constant(COLLECTIVE_OPS));
            }
            Operation::ShuffleIndex => {
                walker.ledger().participants = true;
                walker.ops(Quantity::Constant(SHUFFLE_OPS));
            }
            Operation::LaneIndex => walker.ledger().participants = true,
            Operation::SimdMax
            | Operation::SimdMin
            | Operation::Matrix
            | Operation::MatrixLoad
            | Operation::MatrixLoadTranspose
            | Operation::MatrixStore
            | Operation::MatrixMultiplyAccumulate
            | Operation::MatrixMatmul
            | Operation::MatrixMatmulAdd => {
                return walker.unsupported(&format!("intrinsic `{op}`"));
            }
        }
        Ok(())
    }
}
