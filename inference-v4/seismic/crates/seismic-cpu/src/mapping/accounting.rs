//! The CPU's accounting over the shared structural walk: two work classes (scalar element
//! operations and memory bits), every tile a scratch slot of the executing worker, no target
//! intrinsics and no atomic updates.
use seismic_compiler::selection::quantity::Quantity;
use seismic_compiler::selection::structure::{Accounting, Bound, TileEvent, Walker};
use seismic_compiler::selection::SelectionError;
use seismic_lang::family::CandidateRef;
use seismic_lang::intrinsics::Operation;
use seismic_lang::sir::Expr;
use seismic_lang::syntax::ast::RegionMode;
use seismic_lang::types::{Elem, Shaped, Ty};

/// Element operations of decoding one packed element at its consumption: word load, shift,
/// mask, integer conversion, and two coefficient loads with their conversions. The scalar
/// realization hoists nothing, so every element pays all of them.
const PACKED_ELEMENT_OPS: u64 = 8;

/// Work totals of one execution scope, each a sum of terms already scaled by multiplicity.
#[derive(Clone, Debug, Default)]
pub(crate) struct Work {
    /// Scalar element operations, summed over all pieces. Integer arithmetic over loop
    /// binders, constants and selected geometry is address computation: part of the
    /// per-operation rate, not counted.
    pub ops: Vec<Quantity>,
    /// Bits read from or written to buffers and scratch tiles. External bytes are charged
    /// once per distinct view; further visits are taken as cache-served.
    pub memory_bits: Vec<Quantity>,
}

#[derive(Clone, Debug)]
pub(crate) struct Tile {
    /// Scratch storage in bits at the native element type. A snapshot of external storage is
    /// counted as if materialized: whether the load rule borrows it is known only to `realize`.
    pub bits: Quantity,
    pub launch: Option<(CandidateRef, usize)>,
}

#[derive(Clone, Debug, Default)]
pub(crate) struct Ledger {
    pub tiles: Vec<Tile>,
}

pub(crate) struct CpuAccounting;

impl Accounting for CpuAccounting {
    const NAME: &'static str = "CPU";
    const ATOMIC: bool = false;
    type Work = Work;
    type Ledger = Ledger;

    fn maximum(into: &mut Work, then: Work, els: Work) {
        let join = |into: &mut Vec<Quantity>, a: Vec<Quantity>, b: Vec<Quantity>| {
            if !(a.is_empty() && b.is_empty()) {
                into.push(Quantity::Max(vec![Quantity::Sum(a), Quantity::Sum(b)]));
            }
        };
        join(&mut into.ops, then.ops, els.ops);
        join(&mut into.memory_bits, then.memory_bits, els.memory_bits);
    }

    fn quantities(work: &Work) -> Vec<&Quantity> {
        work.ops.iter().chain(&work.memory_bits).collect()
    }

    fn push_operations(work: &mut Work, scaled: Quantity) {
        work.ops.push(scaled);
    }

    /// Loop bookkeeping of one visit of any region: one element operation.
    fn push_visit(work: &mut Work, _mode: RegionMode, scaled: Quantity) {
        work.ops.push(scaled);
    }

    fn push_traffic(work: &mut Work, _external: bool, scaled: Quantity) {
        work.memory_bits.push(scaled);
    }

    fn packed_element(bound: &Bound<'_>, shaped: &Shaped) -> Option<Quantity> {
        matches!(bound.element(&shaped.elem), Some(Elem::Repr(_))).then_some(Quantity::Constant(PACKED_ELEMENT_OPS))
    }

    /// A reduction folds every element once, in ascending order, on the thread that owns it.
    fn reduction(bound: &Bound<'_>, value: &Expr, _axis: usize, _unordered: bool, _result: &Ty) -> Quantity {
        bound.elements(&value.ty)
    }

    /// Every tile binding is scratch storage of the worker that executes it.
    fn tile(ledger: &mut Ledger, event: TileEvent<'_, '_>) {
        if event.variable.is_some() {
            ledger.tiles.push(Tile { bits: event.bound.tile_bits(event.shaped), launch: event.launch });
        }
    }

    fn intrinsic(walker: &mut Walker<'_, '_, Self>, op: &Operation, _args: &[Expr]) -> Result<(), SelectionError> {
        walker.unsupported(&format!("target intrinsic `{op}`"))
    }
}
