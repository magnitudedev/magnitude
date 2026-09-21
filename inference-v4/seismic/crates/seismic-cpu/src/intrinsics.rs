//! The CPU dialect: its (empty) host-intrinsic enum and its layout algebra.
//!
//! The CPU backend implements the whole portable matrix and offers no
//! host-specific intrinsics, so `CpuIntrinsic` is uninhabited: every encoder
//! arm over `KernelOp<CpuIntrinsic>` is vacuous for the `Intrinsic` variant,
//! and `IntrinsicCatalog::lower` is total vacuously (the CPU profile's
//! `effective_signatures` is empty, so S1 declines every capability proposal
//! before one can be presented to the catalog).
//!
//! Layouts are dense row-major over the storage's own element grid. A
//! representation-plane storage's grid is `[outer axes…, storage elements per
//! row]` at capacity, matching the registry plane geometry that sized it.

use seismic_lang::{
    repr,
    sym::Sym,
    types::{DType, Elem, ExtentExpr, TensorType},
};
use seismic_realization::{
    kernel::{ExecutableDialect, IntrinsicCatalog, IntrinsicConsequences, IntrinsicReferences, PlaneRef},
    plan_space::SolvedValues,
};

/// The closed CPU intrinsic enum: host-specific intrinsics only, of which the
/// CPU target has none.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CpuIntrinsic {}

/// The CPU executable dialect.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CpuDialect;

impl seismic_realization::kernel::sealed::Sealed for CpuDialect {}

/// One axis extent as a solvable size atom: a constant, or the reserved
/// `@runtime<N>` capacity atom of one runtime extent (the binding P1's
/// `SolvedValues` carries). Sealed extents contain nothing else; a surviving
/// unresolved symbolic extent contradicts the seal.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CpuLayoutTemplate {
    pub axes: Vec<Sym>,
    pub dtype: DType,
}

/// A resolved dense row-major layout: the storage's element grid at capacity.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CpuResolvedLayout {
    pub shape: Vec<u64>,
    pub dtype: DType,
}

fn extent_atom(extent: &ExtentExpr) -> Sym {
    match extent {
        ExtentExpr::Static(n) => match i64::try_from(*n) {
            Ok(value) => Sym::constant(value),
            // The layout algebra is i64-sized; a wider extent cannot be
            // sealed into a CPU residence (M1 size-domain proof).
            Err(_) => unreachable!("an axis extent exceeds the 64-bit signed size domain (M1)"),
        },
        ExtentExpr::Runtime(id) => Sym::param(&format!("@runtime{}", id.0)),
        ExtentExpr::Sym(sym) => match sym.as_constant() {
            Some(value) => Sym::constant(value),
            // The seal substitutes every planning symbol before a layout
            // is built (P1 resolution).
            None => unreachable!("an unresolved planning symbol survived into a CPU layout (P1)"),
        },
    }
}

/// One layout over the typed plane descriptor: the element dtype of a dense
/// plane comes from the descriptor itself; a representation plane's grid is
/// `[outer axes…, raw accessor extent along the packing axis]` at capacity
/// (the registry's only packing rule is `Last`), with rows padded to whole
/// storage groups exactly as `Repr::snapshot_layout` computes. No optional
/// member and no default: every fact comes from `PlaneRef`.
fn layout_template(tensor: &TensorType, plane: PlaneRef) -> CpuLayoutTemplate {
    match plane {
        PlaneRef::Dense { elem } => CpuLayoutTemplate {
            axes: tensor.axes.iter().map(extent_atom).collect(),
            dtype: elem,
        },
        PlaneRef::Repr { plane } => {
            // The representation of a representation plane: a residence
            // over a packed element declares exactly the representation's
            // planes (D1), and `PlaneRef::of` rejects an unregistered name,
            // so the remaining pairs are unconstructible.
            let Elem::Repr(name) = &tensor.elem else {
                unreachable!("a representation plane over a non-packed element (D1)")
            };
            let representation = repr::lookup(name)
                // `PlaneRef::of` rejects an unregistered representation
                // before the layout is built (K1).
                .unwrap_or_else(|| unreachable!("a plane of an unregistered representation (K1)"));
            // The packing axis exists for every packed tensor (the registry
            // resolves `Last` against the packed rank).
            let (width, outer) = match tensor.axes.split_last() {
                Some(split) => split,
                None => unreachable!("a packed plane of a scalar (D1)"),
            };
            let group = i64::from(representation.storage_group());
            let width_atom = extent_atom(width);
            // ceil(width / group) * group, as a size expression.
            let physical_width = width_atom
                .add(&Sym::constant(group - 1))
                .quot(&Sym::constant(group))
                .scale(group);
            let mut axes: Vec<Sym> = outer.iter().map(extent_atom).collect();
            axes.push(plane.extent(&physical_width));
            CpuLayoutTemplate {
                axes,
                dtype: plane.dtype(),
            }
        }
    }
}

impl ExecutableDialect for CpuDialect {
    type Intrinsic = CpuIntrinsic;
    type LayoutTemplate = CpuLayoutTemplate;
    type ResolvedLayout = CpuResolvedLayout;

    fn intrinsic_references(op: &CpuIntrinsic) -> IntrinsicReferences {
        // Uninhabited: `match *op {}` is exhaustive over the empty intrinsic
        // enum, so no CPU intrinsic exists to reference anything.
        match *op {}
    }

    fn intrinsic_consequences(op: &CpuIntrinsic) -> IntrinsicConsequences {
        match *op {}
    }

    fn public_layout(tensor: &TensorType, plane: PlaneRef) -> CpuLayoutTemplate {
        layout_template(tensor, plane)
    }

    fn internal_layout(tensor: &TensorType, plane: PlaneRef) -> CpuLayoutTemplate {
        // One host memory model: dense row-major for public and internal
        // residences alike.
        layout_template(tensor, plane)
    }

    fn resolve_layout(
        layout: &CpuLayoutTemplate,
        values: &SolvedValues,
    ) -> CpuResolvedLayout {
        CpuResolvedLayout {
            shape: layout.axes.iter().map(|axis| values.eval(axis)).collect(),
            dtype: layout.dtype,
        }
    }
}

/// The CPU intrinsic catalog. Vacuously total: the CPU profile declares no
/// effective signatures, so no authorized use can ever be presented.
#[derive(Clone, Copy, Debug)]
pub struct CpuIntrinsicCatalog;

impl IntrinsicCatalog<CpuDialect> for CpuIntrinsicCatalog {
    fn lower(
        &self,
        _intrinsic: &seismic_lang::sir::IntrinsicUse,
        _operands: &[seismic_realization::kernel::IntrinsicOperand],
        _result: seismic_realization::kernel::IntrinsicResult,
    ) -> CpuIntrinsic {
        // The CPU profile's `effective_signatures` is empty and S1 declines
        // every capability proposal before presentation, so no authorized
        // use can reach the catalog.
        unreachable!("the CPU target was asked to lower a capability intrinsic (S1)")
    }
}

