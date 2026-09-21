//! The CUDA executable dialect: the closed intrinsic enum (the `cuda.subgroup`
//! family), the total intrinsic catalog over the target's effective
//! signatures, and the dense layout templates.
//!
//! The dialect contributes intrinsics only — universal semantic operations
//! are core-owned (`KernelOp::Core`). `cuda.matrix` is not declared by this
//! backend: its typing, resources, and PTX emission are not complete, so it
//! is never an effective signature of the CUDA profile and a program that
//! applies it is rejected at logical construction. When the family is
//! completed it joins `CudaIntrinsic` as named variants and the target
//! profile admits its signatures — never a fallback.

use seismic_lang::{
    intrinsics::{IntrinsicId, MathOp, ReduceOp},
    sir::IntrinsicUse,
    sym::Sym,
    types::{DType, ExtentExpr, TensorType},
};
use seismic_realization::ids::KernelSsaId;
use seismic_realization::kernel::{
    CostUnit, ExecutableDialect, IntrinsicCatalog, IntrinsicConsequences, IntrinsicOperand,
    IntrinsicReferences, IntrinsicResult, KernelValueRef, KernelValueType, PlaneRef,
};
use seismic_realization::numerics::{CapabilitySignatureId, CountExpr, NumericalTransfer};
use seismic_realization::plan_space::SolvedValues;
use std::collections::BTreeSet;

/// The closed CUDA intrinsic set: the `cuda.subgroup` family only.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CudaIntrinsic {
    /// `%laneid` of this participant (`cuda.subgroup.lane_index: () -> i32`).
    LaneIndex { into: KernelSsaId },
    /// One warp lane exchange (`cuda.subgroup.shuffle: (T, i32) -> T`).
    Shuffle {
        into: KernelSsaId,
        value: KernelValueRef,
        index: KernelValueRef,
        dtype: DType,
    },
    /// One warp butterfly reduction
    /// (`cuda.subgroup.simd_{sum,max,min}: (T) -> T`).
    SubgroupReduce {
        into: KernelSsaId,
        op: ReduceOp,
        value: KernelValueRef,
        dtype: DType,
    },
}

pub(crate) fn subgroup_intrinsic(name: &str) -> IntrinsicId {
    IntrinsicId {
        capability: seismic_lang::intrinsics::CapabilityId::new("cuda", "subgroup"),
        name: name.into(),
    }
}

/// The intrinsic signatures this backend authorizes on every CUDA target
/// (exact-typed uses are matched by `TargetProfile::supports_intrinsic`).
pub(crate) fn subgroup_families() -> BTreeSet<IntrinsicId> {
    ["lane_index", "shuffle", "simd_sum", "simd_max", "simd_min"]
        .into_iter()
        .map(subgroup_intrinsic)
        .collect()
}

impl CudaIntrinsic {
    /// The capability signature this intrinsic realizes.
    pub fn capability(&self) -> IntrinsicId {
        match self {
            CudaIntrinsic::LaneIndex { .. } => subgroup_intrinsic("lane_index"),
            CudaIntrinsic::Shuffle { .. } => subgroup_intrinsic("shuffle"),
            CudaIntrinsic::SubgroupReduce { op, .. } => subgroup_intrinsic(match op {
                ReduceOp::Sum => "simd_sum",
                ReduceOp::Max => "simd_max",
                ReduceOp::Min => "simd_min",
                // The registry declares no `simd_argmax`; `lower` never
                // produces one, so this arm is never constructed.
                ReduceOp::Argmax => "simd_sum",
            }),
        }
    }
}

/// The CUDA executable dialect (sealed by `kernel::sealed`).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Dialect;

impl seismic_realization::kernel::sealed::Sealed for Dialect {}

/// Dense storage layout: the plane's required element dtype, the per-axis
/// extent symbols, and the byte-count symbol (extent product scaled by the
/// element or plane width). Runtime extents contribute their capacity
/// symbol `@runtime<N>`; semantics always use the retained runtime value.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CudaLayoutTemplate {
    pub dtype: DType,
    pub axes: Vec<Sym>,
    pub bytes: Sym,
}

/// One resolved dense layout: the plane's element dtype, per-axis extents,
/// and the total byte count at capacity, all substituted under the
/// solver's validated values.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CudaResolvedLayout {
    pub dtype: DType,
    pub axes: Vec<u64>,
    pub bytes: u64,
}

/// The named-defect idiom for the infallible layout boundary: an input the
/// sealed descriptor and specialization contracts exclude.
fn defect(invariant: &str) -> ! {
    panic!("compiler defect (B1Cuda): {invariant}")
}

/// One axis extent as a solvable size atom: an exact static constant, or
/// the reserved `@runtime<N>` capacity atom of one runtime extent (the
/// binding P1's `SolvedValues` carries).
fn extent_symbol(extent: &ExtentExpr) -> Sym {
    match extent {
        ExtentExpr::Static(n) => match i64::try_from(*n) {
            Ok(value) => Sym::constant(value),
            Err(_) => defect("an axis extent exceeds the 64-bit signed size domain"),
        },
        ExtentExpr::Runtime(id) => Sym::param(&format!("@runtime{}", id.0)),
        ExtentExpr::Sym(sym) => match sym.as_constant() {
            Some(value) => Sym::constant(value),
            None => defect("an unresolved planning symbol survived into a CUDA layout"),
        },
    }
}

/// The layout of one residence plane over a tensor type, expressed over
/// the required plane descriptor: a dense plane is a dense element grid;
/// a representation plane is the outer rows times the plane's own accessor
/// extent along the packing axis (the registry's only packing rule is
/// `Last`).
fn layout_of(tensor: &TensorType, plane: PlaneRef) -> CudaLayoutTemplate {
    match plane {
        PlaneRef::Dense { elem } => {
            let mut axes = Vec::with_capacity(tensor.axes.len());
            let mut elements = Sym::constant(1);
            for axis in &tensor.axes {
                let factor = extent_symbol(axis);
                axes.push(factor.clone());
                elements = elements.mul(&factor);
            }
            CudaLayoutTemplate {
                dtype: elem,
                axes,
                bytes: elements.scale(i64::from(elem.bytes())),
            }
        }
        PlaneRef::Repr { plane } => {
            let Some((width, outer)) = tensor.axes.split_last() else {
                defect("a packed plane of a scalar")
            };
            let width_symbol = extent_symbol(width);
            let mut rows = Sym::constant(1);
            let mut axes = Vec::with_capacity(outer.len() + 1);
            for axis in outer {
                let factor = extent_symbol(axis);
                axes.push(factor.clone());
                rows = rows.mul(&factor);
            }
            let plane_extent = plane.extent(&width_symbol);
            axes.push(plane_extent.clone());
            CudaLayoutTemplate {
                dtype: plane.dtype(),
                axes,
                bytes: rows.mul(&plane_extent),
            }
        }
    }
}

impl ExecutableDialect for Dialect {
    type Intrinsic = CudaIntrinsic;
    type LayoutTemplate = CudaLayoutTemplate;
    type ResolvedLayout = CudaResolvedLayout;

    fn intrinsic_references(op: &CudaIntrinsic) -> IntrinsicReferences {
        match op {
            CudaIntrinsic::LaneIndex { into } => IntrinsicReferences {
                uses: Vec::new(),
                defines: vec![*into],
                places: Vec::new(),
            },
            CudaIntrinsic::Shuffle {
                into, value, index, ..
            } => IntrinsicReferences {
                uses: vec![*value, *index],
                defines: vec![*into],
                places: Vec::new(),
            },
            CudaIntrinsic::SubgroupReduce { into, value, .. } => IntrinsicReferences {
                uses: vec![*value],
                defines: vec![*into],
                places: Vec::new(),
            },
        }
    }

    fn intrinsic_consequences(op: &CudaIntrinsic) -> IntrinsicConsequences {
        // The warp is the subgroup: every participant-group intrinsic
        // requires the target's full 32-lane width.
        IntrinsicConsequences {
            capability: op.capability(),
            numerical: match op {
                CudaIntrinsic::LaneIndex { .. } => NumericalTransfer::Exact,
                // One exchange: exactly one rounding at the lane dtype.
                CudaIntrinsic::Shuffle { dtype, .. } => NumericalTransfer::Round {
                    dtype: *dtype,
                    count: CountExpr::one(),
                },
                CudaIntrinsic::SubgroupReduce { .. } => NumericalTransfer::Capability {
                    signature: CapabilitySignatureId::new(op.capability(), Vec::new()),
                    bound: None,
                },
            },
            private_bytes: 0,
            workgroup_bytes: 0,
            required_subgroup_width: Some(32),
        }
    }

    fn public_layout(tensor: &TensorType, plane: PlaneRef) -> CudaLayoutTemplate {
        layout_of(tensor, plane)
    }

    fn internal_layout(tensor: &TensorType, plane: PlaneRef) -> CudaLayoutTemplate {
        layout_of(tensor, plane)
    }

    /// Mechanical substitution under validated solved values. Infallible:
    /// the seal validated every symbol before substitution.
    fn resolve_layout(layout: &CudaLayoutTemplate, values: &SolvedValues) -> CudaResolvedLayout {
        CudaResolvedLayout {
            dtype: layout.dtype,
            axes: layout.axes.iter().map(|axis| values.eval(axis)).collect(),
            bytes: values.eval(&layout.bytes),
        }
    }
}

/// The typed intrinsic lowering catalog of the CUDA target: total over the
/// effective `cuda.subgroup` signatures. A use outside those signatures is
/// never presented (S1 declines every proposal that requires it), and a
/// reduction presented through an algorithm-intrinsic choice is not a shape
/// this catalog's rules propose.
pub struct CudaIntrinsicCatalog;

impl IntrinsicCatalog<Dialect> for CudaIntrinsicCatalog {
    fn lower(
        &self,
        intrinsic: &IntrinsicUse,
        operands: &[IntrinsicOperand],
        result: IntrinsicResult,
    ) -> CudaIntrinsic {
        let name = intrinsic.id.name.as_str();
        let path = intrinsic.id.path();
        let use_description =
            format!("authorized use `{path}` with operands {operands:?} and result {result:?}");
        let defect = |detail: &str| -> ! {
            panic!(
                "compiler defect (B1Cuda): the CUDA intrinsic catalog cannot lower \
                 {use_description}: {detail}"
            )
        };
        match name {
            "lane_index" => match (operands, result) {
                ([], IntrinsicResult::Ssa { id, ty }) => {
                    debug_assert_eq!(
                        ty,
                        KernelValueType::Scalar(DType::I32),
                        "lane_index defines one i32"
                    );
                    CudaIntrinsic::LaneIndex { into: id }
                }
                _ => defect("lane_index takes no operands and defines one i32"),
            },
            "shuffle" => match (operands, result) {
                (
                    [IntrinsicOperand::Value(value), IntrinsicOperand::Value(index)],
                    IntrinsicResult::Ssa { id, ty },
                ) => {
                    let KernelValueType::Scalar(dtype) = ty else {
                        defect("shuffle defines a scalar")
                    };
                    CudaIntrinsic::Shuffle {
                        into: id,
                        value: *value,
                        index: *index,
                        dtype,
                    }
                }
                _ => defect("shuffle takes two scalar operands and defines one scalar"),
            },
            "simd_sum" | "simd_max" | "simd_min" => match (operands, result) {
                ([IntrinsicOperand::Value(value)], IntrinsicResult::Ssa { id, ty }) => {
                    let KernelValueType::Scalar(dtype) = ty else {
                        defect("a subgroup reduction defines a scalar")
                    };
                    CudaIntrinsic::SubgroupReduce {
                        into: id,
                        op: match name {
                            "simd_sum" => ReduceOp::Sum,
                            "simd_max" => ReduceOp::Max,
                            _ => ReduceOp::Min,
                        },
                        value: *value,
                        dtype,
                    }
                }
                _ => defect(
                    "a subgroup reduction takes one scalar operand and defines one \
                     scalar; the CUDA catalog authorizes it only as a scalar capability use",
                ),
            },
            other => defect(&format!(
                "`{other}` is not an authorized CUDA subgroup intrinsic"
            )),
        }
    }
}

/// The point cost of one cost unit under the uncalibrated ranking identity
/// (`cuda-estimate-unqualified-v0`): ranking only, never legality.
pub(crate) fn point_cost_ns(unit: &CostUnit, model: &crate::mapping::EstimateModel) -> u64 {
    let memory_ns = |bytes: u64| {
        let bits = bytes.saturating_mul(8);
        let bits_per_second = (model.memory_bytes_per_second as u64)
            .saturating_mul(8)
            .max(1);
        bits.saturating_mul(1_000_000_000) / bits_per_second
    };
    let op_ns = 1_000_000_000 / (model.ops_per_second as u64).max(1);
    match unit {
        CostUnit::Scalar => op_ns,
        CostUnit::Load { dtype } | CostUnit::Store { dtype } => {
            memory_ns(u64::from(dtype.bytes())) + op_ns
        }
        CostUnit::PlaneAccess { dtype, .. } => memory_ns(u64::from(dtype.bytes())) + op_ns,
        CostUnit::Math { op, .. } => {
            // The versioned software sequence for the transcendental
            // reference; one instruction for the rest.
            let steps = match op {
                MathOp::Exp | MathOp::Log | MathOp::Sin | MathOp::Cos => 16,
                _ => 1,
            };
            op_ns.saturating_mul(steps)
        }
        CostUnit::Atomic { .. } => op_ns.saturating_mul(8),
        CostUnit::Fold { .. } => op_ns.saturating_mul(4),
        CostUnit::Barrier => 20,
        CostUnit::Intrinsic(_) => op_ns.saturating_mul(4),
    }
}
