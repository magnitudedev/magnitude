//! The Metal dialect: the sealed `MetalIntrinsic` enum (subgroup/matrix
//! capability intrinsics only), the total intrinsic catalog over effective
//! signatures, and the Metal storage layouts.
//!
//! The dialect contributes no universal operations: every portable semantic
//! form is core-owned (`KernelOp::Core`). `MetalIntrinsic` mirrors the
//! registry's `IntrinsicLowering` families — `metal.subgroup.{lane_index,
//! shuffle, simd_sum, simd_max, simd_min}` and `metal.matrix.{matmul,
//! matmul_add}` — with operands and results already resolved to kernel
//! references by core formation, so `lower` is a mechanical table and
//! encoding is exhaustive over the closed enum.
//!
//! Layouts: a residence plane is dense or one plane of a packed
//! representation, sized by a `Sym` expression over qualified tuning names
//! and `@runtime<N>` capacity atoms (the atoms `SolvedValues` binds); the
//! seal proved every symbol before substitution, so `resolve_layout` is
//! infallible.

use seismic_lang::{
    intrinsics::{
        self, reduce_schema, AtomicOp, IntrinsicId, IntrinsicLowering,
        MathOp, ReduceOp,
    },
    sir::IntrinsicUse,
    sym::Sym,
    types::{DType, ExtentExpr, TensorType, ValueType},
};
use seismic_realization::ids::KernelSsaId;
use seismic_realization::kernel::{
    IntrinsicCatalog, IntrinsicConsequences, IntrinsicOperand, IntrinsicReferences,
    IntrinsicResult, KernelPlaceRef, KernelValueRef, KernelViewChain, PlaneRef,
};
use seismic_realization::numerics::{
    CapabilitySignatureId, CountExpr, NumericalTransfer, ReductionTopology,
};
use seismic_realization::plan_space::SolvedValues;
use seismic_realization::failure::{CompilerDefect, Package};

/// The sealed Metal dialect. Intrinsics are subgroup/matrix capability
/// families only; every universal operation arrives as `KernelOp::Core`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct MetalDialect;

impl seismic_realization::kernel::sealed::Sealed for MetalDialect {}

/// The width of one Metal SIMD group. Every subgroup collective the dialect
/// emits operates on exactly this lane topology.
pub const SUBGROUP_WIDTH: u32 = 32;

/// One whole-tensor operand of a matrix intrinsic: the place in its view
/// coordinates together with the in-block view chain over it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MatrixOperand {
    pub place: KernelPlaceRef,
    pub ty: TensorType,
    pub view: KernelViewChain,
}

/// The closed Metal intrinsic enum: subgroup collectives and matrix
/// multiply. Encoded exhaustively by `encode`; never a placeholder.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MetalIntrinsic {
    /// `metal.subgroup.lane_index`: this participant's index in its group.
    ParticipantIndex {
        signature: IntrinsicId,
        into: KernelSsaId,
    },
    /// `metal.subgroup.shuffle`: exchange one value with participant `lane`.
    Exchange {
        signature: IntrinsicId,
        value: KernelValueRef,
        lane: KernelValueRef,
        into: KernelSsaId,
        dtype: DType,
    },
    /// `metal.subgroup.simd_{sum,max,min}`: combine every participant's
    /// value. The lane-strided fold plus collective is emitted by the
    /// encoder; the numerical transfer reassociates the ascending fold.
    SubgroupReduce {
        signature: IntrinsicId,
        op: ReduceOp,
        value: KernelValueRef,
        into: KernelSsaId,
        dtype: DType,
    },
    /// `metal.matrix.matmul`: `into := left @ right` over the output element
    /// domain, one ascending-k accumulation chain per participant.
    MatrixMatmul {
        signature: CapabilitySignatureId,
        left: MatrixOperand,
        right: MatrixOperand,
        into: KernelPlaceRef,
        rows: ExtentExpr,
        columns: ExtentExpr,
        inner: ExtentExpr,
        elem: DType,
        accumulator: DType,
    },
    /// `metal.matrix.matmul_add`: `into := left @ right + addend`.
    MatrixMatmulAdd {
        signature: CapabilitySignatureId,
        left: MatrixOperand,
        right: MatrixOperand,
        addend: MatrixOperand,
        into: KernelPlaceRef,
        rows: ExtentExpr,
        columns: ExtentExpr,
        inner: ExtentExpr,
        elem: DType,
        accumulator: DType,
    },
}

fn subgroup_transfer(op: ReduceOp) -> NumericalTransfer {
    NumericalTransfer::Reassociate {
        op,
        topology: ReductionTopology::Subgroup {
            width: SUBGROUP_WIDTH,
            inner: Box::new(ReductionTopology::SerialAxis {
                axis: 0,
                length: ExtentExpr::Static(0),
            }),
        },
    }
}

impl seismic_realization::kernel::ExecutableDialect for MetalDialect {
    type Intrinsic = MetalIntrinsic;
    type LayoutTemplate = MetalLayout;
    type ResolvedLayout = PhysicalMetalLayout;

    fn intrinsic_references(op: &MetalIntrinsic) -> IntrinsicReferences {
        match op {
            MetalIntrinsic::ParticipantIndex { into, .. } => IntrinsicReferences {
                uses: Vec::new(),
                defines: vec![*into],
                places: Vec::new(),
            },
            MetalIntrinsic::Exchange { value, lane, into, .. } => IntrinsicReferences {
                uses: vec![*value, *lane],
                defines: vec![*into],
                places: Vec::new(),
            },
            MetalIntrinsic::SubgroupReduce { value, into, .. } => IntrinsicReferences {
                uses: vec![*value],
                defines: vec![*into],
                places: Vec::new(),
            },
            MetalIntrinsic::MatrixMatmul {
                left, right, into, ..
            } => IntrinsicReferences {
                uses: Vec::new(),
                defines: Vec::new(),
                places: vec![left.place, right.place, *into],
            },
            MetalIntrinsic::MatrixMatmulAdd {
                left,
                right,
                addend,
                into,
                ..
            } => IntrinsicReferences {
                uses: Vec::new(),
                defines: Vec::new(),
                places: vec![left.place, right.place, addend.place, *into],
            },
        }
    }

    fn intrinsic_consequences(op: &MetalIntrinsic) -> IntrinsicConsequences {
        let (capability, numerical) = match op {
            MetalIntrinsic::ParticipantIndex { signature, .. } => {
                (signature.clone(), NumericalTransfer::Exact)
            }
            MetalIntrinsic::Exchange { signature, dtype, .. } => (
                signature.clone(),
                NumericalTransfer::Round {
                    dtype: *dtype,
                    count: CountExpr::one(),
                },
            ),
            MetalIntrinsic::SubgroupReduce { signature, op, .. } => {
                (signature.clone(), subgroup_transfer(*op))
            }
            MetalIntrinsic::MatrixMatmul { signature, .. }
            | MetalIntrinsic::MatrixMatmulAdd { signature, .. } => (
                signature.intrinsic.clone(),
                NumericalTransfer::Capability {
                    signature: signature.clone(),
                    bound: None,
                },
            ),
        };
        IntrinsicConsequences {
            capability,
            numerical,
            // The emitted forms keep every temporary in registers; staging
            // tiles are later optimization work through new registry
            // families, never silent emitter choices.
            private_bytes: 0,
            workgroup_bytes: 0,
            required_subgroup_width: match op {
                MetalIntrinsic::ParticipantIndex { .. }
                | MetalIntrinsic::Exchange { .. }
                | MetalIntrinsic::SubgroupReduce { .. } => Some(SUBGROUP_WIDTH),
                // The current matrix emission is the exact per-element fma
                // chain; no simdgroup fragment is declared.
                MetalIntrinsic::MatrixMatmul { .. } | MetalIntrinsic::MatrixMatmulAdd { .. } => {
                    None
                }
            },
        }
    }

    fn public_layout(tensor: &TensorType, plane: PlaneRef) -> MetalLayout {
        layout_of(tensor, plane)
    }

    fn internal_layout(tensor: &TensorType, plane: PlaneRef) -> MetalLayout {
        layout_of(tensor, plane)
    }

    fn resolve_layout(
        layout: &MetalLayout,
        values: &SolvedValues,
    ) -> PhysicalMetalLayout {
        // Infallible: every atom the template names was declared and every
        // declared atom is solved (`SolvedValues::eval` reports a violated
        // P1 invariant itself, never a runtime category).
        let elements = values.eval(&layout_elements(layout));
        let resolve_shape = |shape: &[ExtentExpr]| -> Vec<u64> {
            shape
                .iter()
                .map(|extent| match extent {
                    ExtentExpr::Static(value) => *value,
                    // A runtime extent contributes its capacity atom; the
                    // retained runtime value reaches kernels through
                    // `CoreKernelOp::RuntimeExtent` reads, never the layout.
                    ExtentExpr::Runtime(id) => values.eval(&Sym::param(&format!(
                        "@runtime{}",
                        id.0
                    ))),
                    ExtentExpr::Sym(value) => values.eval(value),
                })
                .collect()
        };
        match layout {
            MetalLayout::Dense { dtype, shape, .. } => PhysicalMetalLayout::Dense {
                dtype: *dtype,
                elements,
                shape: resolve_shape(shape),
            },
            MetalLayout::PackedPlane {
                representation,
                plane,
                dtype,
                shape,
                ..
            } => PhysicalMetalLayout::PackedPlane {
                representation: representation.clone(),
                plane: plane.clone(),
                dtype: *dtype,
                elements,
                shape: resolve_shape(shape),
            },
        }
    }
}

/// The total Metal intrinsic catalog. `lower` is a mechanical table over
/// the registry's `IntrinsicLowering` families: core formation presents a
/// use only when its exact signature is effective, with operands typed
/// exactly as the use states, so a structural mismatch contradicts the
/// checker (an impossible internal invariant, reported as a defect).
pub struct MetalIntrinsicCatalog;

fn defect(invariant: impl Into<String>) -> ! {
    panic!(
        "compiler defect ({:?}): {}",
        Package::B1Metal,
        invariant.into()
    )
}

/// The scalar dtype of one capability operand type: scalar uses state it
/// directly; reduction-driven uses state the operand tensor, whose element
/// dtype is the collective's dtype.
fn operand_dtype(use_id: &IntrinsicId, ty: &ValueType) -> DType {
    match ty {
        ValueType::Scalar(dtype) => *dtype,
        ValueType::Tensor(tensor) => match &tensor.elem {
            seismic_lang::types::Elem::Dtype(dtype) => *dtype,
            element => defect(format!(
                "`{use_id}` operand element `{element:?}` is not a registry element dtype"
            )),
        },
        other => defect(format!("`{use_id}` operand `{other}` is not a scalar or tensor")),
    }
}

fn value_operand(
    use_id: &IntrinsicId,
    operand: &IntrinsicOperand,
    role: &str,
) -> KernelValueRef {
    match operand {
        IntrinsicOperand::Value(value) => *value,
        IntrinsicOperand::Tensor { .. } => defect(format!(
            "`{use_id}` {role} operand must be a kernel scalar, found a tensor"
        )),
    }
}

fn matrix_operand(
    use_id: &IntrinsicId,
    operand: &IntrinsicOperand,
    role: &str,
) -> MatrixOperand {
    match operand {
        IntrinsicOperand::Tensor { place, ty, view } => MatrixOperand {
            place: *place,
            ty: ty.clone(),
            view: view.clone(),
        },
        IntrinsicOperand::Value(_) => defect(format!(
            "`{use_id}` {role} operand must be a rank-two tensor, found a scalar"
        )),
    }
}

/// The `[rows, inner]` / `[inner, columns]` / `[rows, columns]` axes of one
/// matrix operand of a checked use.
fn matrix_axes(use_id: &IntrinsicId, operand: &MatrixOperand, role: &str) -> [ExtentExpr; 2] {
    let [first, second] = operand.ty.axes.as_slice() else {
        defect(format!("`{use_id}` {role} operand is not rank two"));
    };
    [first.clone(), second.clone()]
}

fn matrix_elem(use_id: &IntrinsicId, operand: &MatrixOperand, role: &str) -> DType {
    match &operand.ty.elem {
        seismic_lang::types::Elem::Dtype(dtype) => *dtype,
        element => defect(format!(
            "`{use_id}` {role} operand element `{element:?}` is not a registry element dtype"
        )),
    }
}

impl IntrinsicCatalog<MetalDialect> for MetalIntrinsicCatalog {
    fn lower(
        &self,
        intrinsic: &IntrinsicUse,
        operands: &[IntrinsicOperand],
        result: IntrinsicResult,
    ) -> MetalIntrinsic {
        // The registry signature of the exact use: total over checked uses
        // and over the reduction-driven uses core formation synthesizes
        // with the node's operand and result types.
        let signature = intrinsics::capability_signature(intrinsic);
        let expect_arity = |operands: usize| {
            if operands != intrinsic.arguments.len() {
                defect(format!(
                    "`{}` presents {} operands for {} arguments",
                    intrinsic.id,
                    operands,
                    intrinsic.arguments.len()
                ));
            }
        };
        match &signature.lowering {
            IntrinsicLowering::ParticipantIndex => {
                expect_arity(0);
                let IntrinsicResult::Ssa { id, .. } = result else {
                    defect(format!(
                        "`{}` defines the participant index scalar, found `{:?}`",
                        intrinsic.id, result
                    ))
                };
                MetalIntrinsic::ParticipantIndex {
                    signature: intrinsic.id.clone(),
                    into: id,
                }
            }
            IntrinsicLowering::Exchange { dtype } => {
                expect_arity(2);
                MetalIntrinsic::Exchange {
                    signature: intrinsic.id.clone(),
                    value: value_operand(&intrinsic.id, &operands[0], "value"),
                    lane: value_operand(&intrinsic.id, &operands[1], "participant index"),
                    into: ssa_of(&intrinsic.id, result),
                    dtype: *dtype,
                }
            }
            IntrinsicLowering::SubgroupReduce { op, dtype } => {
                let _ = operand_dtype(&intrinsic.id, &intrinsic.arguments[0]);
                expect_arity(1);
                MetalIntrinsic::SubgroupReduce {
                    signature: intrinsic.id.clone(),
                    op: *op,
                    value: value_operand(&intrinsic.id, &operands[0], "value"),
                    into: ssa_of(&intrinsic.id, result),
                    dtype: *dtype,
                }
            }
            IntrinsicLowering::MatrixMatmul { .. } => {
                expect_arity(2);
                let (left, right) = (
                    matrix_operand(&intrinsic.id, &operands[0], "left"),
                    matrix_operand(&intrinsic.id, &operands[1], "right"),
                );
                let [rows, inner] = matrix_axes(&intrinsic.id, &left, "left");
                let [right_inner, columns] = matrix_axes(&intrinsic.id, &right, "right");
                if right_inner != inner {
                    defect(format!(
                        "`{}` inner axes disagree: {inner:?} vs {right_inner:?}",
                        intrinsic.id
                    ));
                }
                let IntrinsicResult::Tensor { place, ty } = result else {
                    defect(format!(
                        "`{}` produces a matrix, found `{:?}`",
                        intrinsic.id, result
                    ))
                };
                let accumulator = matrix_elem(&intrinsic.id, &MatrixOperand {
                    place,
                    ty: ty.clone(),
                    view: KernelViewChain::default(),
                }, "result");
                let elem = matrix_elem(&intrinsic.id, &left, "left");
                MetalIntrinsic::MatrixMatmul {
                    signature: CapabilitySignatureId::new(
                        intrinsic.id.clone(),
                        intrinsic.arguments.clone(),
                    ),
                    left,
                    right,
                    into: place,
                    rows,
                    columns,
                    inner,
                    elem,
                    accumulator,
                }
            }
            IntrinsicLowering::MatrixMatmulAdd { .. } => {
                expect_arity(3);
                let (left, right, addend) = (
                    matrix_operand(&intrinsic.id, &operands[0], "left"),
                    matrix_operand(&intrinsic.id, &operands[1], "right"),
                    matrix_operand(&intrinsic.id, &operands[2], "accumulator"),
                );
                let [rows, inner] = matrix_axes(&intrinsic.id, &left, "left");
                let [right_inner, columns] = matrix_axes(&intrinsic.id, &right, "right");
                if right_inner != inner {
                    defect(format!(
                        "`{}` inner axes disagree: {inner:?} vs {right_inner:?}",
                        intrinsic.id
                    ));
                }
                let IntrinsicResult::Tensor { place, ty } = result else {
                    defect(format!(
                        "`{}` produces a matrix, found `{:?}`",
                        intrinsic.id, result
                    ))
                };
                let accumulator = matrix_elem(&intrinsic.id, &MatrixOperand {
                    place,
                    ty: ty.clone(),
                    view: KernelViewChain::default(),
                }, "result");
                let addend_elem = matrix_elem(&intrinsic.id, &addend, "accumulator");
                if addend_elem != accumulator {
                    defect(format!(
                        "`{}` accumulator element {} differs from its result element {}",
                        intrinsic.id,
                        addend_elem.name(),
                        accumulator.name()
                    ));
                }
                let elem = matrix_elem(&intrinsic.id, &left, "left");
                MetalIntrinsic::MatrixMatmulAdd {
                    signature: CapabilitySignatureId::new(
                        intrinsic.id.clone(),
                        intrinsic.arguments.clone(),
                    ),
                    left,
                    right,
                    addend,
                    into: place,
                    rows,
                    columns,
                    inner,
                    elem,
                    accumulator,
                }
            }
        }
    }
}

fn ssa_of(use_id: &IntrinsicId, result: IntrinsicResult) -> KernelSsaId {
    match result {
        IntrinsicResult::Ssa { id, .. } => id,
        IntrinsicResult::Void | IntrinsicResult::Tensor { .. } => defect(format!(
            "`{use_id}` defines a kernel scalar result, found `{result:?}`"
        )),
    }
}

// ---------------------------------------------------------------------------
// Layouts
// ---------------------------------------------------------------------------

/// One residence plane's layout as a size expression: dense element
/// storage, or one plane of a packed representation. `elements` is a `Sym`
/// over qualified tuning names and `@runtime<N>` capacity atoms.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MetalLayout {
    Dense {
        dtype: DType,
        elements: Sym,
        shape: Vec<ExtentExpr>,
    },
    PackedPlane {
        representation: String,
        plane: String,
        dtype: DType,
        elements: Sym,
        shape: Vec<ExtentExpr>,
    },
}

/// The resolved layout: concrete element count and per-axis extents.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PhysicalMetalLayout {
    Dense {
        dtype: DType,
        elements: u64,
        shape: Vec<u64>,
    },
    PackedPlane {
        representation: String,
        plane: String,
        dtype: DType,
        elements: u64,
        shape: Vec<u64>,
    },
}

impl PhysicalMetalLayout {
    /// The plane's element dtype (the pointer's element type).
    pub fn dtype(&self) -> DType {
        match self {
            PhysicalMetalLayout::Dense { dtype, .. }
            | PhysicalMetalLayout::PackedPlane { dtype, .. } => *dtype,
        }
    }

    /// The residence's per-axis extents in its own coordinates.
    pub fn shape(&self) -> &[u64] {
        match self {
            PhysicalMetalLayout::Dense { shape, .. }
            | PhysicalMetalLayout::PackedPlane { shape, .. } => shape,
        }
    }
}

fn layout_elements(layout: &MetalLayout) -> Sym {
    match layout {
        MetalLayout::Dense { elements, .. } | MetalLayout::PackedPlane { elements, .. } => {
            elements.clone()
        }
    }
}

/// One static extent as a layout size-expression constant. A static extent
/// beyond the coefficient domain of `Sym` is a named defect, never a
/// saturated substitute.
fn static_extent(n: u64) -> Result<Sym, CompilerDefect> {
    i64::try_from(n).map(Sym::constant).map_err(|_| {
        CompilerDefect::new(
            Package::B1Metal,
            format!("a static extent of {n} exceeds the layout size-expression domain"),
        )
    })
}

/// The layout of one declared residence plane over a tensor type, from the
/// `PlaneRef` descriptor (`PlaneRef::of` is the sole construction
/// authority, so the descriptor/tensor pairing is already admitted). The
/// two inadmissible remainders are named defects: a representation plane
/// over a non-packed element (the representation name lives on the packed
/// element), and a packed tensor without a packing axis.
fn layout_of(tensor: &TensorType, plane: PlaneRef) -> MetalLayout {
    use seismic_lang::repr::PlaneEncoding;
    use seismic_lang::types::Elem;
    // Runtime extents contribute their capacity atom to the size
    // expression; the retained runtime value reaches kernels through
    // `CoreKernelOp::RuntimeExtent` reads, never the layout.
    let extent = |axis: &ExtentExpr| -> Result<Sym, CompilerDefect> {
        match axis {
            ExtentExpr::Static(n) => static_extent(*n),
            ExtentExpr::Sym(sym) => Ok(sym.clone()),
            ExtentExpr::Runtime(id) => Ok(Sym::param(&format!("@runtime{}", id.0))),
        }
    };
    let pair = || -> Result<MetalLayout, CompilerDefect> {
        match plane {
            PlaneRef::Dense { elem } => {
                let mut elements = Sym::constant(1);
                for axis in &tensor.axes {
                    elements = elements.mul(&extent(axis)?);
                }
                Ok(MetalLayout::Dense {
                    dtype: elem,
                    elements,
                    shape: tensor.axes.clone(),
                })
            }
            PlaneRef::Repr { plane: schema } => {
                let Elem::Repr(representation) = &tensor.elem else {
                    return Err(CompilerDefect::new(
                        Package::D1,
                        format!(
                            "a representation plane is declared over the non-packed element \
                             `{}`; the descriptor and the tensor disagree",
                            tensor.elem
                        ),
                    ));
                };
                let (columns, outer) = match tensor.axes.split_last() {
                    Some((last, outer)) => (extent(last)?, outer),
                    None => {
                        return Err(CompilerDefect::new(
                            Package::D1,
                            "a packed residence tensor has no packing axis",
                        ))
                    }
                };
                let mut rows = Sym::constant(1);
                for axis in outer {
                    rows = rows.mul(&extent(axis)?);
                }
                let entries = columns
                    .add(&Sym::constant(i64::from(schema.group - 1)))
                    .quot(&Sym::constant(i64::from(schema.group)))
                    .scale(i64::from(schema.fields));
                let plane_elements = match &schema.encoding {
                    PlaneEncoding::Dense(_) => entries,
                    PlaneEncoding::Packed { bits, .. } => entries
                        .scale(i64::from(*bits))
                        .add(&Sym::constant(31))
                        .quot(&Sym::constant(32)),
                };
                Ok(MetalLayout::PackedPlane {
                    representation: representation.clone(),
                    plane: schema.name.to_string(),
                    dtype: schema.dtype(),
                    elements: rows.mul(&plane_elements),
                    shape: tensor.axes.clone(),
                })
            }
        }
    };
    // The frozen `ExecutableDialect` layout signatures carry no failure
    // channel, so the two named inadmissible pairs surface here as the
    // compile-time defect arms of a total boundary (see the B1-Metal lane
    // report for the requested `Result` channel).
    pair().unwrap_or_else(|defect| panic!("{defect} during Metal layout construction"))
}

/// The `metal.subgroup` intrinsic of `name` (diagnostics and proposals).
pub fn subgroup_intrinsic(name: &str) -> IntrinsicId {
    IntrinsicId {
        capability: seismic_lang::intrinsics::CapabilityId::new("metal", "subgroup"),
        name: name.into(),
    }
}

/// The subgroup collective matching one reduction operator, `None` for
/// argmax (which never reassociates).
pub fn collective_of(op: ReduceOp) -> Option<IntrinsicId> {
    let name = match op {
        ReduceOp::Sum => "simd_sum",
        ReduceOp::Max => "simd_max",
        ReduceOp::Min => "simd_min",
        ReduceOp::Argmax => return None,
    };
    Some(subgroup_intrinsic(name))
}

/// The registry reduction schema of `op` over `input` (the encoder's fold
/// arm and the catalog's proposals read the same single owner).
pub fn schema_of(op: ReduceOp, input: DType) -> seismic_lang::intrinsics::ReduceSchema {
    reduce_schema(op, input)
}

/// Whether `op` at element dtype `dtype` composes exactly (the registry
/// admission the encoder's `Math` arm leans on for extremum semantics).
pub fn extremum_is_exact(op: MathOp) -> bool {
    matches!(op, MathOp::Sqrt | MathOp::Fma | MathOp::Abs | MathOp::Max | MathOp::Min)
}

/// Elements a device atomic can combine in one word: exactly the 32-bit
/// kinds the concurrent atomic peer admits.
pub fn device_atomic_dtype(dtype: DType) -> bool {
    matches!(dtype, DType::F32 | DType::I32 | DType::U32)
}

/// The atomic combining law of the registry: `add` rounds once per update,
/// `max`/`min` are exact and NaN-ignoring.
pub fn atomic_rounds_once(op: AtomicOp) -> bool {
    matches!(op, AtomicOp::Add)
}
