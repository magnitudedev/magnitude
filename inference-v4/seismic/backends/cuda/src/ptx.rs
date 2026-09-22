//! Exhaustive mechanical PTX emission for the closed typed kernel IR.
//! PTX is emitted only after physical legality is frozen; this module does
//! not choose algorithms, reinterpret storage, or reject a legal plan.

use crate::capability::CudaIntrinsic;
use crate::profile::CudaFacts;
use crate::Cuda;
use seismic_ir::kernel::ops::*;
use seismic_ir::kernel::{BlockId, Kernel};
use seismic_ir::storage::LaunchLocalKind;
use seismic_ir::target::{
    DenseRepresentationGeometry, KernelEmissionLayout, PackedRepresentationGeometry,
    ReadableRepresentationGeometry, RepresentationGeometry,
};
use seismic_lang::intrinsics::{AtomicOp, MathOp, ReduceOp};
use seismic_lang::registry::{
    CodeInterpretation, DecodeStep, FloatCodeFormat, PlaneEncoding, PlaneInfo, PlaneRepackRecipe,
    RepackExpr, RepresentationKind,
};
use seismic_lang::types::DType;
use seismic_target::{DeviceDescription, NativeCompilationError};
use std::collections::{BTreeSet, HashMap};

pub(crate) struct EmittedPtx {
    pub source: String,
    pub entry: String,
}

enum RepackValue {
    Integer(String),
    Float(String),
}

trait AddressGeometry: Clone {
    fn unit_bytes(&self) -> u64;
    fn packed_group(&self) -> Option<u32>;
}

impl AddressGeometry for DenseRepresentationGeometry {
    fn unit_bytes(&self) -> u64 {
        u64::from(self.dtype.bytes())
    }
    fn packed_group(&self) -> Option<u32> {
        None
    }
}

impl AddressGeometry for PackedRepresentationGeometry {
    fn unit_bytes(&self) -> u64 {
        u64::from(self.layout.packet_size)
    }
    fn packed_group(&self) -> Option<u32> {
        Some(self.layout.group)
    }
}

impl AddressGeometry for ReadableRepresentationGeometry {
    fn unit_bytes(&self) -> u64 {
        match self {
            Self::Dense(geometry) => geometry.unit_bytes(),
            Self::Packed(geometry) => geometry.unit_bytes(),
        }
    }
    fn packed_group(&self) -> Option<u32> {
        match self {
            Self::Dense(geometry) => geometry.packed_group(),
            Self::Packed(geometry) => geometry.packed_group(),
        }
    }
}

fn repack_expression_nodes(expression: &RepackExpr) -> u64 {
    match expression {
        RepackExpr::SourceBits { .. } => 1,
        RepackExpr::ShiftLeft { value, .. }
        | RepackExpr::OffsetI32 { value, .. }
        | RepackExpr::F16ToF32(value)
        | RepackExpr::I32ToF32(value) => 1 + repack_expression_nodes(value),
        RepackExpr::BitOr(left, right) | RepackExpr::MultiplyF32(left, right) => {
            1 + repack_expression_nodes(left) + repack_expression_nodes(right)
        }
    }
}

pub(crate) fn emit(
    target: &DeviceDescription<Cuda>,
    kernel: &Kernel<Cuda>,
    layout: &KernelEmissionLayout,
) -> Result<EmittedPtx, NativeCompilationError> {
    let entry = "seismic_cuda_kernel".to_string();
    let mut emitter = Emitter::new(target.facts(), kernel, layout);
    emitter.collect(kernel.root());
    emitter.header(&entry);
    emitter.emit_block(kernel.root());
    emitter.nvfp4_epilogue();
    emitter.line("ret;");
    emitter.line("}");
    let source = format!("{}\n", target.facts().ptx.header()) + &emitter.text;
    Ok(EmittedPtx { source, entry })
}

struct Emitter<'a> {
    facts: &'a CudaFacts,
    kernel: &'a Kernel<Cuda>,
    layout: &'a KernelEmissionLayout,
    values: HashMap<ErasedValue, u32>,
    ordered: BTreeSet<ErasedValue>,
    text: String,
    temp: u32,
    label: u32,
    op_count: u64,
    temp_bound: u64,
    uses_nvfp4: bool,
}

impl<'a> Emitter<'a> {
    fn new(
        facts: &'a CudaFacts,
        kernel: &'a Kernel<Cuda>,
        layout: &'a KernelEmissionLayout,
    ) -> Self {
        Self {
            facts,
            kernel,
            layout,
            values: HashMap::new(),
            ordered: BTreeSet::new(),
            text: String::new(),
            temp: 0,
            label: 0,
            op_count: 0,
            temp_bound: 32,
            uses_nvfp4: false,
        }
    }
    fn line(&mut self, value: impl AsRef<str>) {
        self.text.push_str("    ");
        self.text.push_str(value.as_ref());
        self.text.push('\n');
    }
    fn raw(&mut self, value: impl AsRef<str>) {
        self.text.push_str(value.as_ref());
        self.text.push('\n');
    }
    fn label(&mut self, stem: &str) -> String {
        let value = format!("L_{}_{}", stem, self.label);
        self.label += 1;
        value
    }
    fn t32(&mut self) -> String {
        let v = format!("%t{}", self.temp);
        self.temp += 1;
        v
    }
    fn h16(&mut self) -> String {
        let v = format!("%h{}", self.temp);
        self.temp += 1;
        v
    }
    fn t64(&mut self) -> String {
        let v = format!("%d{}", self.temp);
        self.temp += 1;
        v
    }
    fn pred(&mut self) -> String {
        let v = format!("%p{}", self.temp);
        self.temp += 1;
        v
    }
    fn f32(&mut self) -> String {
        let v = format!("%f{}", self.temp);
        self.temp += 1;
        v
    }
    fn v(&self, value: ErasedValue) -> String {
        format!("%v{}", self.values[&value])
    }

    fn vector_lane(&self, value: ErasedValue, lane: u16) -> String {
        format!("%v{}_{}", self.values[&value], lane)
    }

    fn collect(&mut self, block: BlockId) {
        for op in &self.kernel.block(block).ops {
            self.op_count = self
                .op_count
                .checked_add(1)
                .unwrap_or_else(|| panic!("typed CUDA operation count overflows u64"));
            self.temp_bound = self
                .temp_bound
                .checked_add(self.operation_temp_bound(op))
                .unwrap_or_else(|| panic!("typed CUDA temporary inventory overflows u64"));
            for value in op_values(op) {
                self.ordered.insert(value);
            }
            match op {
                Op::Intrinsic {
                    op: CudaIntrinsic::NvFp4Matmul { .. } | CudaIntrinsic::NvFp4MatmulAdd { .. },
                    ..
                } => self.uses_nvfp4 = true,
                Op::Branch {
                    then, otherwise, ..
                } => {
                    self.collect(*then);
                    self.collect(*otherwise);
                }
                Op::Repeat { body, .. } => self.collect(*body),
                _ => {}
            }
        }
    }

    fn header(&mut self, entry: &str) {
        for (index, value) in self.ordered.iter().copied().enumerate() {
            self.values.insert(value, index as u32);
        }
        self.raw(".extern .shared .align 16 .b8 seismic_shared[];");
        if self.uses_nvfp4 {
            self.raw(".shared .align 128 .b8 seismic_nvfp4_a[4096];");
            self.raw(".shared .align 128 .b8 seismic_nvfp4_b[256];");
            self.raw(".shared .align 8 .b64 seismic_nvfp4_mbarrier;");
            self.raw(".shared .align 4 .b32 seismic_nvfp4_tmem_addr;");
        }
        self.raw(format!(
            ".visible .entry {entry}(.param .u64 launch_frame) .maxnreg {} {{",
            self.facts.codegen_registers_per_thread
        ));
        let temporaries = self.temp_bound;
        self.line(format!(".reg .b16 %h<{temporaries}>;"));
        self.line(format!(".reg .b32 %t<{temporaries}>;"));
        self.line(format!(".reg .b64 %d<{temporaries}>;"));
        self.line(format!(".reg .f32 %f<{temporaries}>;"));
        self.line(format!(".reg .pred %p<{temporaries}>;"));
        for value in self.ordered.iter().copied().collect::<Vec<_>>() {
            match self.kernel.value_type(value) {
                ValueType::Vector { dtype, lanes } => {
                    for lane in 0..lanes {
                        self.line(format!(
                            ".reg {} {};",
                            scalar_ptx_type(dtype),
                            self.vector_lane(value, lane)
                        ));
                    }
                }
                ty => self.line(format!(".reg {} {};", ptx_type(ty), self.v(value))),
            }
        }
        self.line(".reg .u64 %frame, %buffers, %words, %results, %participant_base, %register_base, %linear_thread;");
        self.line("ld.param.u64 %frame, [launch_frame];");
        for (name, offset) in [
            ("%buffers", 0),
            ("%words", 8),
            ("%results", 16),
            ("%participant_base", 24),
            ("%register_base", 32),
        ] {
            self.line(format!("ld.global.u64 {name}, [%frame+{offset}];"));
        }
        self.line("mov.u32 %t0, %ctaid.x;");
        self.line("mov.u32 %t1, %ctaid.y;");
        self.line("mov.u32 %t2, %ctaid.z;");
        self.line("mov.u32 %t3, %nctaid.x;");
        self.line("mov.u32 %t4, %nctaid.y;");
        self.line("mad.lo.u32 %t5, %t2, %t4, %t1;");
        self.line("mad.lo.u32 %t6, %t5, %t3, %t0;");
        self.line("mov.u32 %t7, %tid.x;");
        self.line("mov.u32 %t8, %tid.y;");
        self.line("mov.u32 %t9, %tid.z;");
        self.line("mov.u32 %t10, %ntid.x;");
        self.line("mov.u32 %t11, %ntid.y;");
        self.line("mad.lo.u32 %t12, %t9, %t11, %t8;");
        self.line("mad.lo.u32 %t13, %t12, %t10, %t7;");
        self.line("mov.u32 %t14, %ntid.z;");
        self.line("mul.lo.u32 %t15, %t10, %t11;");
        self.line("mul.lo.u32 %t15, %t15, %t14;");
        self.line("mul.wide.u32 %linear_thread, %t6, %t15;");
        self.line("cvt.u64.u32 %d16, %t13;");
        self.line("add.u64 %linear_thread, %linear_thread, %d16;");
        self.temp = 32;
    }

    fn operation_temp_bound(&self, op: &Op<Cuda>) -> u64 {
        match op {
            Op::Read { place, index, .. } => {
                let geometry = self.geometry_of(*place);
                let decode = match &geometry.info.kind {
                    RepresentationKind::Packed(_) => {
                        geometry
                            .decode
                            .as_ref()
                            .expect("packed CUDA geometry has no decode recipe")
                            .steps()
                            .len() as u64
                            * 8
                    }
                    RepresentationKind::Dense(_) => 4,
                    RepresentationKind::External(_) => 4,
                };
                24 + index.len() as u64 * 4 + decode
            }
            Op::ReadPlane { index, .. } | Op::Write { index, .. } | Op::Atomic { index, .. } => {
                32 + index.len() as u64 * 4
            }
            Op::VectorRead { out, index, .. } => {
                let lanes = match self.kernel.value_type(*out) {
                    ValueType::Vector { lanes, .. } => u64::from(lanes),
                    _ => panic!("typed CUDA vector read has a scalar result"),
                };
                let geometry = self.geometry_of(match op {
                    Op::VectorRead { place, .. } => *place,
                    _ => unreachable!(),
                });
                let decode = match &geometry.info.kind {
                    RepresentationKind::Packed(_) => {
                        geometry
                            .decode
                            .as_ref()
                            .expect("packed CUDA geometry has no decode recipe")
                            .steps()
                            .len() as u64
                            * 8
                    }
                    RepresentationKind::Dense(_) | RepresentationKind::External(_) => 4,
                };
                lanes * (32 + index.len() as u64 * 4 + decode)
            }
            Op::VectorWrite { value, index, .. } => {
                let lanes = match self.kernel.value_type(*value) {
                    ValueType::Vector { lanes, .. } => u64::from(lanes),
                    _ => panic!("typed CUDA vector write has a scalar value"),
                };
                lanes * (32 + index.len() as u64 * 4)
            }
            Op::RepresentationConvertPacket { conversion, .. } => {
                let conversion =
                    seismic_lang::registry::representation_conversion_info(*conversion);
                conversion.recipe.planes.iter().fold(32u64, |total, plane| {
                    let plane_bound = match plane {
                        PlaneRepackRecipe::BitRoutes(routes) => {
                            routes.len() as u64 * 8 + routes.len().div_ceil(8) as u64 * 2
                        }
                        PlaneRepackRecipe::DenseValues(values) => {
                            values.iter().map(repack_expression_nodes).sum::<u64>() * 4
                        }
                    };
                    total
                        .checked_add(plane_bound)
                        .expect("typed CUDA repack temporary inventory overflows u64")
                })
            }
            Op::Intrinsic {
                op: CudaIntrinsic::MatrixMatmul { .. } | CudaIntrinsic::MatrixMatmulAdd { .. },
                ..
            } => 128,
            Op::Intrinsic {
                op: CudaIntrinsic::NvFp4Matmul { .. } | CudaIntrinsic::NvFp4MatmulAdd { .. },
                ..
            } => 512,
            Op::Branch { .. } | Op::Repeat { .. } => 16,
            _ => 24,
        }
    }

    fn geometry_of(&self, place: PlaceRef) -> RepresentationGeometry {
        self.kernel.closed_place(place, self.layout).geometry
    }

    fn word(&mut self, index: u32) -> String {
        let out = self.t64();
        self.line(format!("ld.global.u64 {out}, [%words+{}];", index * 8));
        out
    }
    fn buffer(&mut self, position: usize) -> String {
        let out = self.t64();
        self.line(format!("ld.global.u64 {out}, [%buffers+{}];", position * 8));
        out
    }

    fn emit_block(&mut self, block: BlockId) -> Option<Vec<ErasedValue>> {
        for op in &self.kernel.block(block).ops {
            match self.kernel.closed_op(op, self.layout) {
                ClosedOpView::Constant { out, value } => self.constant(out.value, out.ty, value),
                ClosedOpView::Binary { op, out, a, b } => {
                    self.binary(op, out.value, out.ty, a.value, b.value)
                }
                ClosedOpView::Unary { op, out, a } => self.unary(op, out.value, out.ty, a.value),
                ClosedOpView::Bit { op, out, a, b } => {
                    self.bit(op, out.value, out.ty, a.value, b.value)
                }
                ClosedOpView::Fma { out, a, b, c } => {
                    self.line(format!(
                        "fma.rn.f32 {}, {}, {}, {};",
                        self.v(out.value),
                        self.v(a.value),
                        self.v(b.value),
                        self.v(c.value)
                    ));
                    // Kernel values use f32 registers for all floating
                    // dtypes. A narrow FMA result therefore needs the same
                    // explicit publication rounding boundary as every other
                    // narrow arithmetic operation.
                    self.round(out.value, out.ty);
                }
                ClosedOpView::VectorSplat { out, value } => {
                    let (_, lanes) = vector_shape(out.ty);
                    for lane in 0..lanes {
                        self.line(format!(
                            "mov{} {}, {};",
                            suffix(value.ty),
                            self.vector_lane(out.value, lane),
                            self.v(value.value)
                        ));
                    }
                }
                ClosedOpView::VectorBinary { op, out, a, b } => self.vector_binary(op, out, a, b),
                ClosedOpView::VectorUnary { op, out, a } => self.vector_unary(op, out, a),
                ClosedOpView::VectorBit { op, out, a, b } => self.vector_bit(op, out, a, b),
                ClosedOpView::VectorFma { out, a, b, c } => self.vector_fma(out, a, b, c),
                ClosedOpView::VectorCast { out, a, to } => self.vector_cast(out, a, to),
                ClosedOpView::VectorLane { out, vector, lane } => self.line(format!(
                    "mov{} {}, {};",
                    suffix(out.ty),
                    self.v(out.value),
                    self.vector_lane(vector.value, lane)
                )),
                ClosedOpView::VectorReduceAdd { out, vector } => {
                    self.vector_reduce_add(out, vector)
                }
                ClosedOpView::ApproximateMath { op, out, a } => {
                    self.math(op, out.value, out.ty, a.value)
                }
                ClosedOpView::Cast { out, a, to } => self.cast(out.value, a.value, a.ty, to),
                ClosedOpView::Bitcast { out, a, to } => self.line(format!(
                    "mov.b{} {}, {};",
                    if to == ValueType::Index { 64 } else { 32 },
                    self.v(out.value),
                    self.v(a.value)
                )),
                ClosedOpView::Cmp { op, out, a, b } => {
                    self.compare(op, out.value(), a.value, a.ty, b.value)
                }
                ClosedOpView::Select {
                    out,
                    condition,
                    a,
                    b,
                } => {
                    let p = self.truth_bool(condition.value());
                    self.line(format!(
                        "selp{} {}, {}, {}, {p};",
                        suffix(out.ty),
                        self.v(out.value),
                        self.v(a.value),
                        self.v(b.value)
                    ));
                }
                ClosedOpView::Logic { op, out, a, b } => self.line(format!(
                    "{}.b32 {}, {}, {};",
                    match op {
                        LogicOp::And => "and",
                        LogicOp::Or => "or",
                    },
                    self.v(out.value()),
                    self.v(a.value()),
                    self.v(b.value())
                )),
                ClosedOpView::Not { out, a } => self.line(format!(
                    "xor.b32 {}, {}, 1;",
                    self.v(out.value()),
                    self.v(a.value())
                )),
                ClosedOpView::Geometry { out, kind } => self.geometry(out.value(), kind),
                ClosedOpView::NatArg { out, index, .. } => {
                    let value = self.word(self.layout.words.nat_first + index);
                    self.line(format!("mov.u64 {}, {value};", self.v(out.value())));
                }
                ClosedOpView::ScalarArg {
                    out, index, dtype, ..
                } => self.scalar_arg(out.value, index, dtype),
                ClosedOpView::Read {
                    out,
                    place,
                    indices,
                } => self.read(
                    out.value,
                    out.ty,
                    &place,
                    &indices
                        .iter()
                        .map(|value| value.value())
                        .collect::<Vec<_>>(),
                ),
                ClosedOpView::VectorRead {
                    out,
                    place,
                    indices,
                    axis,
                    active,
                } => self.vector_read(out, &place, &indices, axis, active),
                ClosedOpView::VectorWrite {
                    place,
                    indices,
                    axis,
                    active,
                    value,
                } => self.vector_write(&place, &indices, axis, active, value),
                ClosedOpView::ReadPlane {
                    out,
                    place,
                    plane_info,
                    indices,
                    ..
                } => self.read_plane(
                    out.value,
                    out.ty,
                    &place,
                    &plane_info,
                    &indices
                        .iter()
                        .map(|value| value.value())
                        .collect::<Vec<_>>(),
                ),
                ClosedOpView::RepresentationConvertPacket {
                    source,
                    destination,
                    recipe,
                    packet,
                    ..
                } => self.convert_packet(&source, &destination, &recipe.recipe, packet.value()),
                ClosedOpView::Write {
                    place,
                    indices,
                    value,
                } => self.write(
                    &place,
                    &indices
                        .iter()
                        .map(|value| value.value())
                        .collect::<Vec<_>>(),
                    value.value,
                ),
                ClosedOpView::Extent { out, place, axis } => {
                    let (_, first, _, _) = self.place(&place);
                    let value = self.word(first + axis);
                    self.line(format!("mov.u64 {}, {value};", self.v(out.value())));
                }
                ClosedOpView::Atomic {
                    op,
                    place,
                    indices,
                    value,
                } => self.atomic(
                    op,
                    &place,
                    &indices
                        .iter()
                        .map(|value| value.value())
                        .collect::<Vec<_>>(),
                    value.value,
                ),
                ClosedOpView::StoreSlot {
                    slot,
                    dtype,
                    value,
                    election: StoreElection::GlobalLeader,
                } => self.store_slot(slot, dtype, value),
                ClosedOpView::Barrier(BarrierScope::Workgroup) => self.line("bar.sync 0;"),
                ClosedOpView::Barrier(BarrierScope::Subgroup) => {
                    self.line("bar.warp.sync 0xffffffff;")
                }
                ClosedOpView::Intrinsic {
                    op,
                    outputs,
                    arguments,
                    ..
                } => self.intrinsic(op, &outputs, &arguments),
                ClosedOpView::Branch {
                    condition,
                    then_block,
                    else_block,
                    outputs,
                    ..
                } => self.branch(condition.value(), then_block, else_block, &outputs),
                ClosedOpView::Repeat {
                    start,
                    end,
                    binder,
                    carries_in,
                    carry_parameters,
                    body,
                    outputs,
                    ..
                } => self.repeat(
                    start.value(),
                    end.value(),
                    binder.value(),
                    &carries_in,
                    &carry_parameters,
                    body,
                    &outputs,
                ),
                ClosedOpView::Yield { values } => {
                    return Some(values.iter().map(|value| value.value).collect());
                }
            }
        }
        None
    }

    fn constant(&mut self, out: ErasedValue, out_type: ValueType, value: ConstantValue) {
        let value = match value {
            ConstantValue::F32(v) => format!("0f{:08x}", v.to_bits()),
            ConstantValue::F16(v) => {
                format!("0f{:08x}", seismic_lang::registry::f16_to_f32(v).to_bits())
            }
            ConstantValue::BF16(v) => format!("0f{:08x}", u32::from(v) << 16),
            ConstantValue::I32(v) => v.to_string(),
            ConstantValue::U32(v) => v.to_string(),
            ConstantValue::Bool(v) => u8::from(v).to_string(),
            ConstantValue::Index(v) => v.to_string(),
        };
        self.line(format!("mov{} {}, {value};", suffix(out_type), self.v(out)));
    }
    fn binary(
        &mut self,
        op: BinaryOp,
        out: ErasedValue,
        ty: ValueType,
        a: ErasedValue,
        b: ErasedValue,
    ) {
        let floating = matches!(ty, ValueType::Scalar(d) if d.is_float());
        let mnemonic = match (op, floating, ty == ValueType::Scalar(DType::I32)) {
            (BinaryOp::Add, true, _) => "add.rn.f32",
            (BinaryOp::Sub, true, _) => "sub.rn.f32",
            (BinaryOp::Mul, true, _) => "mul.rn.f32",
            (BinaryOp::Div, true, _) => "div.rn.f32",
            (BinaryOp::Min, true, _) => "min.f32",
            (BinaryOp::Max, true, _) => "max.f32",
            (BinaryOp::Add, false, _) => {
                if ty == ValueType::Index {
                    "add.u64"
                } else {
                    "add.u32"
                }
            }
            (BinaryOp::Sub, false, _) => {
                if ty == ValueType::Index {
                    "sub.u64"
                } else {
                    "sub.u32"
                }
            }
            (BinaryOp::Mul, false, _) => {
                if ty == ValueType::Index {
                    "mul.lo.u64"
                } else {
                    "mul.lo.u32"
                }
            }
            (BinaryOp::Div, false, true) => "div.s32",
            (BinaryOp::Div, false, _) => {
                if ty == ValueType::Index {
                    "div.u64"
                } else {
                    "div.u32"
                }
            }
            (BinaryOp::Rem, false, true) => "rem.s32",
            (BinaryOp::Rem, false, _) => {
                if ty == ValueType::Index {
                    "rem.u64"
                } else {
                    "rem.u32"
                }
            }
            (BinaryOp::Min, false, true) => "min.s32",
            (BinaryOp::Max, false, true) => "max.s32",
            (BinaryOp::Min, false, _) => {
                if ty == ValueType::Index {
                    "min.u64"
                } else {
                    "min.u32"
                }
            }
            (BinaryOp::Max, false, _) => {
                if ty == ValueType::Index {
                    "max.u64"
                } else {
                    "max.u32"
                }
            }
            (BinaryOp::Rem, true, _) => panic!("floating remainder entered typed kernel IR"),
        };
        self.line(format!(
            "{mnemonic} {}, {}, {};",
            self.v(out),
            self.v(a),
            self.v(b)
        ));
        if matches!(op, BinaryOp::Div | BinaryOp::Rem) && ty == ValueType::Scalar(DType::I32) {
            self.euclidean_fix(op, out, a, b);
        }
        self.round(out, ty);
    }
    fn euclidean_fix(
        &mut self,
        wanted: BinaryOp,
        out: ErasedValue,
        a: ErasedValue,
        b: ErasedValue,
    ) {
        let q = self.t32();
        let r = self.t32();
        let p = self.pred();
        let abs = self.t32();
        let adj = self.t32();
        self.line(format!("div.s32 {q}, {}, {};", self.v(a), self.v(b)));
        self.line(format!("rem.s32 {r}, {}, {};", self.v(a), self.v(b)));
        self.line(format!("setp.lt.s32 {p}, {r}, 0;"));
        self.line(format!("abs.s32 {abs}, {};", self.v(b)));
        match wanted {
            BinaryOp::Div => {
                let sign = self.t32();
                self.line(format!("shr.s32 {sign}, {}, 31;", self.v(b)));
                self.line(format!("or.b32 {sign}, {sign}, 1;"));
                self.line(format!("sub.s32 {adj}, {q}, {sign};"));
                self.line(format!("selp.s32 {}, {adj}, {q}, {p};", self.v(out)));
            }
            BinaryOp::Rem => {
                self.line(format!("add.s32 {adj}, {r}, {abs};"));
                self.line(format!("selp.s32 {}, {adj}, {r}, {p};", self.v(out)));
            }
            _ => {}
        }
    }
    fn unary(&mut self, op: UnaryOp, out: ErasedValue, ty: ValueType, a: ErasedValue) {
        let float = matches!(ty,ValueType::Scalar(d) if d.is_float());
        self.line(format!(
            "{}{} {}, {};",
            match op {
                UnaryOp::Neg => "neg",
                UnaryOp::Abs => "abs",
            },
            if float { ".f32" } else { ".s32" },
            self.v(out),
            self.v(a)
        ));
        self.round(out, ty);
    }
    fn bit(&mut self, op: BitOp, out: ErasedValue, ty: ValueType, a: ErasedValue, b: ErasedValue) {
        let m = match op {
            BitOp::And => "and.b32",
            BitOp::Or => "or.b32",
            BitOp::Xor => "xor.b32",
            BitOp::Shl => {
                if ty == ValueType::Index {
                    "shl.b64"
                } else {
                    "shl.b32"
                }
            }
            BitOp::Shr if ty == ValueType::Scalar(DType::I32) => "shr.s32",
            BitOp::Shr => {
                if ty == ValueType::Index {
                    "shr.u64"
                } else {
                    "shr.u32"
                }
            }
        };
        self.line(format!(
            "{m} {}, {}, {};",
            self.v(out),
            self.v(a),
            self.v(b)
        ));
    }

    fn vector_binary(&mut self, op: BinaryOp, out: ClosedValue, a: ClosedValue, b: ClosedValue) {
        let (dtype, lanes) = vector_shape(out.ty);
        for lane in 0..lanes {
            let destination = self.vector_lane(out.value, lane);
            let left = self.vector_lane(a.value, lane);
            let right = self.vector_lane(b.value, lane);
            let floating = dtype.is_float();
            let mnemonic = match (op, floating, dtype == DType::I32) {
                (BinaryOp::Add, true, _) => "add.rn.f32",
                (BinaryOp::Sub, true, _) => "sub.rn.f32",
                (BinaryOp::Mul, true, _) => "mul.rn.f32",
                (BinaryOp::Div, true, _) => "div.rn.f32",
                (BinaryOp::Min, true, _) => "min.f32",
                (BinaryOp::Max, true, _) => "max.f32",
                (BinaryOp::Add, false, _) => "add.u32",
                (BinaryOp::Sub, false, _) => "sub.u32",
                (BinaryOp::Mul, false, _) => "mul.lo.u32",
                (BinaryOp::Div, false, true) => "div.s32",
                (BinaryOp::Div, false, false) => "div.u32",
                (BinaryOp::Rem, false, true) => "rem.s32",
                (BinaryOp::Rem, false, false) => "rem.u32",
                (BinaryOp::Min, false, true) => "min.s32",
                (BinaryOp::Max, false, true) => "max.s32",
                (BinaryOp::Min, false, false) => "min.u32",
                (BinaryOp::Max, false, false) => "max.u32",
                (BinaryOp::Rem, true, _) => {
                    panic!("floating vector remainder entered typed kernel IR")
                }
            };
            self.line(format!("{mnemonic} {destination}, {left}, {right};"));
            if matches!(op, BinaryOp::Div | BinaryOp::Rem) && dtype == DType::I32 {
                self.euclidean_fix_named(op, &destination, &left, &right);
            }
            self.round_named(&destination, dtype);
        }
    }

    fn vector_unary(&mut self, op: UnaryOp, out: ClosedValue, a: ClosedValue) {
        let (dtype, lanes) = vector_shape(out.ty);
        for lane in 0..lanes {
            let destination = self.vector_lane(out.value, lane);
            self.line(format!(
                "{}{} {destination}, {};",
                match op {
                    UnaryOp::Neg => "neg",
                    UnaryOp::Abs => "abs",
                },
                if dtype.is_float() { ".f32" } else { ".s32" },
                self.vector_lane(a.value, lane)
            ));
            self.round_named(&destination, dtype);
        }
    }

    fn vector_bit(&mut self, op: BitOp, out: ClosedValue, a: ClosedValue, b: ClosedValue) {
        let (dtype, lanes) = vector_shape(out.ty);
        let mnemonic = match op {
            BitOp::And => "and.b32",
            BitOp::Or => "or.b32",
            BitOp::Xor => "xor.b32",
            BitOp::Shl => "shl.b32",
            BitOp::Shr if dtype == DType::I32 => "shr.s32",
            BitOp::Shr => "shr.u32",
        };
        for lane in 0..lanes {
            self.line(format!(
                "{mnemonic} {}, {}, {};",
                self.vector_lane(out.value, lane),
                self.vector_lane(a.value, lane),
                self.vector_lane(b.value, lane)
            ));
        }
    }

    fn vector_fma(&mut self, out: ClosedValue, a: ClosedValue, b: ClosedValue, c: ClosedValue) {
        let (dtype, lanes) = vector_shape(out.ty);
        for lane in 0..lanes {
            let destination = self.vector_lane(out.value, lane);
            self.line(format!(
                "fma.rn.f32 {destination}, {}, {}, {};",
                self.vector_lane(a.value, lane),
                self.vector_lane(b.value, lane),
                self.vector_lane(c.value, lane)
            ));
            self.round_named(&destination, dtype);
        }
    }

    fn vector_cast(&mut self, out: ClosedValue, a: ClosedValue, to: ValueType) {
        let (from_dtype, lanes) = vector_shape(a.ty);
        let (to_dtype, to_lanes) = vector_shape(to);
        assert_eq!(lanes, to_lanes, "typed CUDA vector cast changes lane count");
        for lane in 0..lanes {
            let destination = self.vector_lane(out.value, lane);
            let source = self.vector_lane(a.value, lane);
            self.cast_named(&destination, &source, from_dtype, to_dtype);
        }
    }

    fn vector_reduce_add(&mut self, out: ClosedValue, vector: ClosedValue) {
        let (dtype, lanes) = vector_shape(vector.ty);
        let destination = self.v(out.value);
        self.line(format!(
            "mov{} {destination}, {};",
            scalar_suffix(dtype),
            self.vector_lane(vector.value, 0)
        ));
        for lane in 1..lanes {
            self.line(format!(
                "{} {destination}, {destination}, {};",
                if dtype.is_float() {
                    "add.rn.f32"
                } else {
                    "add.u32"
                },
                self.vector_lane(vector.value, lane)
            ));
            self.round_named(&destination, dtype);
        }
    }

    fn euclidean_fix_named(&mut self, wanted: BinaryOp, out: &str, a: &str, b: &str) {
        let q = self.t32();
        let r = self.t32();
        let p = self.pred();
        let abs = self.t32();
        let adjusted = self.t32();
        self.line(format!("div.s32 {q}, {a}, {b};"));
        self.line(format!("rem.s32 {r}, {a}, {b};"));
        self.line(format!("setp.lt.s32 {p}, {r}, 0;"));
        self.line(format!("abs.s32 {abs}, {b};"));
        match wanted {
            BinaryOp::Div => {
                let sign = self.t32();
                self.line(format!("shr.s32 {sign}, {b}, 31;"));
                self.line(format!("or.b32 {sign}, {sign}, 1;"));
                self.line(format!("sub.s32 {adjusted}, {q}, {sign};"));
                self.line(format!("selp.s32 {out}, {adjusted}, {q}, {p};"));
            }
            BinaryOp::Rem => {
                self.line(format!("add.s32 {adjusted}, {r}, {abs};"));
                self.line(format!("selp.s32 {out}, {adjusted}, {r}, {p};"));
            }
            _ => unreachable!("euclidean fix is only emitted for division/remainder"),
        }
    }

    fn cast_named(&mut self, out: &str, value: &str, from: DType, to: DType) {
        if from == to {
            self.line(format!("mov{} {out}, {value};", scalar_suffix(to)));
        } else if from.is_float() && to.is_float() {
            self.line(format!("mov.f32 {out}, {value};"));
        } else if from.is_float() {
            self.line(format!(
                "cvt.rzi{}.f32 {out}, {value};",
                if to == DType::I32 { ".s32" } else { ".u32" }
            ));
        } else if to.is_float() {
            self.line(format!(
                "cvt.rn.f32{} {out}, {value};",
                if from == DType::I32 { ".s32" } else { ".u32" }
            ));
        } else {
            self.line(format!("mov.b32 {out}, {value};"));
        }
        self.round_named(out, to);
    }
    fn math(&mut self, op: MathOp, out: ErasedValue, out_type: ValueType, a: ErasedValue) {
        match op {
            MathOp::Sqrt => self.line(format!("sqrt.rn.f32 {}, {};", self.v(out), self.v(a))),
            MathOp::Rsqrt => self.line(format!("rsqrt.approx.f32 {}, {};", self.v(out), self.v(a))),
            MathOp::Abs => self.line(format!("abs.f32 {}, {};", self.v(out), self.v(a))),
            MathOp::ExpFast | MathOp::Exp => {
                let scaled = self.f32();
                self.line(format!("mul.rn.f32 {scaled}, {}, 0f3fb8aa3b;", self.v(a)));
                self.line(format!("ex2.approx.f32 {}, {scaled};", self.v(out)));
            }
            MathOp::Log => self.line(format!("lg2.approx.f32 {}, {};", self.v(out), self.v(a))),
            MathOp::Sin => self.line(format!("sin.approx.f32 {}, {};", self.v(out), self.v(a))),
            MathOp::Cos => self.line(format!("cos.approx.f32 {}, {};", self.v(out), self.v(a))),
            MathOp::Max | MathOp::Min | MathOp::Fma => {
                panic!("multi-operand math op encoded as unary IR")
            }
        }
        self.round(out, out_type);
    }
    fn cast(&mut self, out: ErasedValue, a: ErasedValue, from: ValueType, to: ValueType) {
        if from == to {
            self.line(format!("mov{} {}, {};", suffix(to), self.v(out), self.v(a)));
            return;
        }
        let ff = matches!(from,ValueType::Scalar(d)if d.is_float());
        let tf = matches!(to,ValueType::Scalar(d)if d.is_float());
        let fb = matches!(from, ValueType::Bool | ValueType::Scalar(DType::Bool));
        let tb = matches!(to, ValueType::Bool | ValueType::Scalar(DType::Bool));
        if tb {
            let p = self.pred();
            self.line(format!(
                "setp.ne{} {p}, {}, {};",
                if ff {
                    ".f32"
                } else if from == ValueType::Index {
                    ".u64"
                } else {
                    ".u32"
                },
                self.v(a),
                if ff { "0f00000000" } else { "0" }
            ));
            self.line(format!("selp.u32 {}, 1, 0, {p};", self.v(out)));
        } else if ff && tf {
            self.line(format!("mov.f32 {}, {};", self.v(out), self.v(a)));
        } else if ff {
            self.line(format!(
                "cvt.rzi{}{}.f32 {}, {};",
                if to == ValueType::Index {
                    ".u64"
                } else if to == ValueType::Scalar(DType::I32) {
                    ".s32"
                } else {
                    ".u32"
                },
                "",
                self.v(out),
                self.v(a)
            ));
        } else if tf {
            self.line(format!(
                "cvt.rn.f32{} {}, {};",
                if from == ValueType::Scalar(DType::I32) {
                    ".s32"
                } else if from == ValueType::Index {
                    ".u64"
                } else {
                    ".u32"
                },
                self.v(out),
                self.v(a)
            ));
        } else if fb || to == ValueType::Index || from == ValueType::Index {
            self.line(format!(
                "cvt{}{} {}, {};",
                suffix(to),
                suffix(from),
                self.v(out),
                self.v(a)
            ));
        } else {
            self.line(format!("mov.b32 {}, {};", self.v(out), self.v(a)));
        }
        self.round(out, to);
    }
    fn compare(
        &mut self,
        op: CmpOp,
        out: ErasedValue,
        a: ErasedValue,
        ty: ValueType,
        b: ErasedValue,
    ) {
        let p = self.pred();
        let cmp = match op {
            CmpOp::Eq => "eq",
            CmpOp::Ne => "ne",
            CmpOp::Lt => "lt",
            CmpOp::Le => "le",
            CmpOp::Gt => "gt",
            CmpOp::Ge => "ge",
        };
        let class = match ty {
            ValueType::Scalar(d) if d.is_float() => "f32",
            ValueType::Scalar(DType::I32) => "s32",
            ValueType::Index => "u64",
            _ => "u32",
        };
        self.line(format!(
            "setp.{cmp}.{class} {p}, {}, {};",
            self.v(a),
            self.v(b)
        ));
        self.line(format!("selp.u32 {}, 1, 0, {p};", self.v(out)));
    }
    fn truth_bool(&mut self, value: ErasedValue) -> String {
        let p = self.pred();
        self.line(format!("setp.ne.u32 {p}, {}, 0;", self.v(value)));
        p
    }
    fn geometry(&mut self, out: ErasedValue, kind: GeometryValue) {
        let instr = match kind {
            GeometryValue::WorkgroupId(a) => format!("mov.u32 %t31, %ctaid.{};", axis(a)),
            GeometryValue::LocalId(a) => format!("mov.u32 %t31, %tid.{};", axis(a)),
            GeometryValue::WorkgroupSize(a) => format!("mov.u32 %t31, %ntid.{};", axis(a)),
            GeometryValue::GridSize(a) => format!("mov.u32 %t31, %nctaid.{};", axis(a)),
            GeometryValue::SubgroupLane => "mov.u32 %t31, %laneid;".into(),
            GeometryValue::GlobalId(a) => {
                let x = self.t32();
                self.line(format!("mov.u32 {x}, %ctaid.{};", axis(a)));
                let y = self.t32();
                self.line(format!("mov.u32 {y}, %ntid.{};", axis(a)));
                let z = self.t32();
                self.line(format!("mov.u32 {z}, %tid.{};", axis(a)));
                self.line(format!("mad.lo.u32 %t31, {x}, {y}, {z};"));
                String::new()
            }
        };
        if !instr.is_empty() {
            self.line(instr)
        }
        self.line(format!("cvt.u64.u32 {}, %t31;", self.v(out)));
    }
    fn scalar_arg(&mut self, out: ErasedValue, index: u32, dtype: DType) {
        let raw = self.word(self.layout.words.scalar_first + index);
        match dtype {
            DType::F32 => {
                let bits = self.t32();
                self.line(format!("cvt.u32.u64 {bits}, {raw};"));
                self.line(format!("mov.b32 {}, {bits};", self.v(out)));
            }
            DType::F16 => {
                let bits = self.h16();
                self.line(format!("cvt.u16.u64 {bits}, {raw};"));
                self.line(format!("cvt.f32.f16 {}, {bits};", self.v(out)));
            }
            DType::BF16 => {
                let bits = self.h16();
                self.line(format!("cvt.u16.u64 {bits}, {raw};"));
                self.line(format!("cvt.f32.bf16 {}, {bits};", self.v(out)));
            }
            DType::I32 | DType::U32 | DType::Bool => {
                self.line(format!("cvt.u32.u64 {}, {raw};", self.v(out)))
            }
        }
    }

    fn place<G: Clone>(&mut self, place: &ClosedPlace<G>) -> (G, u32, u32, String) {
        match (place.kind, place.words) {
            (ClosedPlaceKind::Global { buffer_ordinal, .. }, ClosedPlaceWords::Binding(words)) => (
                place.geometry.clone(),
                words.first,
                words.rank,
                self.buffer(buffer_ordinal as usize),
            ),
            (ClosedPlaceKind::Local { kind, .. }, ClosedPlaceWords::Local(words)) => {
                let offset = self.word(words.first);
                let base = match kind {
                    LaunchLocalKind::Workgroup => {
                        let p = self.t64();
                        self.line(format!("mov.u64 {p}, seismic_shared;"));
                        p
                    }
                    kind @ (LaunchLocalKind::Participant | LaunchLocalKind::Register) => {
                        let class = if kind == LaunchLocalKind::Participant {
                            1
                        } else {
                            2
                        };
                        let total = self.word(self.layout.words.local_total_first + class);
                        let delta = self.t64();
                        self.line(format!("mul.lo.u64 {delta}, %linear_thread, {total};"));
                        let p = self.t64();
                        self.line(format!(
                            "add.u64 {p}, {}, {delta};",
                            if kind == LaunchLocalKind::Participant {
                                "%participant_base"
                            } else {
                                "%register_base"
                            }
                        ));
                        p
                    }
                };
                let pointer = self.t64();
                self.line(format!("add.u64 {pointer}, {base}, {offset};"));
                (place.geometry.clone(), words.first + 1, words.rank, pointer)
            }
            _ => panic!("closed CUDA place kind and word layout disagree"),
        }
    }

    fn place_raw(&mut self, place: PlaceRef) -> (RepresentationGeometry, u32, u32, String) {
        let closed = self.kernel.closed_place(place, self.layout);
        self.place(&closed)
    }
    fn dense_place(&self, place: PlaceRef) -> ClosedDensePlace {
        self.kernel.closed_dense_place(place, self.layout)
    }
    fn readable_place(&self, place: PlaceRef) -> ClosedReadablePlace {
        self.kernel.closed_readable_place(place, self.layout)
    }

    fn logical_coords(&self, tensor: &LogicalTensorMap, coordinates: &[String]) -> Vec<String> {
        let mut coordinates = coordinates.to_vec();
        for step in tensor.steps.iter().rev() {
            coordinates = match step {
                LogicalViewStep::Slice(axes) => {
                    let mut logical = coordinates.iter();
                    axes.iter()
                        .map(|axis| match axis {
                            LogicalSliceAxis::Point(value) => self.v(*value),
                            LogicalSliceAxis::Range { start, .. } => format!(
                                "({} + {})",
                                self.v(*start),
                                logical.next().expect("closed slice-map rank")
                            ),
                            LogicalSliceAxis::Full => {
                                logical.next().expect("closed slice-map rank").clone()
                            }
                        })
                        .collect()
                }
                LogicalViewStep::Transpose(permutation) => {
                    let mut physical = vec![String::new(); permutation.len()];
                    for (source_axis, coordinate) in permutation.iter().zip(&coordinates) {
                        physical[*source_axis as usize] = coordinate.clone();
                    }
                    physical
                }
                LogicalViewStep::Reshape { from, to } => {
                    let mut linear = "0".to_string();
                    for (coordinate, extent) in coordinates.iter().zip(to) {
                        linear = format!("(({linear}) * {} + ({coordinate}))", self.v(*extent));
                    }
                    let mut physical = vec![String::new(); from.len()];
                    for axis in (0..from.len()).rev() {
                        physical[axis] = format!("(({linear}) % {})", self.v(from[axis]));
                        linear = format!("(({linear}) / {})", self.v(from[axis]));
                    }
                    physical
                }
            };
        }
        coordinates
    }
    fn address<G: AddressGeometry>(
        &mut self,
        place: &ClosedPlace<G>,
        index: &[ErasedValue],
    ) -> (String, G, String) {
        let names = index.iter().map(|value| self.v(*value)).collect::<Vec<_>>();
        self.address_names(place, &names)
    }

    fn address_names<G: AddressGeometry>(
        &mut self,
        place: &ClosedPlace<G>,
        index: &[String],
    ) -> (String, G, String) {
        let (geometry, first, rank, base) = self.place(place);
        if index.len() != rank as usize {
            panic!("typed CUDA memory rank differs from place")
        };
        let mut units = self.t64();
        self.line(format!("mov.u64 {units}, 0;"));
        for (axis, value) in index.iter().enumerate() {
            let mut coord = value.clone();
            if axis + 1 == index.len() {
                if let Some(group) = geometry.packed_group() {
                    let q = self.t64();
                    self.line(format!("div.u64 {q}, {coord}, {group};"));
                    coord = q;
                }
            }
            let stride = self.word(first + rank + axis as u32);
            let term = self.t64();
            self.line(format!("mul.lo.u64 {term}, {coord}, {stride};"));
            let next = self.t64();
            self.line(format!("add.u64 {next}, {units}, {term};"));
            units = next;
        }
        let bytes = geometry.unit_bytes();
        let delta = self.t64();
        self.line(format!("mul.lo.u64 {delta}, {units}, {bytes};"));
        let address = self.t64();
        self.line(format!("add.u64 {address}, {base}, {delta};"));
        let logical = index.last().cloned().unwrap_or_else(|| "0".into());
        (address, geometry, logical)
    }
    fn read(
        &mut self,
        out: ErasedValue,
        out_type: ValueType,
        place: &ClosedReadablePlace,
        index: &[ErasedValue],
    ) {
        let destination = self.v(out);
        self.read_to(&destination, out_type, place, index);
    }

    fn read_to(
        &mut self,
        destination: &str,
        out_type: ValueType,
        place: &ClosedReadablePlace,
        index: &[ErasedValue],
    ) {
        let (address, geometry, logical) = self.address(place, index);
        match &geometry {
            ReadableRepresentationGeometry::Dense(dense) => {
                self.load_to(destination, out_type, &address, dense.dtype)
            }
            ReadableRepresentationGeometry::Packed(packed) => {
                let layout = &packed.layout;
                let recipe = &packed.decode;
                let mut temps: Vec<String> = Vec::with_capacity(recipe.temporary_count());
                for step in recipe.steps() {
                    let value = match step {
                        DecodeStep::ReadPlaneField { into, plane, field } => {
                            let _ = into;
                            self.plane_field(
                                &address,
                                &logical,
                                layout.group,
                                &layout.planes[*plane as usize],
                                *field,
                                recipe.dtype(*into),
                            )
                        }
                        DecodeStep::InterpretCode {
                            into,
                            raw,
                            bits,
                            interpretation,
                        } => {
                            let _ = into;
                            self.interpret(&temps[recipe.ordinal(*raw)], *bits, interpretation)
                        }
                        DecodeStep::DecodeFloatCode { into, raw, format } => {
                            let _ = into;
                            self.decode_float_code(&temps[recipe.ordinal(*raw)], *format)
                        }
                        DecodeStep::ConvertToF32 { into, from } => {
                            let _ = into;
                            let value = self.f32();
                            let from_dtype = recipe.dtype(*from);
                            if from_dtype.is_float() {
                                self.line(format!(
                                    "mov.f32 {value}, {};",
                                    temps[recipe.ordinal(*from)]
                                ));
                            } else {
                                self.line(format!(
                                    "cvt.rn.f32{} {value}, {};",
                                    dtype_suffix(from_dtype),
                                    temps[recipe.ordinal(*from)]
                                ));
                            }
                            value
                        }
                        DecodeStep::Multiply { into, left, right } => {
                            let _ = into;
                            let value = self.f32();
                            self.line(format!(
                                "mul.rn.f32 {value}, {}, {};",
                                temps[recipe.ordinal(*left)],
                                temps[recipe.ordinal(*right)]
                            ));
                            value
                        }
                        DecodeStep::Negate { into, from } => {
                            let _ = into;
                            let value = self.f32();
                            self.line(format!(
                                "neg.f32 {value}, {};",
                                temps[recipe.ordinal(*from)]
                            ));
                            value
                        }
                        DecodeStep::MultiplyAdd {
                            into,
                            factor,
                            multiplicand,
                            addend,
                        } => {
                            let _ = into;
                            let value = self.f32();
                            self.line(format!(
                                "fma.rn.f32 {value}, {}, {}, {};",
                                temps[recipe.ordinal(*factor)],
                                temps[recipe.ordinal(*multiplicand)],
                                temps[recipe.ordinal(*addend)]
                            ));
                            value
                        }
                        DecodeStep::Cast { into, from, to } => {
                            let _ = into;
                            let value = if to.is_float() {
                                self.f32()
                            } else {
                                self.t32()
                            };
                            self.line(format!(
                                "cvt{}{} {value}, {};",
                                dtype_suffix(*to),
                                dtype_suffix(recipe.dtype(*from)),
                                temps[recipe.ordinal(*from)]
                            ));
                            value
                        }
                    };
                    temps.push(value);
                }
                self.line(format!(
                    "mov{} {}, {};",
                    suffix(out_type),
                    destination,
                    temps[recipe.ordinal(recipe.output())]
                ));
                if let ValueType::Scalar(dtype) = out_type {
                    self.round_named(destination, dtype);
                }
            }
        }
    }

    fn vector_read(
        &mut self,
        out: ClosedValue,
        place: &ClosedReadablePlace,
        indices: &[ClosedIndexValue],
        axis: u32,
        active: ClosedIndexValue,
    ) {
        let (dtype, lanes) = vector_shape(out.ty);
        assert!(
            (axis as usize) < indices.len(),
            "typed CUDA vector read axis exceeds rank"
        );
        for lane in 0..lanes {
            let destination = self.vector_lane(out.value, lane);
            let enabled = self.pred();
            let done = self.label("vector_read_done");
            self.line(format!(
                "setp.gt.u64 {enabled}, {}, {lane};",
                self.v(active.value())
            ));
            self.line(format!(
                "mov{} {destination}, {};",
                scalar_suffix(dtype),
                scalar_zero(dtype)
            ));
            self.line(format!("@!{enabled} bra {done};"));
            let mut coordinates = indices
                .iter()
                .map(|value| self.v(value.value()))
                .collect::<Vec<_>>();
            if lane != 0 {
                let coordinate = self.t64();
                self.line(format!(
                    "add.u64 {coordinate}, {}, {lane};",
                    coordinates[axis as usize]
                ));
                coordinates[axis as usize] = coordinate;
            }
            self.read_to_names(&destination, ValueType::Scalar(dtype), place, &coordinates);
            self.raw(format!("{done}:"));
        }
    }

    fn read_to_names(
        &mut self,
        destination: &str,
        out_type: ValueType,
        place: &ClosedReadablePlace,
        index: &[String],
    ) {
        let (address, geometry, logical) = self.address_names(place, index);
        match &geometry {
            ReadableRepresentationGeometry::Dense(dense) => {
                self.load_to(destination, out_type, &address, dense.dtype)
            }
            ReadableRepresentationGeometry::Packed(packed) => {
                let layout = &packed.layout;
                let recipe = &packed.decode;
                let mut temps: Vec<String> = Vec::with_capacity(recipe.temporary_count());
                for step in recipe.steps() {
                    let value = match step {
                        DecodeStep::ReadPlaneField { into, plane, field } => self.plane_field(
                            &address,
                            &logical,
                            layout.group,
                            &layout.planes[*plane as usize],
                            *field,
                            recipe.dtype(*into),
                        ),
                        DecodeStep::InterpretCode {
                            raw,
                            bits,
                            interpretation,
                            ..
                        } => self.interpret(&temps[recipe.ordinal(*raw)], *bits, interpretation),
                        DecodeStep::DecodeFloatCode { raw, format, .. } => {
                            self.decode_float_code(&temps[recipe.ordinal(*raw)], *format)
                        }
                        DecodeStep::ConvertToF32 { from, .. } => {
                            let value = self.f32();
                            let from_dtype = recipe.dtype(*from);
                            if from_dtype.is_float() {
                                self.line(format!(
                                    "mov.f32 {value}, {};",
                                    temps[recipe.ordinal(*from)]
                                ));
                            } else {
                                self.line(format!(
                                    "cvt.rn.f32{} {value}, {};",
                                    dtype_suffix(from_dtype),
                                    temps[recipe.ordinal(*from)]
                                ));
                            }
                            value
                        }
                        DecodeStep::Multiply { left, right, .. } => {
                            let value = self.f32();
                            self.line(format!(
                                "mul.rn.f32 {value}, {}, {};",
                                temps[recipe.ordinal(*left)],
                                temps[recipe.ordinal(*right)]
                            ));
                            value
                        }
                        DecodeStep::Negate { from, .. } => {
                            let value = self.f32();
                            self.line(format!(
                                "neg.f32 {value}, {};",
                                temps[recipe.ordinal(*from)]
                            ));
                            value
                        }
                        DecodeStep::MultiplyAdd {
                            factor,
                            multiplicand,
                            addend,
                            ..
                        } => {
                            let value = self.f32();
                            self.line(format!(
                                "fma.rn.f32 {value}, {}, {}, {};",
                                temps[recipe.ordinal(*factor)],
                                temps[recipe.ordinal(*multiplicand)],
                                temps[recipe.ordinal(*addend)]
                            ));
                            value
                        }
                        DecodeStep::Cast { from, to, .. } => {
                            let value = if to.is_float() {
                                self.f32()
                            } else {
                                self.t32()
                            };
                            self.line(format!(
                                "cvt{}{} {value}, {};",
                                dtype_suffix(*to),
                                dtype_suffix(recipe.dtype(*from)),
                                temps[recipe.ordinal(*from)]
                            ));
                            value
                        }
                    };
                    temps.push(value);
                }
                self.line(format!(
                    "mov{} {destination}, {};",
                    suffix(out_type),
                    temps[recipe.ordinal(recipe.output())]
                ));
                if let ValueType::Scalar(dtype) = out_type {
                    self.round_named(destination, dtype);
                }
            }
        }
    }
    fn read_plane(
        &mut self,
        out: ErasedValue,
        _out_type: ValueType,
        place: &ClosedPackedPlace,
        plane_info: &PlaneInfo,
        index: &[ErasedValue],
    ) {
        let (address, geometry, logical) = self.address(place, index);
        let value = self.plane_field(
            &address,
            &logical,
            geometry.layout.group,
            plane_info,
            0,
            plane_info.storage_dtype,
        );
        self.line(format!("mov.b32 {}, {value};", self.v(out)));
    }

    fn convert_packet(
        &mut self,
        source: &ClosedExternalGlobalPlace,
        destination: &ClosedPackedGlobalPlace,
        recipe: &seismic_lang::registry::PacketRepackRecipe,
        packet: ErasedValue,
    ) {
        let source_layout = &source.geometry.layout;
        let destination_layout = &destination.geometry.layout;
        let source_base = self.buffer(source.buffer_ordinal as usize);
        let source_packet = self.t64();
        self.line(format!(
            "mad.lo.u64 {source_packet}, {}, {}, {source_base};",
            self.v(packet),
            source_layout.packet_size
        ));
        let destination_base = self.buffer(destination.buffer_ordinal as usize);
        let destination_packet = self.t64();
        self.line(format!(
            "mad.lo.u64 {destination_packet}, {}, {}, {destination_base};",
            self.v(packet),
            destination_layout.packet_size
        ));
        for (plane, plane_recipe) in destination_layout.planes.iter().zip(&recipe.planes) {
            let plane_base = self.t64();
            self.line(format!(
                "add.u64 {plane_base}, {destination_packet}, {};",
                plane.offset
            ));
            match plane_recipe {
                PlaneRepackRecipe::BitRoutes(routes) => {
                    for (destination_byte, byte_routes) in routes.chunks_exact(8).enumerate() {
                        let byte = self.t32();
                        self.line(format!("mov.u32 {byte}, 0;"));
                        for (destination_bit, source_bit) in byte_routes.iter().copied().enumerate()
                        {
                            let bit = self.source_bits(&source_packet, source_bit, 1);
                            let shifted = self.t32();
                            self.line(format!("shl.b32 {shifted}, {bit}, {destination_bit};"));
                            self.line(format!("or.b32 {byte}, {byte}, {shifted};"));
                        }
                        self.line(format!(
                            "st.global.u8 [{plane_base}+{destination_byte}], {byte};"
                        ));
                    }
                }
                PlaneRepackRecipe::DenseValues(values) => {
                    for (index, expression) in values.iter().enumerate() {
                        let value = self.repack_expression(&source_packet, expression);
                        let offset = index
                            .checked_mul(plane.storage_dtype.bytes() as usize)
                            .expect("closed CUDA repack plane offset exceeds usize");
                        self.store_repack_value(&plane_base, offset, plane.storage_dtype, value);
                    }
                }
            }
        }
    }

    fn source_bits(&mut self, source_packet: &str, bit: u32, width: u8) -> String {
        let byte = bit / 8;
        let shift = bit % 8;
        let bytes = (u32::from(width) + shift).div_ceil(8);
        let raw = self.t32();
        self.line(format!("mov.u32 {raw}, 0;"));
        for index in 0..bytes {
            let loaded = self.t32();
            self.line(format!(
                "ld.global.u8 {loaded}, [{source_packet}+{}];",
                byte + index
            ));
            let shifted = self.t32();
            self.line(format!("shl.b32 {shifted}, {loaded}, {};", index * 8));
            self.line(format!("or.b32 {raw}, {raw}, {shifted};"));
        }
        let shifted = self.t32();
        self.line(format!("shr.u32 {shifted}, {raw}, {shift};"));
        let output = self.t32();
        let mask = (1u32 << width) - 1;
        self.line(format!("and.b32 {output}, {shifted}, {mask};"));
        output
    }

    fn repack_expression(&mut self, source_packet: &str, expression: &RepackExpr) -> RepackValue {
        match expression {
            RepackExpr::SourceBits { bit, width } => {
                RepackValue::Integer(self.source_bits(source_packet, *bit, *width))
            }
            RepackExpr::ShiftLeft { value, bits } => {
                let RepackValue::Integer(value) = self.repack_expression(source_packet, value)
                else {
                    panic!("closed CUDA repack shifts a floating value")
                };
                let output = self.t32();
                self.line(format!("shl.b32 {output}, {value}, {bits};"));
                RepackValue::Integer(output)
            }
            RepackExpr::BitOr(left, right) => {
                let RepackValue::Integer(left) = self.repack_expression(source_packet, left) else {
                    panic!("closed CUDA repack OR has a floating left operand")
                };
                let RepackValue::Integer(right) = self.repack_expression(source_packet, right)
                else {
                    panic!("closed CUDA repack OR has a floating right operand")
                };
                let output = self.t32();
                self.line(format!("or.b32 {output}, {left}, {right};"));
                RepackValue::Integer(output)
            }
            RepackExpr::OffsetI32 { value, offset } => {
                let RepackValue::Integer(value) = self.repack_expression(source_packet, value)
                else {
                    panic!("closed CUDA repack offsets a floating value")
                };
                let output = self.t32();
                self.line(format!("add.s32 {output}, {value}, {offset};"));
                RepackValue::Integer(output)
            }
            RepackExpr::F16ToF32(value) => {
                let RepackValue::Integer(value) = self.repack_expression(source_packet, value)
                else {
                    panic!("closed CUDA repack converts a floating value from f16 bits")
                };
                let half = self.h16();
                self.line(format!("cvt.u16.u32 {half}, {value};"));
                let output = self.f32();
                self.line(format!("cvt.f32.f16 {output}, {half};"));
                RepackValue::Float(output)
            }
            RepackExpr::I32ToF32(value) => {
                let RepackValue::Integer(value) = self.repack_expression(source_packet, value)
                else {
                    panic!("closed CUDA repack converts an already-floating i32 value")
                };
                let output = self.f32();
                self.line(format!("cvt.rn.f32.s32 {output}, {value};"));
                RepackValue::Float(output)
            }
            RepackExpr::MultiplyF32(left, right) => {
                let RepackValue::Float(left) = self.repack_expression(source_packet, left) else {
                    panic!("closed CUDA repack multiply has a non-floating left operand")
                };
                let RepackValue::Float(right) = self.repack_expression(source_packet, right) else {
                    panic!("closed CUDA repack multiply has a non-floating right operand")
                };
                let output = self.f32();
                self.line(format!("mul.rn.f32 {output}, {left}, {right};"));
                RepackValue::Float(output)
            }
        }
    }

    fn store_repack_value(
        &mut self,
        plane_base: &str,
        offset: usize,
        dtype: DType,
        value: RepackValue,
    ) {
        match (dtype, value) {
            (DType::F32, RepackValue::Float(value)) => {
                self.line(format!("st.global.f32 [{plane_base}+{offset}], {value};"))
            }
            (DType::F16 | DType::BF16, RepackValue::Float(value)) => {
                let bits = self.h16();
                self.line(format!(
                    "cvt.rn{}.f32 {bits}, {value};",
                    dtype_suffix(dtype)
                ));
                self.line(format!("st.global.u16 [{plane_base}+{offset}], {bits};"));
            }
            (DType::I32 | DType::U32, RepackValue::Integer(value)) => {
                self.line(format!("st.global.u32 [{plane_base}+{offset}], {value};"))
            }
            (DType::Bool, RepackValue::Integer(value)) => {
                self.line(format!("st.global.u8 [{plane_base}+{offset}], {value};"))
            }
            _ => panic!("closed CUDA repack expression type disagrees with destination plane"),
        }
    }

    fn plane_field(
        &mut self,
        packet: &str,
        logical: &str,
        packet_group: u32,
        plane: &PlaneInfo,
        field: u32,
        dtype: DType,
    ) -> String {
        let base = self.t64();
        self.line(format!("add.u64 {base}, {packet}, {};", plane.offset));
        let local = self.t64();
        self.line(format!("rem.u64 {local}, {logical}, {packet_group};"));
        let group = self.t64();
        self.line(format!("div.u64 {group}, {local}, {};", plane.group));
        let entry = self.t64();
        self.line(format!(
            "mad.lo.u64 {entry}, {group}, {}, {field};",
            plane.fields
        ));
        match &plane.encoding {
            PlaneEncoding::Dense(storage) => {
                let address = self.t64();
                self.line(format!(
                    "mad.lo.u64 {address}, {entry}, {}, {base};",
                    storage.bytes()
                ));
                self.load_temp(&address, *storage)
            }
            PlaneEncoding::Packed { bits, .. } => {
                let bit = self.t64();
                self.line(format!("mul.lo.u64 {bit}, {entry}, {bits};"));
                let word = self.t64();
                self.line(format!("shr.u64 {word}, {bit}, 5;"));
                let address = self.t64();
                self.line(format!("mad.lo.u64 {address}, {word}, 4, {base};"));
                let raw = self.t32();
                self.line(format!("ld.global.u32 {raw}, [{address}];"));
                let bit32 = self.t32();
                self.line(format!("cvt.u32.u64 {bit32}, {bit};"));
                let shift = self.t32();
                self.line(format!("and.b32 {shift}, {bit32}, 31;"));
                let fits = self.pred();
                self.line(format!("setp.le.u32 {fits}, {shift}, {};", 32 - *bits));
                let joined = self.t32();
                self.line(format!("shr.u32 {joined}, {raw}, {shift};"));
                let complete = self.label("packed_entry");
                self.line(format!("@{fits} bra {complete};"));
                let high = self.t32();
                self.line(format!("ld.global.u32 {high}, [{address}+4];"));
                self.line(format!("shf.r.wrap.b32 {joined}, {high}, {raw}, {shift};"));
                self.raw(format!("{complete}:"));
                let mask = if *bits == 32 {
                    u32::MAX
                } else {
                    (1u32 << *bits) - 1
                };
                self.line(format!("and.b32 {joined}, {joined}, {mask};"));
                let _ = dtype;
                joined
            }
            PlaneEncoding::FloatCode { format } => {
                let bits = format.bits();
                let bit = self.t64();
                self.line(format!("mul.lo.u64 {bit}, {entry}, {bits};"));
                let byte = self.t64();
                self.line(format!("shr.u64 {byte}, {bit}, 3;"));
                let address = self.t64();
                self.line(format!("add.u64 {address}, {base}, {byte};"));
                let raw = self.t32();
                self.line(format!("ld.global.u8 {raw}, [{address}];"));
                let bit32 = self.t32();
                self.line(format!("cvt.u32.u64 {bit32}, {bit};"));
                let shift = self.t32();
                self.line(format!("and.b32 {shift}, {bit32}, 7;"));
                let shifted = self.t32();
                self.line(format!("shr.u32 {shifted}, {raw}, {shift};"));
                let output = self.t32();
                self.line(format!(
                    "and.b32 {output}, {shifted}, {};",
                    (1u32 << bits) - 1
                ));
                output
            }
        }
    }
    fn interpret(&mut self, raw: &str, bits: u32, interpretation: &CodeInterpretation) -> String {
        let out = self.t32();
        match interpretation {
            CodeInterpretation::Unsigned => self.line(format!("mov.u32 {out}, {raw};")),
            CodeInterpretation::TwosComplement => {
                let shift = 32 - bits;
                self.line(format!("shl.b32 {out}, {raw}, {shift};"));
                self.line(format!("shr.s32 {out}, {out}, {shift};"));
            }
            CodeInterpretation::Offset(offset) => {
                self.line(format!("add.s32 {out}, {raw}, {};", -*offset))
            }
            CodeInterpretation::Table(table) => {
                self.line(format!("mov.s32 {out}, {};", table[0]));
                for (index, value) in table.iter().copied().enumerate().skip(1) {
                    let p = self.pred();
                    self.line(format!("setp.eq.u32 {p}, {raw}, {index};"));
                    self.line(format!("selp.s32 {out}, {value}, {out}, {p};"));
                }
            }
        }
        out
    }
    fn decode_float_code(&mut self, raw: &str, format: FloatCodeFormat) -> String {
        let sign_shift = match format {
            FloatCodeFormat::E2M1 => 3,
            FloatCodeFormat::E4M3 | FloatCodeFormat::UE4M3 => 7,
        };
        let exponent_bits = match format {
            FloatCodeFormat::E2M1 => 2,
            FloatCodeFormat::E4M3 | FloatCodeFormat::UE4M3 => 4,
        };
        let mantissa_bits = match format {
            FloatCodeFormat::E2M1 => 1,
            FloatCodeFormat::E4M3 | FloatCodeFormat::UE4M3 => 3,
        };
        let sign = self.t32();
        if format == FloatCodeFormat::UE4M3 {
            self.line(format!("mov.u32 {sign}, 0;"));
        } else {
            self.line(format!("shr.u32 {sign}, {raw}, {sign_shift};"));
            self.line(format!("shl.b32 {sign}, {sign}, 31;"));
        }
        let exponent = self.t32();
        self.line(format!("shr.u32 {exponent}, {raw}, {mantissa_bits};"));
        self.line(format!(
            "and.b32 {exponent}, {exponent}, {};",
            (1u32 << exponent_bits) - 1
        ));
        let mantissa = self.t32();
        self.line(format!(
            "and.b32 {mantissa}, {raw}, {};",
            (1u32 << mantissa_bits) - 1
        ));

        let normal_exponent = self.t32();
        let exponent_bias = match format {
            FloatCodeFormat::E2M1 => 126,
            FloatCodeFormat::E4M3 | FloatCodeFormat::UE4M3 => 120,
        };
        self.line(format!(
            "add.u32 {normal_exponent}, {exponent}, {exponent_bias};"
        ));
        self.line(format!("shl.b32 {normal_exponent}, {normal_exponent}, 23;"));
        let normal_bits = self.t32();
        self.line(format!(
            "shl.b32 {normal_bits}, {mantissa}, {};",
            23 - mantissa_bits
        ));
        self.line(format!(
            "or.b32 {normal_bits}, {normal_bits}, {normal_exponent};"
        ));

        let subnormal = self.f32();
        self.line(format!("cvt.rn.f32.u32 {subnormal}, {mantissa};"));
        let subnormal_scale = match format {
            FloatCodeFormat::E2M1 => "0f3f000000", // 2^-1
            FloatCodeFormat::E4M3 | FloatCodeFormat::UE4M3 => "0f3b000000", // 2^-9
        };
        self.line(format!(
            "mul.rn.f32 {subnormal}, {subnormal}, {subnormal_scale};"
        ));
        let subnormal_bits = self.t32();
        self.line(format!("mov.b32 {subnormal_bits}, {subnormal};"));

        let exponent_zero = self.pred();
        self.line(format!("setp.eq.u32 {exponent_zero}, {exponent}, 0;"));
        let magnitude = self.t32();
        self.line(format!(
            "selp.b32 {magnitude}, {subnormal_bits}, {normal_bits}, {exponent_zero};"
        ));

        if matches!(format, FloatCodeFormat::E4M3 | FloatCodeFormat::UE4M3) {
            let exponent_all_ones = self.pred();
            let mantissa_all_ones = self.pred();
            let nan = self.pred();
            self.line(format!("setp.eq.u32 {exponent_all_ones}, {exponent}, 15;"));
            self.line(format!("setp.eq.u32 {mantissa_all_ones}, {mantissa}, 7;"));
            self.line(format!(
                "and.pred {nan}, {exponent_all_ones}, {mantissa_all_ones};"
            ));
            let finite_or_nan = self.t32();
            self.line(format!(
                "selp.b32 {finite_or_nan}, 0x7fc00000, {magnitude}, {nan};"
            ));
            self.line(format!("mov.b32 {magnitude}, {finite_or_nan};"));
        }

        let signed = self.t32();
        self.line(format!("xor.b32 {signed}, {magnitude}, {sign};"));
        let out = self.f32();
        self.line(format!("mov.b32 {out}, {signed};"));
        out
    }
    fn load_temp(&mut self, address: &str, dtype: DType) -> String {
        match dtype {
            DType::F32 => {
                let v = self.f32();
                self.line(format!("ld.global.f32 {v}, [{address}];"));
                v
            }
            DType::F16 => {
                let b = self.h16();
                let v = self.f32();
                self.line(format!("ld.global.u16 {b}, [{address}];"));
                self.line(format!("cvt.f32.f16 {v}, {b};"));
                v
            }
            DType::BF16 => {
                let b = self.h16();
                let v = self.f32();
                self.line(format!("ld.global.u16 {b}, [{address}];"));
                self.line(format!("cvt.f32.bf16 {v}, {b};"));
                v
            }
            DType::I32 | DType::U32 => {
                let v = self.t32();
                self.line(format!(
                    "ld.global{} {v}, [{address}];",
                    dtype_suffix(dtype)
                ));
                v
            }
            DType::Bool => {
                let v = self.t32();
                self.line(format!("ld.global.u8 {v}, [{address}];"));
                v
            }
        }
    }
    fn load(&mut self, out: ErasedValue, out_type: ValueType, address: &str, dtype: DType) {
        let destination = self.v(out);
        self.load_to(&destination, out_type, address, dtype);
    }
    fn load_to(&mut self, destination: &str, out_type: ValueType, address: &str, dtype: DType) {
        let temp = self.load_temp(address, dtype);
        self.line(format!("mov{} {destination}, {temp};", suffix(out_type)));
    }
    fn load_bits(&mut self, out: ErasedValue, address: &str, dtype: DType) {
        self.line(format!(
            "ld.global{} {}, [{address}];",
            match dtype {
                DType::F32 | DType::I32 | DType::U32 => ".u32",
                DType::F16 | DType::BF16 => ".u16",
                DType::Bool => ".u8",
            },
            self.v(out)
        ));
    }
    fn write(&mut self, place: &ClosedDensePlace, index: &[ErasedValue], value: ErasedValue) {
        let (address, geometry, _) = self.address(place, index);
        let value = self.v(value);
        self.write_address(&address, &geometry, &value);
    }
    fn write_address(
        &mut self,
        address: &str,
        geometry: &DenseRepresentationGeometry,
        value: &str,
    ) {
        match geometry.dtype {
            DType::F32 => self.line(format!("st.global.f32 [{address}], {value};")),
            DType::I32 | DType::U32 => self.line(format!("st.global.u32 [{address}], {value};")),
            DType::Bool => self.line(format!("st.global.u8 [{address}], {value};")),
            DType::F16 | DType::BF16 => {
                let bits = self.h16();
                self.line(format!(
                    "cvt.rn{}{}.f32 {bits}, {value};",
                    dtype_suffix(geometry.dtype),
                    "",
                ));
                self.line(format!("st.global.u16 [{address}], {bits};"));
            }
        }
    }

    fn vector_write(
        &mut self,
        place: &ClosedDensePlace,
        indices: &[ClosedIndexValue],
        axis: u32,
        active: ClosedIndexValue,
        value: ClosedValue,
    ) {
        let (_, lanes) = vector_shape(value.ty);
        assert!(
            (axis as usize) < indices.len(),
            "typed CUDA vector write axis exceeds rank"
        );
        for lane in 0..lanes {
            let enabled = self.pred();
            let done = self.label("vector_write_done");
            self.line(format!(
                "setp.gt.u64 {enabled}, {}, {lane};",
                self.v(active.value())
            ));
            self.line(format!("@!{enabled} bra {done};"));
            let mut coordinates = indices
                .iter()
                .map(|coordinate| self.v(coordinate.value()))
                .collect::<Vec<_>>();
            if lane != 0 {
                let coordinate = self.t64();
                self.line(format!(
                    "add.u64 {coordinate}, {}, {lane};",
                    coordinates[axis as usize]
                ));
                coordinates[axis as usize] = coordinate;
            }
            let (address, geometry, _) = self.address_names(place, &coordinates);
            let lane_value = self.vector_lane(value.value, lane);
            self.write_address(&address, &geometry, &lane_value);
            self.raw(format!("{done}:"));
        }
    }
    fn atomic(
        &mut self,
        op: AtomicOp,
        place: &ClosedDensePlace,
        index: &[ErasedValue],
        value: ErasedValue,
    ) {
        let (address, geometry, _) = self.address(place, index);
        let dtype = geometry.dtype;
        if dtype == DType::F32 && matches!(op, AtomicOp::Max | AtomicOp::Min) {
            let head = self.label("atomic_float");
            let done = self.label("atomic_float_done");
            let observed = self.t32();
            let expected = self.t32();
            let current = self.f32();
            let replacement = self.f32();
            let replacement_bits = self.t32();
            let prior = self.t32();
            self.line(format!("ld.global.u32 {observed}, [{address}];"));
            self.raw(format!("{head}:"));
            self.line(format!("mov.b32 {expected}, {observed};"));
            self.line(format!("mov.b32 {current}, {expected};"));
            self.line(format!(
                "{}.f32 {replacement}, {current}, {};",
                if op == AtomicOp::Max { "max" } else { "min" },
                self.v(value)
            ));
            self.line(format!("mov.b32 {replacement_bits}, {replacement};"));
            self.line(format!(
                "atom.global.cas.b32 {prior}, [{address}], {expected}, {replacement_bits};"
            ));
            let complete = self.pred();
            self.line(format!("setp.eq.u32 {complete}, {prior}, {expected};"));
            self.line(format!("@{complete} bra {done};"));
            self.line(format!("mov.b32 {observed}, {prior};"));
            self.line(format!("bra {head};"));
            self.raw(format!("{done}:"));
            return;
        }
        let mnemonic = match (op, dtype) {
            (AtomicOp::Add, DType::F32) => "add.f32",
            (AtomicOp::Add, DType::I32) => "add.s32",
            (AtomicOp::Add, DType::U32) => "add.u32",
            (AtomicOp::Max, DType::I32) => "max.s32",
            (AtomicOp::Max, DType::U32) => "max.u32",
            (AtomicOp::Min, DType::I32) => "min.s32",
            (AtomicOp::Min, DType::U32) => "min.u32",
            (_, other) => {
                panic!("target profile admitted an unsupported CUDA atomic dtype {other:?}")
            }
        };
        let discard = if dtype == DType::F32 {
            self.f32()
        } else {
            self.t32()
        };
        self.line(format!(
            "atom.global.{mnemonic} {discard}, [{address}], {};",
            self.v(value)
        ));
    }
    fn store_slot(&mut self, slot: u32, dtype: DType, value: ClosedValue) {
        let p = self.pred();
        self.line(format!("setp.eq.u64 {p}, %linear_thread, 0;"));
        match dtype {
            DType::F32 => self.line(format!(
                "@{p} st.global.f32 [%results+{}], {};",
                slot * 8,
                self.v(value.value)
            )),
            DType::F16 | DType::BF16 => {
                let bits = self.h16();
                self.line(format!(
                    "cvt.rn{}{}.f32 {bits}, {};",
                    dtype_suffix(dtype),
                    "",
                    self.v(value.value)
                ));
                self.line(format!(
                    "@{p} st.global.u16 [%results+{}], {bits};",
                    slot * 8
                ));
            }
            DType::I32 | DType::U32 | DType::Bool => self.line(format!(
                "@{p} st.global.u32 [%results+{}], {};",
                slot * 8,
                self.v(value.value)
            )),
        }
    }

    fn intrinsic(&mut self, op: &CudaIntrinsic, outs: &[ClosedValue], args: &[ClosedValue]) {
        match op {
            CudaIntrinsic::LaneIndex => {
                let [out] = outs else {
                    panic!("lane-index result arity")
                };
                self.line(format!("mov.u32 {}, %laneid;", self.v(out.value)));
            }
            CudaIntrinsic::Shuffle { dtype } => {
                let [out] = outs else {
                    panic!("shuffle result arity")
                };
                let [value, lane] = args else {
                    panic!("shuffle argument arity")
                };
                let raw = self.t32();
                if dtype.is_float() {
                    self.line(format!("mov.b32 {raw}, {};", self.v(value.value)))
                } else {
                    self.line(format!("mov.b32 {raw}, {};", self.v(value.value)))
                }
                let shuffled = self.t32();
                self.line(format!(
                    "shfl.sync.idx.b32 {shuffled}|%p0, {raw}, {}, 31, 0xffffffff;",
                    self.v(lane.value)
                ));
                self.line(format!("mov.b32 {}, {shuffled};", self.v(out.value)));
            }
            CudaIntrinsic::SubgroupReduce { op, dtype } => {
                let [out] = outs else {
                    panic!("subgroup reduce result arity")
                };
                let [value] = args else {
                    panic!("subgroup reduce argument arity")
                };
                self.line(format!(
                    "mov{} {}, {};",
                    suffix(out.ty),
                    self.v(out.value),
                    self.v(value.value)
                ));
                for delta in [16, 8, 4, 2, 1] {
                    let shuffled = self.t32();
                    self.line(format!(
                        "shfl.sync.bfly.b32 {shuffled}|%p0, {}, {delta}, 31, 0xffffffff;",
                        self.v(out.value)
                    ));
                    let ty = out.ty;
                    let instruction = match (op, ty) {
                        (ReduceOp::Sum, ValueType::Scalar(d)) if d.is_float() => "add.rn.f32",
                        (ReduceOp::Max, ValueType::Scalar(d)) if d.is_float() => "max.f32",
                        (ReduceOp::Min, ValueType::Scalar(d)) if d.is_float() => "min.f32",
                        (ReduceOp::Argmax, _) => {
                            unreachable!("CUDA subgroup registry has no scalar argmax signature")
                        }
                        (ReduceOp::Sum, _) => "add.u32",
                        (ReduceOp::Max, _) => "max.u32",
                        (ReduceOp::Min, _) => "min.u32",
                    };
                    self.line(format!(
                        "{instruction} {}, {}, {shuffled};",
                        self.v(out.value),
                        self.v(out.value)
                    ));
                    if *op == ReduceOp::Sum && matches!(*dtype, DType::F16 | DType::BF16) {
                        // Registry numerics declare the reduction's
                        // accumulator dtype. Narrow accumulators therefore
                        // round at every reassociated butterfly edge, not
                        // only after the final f32-register operation.
                        self.round(out.value, ValueType::Scalar(*dtype));
                    }
                }
            }
            CudaIntrinsic::MatrixMatmul {
                elem,
                a,
                b,
                destination,
            } => self.matrix(*elem, a, b, None, destination),
            CudaIntrinsic::MatrixMatmulAdd {
                elem,
                a,
                b,
                c,
                destination,
            } => self.matrix(*elem, a, b, Some(c), destination),
            CudaIntrinsic::NvFp4Matmul {
                a,
                b,
                destination,
                tensor_memory,
            } => self.matrix_nvfp4(a, b, None, destination, args, *tensor_memory),
            CudaIntrinsic::NvFp4MatmulAdd {
                a,
                b,
                c,
                destination,
                tensor_memory,
            } => self.matrix_nvfp4(a, b, Some(c), destination, args, *tensor_memory),
        }
    }
    fn matrix(
        &mut self,
        elem: DType,
        a: &LogicalTensorMap,
        b: &LogicalTensorMap,
        c: Option<&LogicalTensorMap>,
        destination: &LogicalTensorMap,
    ) {
        if elem == DType::F32 {
            self.matrix_packed(a, b, c, destination);
            return;
        }
        let a_map = a;
        let b_map = b;
        let destination_map = destination;
        let c_map = c;
        let a = self.dense_place(a.base);
        let b = self.dense_place(b.base);
        let destination = self.dense_place(destination.base);
        let c = c.map(|map| self.dense_place(map.base));
        let ar = a_map.extents.len();
        let br = b_map.extents.len();
        let dr = destination_map.extents.len();
        if ar != 2 || br != 2 || dr != 2 {
            panic!("matrix intrinsic places are not rank two")
        };
        if !matches!(elem, DType::F16 | DType::BF16) {
            panic!("CUDA matrix intrinsic entered PTX emission with an unimplemented element type")
        }
        for place in [&a, &b] {
            if place.geometry.dtype != elem {
                panic!("CUDA matrix intrinsic operand type disagrees with its registered signature")
            }
        }
        let rows = self.v(a_map.extents[0]);
        let inner = self.v(a_map.extents[1]);
        let columns = self.v(b_map.extents[1]);
        let lane = self.t64();
        let lane32 = self.t32();
        self.line(format!("mov.u32 {lane32}, %laneid;"));
        self.line(format!("cvt.u64.u32 {lane}, {lane32};"));
        let group = self.t64();
        let thread = self.t64();
        self.line(format!("shr.u64 {group}, {lane}, 2;"));
        self.line(format!("and.b64 {thread}, {lane}, 3;"));

        let row_tiles = self.t64();
        let column_tiles = self.t64();
        self.line(format!("add.u64 {row_tiles}, {rows}, 15;"));
        self.line(format!("div.u64 {row_tiles}, {row_tiles}, 16;"));
        self.line(format!("add.u64 {column_tiles}, {columns}, 7;"));
        self.line(format!("div.u64 {column_tiles}, {column_tiles}, 8;"));
        let tile_count = self.t64();
        self.line(format!(
            "mul.lo.u64 {tile_count}, {row_tiles}, {column_tiles};"
        ));

        // Every physical warp owns a disjoint strided subset of output
        // tiles. Subgroup legality proves the global participant count is a
        // multiple of 32, so every mma.sync is executed by a complete warp.
        let warp = self.t64();
        self.line(format!("shr.u64 {warp}, %linear_thread, 5;"));
        let grid_threads = self.t64();
        let workgroup_threads = self.t64();
        self.line(format!("mov.u64 {grid_threads}, 1;"));
        self.line(format!("mov.u64 {workgroup_threads}, 1;"));
        for axis in 0..3 {
            let grid_axis = self.word(self.layout.words.grid_first + axis);
            let workgroup_axis = self.word(self.layout.words.workgroup_first + axis);
            self.line(format!(
                "mul.lo.u64 {grid_threads}, {grid_threads}, {grid_axis};"
            ));
            self.line(format!(
                "mul.lo.u64 {workgroup_threads}, {workgroup_threads}, {workgroup_axis};"
            ));
        }
        let warp_count = self.t64();
        self.line(format!(
            "mul.lo.u64 {warp_count}, {grid_threads}, {workgroup_threads};"
        ));
        self.line(format!("shr.u64 {warp_count}, {warp_count}, 5;"));

        let tile_loop = self.label("mma_tile");
        let tile_done = self.label("mma_tile_done");
        self.raw(format!("{tile_loop}:"));
        let tile_end = self.pred();
        self.line(format!("setp.ge.u64 {tile_end}, {warp}, {tile_count};"));
        self.line(format!("@{tile_end} bra {tile_done};"));
        let tile_row = self.t64();
        let tile_column = self.t64();
        self.line(format!("div.u64 {tile_row}, {warp}, {column_tiles};"));
        self.line(format!("mul.lo.u64 {tile_row}, {tile_row}, 16;"));
        self.line(format!("rem.u64 {tile_column}, {warp}, {column_tiles};"));
        self.line(format!("mul.lo.u64 {tile_column}, {tile_column}, 8;"));

        let mut accumulators = Vec::with_capacity(4);
        for i in 0..4u64 {
            let row = self.t64();
            let column = self.t64();
            self.line(format!("add.u64 {row}, {tile_row}, {group};"));
            if i >= 2 {
                self.line(format!("add.u64 {row}, {row}, 8;"));
            }
            self.line(format!("mad.lo.u64 {column}, {thread}, 2, {tile_column};"));
            if i & 1 != 0 {
                self.line(format!("add.u64 {column}, {column}, 1;"));
            }
            let value = self.f32();
            if let Some(c) = &c {
                let loaded = self.matrix_load_f32(
                    c_map.expect("matrix addend map"),
                    c,
                    &row,
                    &column,
                    &rows,
                    &columns,
                );
                self.line(format!("mov.f32 {value}, {loaded};"));
            } else {
                self.line(format!("mov.f32 {value}, 0f00000000;"));
            }
            accumulators.push((value, row, column));
        }

        let k = self.t64();
        self.line(format!("mov.u64 {k}, 0;"));
        let k_loop = self.label("mma_k");
        let k_done = self.label("mma_k_done");
        self.raw(format!("{k_loop}:"));
        let k_end = self.pred();
        self.line(format!("setp.ge.u64 {k_end}, {k}, {inner};"));
        self.line(format!("@{k_end} bra {k_done};"));

        let mut a_registers = Vec::with_capacity(4);
        for pair in 0..4u64 {
            let mut halves = Vec::with_capacity(2);
            for within in 0..2u64 {
                let i = pair * 2 + within;
                let row = self.t64();
                let column = self.t64();
                self.line(format!("add.u64 {row}, {tile_row}, {group};"));
                if !(i < 2 || (4..6).contains(&i)) {
                    self.line(format!("add.u64 {row}, {row}, 8;"));
                }
                self.line(format!("mad.lo.u64 {column}, {thread}, 2, {k};"));
                self.line(format!("add.u64 {column}, {column}, {};", i & 1));
                if i >= 4 {
                    self.line(format!("add.u64 {column}, {column}, 8;"));
                }
                halves.push(self.matrix_load_u16(a_map, &a, &row, &column, &rows, &inner));
            }
            a_registers.push(self.pack_u16(&halves[0], &halves[1]));
        }
        let mut b_registers = Vec::with_capacity(2);
        for pair in 0..2u64 {
            let mut halves = Vec::with_capacity(2);
            for within in 0..2u64 {
                let i = pair * 2 + within;
                let row = self.t64();
                let column = self.t64();
                self.line(format!("mad.lo.u64 {row}, {thread}, 2, {k};"));
                self.line(format!("add.u64 {row}, {row}, {};", i & 1));
                if i >= 2 {
                    self.line(format!("add.u64 {row}, {row}, 8;"));
                }
                self.line(format!("add.u64 {column}, {tile_column}, {group};"));
                halves.push(self.matrix_load_u16(b_map, &b, &row, &column, &inner, &columns));
            }
            b_registers.push(self.pack_u16(&halves[0], &halves[1]));
        }
        let next = (0..4).map(|_| self.f32()).collect::<Vec<_>>();
        let kind = if elem == DType::F16 { "f16" } else { "bf16" };
        self.line(format!(
            "mma.sync.aligned.m16n8k16.row.col.f32.{kind}.{kind}.f32 {{{}, {}, {}, {}}}, {{{}, {}, {}, {}}}, {{{}, {}}}, {{{}, {}, {}, {}}};",
            next[0], next[1], next[2], next[3],
            a_registers[0], a_registers[1], a_registers[2], a_registers[3],
            b_registers[0], b_registers[1],
            accumulators[0].0, accumulators[1].0, accumulators[2].0, accumulators[3].0,
        ));
        for (accumulator, value) in accumulators.iter_mut().zip(next) {
            accumulator.0 = value;
        }
        self.line(format!("add.u64 {k}, {k}, 16;"));
        self.line(format!("bra {k_loop};"));
        self.raw(format!("{k_done}:"));
        for (value, row, column) in accumulators {
            self.matrix_store_f32(
                destination_map,
                &destination,
                &row,
                &column,
                &rows,
                &columns,
                &value,
            );
        }
        self.line(format!("add.u64 {warp}, {warp}, {warp_count};"));
        self.line(format!("bra {tile_loop};"));
        self.raw(format!("{tile_done}:"));
    }

    fn matrix_packed(
        &mut self,
        a: &LogicalTensorMap,
        b: &LogicalTensorMap,
        c: Option<&LogicalTensorMap>,
        destination: &LogicalTensorMap,
    ) {
        let a_map = a;
        let b_map = b;
        let destination_map = destination;
        let c_map = c;
        let a = self.dense_place(a.base);
        let b = self.readable_place(b.base);
        let destination = self.dense_place(destination.base);
        let c = c.map(|map| self.dense_place(map.base));
        let ReadableRepresentationGeometry::Packed(_) = &b.geometry else {
            panic!("registry-selected CUDA packed matrix row has a dense right operand")
        };
        if a.geometry.dtype != DType::F32 || destination.geometry.dtype != DType::F32 {
            panic!("registry-selected CUDA packed matrix row has a non-f32 dense operand")
        }
        let ar = a_map.extents.len();
        let br = b_map.extents.len();
        let dr = destination_map.extents.len();
        if ar != 2 || br != 2 || dr != 2 {
            panic!("registry-selected CUDA packed matrix row is not rank two")
        }
        let rows = self.v(a_map.extents[0]);
        let inner = self.v(a_map.extents[1]);
        let columns = self.v(b_map.extents[1]);
        let total = self.t64();
        self.line(format!("mul.lo.u64 {total}, {rows}, {columns};"));
        let participant_count = self.t64();
        self.line(format!("mov.u64 {participant_count}, 1;"));
        for axis in 0..3 {
            let grid_axis = self.word(self.layout.words.grid_first + axis);
            let workgroup_axis = self.word(self.layout.words.workgroup_first + axis);
            self.line(format!(
                "mul.lo.u64 {participant_count}, {participant_count}, {grid_axis};"
            ));
            self.line(format!(
                "mul.lo.u64 {participant_count}, {participant_count}, {workgroup_axis};"
            ));
        }
        let output = self.t64();
        self.line(format!("mov.u64 {output}, %linear_thread;"));
        let output_loop = self.label("packed_matrix_output");
        let output_done = self.label("packed_matrix_done");
        self.raw(format!("{output_loop}:"));
        let finished = self.pred();
        self.line(format!("setp.ge.u64 {finished}, {output}, {total};"));
        self.line(format!("@{finished} bra {output_done};"));
        let row = self.t64();
        let column = self.t64();
        self.line(format!("div.u64 {row}, {output}, {columns};"));
        self.line(format!("rem.u64 {column}, {output}, {columns};"));
        let accumulator = self.f32();
        if let Some(c) = &c {
            let initial = self.matrix_load_f32(
                c_map.expect("packed matrix addend map"),
                c,
                &row,
                &column,
                &rows,
                &columns,
            );
            self.line(format!("mov.f32 {accumulator}, {initial};"));
        } else {
            self.line(format!("mov.f32 {accumulator}, 0f00000000;"));
        }
        let k = self.t64();
        self.line(format!("mov.u64 {k}, 0;"));
        let k_loop = self.label("packed_matrix_k");
        let k_done = self.label("packed_matrix_k_done");
        self.raw(format!("{k_loop}:"));
        let k_finished = self.pred();
        self.line(format!("setp.ge.u64 {k_finished}, {k}, {inner};"));
        self.line(format!("@{k_finished} bra {k_done};"));
        let (_, a_address) = self.matrix_address(a_map, &a, &row, &k);
        let left = self.f32();
        self.line(format!("ld.global.f32 {left}, [{a_address}];"));
        let right = self.f32();
        self.read_to_names(
            &right,
            ValueType::Scalar(DType::F32),
            &b,
            &self.logical_coords(b_map, &[k.clone(), column.clone()]),
        );
        self.line(format!(
            "fma.rn.f32 {accumulator}, {left}, {right}, {accumulator};"
        ));
        self.line(format!("add.u64 {k}, {k}, 1;"));
        self.line(format!("bra {k_loop};"));
        self.raw(format!("{k_done}:"));
        self.matrix_store_f32(
            destination_map,
            &destination,
            &row,
            &column,
            &rows,
            &columns,
            &accumulator,
        );
        self.line(format!("add.u64 {output}, {output}, {participant_count};"));
        self.line(format!("bra {output_loop};"));
        self.raw(format!("{output_done}:"));
    }

    fn matrix_nvfp4(
        &mut self,
        a_map: &LogicalTensorMap,
        b_map: &LogicalTensorMap,
        c_map: Option<&LogicalTensorMap>,
        destination_map: &LogicalTensorMap,
        arguments: &[ClosedValue],
        tensor_memory: AddressableResourceHandle,
    ) {
        let resource = self
            .kernel
            .closed_addressable_resource(tensor_memory, self.layout);
        assert_eq!(
            arguments.len(),
            2,
            "NVFP4 matrix intrinsic has two global scales"
        );
        assert_eq!(a_map.extents.len(), 2, "NVFP4 left operand is rank two");
        assert_eq!(b_map.extents.len(), 2, "NVFP4 right operand is rank two");
        assert_eq!(
            destination_map.extents.len(),
            2,
            "NVFP4 destination is rank two"
        );
        let a = self.readable_place(a_map.base);
        let b = self.readable_place(b_map.base);
        let destination = self.dense_place(destination_map.base);
        let c = c_map.map(|map| self.dense_place(map.base));
        if !matches!(a.geometry, ReadableRepresentationGeometry::Packed(_))
            || !matches!(b.geometry, ReadableRepresentationGeometry::Packed(_))
            || destination.geometry.dtype != DType::F32
        {
            panic!("NVFP4 matrix intrinsic has non-NVFP4 input or non-f32 output geometry")
        }
        let rows = self.v(a_map.extents[0]);
        let inner = self.v(a_map.extents[1]);
        let columns = self.v(b_map.extents[1]);
        let left_scale = self.v(arguments[0].value);
        let right_scale = self.v(arguments[1].value);
        self.nvfp4_validate_geometry(&a);
        self.nvfp4_validate_geometry(&b);

        let thread = self.t64();
        self.line(format!("cvt.u64.u32 {thread}, %t13;"));
        let first_warp = self.pred();
        self.line(format!("setp.lt.u64 {first_warp}, {thread}, 32;"));
        let after_alloc = self.label("nvfp4_after_alloc");
        self.line(format!("@!{first_warp} bra {after_alloc};"));
        let units64 = self.word(resource.units_word);
        let units = self.t32();
        self.line(format!("cvt.u32.u64 {units}, {units64};"));
        self.line(format!(
            "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [seismic_nvfp4_tmem_addr], {units};"
        ));
        self.raw(format!("{after_alloc}:"));
        self.line("bar.sync 0;");
        let tmem = self.t32();
        self.line(format!("ld.shared.b32 {tmem}, [seismic_nvfp4_tmem_addr];"));

        let init_done = self.label("nvfp4_mbarrier_initialized");
        let thread_zero = self.pred();
        self.line(format!("setp.eq.u64 {thread_zero}, {thread}, 0;"));
        self.line(format!("@!{thread_zero} bra {init_done};"));
        self.line("mbarrier.init.shared::cta.b64 [seismic_nvfp4_mbarrier], 1;");
        self.line("fence.mbarrier_init.release.cluster;");
        self.raw(format!("{init_done}:"));
        self.line("bar.sync 0;");

        let a_shared = self.shared_address("seismic_nvfp4_a");
        let b_shared = self.shared_address("seismic_nvfp4_b");
        let a_desc = self.nvfp4_shared_descriptor(&a_shared, 128, 8);
        let b_desc = self.nvfp4_shared_descriptor(&b_shared, 8, 8);
        let tile_rows = self.t64();
        let tile_columns = self.t64();
        self.line(format!("add.u64 {tile_rows}, {rows}, 127;"));
        self.line(format!("div.u64 {tile_rows}, {tile_rows}, 128;"));
        self.line(format!("add.u64 {tile_columns}, {columns}, 7;"));
        self.line(format!("div.u64 {tile_columns}, {tile_columns}, 8;"));
        let tile_count = self.t64();
        self.line(format!(
            "mul.lo.u64 {tile_count}, {tile_rows}, {tile_columns};"
        ));
        let block = self.t64();
        self.line(format!("cvt.u64.u32 {block}, %t6;"));
        let block_count = self.t64();
        self.line(format!("mov.u64 {block_count}, %nctaid.x;"));
        let grid_y = self.t64();
        let grid_z = self.t64();
        self.line(format!("mov.u64 {grid_y}, %nctaid.y;"));
        self.line(format!("mov.u64 {grid_z}, %nctaid.z;"));
        self.line(format!(
            "mul.lo.u64 {block_count}, {block_count}, {grid_y};"
        ));
        self.line(format!(
            "mul.lo.u64 {block_count}, {block_count}, {grid_z};"
        ));
        let tile = self.t64();
        self.line(format!("mov.u64 {tile}, {block};"));
        let barrier_phase = self.t32();
        self.line(format!("mov.u32 {barrier_phase}, 0;"));
        let tile_loop = self.label("nvfp4_tile");
        let tile_done = self.label("nvfp4_tile_done");
        self.raw(format!("{tile_loop}:"));
        let all_tiles = self.pred();
        self.line(format!("setp.ge.u64 {all_tiles}, {tile}, {tile_count};"));
        self.line(format!("@{all_tiles} bra {tile_done};"));
        let tile_m = self.t64();
        let tile_n = self.t64();
        self.line(format!("div.u64 {tile_m}, {tile}, {tile_columns};"));
        self.line(format!("mul.lo.u64 {tile_m}, {tile_m}, 128;"));
        self.line(format!("rem.u64 {tile_n}, {tile}, {tile_columns};"));
        self.line(format!("mul.lo.u64 {tile_n}, {tile_n}, 8;"));

        let tile_k = self.t64();
        self.line(format!("mov.u64 {tile_k}, 0;"));
        let k_loop = self.label("nvfp4_tile_k");
        let k_done = self.label("nvfp4_tile_k_done");
        self.raw(format!("{k_loop}:"));
        let k_finished = self.pred();
        self.line(format!("setp.ge.u64 {k_finished}, {tile_k}, {inner};"));
        self.line(format!("@{k_finished} bra {k_done};"));
        self.nvfp4_stage_a(
            a_map, &a, &tile_m, &tile_k, &rows, &inner, &thread, &a_shared,
        );
        self.nvfp4_stage_b(
            b_map, &b, &tile_n, &tile_k, &columns, &inner, &thread, &b_shared,
        );
        self.line("fence.proxy.async.shared::cta;");
        self.line("bar.sync 0;");
        self.nvfp4_store_scales(
            a_map,
            b_map,
            &a,
            &b,
            &tile_m,
            &tile_n,
            &tile_k,
            &rows,
            &columns,
            &inner,
            &thread,
            &tmem,
            &first_warp,
        );
        self.line("bar.sync 0;");
        let issue_done = self.label("nvfp4_mma_issued");
        self.line(format!("@!{thread_zero} bra {issue_done};"));
        let sfa = self.t32();
        let sfb = self.t32();
        self.line(format!("add.u32 {sfa}, {tmem}, 8;"));
        self.line(format!("add.u32 {sfb}, {tmem}, 12;"));
        let accumulate = self.pred();
        self.line(format!("setp.ne.u64 {accumulate}, {tile_k}, 0;"));
        let instruction_descriptor = self.t32();
        self.line(format!("mov.u32 {instruction_descriptor}, 0x08020480;"));
        self.line(format!(
            "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.block16 [{tmem}], {a_desc}, {b_desc}, {instruction_descriptor}, [{sfa}], [{sfb}], {accumulate};"
        ));
        self.line(
            "tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [seismic_nvfp4_mbarrier];",
        );
        let wait = self.label("nvfp4_mma_wait");
        self.raw(format!("{wait}:"));
        let complete = self.pred();
        self.line(format!(
            "mbarrier.try_wait.parity.b64 {complete}, [seismic_nvfp4_mbarrier], {barrier_phase};"
        ));
        self.line(format!("@!{complete} bra {wait};"));
        self.line("tcgen05.fence::after_thread_sync;");
        self.line(format!("xor.b32 {barrier_phase}, {barrier_phase}, 1;"));
        self.raw(format!("{issue_done}:"));
        self.line("bar.sync 0;");
        self.line(format!("add.u64 {tile_k}, {tile_k}, 64;"));
        self.line(format!("bra {k_loop};"));
        self.raw(format!("{k_done}:"));

        let warp = self.t32();
        let lane_base = self.t32();
        let load_address = self.t32();
        self.line(format!("shr.u32 {warp}, %t13, 5;"));
        self.line(format!("shl.b32 {lane_base}, {warp}, 21;"));
        self.line(format!("or.b32 {load_address}, {tmem}, {lane_base};"));
        let loaded = (0..8).map(|_| self.t32()).collect::<Vec<_>>();
        self.line(format!(
            "tcgen05.ld.sync.aligned.32x32b.x8.b32 {{{}}}, [{load_address}];",
            loaded.join(", ")
        ));
        self.line("tcgen05.wait::ld.sync.aligned;");
        for (column_offset, bits) in loaded.into_iter().enumerate() {
            let row = self.t64();
            let column = self.t64();
            self.line(format!("add.u64 {row}, {tile_m}, {thread};"));
            self.line(format!("add.u64 {column}, {tile_n}, {column_offset};"));
            let row_ok = self.pred();
            let column_ok = self.pred();
            let valid = self.pred();
            self.line(format!("setp.lt.u64 {row_ok}, {row}, {rows};"));
            self.line(format!("setp.lt.u64 {column_ok}, {column}, {columns};"));
            self.line(format!("and.pred {valid}, {row_ok}, {column_ok};"));
            let skip = self.label("nvfp4_store_skip");
            self.line(format!("@!{valid} bra {skip};"));
            let value = self.f32();
            self.line(format!("mov.b32 {value}, {bits};"));
            let global_scale = self.f32();
            self.line(format!(
                "mul.rn.f32 {global_scale}, {left_scale}, {right_scale};"
            ));
            self.line(format!("mul.rn.f32 {value}, {value}, {global_scale};"));
            if let (Some(c_map), Some(c)) = (c_map, c.as_ref()) {
                let initial = self.matrix_load_f32(c_map, c, &row, &column, &rows, &columns);
                self.line(format!("add.rn.f32 {value}, {value}, {initial};"));
            }
            self.matrix_store_f32(
                destination_map,
                &destination,
                &row,
                &column,
                &rows,
                &columns,
                &value,
            );
            self.raw(format!("{skip}:"));
        }
        self.line("bar.sync 0;");
        self.line(format!("add.u64 {tile}, {tile}, {block_count};"));
        self.line(format!("bra {tile_loop};"));
        self.raw(format!("{tile_done}:"));
        let after_dealloc = self.label("nvfp4_after_dealloc");
        self.line(format!("@!{first_warp} bra {after_dealloc};"));
        self.line(format!(
            "tcgen05.dealloc.cta_group::1.sync.aligned.b32 {tmem}, {units};"
        ));
        self.raw(format!("{after_dealloc}:"));
        self.line("bar.sync 0;");
    }

    fn nvfp4_epilogue(&mut self) {
        if !self.uses_nvfp4 {
            return;
        }
        let warp0 = self.pred();
        let done = self.label("nvfp4_relinquished");
        self.line(format!("setp.lt.u32 {warp0}, %t13, 32;"));
        self.line(format!("@!{warp0} bra {done};"));
        self.line("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
        self.raw(format!("{done}:"));
    }

    fn shared_address(&mut self, symbol: &str) -> String {
        let shared = self.t64();
        self.line(format!("mov.u64 {shared}, {symbol};"));
        shared
    }

    fn nvfp4_shared_descriptor(
        &mut self,
        shared: &str,
        leading_encoded: u64,
        stride_encoded: u64,
    ) -> String {
        let encoded = self.t64();
        let descriptor = self.t64();
        self.line(format!("shr.u64 {encoded}, {shared}, 4;"));
        self.line(format!("and.b64 {encoded}, {encoded}, 0x3fff;"));
        let fixed = (1u64 << 46) | (leading_encoded << 16) | (stride_encoded << 32);
        self.line(format!("or.b64 {descriptor}, {encoded}, 0x{fixed:016x};"));
        descriptor
    }

    fn nvfp4_validate_geometry(&self, place: &ClosedReadablePlace) {
        let ReadableRepresentationGeometry::Packed(packed) = &place.geometry else {
            panic!("NVFP4 matrix operand is not packed")
        };
        let [codes, scales] = packed.layout.planes.as_slice() else {
            panic!("NVFP4 matrix operand does not have code and block-scale planes")
        };
        if !matches!(
            codes.encoding,
            PlaneEncoding::FloatCode {
                format: FloatCodeFormat::E2M1
            }
        ) || !matches!(
            scales.encoding,
            PlaneEncoding::FloatCode {
                format: FloatCodeFormat::UE4M3
            }
        ) || codes.group != 1
            || scales.group != 16
        {
            panic!("NVFP4 matrix operand has a non-E2M1/block16-UE4M3 physical layout")
        }
    }

    fn nvfp4_raw(
        &mut self,
        map: &LogicalTensorMap,
        place: &ClosedReadablePlace,
        row: &str,
        column: &str,
        scale: bool,
    ) -> String {
        let coordinates = self.logical_coords(map, &[row.to_string(), column.to_string()]);
        let (packet, geometry, logical) = self.address_names(place, &coordinates);
        let ReadableRepresentationGeometry::Packed(packed) = geometry else {
            unreachable!("NVFP4 geometry was validated before staging")
        };
        let plane = packed.layout.planes[usize::from(scale)].clone();
        self.plane_field(
            &packet,
            &logical,
            packed.layout.group,
            &plane,
            0,
            DType::U32,
        )
    }

    fn nvfp4_raw_guarded(
        &mut self,
        map: &LogicalTensorMap,
        place: &ClosedReadablePlace,
        row: &str,
        column: &str,
        row_bound: &str,
        column_bound: &str,
        scale: bool,
    ) -> String {
        let value = self.t32();
        self.line(format!("mov.u32 {value}, 0;"));
        let row_ok = self.pred();
        let column_ok = self.pred();
        let valid = self.pred();
        let done = self.label("nvfp4_guarded_read");
        self.line(format!("setp.lt.u64 {row_ok}, {row}, {row_bound};"));
        self.line(format!(
            "setp.lt.u64 {column_ok}, {column}, {column_bound};"
        ));
        self.line(format!("and.pred {valid}, {row_ok}, {column_ok};"));
        self.line(format!("@!{valid} bra {done};"));
        let raw = self.nvfp4_raw(map, place, row, column, scale);
        self.line(format!("mov.u32 {value}, {raw};"));
        self.raw(format!("{done}:"));
        value
    }

    fn nvfp4_stage_a(
        &mut self,
        map: &LogicalTensorMap,
        place: &ClosedReadablePlace,
        tile_m: &str,
        tile_k: &str,
        rows: &str,
        inner: &str,
        thread: &str,
        shared: &str,
    ) {
        let byte = self.t64();
        self.line(format!("mov.u64 {byte}, 0;"));
        let loop_label = self.label("nvfp4_stage_a");
        let done = self.label("nvfp4_stage_a_done");
        self.raw(format!("{loop_label}:"));
        let finished = self.pred();
        self.line(format!("setp.ge.u64 {finished}, {byte}, 32;"));
        self.line(format!("@{finished} bra {done};"));
        let row = self.t64();
        let k0 = self.t64();
        let k1 = self.t64();
        self.line(format!("add.u64 {row}, {tile_m}, {thread};"));
        self.line(format!("mad.lo.u64 {k0}, {byte}, 2, {tile_k};"));
        self.line(format!("add.u64 {k1}, {k0}, 1;"));
        let low = self.nvfp4_raw_guarded(map, place, &row, &k0, rows, inner, false);
        let high = self.nvfp4_raw_guarded(map, place, &row, &k1, rows, inner, false);
        let shifted = self.t32();
        let packed = self.t32();
        self.line(format!("shl.b32 {shifted}, {high}, 4;"));
        self.line(format!("or.b32 {packed}, {low}, {shifted};"));
        let row_minor = self.t64();
        let row_major = self.t64();
        let offset = self.t64();
        self.line(format!("rem.u64 {row_minor}, {thread}, 8;"));
        self.line(format!("div.u64 {row_major}, {thread}, 8;"));
        self.line(format!("mul.lo.u64 {row_minor}, {row_minor}, 16;"));
        self.line(format!("mul.lo.u64 {row_major}, {row_major}, 128;"));
        self.line(format!("add.u64 {offset}, {row_minor}, {row_major};"));
        let second_half = self.pred();
        self.line(format!("setp.ge.u64 {second_half}, {byte}, 16;"));
        self.line(format!("@{second_half} add.u64 {offset}, {offset}, 2048;"));
        let within_half = self.t64();
        self.line(format!("rem.u64 {within_half}, {byte}, 16;"));
        self.line(format!("add.u64 {offset}, {offset}, {within_half};"));
        let address = self.t64();
        self.line(format!("add.u64 {address}, {shared}, {offset};"));
        self.line(format!("st.shared.u8 [{address}], {packed};"));
        self.line(format!("add.u64 {byte}, {byte}, 1;"));
        self.line(format!("bra {loop_label};"));
        self.raw(format!("{done}:"));
    }

    fn nvfp4_stage_b(
        &mut self,
        map: &LogicalTensorMap,
        place: &ClosedReadablePlace,
        tile_n: &str,
        tile_k: &str,
        columns: &str,
        inner: &str,
        thread: &str,
        shared: &str,
    ) {
        for half in 0..2 {
            let offset = self.t64();
            self.line(format!("mad.lo.u64 {offset}, {thread}, 2, {half};"));
            let k_half = self.t64();
            let in_half = self.t64();
            let n_local = self.t64();
            let pair = self.t64();
            self.line(format!("div.u64 {k_half}, {offset}, 128;"));
            self.line(format!("rem.u64 {in_half}, {offset}, 128;"));
            self.line(format!("div.u64 {n_local}, {in_half}, 16;"));
            self.line(format!("rem.u64 {pair}, {in_half}, 16;"));
            let n = self.t64();
            let k0 = self.t64();
            let k1 = self.t64();
            self.line(format!("add.u64 {n}, {tile_n}, {n_local};"));
            self.line(format!("mul.lo.u64 {k_half}, {k_half}, 32;"));
            self.line(format!("mad.lo.u64 {k0}, {pair}, 2, {k_half};"));
            self.line(format!("add.u64 {k0}, {k0}, {tile_k};"));
            self.line(format!("add.u64 {k1}, {k0}, 1;"));
            let low = self.nvfp4_raw_guarded(map, place, &k0, &n, inner, columns, false);
            let high = self.nvfp4_raw_guarded(map, place, &k1, &n, inner, columns, false);
            let shifted = self.t32();
            let packed = self.t32();
            self.line(format!("shl.b32 {shifted}, {high}, 4;"));
            self.line(format!("or.b32 {packed}, {low}, {shifted};"));
            let address = self.t64();
            self.line(format!("add.u64 {address}, {shared}, {offset};"));
            self.line(format!("st.shared.u8 [{address}], {packed};"));
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn nvfp4_store_scales(
        &mut self,
        a_map: &LogicalTensorMap,
        b_map: &LogicalTensorMap,
        a: &ClosedReadablePlace,
        b: &ClosedReadablePlace,
        tile_m: &str,
        tile_n: &str,
        tile_k: &str,
        rows: &str,
        columns: &str,
        inner: &str,
        thread: &str,
        tmem: &str,
        first_warp: &str,
    ) {
        let done = self.label("nvfp4_scales_stored");
        self.line(format!("@!{first_warp} bra {done};"));
        let mut a_registers = Vec::with_capacity(4);
        for row_group in 0..4u64 {
            let row = self.t64();
            self.line(format!("add.u64 {row}, {tile_m}, {thread};"));
            if row_group != 0 {
                self.line(format!("add.u64 {row}, {row}, {};", row_group * 32));
            }
            let mut bytes = Vec::with_capacity(4);
            for scale in 0..4u64 {
                let k = self.t64();
                self.line(format!("add.u64 {k}, {tile_k}, {};", scale * 16));
                bytes.push(self.nvfp4_raw_guarded(a_map, a, &row, &k, rows, inner, true));
            }
            a_registers.push(self.pack_bytes(&bytes));
        }
        let sfa = self.t32();
        self.line(format!("add.u32 {sfa}, {tmem}, 8;"));
        self.line(format!(
            "tcgen05.st.sync.aligned.32x32b.x4.b32 [{sfa}], {{{}}};",
            a_registers.join(", ")
        ));
        self.line("tcgen05.wait::st.sync.aligned;");

        let n = self.t64();
        self.line(format!("add.u64 {n}, {tile_n}, {thread};"));
        let in_n_tile = self.pred();
        self.line(format!("setp.lt.u64 {in_n_tile}, {thread}, 8;"));
        let mut b_bytes = Vec::with_capacity(4);
        for scale in 0..4u64 {
            let k = self.t64();
            self.line(format!("add.u64 {k}, {tile_k}, {};", scale * 16));
            let value = self.nvfp4_raw_guarded(b_map, b, &k, &n, inner, columns, true);
            self.line(format!("@!{in_n_tile} mov.u32 {value}, 0;"));
            b_bytes.push(value);
        }
        let b_register = self.pack_bytes(&b_bytes);
        let sfb = self.t32();
        self.line(format!("add.u32 {sfb}, {tmem}, 12;"));
        self.line(format!(
            "tcgen05.st.sync.aligned.32x32b.x1.b32 [{sfb}], {{{b_register}}};"
        ));
        self.line("tcgen05.wait::st.sync.aligned;");
        self.raw(format!("{done}:"));
    }

    fn pack_bytes(&mut self, bytes: &[String]) -> String {
        assert_eq!(bytes.len(), 4, "NVFP4 scale word has four bytes");
        let result = self.t32();
        self.line(format!("mov.u32 {result}, {};", bytes[0]));
        for (index, byte) in bytes.iter().enumerate().skip(1) {
            let shifted = self.t32();
            self.line(format!("shl.b32 {shifted}, {byte}, {};", index * 8));
            self.line(format!("or.b32 {result}, {result}, {shifted};"));
        }
        result
    }

    fn matrix_load_u16(
        &mut self,
        map: &LogicalTensorMap,
        place: &ClosedDensePlace,
        row: &str,
        column: &str,
        rows: &str,
        columns: &str,
    ) -> String {
        let (dtype, address) = self.matrix_address(map, place, row, column);
        if !matches!(dtype, DType::F16 | DType::BF16) {
            panic!("SM80 matrix operand is not a 16-bit float")
        }
        let row_ok = self.pred();
        let column_ok = self.pred();
        let valid = self.pred();
        self.line(format!("setp.lt.u64 {row_ok}, {row}, {rows};"));
        self.line(format!("setp.lt.u64 {column_ok}, {column}, {columns};"));
        self.line(format!("and.pred {valid}, {row_ok}, {column_ok};"));
        let narrow = self.h16();
        self.line(format!("mov.u16 {narrow}, 0;"));
        self.line(format!("@{valid} ld.global.u16 {narrow}, [{address}];"));
        let value = self.t32();
        self.line(format!("cvt.u32.u16 {value}, {narrow};"));
        value
    }

    fn pack_u16(&mut self, low: &str, high: &str) -> String {
        let shifted = self.t32();
        let packed = self.t32();
        self.line(format!("shl.b32 {shifted}, {high}, 16;"));
        self.line(format!("or.b32 {packed}, {low}, {shifted};"));
        packed
    }

    fn matrix_load_f32(
        &mut self,
        map: &LogicalTensorMap,
        place: &ClosedDensePlace,
        row: &str,
        column: &str,
        rows: &str,
        columns: &str,
    ) -> String {
        let (dtype, address) = self.matrix_address(map, place, row, column);
        let row_ok = self.pred();
        let column_ok = self.pred();
        let valid = self.pred();
        self.line(format!("setp.lt.u64 {row_ok}, {row}, {rows};"));
        self.line(format!("setp.lt.u64 {column_ok}, {column}, {columns};"));
        self.line(format!("and.pred {valid}, {row_ok}, {column_ok};"));
        let value = self.f32();
        self.line(format!("mov.f32 {value}, 0f00000000;"));
        match dtype {
            DType::F32 => self.line(format!("@{valid} ld.global.f32 {value}, [{address}];")),
            DType::F16 | DType::BF16 => {
                let raw = self.h16();
                self.line(format!("mov.u16 {raw}, 0;"));
                self.line(format!("@{valid} ld.global.u16 {raw}, [{address}];"));
                self.line(format!(
                    "@{valid} cvt.f32{} {value}, {raw};",
                    dtype_suffix(dtype)
                ));
            }
            _ => panic!("matrix accumulator is not floating point"),
        }
        value
    }

    fn matrix_store_f32(
        &mut self,
        map: &LogicalTensorMap,
        place: &ClosedDensePlace,
        row: &str,
        column: &str,
        rows: &str,
        columns: &str,
        value: &str,
    ) {
        let (dtype, address) = self.matrix_address(map, place, row, column);
        let row_ok = self.pred();
        let column_ok = self.pred();
        let valid = self.pred();
        self.line(format!("setp.lt.u64 {row_ok}, {row}, {rows};"));
        self.line(format!("setp.lt.u64 {column_ok}, {column}, {columns};"));
        self.line(format!("and.pred {valid}, {row_ok}, {column_ok};"));
        match dtype {
            DType::F32 => self.line(format!("@{valid} st.global.f32 [{address}], {value};")),
            DType::F16 | DType::BF16 => {
                let raw = self.h16();
                self.line(format!(
                    "cvt.rn{}{}.f32 {raw}, {value};",
                    dtype_suffix(dtype),
                    ""
                ));
                self.line(format!("@{valid} st.global.u16 [{address}], {raw};"));
            }
            _ => panic!("matrix destination is not floating point"),
        }
    }

    fn matrix_address(
        &mut self,
        map: &LogicalTensorMap,
        place: &ClosedDensePlace,
        row: &str,
        column: &str,
    ) -> (DType, String) {
        let coordinates = self.logical_coords(map, &[row.to_owned(), column.to_owned()]);
        let (address, geometry, _) = self.address_names(place, &coordinates);
        (geometry.dtype, address)
    }
    fn branch(
        &mut self,
        cond: ErasedValue,
        then_block: BlockId,
        else_block: BlockId,
        outs: &[ClosedValue],
    ) {
        let otherwise = self.label("else");
        let join = self.label("join");
        let p = self.truth_bool(cond);
        self.line(format!("@!{p} bra {otherwise};"));
        let then_values = self
            .emit_block(then_block)
            .expect("typed branch arm yields");
        for (out, value) in outs.iter().zip(then_values) {
            self.copy_value(*out, value);
        }
        self.line(format!("bra {join};"));
        self.raw(format!("{otherwise}:"));
        let else_values = self
            .emit_block(else_block)
            .expect("typed branch arm yields");
        for (out, value) in outs.iter().zip(else_values) {
            self.copy_value(*out, value);
        }
        self.raw(format!("{join}:"));
    }
    fn repeat(
        &mut self,
        start: ErasedValue,
        end: ErasedValue,
        binder: ErasedValue,
        carries: &[ClosedValue],
        params: &[ClosedValue],
        body: BlockId,
        outs: &[ClosedValue],
    ) {
        self.line(format!("mov.u64 {}, {};", self.v(binder), self.v(start)));
        for (param, value) in params.iter().zip(carries) {
            self.copy_value(*param, value.value);
        }
        let head = self.label("repeat");
        let done = self.label("repeat_done");
        self.raw(format!("{head}:"));
        let p = self.pred();
        self.line(format!(
            "setp.ge.u64 {p}, {}, {};",
            self.v(binder),
            self.v(end)
        ));
        self.line(format!("@{p} bra {done};"));
        let yielded = self.emit_block(body).expect("typed repeat body yields");
        for (param, value) in params.iter().zip(yielded) {
            self.copy_value(*param, value);
        }
        self.line(format!(
            "add.u64 {}, {}, 1;",
            self.v(binder),
            self.v(binder)
        ));
        self.line(format!("bra {head};"));
        self.raw(format!("{done}:"));
        for (out, param) in outs.iter().zip(params) {
            self.copy_value(*out, param.value);
        }
    }

    fn copy_value(&mut self, destination: ClosedValue, source: ErasedValue) {
        match destination.ty {
            ValueType::Vector { dtype, lanes } => {
                for lane in 0..lanes {
                    self.line(format!(
                        "mov{} {}, {};",
                        scalar_suffix(dtype),
                        self.vector_lane(destination.value, lane),
                        self.vector_lane(source, lane)
                    ));
                }
            }
            ty => self.line(format!(
                "mov{} {}, {};",
                suffix(ty),
                self.v(destination.value),
                self.v(source)
            )),
        }
    }
    fn round(&mut self, out: ErasedValue, out_type: ValueType) {
        let ValueType::Scalar(dtype) = out_type else {
            return;
        };
        let out = self.v(out);
        self.round_named(&out, dtype);
    }
    fn round_named(&mut self, out: &str, dtype: DType) {
        match dtype {
            DType::F16 => {
                let bits = self.h16();
                self.line(format!("cvt.rn.f16.f32 {bits}, {out};"));
                self.line(format!("cvt.f32.f16 {out}, {bits};"));
            }
            DType::BF16 => {
                let bits = self.h16();
                self.line(format!("cvt.rn.bf16.f32 {bits}, {out};"));
                self.line(format!("cvt.f32.bf16 {out}, {bits};"));
            }
            _ => {}
        }
    }
}

fn ptx_type(ty: ValueType) -> &'static str {
    match ty {
        ValueType::Scalar(dtype) if dtype.is_float() => ".f32",
        ValueType::Index => ".u64",
        ValueType::Scalar(DType::I32) => ".s32",
        ValueType::Scalar(_) | ValueType::Bool => ".u32",
        ValueType::Vector { .. } => {
            panic!("CUDA vectors are declared as scalarized lane registers")
        }
        ValueType::Opaque { .. } => ".b32",
    }
}
fn scalar_ptx_type(dtype: DType) -> &'static str {
    if dtype.is_float() {
        ".f32"
    } else if dtype == DType::I32 {
        ".s32"
    } else {
        ".u32"
    }
}
fn scalar_suffix(dtype: DType) -> &'static str {
    if dtype.is_float() {
        ".f32"
    } else if dtype == DType::I32 {
        ".s32"
    } else {
        ".u32"
    }
}
fn scalar_zero(dtype: DType) -> &'static str {
    if dtype.is_float() {
        "0f00000000"
    } else {
        "0"
    }
}
fn vector_shape(ty: ValueType) -> (DType, u16) {
    match ty {
        ValueType::Vector { dtype, lanes } => (dtype, lanes),
        _ => panic!("typed CUDA vector operation carries a non-vector value"),
    }
}
fn suffix(ty: ValueType) -> &'static str {
    match ty {
        ValueType::Scalar(dtype) if dtype.is_float() => ".f32",
        ValueType::Index => ".u64",
        ValueType::Scalar(DType::I32) => ".s32",
        _ => ".u32",
    }
}
fn dtype_suffix(dtype: DType) -> &'static str {
    match dtype {
        DType::F32 => ".f32",
        DType::F16 => ".f16",
        DType::BF16 => ".bf16",
        DType::I32 => ".s32",
        DType::U32 => ".u32",
        DType::Bool => ".u8",
    }
}
fn axis(axis: u8) -> &'static str {
    match axis {
        0 => "x",
        1 => "y",
        2 => "z",
        _ => panic!("typed geometry axis exceeds rank three"),
    }
}

fn op_values<B: seismic_ir::target::KernelDialect>(op: &Op<B>) -> Vec<ErasedValue> {
    match op {
        Op::Constant { out, .. }
        | Op::Geometry { out, .. }
        | Op::NatArg { out, .. }
        | Op::ScalarArg { out, .. }
        | Op::Extent { out, .. } => vec![*out],
        Op::Binary { out, a, b, .. }
        | Op::Bit { out, a, b, .. }
        | Op::Cmp { out, a, b, .. }
        | Op::Logic { out, a, b, .. } => vec![*out, *a, *b],
        Op::Unary { out, a, .. }
        | Op::Math { out, a, .. }
        | Op::Cast { out, a, .. }
        | Op::Bitcast { out, a, .. }
        | Op::Not { out, a } => vec![*out, *a],
        Op::Fma { out, a, b, c } => vec![*out, *a, *b, *c],
        Op::VectorSplat { out, value }
        | Op::VectorLane {
            out, vector: value, ..
        } => {
            vec![*out, *value]
        }
        Op::VectorBinary { out, a, b, .. } | Op::VectorBit { out, a, b, .. } => {
            vec![*out, *a, *b]
        }
        Op::VectorUnary { out, a, .. }
        | Op::VectorCast { out, a, .. }
        | Op::VectorReduceAdd { out, vector: a } => vec![*out, *a],
        Op::VectorFma { out, a, b, c } => vec![*out, *a, *b, *c],
        Op::Select { out, cond, a, b } => vec![*out, *cond, *a, *b],
        Op::Read { out, index, .. } | Op::ReadPlane { out, index, .. } => {
            let mut v = vec![*out];
            v.extend(index);
            v
        }
        Op::VectorRead {
            out, index, active, ..
        } => {
            let mut values = vec![*out, *active];
            values.extend(index);
            values
        }
        Op::VectorWrite {
            index,
            active,
            value,
            ..
        } => {
            let mut values = vec![*active, *value];
            values.extend(index);
            values
        }
        Op::RepresentationConvertPacket { packet, .. } => vec![*packet],
        Op::Write { index, value, .. } | Op::Atomic { index, value, .. } => {
            let mut v = index.clone();
            v.push(*value);
            v
        }
        Op::StoreSlot { value, .. } => vec![*value],
        Op::Barrier(_) => vec![],
        Op::Intrinsic { outs, args, .. } => outs.iter().chain(args).copied().collect(),
        Op::Branch { cond, outs, .. } => {
            std::iter::once(*cond).chain(outs.iter().copied()).collect()
        }
        Op::Repeat {
            start,
            end,
            binder,
            carries_in,
            carry_params,
            outs,
            ..
        } => {
            let mut values = vec![*start, *end, *binder];
            values.extend(carries_in);
            values.extend(carry_params);
            values.extend(outs);
            values
        }
        Op::Yield { values } => values.clone(),
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn nvfp4_emitter_contains_complete_tcgen05_lifecycle() {
        let source = include_str!("ptx.rs");
        let start = source.find("fn matrix_nvfp4(").expect("NVFP4 emitter");
        let end = source[start..]
            .find("fn matrix_load_u16(")
            .map(|offset| start + offset)
            .expect("end of NVFP4 helpers");
        let emitter = &source[start..end];
        for instruction in [
            "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32",
            "tcgen05.st.sync.aligned.32x32b.x4.b32",
            "tcgen05.st.sync.aligned.32x32b.x1.b32",
            "tcgen05.wait::st.sync.aligned",
            "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.block16",
            "tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64",
            "mbarrier.try_wait.parity.b64",
            "xor.b32 {barrier_phase}, {barrier_phase}, 1",
            "tcgen05.fence::after_thread_sync",
            "tcgen05.ld.sync.aligned.32x32b.x8.b32",
            "tcgen05.wait::ld.sync.aligned",
            "tcgen05.dealloc.cta_group::1.sync.aligned.b32",
            "tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned",
        ] {
            assert!(emitter.contains(instruction), "missing {instruction}");
        }
    }

    #[test]
    fn nvfp4_descriptors_and_tmem_partition_are_exact() {
        let source = include_str!("ptx.rs");
        assert!(source.contains("nvfp4_shared_descriptor(&a_shared, 128, 8)"));
        assert!(source.contains("nvfp4_shared_descriptor(&b_shared, 8, 8)"));
        assert!(source.contains("0x08020480"));
        assert!(source.contains("add.u32 {sfa}, {tmem}, 8"));
        assert!(source.contains("add.u32 {sfb}, {tmem}, 12"));
        assert!(source.contains("setp.lt.u64 {in_n_tile}, {thread}, 8"));
        assert!(source.contains("@!{in_n_tile} mov.u32 {value}, 0"));
        assert!(source.contains("seismic_nvfp4_a[4096]"));
        assert!(source.contains("seismic_nvfp4_b[256]"));
    }
}
