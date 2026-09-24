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
    ReadableRepresentationGeometry,
};
use seismic_lang::intrinsics::{AtomicOp, MathOp, ReduceOp};
use seismic_lang::registry::{
    CodeInterpretation, DecodeStep, FloatCodeFormat, PlaneEncoding, PlaneInfo, PlaneRepackRecipe,
    RepackExpr,
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
            Op::Read { index, .. } => 28 + index.len() as u64 * 4,
            Op::ReadPlaneField { index, .. }
            | Op::ReadPlane { index, .. }
            | Op::Write { index, .. }
            | Op::Atomic { index, .. } => 64 + index.len() as u64 * 4,
            Op::VectorRead { out, index, .. } => {
                let lanes = match self.kernel.value_type(*out) {
                    ValueType::Vector { lanes, .. } => u64::from(lanes),
                    _ => panic!("typed CUDA vector read has a scalar result"),
                };
                lanes * (36 + index.len() as u64 * 4)
            }
            Op::VectorWrite { value, index, .. } => {
                let lanes = match self.kernel.value_type(*value) {
                    ValueType::Vector { lanes, .. } => u64::from(lanes),
                    _ => panic!("typed CUDA vector write has a scalar value"),
                };
                lanes * (32 + index.len() as u64 * 4)
            }
            Op::VectorBinary { out, .. }
            | Op::VectorUnary { out, .. }
            | Op::VectorFma { out, .. }
            | Op::VectorCast { out, .. } => {
                let ValueType::Vector { lanes, .. } = self.kernel.value_type(*out) else {
                    unreachable!()
                };
                u64::from(lanes) * 24
            }
            Op::VectorReduceAdd { vector, .. } => {
                let ValueType::Vector { lanes, .. } = self.kernel.value_type(*vector) else {
                    unreachable!()
                };
                u64::from(lanes) * 24
            }
            Op::Intrinsic {
                op: CudaIntrinsic::SubgroupReduce { .. },
                ..
            } => 128,
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
            Op::Repeat { carry_params, .. } => carry_params.iter().fold(16u64, |count, value| {
                let lanes = match self.kernel.value_type(*value) {
                    ValueType::Vector { lanes, .. } => u64::from(lanes),
                    _ => 1,
                };
                count
                    .checked_add(lanes)
                    .expect("typed CUDA carry temporary inventory overflows u64")
            }),
            Op::Branch { .. } => 16,
            _ => 24,
        }
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
                    let ValueType::Scalar(dtype) = out.ty else {
                        unreachable!()
                    };
                    self.fma_named(
                        dtype,
                        &self.v(out.value),
                        &self.v(a.value),
                        &self.v(b.value),
                        &self.v(c.value),
                    );
                }
                ClosedOpView::VectorFromLanes { out, lanes } => {
                    for (index, lane) in lanes.iter().enumerate() {
                        self.line(format!(
                            "mov{} {}, {};",
                            suffix(lane.ty),
                            self.vector_lane(out.value, index as u16),
                            self.v(lane.value)
                        ));
                    }
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
                ClosedOpView::ScalarBits { out, a } => self.line(format!(
                    "mov.b32 {}, {};",
                    self.v(out.value),
                    self.v(a.value)
                )),
                ClosedOpView::ScalarFromBits { out, a } => self.line(format!(
                    "and.b32 {}, {}, 65535;",
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
                    out, index, kind, ..
                } => self.scalar_arg(out.value, index, kind),
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
                ClosedOpView::ReadPlaneField {
                    out,
                    place,
                    plane_info,
                    field,
                    indices,
                    ..
                } => {
                    let index = indices.iter().map(|v| v.value()).collect::<Vec<_>>();
                    let (address, geometry, logical) = self.address(&place, &index);
                    let ValueType::Scalar(dtype) = out.ty else {
                        unreachable!("typed packed field result")
                    };
                    let value = self.plane_field(
                        &address,
                        &logical,
                        geometry.layout.group,
                        &plane_info,
                        field,
                        dtype,
                    );
                    self.line(format!(
                        "mov{} {}, {value};",
                        suffix(out.ty),
                        self.v(out.value)
                    ));
                }
                ClosedOpView::ReadPlane {
                    out,
                    place,
                    plane_info,
                    element,
                    indices,
                    ..
                } => self.read_plane(
                    out.value,
                    out.ty,
                    &place,
                    &plane_info,
                    element.value(),
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
                    kind,
                    value,
                    election: StoreElection::GlobalLeader,
                } => self.store_slot(slot, kind, value),
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
            ConstantValue::F16(v) | ConstantValue::BF16(v) => v.to_string(),
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
        if let ValueType::Scalar(dtype) = ty {
            let left = self.numeric_operand(dtype, &self.v(a));
            let right = self.numeric_operand(dtype, &self.v(b));
            let destination = self.numeric_destination(dtype, &self.v(out));
            self.line(format!("{mnemonic} {destination}, {left}, {right};"));
            if matches!(op, BinaryOp::Div | BinaryOp::Rem) && dtype == DType::I32 {
                self.euclidean_fix(op, out, a, b);
            }
            self.publish_numeric(dtype, &self.v(out), &destination);
        } else {
            self.line(format!(
                "{mnemonic} {}, {}, {};",
                self.v(out),
                self.v(a),
                self.v(b)
            ));
        }
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
        let ValueType::Scalar(dtype) = ty else {
            panic!("scalar unary type")
        };
        self.unary_named(op, dtype, &self.v(out), &self.v(a));
    }
    fn unary_named(&mut self, op: UnaryOp, dtype: DType, out: &str, value: &str) {
        if dtype.is_float() {
            let bits = self.t32();
            self.line(format!("mov.b32 {bits}, {value};"));
            let (instruction, mask) = match (op, dtype == DType::F32) {
                (UnaryOp::Neg, true) => ("xor.b32", 0x80000000u32),
                (UnaryOp::Neg, false) => ("xor.b32", 0x8000),
                (UnaryOp::Abs, true) => ("and.b32", 0x7fffffff),
                (UnaryOp::Abs, false) => ("and.b32", 0x7fff),
            };
            self.line(format!("{instruction} {bits}, {bits}, {mask};"));
            self.line(format!("mov.b32 {out}, {bits};"));
        } else {
            self.line(format!(
                "{}.s32 {out}, {value};",
                match op {
                    UnaryOp::Neg => "neg",
                    UnaryOp::Abs => "abs",
                }
            ));
        }
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
            let physical_left = self.numeric_operand(dtype, &left);
            let physical_right = self.numeric_operand(dtype, &right);
            let physical_out = self.numeric_destination(dtype, &destination);
            self.line(format!(
                "{mnemonic} {physical_out}, {physical_left}, {physical_right};"
            ));
            if matches!(op, BinaryOp::Div | BinaryOp::Rem) && dtype == DType::I32 {
                self.euclidean_fix_named(op, &destination, &left, &right);
            }
            self.publish_numeric(dtype, &destination, &physical_out);
        }
    }

    fn vector_unary(&mut self, op: UnaryOp, out: ClosedValue, a: ClosedValue) {
        let (dtype, lanes) = vector_shape(out.ty);
        for lane in 0..lanes {
            self.unary_named(
                op,
                dtype,
                &self.vector_lane(out.value, lane),
                &self.vector_lane(a.value, lane),
            );
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

    fn fma_named(&mut self, dtype: DType, out: &str, a: &str, b: &str, c: &str) {
        let a = self.numeric_operand(dtype, a);
        let b = self.numeric_operand(dtype, b);
        let c = self.numeric_operand(dtype, c);
        let destination = self.numeric_destination(dtype, out);
        self.line(format!("fma.rn.f32 {destination}, {a}, {b}, {c};"));
        self.publish_numeric(dtype, out, &destination);
    }
    fn vector_fma(&mut self, out: ClosedValue, a: ClosedValue, b: ClosedValue, c: ClosedValue) {
        let (dtype, lanes) = vector_shape(out.ty);
        for lane in 0..lanes {
            self.fma_named(
                dtype,
                &self.vector_lane(out.value, lane),
                &self.vector_lane(a.value, lane),
                &self.vector_lane(b.value, lane),
                &self.vector_lane(c.value, lane),
            );
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
            let left = self.numeric_operand(dtype, &destination);
            let right = self.numeric_operand(dtype, &self.vector_lane(vector.value, lane));
            let physical_out = self.numeric_destination(dtype, &destination);
            self.line(format!(
                "{} {physical_out}, {left}, {right};",
                if dtype.is_float() {
                    "add.rn.f32"
                } else {
                    "add.u32"
                }
            ));
            self.publish_numeric(dtype, &destination, &physical_out);
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
            return;
        }
        let value = self.numeric_operand(from, value);
        let destination = self.numeric_destination(to, out);
        if from.is_float() && to.is_float() {
            self.line(format!("mov.f32 {destination}, {value};"));
        } else if from.is_float() {
            self.line(format!(
                "cvt.rzi{}.f32 {destination}, {value};",
                if to == DType::I32 { ".s32" } else { ".u32" }
            ));
        } else if to.is_float() {
            self.line(format!(
                "cvt.rn.f32{} {destination}, {value};",
                if from == DType::I32 { ".s32" } else { ".u32" }
            ));
        } else {
            self.line(format!("mov.b32 {destination}, {value};"));
        }
        self.publish_numeric(to, out, &destination);
    }
    fn math(&mut self, op: MathOp, out: ErasedValue, out_type: ValueType, a: ErasedValue) {
        let ValueType::Scalar(dtype) = out_type else {
            unreachable!()
        };
        if op == MathOp::Abs {
            self.unary_named(UnaryOp::Abs, dtype, &self.v(out), &self.v(a));
            return;
        }
        let argument = self.numeric_operand(dtype, &self.v(a));
        let destination = self.numeric_destination(dtype, &self.v(out));
        match op {
            MathOp::Sqrt => self.line(format!("sqrt.rn.f32 {}, {};", destination, argument)),
            MathOp::Rsqrt => self.line(format!("rsqrt.approx.f32 {}, {};", destination, argument)),
            MathOp::Abs => self.line(format!("abs.f32 {}, {};", destination, argument)),
            MathOp::Exp => {
                let scaled = self.f32();
                self.line(format!("mul.rn.f32 {scaled}, {}, 0f3fb8aa3b;", argument));
                self.line(format!("ex2.approx.f32 {}, {scaled};", destination));
            }
            MathOp::Log => self.line(format!("lg2.approx.f32 {}, {};", destination, argument)),
            MathOp::Sin => self.line(format!("sin.approx.f32 {}, {};", destination, argument)),
            MathOp::Cos => self.line(format!("cos.approx.f32 {}, {};", destination, argument)),
            MathOp::Max | MathOp::Min | MathOp::Fma => {
                panic!("multi-operand math op encoded as unary IR")
            }
        }
        self.publish_numeric(dtype, &self.v(out), &destination);
    }
    fn cast(&mut self, out: ErasedValue, a: ErasedValue, from: ValueType, to: ValueType) {
        if from == to {
            self.line(format!("mov{} {}, {};", suffix(to), self.v(out), self.v(a)));
            return;
        }
        let source = match from {
            ValueType::Scalar(dtype) => self.numeric_operand(dtype, &self.v(a)),
            _ => self.v(a),
        };
        let destination = match to {
            ValueType::Scalar(dtype) => self.numeric_destination(dtype, &self.v(out)),
            _ => self.v(out),
        };
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
                source,
                if ff { "0f00000000" } else { "0" }
            ));
            self.line(format!("selp.u32 {}, 1, 0, {p};", destination));
        } else if ff && tf {
            self.line(format!("mov.f32 {}, {};", destination, source));
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
                destination,
                source
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
                destination,
                source
            ));
        } else if fb || to == ValueType::Index || from == ValueType::Index {
            self.line(format!(
                "cvt{}{} {}, {};",
                suffix(to),
                suffix(from),
                destination,
                source
            ));
        } else {
            self.line(format!("mov.b32 {}, {};", destination, source));
        }
        if let ValueType::Scalar(dtype) = to {
            self.publish_numeric(dtype, &self.v(out), &destination);
        }
    }
    fn compare(
        &mut self,
        op: CmpOp,
        out: ErasedValue,
        a: ErasedValue,
        ty: ValueType,
        b: ErasedValue,
    ) {
        let left = match ty {
            ValueType::Scalar(dtype) => self.numeric_operand(dtype, &self.v(a)),
            _ => self.v(a),
        };
        let right = match ty {
            ValueType::Scalar(dtype) => self.numeric_operand(dtype, &self.v(b)),
            _ => self.v(b),
        };
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
        self.line(format!("setp.{cmp}.{class} {p}, {}, {};", left, right));
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
            // %t13 is the stable flattened thread index formed in the launch
            // prologue; %warpid is a scheduling identity and is unsuitable.
            GeometryValue::SubgroupOrdinal => "shr.u32 %t31, %t13, 5;".into(),
            GeometryValue::SubgroupSize => "mov.u32 %t31, %warpsize;".into(),
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
    fn scalar_arg(&mut self, out: ErasedValue, index: u32, kind: seismic_ir::repr::ScalarKind) {
        let raw = self.word(self.layout.words.scalar_first + index);
        match kind {
            seismic_ir::repr::ScalarKind::Nat64 => {
                self.line(format!("mov.u64 {}, {raw};", self.v(out)))
            }
            seismic_ir::repr::ScalarKind::Scalar(DType::F32) => {
                let bits = self.t32();
                self.line(format!("cvt.u32.u64 {bits}, {raw};"));
                self.line(format!("mov.b32 {}, {bits};", self.v(out)));
            }
            seismic_ir::repr::ScalarKind::Scalar(DType::F16 | DType::BF16) => {
                let bits = self.t32();
                self.line(format!("cvt.u32.u64 {bits}, {raw};"));
                self.line(format!("and.b32 {}, {bits}, 65535;", self.v(out)));
            }
            seismic_ir::repr::ScalarKind::Scalar(DType::I32 | DType::U32 | DType::Bool) => {
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

    fn readable_place(&self, place: PlaceRef) -> ClosedReadablePlace {
        self.kernel.closed_readable_place(place, self.layout)
    }

    fn index_binary(&mut self, op: &str, left: &str, right: &str) -> String {
        let out = self.t64();
        self.line(format!("{op}.u64 {out}, {left}, {right};"));
        out
    }

    fn logical_coords(&mut self, tensor: &LogicalTensorMap, coordinates: &[String]) -> Vec<String> {
        let (coordinates, projection) = self.logical_storage_coords(tensor, coordinates);
        assert!(
            projection.is_none(),
            "packed numerical operand cannot be a dense plane projection"
        );
        coordinates
    }

    fn logical_storage_coords(
        &mut self,
        tensor: &LogicalTensorMap,
        coordinates: &[String],
    ) -> (Vec<String>, Option<(u32, String)>) {
        let mut coordinates = coordinates.to_vec();
        let mut projection = None;
        for step in tensor.steps.iter().rev() {
            coordinates = match step {
                LogicalViewStep::Plane { plane, axis } => {
                    let backing = self.readable_place(tensor.base);
                    let ReadableRepresentationGeometry::Packed(geometry) = backing.geometry else {
                        unreachable!("closed plane backing")
                    };
                    let elements = geometry.layout.planes[*plane as usize]
                        .storage_elements_per_packet()
                        .to_string();
                    let coordinate = &coordinates[*axis as usize];
                    let element = self.index_binary("rem", coordinate, &elements);
                    let packet = self.index_binary("div", coordinate, &elements);
                    coordinates[*axis as usize] =
                        self.index_binary("mul.lo", &packet, &geometry.layout.group.to_string());
                    projection = Some((*plane, element));
                    coordinates
                }
                LogicalViewStep::Slice(axes) => {
                    let mut logical = coordinates.iter();
                    axes.iter()
                        .map(|axis| match axis {
                            LogicalSliceAxis::Point(value) => self.v(*value),
                            LogicalSliceAxis::Range { start, .. } => {
                                let start = self.v(*start);
                                self.index_binary(
                                    "add",
                                    &start,
                                    logical.next().expect("closed slice-map rank"),
                                )
                            }
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
                        linear = self.index_binary("mul.lo", &linear, &self.v(*extent));
                        linear = self.index_binary("add", &linear, coordinate);
                    }
                    let mut physical = vec![String::new(); from.len()];
                    for axis in (0..from.len()).rev() {
                        physical[axis] = self.index_binary("rem", &linear, &self.v(from[axis]));
                        linear = self.index_binary("div", &linear, &self.v(from[axis]));
                    }
                    physical
                }
            };
        }
        (coordinates, projection)
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
        place: &ClosedDensePlace,
        index: &[ErasedValue],
    ) {
        let destination = self.v(out);
        let (address, geometry, _) = self.address(place, index);
        self.load_to(&destination, out_type, &address, geometry.dtype);
    }

    fn vector_read(
        &mut self,
        out: ClosedValue,
        place: &ClosedDensePlace,
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
            let (address, geometry, _) = self.address_names(place, &coordinates);
            self.load_to(
                &destination,
                ValueType::Scalar(dtype),
                &address,
                geometry.dtype,
            );
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
                        DecodeStep::ReadPlaneField { into, plane, field } => self
                            .numeric_plane_field(
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
                let ValueType::Scalar(dtype) = out_type else {
                    unreachable!("decoder result is scalar")
                };
                self.publish_numeric(dtype, destination, &temps[recipe.ordinal(recipe.output())]);
            }
        }
    }
    fn read_plane(
        &mut self,
        out: ErasedValue,
        out_type: ValueType,
        place: &ClosedPackedPlace,
        plane_info: &PlaneInfo,
        element: ErasedValue,
        index: &[ErasedValue],
    ) {
        let (address, _, _) = self.address(place, index);
        let value = self.plane_storage(&address, plane_info, &self.v(element));
        self.line(format!("mov{} {}, {value};", suffix(out_type), self.v(out)));
    }
    fn plane_storage(&mut self, packet: &str, plane: &PlaneInfo, element: &str) -> String {
        let base = self.t64();
        self.line(format!("add.u64 {base}, {packet}, {};", plane.offset));
        let offset = self.t64();
        self.line(format!(
            "mul.lo.u64 {offset}, {element}, {};",
            plane.storage_element_bytes()
        ));
        if let PlaneEncoding::Dense(dtype) = plane.encoding {
            let address = self.t64();
            self.line(format!("add.u64 {address}, {base}, {offset};"));
            let out = if dtype == DType::F32 {
                self.f32()
            } else {
                self.t32()
            };
            self.load_to(&out, ValueType::Scalar(dtype), &address, dtype);
            return out;
        }
        let result = self.t32();
        self.line(format!("mov.u32 {result}, 0;"));
        for byte in 0..plane.storage_element_bytes() {
            let position = self.t64();
            self.line(format!("add.u64 {position}, {offset}, {byte};"));
            let within = self.pred();
            self.line(format!(
                "setp.lt.u64 {within}, {position}, {};",
                plane.bytes_per_group
            ));
            let address = self.t64();
            self.line(format!("add.u64 {address}, {base}, {position};"));
            let raw = self.t32();
            self.line(format!("mov.u32 {raw}, 0;"));
            self.line(format!("@{within} ld.global.u8 {raw}, [{address}];"));
            self.line(format!("shl.b32 {raw}, {raw}, {};", byte * 8));
            self.line(format!("or.b32 {result}, {result}, {raw};"));
        }
        result
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

    fn numeric_plane_field(
        &mut self,
        packet: &str,
        logical: &str,
        packet_group: u32,
        plane: &PlaneInfo,
        field: u32,
        dtype: DType,
    ) -> String {
        let value = self.plane_field(packet, logical, packet_group, plane, field, dtype);
        self.numeric_operand(dtype, &value)
    }
    fn plane_field(
        &mut self,
        packet: &str,
        logical: &str,
        packet_group: u32,
        plane: &PlaneInfo,
        field: u32,
        _dtype: DType,
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
                if matches!(storage, DType::F16 | DType::BF16) {
                    let value = self.t32();
                    self.load_to(&value, ValueType::Scalar(*storage), &address, *storage);
                    value
                } else {
                    self.load_temp(&address, *storage)
                }
            }
            PlaneEncoding::Packed { bits, .. } => self.packed_field(&base, &entry, *bits),
            PlaneEncoding::FloatCode { format } => self.packed_field(&base, &entry, format.bits()),
        }
    }
    fn packed_field(&mut self, base: &str, entry: &str, bits: u32) -> String {
        let bit = self.t64();
        self.line(format!("mul.lo.u64 {bit}, {entry}, {bits};"));
        let byte = self.t64();
        self.line(format!("shr.u64 {byte}, {bit}, 3;"));
        let address = self.t64();
        self.line(format!("add.u64 {address}, {base}, {byte};"));
        let bit32 = self.t32();
        self.line(format!("cvt.u32.u64 {bit32}, {bit};"));
        let shift = self.t32();
        self.line(format!("and.b32 {shift}, {bit32}, 7;"));
        let count = self.t32();
        self.line(format!("add.u32 {count}, {shift}, {bits};"));
        let assembled = self.t64();
        self.line(format!("mov.u64 {assembled}, 0;"));
        for i in 0..(bits + 7).div_ceil(8) {
            let raw = self.t32();
            let enabled = self.pred();
            self.line(format!("mov.u32 {raw}, 0;"));
            self.line(format!("setp.gt.u32 {enabled}, {count}, {};", i * 8));
            self.line(format!("@{enabled} ld.global.u8 {raw}, [{address}+{i}];"));
            let wide = self.t64();
            self.line(format!("cvt.u64.u32 {wide}, {raw};"));
            self.line(format!("shl.b64 {wide}, {wide}, {};", i * 8));
            self.line(format!("or.b64 {assembled}, {assembled}, {wide};"));
        }
        self.line(format!("shr.u64 {assembled}, {assembled}, {shift};"));
        let out = self.t32();
        self.line(format!("cvt.u32.u64 {out}, {assembled};"));
        self.line(format!(
            "and.b32 {out}, {out}, {};",
            u32::MAX >> (32 - bits)
        ));
        out
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
            DType::F16 | DType::BF16 => unreachable!("narrow storage uses raw payload loads"),
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
    fn load_to(&mut self, destination: &str, out_type: ValueType, address: &str, dtype: DType) {
        if matches!(dtype, DType::F16 | DType::BF16) {
            let bits = self.h16();
            self.line(format!("ld.global.u16 {bits}, [{address}];"));
            self.line(format!("cvt.u32.u16 {destination}, {bits};"));
        } else {
            let temp = self.load_temp(address, dtype);
            self.line(format!("mov{} {destination}, {temp};", suffix(out_type)));
        }
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
                self.line(format!("cvt.u16.u32 {bits}, {value};"));
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
    fn store_slot(&mut self, slot: u32, kind: seismic_ir::repr::ScalarKind, value: ClosedValue) {
        let p = self.pred();
        self.line(format!("setp.eq.u64 {p}, %linear_thread, 0;"));
        match kind {
            seismic_ir::repr::ScalarKind::Nat64 => self.line(format!(
                "@{p} st.global.u64 [%results+{}], {};",
                slot * 8,
                self.v(value.value)
            )),
            seismic_ir::repr::ScalarKind::Scalar(DType::F32) => self.line(format!(
                "@{p} st.global.f32 [%results+{}], {};",
                slot * 8,
                self.v(value.value)
            )),
            seismic_ir::repr::ScalarKind::Scalar(DType::F16 | DType::BF16) => {
                let bits = self.h16();
                self.line(format!("cvt.u16.u32 {bits}, {};", self.v(value.value)));
                self.line(format!(
                    "@{p} st.global.u16 [%results+{}], {bits};",
                    slot * 8
                ));
            }
            seismic_ir::repr::ScalarKind::Scalar(DType::I32 | DType::U32 | DType::Bool) => self
                .line(format!(
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
                    let destination = self.v(out.value);
                    let left = self.numeric_operand(*dtype, &destination);
                    let right = if *dtype == DType::F32 {
                        let value = self.f32();
                        self.line(format!("mov.b32 {value}, {shuffled};"));
                        value
                    } else {
                        self.numeric_operand(*dtype, &shuffled)
                    };
                    let physical_out = self.numeric_destination(*dtype, &destination);
                    self.line(format!("{instruction} {physical_out}, {left}, {right};"));
                    self.publish_numeric(*dtype, &destination, &physical_out);
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
        let a = self.readable_place(a.base);
        let b = self.readable_place(b.base);
        let destination = self.readable_place(destination.base);
        let c = c.map(|map| self.readable_place(map.base));
        let ar = a_map.extents.len();
        let br = b_map.extents.len();
        let dr = destination_map.extents.len();
        if ar != 2 || br != 2 || dr != 2 {
            panic!("matrix intrinsic places are not rank two")
        };
        if !matches!(elem, DType::F16 | DType::BF16) {
            panic!("CUDA matrix intrinsic entered PTX emission with an unimplemented element type")
        }
        for map in [a_map, b_map] {
            if seismic_lang::registry::representation_info(map.representation).decoded != elem {
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
        let a = self.readable_place(a.base);
        let b = self.readable_place(b.base);
        let destination = self.readable_place(destination.base);
        let c = c.map(|map| self.readable_place(map.base));
        let ReadableRepresentationGeometry::Packed(_) = &b.geometry else {
            panic!("registry-selected CUDA packed matrix row has a dense right operand")
        };
        if seismic_lang::registry::representation_info(a_map.representation).decoded != DType::F32
            || seismic_lang::registry::representation_info(destination_map.representation).decoded
                != DType::F32
        {
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
        let right_coordinates = self.logical_coords(b_map, &[k.clone(), column.clone()]);
        self.read_to_names(
            &right,
            ValueType::Scalar(DType::F32),
            &b,
            &right_coordinates,
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
        let destination = self.readable_place(destination_map.base);
        let c = c_map.map(|map| self.readable_place(map.base));
        if !matches!(a.geometry, ReadableRepresentationGeometry::Packed(_))
            || !matches!(b.geometry, ReadableRepresentationGeometry::Packed(_))
            || seismic_lang::registry::representation_info(destination_map.representation).decoded
                != DType::F32
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
        place: &ClosedReadablePlace,
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
        place: &ClosedReadablePlace,
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
        place: &ClosedReadablePlace,
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
        place: &ClosedReadablePlace,
        row: &str,
        column: &str,
    ) -> (DType, String) {
        let (coordinates, projection) =
            self.logical_storage_coords(map, &[row.to_owned(), column.to_owned()]);
        let (address, geometry, _) = self.address_names(place, &coordinates);
        match (geometry, projection) {
            (ReadableRepresentationGeometry::Dense(geometry), None) => (geometry.dtype, address),
            (ReadableRepresentationGeometry::Packed(geometry), Some((plane, element))) => {
                let plane = &geometry.layout.planes[plane as usize];
                let offset = self.index_binary(
                    "mul.lo",
                    &element,
                    &plane.storage_element_bytes().to_string(),
                );
                let offset = self.index_binary("add", &offset, &plane.offset.to_string());
                let address = self.index_binary("add", &address, &offset);
                (plane.storage_dtype, address)
            }
            _ => panic!("dense intrinsic operand lacks its physical storage projection"),
        }
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
        self.copy_carries(params, &yielded);
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

    fn copy_carries(&mut self, destinations: &[ClosedValue], sources: &[ErasedValue]) {
        // Snapshot every source before modifying a carry register. In particular,
        // a yielded permutation must not observe an earlier destination write.
        let mut snapshots = Vec::new();
        for (destination, source) in destinations.iter().zip(sources) {
            match destination.ty {
                ValueType::Vector { lanes, .. } => {
                    for lane in 0..lanes {
                        let snapshot = self.t32();
                        self.line(format!(
                            "mov.b32 {snapshot}, {};",
                            self.vector_lane(*source, lane)
                        ));
                        snapshots.push((
                            ".b32",
                            self.vector_lane(destination.value, lane),
                            snapshot,
                        ));
                    }
                }
                ty => {
                    let (suffix, snapshot) = if ty == ValueType::Index {
                        (".b64", self.t64())
                    } else {
                        (".b32", self.t32())
                    };
                    self.line(format!("mov{suffix} {snapshot}, {};", self.v(*source)));
                    snapshots.push((suffix, self.v(destination.value), snapshot));
                }
            }
        }
        for (suffix, destination, snapshot) in snapshots {
            self.line(format!("mov{suffix} {destination}, {snapshot};"));
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
    /// Decode a semantic payload only at a physical arithmetic boundary.
    fn numeric_operand(&mut self, dtype: DType, value: &str) -> String {
        if matches!(dtype, DType::F16 | DType::BF16) {
            let bits = self.h16();
            let result = self.f32();
            self.line(format!("cvt.u16.u32 {bits}, {value};"));
            self.line(format!("cvt.f32{} {result}, {bits};", dtype_suffix(dtype)));
            result
        } else {
            value.to_owned()
        }
    }
    fn numeric_destination(&mut self, dtype: DType, out: &str) -> String {
        if matches!(dtype, DType::F16 | DType::BF16) {
            self.f32()
        } else {
            out.to_owned()
        }
    }
    fn publish_numeric(&mut self, dtype: DType, out: &str, value: &str) {
        if matches!(dtype, DType::F16 | DType::BF16) {
            let bits = self.h16();
            self.line(format!(
                "cvt.rn{}.f32 {bits}, {value};",
                dtype_suffix(dtype)
            ));
            self.line(format!("cvt.u32.u16 {out}, {bits};"));
        } else if out != value {
            self.line(format!("mov{} {out}, {value};", scalar_suffix(dtype)));
        }
    }
}

fn ptx_type(ty: ValueType) -> &'static str {
    match ty {
        ValueType::Scalar(DType::F32) => ".f32",
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
    if dtype == DType::F32 {
        ".f32"
    } else if dtype == DType::I32 {
        ".s32"
    } else {
        ".u32"
    }
}
fn scalar_suffix(dtype: DType) -> &'static str {
    if dtype == DType::F32 {
        ".f32"
    } else if dtype == DType::I32 {
        ".s32"
    } else {
        ".u32"
    }
}
fn scalar_zero(dtype: DType) -> &'static str {
    if dtype == DType::F32 {
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
        ValueType::Scalar(DType::F32) => ".f32",
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

fn op_values<B: seismic_ir::target::PhysicalDialect>(op: &Op<B>) -> Vec<ErasedValue> {
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
        | Op::ScalarBits { out, a }
        | Op::ScalarFromBits { out, a }
        | Op::Not { out, a } => vec![*out, *a],
        Op::Fma { out, a, b, c } => vec![*out, *a, *b, *c],
        Op::VectorFromLanes { out, lanes } => {
            let mut values = vec![*out];
            values.extend(lanes);
            values
        }
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
        Op::ReadPlane {
            out,
            index,
            element,
            ..
        } => {
            let mut values = vec![*out, *element];
            values.extend(index);
            values
        }
        Op::Read { out, index, .. } | Op::ReadPlaneField { out, index, .. } => {
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
    use super::*;

    #[test]
    #[should_panic(expected = "bitcast requires equal-width numeric scalar payloads")]
    fn vector_reinterpretation_is_not_a_scalar_bitcast() {
        use seismic_ir::{construction::Construction, target::VectorSupport};

        let facts = facts();
        let mut arena = seismic_lang::expr::ExprArena::new();
        let mut construction = Construction::<Cuda>::new(&mut arena, vec![], false, 0);
        let vectors = VectorSupport::default();
        let mut builder = construction.portable_kernel(&mut arena, &facts, &[], &vectors);
        let word = builder.constant(
            ConstantValue::U32(0x7c01_8000),
            ValueType::Scalar(DType::U32),
        );
        builder.bitcast(
            word,
            ValueType::Vector {
                dtype: DType::F16,
                lanes: 2,
            },
        );
    }

    #[test]
    fn repeat_carry_permutations_read_the_old_state() {
        use seismic_ir::{
            construction::Construction,
            target::{KernelWordLayout, VectorSupport},
        };
        use seismic_lang::registry::IntrinsicUniformity;

        for count in [2, 3, 40] {
            for (constant, ty) in [
                (ConstantValue::U32(1), ValueType::Scalar(DType::U32)),
                (ConstantValue::F32(-0.0), ValueType::Scalar(DType::F32)),
                (ConstantValue::F16(0x7c01), ValueType::Scalar(DType::F16)),
                (ConstantValue::Index(1), ValueType::Index),
            ] {
                let facts = facts();
                let mut arena = seismic_lang::expr::ExprArena::new();
                let mut construction = Construction::<Cuda>::new(&mut arena, vec![], false, 0);
                let vectors = VectorSupport::default();
                let mut builder = construction.portable_kernel(&mut arena, &facts, &[], &vectors);
                let start = builder.index_constant(0);
                let end = builder.index_constant(1);
                let initial = (0..count).map(|_| builder.constant(constant, ty)).collect();
                builder.repeat(
                    start,
                    end,
                    initial,
                    &vec![IntrinsicUniformity::Workgroup; count],
                    |_, _, mut values| {
                        values.rotate_left(1);
                        values
                    },
                );
                builder.close();
                let kernel = &construction.kernels()[0];
                let layout = KernelEmissionLayout {
                    words: KernelWordLayout::for_kernel(kernel),
                    bindings: vec![],
                    locals: vec![],
                    addressable_resources: vec![],
                    scalar_args: vec![],
                    result_types: vec![],
                };
                let mut emitter = Emitter::new(&facts, kernel, &layout);
                emitter.collect(kernel.root());
                emitter.header("carry_probe");
                emitter.emit_block(kernel.root());
                let op = kernel.block(kernel.root()).ops.last().unwrap();
                let ClosedOpView::Repeat {
                    carry_parameters, ..
                } = kernel.closed_op(op, &layout)
                else {
                    panic!("expected repeat")
                };
                assert!(emitter.operation_temp_bound(op) >= count as u64 + 1);
                assert!(u64::from(emitter.temp) <= emitter.temp_bound);
                let registers = carry_parameters
                    .iter()
                    .map(|v| emitter.v(v.value))
                    .collect::<Vec<_>>();
                let mut state = registers
                    .iter()
                    .enumerate()
                    .map(|(i, r)| (r.clone(), i as u64 + 1))
                    .collect::<HashMap<_, _>>();
                // Interpret only the loop's data moves, with distinct old payloads.
                // This fails for the original sequential backedge copies.
                let body = emitter
                    .text
                    .split("L_repeat_0:\n")
                    .nth(1)
                    .unwrap()
                    .split("add.u64")
                    .next()
                    .unwrap();
                for line in body
                    .lines()
                    .map(str::trim)
                    .filter(|line| line.starts_with("mov."))
                {
                    let operands = line.split_once(' ').unwrap().1.trim_end_matches(';');
                    let (destination, source) = operands.split_once(", ").unwrap();
                    let value = state[source];
                    state.insert(destination.to_string(), value);
                }
                for (i, register) in registers.iter().enumerate() {
                    assert_eq!(
                        state[register],
                        ((i + 1) % count + 1) as u64,
                        "{ty:?}: {body}"
                    );
                }
            }
        }
    }

    fn facts() -> CudaFacts {
        use crate::profile::{ComputeCapability, DriverApiVersion, PtxTarget, TensorMemory};
        CudaFacts {
            device_ordinal: 0,
            compute_capability: ComputeCapability::SM80,
            driver_api: DriverApiVersion::PTX_71_MINIMUM,
            ptx: PtxTarget::BASELINE,
            tensor_memory: TensorMemory::Unavailable,
            warp_size: 32,
            multiprocessors: 1,
            max_threads_per_multiprocessor: 2048,
            max_blocks_per_multiprocessor: 32,
            registers_per_block: 65536,
            registers_per_multiprocessor: 65536,
            max_registers_per_thread: 255,
            codegen_registers_per_thread: 64,
            shared_bytes_per_multiprocessor: 65536,
            shared_bytes_per_block: 49152,
            l2_cache_bytes: 1024,
            global_memory_bytes: 1048576,
            cooperative_launch: true,
            kernel_parameter_bytes: 4096,
        }
    }
    #[test]
    fn intrinsic_view_coordinates_emit_ptx_integer_instructions() {
        let source = fixture(|emitter, values| {
            let extent = values[3].value;
            let map = LogicalTensorMap {
                base: PlaceRef::Local { index: 0 },
                representation: seismic_lang::registry::dense(DType::F32),
                extents: vec![extent, extent],
                steps: vec![
                    LogicalViewStep::Slice(vec![
                        LogicalSliceAxis::Range {
                            start: extent,
                            end: extent,
                        },
                        LogicalSliceAxis::Full,
                    ]),
                    LogicalViewStep::Reshape {
                        from: vec![extent, extent],
                        to: vec![extent, extent],
                    },
                ],
            };
            let coordinates = emitter.logical_coords(&map, &["7".into(), "11".into()]);
            assert!(coordinates.iter().all(|value| value.starts_with("%d")));
        });
        assert!(source.contains("mul.lo.u64"));
        assert!(source.contains("add.u64"));
        assert!(source.contains("rem.u64"));
        assert!(source.contains("div.u64"));
        assert!(
            !source.contains("(("),
            "PTX operands cannot contain C-style address expressions"
        );
    }

    fn fixture(run: impl FnOnce(&mut Emitter<'_>, &[ClosedValue])) -> String {
        use seismic_ir::{
            construction::Construction,
            target::{KernelWordLayout, VectorSupport},
        };
        let facts = facts();
        let mut arena = seismic_lang::expr::ExprArena::new();
        let mut construction = Construction::<Cuda>::new(&mut arena, vec![], false, 0);
        let vectors = VectorSupport::default();
        let mut builder = construction.portable_kernel(&mut arena, &facts, &[], &vectors);
        let half = builder.constant(ConstantValue::F16(0xfc01), ValueType::Scalar(DType::F16));
        let bf = builder.constant(ConstantValue::BF16(0xff81), ValueType::Scalar(DType::BF16));
        builder.constant(
            ConstantValue::F32(f32::from_bits(0xff800001)),
            ValueType::Scalar(DType::F32),
        );
        builder.constant(ConstantValue::Index((1u64 << 54) + 1), ValueType::Index);
        let bits = builder.scalar_bits(half.clone());
        let restored = builder.scalar_from_bits(bits, DType::F16);
        let condition = builder.constant(ConstantValue::Bool(true), ValueType::Bool);
        builder.select(condition, half, restored);
        builder.scalar_bits(bf);
        builder.close();
        let kernel = &construction.kernels()[0];
        let layout = KernelEmissionLayout {
            words: KernelWordLayout::for_kernel(kernel),
            bindings: vec![],
            locals: vec![],
            addressable_resources: vec![],
            scalar_args: vec![],
            result_types: vec![],
        };
        let mut emitter = Emitter::new(&facts, kernel, &layout);
        emitter.collect(kernel.root());
        emitter.header("payload_probe");
        emitter.emit_block(kernel.root());
        let values = kernel
            .block(kernel.root())
            .ops
            .iter()
            .filter_map(|op| match kernel.closed_op(op, &layout) {
                ClosedOpView::Constant { out, .. } => Some(out),
                _ => None,
            })
            .collect::<Vec<_>>();
        run(&mut emitter, &values);
        emitter.text
    }
    #[test]
    fn source_float_cast_recipe_emits_word_operations_not_native_conversions() {
        use seismic_ir::{construction::Construction, target::{KernelWordLayout, VectorSupport}};
        let facts = facts();
        for (from, to) in [(DType::F16,DType::F32),(DType::BF16,DType::F32),(DType::F32,DType::F16),(DType::F32,DType::BF16)] {
            let mut arena = seismic_lang::expr::ExprArena::new();
            let mut construction = Construction::<Cuda>::new(&mut arena, vec![], false, 0);
            let vectors = VectorSupport::default();
            let mut builder = construction.portable_kernel(&mut arena, &facts, &[], &vectors);
            let constant = match from {
                DType::F16 => ConstantValue::F16(0xfc01),
                DType::BF16 => ConstantValue::BF16(0xff81),
                DType::F32 => ConstantValue::F32(f32::from_bits(0xff800001)),
                _ => unreachable!(),
            };
            let input = builder.constant(constant,ValueType::Scalar(from));
            builder.cast(input,ValueType::Scalar(to));
            builder.close();
            let kernel = &construction.kernels()[0];
            let layout = KernelEmissionLayout {
                words: KernelWordLayout::for_kernel(kernel), bindings: vec![], locals: vec![],
                addressable_resources: vec![], scalar_args: vec![], result_types: vec![],
            };
            assert!(kernel.block(kernel.root()).ops.iter().all(|op| !matches!(kernel.closed_op(op,&layout),ClosedOpView::Cast { .. })), "source scalar conversion must expand its recipe");
            let mut emitter = Emitter::new(&facts,kernel,&layout);
            emitter.collect(kernel.root());
            emitter.header("source_cast");
            emitter.emit_block(kernel.root());
            assert!(emitter.text.contains("and.b32"));
            for forbidden in ["cvt.f32.f16","cvt.f32.bf16","cvt.rn.f16.f32","cvt.rn.bf16.f32"] {
                assert!(!emitter.text.contains(forbidden), "{from:?}->{to:?}: {forbidden}");
            }
        }
    }

    #[test]
    fn selected_launch_descriptor_survives_closure_and_normalization() {
        use seismic_ir::{
            construction::{AllocationPlan, Construction},
            schedule::{Launch, LaunchParticipation},
            target::{LocalRealization, LocalRealizationPolicy, PhysicalDialect, VectorSupport},
        };
        let mut facts = facts();
        facts.cooperative_launch = false;
        assert_eq!(
            Cuda::launch_for_participation(&facts, LaunchParticipation::CooperativeGrid),
            None
        );
        assert_eq!(Cuda::ordinary_launch(), crate::CudaLaunchMode::Independent);
        facts.cooperative_launch = true;
        let cooperative =
            Cuda::launch_for_participation(&facts, LaunchParticipation::CooperativeGrid).unwrap();
        let mut arena = seismic_lang::expr::ExprArena::new();
        let mut construction = Construction::<Cuda>::new(&mut arena, vec![], false, 0);
        let vectors = VectorSupport::default();
        let kernel = construction
            .portable_kernel(&mut arena, &facts, &[], &vectors)
            .close();
        let one = arena.nat(1);
        let empty = arena.bool(false);
        let mut schedule = construction.schedule(&mut arena, 0);
        for descriptor in [Cuda::ordinary_launch(), cooperative] {
            let id = schedule.launch(Launch {
                kernel,
                descriptor,
                grid: [one; 3],
                workgroup: [one; 3],
                empty,
                parallel_extent: None,
                logical_base: None,
            });
            schedule.step_launch(id);
        }
        let schedule = schedule.close();
        let executable = construction
            .close(schedule)
            .normalize_launches(&mut arena, 65535, 64)
            .unwrap()
            .analyze_allocations()
            .apply_allocation_plan(&mut arena, AllocationPlan::distinct())
            .finish()
            .close_execution(
                &mut arena,
                LocalRealizationPolicy {
                    workgroup: LocalRealization::NativeDynamic,
                    participant: LocalRealization::NativeStatic,
                    register: LocalRealization::NativeStatic,
                },
                &crate::profile::CudaKernelAbi,
            );
        let launches = executable.schedule().launches();
        assert_eq!(launches[0].descriptor, crate::CudaLaunchMode::Independent);
        assert_eq!(
            launches[1].descriptor,
            crate::CudaLaunchMode::CooperativeGrid
        );
        assert_eq!(launches[0].kernel, launches[1].kernel);
        let mut ordinary = seismic_ir::identity::StructureDigest::new("launch-test");
        ordinary.hashed(&launches[0].descriptor);
        let mut cooperative = seismic_ir::identity::StructureDigest::new("launch-test");
        cooperative.hashed(&launches[1].descriptor);
        assert_ne!(ordinary.finish(), cooperative.finish());
    }

    #[test]
    fn narrow_payload_constants_select_and_bits_never_convert() {
        let text = fixture(|_, _| {});
        assert!(text.contains("mov.u32 %v0, 64513;"));
        assert!(text.contains("mov.u32 %v1, 65409;"));
        assert!(text.contains("0fff800001"));
        assert!(text.contains("18014398509481985"));
        assert!(text.contains("selp.u32"));
        assert!(text.contains("and.b32"));
        assert!(!text.contains("cvt.f32.f16"));
        assert!(!text.contains("cvt.f32.bf16"));
        assert!(!text.contains("cvt.rn.f16.f32"));
    }
    #[test]
    fn narrow_load_abi_copy_and_publication_are_integer_transport() {
        let text = fixture(|emitter, values| {
            for (value, dtype) in values.iter().zip([DType::F16, DType::BF16]) {
                let name = emitter.v(value.value);
                emitter.load_to(&name, value.ty, "%address", dtype);
                emitter.scalar_arg(value.value, 0, seismic_ir::repr::ScalarKind::Scalar(dtype));
                emitter.copy_value(*value, value.value);
                let geometry =
                    seismic_ir::target::RepresentationGeometry::of(seismic_lang::registry::dense(dtype))
                        .dense();
                emitter.write_address("%address", &geometry, &name);
                emitter.store_slot(0, seismic_ir::repr::ScalarKind::Scalar(dtype), *value);
            }
        });
        assert!(text.contains("ld.global.u16"));
        assert!(text.contains("cvt.u32.u16"));
        assert!(text.contains("cvt.u16.u32"));
        assert!(text.contains("st.global.u16"));
        assert!(!text.contains("cvt.f32.f16"));
        assert!(!text.contains("cvt.f32.bf16"));
        assert!(!text.contains("cvt.rn.f16.f32"));
        assert!(!text.contains("cvt.rn.bf16.f32"));
    }
    #[test]
    fn native_arithmetic_decodes_only_at_the_operation_boundary() {
        let text = fixture(|emitter, values| {
            let half = values[0];
            emitter.binary(BinaryOp::Add, half.value, half.ty, half.value, half.value);
        });
        assert_eq!(text.matches("cvt.f32.f16").count(), 2);
        assert_eq!(text.matches("cvt.rn.f16.f32").count(), 1);
        assert!(text.contains("add.rn.f32 %f"));
        assert!(text.contains("cvt.u32.u16 %v0"));
    }

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
