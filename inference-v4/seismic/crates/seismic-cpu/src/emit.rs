//! Exhaustive Cranelift emission of the closed typed kernel IR.

use crate::Cpu;
use cranelift_codegen::ir::{
    self,
    condcodes::{FloatCC, IntCC},
    types, AbiParam, BlockArg, InstBuilder, MemFlags, Value,
};
use cranelift_codegen::isa::CallConv;
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift_jit::JITModule;
use cranelift_module::Module;
use seismic_compiler::errors::NativeCompilationError;
use seismic_compiler::kernel::ops::{
    BarrierScope, BinaryOp, BitOp, Block, ClosedDensePlace, ClosedExternalGlobalPlace,
    ClosedOpView, ClosedPackedGlobalPlace, ClosedPackedPlace, ClosedPlace, ClosedPlaceKind,
    ClosedPlaceWords, ClosedReadablePlace, ClosedValue, CmpOp, ConstantValue, ErasedValue,
    GeometryValue, LogicOp, Op, StoreElection, UnaryOp, ValueType,
};
use seismic_compiler::kernel::{BlockId, Kernel};
use seismic_compiler::storage::LaunchLocalKind;
use seismic_compiler::target::{
    DenseRepresentationGeometry, KernelEmissionLayout, PackedRepresentationGeometry,
    ReadableRepresentationGeometry,
};
use seismic_lang::intrinsics::MathOp;
use seismic_lang::registry::{
    CodeInterpretation, DecodeStep, FloatCodeFormat, PlaneEncoding, PlaneInfo, PlaneRepackRecipe,
    RepackExpr, RepresentationKind,
};
use seismic_lang::types::DType;
use std::collections::{BTreeSet, HashMap};

pub(crate) struct Import {
    pub name: &'static str,
    pub signature: ir::Signature,
    pub reference: ir::FuncRef,
}

pub(crate) struct EmittedKernel {
    pub function: ir::Function,
    pub imports: Vec<Import>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum HostImport {
    ApproxExp,
    ApproxLog,
    ApproxSin,
    ApproxCos,
    Fmax,
    Fmin,
    Round(RoundImport),
    F16Load,
    F16Store,
    PackedBits,
    FloatCode(FloatCodeImport),
    Barrier,
    Atomic(AtomicImport),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum RoundImport {
    F32,
    F16,
    BF16,
    I32,
    U32,
    Bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum FloatCodeImport {
    E2M1,
    E4M3,
    UE4M3,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum AtomicImport {
    AddF32,
    MaxF32,
    MinF32,
    AddF16,
    MaxF16,
    MinF16,
    AddBF16,
    MaxBF16,
    MinBF16,
    AddI32,
    MaxI32,
    MinI32,
    AddU32,
    MaxU32,
    MinU32,
}

impl HostImport {
    fn name(self) -> &'static str {
        match self {
            Self::ApproxExp => "seismic_approx_exp_f32",
            Self::ApproxLog => "seismic_approx_log_f32",
            Self::ApproxSin => "seismic_approx_sin_f32",
            Self::ApproxCos => "seismic_approx_cos_f32",
            Self::Fmax => "seismic_fmax",
            Self::Fmin => "seismic_fmin",
            Self::Round(kind) => match kind {
                RoundImport::F32 => "seismic_round_f32",
                RoundImport::F16 => "seismic_round_f16",
                RoundImport::BF16 => "seismic_round_bf16",
                RoundImport::I32 => "seismic_round_i32",
                RoundImport::U32 => "seismic_round_u32",
                RoundImport::Bool => "seismic_round_bool",
            },
            Self::F16Load => "seismic_f16_load",
            Self::F16Store => "seismic_f16_store",
            Self::PackedBits => "seismic_packed_bits",
            Self::FloatCode(kind) => match kind {
                FloatCodeImport::E2M1 => "seismic_float_code_e2m1",
                FloatCodeImport::E4M3 => "seismic_float_code_e4m3",
                FloatCodeImport::UE4M3 => "seismic_float_code_ue4m3",
            },
            Self::Barrier => "seismic_cpu_barrier",
            Self::Atomic(kind) => match kind {
                AtomicImport::AddF32 => "seismic_atomic_add_f32",
                AtomicImport::MaxF32 => "seismic_atomic_max_f32",
                AtomicImport::MinF32 => "seismic_atomic_min_f32",
                AtomicImport::AddF16 => "seismic_atomic_add_f16",
                AtomicImport::MaxF16 => "seismic_atomic_max_f16",
                AtomicImport::MinF16 => "seismic_atomic_min_f16",
                AtomicImport::AddBF16 => "seismic_atomic_add_bf16",
                AtomicImport::MaxBF16 => "seismic_atomic_max_bf16",
                AtomicImport::MinBF16 => "seismic_atomic_min_bf16",
                AtomicImport::AddI32 => "seismic_atomic_add_i32",
                AtomicImport::MaxI32 => "seismic_atomic_max_i32",
                AtomicImport::MinI32 => "seismic_atomic_min_i32",
                AtomicImport::AddU32 => "seismic_atomic_add_u32",
                AtomicImport::MaxU32 => "seismic_atomic_max_u32",
                AtomicImport::MinU32 => "seismic_atomic_min_u32",
            },
        }
    }

    fn signature(self, call_conv: CallConv) -> ir::Signature {
        let mut signature = ir::Signature::new(call_conv);
        match self {
            Self::ApproxExp | Self::ApproxLog | Self::ApproxSin | Self::ApproxCos => {
                signature.params.push(AbiParam::new(types::F32));
                signature.returns.push(AbiParam::new(types::F32));
            }
            Self::Fmax | Self::Fmin => {
                signature
                    .params
                    .extend([AbiParam::new(types::F64), AbiParam::new(types::F64)]);
                signature.returns.push(AbiParam::new(types::F64));
            }
            Self::Round(_) => {
                signature.params.push(AbiParam::new(types::F64));
                signature.returns.push(AbiParam::new(types::F64));
            }
            Self::F16Load => {
                signature.params.push(AbiParam::new(types::I32));
                signature.returns.push(AbiParam::new(types::F32));
            }
            Self::F16Store => {
                signature.params.push(AbiParam::new(types::F32));
                signature.returns.push(AbiParam::new(types::I32));
            }
            Self::PackedBits => {
                signature.params.extend([
                    AbiParam::new(types::I64),
                    AbiParam::new(types::I64),
                    AbiParam::new(types::I32),
                ]);
                signature.returns.push(AbiParam::new(types::I32));
            }
            Self::FloatCode(_) => {
                signature.params.push(AbiParam::new(types::I32));
                signature.returns.push(AbiParam::new(types::F32));
            }
            Self::Barrier => {
                signature.params.push(AbiParam::new(types::I64));
                signature.returns.push(AbiParam::new(types::I32));
            }
            Self::Atomic(_) => {
                signature
                    .params
                    .extend([AbiParam::new(types::I64), AbiParam::new(types::I64)]);
            }
        }
        signature
    }
}

fn round_import(dtype: DType) -> HostImport {
    HostImport::Round(match dtype {
        DType::F32 => RoundImport::F32,
        DType::F16 => RoundImport::F16,
        DType::BF16 => RoundImport::BF16,
        DType::I32 => RoundImport::I32,
        DType::U32 => RoundImport::U32,
        DType::Bool => RoundImport::Bool,
    })
}

fn float_code_import(format: FloatCodeFormat) -> HostImport {
    HostImport::FloatCode(match format {
        FloatCodeFormat::E2M1 => FloatCodeImport::E2M1,
        FloatCodeFormat::E4M3 => FloatCodeImport::E4M3,
        FloatCodeFormat::UE4M3 => FloatCodeImport::UE4M3,
    })
}

fn atomic_import(dtype: DType, op: seismic_lang::intrinsics::AtomicOp) -> HostImport {
    use seismic_lang::intrinsics::AtomicOp;
    HostImport::Atomic(match (dtype, op) {
        (DType::F32, AtomicOp::Add) => AtomicImport::AddF32,
        (DType::F32, AtomicOp::Max) => AtomicImport::MaxF32,
        (DType::F32, AtomicOp::Min) => AtomicImport::MinF32,
        (DType::F16, AtomicOp::Add) => AtomicImport::AddF16,
        (DType::F16, AtomicOp::Max) => AtomicImport::MaxF16,
        (DType::F16, AtomicOp::Min) => AtomicImport::MinF16,
        (DType::BF16, AtomicOp::Add) => AtomicImport::AddBF16,
        (DType::BF16, AtomicOp::Max) => AtomicImport::MaxBF16,
        (DType::BF16, AtomicOp::Min) => AtomicImport::MinBF16,
        (DType::I32, AtomicOp::Add) => AtomicImport::AddI32,
        (DType::I32, AtomicOp::Max) => AtomicImport::MaxI32,
        (DType::I32, AtomicOp::Min) => AtomicImport::MinI32,
        (DType::U32, AtomicOp::Add) => AtomicImport::AddU32,
        (DType::U32, AtomicOp::Max) => AtomicImport::MaxU32,
        (DType::U32, AtomicOp::Min) => AtomicImport::MinU32,
        (DType::Bool, _) => panic!("typed CPU kernel admitted an atomic boolean"),
    })
}

pub(crate) fn emit(
    module: &mut JITModule,
    kernel: &Kernel<Cpu>,
    layout: &KernelEmissionLayout,
    call_conv: CallConv,
) -> Result<EmittedKernel, NativeCompilationError> {
    let mut signature = module.make_signature();
    signature.call_conv = call_conv;
    signature.params.extend([
        AbiParam::new(types::I64), // frame
        AbiParam::new(types::I64), // barrier
        AbiParam::new(types::I64), // linear workgroup
        AbiParam::new(types::I64), // linear local
        AbiParam::new(types::I64), // workgroup scratch
        AbiParam::new(types::I64), // participant scratch
        AbiParam::new(types::I64), // register scratch
    ]);
    let mut function = ir::Function::with_name_signature(
        ir::UserFuncName::testcase("seismic_cpu_kernel"),
        signature,
    );
    let mut context = FunctionBuilderContext::new();
    let mut builder = FunctionBuilder::new(&mut function, &mut context);
    let entry = builder.create_block();
    builder.append_block_params_for_function_params(entry);
    builder.switch_to_block(entry);
    builder.seal_block(entry);
    let params = builder.block_params(entry).to_vec();
    let buffers = builder
        .ins()
        .load(types::I64, MemFlags::trusted(), params[0], 0);
    let words = builder
        .ins()
        .load(types::I64, MemFlags::trusted(), params[0], 8);
    let results = builder
        .ins()
        .load(types::I64, MemFlags::trusted(), params[0], 16);
    let imports_used = collect_imports(kernel);
    let mut imports = HashMap::new();
    for import in imports_used {
        let signature = builder.import_signature(import.signature(call_conv));
        let reference = builder.import_function(ir::ExtFuncData {
            name: ir::ExternalName::testcase(import.name()),
            signature,
            colocated: false,
        });
        imports.insert(import, reference);
    }
    let mut emitter = Emitter {
        builder,
        kernel,
        layout,
        imports,
        values: HashMap::new(),
        buffers,
        words,
        results,
        barrier: params[1],
        linear_workgroup: params[2],
        linear_local: params[3],
        workgroup_scratch: params[4],
        participant_scratch: params[5],
        register_scratch: params[6],
    };
    let yielded = emitter.emit_block(kernel.root());
    if yielded.is_some() {
        panic!("root kernel block yielded values");
    }
    emitter.builder.ins().return_(&[]);
    let import_records = emitter
        .imports
        .iter()
        .map(|(kind, reference)| Import {
            name: kind.name(),
            signature: kind.signature(call_conv),
            reference: *reference,
        })
        .collect::<Vec<_>>();
    drop(emitter);
    Ok(EmittedKernel {
        function,
        imports: import_records,
    })
}

fn collect_imports(kernel: &Kernel<Cpu>) -> BTreeSet<HostImport> {
    let mut imports = BTreeSet::new();
    imports.extend([
        round_import(DType::F32),
        round_import(DType::F16),
        round_import(DType::BF16),
        round_import(DType::I32),
        round_import(DType::U32),
        round_import(DType::Bool),
    ]);
    imports.insert(HostImport::F16Load);
    imports.insert(HostImport::F16Store);
    imports.insert(HostImport::PackedBits);
    imports.extend([
        float_code_import(FloatCodeFormat::E2M1),
        float_code_import(FloatCodeFormat::E4M3),
        float_code_import(FloatCodeFormat::UE4M3),
    ]);
    for dtype in [DType::F32, DType::F16, DType::BF16, DType::I32, DType::U32] {
        for op in [
            seismic_lang::intrinsics::AtomicOp::Add,
            seismic_lang::intrinsics::AtomicOp::Max,
            seismic_lang::intrinsics::AtomicOp::Min,
        ] {
            imports.insert(atomic_import(dtype, op));
        }
    }
    collect_block_imports(kernel, kernel.root(), &mut imports);
    imports
}

fn collect_block_imports(kernel: &Kernel<Cpu>, block: BlockId, imports: &mut BTreeSet<HostImport>) {
    for op in &kernel.block(block).ops {
        match op {
            Op::Math { op, .. } => match op {
                MathOp::Exp | MathOp::ExpFast => {
                    imports.insert(HostImport::ApproxExp);
                }
                MathOp::Log => {
                    imports.insert(HostImport::ApproxLog);
                }
                MathOp::Sin => {
                    imports.insert(HostImport::ApproxSin);
                }
                MathOp::Cos => {
                    imports.insert(HostImport::ApproxCos);
                }
                MathOp::Max => {
                    imports.insert(HostImport::Fmax);
                }
                MathOp::Min => {
                    imports.insert(HostImport::Fmin);
                }
                MathOp::Fma | MathOp::Rsqrt | MathOp::Sqrt | MathOp::Abs => {}
            },
            Op::Binary {
                op: BinaryOp::Min, ..
            }
            | Op::VectorBinary {
                op: BinaryOp::Min, ..
            } => {
                imports.insert(HostImport::Fmin);
            }
            Op::Binary {
                op: BinaryOp::Max, ..
            }
            | Op::VectorBinary {
                op: BinaryOp::Max, ..
            } => {
                imports.insert(HostImport::Fmax);
            }
            Op::Barrier(BarrierScope::Workgroup) => {
                imports.insert(HostImport::Barrier);
            }
            Op::Atomic { .. } => {}
            Op::Branch {
                then, otherwise, ..
            } => {
                collect_block_imports(kernel, *then, imports);
                collect_block_imports(kernel, *otherwise, imports);
            }
            Op::Repeat { body, .. } => collect_block_imports(kernel, *body, imports),
            Op::Intrinsic { op, .. } => match *op {},
            _ => {}
        }
    }
}

struct Emitter<'a, 'b> {
    builder: FunctionBuilder<'a>,
    kernel: &'b Kernel<Cpu>,
    layout: &'b KernelEmissionLayout,
    imports: HashMap<HostImport, ir::FuncRef>,
    values: HashMap<ErasedValue, Value>,
    buffers: Value,
    words: Value,
    results: Value,
    barrier: Value,
    linear_workgroup: Value,
    linear_local: Value,
    workgroup_scratch: Value,
    participant_scratch: Value,
    register_scratch: Value,
}

impl Emitter<'_, '_> {
    fn emit_block(&mut self, block: BlockId) -> Option<Vec<Value>> {
        let Block { ops } = self.kernel.block(block);
        for op in ops {
            match self.kernel.closed_op(op, self.layout) {
                ClosedOpView::Constant { out, value } => {
                    let value = self.constant(value);
                    self.define(out.value, value);
                }
                ClosedOpView::Binary { op, out, a, b } => {
                    let value = self.binary(op, out.ty, self.value(a.value), self.value(b.value));
                    self.define(out.value, value);
                }
                ClosedOpView::Unary { op, out, a } => {
                    let a = self.value(a.value);
                    let value = match out.ty {
                        ValueType::Scalar(dtype) if dtype.is_float() => match op {
                            UnaryOp::Neg => self.builder.ins().fneg(a),
                            UnaryOp::Abs => self.builder.ins().fabs(a),
                        },
                        _ => match op {
                            UnaryOp::Neg => self.builder.ins().ineg(a),
                            UnaryOp::Abs => {
                                let ty = self.builder.func.dfg.value_type(a);
                                let zero = self.builder.ins().iconst(ty, 0);
                                let neg = self.builder.ins().ineg(a);
                                let negative =
                                    self.builder.ins().icmp(IntCC::SignedLessThan, a, zero);
                                self.builder.ins().select(negative, neg, a)
                            }
                        },
                    };
                    let value = self.round_value(value, out.ty);
                    self.define(out.value, value);
                }
                ClosedOpView::Bit { op, out, a, b } => {
                    let a = self.value(a.value);
                    let b = self.value(b.value);
                    let value = match op {
                        BitOp::And => self.builder.ins().band(a, b),
                        BitOp::Or => self.builder.ins().bor(a, b),
                        BitOp::Xor => self.builder.ins().bxor(a, b),
                        BitOp::Shl => self.builder.ins().ishl(a, b),
                        BitOp::Shr if out.ty == ValueType::Scalar(DType::I32) => {
                            self.builder.ins().sshr(a, b)
                        }
                        BitOp::Shr => self.builder.ins().ushr(a, b),
                    };
                    self.define(out.value, value);
                }
                ClosedOpView::Fma { out, a, b, c } => {
                    let a = self.value(a.value);
                    let b = self.value(b.value);
                    let c = self.value(c.value);
                    let value = self.builder.ins().fma(a, b, c);
                    let value = self.round_value(value, out.ty);
                    self.define(out.value, value);
                }
                ClosedOpView::VectorSplat { out, value } => {
                    let scalar = self.value(value.value);
                    let value = self.builder.ins().splat(native_type(out.ty), scalar);
                    self.define(out.value, value);
                }
                ClosedOpView::VectorBinary { op, out, a, b } => {
                    let value =
                        self.vector_binary(op, out.ty, self.value(a.value), self.value(b.value));
                    self.define(out.value, value);
                }
                ClosedOpView::VectorUnary { op, out, a } => {
                    let value = self.vector_unary(op, out.ty, self.value(a.value));
                    self.define(out.value, value);
                }
                ClosedOpView::VectorBit { op, out, a, b } => {
                    let a = self.value(a.value);
                    let b = self.value(b.value);
                    let value = match op {
                        BitOp::And => self.builder.ins().band(a, b),
                        BitOp::Or => self.builder.ins().bor(a, b),
                        BitOp::Xor => self.builder.ins().bxor(a, b),
                        BitOp::Shl => self.builder.ins().ishl(a, b),
                        BitOp::Shr if vector_dtype(out.ty) == DType::I32 => {
                            self.builder.ins().sshr(a, b)
                        }
                        BitOp::Shr => self.builder.ins().ushr(a, b),
                    };
                    self.define(out.value, value);
                }
                ClosedOpView::VectorFma { out, a, b, c } => {
                    let a = self.value(a.value);
                    let b = self.value(b.value);
                    let c = self.value(c.value);
                    let value = self.builder.ins().fma(a, b, c);
                    let value = self.round_vector(value, out.ty);
                    self.define(out.value, value);
                }
                ClosedOpView::VectorCast { out, a, to } => {
                    let value = self.vector_cast(self.value(a.value), a.ty, to);
                    self.define(out.value, value);
                }
                ClosedOpView::VectorLane { out, vector, lane } => {
                    let vector = self.value(vector.value);
                    let value = self.builder.ins().extractlane(vector, lane as u8);
                    self.define(out.value, value);
                }
                ClosedOpView::VectorReduceAdd { out, vector } => {
                    let (_, lanes) = vector_shape(vector.ty);
                    let source = self.value(vector.value);
                    let mut value = self.builder.ins().extractlane(source, 0);
                    for lane in 1..lanes {
                        let next = self.builder.ins().extractlane(source, lane as u8);
                        value = if vector_dtype(vector.ty).is_float() {
                            self.builder.ins().fadd(value, next)
                        } else {
                            self.builder.ins().iadd(value, next)
                        };
                        value = self.round_value(value, out.ty);
                    }
                    self.define(out.value, value);
                }
                ClosedOpView::ApproximateMath { op, out, a } => {
                    let value = self.math(op, self.value(a.value));
                    let value = self.round_value(value, out.ty);
                    self.define(out.value, value);
                }
                ClosedOpView::Cast { out, a, to } => {
                    let value = self.cast(self.value(a.value), a.ty, to);
                    self.define(out.value, value);
                }
                ClosedOpView::Bitcast { out, a, to } => {
                    let a = self.value(a.value);
                    let value = self
                        .builder
                        .ins()
                        .bitcast(native_type(to), MemFlags::new(), a);
                    self.define(out.value, value);
                }
                ClosedOpView::Cmp { op, out, a, b } => {
                    let value = self.compare(op, a.ty, self.value(a.value), self.value(b.value));
                    self.define(out.value(), value);
                }
                ClosedOpView::Select {
                    out,
                    condition,
                    a,
                    b,
                } => {
                    let condition = self.value(condition.value());
                    let condition = self.truth(condition);
                    let a = self.value(a.value);
                    let b = self.value(b.value);
                    let value = self.builder.ins().select(condition, a, b);
                    self.define(out.value, value);
                }
                ClosedOpView::Logic { op, out, a, b } => {
                    let a = self.value(a.value());
                    let b = self.value(b.value());
                    let value = match op {
                        LogicOp::And => self.builder.ins().band(a, b),
                        LogicOp::Or => self.builder.ins().bor(a, b),
                    };
                    self.define(out.value(), value);
                }
                ClosedOpView::Not { out, a } => {
                    let one = self.builder.ins().iconst(types::I32, 1);
                    let a = self.value(a.value());
                    let value = self.builder.ins().bxor(a, one);
                    self.define(out.value(), value);
                }
                ClosedOpView::Geometry { out, kind } => {
                    let value = self.geometry(kind);
                    self.define(out.value(), value);
                }
                ClosedOpView::NatArg { out, index, .. } => {
                    let value = self.word(self.layout.words.nat_first + index);
                    self.define(out.value(), value);
                }
                ClosedOpView::ScalarArg {
                    out, index, dtype, ..
                } => {
                    let raw = self.word(self.layout.words.scalar_first + index);
                    let value = self.decode_word(raw, dtype);
                    self.define(out.value, value);
                }
                ClosedOpView::Read {
                    out,
                    place,
                    indices,
                } => {
                    let coordinates = indices
                        .iter()
                        .map(|value| self.value(value.value()))
                        .collect::<Vec<_>>();
                    let value = self.read(&place, &coordinates);
                    self.define(out.value, value);
                }
                ClosedOpView::VectorRead {
                    out,
                    place,
                    indices,
                    axis,
                    active,
                } => {
                    let coordinates = indices
                        .iter()
                        .map(|value| self.value(value.value()))
                        .collect::<Vec<_>>();
                    let value = self.vector_read(
                        &place,
                        &coordinates,
                        axis,
                        self.value(active.value()),
                        out.ty,
                    );
                    self.define(out.value, value);
                }
                ClosedOpView::ReadPlane {
                    out,
                    place,
                    plane,
                    indices,
                    ..
                } => {
                    let coordinates = indices
                        .iter()
                        .map(|value| self.value(value.value()))
                        .collect::<Vec<_>>();
                    let value = self.read_plane(&place, plane as usize, &coordinates);
                    self.define(out.value, value);
                }
                ClosedOpView::RepresentationConvertPacket {
                    source,
                    destination,
                    recipe,
                    packet,
                    ..
                } => {
                    let packet = self.value(packet.value());
                    self.convert_packet(&source, &destination, &recipe.recipe, packet);
                }
                ClosedOpView::Write {
                    place,
                    indices,
                    value,
                } => {
                    let coordinates = indices
                        .iter()
                        .map(|value| self.value(value.value()))
                        .collect::<Vec<_>>();
                    self.write(&place, &coordinates, self.value(value.value));
                }
                ClosedOpView::VectorWrite {
                    place,
                    indices,
                    axis,
                    active,
                    value,
                } => {
                    let coordinates = indices
                        .iter()
                        .map(|index| self.value(index.value()))
                        .collect::<Vec<_>>();
                    let active = self.value(active.value());
                    let value_id = self.value(value.value);
                    self.vector_write(&place, &coordinates, axis, active, value_id, value.ty);
                }
                ClosedOpView::Extent { out, place, axis } => {
                    let value = self.extent(&place, axis);
                    self.define(out.value(), value);
                }
                ClosedOpView::Atomic {
                    op,
                    place,
                    indices,
                    value,
                } => {
                    let coordinates = indices
                        .iter()
                        .map(|value| self.value(value.value()))
                        .collect::<Vec<_>>();
                    let (address, geometry, _) = self.address(&place, &coordinates);
                    let raw = self.encode_word(self.value(value.value), geometry.dtype);
                    self.call(atomic_import(geometry.dtype, op), &[address, raw]);
                }
                ClosedOpView::StoreSlot {
                    slot,
                    dtype,
                    value,
                    election: StoreElection::GlobalLeader,
                } => {
                    let raw = self.encode_word(self.value(value.value), dtype);
                    let workgroup_leader =
                        self.builder
                            .ins()
                            .icmp_imm(IntCC::Equal, self.linear_workgroup, 0);
                    let participant_leader =
                        self.builder
                            .ins()
                            .icmp_imm(IntCC::Equal, self.linear_local, 0);
                    let elected = self
                        .builder
                        .ins()
                        .band(workgroup_leader, participant_leader);
                    let store = self.builder.create_block();
                    let continuation = self.builder.create_block();
                    self.builder
                        .ins()
                        .brif(elected, store, &[], continuation, &[]);
                    self.builder.switch_to_block(store);
                    self.builder.ins().store(
                        MemFlags::trusted(),
                        raw,
                        self.results,
                        (slot as i32) * 8,
                    );
                    self.builder.ins().jump(continuation, &[]);
                    self.builder.seal_block(store);
                    self.builder.switch_to_block(continuation);
                    self.builder.seal_block(continuation);
                }
                ClosedOpView::Barrier(BarrierScope::Workgroup) => {
                    let cancelled = self.call(HostImport::Barrier, &[self.barrier]);
                    let cancelled = self.builder.ins().icmp_imm(IntCC::NotEqual, cancelled, 0);
                    let leave = self.builder.create_block();
                    let continuation = self.builder.create_block();
                    self.builder
                        .ins()
                        .brif(cancelled, leave, &[], continuation, &[]);
                    self.builder.switch_to_block(leave);
                    self.builder.ins().return_(&[]);
                    self.builder.seal_block(leave);
                    self.builder.switch_to_block(continuation);
                    self.builder.seal_block(continuation);
                }
                ClosedOpView::Barrier(BarrierScope::Subgroup) => {}
                ClosedOpView::Intrinsic { op, .. } => match *op {},
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
                    return Some(values.iter().map(|value| self.value(value.value)).collect());
                }
            }
        }
        None
    }

    fn value(&self, value: ErasedValue) -> Value {
        self.values.get(&value).copied().unwrap_or_else(|| {
            panic!("typed kernel value {value:?} has no dominating native definition")
        })
    }

    fn define(&mut self, out: ErasedValue, value: Value) {
        if self.values.insert(out, value).is_some() {
            panic!("typed kernel value {out:?} has two native definitions");
        }
    }

    fn word(&mut self, index: u32) -> Value {
        self.builder.ins().load(
            types::I64,
            MemFlags::trusted(),
            self.words,
            (index as i32) * 8,
        )
    }

    fn constant(&mut self, value: ConstantValue) -> Value {
        match value {
            ConstantValue::F32(value) => self.builder.ins().f32const(value),
            ConstantValue::F16(value) => self
                .builder
                .ins()
                .f32const(seismic_lang::registry::f16_to_f32(value)),
            ConstantValue::BF16(value) => self
                .builder
                .ins()
                .f32const(f32::from_bits(u32::from(value) << 16)),
            ConstantValue::I32(value) => self.builder.ins().iconst(types::I32, i64::from(value)),
            ConstantValue::U32(value) => self.builder.ins().iconst(types::I32, i64::from(value)),
            ConstantValue::Bool(value) => self.builder.ins().iconst(types::I32, i64::from(value)),
            ConstantValue::Index(value) => self.builder.ins().iconst(types::I64, value as i64),
        }
    }

    fn binary(&mut self, op: BinaryOp, ty: ValueType, a: Value, b: Value) -> Value {
        let value = match ty {
            ValueType::Scalar(dtype) if dtype.is_float() => match op {
                BinaryOp::Add => self.builder.ins().fadd(a, b),
                BinaryOp::Sub => self.builder.ins().fsub(a, b),
                BinaryOp::Mul => self.builder.ins().fmul(a, b),
                BinaryOp::Div => self.builder.ins().fdiv(a, b),
                BinaryOp::Rem => panic!("floating remainder is not in typed kernel IR"),
                BinaryOp::Min => self.call_float2(HostImport::Fmin, a, b),
                BinaryOp::Max => self.call_float2(HostImport::Fmax, a, b),
            },
            ValueType::Scalar(DType::I32) => match op {
                BinaryOp::Add => self.builder.ins().iadd(a, b),
                BinaryOp::Sub => self.builder.ins().isub(a, b),
                BinaryOp::Mul => self.builder.ins().imul(a, b),
                BinaryOp::Div => self.signed_euclid(a, b).0,
                BinaryOp::Rem => self.signed_euclid(a, b).1,
                BinaryOp::Min => {
                    let c = self.builder.ins().icmp(IntCC::SignedLessThan, a, b);
                    self.builder.ins().select(c, a, b)
                }
                BinaryOp::Max => {
                    let c = self.builder.ins().icmp(IntCC::SignedGreaterThan, a, b);
                    self.builder.ins().select(c, a, b)
                }
            },
            _ => match op {
                BinaryOp::Add => self.builder.ins().iadd(a, b),
                BinaryOp::Sub => self.builder.ins().isub(a, b),
                BinaryOp::Mul => self.builder.ins().imul(a, b),
                BinaryOp::Div => self.builder.ins().udiv(a, b),
                BinaryOp::Rem => self.builder.ins().urem(a, b),
                BinaryOp::Min => {
                    let c = self.builder.ins().icmp(IntCC::UnsignedLessThan, a, b);
                    self.builder.ins().select(c, a, b)
                }
                BinaryOp::Max => {
                    let c = self.builder.ins().icmp(IntCC::UnsignedGreaterThan, a, b);
                    self.builder.ins().select(c, a, b)
                }
            },
        };
        self.round_value(value, ty)
    }

    fn vector_binary(&mut self, op: BinaryOp, ty: ValueType, a: Value, b: Value) -> Value {
        let (dtype, _) = vector_shape(ty);
        let value = if dtype.is_float() {
            match op {
                BinaryOp::Add => self.builder.ins().fadd(a, b),
                BinaryOp::Sub => self.builder.ins().fsub(a, b),
                BinaryOp::Mul => self.builder.ins().fmul(a, b),
                BinaryOp::Div => self.builder.ins().fdiv(a, b),
                BinaryOp::Rem => panic!("floating vector remainder is not in typed kernel IR"),
                BinaryOp::Min => self.map_vector_binary(ty, a, b, |this, x, y| {
                    this.call_float2(HostImport::Fmin, x, y)
                }),
                BinaryOp::Max => self.map_vector_binary(ty, a, b, |this, x, y| {
                    this.call_float2(HostImport::Fmax, x, y)
                }),
            }
        } else if matches!(
            op,
            BinaryOp::Div | BinaryOp::Rem | BinaryOp::Min | BinaryOp::Max
        ) {
            self.map_vector_binary(ty, a, b, |this, x, y| {
                if dtype == DType::I32 {
                    match op {
                        BinaryOp::Div => this.signed_euclid(x, y).0,
                        BinaryOp::Rem => this.signed_euclid(x, y).1,
                        BinaryOp::Min => {
                            let c = this.builder.ins().icmp(IntCC::SignedLessThan, x, y);
                            this.builder.ins().select(c, x, y)
                        }
                        BinaryOp::Max => {
                            let c = this.builder.ins().icmp(IntCC::SignedGreaterThan, x, y);
                            this.builder.ins().select(c, x, y)
                        }
                        _ => unreachable!("lane-mapped signed vector operation"),
                    }
                } else {
                    match op {
                        BinaryOp::Div => this.builder.ins().udiv(x, y),
                        BinaryOp::Rem => this.builder.ins().urem(x, y),
                        BinaryOp::Min => {
                            let c = this.builder.ins().icmp(IntCC::UnsignedLessThan, x, y);
                            this.builder.ins().select(c, x, y)
                        }
                        BinaryOp::Max => {
                            let c = this.builder.ins().icmp(IntCC::UnsignedGreaterThan, x, y);
                            this.builder.ins().select(c, x, y)
                        }
                        _ => unreachable!("lane-mapped unsigned vector operation"),
                    }
                }
            })
        } else {
            match op {
                BinaryOp::Add => self.builder.ins().iadd(a, b),
                BinaryOp::Sub => self.builder.ins().isub(a, b),
                BinaryOp::Mul => self.builder.ins().imul(a, b),
                _ => unreachable!("non-lane-mapped integer vector operation"),
            }
        };
        self.round_vector(value, ty)
    }

    fn vector_unary(&mut self, op: UnaryOp, ty: ValueType, value: Value) -> Value {
        let dtype = vector_dtype(ty);
        let value = if dtype.is_float() {
            match op {
                UnaryOp::Neg => self.builder.ins().fneg(value),
                UnaryOp::Abs => self.builder.ins().fabs(value),
            }
        } else if dtype == DType::I32 {
            match op {
                UnaryOp::Neg => self.builder.ins().ineg(value),
                UnaryOp::Abs => self.map_vector_unary(ty, value, |this, lane| {
                    let zero = this.builder.ins().iconst(types::I32, 0);
                    let neg = this.builder.ins().ineg(lane);
                    let negative = this.builder.ins().icmp(IntCC::SignedLessThan, lane, zero);
                    this.builder.ins().select(negative, neg, lane)
                }),
            }
        } else {
            panic!("unsigned vector reached a signed unary operation")
        };
        self.round_vector(value, ty)
    }

    fn map_vector_unary(
        &mut self,
        ty: ValueType,
        value: Value,
        mut operation: impl FnMut(&mut Self, Value) -> Value,
    ) -> Value {
        let (_, lanes) = vector_shape(ty);
        let mut result = None;
        for lane in 0..lanes {
            let scalar = self.builder.ins().extractlane(value, lane as u8);
            let scalar = operation(self, scalar);
            result = Some(match result {
                None => self.builder.ins().splat(native_type(ty), scalar),
                Some(vector) => self.builder.ins().insertlane(vector, scalar, lane as u8),
            });
        }
        result.expect("typed vectors have at least one lane")
    }

    fn map_vector_binary(
        &mut self,
        ty: ValueType,
        a: Value,
        b: Value,
        mut operation: impl FnMut(&mut Self, Value, Value) -> Value,
    ) -> Value {
        let (_, lanes) = vector_shape(ty);
        let mut result = None;
        for lane in 0..lanes {
            let a = self.builder.ins().extractlane(a, lane as u8);
            let b = self.builder.ins().extractlane(b, lane as u8);
            let scalar = operation(self, a, b);
            result = Some(match result {
                None => self.builder.ins().splat(native_type(ty), scalar),
                Some(vector) => self.builder.ins().insertlane(vector, scalar, lane as u8),
            });
        }
        result.expect("typed vectors have at least one lane")
    }

    fn vector_cast(&mut self, value: Value, from: ValueType, to: ValueType) -> Value {
        let (from_dtype, from_lanes) = vector_shape(from);
        let (to_dtype, to_lanes) = vector_shape(to);
        assert_eq!(from_lanes, to_lanes, "typed vector cast changed lane count");
        let scalar_from = ValueType::Scalar(from_dtype);
        let scalar_to = ValueType::Scalar(to_dtype);
        let mut result = None;
        for lane in 0..from_lanes {
            let scalar = self.builder.ins().extractlane(value, lane as u8);
            let scalar = self.cast(scalar, scalar_from, scalar_to);
            result = Some(match result {
                None => self.builder.ins().splat(native_type(to), scalar),
                Some(vector) => self.builder.ins().insertlane(vector, scalar, lane as u8),
            });
        }
        result.expect("typed vectors have at least one lane")
    }

    fn round_vector(&mut self, value: Value, ty: ValueType) -> Value {
        let dtype = vector_dtype(ty);
        if !dtype.is_float() || dtype == DType::F32 {
            return value;
        }
        self.map_vector_unary(ty, value, |this, lane| {
            this.round_value(lane, ValueType::Scalar(dtype))
        })
    }

    fn math(&mut self, op: MathOp, a: Value) -> Value {
        match op {
            MathOp::Exp | MathOp::ExpFast => self.call_float1(HostImport::ApproxExp, a),
            MathOp::Log => self.call_float1(HostImport::ApproxLog, a),
            MathOp::Sin => self.call_float1(HostImport::ApproxSin, a),
            MathOp::Cos => self.call_float1(HostImport::ApproxCos, a),
            MathOp::Sqrt => self.builder.ins().sqrt(a),
            MathOp::Rsqrt => {
                let one = self.builder.ins().f32const(1.0);
                let root = self.builder.ins().sqrt(a);
                self.builder.ins().fdiv(one, root)
            }
            MathOp::Abs => self.builder.ins().fabs(a),
            MathOp::Max | MathOp::Min | MathOp::Fma => {
                panic!("multi-operand math operation was encoded as unary typed IR")
            }
        }
    }

    fn compare(&mut self, op: CmpOp, operand_type: ValueType, a: Value, b: Value) -> Value {
        let condition = match operand_type {
            ValueType::Scalar(dtype) if dtype.is_float() => self.builder.ins().fcmp(
                match op {
                    CmpOp::Eq => FloatCC::Equal,
                    CmpOp::Ne => FloatCC::NotEqual,
                    CmpOp::Lt => FloatCC::LessThan,
                    CmpOp::Le => FloatCC::LessThanOrEqual,
                    CmpOp::Gt => FloatCC::GreaterThan,
                    CmpOp::Ge => FloatCC::GreaterThanOrEqual,
                },
                a,
                b,
            ),
            ValueType::Scalar(DType::I32) => self.builder.ins().icmp(
                match op {
                    CmpOp::Eq => IntCC::Equal,
                    CmpOp::Ne => IntCC::NotEqual,
                    CmpOp::Lt => IntCC::SignedLessThan,
                    CmpOp::Le => IntCC::SignedLessThanOrEqual,
                    CmpOp::Gt => IntCC::SignedGreaterThan,
                    CmpOp::Ge => IntCC::SignedGreaterThanOrEqual,
                },
                a,
                b,
            ),
            _ => self.builder.ins().icmp(
                match op {
                    CmpOp::Eq => IntCC::Equal,
                    CmpOp::Ne => IntCC::NotEqual,
                    CmpOp::Lt => IntCC::UnsignedLessThan,
                    CmpOp::Le => IntCC::UnsignedLessThanOrEqual,
                    CmpOp::Gt => IntCC::UnsignedGreaterThan,
                    CmpOp::Ge => IntCC::UnsignedGreaterThanOrEqual,
                },
                a,
                b,
            ),
        };
        self.builder.ins().uextend(types::I32, condition)
    }

    fn truth(&mut self, value: Value) -> Value {
        self.builder.ins().icmp_imm(IntCC::NotEqual, value, 0)
    }

    fn cast(&mut self, value: Value, from: ValueType, to: ValueType) -> Value {
        if from == to {
            return value;
        }
        let from_float = matches!(from, ValueType::Scalar(dtype) if dtype.is_float());
        let to_float = matches!(to, ValueType::Scalar(dtype) if dtype.is_float());
        let from_bool = matches!(from, ValueType::Bool | ValueType::Scalar(DType::Bool));
        let to_bool = matches!(to, ValueType::Bool | ValueType::Scalar(DType::Bool));
        let result = if to_bool {
            let condition = if from_float {
                let zero = self.builder.ins().f32const(0.0);
                self.builder.ins().fcmp(FloatCC::NotEqual, value, zero)
            } else {
                self.builder.ins().icmp_imm(IntCC::NotEqual, value, 0)
            };
            self.builder.ins().uextend(types::I32, condition)
        } else if from_bool {
            if to_float {
                self.builder.ins().fcvt_from_uint(types::F32, value)
            } else if to == ValueType::Index {
                self.builder.ins().uextend(types::I64, value)
            } else {
                value
            }
        } else if from_float && to_float {
            value
        } else if from_float {
            let word = if to == ValueType::Scalar(DType::I32) {
                self.builder.ins().fcvt_to_sint_sat(types::I32, value)
            } else {
                self.builder.ins().fcvt_to_uint_sat(types::I32, value)
            };
            if to == ValueType::Index {
                self.builder.ins().uextend(types::I64, word)
            } else {
                word
            }
        } else if to_float {
            match from {
                ValueType::Scalar(DType::I32) => {
                    self.builder.ins().fcvt_from_sint(types::F32, value)
                }
                ValueType::Scalar(DType::U32) | ValueType::Index => {
                    self.builder.ins().fcvt_from_uint(types::F32, value)
                }
                _ => panic!("typed scalar cast contains a non-scalar source"),
            }
        } else {
            // Integer-to-integer casts preserve the low 32 bits. `Index`
            // is represented as I64 natively but is the language's u32
            // index domain, matching the other native backends.
            let word = if from == ValueType::Index {
                self.builder.ins().ireduce(types::I32, value)
            } else {
                value
            };
            if to == ValueType::Index {
                self.builder.ins().uextend(types::I64, word)
            } else {
                word
            }
        };
        self.round_value(result, to)
    }

    fn round_value(&mut self, value: Value, ty: ValueType) -> Value {
        let ValueType::Scalar(dtype) = ty else {
            return value;
        };
        if !dtype.is_float() || dtype == DType::F32 {
            return value;
        }
        let promoted = self.builder.ins().fpromote(types::F64, value);
        let rounded = self.call(round_import(dtype), &[promoted]);
        self.builder.ins().fdemote(types::F32, rounded)
    }

    fn signed_euclid(&mut self, lhs: Value, rhs: Value) -> (Value, Value) {
        let quotient = self.builder.ins().sdiv(lhs, rhs);
        let remainder = self.builder.ins().srem(lhs, rhs);
        let zero = self.builder.ins().iconst(types::I32, 0);
        let negative_remainder = self
            .builder
            .ins()
            .icmp(IntCC::SignedLessThan, remainder, zero);
        let negative_divisor = self.builder.ins().icmp(IntCC::SignedLessThan, rhs, zero);
        let minus_one = self.builder.ins().iconst(types::I32, -1);
        let one = self.builder.ins().iconst(types::I32, 1);
        let adjustment = self.builder.ins().select(negative_divisor, one, minus_one);
        let adjusted_quotient = self.builder.ins().iadd(quotient, adjustment);
        let neg_rhs = self.builder.ins().ineg(rhs);
        let abs_rhs = self.builder.ins().select(negative_divisor, neg_rhs, rhs);
        let adjusted_remainder = self.builder.ins().iadd(remainder, abs_rhs);
        (
            self.builder
                .ins()
                .select(negative_remainder, adjusted_quotient, quotient),
            self.builder
                .ins()
                .select(negative_remainder, adjusted_remainder, remainder),
        )
    }

    fn geometry(&mut self, geometry: GeometryValue) -> Value {
        match geometry {
            GeometryValue::WorkgroupId(axis) => {
                self.axis(self.linear_workgroup, self.layout.words.grid_first, axis)
            }
            GeometryValue::LocalId(axis) => {
                self.axis(self.linear_local, self.layout.words.workgroup_first, axis)
            }
            GeometryValue::GlobalId(axis) => {
                let group = self.axis(self.linear_workgroup, self.layout.words.grid_first, axis);
                let local = self.axis(self.linear_local, self.layout.words.workgroup_first, axis);
                let size = self.word(self.layout.words.workgroup_first + u32::from(axis));
                let base = self.builder.ins().imul(group, size);
                self.builder.ins().iadd(base, local)
            }
            GeometryValue::WorkgroupSize(axis) => {
                self.word(self.layout.words.workgroup_first + u32::from(axis))
            }
            GeometryValue::GridSize(axis) => {
                self.word(self.layout.words.grid_first + u32::from(axis))
            }
            GeometryValue::SubgroupLane => self.builder.ins().iconst(types::I64, 0),
        }
    }

    fn axis(&mut self, linear: Value, first: u32, axis: u8) -> Value {
        let x = self.word(first);
        match axis {
            0 => self.builder.ins().urem(linear, x),
            1 => {
                let q = self.builder.ins().udiv(linear, x);
                let y = self.word(first + 1);
                self.builder.ins().urem(q, y)
            }
            2 => {
                let y = self.word(first + 1);
                let xy = self.builder.ins().imul(x, y);
                self.builder.ins().udiv(linear, xy)
            }
            _ => panic!("kernel geometry axis exceeds rank three"),
        }
    }

    fn place_geometry<G: Clone>(&mut self, place: &ClosedPlace<G>) -> (G, u32, u32, Value) {
        match (place.kind, place.words) {
            (ClosedPlaceKind::Global { buffer_ordinal, .. }, ClosedPlaceWords::Binding(words)) => {
                let pointer = self.buffer(buffer_ordinal as usize);
                (place.geometry.clone(), words.first, words.rank, pointer)
            }
            (ClosedPlaceKind::Local { kind, .. }, ClosedPlaceWords::Local(words)) => {
                let pointer = match kind {
                    LaunchLocalKind::Workgroup => self.workgroup_scratch,
                    LaunchLocalKind::Participant => self.participant_scratch,
                    LaunchLocalKind::Register => self.register_scratch,
                };
                let offset = self.word(words.first);
                let pointer = self.builder.ins().iadd(pointer, offset);
                (place.geometry.clone(), words.first + 1, words.rank, pointer)
            }
            _ => panic!("closed CPU place kind and word layout disagree"),
        }
    }

    fn buffer(&mut self, position: usize) -> Value {
        self.builder.ins().load(
            types::I64,
            MemFlags::trusted(),
            self.buffers,
            (position as i32) * 8,
        )
    }

    fn extent(&mut self, place: &ClosedPlace, axis: u32) -> Value {
        let (_, first, rank, _) = self.place_geometry(place);
        if axis >= rank {
            panic!("typed extent operation names an absent axis");
        }
        self.word(first + axis)
    }

    fn address<G: CpuAddressGeometry + Clone>(
        &mut self,
        place: &ClosedPlace<G>,
        coordinates: &[Value],
    ) -> (Value, G, Value) {
        let (geometry, first, rank, pointer) = self.place_geometry(place);
        if coordinates.len() != rank as usize {
            panic!("typed memory operation rank disagrees with its place");
        }
        let mut units = self.builder.ins().iconst(types::I64, 0);
        for (axis, coordinate) in coordinates.iter().copied().enumerate() {
            let coordinate = if axis + 1 == coordinates.len() {
                match geometry.packet_group() {
                    Some(group) => self.builder.ins().udiv_imm(coordinate, i64::from(group)),
                    None => coordinate,
                }
            } else {
                coordinate
            };
            let stride = self.word(first + rank + axis as u32);
            let term = self.builder.ins().imul(coordinate, stride);
            units = self.builder.ins().iadd(units, term);
        }
        let bytes = self
            .builder
            .ins()
            .imul_imm(units, geometry.unit_bytes() as i64);
        (
            self.builder.ins().iadd(pointer, bytes),
            geometry,
            *coordinates.last().unwrap_or(&units),
        )
    }

    fn read(&mut self, place: &ClosedReadablePlace, coordinates: &[Value]) -> Value {
        let (address, geometry, logical_last) = self.address(place, coordinates);
        match &geometry {
            ReadableRepresentationGeometry::Dense(geometry) => {
                self.load_element(address, geometry.dtype)
            }
            ReadableRepresentationGeometry::Packed(geometry) => {
                let layout = &geometry.layout;
                let recipe = &geometry.decode;
                let mut temps = Vec::with_capacity(recipe.temporary_count());
                for step in recipe.steps() {
                    let value = match step {
                        DecodeStep::ReadPlaneField { into, plane, field } => {
                            let _ = into;
                            self.read_plane_field(
                                address,
                                logical_last,
                                layout.group,
                                &layout.planes[*plane as usize],
                                *field,
                            )
                        }
                        DecodeStep::InterpretCode {
                            into,
                            raw,
                            bits,
                            interpretation,
                        } => {
                            let _ = into;
                            self.interpret(temps[recipe.ordinal(*raw)], *bits, interpretation)
                        }
                        DecodeStep::DecodeFloatCode { into, raw, format } => {
                            let _ = into;
                            let raw = temps[recipe.ordinal(*raw)];
                            self.call(float_code_import(*format), &[raw])
                        }
                        DecodeStep::ConvertToF32 { into, from } => {
                            let _ = into;
                            let from_ty = recipe.dtype(*from);
                            let raw = temps[recipe.ordinal(*from)];
                            self.to_f32(raw, from_ty)
                        }
                        DecodeStep::Multiply { into, left, right } => {
                            let _ = into;
                            let l = temps[recipe.ordinal(*left)];
                            let r = temps[recipe.ordinal(*right)];
                            self.builder.ins().fmul(l, r)
                        }
                        DecodeStep::Negate { into, from } => {
                            let _ = into;
                            let value = temps[recipe.ordinal(*from)];
                            self.builder.ins().fneg(value)
                        }
                        DecodeStep::MultiplyAdd {
                            into,
                            factor,
                            multiplicand,
                            addend,
                        } => {
                            let _ = into;
                            let f = temps[recipe.ordinal(*factor)];
                            let m = temps[recipe.ordinal(*multiplicand)];
                            let a = temps[recipe.ordinal(*addend)];
                            self.builder.ins().fma(f, m, a)
                        }
                        DecodeStep::Cast { into, from, to } => {
                            let _ = into;
                            let value = temps[recipe.ordinal(*from)];
                            self.round_value(value, ValueType::Scalar(*to))
                        }
                    };
                    temps.push(value);
                }
                temps[recipe.ordinal(recipe.output())]
            }
        }
    }

    fn vector_read(
        &mut self,
        place: &ClosedReadablePlace,
        coordinates: &[Value],
        axis: u32,
        active: Value,
        ty: ValueType,
    ) -> Value {
        let (_, lanes) = vector_shape(ty);
        assert!(
            (axis as usize) < coordinates.len(),
            "typed vector read axis exceeds its coordinate rank"
        );
        let mut result: Option<Value> = None;
        for lane in 0..lanes {
            let enabled =
                self.builder
                    .ins()
                    .icmp_imm(IntCC::UnsignedGreaterThan, active, i64::from(lane));
            let read_block = self.builder.create_block();
            let zero_block = self.builder.create_block();
            let join = self.builder.create_block();
            self.builder
                .append_block_param(join, native_scalar_type(vector_dtype(ty)));
            self.builder
                .ins()
                .brif(enabled, read_block, &[], zero_block, &[]);

            self.builder.switch_to_block(read_block);
            self.builder.seal_block(read_block);
            let mut lane_coordinates = coordinates.to_vec();
            lane_coordinates[axis as usize] = self
                .builder
                .ins()
                .iadd_imm(lane_coordinates[axis as usize], i64::from(lane));
            let loaded = self.read(place, &lane_coordinates);
            self.builder.ins().jump(join, &[loaded.into()]);

            self.builder.switch_to_block(zero_block);
            self.builder.seal_block(zero_block);
            let zero = scalar_zero(&mut self.builder, vector_dtype(ty));
            self.builder.ins().jump(join, &[zero.into()]);

            self.builder.switch_to_block(join);
            self.builder.seal_block(join);
            let loaded = self.builder.block_params(join)[0];
            result = Some(match result {
                None => self.builder.ins().splat(native_type(ty), loaded),
                Some(vector) => self.builder.ins().insertlane(vector, loaded, lane as u8),
            });
        }
        result.expect("typed vectors have at least one lane")
    }

    fn read_plane(
        &mut self,
        place: &ClosedPackedPlace,
        plane: usize,
        coordinates: &[Value],
    ) -> Value {
        let (packet, geometry, logical_last) = self.address(place, coordinates);
        self.read_plane_field(
            packet,
            logical_last,
            geometry.layout.group,
            &geometry.layout.planes[plane],
            0,
        )
    }

    fn read_plane_field(
        &mut self,
        packet: Value,
        logical: Value,
        packet_group: u32,
        plane: &PlaneInfo,
        field: u32,
    ) -> Value {
        let base = self.builder.ins().iadd_imm(packet, i64::from(plane.offset));
        let local = self
            .builder
            .ins()
            .urem_imm(logical, i64::from(packet_group));
        let group = self.builder.ins().udiv_imm(local, i64::from(plane.group));
        let entry = self.builder.ins().imul_imm(group, i64::from(plane.fields));
        let entry = self.builder.ins().iadd_imm(entry, i64::from(field));
        match &plane.encoding {
            PlaneEncoding::Dense(dtype) => {
                let offset = self.builder.ins().imul_imm(entry, i64::from(dtype.bytes()));
                let address = self.builder.ins().iadd(base, offset);
                self.load_element(address, *dtype)
            }
            PlaneEncoding::Packed { bits, .. } => {
                let bit = self.builder.ins().imul_imm(entry, i64::from(*bits));
                let width = self.builder.ins().iconst(types::I32, i64::from(*bits));
                self.call(HostImport::PackedBits, &[base, bit, width])
            }
            PlaneEncoding::FloatCode { format } => {
                let bits = format.bits();
                let bit = self.builder.ins().imul_imm(entry, i64::from(bits));
                let width = self.builder.ins().iconst(types::I32, i64::from(bits));
                self.call(HostImport::PackedBits, &[base, bit, width])
            }
        }
    }

    fn interpret(&mut self, raw: Value, bits: u32, interpretation: &CodeInterpretation) -> Value {
        match interpretation {
            CodeInterpretation::Unsigned => raw,
            CodeInterpretation::TwosComplement => {
                let shift = 32 - bits;
                let shifted = self.builder.ins().ishl_imm(raw, i64::from(shift));
                self.builder.ins().sshr_imm(shifted, i64::from(shift))
            }
            CodeInterpretation::Offset(offset) => {
                self.builder.ins().iadd_imm(raw, -i64::from(*offset))
            }
            CodeInterpretation::Table(table) => {
                let mut value = self.builder.ins().iconst(types::I32, i64::from(table[0]));
                for (index, entry) in table.iter().copied().enumerate().skip(1) {
                    let matches = self.builder.ins().icmp_imm(IntCC::Equal, raw, index as i64);
                    let entry = self.builder.ins().iconst(types::I32, i64::from(entry));
                    value = self.builder.ins().select(matches, entry, value);
                }
                value
            }
        }
    }

    fn to_f32(&mut self, value: Value, dtype: DType) -> Value {
        match dtype {
            DType::F32 | DType::F16 | DType::BF16 => value,
            DType::I32 => self.builder.ins().fcvt_from_sint(types::F32, value),
            DType::U32 | DType::Bool => self.builder.ins().fcvt_from_uint(types::F32, value),
        }
    }

    fn write(&mut self, place: &ClosedDensePlace, coordinates: &[Value], value: Value) {
        let (address, geometry, _) = self.address(place, coordinates);
        self.store_element(address, geometry.dtype, value)
    }

    fn vector_write(
        &mut self,
        place: &ClosedDensePlace,
        coordinates: &[Value],
        axis: u32,
        active: Value,
        value: Value,
        ty: ValueType,
    ) {
        let (_, lanes) = vector_shape(ty);
        assert!(
            (axis as usize) < coordinates.len(),
            "typed vector write axis exceeds its coordinate rank"
        );
        for lane in 0..lanes {
            let enabled =
                self.builder
                    .ins()
                    .icmp_imm(IntCC::UnsignedGreaterThan, active, i64::from(lane));
            let store = self.builder.create_block();
            let continuation = self.builder.create_block();
            self.builder
                .ins()
                .brif(enabled, store, &[], continuation, &[]);
            self.builder.switch_to_block(store);
            self.builder.seal_block(store);
            let mut lane_coordinates = coordinates.to_vec();
            lane_coordinates[axis as usize] = self
                .builder
                .ins()
                .iadd_imm(lane_coordinates[axis as usize], i64::from(lane));
            let scalar = self.builder.ins().extractlane(value, lane as u8);
            self.write(place, &lane_coordinates, scalar);
            self.builder.ins().jump(continuation, &[]);
            self.builder.switch_to_block(continuation);
            self.builder.seal_block(continuation);
        }
    }

    fn convert_packet(
        &mut self,
        source: &ClosedExternalGlobalPlace,
        destination: &ClosedPackedGlobalPlace,
        recipe: &seismic_lang::registry::PacketRepackRecipe,
        packet: Value,
    ) {
        let source_layout = &source.geometry.layout;
        let destination_layout = &destination.geometry.layout;
        let source_base = self.buffer(source.buffer_ordinal as usize);
        let source_offset = self
            .builder
            .ins()
            .imul_imm(packet, i64::from(source_layout.packet_size));
        let source_packet = self.builder.ins().iadd(source_base, source_offset);
        let destination_base = self.buffer(destination.buffer_ordinal as usize);
        let destination_offset = self
            .builder
            .ins()
            .imul_imm(packet, i64::from(destination_layout.packet_size));
        let destination_packet = self
            .builder
            .ins()
            .iadd(destination_base, destination_offset);

        for (plane, plane_recipe) in destination_layout.planes.iter().zip(&recipe.planes) {
            let plane_base = self
                .builder
                .ins()
                .iadd_imm(destination_packet, i64::from(plane.offset));
            match plane_recipe {
                PlaneRepackRecipe::BitRoutes(routes) => {
                    for (destination_byte, byte_routes) in routes.chunks_exact(8).enumerate() {
                        let mut byte = self.builder.ins().iconst(types::I32, 0);
                        for (destination_bit, source_bit) in byte_routes.iter().copied().enumerate()
                        {
                            let bit = self.source_bits(source_packet, source_bit, 1);
                            let bit = self.builder.ins().ishl_imm(bit, destination_bit as i64);
                            byte = self.builder.ins().bor(byte, bit);
                        }
                        self.builder.ins().istore8(
                            MemFlags::trusted(),
                            byte,
                            plane_base,
                            destination_byte as i32,
                        );
                    }
                }
                PlaneRepackRecipe::DenseValues(values) => {
                    for (index, expression) in values.iter().enumerate() {
                        let value = self.repack_expression(source_packet, expression);
                        let offset = index
                            .checked_mul(plane.storage_dtype.bytes() as usize)
                            .expect("closed CPU repack plane offset exceeds usize");
                        let address = self.builder.ins().iadd_imm(plane_base, offset as i64);
                        self.store_element(address, plane.storage_dtype, value);
                    }
                }
            }
        }
    }

    fn source_bits(&mut self, source_packet: Value, bit: u32, width: u8) -> Value {
        let byte = bit / 8;
        let shift = bit % 8;
        let bytes = (u32::from(width) + shift).div_ceil(8);
        let mut raw = self.builder.ins().iconst(types::I32, 0);
        for index in 0..bytes {
            let loaded = self.builder.ins().load(
                types::I8,
                MemFlags::trusted(),
                source_packet,
                (byte + index) as i32,
            );
            let loaded = self.builder.ins().uextend(types::I32, loaded);
            let loaded = self.builder.ins().ishl_imm(loaded, i64::from(index * 8));
            raw = self.builder.ins().bor(raw, loaded);
        }
        let shifted = self.builder.ins().ushr_imm(raw, i64::from(shift));
        let mask = (1u32 << width) - 1;
        self.builder.ins().band_imm(shifted, i64::from(mask))
    }

    fn repack_expression(&mut self, source_packet: Value, expression: &RepackExpr) -> Value {
        match expression {
            RepackExpr::SourceBits { bit, width } => self.source_bits(source_packet, *bit, *width),
            RepackExpr::ShiftLeft { value, bits } => {
                let value = self.repack_expression(source_packet, value);
                self.builder.ins().ishl_imm(value, i64::from(*bits))
            }
            RepackExpr::BitOr(left, right) => {
                let left = self.repack_expression(source_packet, left);
                let right = self.repack_expression(source_packet, right);
                self.builder.ins().bor(left, right)
            }
            RepackExpr::OffsetI32 { value, offset } => {
                let value = self.repack_expression(source_packet, value);
                self.builder.ins().iadd_imm(value, i64::from(*offset))
            }
            RepackExpr::F16ToF32(value) => {
                let value = self.repack_expression(source_packet, value);
                self.call(HostImport::F16Load, &[value])
            }
            RepackExpr::I32ToF32(value) => {
                let value = self.repack_expression(source_packet, value);
                self.builder.ins().fcvt_from_sint(types::F32, value)
            }
            RepackExpr::MultiplyF32(left, right) => {
                let left = self.repack_expression(source_packet, left);
                let right = self.repack_expression(source_packet, right);
                self.builder.ins().fmul(left, right)
            }
        }
    }

    fn load_element(&mut self, address: Value, dtype: DType) -> Value {
        match dtype {
            DType::F32 => self
                .builder
                .ins()
                .load(types::F32, MemFlags::trusted(), address, 0),
            DType::I32 | DType::U32 => {
                self.builder
                    .ins()
                    .load(types::I32, MemFlags::trusted(), address, 0)
            }
            DType::Bool => {
                let value = self
                    .builder
                    .ins()
                    .load(types::I8, MemFlags::trusted(), address, 0);
                self.builder.ins().uextend(types::I32, value)
            }
            DType::F16 => {
                let bits = self
                    .builder
                    .ins()
                    .load(types::I16, MemFlags::trusted(), address, 0);
                let bits = self.builder.ins().uextend(types::I32, bits);
                self.call(HostImport::F16Load, &[bits])
            }
            DType::BF16 => {
                let bits = self
                    .builder
                    .ins()
                    .load(types::I16, MemFlags::trusted(), address, 0);
                let bits = self.builder.ins().uextend(types::I32, bits);
                let shifted = self.builder.ins().ishl_imm(bits, 16);
                self.builder
                    .ins()
                    .bitcast(types::F32, MemFlags::new(), shifted)
            }
        }
    }

    fn store_element(&mut self, address: Value, dtype: DType, value: Value) {
        let stored = match dtype {
            DType::F32 | DType::I32 | DType::U32 => value,
            DType::Bool => self.builder.ins().ireduce(types::I8, value),
            DType::F16 => {
                let bits = self.call(HostImport::F16Store, &[value]);
                self.builder.ins().ireduce(types::I16, bits)
            }
            DType::BF16 => {
                let bits = self
                    .builder
                    .ins()
                    .bitcast(types::I32, MemFlags::new(), value);
                let shifted = self.builder.ins().ushr_imm(bits, 16);
                self.builder.ins().ireduce(types::I16, shifted)
            }
        };
        self.builder
            .ins()
            .store(MemFlags::trusted(), stored, address, 0);
    }

    fn decode_word(&mut self, word: Value, dtype: DType) -> Value {
        match dtype {
            DType::F32 => {
                let bits = self.builder.ins().ireduce(types::I32, word);
                self.builder
                    .ins()
                    .bitcast(types::F32, MemFlags::new(), bits)
            }
            DType::I32 | DType::U32 | DType::Bool => self.builder.ins().ireduce(types::I32, word),
            DType::F16 => {
                let bits = self.builder.ins().ireduce(types::I32, word);
                self.call(HostImport::F16Load, &[bits])
            }
            DType::BF16 => {
                let bits = self.builder.ins().ireduce(types::I32, word);
                let bits = self.builder.ins().ishl_imm(bits, 16);
                self.builder
                    .ins()
                    .bitcast(types::F32, MemFlags::new(), bits)
            }
        }
    }

    fn encode_word(&mut self, value: Value, dtype: DType) -> Value {
        let raw = match dtype {
            DType::F32 => self
                .builder
                .ins()
                .bitcast(types::I32, MemFlags::new(), value),
            DType::I32 | DType::U32 | DType::Bool => value,
            DType::F16 => self.call(HostImport::F16Store, &[value]),
            DType::BF16 => {
                let bits = self
                    .builder
                    .ins()
                    .bitcast(types::I32, MemFlags::new(), value);
                self.builder.ins().ushr_imm(bits, 16)
            }
        };
        self.builder.ins().uextend(types::I64, raw)
    }

    fn branch(
        &mut self,
        cond: ErasedValue,
        then_id: BlockId,
        else_id: BlockId,
        outs: &[ClosedValue],
    ) {
        let then_block = self.builder.create_block();
        let else_block = self.builder.create_block();
        let join = self.builder.create_block();
        for out in outs {
            self.builder.append_block_param(join, native_type(out.ty));
        }
        let condition = self.truth(self.value(cond));
        self.builder
            .ins()
            .brif(condition, then_block, &[], else_block, &[]);
        let parent = self.values.clone();
        self.builder.switch_to_block(then_block);
        self.builder.seal_block(then_block);
        self.values = parent.clone();
        let then_values = self
            .emit_block(then_id)
            .expect("typed branch arm ends in Yield");
        let then_args = then_values
            .into_iter()
            .map(BlockArg::from)
            .collect::<Vec<_>>();
        self.builder.ins().jump(join, &then_args);
        self.builder.switch_to_block(else_block);
        self.builder.seal_block(else_block);
        self.values = parent.clone();
        let else_values = self
            .emit_block(else_id)
            .expect("typed branch arm ends in Yield");
        let else_args = else_values
            .into_iter()
            .map(BlockArg::from)
            .collect::<Vec<_>>();
        self.builder.ins().jump(join, &else_args);
        self.builder.switch_to_block(join);
        self.builder.seal_block(join);
        self.values = parent;
        let params = self.builder.block_params(join).to_vec();
        for (out, value) in outs.iter().zip(params) {
            self.define(out.value, value);
        }
    }

    fn repeat(
        &mut self,
        start: ErasedValue,
        end: ErasedValue,
        binder: ErasedValue,
        carries: &[ClosedValue],
        carry_params: &[ClosedValue],
        body: BlockId,
        outs: &[ClosedValue],
    ) {
        let header = self.builder.create_block();
        self.builder.append_block_param(header, types::I64);
        for value in carries {
            self.builder
                .append_block_param(header, native_type(value.ty));
        }
        let done = self.builder.create_block();
        for out in outs {
            self.builder.append_block_param(done, native_type(out.ty));
        }
        let initial = carries
            .iter()
            .map(|value| self.value(value.value))
            .collect::<Vec<_>>();
        let mut args = vec![self.value(start)];
        args.extend(initial);
        let args = args.into_iter().map(BlockArg::from).collect::<Vec<_>>();
        self.builder.ins().jump(header, &args);
        self.builder.switch_to_block(header);
        let params = self.builder.block_params(header).to_vec();
        let current = params[0];
        let end = self.value(end);
        let condition = self
            .builder
            .ins()
            .icmp(IntCC::UnsignedLessThan, current, end);
        let body_block = self.builder.create_block();
        self.builder.ins().brif(
            condition,
            body_block,
            &[],
            done,
            &params[1..]
                .iter()
                .copied()
                .map(BlockArg::from)
                .collect::<Vec<_>>(),
        );
        self.builder.switch_to_block(body_block);
        self.builder.seal_block(body_block);
        let parent = self.values.clone();
        self.values.insert(binder, current);
        for (parameter, value) in carry_params.iter().zip(params.iter().copied().skip(1)) {
            self.values.insert(parameter.value, value);
        }
        let yielded = self
            .emit_block(body)
            .expect("typed repeat body ends in Yield");
        let one = self.builder.ins().iconst(types::I64, 1);
        let next = self.builder.ins().iadd(current, one);
        let mut next_args = vec![next];
        next_args.extend(yielded);
        let next_args = next_args
            .into_iter()
            .map(BlockArg::from)
            .collect::<Vec<_>>();
        self.builder.ins().jump(header, &next_args);
        self.builder.seal_block(header);
        self.builder.switch_to_block(done);
        self.builder.seal_block(done);
        self.values = parent;
        let results = self.builder.block_params(done).to_vec();
        for (out, value) in outs.iter().zip(results) {
            self.define(out.value, value);
        }
    }

    fn call(&mut self, import: HostImport, args: &[Value]) -> Value {
        let reference = self.imports[&import];
        let call = self.builder.ins().call(reference, args);
        self.builder
            .inst_results(call)
            .first()
            .copied()
            .unwrap_or_else(|| self.builder.ins().iconst(types::I32, 0))
    }
    fn call_float1(&mut self, import: HostImport, value: Value) -> Value {
        self.call(import, &[value])
    }
    fn call_float2(&mut self, import: HostImport, a: Value, b: Value) -> Value {
        let a = self.builder.ins().fpromote(types::F64, a);
        let b = self.builder.ins().fpromote(types::F64, b);
        let result = self.call(import, &[a, b]);
        self.builder.ins().fdemote(types::F32, result)
    }
}

trait CpuAddressGeometry {
    fn packet_group(&self) -> Option<u32>;
    fn unit_bytes(&self) -> u64;
}

impl CpuAddressGeometry for DenseRepresentationGeometry {
    fn packet_group(&self) -> Option<u32> {
        None
    }
    fn unit_bytes(&self) -> u64 {
        u64::from(self.dtype.bytes())
    }
}

impl CpuAddressGeometry for PackedRepresentationGeometry {
    fn packet_group(&self) -> Option<u32> {
        Some(self.layout.group)
    }
    fn unit_bytes(&self) -> u64 {
        u64::from(self.layout.packet_size)
    }
}

impl CpuAddressGeometry for ReadableRepresentationGeometry {
    fn packet_group(&self) -> Option<u32> {
        match self {
            Self::Dense(geometry) => geometry.packet_group(),
            Self::Packed(geometry) => geometry.packet_group(),
        }
    }
    fn unit_bytes(&self) -> u64 {
        match self {
            Self::Dense(geometry) => geometry.unit_bytes(),
            Self::Packed(geometry) => geometry.unit_bytes(),
        }
    }
}

fn native_type(ty: ValueType) -> ir::Type {
    match ty {
        ValueType::Scalar(dtype) => native_scalar_type(dtype),
        ValueType::Vector { dtype, lanes } => native_scalar_type(dtype)
            .by(u32::from(lanes))
            .expect("CPU vector width is not a fixed Cranelift vector type"),
        ValueType::Bool => types::I32,
        ValueType::Index => types::I64,
        ValueType::Opaque { .. } => panic!("CPU has no opaque intrinsic values"),
    }
}

fn native_scalar_type(dtype: DType) -> ir::Type {
    if dtype.is_float() {
        // F16/BF16 values remain widened between explicit registry rounding
        // boundaries, exactly like scalar kernel values.
        types::F32
    } else {
        types::I32
    }
}

fn vector_shape(ty: ValueType) -> (DType, u16) {
    match ty {
        ValueType::Vector { dtype, lanes } => (dtype, lanes),
        _ => panic!("typed vector operation carries a non-vector value"),
    }
}

fn vector_dtype(ty: ValueType) -> DType {
    vector_shape(ty).0
}

fn scalar_zero(builder: &mut FunctionBuilder<'_>, dtype: DType) -> Value {
    if dtype.is_float() {
        builder.ins().f32const(0.0)
    } else {
        builder.ins().iconst(types::I32, 0)
    }
}
