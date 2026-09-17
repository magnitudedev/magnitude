use cranelift_codegen::ir::{
    self,
    condcodes::{FloatCC, IntCC},
    types, AbiParam, InstBuilder, MemFlags, Value,
};
use cranelift_frontend::{FunctionBuilder, Variable};
use seismic_lang::abi::ScalarParameter;
use seismic_lang::{
    ast::{AssignOp, BinaryOp, UnaryOp},
    hir::{self, Builtin, Expr, ExprKind, Index, Stmt, StmtKind, VarId, VarKind},
    lower::Lowered,
    repr,
    sym::{Atom, Sym},
    types::{DType, Elem, Ty},
};
use seismic_realization::LoadStrategy;
use seismic_realization::{
    execution::{ExecutionEvidence, MemoryObject, Multiplicity},
    BufferSpec, MathFunction,
};
use std::collections::HashMap;
use std::sync::Arc;

#[derive(Clone)]
enum Storage {
    Dense {
        pointer: Value,
        dtype: DType,
    },
    Packed {
        words: Value,
        scale: Value,
        bias: Option<Value>,
        name: String,
    },
}
/// Logical extent and its proven storage capacity. Runtime extents never change
/// the physical strides of an owning allocation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Dimension {
    capacity: i64,
    extent: Option<Value>,
}
impl Dimension {
    fn fixed(capacity: i64) -> Self {
        Self {
            capacity,
            extent: None,
        }
    }
}
#[derive(Clone)]
struct View {
    storage: Storage,
    offset: Value,
    shape: Vec<Dimension>,
    strides: Vec<i64>,
}
#[derive(Clone)]
enum Binding {
    Scalar(Variable, DType),
    View(View),
}
enum ResultValue {
    Scalar(Value, DType),
    View(View),
    Tuple(Vec<ResultValue>),
    Void,
}
impl ResultValue {
    fn scalar(self) -> Result<(Value, DType), String> {
        if let Self::Scalar(v, d) = self {
            Ok((v, d))
        } else {
            Err("expected scalar".into())
        }
    }
    fn view(self) -> Result<View, String> {
        if let Self::View(v) = self {
            Ok(v)
        } else {
            Err("expected tile/tensor view".into())
        }
    }
}

pub(crate) struct Emitter<'a, 'b> {
    pub builder: FunctionBuilder<'a>,
    loads: LoadStrategy,
    pub imports: Vec<(ir::FuncRef, MathFunction)>,
    lowered: &'b Lowered,
    bindings: HashMap<VarId, Binding>,
    indices: HashMap<String, Value>,
    constants: HashMap<String, i64>,
    dimensions: HashMap<String, Dimension>,
    scratch: Value,
    pub scratch_bytes: usize,
    pub buffers: Vec<BufferSpec>,
    pub scalars: Vec<ScalarParameter>,
    pub execution: ExecutionEvidence,
    multiplicity: Arc<Multiplicity>,
}
fn scalar_type(d: DType) -> Result<ir::Type, String> {
    Ok(match d {
        DType::F32 | DType::BF16 | DType::F16 => types::F32,
        DType::I32 | DType::U32 => types::I32,
        DType::Bool => types::I8,
    })
}
fn strides(shape: &[Dimension]) -> Result<Vec<i64>, String> {
    let mut out = vec![1; shape.len()];
    let mut stride = 1i64;
    for i in (0..shape.len()).rev() {
        if shape[i].capacity < 0 {
            return Err("negative scalar realization shape".into());
        }
        out[i] = stride;
        stride = stride
            .checked_mul(shape[i].capacity)
            .ok_or("scalar realization shape overflow")?;
    }
    Ok(out)
}
fn elements(shape: &[Dimension]) -> Result<i64, String> {
    shape.iter().try_fold(1i64, |a, b| {
        if b.capacity < 0 {
            Err("negative scalar realization shape".into())
        } else {
            a.checked_mul(b.capacity)
                .ok_or_else(|| "scalar realization shape overflow".into())
        }
    })
}

impl<'a, 'b> Emitter<'a, 'b> {
    pub fn new(
        lowered: &'b Lowered,
        mut builder: FunctionBuilder<'a>,
        buffers: Value,
        scalars: Value,
        scratch: Value,
        loads: LoadStrategy,
    ) -> Result<Self, String> {
        let zero = builder.ins().iconst(types::I64, 0);
        let entry = builder.current_block().unwrap();
        let mut s = Self {
            builder,
            loads,
            imports: Vec::new(),
            lowered,
            bindings: HashMap::new(),
            indices: HashMap::new(),
            constants: lowered.shapes.clone(),
            dimensions: HashMap::new(),
            scratch,
            scratch_bytes: 0,
            buffers: Vec::new(),
            scalars: Vec::new(),
            execution: ExecutionEvidence {
                blocks: HashMap::from([(entry, Arc::new(Multiplicity::Constant(1)))]),
                validity_guards: Vec::new(),
                memory_roots: HashMap::from([
                    (buffers, MemoryObject::BufferTable),
                    (scalars, MemoryObject::ScalarArguments),
                    (scratch, MemoryObject::PrivateScratch),
                ]),
            },
            multiplicity: Arc::new(Multiplicity::Constant(1)),
        };
        for (id, (name, ty)) in lowered.params.iter().enumerate() {
            match ty {
                Ty::Tensor(sh) => {
                    let shape = s.shape(&sh.shape)?;
                    let count = elements(&shape)?;
                    let storage = match &sh.elem {
                        Elem::Dtype(dtype) => {
                            scalar_type(*dtype)?;
                            let pointer =
                                s.parameter(buffers, name, "", count, dtype.bytes() as i64)?;
                            Storage::Dense {
                                pointer,
                                dtype: *dtype,
                            }
                        }
                        Elem::Repr(name_) => {
                            let r = repr::lookup(name_).ok_or("unknown packed representation")?;
                            if shape
                                .last()
                                .is_none_or(|n| n.capacity % i64::from(r.group) != 0)
                            {
                                return Err(
                                    "scalar realization packed parameter requires complete groups"
                                        .into(),
                                );
                            }
                            let words = s.parameter(
                                buffers,
                                name,
                                "words",
                                count / i64::from(r.codes_per_word()),
                                4,
                            )?;
                            let scale = s.parameter(
                                buffers,
                                name,
                                "scale",
                                count / i64::from(r.group),
                                r.coefficient.bytes() as i64,
                            )?;
                            let bias = if r.has_bias {
                                Some(s.parameter(
                                    buffers,
                                    name,
                                    "bias",
                                    count / i64::from(r.group),
                                    r.coefficient.bytes() as i64,
                                )?)
                            } else {
                                None
                            };
                            Storage::Packed {
                                words,
                                scale,
                                bias,
                                name: name_.clone(),
                            }
                        }
                        Elem::Param(_) => {
                            return Err("unbound scalar realization element parameter".into())
                        }
                    };
                    s.bindings.insert(
                        id,
                        Binding::View(View {
                            storage,
                            offset: zero,
                            strides: strides(&shape)?,
                            shape,
                        }),
                    );
                }
                Ty::Scalar(dtype) => {
                    let index =
                        i32::try_from(s.scalars.len() * 8).map_err(|_| "scalar ABI overflow")?;
                    let pointer = s.builder.ins().iadd_imm(scalars, i64::from(index));
                    let v = s.read_dense(pointer, zero, *dtype)?.0;
                    let var = s.builder.declare_var(scalar_type(*dtype)?);
                    s.builder.def_var(var, v);
                    s.bindings.insert(id, Binding::Scalar(var, *dtype));
                    let parameter = ScalarParameter::from_lowered(lowered, name, *dtype)?;
                    if let Some(bound) = parameter.index_bound {
                        let wide = s.builder.ins().sextend(types::I64, v);
                        let valid =
                            s.builder
                                .ins()
                                .icmp_imm(IntCC::UnsignedLessThan, wide, bound as i64);
                        s.require(valid);
                    }
                    s.scalars.push(parameter);
                }
                _ => {
                    return Err(
                        "scalar realization entry parameters must be tensors or scalars".into(),
                    )
                }
            }
        }
        Ok(s)
    }
    fn parameter(
        &mut self,
        table: Value,
        name: &str,
        plane: &str,
        count: i64,
        width: i64,
    ) -> Result<Value, String> {
        let bytes = count
            .checked_mul(width)
            .and_then(|n| usize::try_from(n).ok())
            .ok_or("scalar realization parameter size overflow")?;
        let index = i32::try_from(self.buffers.len() * 8)
            .map_err(|_| "scalar realization buffer ABI overflow")?;
        let ptr = self
            .builder
            .ins()
            .load(types::I64, MemFlags::new(), table, index);
        self.execution.memory_roots.insert(
            ptr,
            MemoryObject::Buffer {
                parameter: name.into(),
                plane: plane.into(),
            },
        );
        self.buffers.push(BufferSpec {
            parameter: name.into(),
            plane: plane.into(),
            bytes,
            alignment: usize::try_from(width).map_err(|_| "invalid storage alignment")?,
        });
        Ok(ptr)
    }
    fn shape(&self, shape: &[Sym]) -> Result<Vec<Dimension>, String> {
        shape
            .iter()
            .map(|s| {
                if let Some(n) = s
                    .eval(&|p| self.constants.get(p).copied())
                    .filter(|n| *n >= 0)
                {
                    return Ok(Dimension::fixed(n));
                }
                self.dimensions
                    .iter()
                    .find_map(|(name, dimension)| (*s == Sym::param(name)).then_some(*dimension))
                    .ok_or_else(|| format!("runtime tile extent `{s}` has no proven capacity"))
            })
            .collect()
    }
    fn extent(&mut self, dim: Dimension) -> Value {
        dim.extent
            .unwrap_or_else(|| self.builder.ins().iconst(types::I64, dim.capacity))
    }
    fn sym(&mut self, s: &Sym) -> Result<Value, String> {
        if let Some(n) = s.eval(&|p| self.constants.get(p).copied()) {
            return Ok(self.builder.ins().iconst(types::I64, n));
        }
        let mut total = self.builder.ins().iconst(types::I64, 0);
        for (monomial, coefficient) in s.monomials() {
            let mut term = self.builder.ins().iconst(types::I64, coefficient);
            for (atom, power) in monomial {
                let value = self.atom(atom)?;
                for _ in 0..*power {
                    term = self.builder.ins().imul(term, value);
                }
            }
            total = self.builder.ins().iadd(total, term);
        }
        Ok(total)
    }
    fn atom(&mut self, a: &Atom) -> Result<Value, String> {
        match a {
            Atom::Param(name) => {
                if let Some(dimension) = self.dimensions.get(name).copied() {
                    return Ok(self.extent(dimension));
                }
                if let Some(value) = self.indices.get(name) {
                    return Ok(*value);
                }
                if let Some(value) = self.constants.get(name) {
                    return Ok(self.builder.ins().iconst(types::I64, *value));
                }
                // Runtime index parameters and scalar argmax bindings retain a
                // semantic atom but obtain their current value from native SSA.
                let binding = self.lowered.vars.iter().enumerate().find_map(|(id, var)| {
                    (matches!(&var.kind,VarKind::Index(atom) if atom==a))
                        .then(|| self.bindings.get(&id).cloned())
                        .flatten()
                });
                if let Some(Binding::Scalar(var, dtype)) = binding {
                    let value = self.builder.use_var(var);
                    return match dtype {
                        DType::I32 => Ok(self.builder.ins().sextend(types::I64, value)),
                        DType::U32 => Ok(self.builder.ins().uextend(types::I64, value)),
                        _ => Err("runtime index requires integer storage".into()),
                    };
                }
                Err(format!("scalar realization unresolved index `{name}`"))
            }
            Atom::Quot(n, d) | Atom::Rem(n, d) => {
                let divisor = d
                    .eval(&|p| self.constants.get(p).copied())
                    .filter(|n| *n > 0)
                    .ok_or("scalar realization symbolic divisor must be a positive constant")?;
                let value = self.sym(n)?;
                let divisor = self.builder.ins().iconst(types::I64, divisor);
                // Symbolic numerators may be negative. Match Sym's Euclidean semantics.
                let q = self.builder.ins().sdiv(value, divisor);
                let r = self.builder.ins().srem(value, divisor);
                let neg = self.builder.ins().icmp_imm(IntCC::SignedLessThan, r, 0);
                if matches!(a, Atom::Quot(..)) {
                    let lower = self.builder.ins().iadd_imm(q, -1);
                    Ok(self.builder.ins().select(neg, lower, q))
                } else {
                    let positive = self.builder.ins().iadd(r, divisor);
                    Ok(self.builder.ins().select(neg, positive, r))
                }
            }
        }
    }
    fn block_with(&mut self, count: Arc<Multiplicity>) -> ir::Block {
        let block = self.builder.create_block();
        self.execution.blocks.insert(block, count);
        block
    }
    fn switch_to_block(&mut self, block: ir::Block) {
        self.multiplicity = self.execution.blocks[&block].clone();
        self.builder.switch_to_block(block);
    }
    fn loop_range(
        &mut self,
        lo: Value,
        hi: Value,
        body: impl FnOnce(&mut Self, Value) -> Result<(), String>,
    ) -> Result<(), String> {
        let parent = self.multiplicity.clone();
        let iterations = Arc::new(Multiplicity::Iterations {
            lower: lo,
            upper: hi,
        });
        let head = self.block_with(Multiplicity::product(
            parent.clone(),
            Arc::new(Multiplicity::PlusOne(iterations.clone())),
        ));
        let run = self.block_with(Multiplicity::product(parent.clone(), iterations));
        let done = self.block_with(parent);
        self.builder.append_block_param(head, types::I64);
        self.builder.ins().jump(head, &[lo.into()]);
        self.switch_to_block(head);
        let index = self.builder.block_params(head)[0];
        let cond = self.builder.ins().icmp(IntCC::SignedLessThan, index, hi);
        self.builder.ins().brif(cond, run, &[], done, &[]);
        self.switch_to_block(run);
        body(self, index)?;
        let next = self.builder.ins().iadd_imm(index, 1);
        self.builder.ins().jump(head, &[next.into()]);
        self.switch_to_block(done);
        Ok(())
    }
    fn each(
        &mut self,
        shape: &[Dimension],
        body: impl FnOnce(&mut Self, &[Value]) -> Result<(), String>,
    ) -> Result<(), String> {
        if shape.iter().any(|d| d.extent.is_some()) {
            return self.each_dynamic(shape, &mut Vec::new(), body);
        }
        let count = elements(shape)?;
        if count == 0 {
            return Ok(());
        }
        let strides = strides(shape)?;
        let lo = self.builder.ins().iconst(types::I64, 0);
        let hi = self.builder.ins().iconst(types::I64, count);
        self.loop_range(lo, hi, |s, flat| {
            let mut indices = Vec::new();
            for (dim, stride) in shape.iter().zip(strides) {
                let q = s.builder.ins().udiv_imm(flat, stride);
                indices.push(s.builder.ins().urem_imm(q, dim.capacity));
            }
            body(s, &indices)
        })
    }
    fn each_dynamic(
        &mut self,
        shape: &[Dimension],
        indices: &mut Vec<Value>,
        body: impl FnOnce(&mut Self, &[Value]) -> Result<(), String>,
    ) -> Result<(), String> {
        let Some((dim, rest)) = shape.split_first() else {
            return body(self, indices);
        };
        let lo = self.builder.ins().iconst(types::I64, 0);
        let hi = self.extent(*dim);
        self.loop_range(lo, hi, |s, index| {
            indices.push(index);
            let result = s.each_dynamic(rest, indices, body);
            indices.pop();
            result
        })
    }
    fn tile(&mut self, shape: Vec<Dimension>, dtype: DType) -> Result<View, String> {
        scalar_type(dtype)?;
        let bytes = elements(&shape)?
            .checked_mul(dtype.bytes() as i64)
            .and_then(|n| usize::try_from(n).ok())
            .ok_or("scalar realization tile size overflow")?;
        let start = self
            .scratch_bytes
            .checked_add(7)
            .map(|n| n & !7)
            .ok_or("scalar realization scratch overflow")?;
        self.scratch_bytes = start
            .checked_add(bytes)
            .ok_or("scalar realization scratch overflow")?;
        let pointer = self.builder.ins().iadd_imm(
            self.scratch,
            i64::try_from(start).map_err(|_| "scalar realization scratch exceeds address range")?,
        );
        let offset = self.builder.ins().iconst(types::I64, 0);
        Ok(View {
            storage: Storage::Dense { pointer, dtype },
            offset,
            strides: strides(&shape)?,
            shape,
        })
    }
    fn offset(&mut self, view: &View, indices: &[Value]) -> Result<Value, String> {
        if indices.len() != view.shape.len() {
            return Err("scalar realization index rank mismatch".into());
        }
        let mut offset = view.offset;
        for (index, stride) in indices.iter().zip(&view.strides) {
            let delta = self.builder.ins().imul_imm(*index, *stride);
            offset = self.builder.ins().iadd(offset, delta);
        }
        Ok(offset)
    }
    fn address(&mut self, base: Value, index: Value, width: u32) -> Value {
        let bytes = self.builder.ins().imul_imm(index, i64::from(width));
        self.builder.ins().iadd(base, bytes)
    }
    fn read_dense(
        &mut self,
        base: Value,
        offset: Value,
        dtype: DType,
    ) -> Result<(Value, DType), String> {
        let ptr = self.address(base, offset, dtype.bytes());
        Ok(match dtype {
            DType::F32 | DType::I32 | DType::U32 | DType::Bool => (
                self.builder
                    .ins()
                    .load(scalar_type(dtype)?, MemFlags::new(), ptr, 0),
                dtype,
            ),
            DType::BF16 => {
                let bits = self.builder.ins().load(types::I16, MemFlags::new(), ptr, 0);
                let bits = self.builder.ins().uextend(types::I32, bits);
                let bits = self.builder.ins().ishl_imm(bits, 16);
                (
                    self.builder
                        .ins()
                        .bitcast(types::F32, MemFlags::new(), bits),
                    dtype,
                )
            }
            DType::F16 => {
                let bits = self.builder.ins().load(types::I16, MemFlags::new(), ptr, 0);
                let bits = self.builder.ins().uextend(types::I32, bits);
                (self.decode_f16(bits), dtype)
            }
        })
    }
    fn read(&mut self, view: &View, indices: &[Value]) -> Result<(Value, DType), String> {
        let offset = self.offset(view, indices)?;
        match &view.storage {
            Storage::Dense { pointer, dtype } => self.read_dense(*pointer, offset, *dtype),
            Storage::Packed {
                words,
                scale,
                bias,
                name,
            } => {
                let r = repr::lookup(name).ok_or("unknown packed representation")?;
                let cpw = i64::from(r.codes_per_word());
                let group = self.builder.ins().udiv_imm(offset, i64::from(r.group));
                let word_index = self.builder.ins().udiv_imm(offset, cpw);
                let (word, _) = self.read_dense(*words, word_index, DType::U32)?;
                let pos = self.builder.ins().urem_imm(offset, cpw);
                let shift = self.builder.ins().imul_imm(pos, i64::from(r.bits));
                let shift = self.builder.ins().ireduce(types::I32, shift);
                let code = self.builder.ins().ushr(word, shift);
                let code = self.builder.ins().band_imm(code, (1i64 << r.bits) - 1);
                let code = match r.code {
                    repr::CodeInterpretation::Unsigned=>self.builder.ins().fcvt_from_uint(types::F32,code),
                    repr::CodeInterpretation::Offset(zero)=>{
                        let code=self.builder.ins().iadd_imm(code,-i64::from(zero));
                        self.builder.ins().fcvt_from_sint(types::F32,code)
                    }
                    repr::CodeInterpretation::TwosComplement=>{
                        let code=self.builder.ins().ishl_imm(code,i64::from(32-r.bits));
                        let code=self.builder.ins().sshr_imm(code,i64::from(32-r.bits));
                        self.builder.ins().fcvt_from_sint(types::F32,code)
                    }
                    repr::CodeInterpretation::Table(table)=>{
                        let mut decoded=self.builder.ins().iconst(types::I32,i64::from(*table.last().ok_or("empty code table")?));
                        for (i,value) in table.iter().enumerate().rev().skip(1) {
                            let equal=self.builder.ins().icmp_imm(IntCC::Equal,code,i as i64);
                            let value=self.builder.ins().iconst(types::I32,i64::from(*value));
                            decoded=self.builder.ins().select(equal,value,decoded);
                        }
                        self.builder.ins().fcvt_from_sint(types::F32,decoded)
                    }
                };
                let (scale, _) = self.read_dense(*scale, group, r.coefficient)?;
                let bias = if let Some(bias) = bias {
                    self.read_dense(*bias, group, r.coefficient)?.0
                } else {
                    self.builder.ins().f32const(0.0)
                };
                Ok((self.builder.ins().fma(code, scale, bias), DType::F32))
            }
        }
    }
    /// IEEE binary16 conversion expressed in the shared typed instruction IR.
    /// All shifts are bounded, including values whose selected result is zero/Inf.
    fn encode_f16(&mut self, value: Value) -> Value {
        let bits = self
            .builder
            .ins()
            .bitcast(types::I32, MemFlags::new(), value);
        let sign = self.builder.ins().ushr_imm(bits, 16);
        let sign = self.builder.ins().band_imm(sign, 0x8000);
        let magnitude = self.builder.ins().band_imm(bits, 0x7fff_ffff);
        let exponent = self.builder.ins().ushr_imm(magnitude, 23);
        let mantissa = self.builder.ins().band_imm(magnitude, 0x7f_ffff);
        let mantissa = self.builder.ins().bor_imm(mantissa, 0x80_0000);
        let normal_lsb = self.builder.ins().ushr_imm(magnitude, 13);
        let normal_lsb = self.builder.ins().band_imm(normal_lsb, 1);
        let normal_bias = self.builder.ins().iadd_imm(normal_lsb, 0xfff);
        let normal = self.builder.ins().iadd(magnitude, normal_bias);
        let normal = self.builder.ins().ushr_imm(normal, 13);
        let normal = self.builder.ins().iadd_imm(normal, -(112 << 10));
        let base = self.builder.ins().iconst(types::I32, 126);
        let shift = self.builder.ins().isub(base, exponent);
        let fourteen = self.builder.ins().iconst(types::I32, 14);
        let twenty_four = self.builder.ins().iconst(types::I32, 24);
        let low = self
            .builder
            .ins()
            .icmp_imm(IntCC::SignedLessThan, shift, 14);
        let shift = self.builder.ins().select(low, fourteen, shift);
        let high = self
            .builder
            .ins()
            .icmp_imm(IntCC::SignedGreaterThan, shift, 24);
        let shift = self.builder.ins().select(high, twenty_four, shift);
        let lsb = self.builder.ins().ushr(mantissa, shift);
        let lsb = self.builder.ins().band_imm(lsb, 1);
        let one = self.builder.ins().iconst(types::I32, 1);
        let previous = self.builder.ins().iadd_imm(shift, -1);
        let bias = self.builder.ins().ishl(one, previous);
        let bias = self.builder.ins().iadd_imm(bias, -1);
        let bias = self.builder.ins().iadd(bias, lsb);
        let rounded = self.builder.ins().iadd(mantissa, bias);
        let subnormal = self.builder.ins().ushr(rounded, shift);
        let small = self
            .builder
            .ins()
            .icmp_imm(IntCC::UnsignedLessThan, exponent, 113);
        let rounded = self.builder.ins().select(small, subnormal, normal);
        let zero = self.builder.ins().iconst(types::I32, 0);
        let tiny = self
            .builder
            .ins()
            .icmp_imm(IntCC::UnsignedLessThan, exponent, 102);
        let rounded = self.builder.ins().select(tiny, zero, rounded);
        let infinity = self.builder.ins().iconst(types::I32, 0x7c00);
        let overflow =
            self.builder
                .ins()
                .icmp_imm(IntCC::UnsignedGreaterThanOrEqual, magnitude, 0x477f_f000);
        let rounded = self.builder.ins().select(overflow, infinity, rounded);
        let nan = self
            .builder
            .ins()
            .icmp_imm(IntCC::UnsignedGreaterThan, magnitude, 0x7f80_0000);
        let quiet_nan = self.builder.ins().iconst(types::I32, 0x7e00);
        let rounded = self.builder.ins().select(nan, quiet_nan, rounded);
        self.builder.ins().bor(sign, rounded)
    }
    fn decode_f16(&mut self, bits: Value) -> Value {
        let sign = self.builder.ins().band_imm(bits, 0x8000);
        let sign = self.builder.ins().ishl_imm(sign, 16);
        let magnitude = self.builder.ins().band_imm(bits, 0x7fff);
        let shifted = self.builder.ins().ishl_imm(magnitude, 13);
        let normal = self.builder.ins().iadd_imm(shifted, 112 << 23);
        let special = self.builder.ins().bor_imm(shifted, 0x7f80_0000);
        let exceptional =
            self.builder
                .ins()
                .icmp_imm(IntCC::UnsignedGreaterThanOrEqual, magnitude, 0x7c00);
        let decoded = self.builder.ins().select(exceptional, special, normal);
        let mantissa = self.builder.ins().band_imm(bits, 0x3ff);
        let small = self.builder.ins().fcvt_from_uint(types::F32, mantissa);
        let quantum = self.builder.ins().f32const(2.0f32.powi(-24));
        let small = self.builder.ins().fmul(small, quantum);
        let small = self
            .builder
            .ins()
            .bitcast(types::I32, MemFlags::new(), small);
        let subnormal = self
            .builder
            .ins()
            .icmp_imm(IntCC::UnsignedLessThan, magnitude, 0x400);
        let decoded = self.builder.ins().select(subnormal, small, decoded);
        let decoded = self.builder.ins().bor(sign, decoded);
        self.builder
            .ins()
            .bitcast(types::F32, MemFlags::new(), decoded)
    }
    fn publish(&mut self, value: Value, dtype: DType) -> Value {
        match dtype {
            DType::BF16 => self.narrow_bf16(value),
            DType::F16 => {
                let bits = self.encode_f16(value);
                self.decode_f16(bits)
            }
            _ => value,
        }
    }
    fn narrow_bf16(&mut self, value: Value) -> Value {
        let bits = self
            .builder
            .ins()
            .bitcast(types::I32, MemFlags::new(), value);
        let lsb = self.builder.ins().ushr_imm(bits, 16);
        let lsb = self.builder.ins().band_imm(lsb, 1);
        let bias = self.builder.ins().iadd_imm(lsb, 0x7fff);
        let rounded = self.builder.ins().iadd(bits, bias);
        let rounded = self
            .builder
            .ins()
            .band_imm(rounded, 0xffff0000u32 as i32 as i64);
        let nan = self.builder.ins().fcmp(FloatCC::Unordered, value, value);
        let nanbits = self.builder.ins().bor_imm(bits, 0x00400000);
        let bits = self.builder.ins().select(nan, nanbits, rounded);
        self.builder
            .ins()
            .bitcast(types::F32, MemFlags::new(), bits)
    }
    fn cast(&mut self, value: Value, from: DType, to: DType) -> Result<Value, String> {
        if from == to {
            return Ok(value);
        }
        scalar_type(to)?;
        let result = if from.is_float() && to.is_float() {
            value
        } else if from.is_int() && to.is_float() {
            if from == DType::U32 {
                self.builder.ins().fcvt_from_uint(types::F32, value)
            } else {
                self.builder.ins().fcvt_from_sint(types::F32, value)
            }
        } else if from.is_float() && to.is_int() {
            if to == DType::U32 {
                self.builder.ins().fcvt_to_uint_sat(types::I32, value)
            } else {
                self.builder.ins().fcvt_to_sint_sat(types::I32, value)
            }
        } else if from.is_int() && to.is_int() {
            value
        } else {
            return Err("scalar realization boolean conversion is not implemented".into());
        };
        Ok(self.publish(result, to))
    }
    fn write(
        &mut self,
        view: &View,
        indices: &[Value],
        value: Value,
        from: DType,
    ) -> Result<(), String> {
        let Storage::Dense { pointer, dtype } = view.storage else {
            return Err("scalar realization packed stores need encoding semantics".into());
        };
        let value = self.cast(value, from, dtype)?;
        let offset = self.offset(view, indices)?;
        let ptr = self.address(pointer, offset, dtype.bytes());
        let value = if dtype == DType::BF16 {
            let bits = self
                .builder
                .ins()
                .bitcast(types::I32, MemFlags::new(), value);
            let bits = self.builder.ins().ushr_imm(bits, 16);
            self.builder.ins().ireduce(types::I16, bits)
        } else if dtype == DType::F16 {
            let bits = self.encode_f16(value);
            self.builder.ins().ireduce(types::I16, bits)
        } else {
            value
        };
        self.builder.ins().store(MemFlags::new(), value, ptr, 0);
        Ok(())
    }
    fn equal_shape(&mut self, a: &[Dimension], b: &[Dimension]) -> Result<(), String> {
        if a.len() != b.len() {
            return Err("copy rank mismatch".into());
        }
        for (a, b) in a.iter().zip(b) {
            if a == b {
                continue;
            }
            if a.extent.is_none() && b.extent.is_none() {
                return Err("copy shape mismatch".into());
            }
            let a = self.extent(*a);
            let b = self.extent(*b);
            let valid = self.builder.ins().icmp(IntCC::Equal, a, b);
            self.require(valid);
        }
        Ok(())
    }
    fn copy(&mut self, source: &View, target: &View) -> Result<(), String> {
        self.equal_shape(&source.shape, &target.shape)?;
        self.each(&source.shape, |s, indices| {
            let (value, dtype) = s.read(source, indices)?;
            s.write(target, indices, value, dtype)
        })
    }
    fn materialize(&mut self, source: View) -> Result<View, String> {
        let dtype = match &source.storage {
            Storage::Dense { dtype, .. } => *dtype,
            Storage::Packed { .. } => DType::F32,
        };
        let target = self.tile(source.shape.clone(), dtype)?;
        self.copy(&source, &target)?;
        Ok(target)
    }
    pub fn parallel_root(&mut self, body: &[Stmt], flat: Value) -> Result<u64, String> {
        let [stmt] = body else {
            return Err("parallel dispatch requires exactly one outer parallel domain".into());
        };
        let StmtKind::Parallel {
            vars,
            extents,
            body,
        } = &stmt.kind
        else {
            return Err("parallel dispatch requires an outer parallel domain".into());
        };
        let shape = self.shape(extents)?;
        if shape.iter().any(|d| d.extent.is_some()) {
            return Err("parallel dispatch requires a static root domain".into());
        }
        let count = elements(&shape)?;
        if count == 0 {
            return Ok(0);
        }
        let valid = self
            .builder
            .ins()
            .icmp_imm(IntCC::UnsignedLessThan, flat, count);
        self.require(valid);
        for ((var, dimension), stride) in vars.iter().zip(&shape).zip(strides(&shape)?) {
            let q = self.builder.ins().udiv_imm(flat, stride);
            let index = self.builder.ins().urem_imm(q, dimension.capacity);
            self.bind_index(*var, index)?;
        }
        self.body(body)?;
        Ok(count as u64)
    }
    pub fn body(&mut self, body: &[Stmt]) -> Result<(), String> {
        for stmt in body {
            self.stmt(stmt)
                .map_err(|e| format!("{}:{}: {e}", self.lowered.name, stmt.span.start))?;
        }
        Ok(())
    }
    fn bind_index(&mut self, id: VarId, value: Value) -> Result<(), String> {
        let VarKind::Index(Atom::Param(name)) = &self.lowered.vars[id].kind else {
            return Err("invalid scalar realization loop binding".into());
        };
        self.indices.insert(name.clone(), value);
        Ok(())
    }
    fn stmt(&mut self, s: &Stmt) -> Result<(), String> {
        match &s.kind {
            StmtKind::Parallel {
                vars,
                extents,
                body,
            } => {
                let shape = self.shape(extents)?;
                self.each(&shape, |s, indices| {
                    for (var, index) in vars.iter().zip(indices) {
                        s.bind_index(*var, *index)?;
                    }
                    s.body(body)
                })
            }
            StmtKind::Owned { vars, tile, body } => {
                let tile = self.expr(tile)?.view()?;
                self.each(&tile.shape, |s, indices| {
                    for (var, index) in vars.iter().zip(indices) {
                        s.bind_index(*var, *index)?;
                    }
                    s.body(body)
                })
            }
            StmtKind::Range { var, lo, hi, body } => {
                let lo = self.sym(lo)?;
                let hi = self.sym(hi)?;
                self.loop_range(lo, hi, |s, index| {
                    s.bind_index(*var, index)?;
                    s.body(body)
                })
            }
            StmtKind::LoadLoop {
                vars,
                views,
                axis,
                piece,
                body,
                capacity,
            } => {
                let sources = views
                    .iter()
                    .map(|expr| self.expr(expr)?.view())
                    .collect::<Result<Vec<_>, _>>()?;
                let first = sources.first().ok_or("empty stream")?;
                let dimension = *first.shape.get(*axis).ok_or("invalid stream axis")?;
                for view in &sources[1..] {
                    self.equal_shape(
                        &[dimension],
                        &[*view.shape.get(*axis).ok_or("invalid stream axis")?],
                    )?;
                }
                let Atom::Param(name) = piece else {
                    return Err("unresolved stream piece".into());
                };
                if dimension.capacity == 0 {
                    return Ok(());
                }
                let old_dimensions = self.dimensions.clone();
                let old_constants = self.constants.clone();
                let old_bindings = self.bindings.clone();
                let emit_piece =
                    |s: &mut Self, start: Value, dim: Dimension| -> Result<(), String> {
                        if dim.extent.is_none() {
                            s.constants.insert(name.clone(), dim.capacity);
                        } else {
                            s.constants.remove(name);
                        }
                        s.dimensions.insert(name.clone(), dim);
                        for (var, source) in vars.iter().zip(&sources) {
                            let mut view = source.clone();
                            view.shape[*axis] = dim;
                            let delta = s.builder.ins().imul_imm(start, view.strides[*axis]);
                            view.offset = s.builder.ins().iadd(view.offset, delta);
                            let borrow = s.loads == LoadStrategy::BorrowProvenReadOnly
                                && !body.iter().any(|stmt| {
                                    seismic_lang::effects::tensor_effect(stmt, &s.lowered.backend)
                                        || seismic_lang::effects::tile_mutated(
                                            stmt,
                                            *var,
                                            &s.lowered.backend,
                                        )
                                });
                            let tile = if borrow { view } else { s.materialize(view)? };
                            s.bindings.insert(*var, Binding::View(tile));
                        }
                        s.body(body)
                    };
                if let Some(capacity) = capacity {
                    if *capacity <= 0 {
                        return Err("stream piece capacity must be positive".into());
                    }
                    let extent = self.extent(dimension);
                    let divisor = self.builder.ins().iconst(types::I64, *capacity);
                    // Quotient + nonzero remainder avoids overflowing extent + capacity - 1.
                    let whole = self.builder.ins().udiv(extent, divisor);
                    let tail = self.builder.ins().urem(extent, divisor);
                    let has_tail = self.builder.ins().icmp_imm(IntCC::NotEqual, tail, 0);
                    let extra = self.builder.ins().uextend(types::I64, has_tail);
                    let chunks = self.builder.ins().iadd(whole, extra);
                    let zero = self.builder.ins().iconst(types::I64, 0);
                    self.loop_range(zero, chunks, |s, chunk| {
                        let start = s.builder.ins().imul_imm(chunk, *capacity);
                        let remaining = s.builder.ins().isub(extent, start);
                        let short =
                            s.builder
                                .ins()
                                .icmp_imm(IntCC::UnsignedLessThan, remaining, *capacity);
                        let size = s.builder.ins().select(short, remaining, divisor);
                        emit_piece(
                            s,
                            start,
                            Dimension {
                                capacity: (*capacity).min(dimension.capacity),
                                extent: Some(size),
                            },
                        )
                    })?;
                } else {
                    let zero = self.builder.ins().iconst(types::I64, 0);
                    emit_piece(self, zero, dimension)?;
                }
                // Retain writes to pre-existing scalar variables and tile storage,
                // but do not leak piece-local SSA bindings outside their loop.
                self.bindings = old_bindings;
                self.dimensions = old_dimensions;
                self.constants = old_constants;
                Ok(())
            }
            StmtKind::Assign { target, op, value } => {
                if self.loads == LoadStrategy::BorrowProvenReadOnly && *op == AssignOp::Assign {
                    if let (
                        ExprKind::Var(var),
                        ExprKind::Builtin {
                            name: Builtin::Load,
                            args,
                        },
                    ) = (&target.kind, &value.kind)
                    {
                        if !self.bindings.contains_key(var)
                            && seismic_lang::effects::load_can_borrow(
                                &self.lowered.body,
                                *var,
                                &self.lowered.backend,
                            )
                        {
                            let view = self.expr(&args[0])?.view()?;
                            self.bindings.insert(*var, Binding::View(view));
                            return Ok(());
                        }
                    }
                }
                let value = self.expr(value)?;
                match &target.kind {
                    ExprKind::Var(id) => match value {
                        ResultValue::View(view) => {
                            if *op != AssignOp::Assign {
                                return Err("compound view assignment".into());
                            }
                            if matches!(target.ty, Ty::Tile(_)) {
                                if let Some(Binding::View(destination)) =
                                    self.bindings.get(id).cloned()
                                {
                                    if matches!(value_kind(s), Some(ExprKind::TileAlloc { .. })) {
                                        self.equal_shape(&destination.shape, &view.shape)?;
                                        // The existing variable owns storage. Allocation does
                                        // not publish a value; subsequent checked writes initialize it.
                                        return Ok(());
                                    }
                                    // Tile assignment is a value copy, matching the interpreter.
                                    // Snapshot first handles self-transpose/slices without alias loss.
                                    let source = self.materialize(view)?;
                                    return self.copy(&source, &destination);
                                }
                                let view = if matches!(
                                    value_kind(s),
                                    Some(
                                        ExprKind::Var(_)
                                            | ExprKind::Index { .. }
                                            | ExprKind::Transpose(_)
                                    )
                                ) {
                                    self.materialize(view)?
                                } else {
                                    view
                                };
                                self.bindings.insert(*id, Binding::View(view));
                            } else {
                                if self.bindings.contains_key(id) {
                                    return Err("tensor view rebinding needs explicit control-flow alias analysis".into());
                                }
                                self.bindings.insert(*id, Binding::View(view));
                            }
                            Ok(())
                        }
                        ResultValue::Scalar(value, from) => {
                            let dtype = match target.ty {
                                Ty::Scalar(d) => d,
                                _ => return Err("scalar assignment to non-scalar".into()),
                            };
                            let value = self.cast(value, from, dtype)?;
                            let var = if let Some(Binding::Scalar(var, _)) = self.bindings.get(id) {
                                *var
                            } else {
                                let var = self.builder.declare_var(scalar_type(dtype)?);
                                self.bindings.insert(*id, Binding::Scalar(var, dtype));
                                var
                            };
                            let value = if *op != AssignOp::Assign {
                                let old = self.builder.use_var(var);
                                self.binary(assign_binary(*op)?, old, value, dtype)?
                            } else {
                                value
                            };
                            self.builder.def_var(var, value);
                            Ok(())
                        }
                        _ => Err("unsupported scalar realization local binding".into()),
                    },
                    ExprKind::Index { .. } => {
                        let view = self.index_view(target)?;
                        let (mut value, from) = value.scalar()?;
                        let dtype = match view.storage {
                            Storage::Dense { dtype, .. } => dtype,
                            _ => return Err("packed element assignment".into()),
                        };
                        value = self.cast(value, from, dtype)?;
                        if *op != AssignOp::Assign {
                            let old = self.read(&view, &[])?.0;
                            value = self.binary(assign_binary(*op)?, old, value, dtype)?;
                        }
                        self.write(&view, &[], value, dtype)
                    }
                    _ => Err("unsupported scalar realization assignment target".into()),
                }
            }
            StmtKind::Expr(e) => {
                self.expr(e)?;
                Ok(())
            }
            StmtKind::If { cond, then, els } => {
                let (condition, dtype) = self.expr(cond)?.scalar()?;
                if dtype != DType::Bool {
                    return Err("non-boolean branch condition".into());
                }
                let parent = self.multiplicity.clone();
                let yes = self.block_with(Multiplicity::product(
                    parent.clone(),
                    Arc::new(Multiplicity::Predicate {
                        value: condition,
                        expected: true,
                    }),
                ));
                let no = self.block_with(Multiplicity::product(
                    parent.clone(),
                    Arc::new(Multiplicity::Predicate {
                        value: condition,
                        expected: false,
                    }),
                ));
                let join = self.block_with(parent);
                self.builder.ins().brif(condition, yes, &[], no, &[]);
                let bindings = self.bindings.clone();
                let indices = self.indices.clone();
                let constants = self.constants.clone();
                let dimensions = self.dimensions.clone();
                self.switch_to_block(yes);
                self.body(then)?;
                self.builder.ins().jump(join, &[]);
                self.bindings = bindings.clone();
                self.indices = indices.clone();
                self.constants = constants.clone();
                self.dimensions = dimensions.clone();
                self.switch_to_block(no);
                self.body(els)?;
                self.builder.ins().jump(join, &[]);
                self.bindings = bindings;
                self.indices = indices;
                self.constants = constants;
                self.dimensions = dimensions;
                self.switch_to_block(join);
                Ok(())
            }
            StmtKind::Lanes { .. } => {
                Err("GPU lane operation in scalar realization realization".into())
            }
        }
    }
    fn index_value(&mut self, e: &Expr) -> Result<Value, String> {
        if let Some(sym) = &e.sym {
            return self.sym(sym);
        }
        let (value, dtype) = self.expr(e)?.scalar()?;
        match dtype {
            DType::I32 => Ok(self.builder.ins().sextend(types::I64, value)),
            DType::U32 => Ok(self.builder.ins().uextend(types::I64, value)),
            _ => Err("view index requires integer storage".into()),
        }
    }
    fn index_view(&mut self, e: &Expr) -> Result<View, String> {
        let ExprKind::Index { base, indices } = &e.kind else {
            return self.expr(e)?.view();
        };
        let base = self.expr(base)?.view()?;
        let mut out = View {
            shape: Vec::new(),
            strides: Vec::new(),
            ..base.clone()
        };
        for axis in 0..base.shape.len() {
            let parent = self.extent(base.shape[axis]);
            let (start, end, keep) = match indices.get(axis) {
                Some(Index::Point(e)) => {
                    let start = self.index_value(e)?;
                    let valid = self
                        .builder
                        .ins()
                        .icmp(IntCC::UnsignedLessThan, start, parent);
                    self.require(valid);
                    (start, start, false)
                }
                Some(Index::Slice { start, end }) => {
                    let start = match start {
                        Some(e) => self.index_value(e)?,
                        None => self.builder.ins().iconst(types::I64, 0),
                    };
                    let end = match end {
                        Some(e) => self.index_value(e)?,
                        None => parent,
                    };
                    let ordered =
                        self.builder
                            .ins()
                            .icmp(IntCC::UnsignedLessThanOrEqual, start, end);
                    let bounded =
                        self.builder
                            .ins()
                            .icmp(IntCC::UnsignedLessThanOrEqual, end, parent);
                    let valid = self.builder.ins().band(ordered, bounded);
                    self.require(valid);
                    (start, end, true)
                }
                None => (self.builder.ins().iconst(types::I64, 0), parent, true),
            };
            let delta = self.builder.ins().imul_imm(start, base.strides[axis]);
            out.offset = self.builder.ins().iadd(out.offset, delta);
            if keep {
                let logical = self.builder.ins().isub(end, start);
                let symbolic =
                    e.ty.shaped()
                        .and_then(|sh| sh.shape.get(out.shape.len()))
                        .ok_or("view shape mismatch")?;
                let dimension =
                    if let Some(capacity) = symbolic.eval(&|p| self.constants.get(p).copied()) {
                        if capacity < 0 || capacity > base.shape[axis].capacity {
                            return Err("slice exceeds parent capacity".into());
                        }
                        let valid = self.builder.ins().icmp_imm(IntCC::Equal, logical, capacity);
                        self.require(valid);
                        Dimension::fixed(capacity)
                    } else {
                        let dim = Dimension {
                            capacity: base.shape[axis].capacity,
                            extent: Some(logical),
                        };
                        if let [Atom::Param(name)] = symbolic.atoms().as_slice() {
                            if *symbolic == Sym::param(name) {
                                self.dimensions.insert(name.clone(), dim);
                            }
                        }
                        dim
                    };
                out.shape.push(dimension);
                out.strides.push(base.strides[axis]);
            }
        }
        Ok(out)
    }
    fn require(&mut self, valid: Value) {
        self.execution.validity_guards.push(valid);
        let proceed = self.block_with(self.multiplicity.clone());
        let fail = self.block_with(Arc::new(Multiplicity::Constant(0)));
        self.builder.ins().brif(valid, proceed, &[], fail, &[]);
        self.switch_to_block(fail);
        let status = self.builder.ins().iconst(types::I32, 1);
        self.builder.ins().return_(&[status]);
        self.switch_to_block(proceed);
    }
    fn binary(&mut self, op: BinaryOp, a: Value, b: Value, d: DType) -> Result<Value, String> {
        let float = d.is_float();
        let value = match op {
            BinaryOp::Add => {
                if float {
                    self.builder.ins().fadd(a, b)
                } else {
                    self.builder.ins().iadd(a, b)
                }
            }
            BinaryOp::Sub => {
                if float {
                    self.builder.ins().fsub(a, b)
                } else {
                    self.builder.ins().isub(a, b)
                }
            }
            BinaryOp::Mul => {
                if float {
                    self.builder.ins().fmul(a, b)
                } else {
                    self.builder.ins().imul(a, b)
                }
            }
            BinaryOp::Shl | BinaryOp::Shr if d.is_int()=>{
                let valid=self.builder.ins().icmp_imm(IntCC::UnsignedLessThan,b,32);
                self.require(valid);
                if op==BinaryOp::Shl {self.builder.ins().ishl(a,b)}
                else if d==DType::U32 {self.builder.ins().ushr(a,b)}
                else {self.builder.ins().sshr(a,b)}
            },
            BinaryOp::Div if float => self.builder.ins().fdiv(a, b),
            BinaryOp::Div | BinaryOp::Rem if d.is_int() => {
                let nonzero = self.builder.ins().icmp_imm(IntCC::NotEqual, b, 0);
                self.require(nonzero);
                if d == DType::U32 {
                    if op == BinaryOp::Div {
                        self.builder.ins().udiv(a, b)
                    } else {
                        self.builder.ins().urem(a, b)
                    }
                } else {
                    let minimum = self
                        .builder
                        .ins()
                        .icmp_imm(IntCC::Equal, a, i64::from(i32::MIN));
                    let minus_one = self.builder.ins().icmp_imm(IntCC::Equal, b, -1);
                    let overflow = self.builder.ins().band(minimum, minus_one);
                    let valid = self.builder.ins().icmp_imm(IntCC::Equal, overflow, 0);
                    self.require(valid);
                    let quotient = self.builder.ins().sdiv(a, b);
                    let remainder = self.builder.ins().srem(a, b);
                    let negative = self
                        .builder
                        .ins()
                        .icmp_imm(IntCC::SignedLessThan, remainder, 0);
                    let divisor_negative = self.builder.ins().icmp_imm(IntCC::SignedLessThan, b, 0);
                    if op == BinaryOp::Div {
                        let one = self.builder.ins().iconst(types::I32, 1);
                        let minus_one = self.builder.ins().iconst(types::I32, -1);
                        let correction =
                            self.builder.ins().select(divisor_negative, one, minus_one);
                        let adjusted = self.builder.ins().iadd(quotient, correction);
                        self.builder.ins().select(negative, adjusted, quotient)
                    } else {
                        let negated = self.builder.ins().ineg(b);
                        let magnitude = self.builder.ins().select(divisor_negative, negated, b);
                        let adjusted = self.builder.ins().iadd(remainder, magnitude);
                        self.builder.ins().select(negative, adjusted, remainder)
                    }
                }
            }
            BinaryOp::Eq
            | BinaryOp::Ne
            | BinaryOp::Lt
            | BinaryOp::Le
            | BinaryOp::Gt
            | BinaryOp::Ge => {
                if float {
                    let cc = match op {
                        BinaryOp::Eq => FloatCC::Equal,
                        BinaryOp::Ne => FloatCC::NotEqual,
                        BinaryOp::Lt => FloatCC::LessThan,
                        BinaryOp::Le => FloatCC::LessThanOrEqual,
                        BinaryOp::Gt => FloatCC::GreaterThan,
                        _ => FloatCC::GreaterThanOrEqual,
                    };
                    self.builder.ins().fcmp(cc, a, b)
                } else {
                    let cc = match (op, d == DType::U32) {
                        (BinaryOp::Eq, _) => IntCC::Equal,
                        (BinaryOp::Ne, _) => IntCC::NotEqual,
                        (BinaryOp::Lt, false) => IntCC::SignedLessThan,
                        (BinaryOp::Le, false) => IntCC::SignedLessThanOrEqual,
                        (BinaryOp::Gt, false) => IntCC::SignedGreaterThan,
                        (BinaryOp::Ge, false) => IntCC::SignedGreaterThanOrEqual,
                        (BinaryOp::Lt, true) => IntCC::UnsignedLessThan,
                        (BinaryOp::Le, true) => IntCC::UnsignedLessThanOrEqual,
                        (BinaryOp::Gt, true) => IntCC::UnsignedGreaterThan,
                        _ => IntCC::UnsignedGreaterThanOrEqual,
                    };
                    self.builder.ins().icmp(cc, a, b)
                }
            }
            BinaryOp::And | BinaryOp::BitAnd => self.builder.ins().band(a, b),
            BinaryOp::Or | BinaryOp::BitOr => self.builder.ins().bor(a, b),
            BinaryOp::BitXor => self.builder.ins().bxor(a, b),
            _ => {
                return Err(format!(
                    "scalar realization operator {} is not implemented for {}",
                    op.text(),
                    d.name()
                ))
            }
        };
        Ok(
            if matches!(d, DType::BF16 | DType::F16)
                && !matches!(
                    op,
                    BinaryOp::Eq
                        | BinaryOp::Ne
                        | BinaryOp::Lt
                        | BinaryOp::Le
                        | BinaryOp::Gt
                        | BinaryOp::Ge
                )
            {
                self.publish(value, d)
            } else {
                value
            },
        )
    }
    fn expr(&mut self, e: &Expr) -> Result<ResultValue, String> {
        use ResultValue as R;
        match &e.kind {
            ExprKind::Int(n) => {
                let d = match e.ty {
                    Ty::Scalar(d) => d,
                    _ => DType::I32,
                };
                Ok(R::Scalar(self.builder.ins().iconst(scalar_type(d)?, *n), d))
            }
            ExprKind::Float(n) => {
                let d = match e.ty {
                    Ty::Scalar(d) => d,
                    _ => DType::F32,
                };
                let value = self.builder.ins().f32const(*n as f32);
                Ok(R::Scalar(self.cast(value, DType::F32, d)?, d))
            }
            ExprKind::Bool(v) => Ok(R::Scalar(
                self.builder.ins().iconst(types::I8, i64::from(*v)),
                DType::Bool,
            )),
            ExprKind::ShapeParam(_) => {
                let v = self.sym(e.sym.as_ref().ok_or("missing shape expression")?)?;
                Ok(R::Scalar(
                    self.builder.ins().ireduce(types::I32, v),
                    DType::I32,
                ))
            }
            ExprKind::Var(id) => match self.bindings.get(id).cloned() {
                Some(Binding::Scalar(var, d)) => Ok(R::Scalar(self.builder.use_var(var), d)),
                Some(Binding::View(view)) => Ok(R::View(view)),
                None => {
                    if let VarKind::Index(atom) = &self.lowered.vars[*id].kind {
                        let v = self.atom(atom)?;
                        Ok(R::Scalar(
                            self.builder.ins().ireduce(types::I32, v),
                            DType::I32,
                        ))
                    } else {
                        Err(format!(
                            "unbound scalar realization local {}",
                            self.lowered.vars[*id].name
                        ))
                    }
                }
            },
            ExprKind::TileAlloc { shape, dtype } => {
                let shape = self.shape(shape)?;
                let Elem::Dtype(dtype) = dtype else {return Err("unresolved or packed local tile dtype".into())};
                Ok(R::View(self.tile(shape, *dtype)?))
            }
            ExprKind::Index { .. } => {
                let view = self.index_view(e)?;
                if let Ty::Scalar(dtype) = e.ty {
                    let (v, d) = self.read(&view, &[])?;
                    Ok(R::Scalar(self.cast(v, d, dtype)?, dtype))
                } else {
                    Ok(R::View(view))
                }
            }
            ExprKind::Transpose(base) => {
                let mut view = self.expr(base)?.view()?;
                if view.shape.len() != 2 {
                    return Err("scalar realization transpose requires rank two".into());
                }
                view.shape.swap(0, 1);
                view.strides.swap(0, 1);
                Ok(R::View(view))
            }
            ExprKind::Tuple(items) => Ok(R::Tuple(
                items
                    .iter()
                    .map(|e| self.expr(e))
                    .collect::<Result<_, _>>()?,
            )),
            ExprKind::Cast { dtype, expr } => {
                let (v, from) = self.expr(expr)?.scalar()?;
                Ok(R::Scalar(self.cast(v, from, *dtype)?, *dtype))
            }
            ExprKind::Binary { op, lhs, rhs } => {
                let (a, ad) = self.expr(lhs)?.scalar()?;
                let (b, bd) = self.expr(rhs)?.scalar()?;
                let d = if matches!(op,BinaryOp::Shl|BinaryOp::Shr) {
                    // The checked language permits any integer shift count;
                    // signedness and result type belong to the shifted value.
                    ad
                } else { DType::promote(ad, bd).ok_or("scalar realization incompatible binary types")? };
                let a = self.cast(a, ad, d)?;
                let b = self.cast(b, bd, d)?;
                let value = self.binary(*op, a, b, d)?;
                let result = if matches!(
                    op,
                    BinaryOp::Eq
                        | BinaryOp::Ne
                        | BinaryOp::Lt
                        | BinaryOp::Le
                        | BinaryOp::Gt
                        | BinaryOp::Ge
                ) {
                    DType::Bool
                } else {
                    d
                };
                Ok(R::Scalar(value, result))
            }
            ExprKind::Unary { op, expr } => {
                let (v, d) = self.expr(expr)?.scalar()?;
                let value = match op {
                    UnaryOp::Neg => {
                        if d.is_float() {
                            self.builder.ins().fneg(v)
                        } else {
                            self.builder.ins().ineg(v)
                        }
                    }
                    UnaryOp::Not => self.builder.ins().icmp_imm(IntCC::Equal, v, 0),
                    UnaryOp::BitNot=>self.builder.ins().bnot(v),
                };
                Ok(R::Scalar(value, d))
            }
            ExprKind::Builtin { name, args } => self.builtin(*name, args, e),
            ExprKind::Call { .. } => {
                Err("scalar realization input contains an uninlined call".into())
            }
            _ => Err("backend-specific expression has no scalar realization realization".into()),
        }
    }
    fn builtin(&mut self, name: Builtin, args: &[Expr], e: &Expr) -> Result<ResultValue, String> {
        use ResultValue as R;
        match name {
            Builtin::Reshape => {
                let mut view = self.expr(&args[0])?.view()?;
                let target =
                    self.shape(&e.ty.shaped().ok_or("reshape requires shaped result")?.shape)?;
                if view.shape.iter().chain(&target).any(|d| d.extent.is_some()) {
                    return Err("reshape requires statically resolved extents".into());
                }
                let source_shape = view.shape.iter().map(|d| d.capacity).collect::<Vec<_>>();
                let target_shape = target.iter().map(|d| d.capacity).collect::<Vec<_>>();
                view.strides = seismic_lang::layout::reshape_strides(
                    &source_shape,
                    &view.strides,
                    &target_shape,
                )?;
                view.shape = target;
                Ok(R::View(view))
            }
            Builtin::Load => {
                let input = self.expr(&args[0])?;
                match input {
                    R::View(view) => Ok(R::View(self.materialize(view)?)),
                    R::Tuple(items) => Ok(R::Tuple(
                        items
                            .into_iter()
                            .map(|v| self.materialize(v.view()?).map(R::View))
                            .collect::<Result<_, String>>()?,
                    )),
                    _ => Err("invalid scalar realization load".into()),
                }
            }
            Builtin::Store => {
                let source = self.expr(&args[0])?.view()?;
                let target = self.expr(&args[1])?.view()?;
                self.copy(&source, &target)?;
                Ok(R::Void)
            }
            Builtin::Extent => {
                let n = self.sym(e.sym.as_ref().ok_or("unresolved extent")?)?;
                Ok(R::Scalar(
                    self.builder.ins().ireduce(types::I32, n),
                    DType::I32,
                ))
            }
            Builtin::Reduce => self.reduce(args, e),
            Builtin::Atomic => Err("scalar realization atomic effects are not implemented".into()),
            _ => {
                let dtype = match e.ty {
                    Ty::Scalar(d) => d,
                    _ => {
                        return Err(
                            "scalar realization arithmetic builtin needs scalar result".into()
                        )
                    }
                };
                let mut values = Vec::new();
                for arg in args {
                    let (v, d) = self.expr(arg)?.scalar()?;
                    values.push(self.cast(v, d, dtype)?);
                }
                if !dtype.is_float() {
                    return Err(
                        "scalar realization integer arithmetic builtin is not implemented".into(),
                    );
                }
                let value = match name {
                    Builtin::Fma => self.builder.ins().fma(values[0], values[1], values[2]),
                    Builtin::Sqrt => self.builder.ins().sqrt(values[0]),
                    Builtin::Rsqrt => {
                        let root = self.builder.ins().sqrt(values[0]);
                        let one = self.builder.ins().f32const(1.0);
                        self.builder.ins().fdiv(one, root)
                    }
                    Builtin::Abs => self.builder.ins().fabs(values[0]),
                    Builtin::Exp
                    | Builtin::ExpFast
                    | Builtin::Log
                    | Builtin::Sin
                    | Builtin::Cos => {
                        let operation = match name {
                            Builtin::Exp => MathFunction::Exp,
                            Builtin::ExpFast => MathFunction::ExpFast,
                            Builtin::Log => MathFunction::Log,
                            Builtin::Sin => MathFunction::Sin,
                            _ => MathFunction::Cos,
                        };
                        let mut signature =
                            ir::Signature::new(self.builder.func.signature.call_conv);
                        signature.params.push(AbiParam::new(types::F32));
                        signature.returns.push(AbiParam::new(types::F32));
                        let signature = self.builder.import_signature(signature);
                        let external = self.builder.func.declare_imported_user_function(
                            ir::UserExternalName {
                                namespace: 0,
                                index: operation as u32,
                            },
                        );
                        let reference = self.builder.import_function(ir::ExtFuncData {
                            name: ir::ExternalName::user(external),
                            signature,
                            colocated: false,
                        });
                        self.imports.push((reference, operation));
                        let call = self.builder.ins().call(reference, &values);
                        self.builder.inst_results(call)[0]
                    }
                    Builtin::Max | Builtin::Min => {
                        self.extremum(values[0], values[1], dtype, name == Builtin::Max)?
                    }
                    _ => return Err("scalar realization builtin not implemented".into()),
                };
                Ok(R::Scalar(self.publish(value, dtype), dtype))
            }
        }
    }
    fn extremum(
        &mut self,
        a: Value,
        b: Value,
        dtype: DType,
        maximum: bool,
    ) -> Result<Value, String> {
        if dtype.is_float() {
            let value = if maximum {
                self.builder.ins().fmax(a, b)
            } else {
                self.builder.ins().fmin(a, b)
            };
            let anan = self.builder.ins().fcmp(FloatCC::Unordered, a, a);
            let bnan = self.builder.ins().fcmp(FloatCC::Unordered, b, b);
            let value = self.builder.ins().select(anan, b, value);
            Ok(self.builder.ins().select(bnan, a, value))
        } else if dtype.is_int() {
            let cc = match (dtype == DType::U32, maximum) {
                (true, true) => IntCC::UnsignedGreaterThan,
                (true, false) => IntCC::UnsignedLessThan,
                (false, true) => IntCC::SignedGreaterThan,
                (false, false) => IntCC::SignedLessThan,
            };
            let choose_a = self.builder.ins().icmp(cc, a, b);
            Ok(self.builder.ins().select(choose_a, a, b))
        } else {
            Err("extremum requires numeric operands".into())
        }
    }
    fn reduce(&mut self, args: &[Expr], e: &Expr) -> Result<ResultValue, String> {
        let source = self.expr(&args[0])?.view()?;
        let axis = args[1]
            .sym
            .as_ref()
            .and_then(Sym::as_constant)
            .and_then(|n| usize::try_from(n).ok())
            .ok_or("scalar realization reduction axis unresolved")?;
        let operation = match args[2].kind {
            ExprKind::Int(0) => hir::ReduceOp::Sum,
            ExprKind::Int(1) => hir::ReduceOp::Max,
            ExprKind::Int(2) => hir::ReduceOp::Min,
            ExprKind::Int(3) => hir::ReduceOp::Argmax,
            _ => return Err("scalar realization reduction operation unresolved".into()),
        };
        let dtype = match &source.storage {
            Storage::Dense { dtype, .. } => *dtype,
            Storage::Packed { .. } => {
                return Err(
                    "scalar realization packed reduction requires an explicit decode".into(),
                )
            }
        };
        if !dtype.is_numeric() || (dtype.is_int() && operation == hir::ReduceOp::Sum) {
            return Err(
                "scalar realization reduction needs floating sum or numeric extrema".into(),
            );
        }
        let extent = *source
            .shape
            .get(axis)
            .ok_or("scalar realization reduction axis out of bounds")?;
        if operation == hir::ReduceOp::Argmax
            && (extent.extent.is_some()
                || !(1..=i64::from(i32::MAX) + 1).contains(&extent.capacity))
        {
            return Err("argmax requires a nonempty axis with i32-representable indices".into());
        }
        let output_dtype = if operation == hir::ReduceOp::Argmax {
            DType::I32
        } else {
            dtype
        };
        let mut output_shape = source.shape.clone();
        output_shape.remove(axis);
        let out = self.tile(output_shape.clone(), output_dtype)?;
        self.each(&output_shape, |s, indices| {
            let accumulator = s.builder.declare_var(scalar_type(dtype)?);
            let initial = if dtype.is_float() {
                let value = match operation {
                    hir::ReduceOp::Sum => 0.0,
                    hir::ReduceOp::Min => f32::INFINITY,
                    _ => f32::NEG_INFINITY,
                };
                s.builder.ins().f32const(value)
            } else {
                let value = match (dtype, operation) {
                    (DType::I32, hir::ReduceOp::Min) => i64::from(i32::MAX),
                    (DType::I32, _) => i64::from(i32::MIN),
                    (DType::U32, hir::ReduceOp::Min) => i64::from(u32::MAX),
                    _ => 0,
                };
                s.builder.ins().iconst(types::I32, value)
            };
            s.builder.def_var(accumulator, initial);
            let winner = if operation == hir::ReduceOp::Argmax {
                let var = s.builder.declare_var(types::I32);
                let zero = s.builder.ins().iconst(types::I32, 0);
                s.builder.def_var(var, zero);
                Some(var)
            } else {
                None
            };
            let lo = s.builder.ins().iconst(types::I64, 0);
            let hi = s.extent(extent);
            s.loop_range(lo, hi, |s, k| {
                let mut input = indices.to_vec();
                input.insert(axis, k);
                let value = s.read(&source, &input)?.0;
                let acc = s.builder.use_var(accumulator);
                let next = match operation {
                    hir::ReduceOp::Sum => {
                        let sum = s.builder.ins().fadd(acc, value);
                        s.publish(sum, dtype)
                    }
                    hir::ReduceOp::Max => s.extremum(acc, value, dtype, true)?,
                    hir::ReduceOp::Min => s.extremum(acc, value, dtype, false)?,
                    hir::ReduceOp::Argmax => {
                        // Strict improvement preserves the first index on ties;
                        // unordered floats never replace a valid winner.
                        let better = if dtype.is_float() {
                            s.builder.ins().fcmp(FloatCC::GreaterThan, value, acc)
                        } else {
                            s.builder.ins().icmp(
                                if dtype == DType::U32 {
                                    IntCC::UnsignedGreaterThan
                                } else {
                                    IntCC::SignedGreaterThan
                                },
                                value,
                                acc,
                            )
                        };
                        let var = winner.unwrap();
                        let old = s.builder.use_var(var);
                        let at = s.builder.ins().ireduce(types::I32, k);
                        let at = s.builder.ins().select(better, at, old);
                        s.builder.def_var(var, at);
                        s.builder.ins().select(better, value, acc)
                    }
                };
                s.builder.def_var(accumulator, next);
                Ok(())
            })?;
            let value = s.builder.use_var(winner.unwrap_or(accumulator));
            s.write(&out, indices, value, output_dtype)
        })?;
        if matches!(e.ty, Ty::Scalar(_)) {
            let (value, dtype) = self.read(&out, &[])?;
            Ok(ResultValue::Scalar(value, dtype))
        } else {
            Ok(ResultValue::View(out))
        }
    }
}
fn assign_binary(op: AssignOp) -> Result<BinaryOp, String> {
    match op {
        AssignOp::Add => Ok(BinaryOp::Add),
        AssignOp::Sub => Ok(BinaryOp::Sub),
        AssignOp::Mul => Ok(BinaryOp::Mul),
        AssignOp::Assign => Err("not a compound assignment".into()),
    }
}

fn value_kind(statement: &Stmt) -> Option<&ExprKind> {
    if let StmtKind::Assign { value, .. } = &statement.kind {
        Some(&value.kind)
    } else {
        None
    }
}
