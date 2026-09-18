use cranelift_codegen::ir::{
    self,
    condcodes::{FloatCC, IntCC},
    types, AbiParam, InstBuilder, MemFlags, Value,
};
use cranelift_frontend::{FunctionBuilder, Variable};
use seismic_lang::abi::ScalarParameter;
use seismic_lang::{
    ast::{AssignOp, BinaryOp, UnaryOp},
    ir::{Builtin, Expr, ExprKind, Index, ReduceOp, Stmt, StmtKind, VarId, VarKind},
    lowered_ir::LoweredIr,
    repr,
    sym::{Atom, Sym},
    types::{DType, Elem, Ty},
};
use seismic_realization::{
    execution::{ExecutionEvidence, MemoryObject, Multiplicity},
    BufferSpec, MathFunction,
};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

#[derive(Clone)]
enum Storage {
    /// Logical shape/layout retained after all element-data uses disappear.
    Geometry,
    Dense {
        pointer: Value,
        dtype: DType,
    },
    Packed {
        planes: Vec<Value>,
        name: String,
    },
    Coefficient { planes: Vec<Value>, name: String, bias: bool },
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
    publication: bool,
    offset: Value,
    shape: Vec<Dimension>,
    strides: Vec<i64>,
}
#[derive(Clone)]
enum Binding {
    Scalar(Variable, DType),
    View(View),
    /// Owning packet planes retain their allocation while the logical packet
    /// prefix is an SSA variable carried across branches and loop iterations.
    Packed { view: View, offset: Variable },
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
    pub imports: Vec<(ir::FuncRef, MathFunction)>,
    pub backend_calls: Vec<(ir::FuncRef, seismic_realization::ParticipantOperation)>,
    participation: seismic_realization::dispatch::Participation,
    lowered: &'b LoweredIr,
    data_variables: HashSet<VarId>,
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
        lowered: &'b LoweredIr,
        mut builder: FunctionBuilder<'a>,
        buffers: Value,
        scalars: Value,
        scratch: Value,
        participation: seismic_realization::dispatch::Participation,
    ) -> Result<Self, String> {
        let zero = builder.ins().iconst(types::I64, 0);
        let entry = builder.current_block().unwrap();
        let mut s = Self {
            builder,
            imports: Vec::new(),
            backend_calls: Vec::new(), participation,
            lowered,
            data_variables: seismic_lang::demand::data_variables(&lowered.body),
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
                                .is_none_or(|n| n.capacity % i64::from(r.storage_group()) != 0)
                            {
                                return Err(
                                    "scalar realization packed parameter requires complete groups"
                                        .into(),
                                );
                            }
                            let mut planes = Vec::new();
                            for plane in r.planes() {
                                let n = plane.storage_elements(count as u64).and_then(|n| i64::try_from(n).ok()).ok_or("packed plane extent overflow")?;
                                planes.push(s.parameter(buffers, name, plane.name, n, i64::from(plane.dtype().bytes()))?);
                            }
                            Storage::Packed { planes, name: name_.clone() }
                        }
                        Elem::Param(_) => {
                            return Err("unbound scalar realization element parameter".into())
                        }
                    };
                    s.bindings.insert(
                        id,
                        Binding::View(View {
                            storage, publication: true,
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
        // Lexical loop locals cannot supply pointers, indices or dynamic shape
        // values after an empty iteration domain. Existing scalar Variables and
        // allocated tile storage retain their writes through SSA/memory.
        let bindings=self.bindings.clone();
        let indices=self.indices.clone();
        let constants=self.constants.clone();
        let dimensions=self.dimensions.clone();
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
        self.bindings=bindings;
        self.indices=indices;
        self.constants=constants;
        self.dimensions=dimensions;
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
            storage: Storage::Dense { pointer, dtype }, publication: false,
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
            Storage::Geometry => Err("element read from geometry-only value".into()),
            Storage::Dense { pointer, dtype } => self.read_dense(*pointer, offset, *dtype),
            Storage::Packed { planes, name } => {
                let r = repr::lookup(name).ok_or("unknown packed representation")?;
                let code = self.read_bits(planes[0], offset, r.bits)?;
                let code = self.decode_code(code, r.bits, &r.code)?;
                let scale = self.read_coefficient(r, planes, offset, false)?;
                let bias = self.read_coefficient(r, planes, offset, true)?;
                Ok((self.builder.ins().fma(code, scale, bias), DType::F32))
            }
            Storage::Coefficient { planes, name, bias } => {
                let r = repr::lookup(name).ok_or("unknown packed representation")?;
                let logical = self.builder.ins().imul_imm(offset, i64::from(r.group));
                Ok((self.read_coefficient(r, planes, logical, *bias)?, r.coefficient_dtype()))
            }
        }
    }
    fn read_bits(&mut self, pointer: Value, entry: Value, bits: u32) -> Result<Value, String> {
        let bit = self.builder.ins().imul_imm(entry, i64::from(bits));
        let word_index = self.builder.ins().udiv_imm(bit, 32);
        let shift = self.builder.ins().urem_imm(bit, 32);
        let shift = self.builder.ins().ireduce(types::I32, shift);
        let (word, _) = self.read_dense(pointer, word_index, DType::U32)?;
        let mut code = self.builder.ins().ushr(word, shift);
        if 32 % bits != 0 {
            let crossing = self.builder.ins().icmp_imm(IntCC::UnsignedGreaterThan, shift, i64::from(32 - bits));
            let next = self.builder.ins().iadd_imm(word_index, 1);
            // A noncrossing final code must never read past its last stored word.
            let next = self.builder.ins().select(crossing, next, word_index);
            let (high, _) = self.read_dense(pointer, next, DType::U32)?;
            let left = self.builder.ins().irsub_imm(shift, 32);
            let left = self.builder.ins().band_imm(left, 31);
            let high = self.builder.ins().ishl(high, left);
            let zero = self.builder.ins().iconst(types::I32, 0);
            let high = self.builder.ins().select(crossing, high, zero);
            code = self.builder.ins().bor(code, high);
        }
        Ok(self.builder.ins().band_imm(code, (1i64 << bits) - 1))
    }
    fn decode_code(&mut self, code: Value, bits: u32, interpretation: &repr::CodeInterpretation) -> Result<Value, String> {
        let decoded = match interpretation {
            repr::CodeInterpretation::Unsigned => return Ok(self.builder.ins().fcvt_from_uint(types::F32, code)),
            repr::CodeInterpretation::Offset(zero) => self.builder.ins().iadd_imm(code, -i64::from(*zero)),
            repr::CodeInterpretation::TwosComplement => {
                let code = self.builder.ins().ishl_imm(code, i64::from(32 - bits));
                self.builder.ins().sshr_imm(code, i64::from(32 - bits))
            }
            repr::CodeInterpretation::Table(table) => {
                let mut decoded = self.builder.ins().iconst(types::I32, i64::from(*table.last().ok_or("empty code table")?));
                for (i, value) in table.iter().enumerate().rev().skip(1) {
                    let equal = self.builder.ins().icmp_imm(IntCC::Equal, code, i as i64);
                    let value = self.builder.ins().iconst(types::I32, i64::from(*value));
                    decoded = self.builder.ins().select(equal, value, decoded);
                }
                decoded
            }
        };
        Ok(self.builder.ins().fcvt_from_sint(types::F32, decoded))
    }
    fn read_coefficient(&mut self, r: &repr::Repr, planes: &[Value], logical: Value, bias: bool) -> Result<Value, String> {
        Ok(match r.coefficient(bias) {
            None => self.builder.ins().f32const(0.0),
            Some(repr::Coefficient::Direct { plane }) => {
                let entry = self.builder.ins().udiv_imm(logical, i64::from(plane.group));
                self.read_dense(planes[r.plane_index(plane.name).unwrap()], entry, plane.dtype())?.0
            }
            Some(repr::Coefficient::Product { factor, coefficients, field, sign }) => {
                let group = self.builder.ins().udiv_imm(logical, i64::from(coefficients.group));
                let entry = self.builder.ins().imul_imm(group, i64::from(coefficients.fields));
                let entry = self.builder.ins().iadd_imm(entry, i64::from(field));
                let repr::PlaneEncoding::Packed { bits, interpretation } = &coefficients.encoding else { return Err("hierarchical coefficients must be packed".into()); };
                let code = self.read_bits(planes[r.plane_index(coefficients.name).unwrap()], entry, *bits)?;
                let code = self.decode_code(code, *bits, interpretation)?;
                let entry = self.builder.ins().udiv_imm(logical, i64::from(factor.group));
                let factor = self.read_dense(planes[r.plane_index(factor.name).unwrap()], entry, factor.dtype())?.0;
                let value = self.builder.ins().fmul(factor, code);
                if sign == -1 { self.builder.ins().fneg(value) } else if sign == 1 { value } else { return Err("unsupported coefficient sign".into()); }
            }
        })
    }
    fn packet_accessor(&mut self, source: View, name: &str, shape: Vec<Dimension>) -> Result<View, String> {
        let Storage::Packed { planes, name: representation } = &source.storage else { return Err("packet accessor requires retained packed storage".into()); };
        let r = repr::lookup(representation).ok_or("unknown packed representation")?;
        let logical_coefficient = name == "scale" || name == "bias";
        let (numerator, denominator, storage) = if logical_coefficient {
            (1i64, i64::from(r.group), Storage::Coefficient { planes: planes.clone(), name: representation.clone(), bias: name == "bias" })
        } else {
            let plane = r.plane(name).ok_or("unknown packed plane")?;
            (i64::from(plane.fields) * i64::from(plane.entry_bits()), i64::from(plane.group) * i64::from(plane.dtype().bytes()) * 8,
             Storage::Dense { pointer: planes[r.plane_index(name).unwrap()], dtype: plane.dtype() })
        };
        let scaled = self.builder.ins().imul_imm(source.offset, numerator);
        let rem = self.builder.ins().urem_imm(scaled, denominator);
        let valid = self.builder.ins().icmp_imm(IntCC::Equal, rem, 0);
        self.require(valid);
        let offset = self.builder.ins().udiv_imm(scaled, denominator);
        let mut strides = Vec::new();
        for (axis, stride) in source.strides.iter().enumerate() {
            if axis + 1 == source.strides.len() { strides.push(1); }
            else {
                let scaled = stride.checked_mul(numerator).ok_or("packet accessor stride overflow")?;
                if scaled % denominator != 0 { return Err("packet accessor outer stride is not plane aligned".into()); }
                strides.push(scaled / denominator);
            }
        }
        Ok(View { storage, offset, shape, strides, publication: false })
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
        if view.publication && self.participation.lanes() > 1 {
            let lane = self.lane_index()?;
            let leader = self.builder.ins().icmp_imm(IntCC::Equal, lane, 0);
            let write = self.block_with(self.multiplicity.clone());
            let done = self.block_with(self.multiplicity.clone());
            self.builder.ins().brif(leader, write, &[], done, &[]);
            self.switch_to_block(write);
            self.builder.ins().store(MemFlags::new(), value, ptr, 0);
            self.builder.ins().jump(done, &[]);
            self.switch_to_block(done);
        } else { self.builder.ins().store(MemFlags::new(), value, ptr, 0); }
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
    fn bind_tile(&mut self, id: VarId, view: View) {
        let binding = if matches!(view.storage, Storage::Packed { .. }) {
            let offset = self.builder.declare_var(types::I64);
            self.builder.def_var(offset, view.offset);
            Binding::Packed { view, offset }
        } else {
            Binding::View(view)
        };
        self.bindings.insert(id, binding);
    }
    /// Copy representation-owned packets into an owning snapshot without
    /// decoding or re-encoding coefficients/codes. The returned prefix belongs
    /// to the copied logical value and must travel with the destination binding.
    fn copy_packed(&mut self, source: &View, target: &View) -> Result<Value, String> {
        self.equal_shape(&source.shape, &target.shape)?;
        let (
            Storage::Packed { planes: from, name },
            Storage::Packed {
                planes: to,
                name: target_name,
            },
        ) = (&source.storage, &target.storage)
        else {
            return Err("packet copy requires packed storage".into());
        };
        if name != target_name {
            return Err("packet copy representation mismatch".into());
        }
        let r = repr::lookup(name).ok_or("unknown packed representation")?;
        let group = i64::from(r.storage_group());
        for view in [source, target] {
            if view.strides.last().is_some_and(|&stride| stride != 1)
                || view
                    .strides
                    .iter()
                    .take(view.strides.len().saturating_sub(1))
                    .any(|&stride| stride % group != 0)
            {
                return Err("packed snapshot requires a retained packed row layout".into());
            }
        }
        let prefix = self.builder.ins().urem_imm(source.offset, group);
        let target_prefix = self.builder.ins().urem_imm(target.offset, group);
        let width = self.extent(source.shape.last().copied().unwrap_or(Dimension::fixed(1)));
        let used = self.builder.ins().iadd(prefix, width);
        let zero = self.builder.ins().iconst(types::I64, 0);
        let nonempty = self
            .builder
            .ins()
            .icmp_imm(IntCC::UnsignedGreaterThan, width, 0);
        let descriptors = r.planes();
        self.each(
            &source.shape[..source.shape.len().saturating_sub(1)],
            |s, indices| {
                let mut source_origin = s.builder.ins().isub(source.offset, prefix);
                let mut target_origin = s.builder.ins().isub(target.offset, target_prefix);
                for ((&index, &source_stride), &target_stride) in
                    indices.iter().zip(&source.strides).zip(&target.strides)
                {
                    let term = s.builder.ins().imul_imm(index, source_stride);
                    source_origin = s.builder.ins().iadd(source_origin, term);
                    let term = s.builder.ins().imul_imm(index, target_stride);
                    target_origin = s.builder.ins().iadd(target_origin, term);
                }
                for ((plane, &from), &to) in descriptors.iter().zip(from).zip(to) {
                    let entry_bits = i64::from(plane.fields) * i64::from(plane.entry_bits());
                    let storage_bits = i64::from(plane.dtype().bytes()) * 8;
                    let address_index = |s: &mut Self, origin| {
                        let first = s.builder.ins().udiv_imm(origin, i64::from(plane.group));
                        let first = s.builder.ins().imul_imm(first, entry_bits);
                        s.builder.ins().udiv_imm(first, storage_bits)
                    };
                    let first = address_index(s, source_origin);
                    let destination = address_index(s, target_origin);
                    let count = s.builder.ins().iadd_imm(used, i64::from(plane.group) - 1);
                    let count = s.builder.ins().udiv_imm(count, i64::from(plane.group));
                    let count = s.builder.ins().imul_imm(count, entry_bits);
                    let count = s.builder.ins().iadd_imm(count, storage_bits - 1);
                    let count = s.builder.ins().udiv_imm(count, storage_bits);
                    let count = s.builder.ins().select(nonempty, count, zero);
                    s.loop_range(zero, count, |s, index| {
                        let source_index = s.builder.ins().iadd(first, index);
                        let destination_index = s.builder.ins().iadd(destination, index);
                        let bytes = plane.dtype().bytes();
                        let source_address = s.address(from, source_index, bytes);
                        let destination_address = s.address(to, destination_index, bytes);
                        let raw_type = match bytes {
                            1 => types::I8,
                            2 => types::I16,
                            4 => types::I32,
                            _ => return Err("unsupported packed plane storage width".into()),
                        };
                        let raw =
                            s.builder
                                .ins()
                                .load(raw_type, MemFlags::new(), source_address, 0);
                        s.builder
                            .ins()
                            .store(MemFlags::new(), raw, destination_address, 0);
                        Ok(())
                    })?;
                }
                Ok(())
            },
        )?;
        Ok(prefix)
    }
    fn materialize(&mut self, source: View) -> Result<View, String> {
        if let Storage::Packed { name, .. } = &source.storage {
            let r = repr::lookup(name).ok_or("unknown packed representation")?;
            let capacities = source
                .shape
                .iter()
                .map(|d| u64::try_from(d.capacity).map_err(|_| "negative packed shape"))
                .collect::<Result<Vec<_>, _>>()?;
            let layout = r
                .snapshot_layout(&capacities)
                .ok_or("packed snapshot layout overflow")?;
            let mut planes = Vec::new();
            for plane in layout.planes {
                let count =
                    i64::try_from(plane.elements).map_err(|_| "packed plane capacity overflow")?;
                let allocation = self.tile(vec![Dimension::fixed(count)], plane.plane.dtype())?;
                let Storage::Dense { pointer, .. } = allocation.storage else {
                    unreachable!()
                };
                planes.push(pointer);
            }
            let packed_strides = layout
                .strides
                .into_iter()
                .map(|n| i64::try_from(n).map_err(|_| "packed row stride overflow"))
                .collect::<Result<Vec<_>, _>>()?;
            let mut target = View {
                storage: Storage::Packed {
                    planes,
                    name: name.clone(),
                },
                publication: false,
                offset: self.builder.ins().iconst(types::I64, 0),
                shape: source.shape.clone(),
                strides: packed_strides,
            };
            target.offset = self.copy_packed(&source, &target)?;
            return Ok(target);
        }
        let dtype = match &source.storage {
            Storage::Geometry => return Err("cannot materialize geometry-only value data".into()),
            Storage::Dense { dtype, .. } => *dtype,
            Storage::Coefficient { name, .. } => repr::lookup(name)
                .ok_or("unknown representation")?
                .coefficient_dtype(),
            Storage::Packed { .. } => unreachable!(),
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
            StmtKind::Reduction(_) => Err("scalar emission requires materialized reduction choices".into()),
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
                let tile = self.geometry(tile)?;
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
                domain,
                offset,
                vars,
                views,
                axes,
                piece,
                body,
                capacity,
                modes,
            } => {
                let modes = modes.as_ref().filter(|m| m.len() == vars.len()).ok_or("unresolved stream loads reached scalar emission")?;
                if vars.len()!=views.len() || axes.len()!=views.len() {return Err("stream transfer binding geometry mismatch".into());}
                let geometry=self.geometry(&domain.view)?;
                let dimension=*geometry.shape.get(domain.axis).ok_or("invalid stream domain axis")?;
                let sources = views
                    .iter()
                    .map(|expr| self.expr(expr)?.view())
                    .collect::<Result<Vec<_>, _>>()?;
                for (view,&axis) in sources.iter().zip(axes) {
                    self.equal_shape(
                        &[dimension],
                        &[*view.shape.get(axis).ok_or("invalid stream axis")?],
                    )?;
                }
                let Atom::Param(name) = piece else {
                    return Err("unresolved stream piece".into());
                };
                if dimension.capacity == 0 {
                    return Ok(());
                }
                let old_indices = self.indices.clone();
                let old_dimensions = self.dimensions.clone();
                let old_constants = self.constants.clone();
                let old_bindings = self.bindings.clone();
                let emit_piece =
                    |s: &mut Self, start: Value, dim: Dimension| -> Result<(), String> {
                        if let Some(variable)=offset {s.bind_index(*variable,start)?;}
                        if dim.extent.is_none() {
                            s.constants.insert(name.clone(), dim.capacity);
                        } else {
                            s.constants.remove(name);
                        }
                        s.dimensions.insert(name.clone(), dim);
                        for (((var, source), mode), &axis) in vars.iter().zip(&sources).zip(modes).zip(axes) {
                            let mut view = source.clone();
                            view.shape[axis] = dim;
                            let delta = s.builder.ins().imul_imm(start, view.strides[axis]);
                            view.offset = s.builder.ins().iadd(view.offset, delta);
                            let tile = if *mode == seismic_lang::ir::LoadMode::Borrow { view } else { s.materialize(view)? };
                            if *mode == seismic_lang::ir::LoadMode::Borrow { s.bindings.insert(*var,Binding::View(tile)); } else { s.bind_tile(*var,tile); }
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
                self.indices = old_indices;
                self.dimensions = old_dimensions;
                self.constants = old_constants;
                Ok(())
            }
            StmtKind::Assign { target, op, value } => {
                if let ExprKind::Var(variable) = target.kind {
                    if matches!(target.ty, Ty::Tile(_)) && !self.data_variables.contains(&variable) {
                        if *op != AssignOp::Assign {
                            return Err("compound geometry-only tile assignment".into());
                        }
                        let source = self.geometry(value)?;
                        if let Some(Binding::View(destination)) = self.bindings.get(&variable).cloned() {
                            self.equal_shape(&destination.shape, &source.shape)?;
                        } else {
                            let geometry = self.geometry_snapshot(source.shape)?;
                            self.bind_tile(variable, geometry);
                        }
                        return Ok(());
                    }
                }
                if let (ExprKind::Var(var), ExprKind::Load { view, mode: seismic_lang::ir::LoadMode::Borrow }) = (&target.kind, &value.kind) {
                    if *op != AssignOp::Assign || self.bindings.contains_key(var) {
                        return Err("borrowed load must define fresh tile storage".into());
                    }
                    let view = self.expr(view)?.view()?;
                    self.bindings.insert(*var, Binding::View(view));
                    return Ok(());
                }
                let value = self.expr(value)?;
                match &target.kind {
                    ExprKind::Var(id) => match value {
                        ResultValue::View(view) => {
                            if *op != AssignOp::Assign {
                                return Err("compound view assignment".into());
                            }
                            if matches!(target.ty, Ty::Tile(_)) {
                                if let Some(binding @ (Binding::View(_) | Binding::Packed { .. })) = self.bindings.get(id).cloned() {
                                    let (destination, offset) = match binding {
                                        Binding::View(view) => (view, None),
                                        Binding::Packed { mut view, offset } => {view.offset=self.builder.use_var(offset);(view,Some(offset))},
                                        _ => unreachable!(),
                                    };
                                    if matches!(value_kind(s), Some(ExprKind::TileAlloc { .. })) {
                                        self.equal_shape(&destination.shape, &view.shape)?;
                                        // The existing variable owns storage. Allocation does
                                        // not publish a value; subsequent checked writes initialize it.
                                        return Ok(());
                                    }
                                    // Tile assignment is a value copy, matching the interpreter.
                                    // Snapshot first handles self-transpose/slices without alias loss.
                                    let source = self.materialize(view)?;
                                    if let Some(offset) = offset {
                                        let prefix=self.copy_packed(&source,&destination)?;
                                        self.builder.def_var(offset,prefix);
                                        return Ok(());
                                    }
                                    return self.copy(&source, &destination);
                                }
                                let owns_storage = matches!(
                                    value_kind(s),
                                    Some(
                                        ExprKind::TileAlloc { .. }
                                            | ExprKind::Load { .. }
                                            | ExprKind::Builtin { name: Builtin::Reduce, .. }
                                    )
                                );
                                let view = if owns_storage {
                                    view
                                } else {
                                    // Every other view expression (including
                                    // reshape and physical-plane access) still
                                    // borrows its operand's storage. A tile
                                    // value binding must snapshot that view.
                                    self.materialize(view)?
                                };
                                self.bind_tile(*id, view);
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
            StmtKind::Lanes { var, extent, width, body } => {
                if self.participation.lanes() == 1 { return Err("lane iteration requires a subgroup realization".into()); }
                let mapping = seismic_realization::dispatch::Ownership::Cyclic { consecutive: u64::try_from(*width).map_err(|_| "negative lane width")? };
                let period = i64::try_from(mapping.period(u64::from(self.participation.lanes()))?).map_err(|_| "lane period overflow")?;
                let lane = self.lane_index()?;
                let lane = self.builder.ins().uextend(types::I64, lane);
                let offset = self.builder.ins().imul_imm(lane, *width);
                let extent = self.sym(extent)?;
                let valid_extent = self.builder.ins().icmp_imm(IntCC::UnsignedLessThanOrEqual, extent, i64::from(i32::MAX) + 1);
                self.require(valid_extent);
                let whole = self.builder.ins().udiv_imm(extent, period);
                let remainder = self.builder.ins().urem_imm(extent, period);
                let tail = self.builder.ins().icmp_imm(IntCC::NotEqual, remainder, 0);
                let tail = self.builder.ins().uextend(types::I64, tail);
                let rounds = self.builder.ins().iadd(whole, tail);
                let zero = self.builder.ins().iconst(types::I64, 0);
                self.loop_range(zero, rounds, |s, round| {
                    let base = s.builder.ins().imul_imm(round, period);
                    let base = s.builder.ins().iadd(base, offset);
                    let width = s.builder.ins().iconst(types::I64, *width);
                    s.loop_range(zero, width, |s, within| {
                        let index = s.builder.ins().iadd(base, within);
                        let valid = s.builder.ins().icmp(IntCC::UnsignedLessThan, index, extent);
                        let run = s.block_with(s.multiplicity.clone());
                        let done = s.block_with(s.multiplicity.clone());
                        s.builder.ins().brif(valid, run, &[], done, &[]);
                        s.switch_to_block(run);s.bind_index(*var,index)?;s.body(body)?;
                        s.builder.ins().jump(done,&[]);s.switch_to_block(done);Ok(())
                    })
                })
            }
        }
    }
    fn participant_call(&mut self, operation: seismic_realization::ParticipantOperation, args: &[Value], result: ir::Type) -> Value {
        let mut signature = ir::Signature::new(self.builder.func.signature.call_conv);
        for &value in args { signature.params.push(AbiParam::new(self.builder.func.dfg.value_type(value))); }
        signature.returns.push(AbiParam::new(result));
        let signature = self.builder.import_signature(signature);
        let external = self.builder.func.declare_imported_user_function(ir::UserExternalName { namespace: 1, index: self.backend_calls.len() as u32 });
        let reference = self.builder.import_function(ir::ExtFuncData { name: ir::ExternalName::user(external), signature, colocated: false });
        self.backend_calls.push((reference, operation));
        let call = self.builder.ins().call(reference, args);
        self.builder.inst_results(call)[0]
    }
    fn lane_index(&mut self) -> Result<Value, String> {
        if self.participation.lanes() == 1 { return Err("lane index requires subgroup participation".into()); }
        let lane = self.participant_call(seismic_realization::ParticipantOperation::LaneIndex, &[], types::I32);
        // Do not cache across arbitrary source blocks: a value first requested in
        // a conditional region need not dominate a later publication.
        Ok(lane)
    }
    fn index_value(&mut self, e: &Expr) -> Result<Value, String> {
        if seismic_lang::effects::can_substitute_symbolic_value(e) {
            if let Some(sym) = &e.sym {
                return self.sym(sym);
            }
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
        self.index_geometry(e, indices, base)
    }
    fn geometry_snapshot(&mut self, shape: Vec<Dimension>) -> Result<View, String> {
        Ok(View {
            storage: Storage::Geometry,
            publication: false,
            offset: self.builder.ins().iconst(types::I64, 0),
            strides: strides(&shape)?,
            shape,
        })
    }
    /// Evaluate the same view operations and checks without consuming elements.
    /// A logical tile snapshot retains contiguous layout independently of data.
    fn geometry(&mut self, e: &Expr) -> Result<View, String> {
        match &e.kind {
            ExprKind::TileAlloc { shape, .. } => {
                let shape = self.shape(shape)?;
                self.geometry_snapshot(shape)
            }
            ExprKind::Load { view, .. } => {
                let view = self.geometry(view)?;
                self.geometry_snapshot(view.shape)
            }
            ExprKind::Index { base, indices } => {
                let base = self.geometry(base)?;
                self.index_geometry(e, indices, base)
            }
            ExprKind::Transpose(base) => {
                let mut view = self.geometry(base)?;
                if view.shape.len() != 2 {
                    return Err("scalar realization transpose requires rank two".into());
                }
                view.shape.swap(0, 1);
                view.strides.swap(0, 1);
                Ok(view)
            }
            ExprKind::Builtin { name: Builtin::Reshape, args } => {
                let view = self.geometry(&args[0])?;
                self.reshape_geometry(e, view)
            }
            _ => self.expr(e)?.view(),
        }
    }
    fn reshape_geometry(&mut self, e: &Expr, mut view: View) -> Result<View, String> {
        let target = self.shape(&e.ty.shaped().ok_or("reshape requires shaped result")?.shape)?;
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
        Ok(view)
    }
    fn index_geometry(&mut self, e: &Expr, indices: &[Index], base: View) -> Result<View, String> {
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
                    let dynamic = start.iter().chain(end).any(|e| e.sym.is_none());
                    let start = match start {
                        Some(e) => self.index_value(e)?,
                        None => self.builder.ins().iconst(types::I64, 0),
                    };
                    let end = match end {
                        Some(e) => self.index_value(e)?,
                        None => parent,
                    };
                    if dynamic {
                        // The checked language clamps data-dependent windows.
                        // Its extent inherits the parent view's storage bound;
                        // empty/reversed windows remain valid empty views.
                        let zero = self.builder.ins().iconst(types::I64, 0);
                        let negative = self.builder.ins()
                            .icmp_imm(IntCC::SignedLessThan, end, 0);
                        let end = self.builder.ins().select(negative, zero, end);
                        let beyond = self.builder.ins()
                            .icmp(IntCC::UnsignedGreaterThan, end, parent);
                        let end = self.builder.ins().select(beyond, parent, end);
                        let negative = self.builder.ins()
                            .icmp_imm(IntCC::SignedLessThan, start, 0);
                        let start = self.builder.ins().select(negative, zero, start);
                        let beyond = self.builder.ins()
                            .icmp(IntCC::UnsignedGreaterThan, start, end);
                        let start = self.builder.ins().select(beyond, end, start);
                        (start, end, true)
                    } else {
                        let ordered = self.builder.ins()
                            .icmp(IntCC::UnsignedLessThanOrEqual, start, end);
                        let bounded = self.builder.ins()
                            .icmp(IntCC::UnsignedLessThanOrEqual, end, parent);
                        let valid = self.builder.ins().band(ordered, bounded);
                        self.require(valid);
                        (start, end, true)
                    }
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
                        // A selected piece already carries a source-derived
                        // capacity. Taking a view of a larger backing tensor
                        // must not replace that bound with the whole parent.
                        let capacity = self.shape(std::slice::from_ref(symbolic))
                            .ok().and_then(|shape| shape.first().copied())
                            .map_or(base.shape[axis].capacity, |dimension| {
                                dimension.capacity.min(base.shape[axis].capacity)
                            });
                        if capacity < base.shape[axis].capacity {
                            let valid = self.builder.ins()
                                .icmp_imm(IntCC::UnsignedLessThanOrEqual, logical, capacity);
                            self.require(valid);
                        }
                        let dim = Dimension {
                            capacity,
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
            ExprKind::Load { view, mode } => {
                let input = self.expr(view)?;
                if *mode == seismic_lang::ir::LoadMode::Borrow {
                    return Err("borrowed load requires an explicit IR value binding".into());
                }
                match input {
                    R::View(view) => Ok(R::View(self.materialize(view)?)),
                    R::Tuple(items) => Ok(R::Tuple(items.into_iter().map(|v| self.materialize(v.view()?).map(R::View)).collect::<Result<_,String>>()?)),
                    _ => Err("invalid scalar realization load".into()),
                }
            }
            ExprKind::Intrinsic { op, args } => {
                use seismic_lang::intrinsics::Operation as I;
                use seismic_realization::ParticipantOperation as P;
                if self.participation.lanes() == 1 { return Err(format!("intrinsic {op} has no selected scalar participant implementation")); }
                if *op == I::LaneIndex {
                    let value=self.participant_call(P::LaneIndex,&[],types::I32);
                    return Ok(R::Scalar(value,DType::I32));
                }
                let (value, dtype) = self.expr(&args[0])?.scalar()?;
                if dtype != DType::F32 { return Err("subgroup exchange currently requires explicit f32 input".into()); }
                let value=match op {
                    I::SimdSum=>self.participant_call(P::Reduce(ReduceOp::Sum), &[value], types::F32),
                    I::ShuffleIndex=>{
                        let (lane,dtype)=self.expr(&args[1])?.scalar()?;
                        if dtype!=DType::I32 && dtype!=DType::U32 {return Err("shuffle source lane must be a 32-bit integer".into());}
                        let valid=self.builder.ins().icmp_imm(IntCC::UnsignedLessThan,lane,i64::from(self.participation.lanes()));
                        self.require(valid);
                        self.participant_call(P::ShuffleIndex,&[value,lane],types::F32)
                    }
                    _=>return Err(format!("intrinsic {op} has no selected scalar participant implementation")),
                };
                Ok(R::Scalar(value,DType::F32))
            }
            ExprKind::Accessor { base, name } => {
                let source = self.expr(base)?.view()?;
                let shape = self.shape(&e.ty.shaped().ok_or("packet accessor requires shaped result")?.shape)?;
                Ok(R::View(self.packet_accessor(source, name, shape)?))
            }
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
                Some(Binding::Packed { mut view, offset }) => {view.offset=self.builder.use_var(offset);Ok(R::View(view))},
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
                let view = self.expr(&args[0])?.view()?;
                Ok(R::View(self.reshape_geometry(e, view)?))
            }
            Builtin::Load => Err("unresolved load reached scalar emission".into()),
            Builtin::Store => {
                let source = self.expr(&args[0])?.view()?;
                let target = self.expr(&args[1])?.view()?;
                self.copy(&source, &target)?;
                Ok(R::Void)
            }
            Builtin::Extent => {
                // Extent belongs to the actual view, which establishes dynamic
                // window metadata and point guards without copying its data.
                let view = self.geometry(&args[0])?;
                // A statically known axis can still carry a nested view check.
                // Its evaluation follows the base view in source argument order.
                if !seismic_lang::effects::can_substitute_symbolic_value(&args[1]) {
                    self.expr(&args[1])?.scalar()?;
                }
                let axis = args[1].sym.as_ref().and_then(Sym::as_constant)
                    .and_then(|n| usize::try_from(n).ok())
                    .ok_or("extent axis must be a nonnegative constant")?;
                let dimension = *view.shape.get(axis).ok_or("extent axis outside view")?;
                let n = self.extent(dimension);
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
                if !dtype.is_float() && !matches!(name, Builtin::Min | Builtin::Max) {
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
        } else if dtype == DType::Bool {
            Ok(if maximum { self.builder.ins().bor(a,b) } else { self.builder.ins().band(a,b) })
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
        let operation = match args[2].kind { ExprKind::Int(tag) => seismic_lang::ir::ReduceOp::from_tag(tag), _ => None }
            .ok_or("scalar realization reduction operation unresolved")?;
        let dtype = match &source.storage {
            Storage::Geometry => return Err("reduction reads geometry-only value data".into()),
            Storage::Dense { dtype, .. } => *dtype,
            Storage::Coefficient { name, .. } => repr::lookup(name).ok_or("unknown representation")?.coefficient_dtype(),
            Storage::Packed { .. } => {
                return Err(
                    "scalar realization packed reduction requires an explicit decode".into(),
                )
            }
        };
        let contract = seismic_lang::reduction::Contract::new(operation, dtype,
            matches!(args.get(3).map(|e| &e.kind), Some(ExprKind::Bool(true))));
        let extent = *source
            .shape
            .get(axis)
            .ok_or("scalar realization reduction axis out of bounds")?;
        if operation == ReduceOp::Argmax
            && (extent.extent.is_some()
                || !(1..=i64::from(i32::MAX) + 1).contains(&extent.capacity))
        {
            return Err("argmax requires a nonempty axis with i32-representable indices".into());
        }
        let output_dtype = contract.output();
        let mut output_shape = source.shape.clone();
        output_shape.remove(axis);
        let out = self.tile(output_shape.clone(), output_dtype)?;
        self.each(&output_shape, |s, indices| {
            let accumulator = s.builder.declare_var(scalar_type(dtype)?);
            let initial = if dtype.is_float() {
                s.builder.ins().f32const(contract.identity().value() as f32)
            } else {
                s.builder.ins().iconst(scalar_type(dtype)?, contract.identity().value() as i64)
            };
            s.builder.def_var(accumulator, initial);
            let winner = if operation == ReduceOp::Argmax {
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
                    ReduceOp::Sum => {
                        if dtype.is_float() {
                            let sum = s.builder.ins().fadd(acc, value);
                            s.publish(sum, dtype)
                        } else if dtype == DType::Bool {
                            s.builder.ins().bor(acc, value)
                        } else {
                            // Integer reduction publication saturates after each step.
                            let extend = |s: &mut Self, v| if dtype == DType::I32 { s.builder.ins().sextend(types::I64, v) } else { s.builder.ins().uextend(types::I64, v) };
                            let a = extend(s, acc);
                            let b = extend(s, value);
                            let sum = s.builder.ins().iadd(a,b);
                            let lo = s.builder.ins().iconst(types::I64, if dtype == DType::I32 { i64::from(i32::MIN) } else { 0 });
                            let hi = s.builder.ins().iconst(types::I64, if dtype == DType::I32 { i64::from(i32::MAX) } else { i64::from(u32::MAX) });
                            let below = s.builder.ins().icmp(IntCC::SignedLessThan, sum, lo);
                            let sum = s.builder.ins().select(below, lo, sum);
                            let above = s.builder.ins().icmp(IntCC::SignedGreaterThan, sum, hi);
                            let sum = s.builder.ins().select(above, hi, sum);
                            s.builder.ins().ireduce(types::I32, sum)
                        }
                    }
                    ReduceOp::Max => s.extremum(acc, value, dtype, true)?,
                    ReduceOp::Min => s.extremum(acc, value, dtype, false)?,
                    ReduceOp::Argmax => {
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
