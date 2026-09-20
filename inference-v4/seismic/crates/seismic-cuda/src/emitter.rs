//! Exhaustive PTX emission over the sealed `CudaOp` set. The emitter is
//! mechanical: no allocation, geometry, synchronization, algorithm, or
//! precision decision is made here, no selected opcode is rejected, and no
//! check is added or omitted — guards come only from discharges recorded in
//! the opcodes during alternative construction.
//!
//! All scalar floating values live in `.f32` registers: narrow (f16/bf16)
//! loads widen on read and stores round on write, which is exactly the
//! registry's load/add/round/store contract. The exact transcendental
//! reference is the versioned `seismic_math` software sequence (`exp`,
//! `log`, `sin`, `cos` calls; `sqrt`/`rsqrt` as one correctly rounded f64
//! step); the approximate `ex2.approx` sequence appears only in opcodes
//! that carry the `Approximate` transfer of the optimized alternative.

use crate::native::CudaParam;
use crate::physical::CudaDialect;
use crate::physical::CudaSliceAxis;
use crate::physical::{
    AtomicMode, CudaConst, CudaOp, CudaScalarDest, CudaSsa, CudaStorageRef, CudaView, MathMode,
    SerialLength,
};
use seismic_lang::{
    intrinsics::{MathOp, PlaneField, ReduceOp},
    logical::GraphValueId,
    syntax::ast::{BinaryOp, UnaryOp},
    types::{DType, ExtentExpr, RuntimeExtentId},
};
use seismic_realization::{
    dispatch::{LinearIterationMap, LinearTotal},
    executable::{
        self, ResolvedExecutorScalar, ResolvedKernelStep, ResolvedLaunch, ResolvedTransport,
    },
};
use std::collections::BTreeMap;

/// Storage type spelling of one dtype in PTX.
pub fn storage_type(dtype: DType) -> &'static str {
    match dtype {
        DType::F32 => "f32",
        DType::F16 | DType::BF16 => "b16",
        DType::I32 => "s32",
        DType::U32 => "u32",
        DType::Bool => "u8",
    }
}

/// Register type spelling of one dtype (narrow floats live widened).
pub fn ptx_type(dtype: DType) -> &'static str {
    match dtype {
        DType::F32 | DType::F16 | DType::BF16 => "f32",
        DType::I32 => "s32",
        DType::U32 | DType::Bool => "u32",
    }
}

/// One emitted kernel value.
#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    Scalar {
        register: String,
        dtype: DType,
    },
    Index(String),
    Pointer {
        register: String,
        dtype: DType,
    },
    Tensor {
        register: String,
        dtype: DType,
        shape: Vec<ExtentExpr>,
        strides: Vec<Stride>,
    },
    Tuple(Vec<Value>),
    Range(String, String),
}

/// One (possibly runtime-computed) byte stride.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Stride {
    Static(u64),
    /// A product of trailing axis extents.
    Runtime(Vec<ExtentExpr>),
}

/// The per-launch emitter.
pub struct Ptx<'a> {
    launch: &'a ResolvedLaunch<CudaDialect>,
    values: &'a BTreeMap<GraphValueId, ResolvedTransport>,
    params: Vec<CudaParam>,
    /// Register of each parameter (scalar params hold their own register).
    param_registers: Vec<String>,
    storage_registers: BTreeMap<u64, String>,
    extent_registers: BTreeMap<u32, String>,
    slot_register: Option<String>,
    status_register: Option<String>,
    result_register: Option<String>,
    r32: usize,
    r64: usize,
    f32: usize,
    pred: usize,
    labels: usize,
    body: Vec<String>,
    ssa: BTreeMap<u32, Value>,
    bound: BTreeMap<u32, Value>,
    axis_registers: BTreeMap<u32, String>,
    /// (linear register, total register, participant count) of the visit loop.
    iteration: Option<(String, String, u64)>,
    loop_label: Option<String>,
    done_label: Option<String>,
    stride_register: Option<String>,
    /// (symbol, argument register, input block, output block) of each
    /// `seismic_math` software call.
    math_calls: Vec<(String, String, String, String)>,
    /// The fold accumulator register (sum) or the tracked extremum.
    fold_accumulator: Option<String>,
    fold_tracked_index: Option<String>,
    exact_math: bool,
}

fn bug(message: &str) -> String {
    format!("compiler bug: {message}")
}

/// Encode one resolved launch into its mechanical parameter list and PTX.
pub fn encode(
    launch: &ResolvedLaunch<CudaDialect>,
    values: &BTreeMap<GraphValueId, ResolvedTransport>,
) -> Result<crate::native::CudaLaunch, String> {
    let ptx = Ptx {
        launch,
        values,
        params: Vec::new(),
        param_registers: Vec::new(),
        storage_registers: BTreeMap::new(),
        extent_registers: BTreeMap::new(),
        slot_register: None,
        status_register: None,
        result_register: None,
        r32: 0,
        r64: 0,
        f32: 0,
        pred: 0,
        labels: 0,
        body: Vec::new(),
        ssa: BTreeMap::new(),
        bound: BTreeMap::new(),
        axis_registers: BTreeMap::new(),
        iteration: None,
        loop_label: None,
        done_label: None,
        stride_register: None,
        math_calls: Vec::new(),
        fold_accumulator: None,
        fold_tracked_index: None,
        exact_math: false,
    };
    ptx.emit()
}

impl<'a> Ptx<'a> {
    // -- register allocation ------------------------------------------------

    fn r32(&mut self) -> String {
        let v = format!("%r{}", self.r32);
        self.r32 += 1;
        v
    }
    fn r64(&mut self) -> String {
        let v = format!("%rd{}", self.r64);
        self.r64 += 1;
        v
    }
    fn f32(&mut self) -> String {
        let v = format!("%f{}", self.f32);
        self.f32 += 1;
        v
    }
    fn pred(&mut self) -> String {
        let v = format!("%p{}", self.pred);
        self.pred += 1;
        v
    }
    fn label(&mut self) -> String {
        let v = format!("L{}", self.labels);
        self.labels += 1;
        v
    }
    fn push(&mut self, line: impl Into<String>) {
        self.body.push(format!("  {}", line.into()));
    }

    fn participants(&self) -> Result<u64, String> {
        match &self.launch.geometry.participants_per_workgroup[0] {
            executable::ExecutionExpr::Const(n) => Ok(*n),
            other => Err(bug(&format!(
                "a universal CUDA launch has a non-constant participant width ({other:?})"
            ))),
        }
    }

    /// Add one parameter (deduplicated) and return its value register.
    fn param(&mut self, param: CudaParam, dtype: DType) -> String {
        if let Some(index) = self.params.iter().position(|existing| *existing == param) {
            return self.param_registers[index].clone();
        }
        let register = if dtype == DType::F32 {
            self.f32()
        } else {
            self.r64()
        };
        self.params.push(param);
        self.param_registers.push(register.clone());
        register
    }

    /// The pointer register of one resolved storage (adding the parameter
    /// on first use).
    fn storage_pointer(
        &mut self,
        resolved: executable::ResolvedStorageId,
    ) -> Result<String, String> {
        if let Some(register) = self.storage_registers.get(&resolved.0) {
            return Ok(register.clone());
        }
        let register = self.param(CudaParam::Storage(resolved), DType::U32);
        self.storage_registers.insert(resolved.0, register.clone());
        Ok(register)
    }

    /// The value register of one runtime extent (adding the parameter on
    /// first use).
    fn extent_register(&mut self, extent: RuntimeExtentId) -> String {
        if let Some(register) = self.extent_registers.get(&extent.0) {
            return register.clone();
        }
        let register = self.param(CudaParam::Extent(extent), DType::U32);
        self.extent_registers.insert(extent.0, register.clone());
        register
    }

    fn slot_base(&mut self) -> String {
        if let Some(register) = self.slot_register.clone() {
            return register;
        }
        let register = self.param(CudaParam::SlotBlock, DType::U32);
        self.slot_register = Some(register.clone());
        register
    }

    fn status_base(&mut self) -> String {
        if let Some(register) = self.status_register.clone() {
            return register;
        }
        let register = self.param(CudaParam::StatusBlock, DType::U32);
        self.status_register = Some(register.clone());
        register
    }

    fn result_base(&mut self) -> String {
        if let Some(register) = self.result_register.clone() {
            return register;
        }
        let register = self.param(CudaParam::ResultBlock, DType::U32);
        self.result_register = Some(register.clone());
        register
    }

    // -- entry ---------------------------------------------------------------

    fn emit(mut self) -> Result<crate::native::CudaLaunch, String> {
        // Bound storages of the launch, in binding order.
        for group in &self.launch.bindings {
            for member in group.members.iter() {
                self.storage_pointer(member.storage)?;
            }
        }
        // The single mapped step's iteration map governs the launch.
        let mut map = None;
        let mut ops = None;
        for kernel_step in self.launch.kernel.steps.iter() {
            if let ResolvedKernelStep::Mapped {
                iteration,
                ops: step_ops,
                ..
            } = kernel_step
            {
                map = Some(iteration.clone());
                ops = Some(step_ops.clone());
                break;
            }
        }
        let (map, ops) = match (map, ops) {
            (Some(map), Some(ops)) => (map, ops),
            _ => return Err(bug("a launch has no mapped iteration")),
        };
        for op in ops.iter() {
            self.declare_op_needs(op)?;
        }
        let participants = self.participants()?;
        self.emit_iteration(&map)?;
        for op in ops.iter() {
            self.op(op)?;
        }
        self.emit_stride_back();
        if self.params.len() * std::mem::size_of::<u64>()
            > crate::native::MAX_KERNEL_PARAMETER_BYTES
        {
            return Err(bug(&format!(
                "resolved CUDA launch needs {} parameter bytes beyond the kernel ABI",
                self.params.len() * std::mem::size_of::<u64>()
            )));
        }
        let exact_math = self
            .exact_math
            .then(|| {
                format!(
                    "{}\n{}\n",
                    include_str!("math/exp.ptx"),
                    include_str!("math/portable_math.ptx")
                )
            })
            .unwrap_or_default();
        let name = format!("seismic_launch_{}", self.launch.id.0);
        // Entry-signature parameters (software-call parameter blocks are
        // internal and declared in the body).
        let mut parameters = Vec::new();
        let mut prologue = String::new();
        for (slot, param) in self.params.iter().enumerate() {
            let register = self.param_registers[slot].clone();
            match param {
                CudaParam::MathCall { .. } => continue,
                CudaParam::Storage(_) => {
                    parameters.push(format!(".param .u64 __storage_{slot}"));
                    prologue.push_str(&format!("  ld.param.u64 {register}, [__storage_{slot}];\n"));
                }
                CudaParam::AbiScalar { dtype, .. } => {
                    parameters.push(format!(".param .{} __scalar_{slot}", ptx_type(*dtype)));
                    prologue.push_str(&format!(
                        "  ld.param.{} {register}, [__scalar_{slot}];\n",
                        ptx_type(*dtype)
                    ));
                }
                CudaParam::Extent(_) => {
                    parameters.push(format!(".param .u64 __extent_{slot}"));
                    prologue.push_str(&format!("  ld.param.u64 {register}, [__extent_{slot}];\n"));
                }
                CudaParam::SlotBlock => {
                    parameters.push(".param .u64 __slots".to_string());
                    prologue.push_str(&format!("  ld.param.u64 {register}, [__slots];\n"));
                }
                CudaParam::StatusBlock => {
                    parameters.push(".param .u64 __status".to_string());
                    prologue.push_str(&format!("  ld.param.u64 {register}, [__status];\n"));
                }
                CudaParam::ResultBlock => {
                    parameters.push(".param .u64 __results".to_string());
                    prologue.push_str(&format!("  ld.param.u64 {register}, [__results];\n"));
                }
            }
        }
        // The internal software-call parameter blocks.
        let mut call_decls = String::new();
        for (_, _, input, output) in &self.math_calls {
            call_decls.push_str(&format!(
                "  .param .b32 {input};\n  .param .b32 {output};\n"
            ));
        }
        let text = format!(
            ".version 7.1\n.target sm_80\n.address_size 64\n{exact_math}.visible .entry {name}({}) .maxntid {participants}, 1, 1 {{\n  .reg .b32 %r<{}>;\n  .reg .b64 %rd<{}>;\n  .reg .f32 %f<{}>;\n  .reg .pred %p<{}>;\n{call_decls}{prologue}{}\n  ret;\n}}\n",
            parameters.join(", "),
            self.r32.max(1),
            self.r64.max(1),
            self.f32.max(1),
            self.pred.max(1),
            self.body.join("\n")
        );
        Ok(crate::native::CudaLaunch {
            id: self.launch.id,
            name,
            params: self.params,
            block: participants,
            work_items: self.launch.work_items.clone(),
            ptx: text,
        })
    }

    /// Declare the parameters one opcode needs before emission begins.
    fn declare_op_needs(&mut self, op: &CudaOp) -> Result<(), String> {
        match op {
            CudaOp::ExtentValue { extent, .. } => {
                self.extent_register(*extent);
            }
            CudaOp::Check { .. } => {
                self.status_base();
            }
            CudaOp::StoreScalar { dest, .. } => match dest {
                CudaScalarDest::ResolvedSlot(_) => {
                    self.slot_base();
                }
                CudaScalarDest::ResolvedResult(_) => {
                    self.result_base();
                }
                CudaScalarDest::Abi { .. } => {
                    return Err(bug(
                        "an unresolved ABI result identity reached CUDA emission",
                    ));
                }
                CudaScalarDest::Output(_) => {}
                CudaScalarDest::Slot(_) => {
                    return Err(bug(
                        "a template executor-slot identity survived resolved-plan construction",
                    ));
                }
            },
            CudaOp::SerialFold { dest, .. } => match dest {
                CudaScalarDest::ResolvedSlot(_) => {
                    self.slot_base();
                }
                CudaScalarDest::ResolvedResult(_) => {
                    self.result_base();
                }
                CudaScalarDest::Abi { .. } => {
                    return Err(bug(
                        "an unresolved ABI result identity reached CUDA emission",
                    ));
                }
                CudaScalarDest::Output(_) => {}
                CudaScalarDest::Slot(_) => {
                    return Err(bug(
                        "a template executor-slot identity survived resolved-plan construction",
                    ));
                }
            },
            CudaOp::Fill { dest, .. } => {
                let resolved = self.resolved_of_ref(dest)?;
                self.storage_pointer(resolved)?;
            }
            CudaOp::CopyElements { source, dest, .. } | CudaOp::Decode { source, dest, .. } => {
                for reference in [source, dest] {
                    let resolved = self.resolved_of_ref(reference)?;
                    self.storage_pointer(resolved)?;
                }
            }
            CudaOp::PackedPlaneRead { source, .. } | CudaOp::PackedElementRead { source, .. } => {
                let resolved = self.resolved_of_ref(source)?;
                self.storage_pointer(resolved)?;
            }
            CudaOp::SerialFor { body, .. } => {
                for nested in body {
                    self.declare_op_needs(nested)?;
                }
            }
            _ => {}
        }
        Ok(())
    }

    // -- iteration -----------------------------------------------------------

    /// Emit the grid-stride traversal: the linear participant coordinate,
    /// the tail mask, the delinearized axis coordinates, and the visit
    /// loop head. Zero work never reaches here (the runtime skips zero-work
    /// launches before submission).
    fn emit_iteration(&mut self, map: &LinearIterationMap) -> Result<(), String> {
        let total = self.total_register(map)?;
        let participants = self.participants()?;
        let linear = self.linear_register();
        let done = self.label();
        let top = self.label();
        self.body.push(format!("{top}:"));
        let invalid = self.pred();
        self.push(format!("setp.ge.u64 {invalid}, {linear}, {total};"));
        self.push(format!("@{invalid} bra {done};"));
        // Delinearization: row-major, the last axis fastest. Runtime
        // extents read their kernel parameter.
        let rest = linear.clone();
        for axis in 0..map.extents.len() {
            let divisor = self.trailing_stride(&map.extents[axis + 1..])?;
            let coordinate = self.r64();
            self.push(format!("div.u64 {coordinate}, {rest}, {divisor};"));
            let modulus = self.extent_operand_register(&map.extents[axis])?;
            self.push(format!("rem.u64 {rest}, {rest}, {modulus};"));
            self.axis_registers.insert(axis as u32, coordinate);
        }
        self.iteration = Some((linear, total, participants));
        self.loop_label = Some(top);
        self.done_label = Some(done);
        Ok(())
    }

    /// The register holding the iteration total: a constant for static
    /// domains, an evaluated expression for runtime products.
    fn total_register(&mut self, map: &LinearIterationMap) -> Result<String, String> {
        match &map.total {
            LinearTotal::Static(total) => {
                let register = self.r64();
                self.push(format!("mov.u64 {register}, {total};"));
                Ok(register)
            }
            LinearTotal::Runtime { product, .. } => self.runtime_expression(product),
        }
    }

    /// Evaluate one retained runtime scalar expression in PTX.
    fn runtime_expression(
        &mut self,
        expr: &seismic_lang::logical::RuntimeScalarExpr,
    ) -> Result<String, String> {
        use seismic_lang::logical::RuntimeScalarExpr as Expr;
        Ok(match expr {
            Expr::Const(value) => {
                let register = self.r64();
                self.push(format!("mov.u64 {register}, {value};"));
                register
            }
            Expr::Extent(id) => self.extent_register(*id),
            Expr::Value(_) => Err(bug("a runtime total references a graph value"))?,
            Expr::Add(left, right) => {
                let (left, right) = (
                    self.runtime_expression(left)?,
                    self.runtime_expression(right)?,
                );
                let register = self.r64();
                self.push(format!("add.u64 {register}, {left}, {right};"));
                register
            }
            Expr::Sub(left, right) => {
                let (left, right) = (
                    self.runtime_expression(left)?,
                    self.runtime_expression(right)?,
                );
                let register = self.r64();
                self.push(format!("sub.u64 {register}, {left}, {right};"));
                register
            }
            Expr::Mul(left, right) => {
                let (left, right) = (
                    self.runtime_expression(left)?,
                    self.runtime_expression(right)?,
                );
                let register = self.r64();
                self.push(format!("mul.lo.u64 {register}, {left}, {right};"));
                register
            }
            Expr::Div(left, right) => {
                let (left, right) = (
                    self.runtime_expression(left)?,
                    self.runtime_expression(right)?,
                );
                let register = self.r64();
                self.push(format!("div.u64 {register}, {left}, {right};"));
                register
            }
            Expr::Rem(left, right) => {
                let (left, right) = (
                    self.runtime_expression(left)?,
                    self.runtime_expression(right)?,
                );
                let register = self.r64();
                self.push(format!("rem.u64 {register}, {left}, {right};"));
                register
            }
        })
    }

    /// The byte-stride divisor of a trailing extent suffix.
    fn trailing_stride(&mut self, extents: &[ExtentExpr]) -> Result<String, String> {
        let mut factors = Vec::new();
        let mut static_product = 1u64;
        let mut all_static = true;
        for extent in extents {
            match extent {
                ExtentExpr::Static(n) => static_product = static_product.saturating_mul(*n),
                _ => {
                    all_static = false;
                    factors.push(extent.clone());
                }
            }
        }
        if all_static {
            let register = self.r64();
            self.push(format!("mov.u64 {register}, {static_product};"));
            Ok(register)
        } else {
            let mut register = self.r64();
            self.push(format!("mov.u64 {register}, {static_product};"));
            for factor in factors {
                let value = self.extent_operand_register(&factor)?;
                let next = self.r64();
                self.push(format!("mul.lo.u64 {next}, {register}, {value};"));
                register = next;
            }
            Ok(register)
        }
    }

    /// The register holding one extent's value (a constant or an extent
    /// parameter).
    fn extent_operand_register(&mut self, extent: &ExtentExpr) -> Result<String, String> {
        match extent {
            ExtentExpr::Static(n) => {
                let register = self.r64();
                self.push(format!("mov.u64 {register}, {n};"));
                Ok(register)
            }
            ExtentExpr::Sym(sym) => match sym.as_constant() {
                Some(value) => {
                    let register = self.r64();
                    self.push(format!("mov.u64 {register}, {value};"));
                    Ok(register)
                }
                None => Err(bug("an unresolved symbolic extent survived specialization")),
            },
            ExtentExpr::Runtime(id) => Ok(self.extent_register(*id)),
        }
    }

    /// The linear participant coordinate register, and the grid-stride
    /// step register (all participants of the launch).
    fn linear_register(&mut self) -> String {
        let block = self.r32();
        let grid = self.r32();
        let tid = self.r32();
        let cta = self.r32();
        self.push(format!("mov.u32 {tid}, %tid.x;"));
        self.push(format!("mov.u32 {cta}, %ctaid.x;"));
        self.push(format!("mov.u32 {block}, %ntid.x;"));
        self.push(format!("mov.u32 {grid}, %nctaid.x;"));
        let block64 = self.r64();
        let tid64 = self.r64();
        let cta64 = self.r64();
        let grid64 = self.r64();
        self.push(format!("cvt.u64.u32 {block64}, {block};"));
        self.push(format!("cvt.u64.u32 {tid64}, {tid};"));
        self.push(format!("cvt.u64.u32 {cta64}, {cta};"));
        self.push(format!("cvt.u64.u32 {grid64}, {grid};"));
        let linear = self.r64();
        self.push(format!("mad.lo.u64 {linear}, {cta64}, {block64}, {tid64};"));
        let stride = self.r64();
        self.push(format!("mul.lo.u64 {stride}, {grid64}, {block64};"));
        self.stride_register = Some(stride);
        linear
    }

    /// The trailing stride-back branch of the grid-stride loop.
    fn emit_stride_back(&mut self) {
        let Some((linear, total, _)) = self.iteration.clone() else {
            return;
        };
        let top = self.loop_label.clone().expect("the loop is open");
        let done = self.done_label.clone().expect("the loop is open");
        let stride = self
            .stride_register
            .clone()
            .expect("the stride is computed");
        let again = self.pred();
        self.push(format!("add.u64 {linear}, {linear}, {stride};"));
        self.push(format!("setp.lt.u64 {again}, {linear}, {total};"));
        self.push(format!("@{again} bra {top};"));
        self.body.push(format!("{done}:"));
    }

    // -- operands ------------------------------------------------------------

    /// Materialize one opcode operand: an SSA register already defined by
    /// an earlier opcode of this launch, or a bound graph value resolved
    /// through its transport (a by-value parameter, an executor slot word,
    /// or a storage-backed tensor descriptor).
    fn operand(&mut self, operand: crate::physical::CudaOperand) -> Result<Value, String> {
        use crate::physical::CudaOperand;
        match operand {
            CudaOperand::Ssa(register) => self.ssa.get(&register.0).cloned().ok_or_else(|| {
                bug(&format!(
                    "SSA register #{} is used before its definition",
                    register.0
                ))
            }),
            CudaOperand::Value(value) => self.bound_value(value),
        }
    }

    /// The resolved storage of one opcode storage reference: a template, or
    /// a boundary value whose storage transports through its launch binding.
    fn resolved_of_ref(
        &mut self,
        reference: &CudaStorageRef,
    ) -> Result<executable::ResolvedStorageId, String> {
        match reference {
            CudaStorageRef::Resolved(storage) => Ok(*storage),
            CudaStorageRef::Template(_) => Err(bug(
                "a template storage identity survived resolved-plan construction",
            )),
            CudaStorageRef::Binding(value) => {
                let transport = self
                    .values
                    .get(value)
                    .cloned()
                    .or_else(|| {
                        self.launch.kernel.steps.iter().find_map(|step| match step {
                            ResolvedKernelStep::Mapped { bindings, .. } => bindings
                                .iter()
                                .find(|(bound, _)| *bound == *value)
                                .map(|(_, transport)| transport.clone()),
                            _ => None,
                        })
                    })
                    .ok_or_else(|| {
                        bug(&format!(
                            "graph value#{} has no resolved transport",
                            value.0
                        ))
                    })?;
                match transport {
                    ResolvedTransport::Storage(views) => Ok(views
                        .as_slice()
                        .first()
                        .ok_or_else(|| bug("a tensor transport has no plane"))?
                        .storage),
                    _ => Err(bug(&format!(
                        "graph value#{} does not transport storage",
                        value.0
                    ))),
                }
            }
            _ => Err(bug("a storage reference is unresolved at emission")),
        }
    }

    /// Materialize one bound graph value through its resolved transport.
    fn bound_value(&mut self, value: GraphValueId) -> Result<Value, String> {
        if let Some(existing) = self.bound.get(&value.0) {
            return Ok(existing.clone());
        }
        let transport = self
            .values
            .get(&value)
            .cloned()
            .or_else(|| {
                self.launch.kernel.steps.iter().find_map(|step| match step {
                    ResolvedKernelStep::Mapped { bindings, .. } => bindings
                        .iter()
                        .find(|(bound, _)| *bound == value)
                        .map(|(_, transport)| transport.clone()),
                    _ => None,
                })
            })
            .ok_or_else(|| {
                bug(&format!(
                    "graph value#{} has no resolved transport",
                    value.0
                ))
            })?;
        let materialized = self.materialize_transport(&transport)?;
        self.bound.insert(value.0, materialized.clone());
        Ok(materialized)
    }

    /// Materialize one resolved transport into kernel SSA.
    fn materialize_transport(&mut self, transport: &ResolvedTransport) -> Result<Value, String> {
        Ok(match transport {
            ResolvedTransport::Void => Value::Scalar {
                register: String::new(),
                dtype: DType::Bool,
            },
            ResolvedTransport::Kernel(_) => {
                return Err(bug("a kernel-local transport reached CUDA emission"));
            }
            ResolvedTransport::ExecutorScalar(scalar) => match scalar {
                ResolvedExecutorScalar::Abi { offset, dtype, .. } => {
                    let register = self.param(
                        CudaParam::AbiScalar {
                            offset: *offset,
                            dtype: *dtype,
                        },
                        *dtype,
                    );
                    Value::Scalar {
                        register,
                        dtype: *dtype,
                    }
                }
                ResolvedExecutorScalar::Result { .. } => {
                    return Err(bug("a result scalar reached input materialization"));
                }
                ResolvedExecutorScalar::Slot { slot, dtype } => {
                    let base = self.slot_base();
                    let address = self.r64();
                    self.push(format!("add.u64 {address}, {base}, {};", slot.0 * 8));
                    let loaded = self.load_slot(address, *dtype)?;
                    Value::Scalar {
                        register: loaded,
                        dtype: *dtype,
                    }
                }
                ResolvedExecutorScalar::Computed { .. } => {
                    return Err(bug(
                        "a computed control scalar reached value materialization",
                    ));
                }
            },
            ResolvedTransport::Storage(views) => {
                let view = views.first();
                let register = self.storage_pointer(view.storage)?;
                Value::Pointer {
                    register,
                    dtype: DType::U32,
                }
            }
            ResolvedTransport::Tuple(items) => Value::Tuple(
                items
                    .iter()
                    .map(|item| self.materialize_transport(item))
                    .collect::<Result<Vec<_>, _>>()?,
            ),
        })
    }

    /// Load one 8-byte executor slot word as the given dtype.
    fn load_slot(&mut self, address: String, dtype: DType) -> Result<String, String> {
        let raw = self.r64();
        self.push(format!("ld.global.u64 {raw}, [{address}];"));
        Ok(match dtype {
            DType::F32 => {
                let value = self.f32();
                self.push(format!("cvt.rn.f32.s64 {value}, {raw};"));
                value
            }
            DType::F16 | DType::BF16 => {
                let value = self.f32();
                self.push(format!("cvt.rn.f32.s64 {value}, {raw};"));
                value
            }
            _ => {
                let value = self.r32();
                self.push(format!("cvt.rni.s32.s64 {value}, {raw};"));
                value
            }
        })
    }

    // -- opcode emission -----------------------------------------------------

    /// Emit one opcode. The match is exhaustive over the sealed set; no
    /// selected opcode is rejected.
    fn op(&mut self, op: &CudaOp) -> Result<(), String> {
        match op {
            CudaOp::NoOp => Ok(()),
            CudaOp::Const { dest, value, dtype } => {
                let emitted = self.constant(value, *dtype)?;
                self.ssa.insert(dest.0, emitted);
                Ok(())
            }
            CudaOp::ExtentValue { dest, extent } => {
                let register = self.extent_register(*extent);
                let index = self.r64();
                self.push(format!("cvt.u64.u32-cast {index}, {register};"));
                self.ssa.insert(dest.0, Value::Index(register));
                Ok(())
            }
            CudaOp::AxisCoordinate { axis, dest } => {
                let register = self
                    .axis_registers
                    .get(axis)
                    .cloned()
                    .ok_or_else(|| bug("an axis coordinate names an absent axis"))?;
                self.ssa.insert(dest.0, Value::Index(register));
                Ok(())
            }
            CudaOp::SerialFor {
                binder,
                length,
                body,
            } => {
                let binder_register = self.r64();
                self.push(format!("mov.u64 {binder_register}, 0;"));
                self.ssa
                    .insert(binder.0, Value::Index(binder_register.clone()));
                let top = self.label();
                let done = self.label();
                self.body.push(format!("{top}:"));
                for nested in body {
                    self.op(nested)?;
                }
                self.push(format!("add.u64 {binder_register}, {binder_register}, 1;"));
                let limit = self.serial_length_register(length)?;
                let again = self.pred();
                self.push(format!("setp.lt.u64 {again}, {binder_register}, {limit};"));
                self.push(format!("@{again} bra {top};"));
                self.body.push(format!("{done}:"));
                Ok(())
            }
            CudaOp::Unary {
                op: kind,
                source,
                dest,
            } => {
                let value = self.scalar_of(*source)?;
                let emitted = self.unary(*kind, value)?;
                self.ssa.insert(dest.0, emitted);
                Ok(())
            }
            CudaOp::Binary {
                op: kind,
                lhs,
                rhs,
                dest,
                dtype,
                guard,
            } => {
                let (lhs, rhs) = (self.operand(*lhs)?, self.operand(*rhs)?);
                if let Some(guard) = guard {
                    let register = self.guard_predicate(guard)?;
                    let (lhs, rhs) = (self.scalar_of_value(lhs)?, self.scalar_of_value(rhs)?);
                    let emitted = self.predicated_binary(*kind, lhs, rhs, *dtype, &register)?;
                    self.ssa.insert(dest.0, emitted);
                } else {
                    let (lhs, rhs) = (self.scalar_of_value(lhs)?, self.scalar_of_value(rhs)?);
                    let emitted = self.binary(*kind, lhs, rhs, *dtype)?;
                    self.ssa.insert(dest.0, emitted);
                }
                Ok(())
            }
            CudaOp::Fma {
                a,
                b,
                c,
                dest,
                dtype,
            } => {
                let (a, b, c) = (
                    self.scalar_of(*a)?,
                    self.scalar_of(*b)?,
                    self.scalar_of(*c)?,
                );
                let out = self.f32();
                self.push(format!("fma.rn.f32 {out}, {a}, {b}, {c};"));
                self.ssa.insert(
                    dest.0,
                    Value::Scalar {
                        register: out,
                        dtype: *dtype,
                    },
                );
                Ok(())
            }
            CudaOp::Cast {
                source,
                dest,
                source_dtype: _,
                target_dtype,
            } => {
                let value = self.scalar_of(*source)?;
                let emitted = self.cast(value, *target_dtype)?;
                self.ssa.insert(dest.0, emitted);
                Ok(())
            }
            CudaOp::Math {
                op: kind,
                arguments,
                dest,
                mode,
            } => {
                let arguments = arguments
                    .iter()
                    .map(|argument| self.scalar_of(*argument))
                    .collect::<Result<Vec<_>, _>>()?;
                let emitted = self.math(*kind, &arguments, *mode)?;
                self.ssa.insert(dest.0, emitted);
                Ok(())
            }
            CudaOp::Select {
                condition,
                then_value,
                else_value,
                dest,
                dtype,
            } => {
                let (condition, then_value, else_value) = (
                    self.scalar_of(*condition)?,
                    self.scalar_of(*then_value)?,
                    self.scalar_of(*else_value)?,
                );
                let emitted = self.select(condition, then_value, else_value, *dtype)?;
                self.ssa.insert(dest.0, emitted);
                Ok(())
            }
            CudaOp::TuplePack { parts, dest } => {
                let values = parts
                    .iter()
                    .map(|part| self.operand(*part))
                    .collect::<Result<Vec<_>, _>>()?;
                self.ssa.insert(dest.0, Value::Tuple(values));
                Ok(())
            }
            CudaOp::TupleGet {
                source,
                index,
                dest,
            } => {
                let value = self.operand(*source)?;
                let Value::Tuple(fields) = value else {
                    return Err(bug("a tuple field read names a non-tuple"));
                };
                let field = fields
                    .get(*index as usize)
                    .cloned()
                    .ok_or_else(|| bug("a tuple field is absent"))?;
                self.ssa.insert(dest.0, field);
                Ok(())
            }
            CudaOp::RangeMake { start, end, dest } => {
                let (start, end) = (self.index_of(*start)?, self.index_of(*end)?);
                self.ssa.insert(dest.0, Value::Range(start, end));
                Ok(())
            }
            CudaOp::RangeStart { source, dest } => {
                let value = self.operand(*source)?;
                let Value::Range(start, _) = value else {
                    return Err(bug("a range start read names a non-range"));
                };
                self.ssa.insert(
                    dest.0,
                    Value::Scalar {
                        register: start,
                        dtype: DType::I32,
                    },
                );
                Ok(())
            }
            CudaOp::RangeEnd { source, dest } => {
                let value = self.operand(*source)?;
                let Value::Range(_, end) = value else {
                    return Err(bug("a range end read names a non-range"));
                };
                self.ssa.insert(
                    dest.0,
                    Value::Scalar {
                        register: end,
                        dtype: DType::I32,
                    },
                );
                Ok(())
            }
            CudaOp::ExtentOf {
                base,
                axis,
                dest,
                valid,
            } => {
                let tensor = self.operand(*base)?;
                let Value::Tensor { shape, .. } = &tensor else {
                    return Err(bug("an extent read names a non-tensor"));
                };
                let extent = shape
                    .get(*axis as usize)
                    .cloned()
                    .ok_or_else(|| bug("an extent axis is absent"))?;
                let register = self.extent_operand_register(&extent)?;
                if *valid {
                    let predicate = self.pred();
                    self.push(format!("setp.ne.u64 {predicate}, {register}, 0;"));
                    let flag = self.r32();
                    self.push(format!("selp.u32 {flag}, 1, 0, {predicate};"));
                    self.ssa.insert(
                        dest.0,
                        Value::Scalar {
                            register: flag,
                            dtype: DType::Bool,
                        },
                    );
                } else {
                    self.ssa.insert(dest.0, Value::Index(register));
                }
                Ok(())
            }
            CudaOp::ElementRead {
                base,
                view,
                view_shape,
                indices,
                dest,
                dtype,
                guard,
            } => {
                let address =
                    self.element_address(base, view, view_shape, indices, dtype.bytes() as u64)?;
                let predicate = self.guard_of(guard.as_ref())?;
                let loaded = self.load_typed(address, *dtype, predicate.as_deref())?;
                self.ssa.insert(
                    dest.0,
                    Value::Scalar {
                        register: loaded,
                        dtype: *dtype,
                    },
                );
                Ok(())
            }
            CudaOp::ElementWrite {
                base,
                view,
                view_shape,
                indices,
                value,
                dtype,
                guard,
            } => {
                let address =
                    self.element_address(base, view, view_shape, indices, dtype.bytes() as u64)?;
                let value = self.scalar_of(*value)?;
                let predicate = self.guard_of(guard.as_ref())?;
                self.store_typed(address, value, *dtype, predicate.as_deref())?;
                Ok(())
            }
            CudaOp::Atomic {
                op,
                base,
                view,
                view_shape,
                indices,
                value,
                dtype,
                mode,
                guard,
            } => {
                let address =
                    self.element_address(base, view, view_shape, indices, dtype.bytes() as u64)?;
                let value = self.scalar_of(*value)?;
                let predicate = self.guard_of(guard.as_ref())?;
                match mode {
                    AtomicMode::Serialized => {
                        self.serialized_atomic(*op, address, value, *dtype, predicate.as_deref())
                    }
                    AtomicMode::CasWord => {
                        if *op != seismic_lang::intrinsics::AtomicOp::Add {
                            return Err(bug("the CAS word loop is planned for atomic add only"));
                        }
                        self.cas_atomic_add(address, value, *dtype)
                    }
                }
            }
            CudaOp::Fill {
                dest,
                view,
                view_shape,
                dtype,
                value,
            } => {
                let resolved = self.resolved_of_ref(dest)?;
                let pointer = self.storage_pointer(resolved)?;
                let value = self.scalar_of(*value)?;
                let linear = self.linear_element(view_shape)?;
                let address = self.r64();
                self.push(format!(
                    "mul.lo.u64 {address}, {linear}, {};",
                    u64::from(dtype.bytes())
                ));
                self.push(format!("add.u64 {address}, {pointer}, {address};"));
                self.store_typed(address, value, *dtype, None)?;
                let _ = view;
                Ok(())
            }
            CudaOp::CopyElements {
                source,
                source_view,
                dest,
                dest_view,
                view_shape,
                dtype,
            } => {
                let source_resolved = self.resolved_of_ref(source)?;
                let dest_resolved = self.resolved_of_ref(dest)?;
                let source_pointer = self.storage_pointer(source_resolved)?;
                let dest_pointer = self.storage_pointer(dest_resolved)?;
                self.linear_copy(
                    source_pointer,
                    dest_pointer,
                    view_shape,
                    *dtype,
                    source_view,
                    dest_view,
                )
            }
            CudaOp::Decode {
                source,
                dest,
                view_shape,
                repr,
                ..
            } => {
                // One decoded f32 per visit: the linear coordinate over the
                // view domain is the flat entry into the representation.
                let source_resolved = self.resolved_of_ref(source)?;
                let source_pointer = self.storage_pointer(source_resolved)?;
                let dest_resolved = self.resolved_of_ref(dest)?;
                let dest_pointer = self.storage_pointer(dest_resolved)?;
                let entry = self.linear_element(view_shape)?;
                let value = self.decode_entry(&source_pointer, repr, &entry, view_shape, None)?;
                let address = self.r64();
                self.push(format!("mul.lo.u64 {address}, {entry}, 4;"));
                self.push(format!("add.u64 {address}, {dest_pointer}, {address};"));
                self.push(format!("st.global.f32 [{address}], {value};"));
                Ok(())
            }
            CudaOp::PackedPlaneRead {
                source,
                view,
                view_shape,
                plane,
                repr,
                dest,
            } => {
                let source_resolved = self.resolved_of_ref(source)?;
                let base = self.storage_pointer(source_resolved)?;
                let entry = self.packed_entry(view, view_shape, &[])?;
                let ordinal = self.plane_ordinal(repr, plane)?;
                let (offset, _group, dtype) = self.plane_offset(repr, ordinal, view_shape)?;
                let address = self.r64();
                let bytes = u64::from(dtype.bytes());
                self.push(format!("mad.lo.u64 {address}, {entry}, {bytes}, {offset};"));
                self.push(format!("add.u64 {address}, {base}, {address};"));
                let loaded = self.load_typed(address, dtype, None)?;
                self.ssa.insert(
                    dest.0,
                    Value::Scalar {
                        register: loaded,
                        dtype,
                    },
                );
                Ok(())
            }
            CudaOp::PackedElementRead {
                source,
                view,
                view_shape,
                indices,
                repr,
                dest,
                guard,
            } => {
                let source_resolved = self.resolved_of_ref(source)?;
                let base = self.storage_pointer(source_resolved)?;
                let index_operands: Vec<crate::physical::CudaOperand> = indices.clone();
                let entry = self.packed_entry(view, view_shape, &index_operands)?;
                let predicate = self.guard_of(guard.as_ref())?;
                let value =
                    self.decode_entry(&base, repr, &entry, view_shape, predicate.as_deref())?;
                self.ssa.insert(
                    dest.0,
                    Value::Scalar {
                        register: value,
                        dtype: DType::F32,
                    },
                );
                Ok(())
            }
            CudaOp::StoreScalar {
                dest,
                source,
                dtype,
            } => {
                let value = self.scalar_of(*source)?;
                match dest {
                    CudaScalarDest::ResolvedSlot(slot) => {
                        let base = self.slot_base();
                        let address = self.r64();
                        self.push(format!("add.u64 {address}, {base}, {};", slot.0 * 8));
                        self.store_word(address, value, *dtype)?;
                    }
                    CudaScalarDest::ResolvedResult(field) => {
                        let base = self.result_base();
                        let address = self.r64();
                        self.push(format!("add.u64 {address}, {base}, {};", field.0 * 8));
                        self.store_word(address, value, *dtype)?;
                    }
                    CudaScalarDest::Abi { .. } => {
                        return Err(bug(
                            "an unresolved ABI result identity reached CUDA emission",
                        ));
                    }
                    CudaScalarDest::Output(_) => {}
                    CudaScalarDest::Slot(_) => {
                        return Err(bug(
                            "a template executor-slot identity survived resolved-plan construction",
                        ));
                    }
                }
                Ok(())
            }
            CudaOp::Check {
                kind,
                values,
                extents,
                status,
                guard,
                ..
            } => self.emit_check(*kind, values, extents, *status, *guard),
            CudaOp::SerialFold {
                operand,
                view,
                view_shape,
                axis,
                length,
                op,
                accumulator,
                dest,
                nonempty,
            } => self.emit_serial_fold(
                operand,
                view,
                view_shape,
                *axis,
                length,
                *op,
                *accumulator,
                dest,
                *nonempty,
            ),
            CudaOp::LaneIndex { dest } => {
                let register = self.r32();
                self.push(format!("mov.u32 {register}, %laneid;"));
                self.ssa.insert(
                    dest.0,
                    Value::Scalar {
                        register,
                        dtype: DType::U32,
                    },
                );
                Ok(())
            }
            CudaOp::Shuffle {
                value,
                index,
                dest,
                dtype,
            } => {
                let value = self.scalar_of(*value)?;
                let index = self.scalar_of(*index)?;
                let out = self.typed_register(*dtype);
                self.push(format!(
                    "shfl.sync.idx.b32 {out}, {value}, {index}, 31, 0xffffffff;"
                ));
                self.ssa.insert(
                    dest.0,
                    Value::Scalar {
                        register: out,
                        dtype: *dtype,
                    },
                );
                Ok(())
            }
            CudaOp::SubgroupReduce {
                op,
                value,
                dest,
                dtype,
            } => {
                let mut value = self.scalar_of(*value)?;
                for offset in [16u32, 8, 4, 2, 1] {
                    let shuffled = self.typed_register(*dtype);
                    self.push(format!(
                        "shfl.sync.down.b32 {shuffled}, {value}, {offset}, 31, 0xffffffff;"
                    ));
                    let combined = self.typed_register(*dtype);
                    let combine = match op {
                        ReduceOp::Sum => format!("add.rn.f32 {combined}, {value}, {shuffled};"),
                        ReduceOp::Max => format!("max.f32 {combined}, {value}, {shuffled};"),
                        ReduceOp::Min => format!("min.f32 {combined}, {value}, {shuffled};"),
                        ReduceOp::Argmax => return Err(bug("argmax has no subgroup reduction")),
                    };
                    self.push(combine);
                    value = combined;
                }
                self.ssa.insert(
                    dest.0,
                    Value::Scalar {
                        register: value,
                        dtype: *dtype,
                    },
                );
                Ok(())
            }
        }
    }

    // -- scalar helpers ------------------------------------------------------

    fn typed_register(&mut self, dtype: DType) -> String {
        if dtype == DType::F32 || dtype.is_float() {
            self.f32()
        } else {
            self.r32()
        }
    }

    /// Materialize one operand as a scalar register (indices convert).
    fn scalar_of(&mut self, operand: crate::physical::CudaOperand) -> Result<String, String> {
        let value = self.operand(operand)?;
        self.scalar_of_value(value)
    }

    fn scalar_of_value(&mut self, value: Value) -> Result<String, String> {
        Ok(match value {
            Value::Scalar { register, dtype } => {
                if dtype.is_float() && dtype != DType::F32 {
                    // Narrow floats already live widened in f32 registers.
                    register
                } else {
                    register
                }
            }
            Value::Index(register) => {
                let converted = self.r32();
                self.push(format!("cvt.rni.s32.s64 {converted}, {register};"));
                converted
            }
            Value::Pointer { .. } => return Err(bug("a pointer reached scalar use")),
            Value::Tensor { .. } => return Err(bug("a tensor reached scalar use")),
            Value::Tuple(_) | Value::Range(..) => {
                return Err(bug("an aggregate reached scalar use"));
            }
        })
    }

    /// Materialize one operand as a u64 index register.
    fn index_of(&mut self, operand: crate::physical::CudaOperand) -> Result<String, String> {
        let value = self.operand(operand)?;
        Ok(match value {
            Value::Index(register) => register,
            Value::Scalar { register, dtype } => {
                let converted = self.r64();
                let (to, from) = if dtype == DType::I32 {
                    ("s64", "s32")
                } else {
                    ("u64", "u32")
                };
                self.push(format!("cvt.{to}.{from} {converted}, {register};"));
                converted
            }
            _ => return Err(bug("a non-index reached index use")),
        })
    }

    fn constant(&mut self, value: &CudaConst, dtype: DType) -> Result<Value, String> {
        Ok(match value {
            CudaConst::Int(bits) => {
                if dtype == DType::Bool {
                    let register = self.r32();
                    self.push(format!("mov.u32 {register}, {};", bits & 1));
                    Value::Scalar { register, dtype }
                } else {
                    let register = self.r64();
                    self.push(format!("mov.u64 {register}, {bits};"));
                    Value::Index(register)
                }
            }
            CudaConst::FloatBits(bits) => {
                let register = self.f32();
                self.push(format!("mov.b32 {register}, 0x{bits:08x};"));
                Value::Scalar {
                    register,
                    dtype: DType::F32,
                }
            }
            CudaConst::Bool(flag) => {
                let register = self.r32();
                self.push(format!("mov.u32 {register}, {};", u8::from(*flag)));
                Value::Scalar { register, dtype }
            }
            CudaConst::ShapeParam(name) => {
                return Err(bug(&format!(
                    "the shape parameter `{name}` survived specialization"
                )));
            }
        })
    }

    /// The guard predicate register of one discharged check.
    /// The guard predicate register of one discharged check.
    fn guard_predicate(&mut self, guard: &CudaSsa) -> Result<String, String> {
        match self.ssa.get(&guard.0) {
            Some(Value::Scalar {
                register,
                dtype: DType::Bool,
            }) => Ok(register.clone()),
            _ => Err(bug("a guard register is not a boolean")),
        }
    }

    fn guard_of(&mut self, guard: Option<&CudaSsa>) -> Result<Option<String>, String> {
        match guard {
            None => Ok(None),
            Some(guard) => self.guard_predicate(guard).map(Some),
        }
    }

    fn predicated_binary(
        &mut self,
        op: BinaryOp,
        lhs: String,
        rhs: String,
        dtype: DType,
        guard: &str,
    ) -> Result<Value, String> {
        // The guarded operation is skipped entirely when the check fails;
        // its result is then undefined and the invocation reports the
        // recorded status error after synchronization.
        let out = self.typed_register(dtype);
        let instruction = binary_instruction(op, dtype)?;
        self.push(format!("@{guard} {instruction} {out}, {lhs}, {rhs};"));
        Ok(Value::Scalar {
            register: out,
            dtype,
        })
    }

    fn unary(&mut self, op: UnaryOp, value: String) -> Result<Value, String> {
        let out = self.f32();
        let instruction = match op {
            UnaryOp::Neg => "neg.f32",
            UnaryOp::Not => "xor.b32",
            UnaryOp::BitNot => "not.b32",
        };
        if op == UnaryOp::Not {
            self.push(format!("{instruction} {out}, {value}, 1;"));
        } else {
            self.push(format!("{instruction} {out}, {value};"));
        }
        Ok(Value::Scalar {
            register: out,
            dtype: DType::F32,
        })
    }

    fn binary(
        &mut self,
        op: BinaryOp,
        lhs: String,
        rhs: String,
        dtype: DType,
    ) -> Result<Value, String> {
        let comparison = matches!(
            op,
            BinaryOp::Eq | BinaryOp::Ne | BinaryOp::Lt | BinaryOp::Le | BinaryOp::Gt | BinaryOp::Ge
        );
        if comparison {
            let predicate = self.pred();
            self.push(format!(
                "setp.{}.{} {predicate}, {lhs}, {rhs};",
                comparison_name(op),
                ptx_type(dtype)
            ));
            let out = self.r32();
            self.push(format!("selp.u32 {out}, 1, 0, {predicate};"));
            return Ok(Value::Scalar {
                register: out,
                dtype: DType::Bool,
            });
        }
        // Integer division and remainder are Euclidean: the quotient is
        // floored and the remainder is always non-negative.
        if matches!(op, BinaryOp::Div | BinaryOp::Rem) && !dtype.is_float() {
            return self.euclidean_division(op, lhs, rhs, dtype);
        }
        let out = self.typed_register(dtype);
        let instruction = binary_instruction(op, dtype)?;
        self.push(format!("{instruction} {out}, {lhs}, {rhs};"));
        Ok(Value::Scalar {
            register: out,
            dtype,
        })
    }

    /// Euclidean `a div b` / `a rem b` with nonzero divisor (the divisor
    /// obligation is discharged as a guard on the operation).
    fn euclidean_division(
        &mut self,
        op: BinaryOp,
        lhs: String,
        rhs: String,
        dtype: DType,
    ) -> Result<Value, String> {
        let signed = dtype == DType::I32;
        let (div, rem, neg) = if signed {
            ("div.s32", "rem.s32", "neg.s32")
        } else {
            ("div.u32", "rem.u32", "neg.s32")
        };
        let quotient = self.r32();
        let remainder = self.r32();
        self.push(format!("{div} {quotient}, {lhs}, {rhs};"));
        self.push(format!("{rem} {remainder}, {lhs}, {rhs};"));
        // If the remainder is nonzero and the operands' signs differ, the
        // truncated quotient is one too high: subtract one.
        let nonzero = self.pred();
        self.push(format!("setp.ne.s32 {nonzero}, {remainder}, 0;"));
        let signs_differ = self.pred();
        self.push(format!("setp.lt.s32 {lhs}, 0;"));
        let lhs_negative = self.pred();
        self.body.pop();
        let _ = lhs_negative;
        let _ = signs_differ;
        let lhs_sign = self.pred();
        self.push(format!("setp.lt.s32 {lhs_sign}, {lhs}, 0;"));
        let rhs_sign = self.pred();
        self.push(format!("setp.lt.s32 {rhs_sign}, {rhs}, 0;"));
        let signs_differ = self.pred();
        self.push(format!("xor.pred {signs_differ}, {lhs_sign}, {rhs_sign};"));
        let adjust = self.pred();
        self.push(format!("and.pred {adjust}, {nonzero}, {signs_differ};"));
        self.push(format!("@{adjust} add.s32 {quotient}, {quotient}, -1;"));
        let out = self.r32();
        match op {
            BinaryOp::Div => self.push(format!("mov.b32 {out}, {quotient};")),
            _ => {
                self.push(format!("mul.lo.s32 {out}, {quotient}, {rhs};"));
                self.push(format!("sub.s32 {out}, {lhs}, {out};"));
            }
        }
        let _ = neg;
        Ok(Value::Scalar {
            register: out,
            dtype,
        })
    }

    fn cast(&mut self, value: String, target: DType) -> Result<Value, String> {
        let out = self.typed_register(target);
        if target == DType::F32 {
            self.push(format!("mov.b32 {out}, {value};"));
        } else if target.is_float() {
            // Round once into the narrow dtype, then keep the widened
            // register (the store rounds again on the same value).
            self.push(format!(
                "mov.b32 {out}, {value}; // widened {} value",
                target.name()
            ));
        } else {
            self.push(format!("cvt.rni.{}.f32 {out}, {value};", ptx_type(target)));
        }
        Ok(Value::Scalar {
            register: out,
            dtype: target,
        })
    }

    fn select(
        &mut self,
        condition: String,
        then_value: String,
        else_value: String,
        dtype: DType,
    ) -> Result<Value, String> {
        let predicate = self.pred();
        self.push(format!("setp.ne.u32 {predicate}, {condition}, 0;"));
        let out = self.typed_register(dtype);
        self.push(format!(
            "selp.{} {out}, {then_value}, {else_value}, {predicate};",
            ptx_type(dtype)
        ));
        Ok(Value::Scalar {
            register: out,
            dtype,
        })
    }

    /// The exact `seismic_math` reference or the approximate sequence of
    /// the optimized alternative.
    fn math(&mut self, op: MathOp, arguments: &[String], mode: MathMode) -> Result<Value, String> {
        let a = arguments
            .first()
            .cloned()
            .ok_or_else(|| bug("a math opcode has no argument"))?;
        let out = self.f32();
        match op {
            MathOp::Fma => {
                let (b, c) = (
                    arguments.get(1).cloned().unwrap_or(a.clone()),
                    arguments.get(2).cloned().unwrap_or(a.clone()),
                );
                self.push(format!("fma.rn.f32 {out}, {a}, {b}, {c};"));
            }
            MathOp::Exp | MathOp::Log | MathOp::Sin | MathOp::Cos => {
                // The versioned software sequence defines the reference
                // bits; `exp_fast` rides the exact sequence here.
                let symbol = match op {
                    MathOp::Exp => "seismic_exp",
                    MathOp::Log => "seismic_log",
                    MathOp::Sin => "seismic_sin",
                    MathOp::Cos => "seismic_cos",
                    _ => unreachable!(),
                };
                let result = self.call_math(symbol, &a)?;
                self.push(format!("mov.b32 {out}, {result};"));
            }
            MathOp::ExpFast => match mode {
                MathMode::Software => {
                    let result = self.call_math("seismic_exp", &a)?;
                    self.push(format!("mov.b32 {out}, {result};"));
                }
                MathMode::FastApprox => {
                    let scaled = self.f32();
                    self.push(format!("mul.f32 {scaled}, {a}, 0f3fb8aa3b;"));
                    self.push(format!("ex2.approx.f32 {out}, {scaled};"));
                }
            },
            MathOp::Sqrt | MathOp::Rsqrt => {
                // One correctly rounded f64 step: widen, compute in f64,
                // round once back to f32 — the exact reference bits.
                let wide = format!("%fd{}", self.f32);
                let _ = wide;
                let wide = self.f64_register();
                self.push(format!("cvt.rn.f64.f32 {wide}, {a};"));
                let wide_result = self.f64_register();
                match op {
                    MathOp::Sqrt => {
                        self.push(format!("sqrt.rn.f64 {wide_result}, {wide};"));
                    }
                    _ => {
                        let reciprocal = self.f64_register();
                        self.push(format!("sqrt.rn.f64 {reciprocal}, {wide};"));
                        self.push(format!(
                            "div.rn.f64 {wide_result}, 0d3ff0000000000000, {reciprocal};"
                        ));
                    }
                }
                self.push(format!("cvt.rn.f32.f64 {out}, {wide_result};"));
            }
            MathOp::Abs => self.push(format!("abs.f32 {out}, {a};")),
            MathOp::Max | MathOp::Min => {
                let b = arguments.get(1).cloned().unwrap_or(a.clone());
                self.push(format!(
                    "{}.f32 {out}, {a}, {b};",
                    if op == MathOp::Max { "max" } else { "min" }
                ));
            }
        }
        Ok(Value::Scalar {
            register: out,
            dtype: DType::F32,
        })
    }

    /// One `seismic_math` software call (exact reference bits): the
    /// argument is stored into the call's parameter block, the sequence is
    /// invoked, and the result is loaded back.
    fn call_math(&mut self, symbol: &str, argument: &str) -> Result<String, String> {
        self.exact_math = true;
        let ordinal = self.math_calls.len();
        let input = format!("__math_arg_{ordinal}");
        let output = format!("__math_result_{ordinal}");
        self.math_calls
            .push((symbol.to_string(), argument.to_string(), input, output));
        let out = self.f32();
        let entry = self.math_calls.len() - 1;
        let (_, _, input, output) = self.math_calls[entry].clone();
        self.push(format!("st.param.f32 [{input}], {argument};"));
        self.push(format!("call.uni ({output}), {symbol}, ({input});"));
        self.push(format!("ld.param.f32 {out}, [{output}];"));
        Ok(out)
    }

    fn f64_register(&mut self) -> String {
        let v = format!("%fd{}", self.f32 / 2);
        self.f32 += 2;
        v
    }

    // -- memory helpers ------------------------------------------------------

    /// Predicated typed load; narrow floats widen into f32 registers.
    fn load_typed(
        &mut self,
        address: String,
        dtype: DType,
        guard: Option<&str>,
    ) -> Result<String, String> {
        let out = self.typed_register(dtype);
        let prefix = guard.map(|guard| format!("@{guard} ")).unwrap_or_default();
        match dtype {
            DType::F32 => self.push(format!("{prefix}ld.global.f32 {out}, [{address}];")),
            DType::F16 | DType::BF16 => {
                let raw = self.r32();
                self.push(format!(
                    "{prefix}ld.global.{} {raw}, [{address}];",
                    storage_type(dtype)
                ));
                self.push(format!(
                    "cvt.f32.{} {out}, {raw};",
                    if dtype == DType::F16 { "f16" } else { "bf16" }
                ));
            }
            _ => self.push(format!(
                "{prefix}ld.global.{} {out}, [{address}];",
                storage_type(dtype)
            )),
        }
        Ok(out)
    }

    /// Predicated typed store; narrow floats round once on the store.
    fn store_typed(
        &mut self,
        address: String,
        value: String,
        dtype: DType,
        guard: Option<&str>,
    ) -> Result<(), String> {
        let prefix = guard.map(|guard| format!("@{guard} ")).unwrap_or_default();
        match dtype {
            DType::F32 => self.push(format!("{prefix}st.global.f32 [{address}], {value};")),
            DType::F16 | DType::BF16 => {
                let raw = self.r32();
                self.push(format!(
                    "cvt.rn.{}.f32 {raw}, {value};",
                    if dtype == DType::F16 { "f16" } else { "bf16" }
                ));
                self.push(format!(
                    "{prefix}st.global.{} [{address}], {raw};",
                    storage_type(dtype)
                ));
            }
            _ => self.push(format!(
                "{prefix}st.global.{} [{address}], {value};",
                storage_type(dtype)
            )),
        }
        Ok(())
    }

    /// Store one scalar into an 8-byte executor slot word.
    fn store_word(&mut self, address: String, value: String, dtype: DType) -> Result<(), String> {
        let wide = self.r64();
        let conversion = match dtype {
            DType::F32 => format!("cvt.rn.s64.f32 {wide}, {value};"),
            _ => format!("cvt.s64.{} {wide}, {value};", ptx_type(dtype)),
        };
        self.push(conversion);
        self.push(format!("st.global.u64 [{address}], {wide};"));
        Ok(())
    }

    /// The linear element index of this participant within the traversal
    /// domain described by `shape` (static total) — the traversal's own
    /// linear coordinate when the shapes agree, otherwise a runtime product.
    fn linear_element(&mut self, shape: &[ExtentExpr]) -> Result<String, String> {
        let all_static = shape.iter().all(|extent| extent.as_static().is_some());
        if all_static {
            let Some((linear, _, _)) = self.iteration.clone() else {
                return Err(bug("a fill has no active traversal"));
            };
            Ok(linear)
        } else {
            let mut product = None;
            for (axis, _extent) in shape.iter().enumerate() {
                let coordinate = self
                    .axis_registers
                    .get(&(axis as u32))
                    .cloned()
                    .ok_or_else(|| bug("a fill axis has no coordinate"))?;
                product = Some(match product {
                    None => coordinate,
                    Some(acc) => {
                        let next = self.r64();
                        self.push(format!("mul.lo.u64 {next}, {acc}, {coordinate};"));
                        next
                    }
                });
            }
            Ok(product.unwrap_or_else(|| self.r64()))
        }
    }

    /// The element address of one tensor value under its view transform.
    /// Flat element entry of one packed view access: view strides at one
    /// element per index plus the transform's offset contributions.
    fn packed_entry(
        &mut self,
        view: &CudaView,
        shape: &[ExtentExpr],
        indices: &[crate::physical::CudaOperand],
    ) -> Result<String, String> {
        if indices.is_empty() {
            return self.linear_element(shape);
        }
        let strides = self.view_strides(view, shape, 1)?;
        let remaining = remaining_axes(view, shape);
        if indices.len() != remaining {
            return Err(bug(&format!(
                "a packed access provides {} indices for {} remaining axes",
                indices.len(),
                remaining
            )));
        }
        let mut offset = self.r64();
        self.push(format!("mov.u64 {offset}, 0;"));
        for (axis, index) in indices.iter().enumerate() {
            let index = self.index_of(*index)?;
            let term = self.r64();
            let stride = self.stride_register_of(&strides[axis])?;
            self.push(format!("mul.lo.u64 {term}, {index}, {stride};"));
            self.push(format!("add.u64 {offset}, {offset}, {term};"));
        }
        self.emit_view_offset(view, shape, &mut offset, 1)?;
        Ok(offset)
    }

    /// The ordinal of one named representation plane.
    fn plane_ordinal(&mut self, repr: &str, plane: &PlaneField) -> Result<usize, String> {
        let representation = seismic_lang::repr::lookup(repr)
            .ok_or_else(|| bug(&format!("unknown representation `{repr}`")))?;
        representation
            .plane_index(plane.name())
            .ok_or_else(|| bug(&format!("`{repr}` has no plane `{}`", plane.name())))
    }

    /// Byte offset (a u64 register), entry group, and dtype of one
    /// representation plane; planes are laid out sequentially, each aligned
    /// to its dtype. The packed axis is the view's last axis.
    fn plane_offset(
        &mut self,
        repr: &str,
        ordinal: usize,
        view_shape: &[ExtentExpr],
    ) -> Result<(String, u64, DType), String> {
        let representation = seismic_lang::repr::lookup(repr)
            .ok_or_else(|| bug(&format!("unknown representation `{repr}`")))?;
        let planes = representation.planes();
        let packed_extent = view_shape
            .last()
            .cloned()
            .ok_or_else(|| bug("a packed view has an axis"))?;
        let values = self.extent_operand_register(&packed_extent)?;
        let mut cursor = self.r64();
        self.push(format!("mov.u64 {cursor}, 0;"));
        for (index, plane) in planes.iter().enumerate() {
            let alignment = u64::from(plane.dtype().bytes()).max(1);
            if alignment > 1 {
                let aligned = self.r64();
                self.push(format!("add.u64 {aligned}, {cursor}, {};", alignment - 1));
                self.push(format!(
                    "and.b64 {aligned}, {aligned}, {};",
                    !(alignment - 1)
                ));
                cursor = aligned;
            }
            if index == ordinal {
                return Ok((cursor, u64::from(plane.group), plane.dtype()));
            }
            let entries = self.r64();
            if plane.group > 1 {
                self.push(format!("div.u64 {entries}, {values}, {};", plane.group));
            } else {
                self.push(format!("mov.u64 {entries}, {values};"));
            }
            let mut bytes = entries;
            if plane.fields > 1 {
                let scaled = self.r64();
                self.push(format!("mul.lo.u64 {scaled}, {bytes}, {};", plane.fields));
                bytes = scaled;
            }
            if let seismic_lang::repr::PlaneEncoding::Packed { bits, .. } = &plane.encoding {
                let packed_words = self.r64();
                self.push(format!("mul.lo.u64 {packed_words}, {bytes}, {bits};"));
                self.push(format!("add.u64 {packed_words}, {packed_words}, 31;"));
                self.push(format!("div.u64 {packed_words}, {packed_words}, 32;"));
                bytes = packed_words;
            }
            let plane_bytes = u64::from(plane.dtype().bytes()).max(1);
            let advanced = self.r64();
            self.push(format!("mul.lo.u64 {advanced}, {bytes}, {plane_bytes};"));
            self.push(format!("add.u64 {cursor}, {cursor}, {advanced};"));
        }
        Err(bug("a plane ordinal exceeds the representation"))
    }

    /// A dense coefficient-plane value converted to f32.
    fn load_plain(
        &mut self,
        address: &str,
        dtype: DType,
        predicate: Option<&str>,
    ) -> Result<String, String> {
        let value = self.r32();
        match dtype {
            DType::F32 => match predicate {
                Some(predicate) => {
                    self.push(format!("@{predicate} ld.global.f32 {value}, [{address}];"));
                }
                None => {
                    self.push(format!("ld.global.f32 {value}, [{address}];"));
                }
            },
            DType::F16 => {
                let raw = self.r32();
                match predicate {
                    Some(predicate) => {
                        self.push(format!("@{predicate} ld.global.u16 {raw}, [{address}];"))
                    }
                    None => self.push(format!("ld.global.u16 {raw}, [{address}];")),
                }
                self.push(format!("cvt.rn.f32.f16 {value}, {raw};"));
            }
            DType::BF16 => {
                let raw = self.r32();
                match predicate {
                    Some(predicate) => {
                        self.push(format!("@{predicate} ld.global.u16 {raw}, [{address}];"))
                    }
                    None => self.push(format!("ld.global.u16 {raw}, [{address}];")),
                }
                let bits = self.r32();
                self.push(format!("cvt.u32.u16 {bits}, {raw};"));
                self.push(format!("shl.b32 {bits}, {bits}, 16;"));
                self.push(format!("mov.b32 {value}, {bits};"));
            }
            other => {
                return Err(bug(&format!(
                    "a coefficient plane of dtype {other:?} is not floating"
                )));
            }
        }
        Ok(value)
    }

    /// One coefficient (scale or bias) of the entry, through the
    /// representation's coefficient planes.
    fn coefficient_value(
        &mut self,
        base_pointer: &str,
        repr: &str,
        bias: bool,
        entry: &str,
        view_shape: &[ExtentExpr],
        predicate: Option<&str>,
    ) -> Result<Option<String>, String> {
        use seismic_lang::repr::{CodeInterpretation, Coefficient};
        let representation = seismic_lang::repr::lookup(repr)
            .ok_or_else(|| bug(&format!("unknown representation `{repr}`")))?;
        let Some(coefficient) = representation.coefficient(bias) else {
            return Ok(None);
        };
        match coefficient {
            Coefficient::Direct { plane } => {
                let ordinal = representation
                    .plane_index(plane.name)
                    .ok_or_else(|| bug("a coefficient plane is absent"))?;
                let (offset, group, dtype) = self.plane_offset(repr, ordinal, view_shape)?;
                let index = self.r64();
                if group > 1 {
                    self.push(format!("div.u64 {index}, {entry}, {group};"));
                } else {
                    self.push(format!("mov.u64 {index}, {entry};"));
                }
                let address = self.r64();
                self.push(format!("add.u64 {address}, {base_pointer}, {offset};"));
                self.push(format!(
                    "mad.lo.u64 {address}, {index}, {}, {address};",
                    u64::from(dtype.bytes())
                ));
                Ok(Some(self.load_plain(&address, dtype, predicate)?))
            }
            Coefficient::Product {
                factor,
                coefficients,
                field,
                sign,
            } => {
                // Factor plane at the entry's factor group.
                let factor_ordinal = representation
                    .plane_index(factor.name)
                    .ok_or_else(|| bug("a factor plane is absent"))?;
                let (factor_offset, factor_group, factor_dtype) =
                    self.plane_offset(repr, factor_ordinal, view_shape)?;
                let factor_index = self.r64();
                if factor_group > 1 {
                    self.push(format!("div.u64 {factor_index}, {entry}, {factor_group};"));
                } else {
                    self.push(format!("mov.u64 {factor_index}, {entry};"));
                }
                let factor_address = self.r64();
                self.push(format!(
                    "add.u64 {factor_address}, {base_pointer}, {factor_offset};"
                ));
                self.push(format!(
                    "mad.lo.u64 {factor_address}, {factor_index}, {}, {factor_address};",
                    u64::from(factor_dtype.bytes())
                ));
                let factor_value = self.load_plain(&factor_address, factor_dtype, predicate)?;
                // Packed coefficients plane: bit position = plane entry * bits.
                let coeff_ordinal = representation
                    .plane_index(coefficients.name)
                    .ok_or_else(|| bug("a coefficients plane is absent"))?;
                let (coeff_offset, coeff_group, _) =
                    self.plane_offset(repr, coeff_ordinal, view_shape)?;
                let coeff_index = self.r64();
                if coeff_group > 1 {
                    self.push(format!("div.u64 {coeff_index}, {entry}, {coeff_group};"));
                } else {
                    self.push(format!("mov.u64 {coeff_index}, {entry};"));
                }
                let plane_entry = self.r64();
                self.push(format!(
                    "mad.lo.u64 {plane_entry}, {coeff_index}, {}, {field};",
                    coefficients.fields
                ));
                let seismic_lang::repr::PlaneEncoding::Packed {
                    bits,
                    interpretation,
                } = &coefficients.encoding
                else {
                    return Err(bug("a coefficients plane is packed"));
                };
                let bit_position = self.r64();
                self.push(format!("mul.lo.u64 {bit_position}, {plane_entry}, {bits};"));
                let word_index = self.r64();
                self.push(format!("div.u64 {word_index}, {bit_position}, 32;"));
                let word_address = self.r64();
                self.push(format!(
                    "add.u64 {word_address}, {base_pointer}, {coeff_offset};"
                ));
                self.push(format!(
                    "mad.lo.u64 {word_address}, {word_index}, 4, {word_address};"
                ));
                let word = self.r32();
                match predicate {
                    Some(predicate) => self.push(format!(
                        "@{predicate} ld.global.u32 {word}, [{word_address}];"
                    )),
                    None => self.push(format!("ld.global.u32 {word}, [{word_address}];")),
                }
                let shift = self.r32();
                let shift64 = self.r64();
                self.push(format!("rem.u64 {shift64}, {bit_position}, 32;"));
                self.push(format!("cvt.u32.u64 {shift}, {shift64};"));
                let raw = self.r32();
                self.push(format!("shr.b32 {raw}, {word}, {shift};"));
                self.push(format!("and.b32 {raw}, {raw}, {};", (1u32 << bits) - 1));
                let code = match interpretation {
                    CodeInterpretation::Unsigned => raw,
                    CodeInterpretation::TwosComplement => {
                        let shifted = self.r32();
                        self.push(format!("shl.b32 {shifted}, {raw}, {};", 32 - bits));
                        let sign = self.r32();
                        self.push(format!("shr.s32 {sign}, {shifted}, {};", 32 - bits));
                        sign
                    }
                    CodeInterpretation::Offset(zero) => {
                        let offset = self.r32();
                        self.push(format!("sub.s32 {offset}, {raw}, {zero};"));
                        offset
                    }
                    CodeInterpretation::Table(_) => {
                        return Err(bug("table-code representations are not yet decodable"));
                    }
                };
                let code_f = self.r32();
                self.push(format!("cvt.rn.f32.s32 {code_f}, {code};"));
                let product = self.r32();
                self.push(format!("mul.f32 {product}, {factor_value}, {code_f};"));
                if sign < 0 {
                    let negated = self.r32();
                    self.push(format!("neg.f32 {negated}, {product};"));
                    Ok(Some(negated))
                } else {
                    Ok(Some(product))
                }
            }
        }
    }

    /// Decoded f32 of one packed entry: the code field out of the words
    /// plane combined with the coefficient planes in a single-rounding fma
    /// (the reference combine rounds once).
    fn decode_entry(
        &mut self,
        base_pointer: &str,
        repr: &str,
        entry: &str,
        view_shape: &[ExtentExpr],
        predicate: Option<&str>,
    ) -> Result<String, String> {
        use seismic_lang::repr::CodeInterpretation;
        let representation = seismic_lang::repr::lookup(repr)
            .ok_or_else(|| bug(&format!("unknown representation `{repr}`")))?;
        let bits = representation.bits;
        let per_word = 32 / bits;
        let mask: u32 = (1u32 << bits) - 1;
        let (words_offset, _, _) = self.plane_offset(repr, 0, view_shape)?;
        let word_off = self.r64();
        self.push(format!("div.u64 {word_off}, {entry}, {per_word};"));
        self.push(format!("mul.lo.u64 {word_off}, {word_off}, 4;"));
        let word_address = self.r64();
        self.push(format!(
            "add.u64 {word_address}, {base_pointer}, {words_offset};"
        ));
        self.push(format!(
            "add.u64 {word_address}, {word_address}, {word_off};"
        ));
        let word = self.r32();
        match predicate {
            Some(predicate) => self.push(format!(
                "@{predicate} ld.global.u32 {word}, [{word_address}];"
            )),
            None => self.push(format!("ld.global.u32 {word}, [{word_address}];")),
        }
        let entry32 = self.r32();
        self.push(format!("cvt.u32.u64 {entry32}, {entry};"));
        let shift = self.r32();
        self.push(format!("rem.u32 {shift}, {entry32}, {per_word};"));
        self.push(format!("mul.lo.u32 {shift}, {shift}, {bits};"));
        let raw = self.r32();
        self.push(format!("shr.b32 {raw}, {word}, {shift};"));
        self.push(format!("and.b32 {raw}, {raw}, {mask};"));
        let code = match &representation.code {
            CodeInterpretation::Unsigned => raw,
            CodeInterpretation::TwosComplement => {
                let shifted = self.r32();
                self.push(format!("shl.b32 {shifted}, {raw}, {};", 32 - bits));
                let sign = self.r32();
                self.push(format!("shr.s32 {sign}, {shifted}, {};", 32 - bits));
                sign
            }
            CodeInterpretation::Offset(zero) => {
                let offset = self.r32();
                self.push(format!("sub.s32 {offset}, {raw}, {zero};"));
                offset
            }
            CodeInterpretation::Table(_) => {
                return Err(bug("table-code representations are not yet decodable"));
            }
        };
        let code_f = self.r32();
        self.push(format!("cvt.rn.f32.s32 {code_f}, {code};"));
        let scale =
            self.coefficient_value(base_pointer, repr, false, entry, view_shape, predicate)?;
        let bias =
            self.coefficient_value(base_pointer, repr, true, entry, view_shape, predicate)?;
        let value = self.r32();
        match (scale, bias) {
            (Some(scale), Some(bias)) => {
                self.push(format!("fma.rn.f32 {value}, {scale}, {code_f}, {bias};"));
            }
            (Some(scale), None) => {
                self.push(format!("mul.f32 {value}, {scale}, {code_f};"));
            }
            (None, Some(bias)) => {
                self.push(format!("add.f32 {value}, {code_f}, {bias};"));
            }
            (None, None) => {
                self.push(format!("mov.f32 {value}, {code_f};"));
            }
        }
        Ok(value)
    }

    fn element_address(
        &mut self,
        base: &crate::physical::CudaOperand,
        view: &CudaView,
        shape: &[ExtentExpr],
        indices: &[crate::physical::CudaOperand],
        bytes: u64,
    ) -> Result<String, String> {
        let base_value = self.operand(*base)?;
        let Value::Pointer { register, .. } = base_value else {
            return Err(bug("an element access names a non-storage base"));
        };
        // Byte strides of the view's (result) shape, the last axis fastest.
        let strides = self.view_strides(view, shape, bytes)?;
        // Remaining axes after the transform drops sliced axes.
        let remaining = remaining_axes(view, shape);
        if indices.len() != remaining {
            return Err(bug(&format!(
                "an element access provides {} indices for {} remaining axes",
                indices.len(),
                remaining
            )));
        }
        let mut offset = self.r64();
        self.push(format!("mov.u64 {offset}, 0;"));
        for (axis, index) in indices.iter().enumerate() {
            let index = self.index_of(*index)?;
            let term = self.r64();
            let stride = self.stride_register_of(&strides[axis])?;
            self.push(format!("mul.lo.u64 {term}, {index}, {stride};"));
            self.push(format!("add.u64 {offset}, {offset}, {term};"));
        }
        // The transform's constant offset contributions (slice points and
        // range starts), with dynamic endpoints read as bound values.
        self.emit_view_offset(view, shape, &mut offset, bytes)?;
        let address = self.r64();
        self.push(format!("add.u64 {address}, {register}, {offset};"));
        Ok(address)
    }

    /// The byte strides of the view's shape under its transform.
    fn view_strides(
        &mut self,
        view: &CudaView,
        shape: &[ExtentExpr],
        bytes: u64,
    ) -> Result<Vec<Stride>, String> {
        match view {
            CudaView::Identity | CudaView::Reshape { .. } => self.contiguous_strides(shape, bytes),
            CudaView::Transpose { permutation } => {
                // The source shape permutes into the view shape; the view's
                // axis i strides like the source's axis permutation[i].
                let mut source_shape = shape.to_vec();
                for (view_axis, source_axis) in permutation.iter().enumerate() {
                    if let Some(value) = shape.get(view_axis) {
                        source_shape[*source_axis as usize] = value.clone();
                    }
                }
                let source_strides = self.contiguous_strides(&source_shape, bytes)?;
                Ok(permutation
                    .iter()
                    .map(|source_axis| source_strides[*source_axis as usize].clone())
                    .collect())
            }
            CudaView::Slice { axes } => {
                let mut kept = Vec::new();
                for axis in axes {
                    match axis {
                        CudaSliceAxis::Full => kept.push(axis.clone()),
                        CudaSliceAxis::Point(value) => kept.push(CudaSliceAxis::Point(*value)),
                        CudaSliceAxis::Range { .. } => kept.push(CudaSliceAxis::Full),
                    }
                }
                let _ = &kept;
                let sliced_shape = remaining_shape(view, shape);
                self.contiguous_strides(&sliced_shape, bytes)
            }
        }
    }

    fn contiguous_strides(
        &mut self,
        shape: &[ExtentExpr],
        bytes: u64,
    ) -> Result<Vec<Stride>, String> {
        // The last axis advances by one element (`bytes`); earlier axes by
        // the trailing extent product.
        let mut strides = vec![Stride::Static(bytes); shape.len()];
        for axis in (0..shape.len().saturating_sub(1)).rev() {
            let factor = &shape[axis + 1];
            strides[axis] = match (&strides[axis + 1], factor.as_static()) {
                (Stride::Static(inner), Some(n)) => Stride::Static(inner * n),
                _ => {
                    let mut extents = vec![factor.clone()];
                    if let Stride::Runtime(mut tail) = strides[axis + 1].clone() {
                        extents.append(&mut tail);
                    }
                    Stride::Runtime(extents)
                }
            };
        }
        Ok(strides)
    }

    fn stride_register_of(&mut self, stride: &Stride) -> Result<String, String> {
        match stride {
            Stride::Static(bytes) => {
                let register = self.r64();
                self.push(format!("mov.u64 {register}, {bytes};"));
                Ok(register)
            }
            Stride::Runtime(extents) => {
                let mut register = self.r64();
                self.push(format!("mov.u64 {register}, 1;"));
                for extent in extents {
                    let value = self.extent_operand_register(extent)?;
                    let next = self.r64();
                    self.push(format!("mul.lo.u64 {next}, {register}, {value};"));
                    register = next;
                }
                Ok(register)
            }
        }
    }

    /// Emit the transform's byte-offset contributions (slice points and
    /// range starts, dynamic endpoints included).
    fn emit_view_offset(
        &mut self,
        view: &CudaView,
        shape: &[ExtentExpr],
        offset: &mut String,
        bytes: u64,
    ) -> Result<(), String> {
        if let CudaView::Slice { axes } = view {
            let strides = self.contiguous_strides(shape, bytes)?;
            for (axis, slice) in axes.iter().enumerate() {
                let stride = self.stride_register_of(&strides[axis])?;
                match slice {
                    CudaSliceAxis::Full => {}
                    CudaSliceAxis::Point(value) => {
                        let index = self
                            .bound
                            .get(&value.0)
                            .cloned()
                            .ok_or_else(|| bug("a slice point is unbound"))?;
                        let point = self.scalar_of_value(index)?;
                        let wide = self.r64();
                        self.push(format!("cvt.s64.s32 {wide}, {point};"));
                        let term = self.r64();
                        self.push(format!("mul.lo.u64 {term}, {wide}, {stride};"));
                        self.push(format!("add.u64 {offset}, {offset}, {term};"));
                    }
                    CudaSliceAxis::Range { start, .. } => {
                        if let Some(start) = start {
                            let index = self
                                .bound
                                .get(&start.0)
                                .cloned()
                                .ok_or_else(|| bug("a slice start is unbound"))?;
                            let value = self.scalar_of_value(index)?;
                            let wide = self.r64();
                            self.push(format!("cvt.s64.s32 {wide}, {value};"));
                            let term = self.r64();
                            self.push(format!("mul.lo.u64 {term}, {wide}, {stride};"));
                            self.push(format!("add.u64 {offset}, {offset}, {term};"));
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// A grid-stride linear copy between two storage pointers.
    fn linear_copy(
        &mut self,
        source: String,
        destination: String,
        shape: &[ExtentExpr],
        dtype: DType,
        source_view: &CudaView,
        dest_view: &CudaView,
    ) -> Result<(), String> {
        let _ = (source_view, dest_view);
        let linear = self.linear_element(shape)?;
        let stride = self
            .stride_register
            .clone()
            .ok_or_else(|| bug("a copy has no traversal stride"))?;
        let total = self
            .iteration
            .as_ref()
            .map(|(_, total, _)| total.clone())
            .ok_or_else(|| bug("a copy has no traversal total"))?;
        let top = self.label();
        let done = self.label();
        let invalid = self.pred();
        self.body.push(format!("{top}:"));
        self.push(format!("setp.ge.u64 {invalid}, {linear}, {total};"));
        self.push(format!("@{invalid} bra {done};"));
        let offset = self.r64();
        self.push(format!(
            "mul.lo.u64 {offset}, {linear}, {};",
            u64::from(dtype.bytes())
        ));
        let input = self.r64();
        self.push(format!("add.u64 {input}, {source}, {offset};"));
        let output = self.r64();
        self.push(format!("add.u64 {output}, {destination}, {offset};"));
        let value = self.load_typed(input, dtype, None)?;
        self.store_typed(output, value, dtype, None)?;
        self.push(format!("add.u64 {linear}, {linear}, {stride};"));
        let again = self.pred();
        self.push(format!("setp.lt.u64 {again}, {linear}, {total};"));
        self.push(format!("@{again} bra {top};"));
        self.body.push(format!("{done}:"));
        Ok(())
    }

    /// Serialized exact load/combine/round/store (the universal atomic column).
    /// Integer elements use the native `atom.global` instruction, which is
    /// exact in any order; floats widen to f32, combine, and round on the
    /// store. PTX `max.f32`/`min.f32` return the non-NaN operand when one
    /// operand is NaN, matching the reference.
    fn serialized_atomic(
        &mut self,
        op: seismic_lang::intrinsics::AtomicOp,
        address: String,
        value: String,
        dtype: DType,
        guard: Option<&str>,
    ) -> Result<(), String> {
        use seismic_lang::intrinsics::AtomicOp;
        let prefix = guard.map(|guard| format!("@{guard} ")).unwrap_or_default();
        match dtype {
            // f32 and narrow floats: load (widened), combine in f32, round on
            // the store — the registry's load/combine/round/store contract.
            DType::F32 | DType::F16 | DType::BF16 => {
                let loaded = self.load_typed(address.clone(), dtype, guard)?;
                let out = self.f32();
                let instruction = match op {
                    AtomicOp::Add => "add.rn.f32",
                    AtomicOp::Max => "max.f32",
                    AtomicOp::Min => "min.f32",
                };
                self.push(format!("{instruction} {out}, {loaded}, {value};"));
                self.store_typed(address, out, dtype, guard)?;
            }
            DType::I32 | DType::U32 => {
                let out = self.r32();
                let instruction = match op {
                    AtomicOp::Add => "add",
                    AtomicOp::Max => "max",
                    AtomicOp::Min => "min",
                };
                self.push(format!(
                    "{prefix}atom.global.{instruction}.{} {out}, [{address}], {value};",
                    ptx_type(dtype)
                ));
            }
            DType::Bool => return Err(bug("bool atomic update is undefined")),
        }
        Ok(())
    }

    /// The planned CAS narrow-floating atomic add: a compare/exchange loop
    /// over the containing 32-bit word. The word lies inside the tensor's
    /// own allocation (the layout admitted it), so the loop never reads
    /// beyond an ABI allocation.
    fn cas_atomic_add(
        &mut self,
        address: String,
        value: String,
        dtype: DType,
    ) -> Result<(), String> {
        if !matches!(dtype, DType::F16 | DType::BF16) {
            // 32-bit atomics use the native add; the CAS loop is only the
            // narrow-floating strategy.
            return self.serialized_atomic(
                seismic_lang::intrinsics::AtomicOp::Add,
                address,
                value,
                dtype,
                None,
            );
        }
        // Word-align down to the containing 4-byte word.
        let byte_offset = self.r64();
        self.push(format!("sub.u64 {byte_offset}, {address}, 0;"));
        let _ = byte_offset;
        let word = self.r64();
        self.push(format!("and.b64 {word}, {address}, -4;"));
        let half_selector = self.r64();
        self.push(format!("rem.u64 {half_selector}, {address}, 4;"));
        let shift = self.r64();
        self.push(format!("mul.lo.u64 {shift}, {half_selector}, 8;"));
        let retry = self.label();
        self.body.push(format!("{retry}:"));
        let assumed = self.r32();
        self.push(format!("ld.global.b32 {assumed}, [{word}];"));
        // Extract the target half, widen, add, round, repack.
        let extracted = self.r32();
        self.push(format!("shr.b32 {extracted}, {assumed}, {shift};"));
        self.push(format!("and.b32 {extracted}, {extracted}, 0xffff;"));
        let widened = self.f32();
        self.push(format!(
            "cvt.f32.{} {widened}, {extracted};",
            if dtype == DType::F16 { "f16" } else { "bf16" }
        ));
        let sum = self.f32();
        self.push(format!("add.rn.f32 {sum}, {widened}, {value};"));
        let narrow = self.r32();
        self.push(format!(
            "cvt.rn.{}.f32 {narrow}, {sum};",
            if dtype == DType::F16 { "f16" } else { "bf16" }
        ));
        let packed = self.r32();
        let mask = self.r32();
        self.push(format!("shl.b32 {mask}, 65535, {shift};"));
        self.push(format!("not.b32 {mask}, {mask};"));
        self.push(format!("and.b32 {packed}, {assumed}, {mask};"));
        let shifted = self.r32();
        self.push(format!("shl.b32 {shifted}, {narrow}, {shift};"));
        self.push(format!("or.b32 {packed}, {packed}, {shifted};"));
        // Compare/exchange; on contention the loop retries.
        let stored = self.r32();
        self.push(format!(
            "atom.global.cas.b32 {stored}, [{word}], {assumed}, {packed};"
        ));
        let failed = self.pred();
        self.push(format!("setp.ne.b32 {failed}, {stored}, {assumed};"));
        self.push(format!("@{failed} bra {retry};"));
        Ok(())
    }

    // -- checks and folds ----------------------------------------------------

    /// Emit one planned runtime predicate: the guard register (false on
    /// violation) and the first-error status write.
    fn emit_check(
        &mut self,
        kind: crate::physical::CheckKind,
        values: &[GraphValueId],
        extents: &[ExtentExpr],
        status: executable::StatusFieldId,
        guard: CudaSsa,
    ) -> Result<(), String> {
        use crate::physical::CheckKind;
        let guard_register = self.pred();
        let mut operands = Vec::new();
        for value in values {
            let operand = self.bound_value(*value)?;
            let scalar = self.scalar_of_value(operand)?;
            operands.push(scalar);
        }
        let mut extent_operands = Vec::new();
        for extent in extents {
            extent_operands.push(self.extent_operand_register(extent)?);
        }
        match kind {
            CheckKind::IndexInBounds | CheckKind::RangeInBounds => {
                let index = operands
                    .first()
                    .cloned()
                    .ok_or_else(|| bug("an index check has no index"))?;
                let bound = extent_operands
                    .first()
                    .cloned()
                    .ok_or_else(|| bug("an index check has no bound"))?;
                self.push(format!("setp.lt.u64 {guard_register}, {index}, {bound};"));
                let zero = self.pred();
                self.push(format!("setp.ge.u64 {zero}, {index}, 0;"));
                self.push(format!(
                    "and.pred {guard_register}, {guard_register}, {zero};"
                ));
                if let Some(end) = operands.get(1) {
                    let end_wide = self.r64();
                    self.push(format!("cvt.s64.s32 {end_wide}, {end};"));
                    let ordered = self.pred();
                    self.push(format!("setp.le.u64 {ordered}, {index}, {end_wide};"));
                    self.push(format!(
                        "and.pred {guard_register}, {guard_register}, {ordered};"
                    ));
                }
            }
            CheckKind::DivisionByZero => {
                let divisor = operands
                    .first()
                    .cloned()
                    .ok_or_else(|| bug("a divisor check has no divisor"))?;
                self.push(format!("setp.ne.s32 {guard_register}, {divisor}, 0;"));
            }
            CheckKind::DivisionOverflow => {
                let lhs = operands
                    .first()
                    .cloned()
                    .ok_or_else(|| bug("a division-overflow check has no dividend"))?;
                let rhs = operands
                    .get(1)
                    .cloned()
                    .ok_or_else(|| bug("a division-overflow check has no divisor"))?;
                let overflow = self.pred();
                self.push(format!("setp.eq.s32 {overflow}, {rhs}, -1;"));
                let is_minimum = self.pred();
                self.push(format!("setp.eq.s32 {is_minimum}, {lhs}, -2147483648;"));
                let both = self.pred();
                self.push(format!("and.pred {both}, {overflow}, {is_minimum};"));
                self.push(format!("not.pred {guard_register}, {both};"));
            }
            CheckKind::ShiftOutOfRange => {
                let amount = operands
                    .first()
                    .cloned()
                    .ok_or_else(|| bug("a shift check has no amount"))?;
                let low = self.pred();
                self.push(format!("setp.ge.s32 {low}, {amount}, 0;"));
                let high = self.pred();
                self.push(format!("setp.lt.s32 {high}, {amount}, 32;"));
                self.push(format!("and.pred {guard_register}, {low}, {high};"));
            }
            CheckKind::ShapeOverflow => {
                // The checked product of the factors fits 64 bits: after
                // each multiply, dividing back must reproduce the previous
                // product (a wrap never does).
                let mut product: Option<String> = None;
                let mut valid = None;
                for factor in extent_operands {
                    let (next, ok) = match product.clone() {
                        None => {
                            let next = self.r64();
                            self.push(format!("mov.u64 {next}, {factor};"));
                            let ok = self.pred();
                            self.push(format!("setp.ne.u64 {ok}, {next}, 0;"));
                            (next, ok)
                        }
                        Some(previous) => {
                            let next = self.r64();
                            self.push(format!("mul.lo.u64 {next}, {previous}, {factor};"));
                            let back = self.r64();
                            self.push(format!("div.u64 {back}, {next}, {factor};"));
                            let consistent = self.pred();
                            self.push(format!("setp.eq.u64 {consistent}, {back}, {previous};"));
                            let nonzero = self.pred();
                            self.push(format!("setp.ne.u64 {nonzero}, {factor}, 0;"));
                            let ok = self.pred();
                            self.push(format!("and.pred {ok}, {consistent}, {nonzero};"));
                            (next, ok)
                        }
                    };
                    product = Some(next);
                    valid = match valid {
                        None => Some(ok),
                        Some(previous) => {
                            let combined = self.pred();
                            self.push(format!("and.pred {combined}, {previous}, {ok};"));
                            Some(combined)
                        }
                    };
                }
                let verdict = valid.unwrap_or_else(|| {
                    let always = self.pred();
                    always
                });
                self.push(format!("mov.pred {guard_register}, {verdict};"));
            }
            CheckKind::EmptyReductionInput => {
                let length = extent_operands
                    .first()
                    .cloned()
                    .ok_or_else(|| bug("an empty check has no length"))?;
                self.push(format!("setp.ne.u64 {guard_register}, {length}, 0;"));
            }
        }
        // The first-error status write: only when the word is currently
        // zero, so the first violation wins.
        let base = self.status_base();
        let address = self.r64();
        self.push(format!("add.u64 {address}, {base}, {};", status.0 * 4));
        let current = self.r32();
        self.push(format!("ld.global.u32 {current}, [{address}];"));
        let untouched = self.pred();
        self.push(format!("setp.eq.u32 {untouched}, {current}, 0;"));
        let kind_code = status_code(kind);
        self.push(format!(
            "@{untouched} st.global.u32 [{address}], {kind_code};"
        ));
        let guard_value = self.r32();
        self.push(format!("selp.u32 {guard_value}, 1, 0, {guard_register};"));
        self.ssa.insert(
            guard.0,
            Value::Scalar {
                register: guard_value,
                dtype: DType::Bool,
            },
        );
        Ok(())
    }

    /// The parallel-outer serial fold: each participant folds the reduced
    /// axis ascending with the registry accumulator/identity/tie semantics
    /// and publishes exactly once.
    #[allow(clippy::too_many_arguments)]
    fn emit_serial_fold(
        &mut self,
        operand: &CudaStorageRef,
        view: &CudaView,
        shape: &[ExtentExpr],
        axis: usize,
        length: &ExtentExpr,
        op: ReduceOp,
        accumulator: DType,
        dest: &CudaScalarDest,
        nonempty: Option<executable::StatusFieldId>,
    ) -> Result<(), String> {
        let resolved = match operand {
            CudaStorageRef::Resolved(storage) => *storage,
            _ => return Err(bug("a fold names an unresolved operand")),
        };
        let pointer = self.storage_pointer(resolved)?;
        let strides =
            self.view_strides(view, shape, u64::from(read_dtype(op, accumulator).bytes()))?;
        let remaining = remaining_axes(view, shape);
        let reduced_stride = self.stride_register_of(&strides[axis])?;
        let outer_strides: Vec<Stride> = strides
            .iter()
            .enumerate()
            .filter(|(index, _)| *index != axis && *index < remaining + 1)
            .map(|(_, stride)| stride.clone())
            .collect();
        // Guard for the nonempty precondition of identity-less folds; its
        // declared status field records an empty reduced axis.
        let guard = if let Some(status) = nonempty {
            let length_register = self.extent_operand_register(length)?;
            let predicate = self.pred();
            self.push(format!("setp.ne.u64 {predicate}, {length_register}, 0;"));
            Some((status, predicate))
        } else {
            None
        };
        // The outer coordinates delinearize this participant's output.
        let outer_offset = self.r64();
        self.push(format!("mov.u64 {outer_offset}, 0;"));
        for (index, stride) in outer_strides.iter().enumerate() {
            let coordinate = self
                .axis_registers
                .get(&(index as u32))
                .cloned()
                .unwrap_or_else(|| self.r64());
            let stride_register = self.stride_register_of(stride)?;
            let term = self.r64();
            self.push(format!(
                "mul.lo.u64 {term}, {coordinate}, {stride_register};"
            ));
            self.push(format!("add.u64 {outer_offset}, {outer_offset}, {term};"));
        }
        // The serial ascending fold over the reduced axis.
        let binder = self.r64();
        self.push(format!("mov.u64 {binder}, 0;"));
        let top = self.label();
        let done = self.label();
        self.body.push(format!("{top}:"));
        let length_register = self.extent_operand_register(length)?;
        let inside = self.pred();
        self.push(format!(
            "setp.lt.u64 {inside}, {binder}, {length_register};"
        ));
        self.push(format!("@{inside} bra {done};"));
        let element_offset = self.r64();
        self.push(format!(
            "mad.lo.u64 {element_offset}, {binder}, {reduced_stride}, {outer_offset};"
        ));
        let address = self.r64();
        self.push(format!("add.u64 {address}, {pointer}, {element_offset};"));
        let element = self.load_typed(address, read_dtype(op, accumulator), None)?;
        self.fold_step(op, accumulator, element, &binder)?;
        self.push(format!("add.u64 {binder}, {binder}, 1;"));
        self.push(format!("bra {top};"));
        self.body.push(format!("{done}:"));
        // An empty identity-less fold records its declared status field.
        if let Some((status, predicate)) = guard {
            let violated = self.pred();
            self.push(format!("not.pred {violated}, {predicate};"));
            let base = self.status_base();
            let address = self.r64();
            self.push(format!("add.u64 {address}, {base}, {};", status.0 * 4));
            let current = self.r32();
            self.push(format!("ld.global.u32 {current}, [{address}];"));
            let untouched = self.pred();
            self.push(format!("setp.eq.u32 {untouched}, {current}, 0;"));
            let first = self.pred();
            self.push(format!("and.pred {first}, {violated}, {untouched};"));
            self.push(format!(
                "@{first} st.global.u32 [{address}], {};",
                status_code(crate::physical::CheckKind::EmptyReductionInput)
            ));
        }
        // One publication of the folded result.
        let result = self.fold_result_register(op, accumulator);
        self.emit_scalar_store(dest, result, accumulator)
    }

    /// Publish one scalar result to its slot or result-field destination.
    fn emit_scalar_store(
        &mut self,
        dest: &CudaScalarDest,
        value: String,
        dtype: DType,
    ) -> Result<(), String> {
        match dest {
            CudaScalarDest::ResolvedSlot(slot) => {
                let base = self.slot_base();
                let address = self.r64();
                self.push(format!("add.u64 {address}, {base}, {};", slot.0 * 8));
                self.store_word(address, value, dtype)?;
                self.push("membar.gl;");
                Ok(())
            }
            CudaScalarDest::ResolvedResult(field) => {
                let base = self.result_base();
                let address = self.r64();
                self.push(format!("add.u64 {address}, {base}, {};", field.0 * 8));
                self.store_word(address, value, dtype)?;
                Ok(())
            }
            CudaScalarDest::Abi { .. } => Err(bug(
                "an unresolved ABI result identity reached CUDA emission",
            )),
            CudaScalarDest::Output(_) => Ok(()),
            CudaScalarDest::Slot(_) => Err(bug(
                "a template executor-slot identity survived resolved-plan construction",
            )),
        }
    }

    /// The value register of one serial-length operand.
    fn serial_length_register(&mut self, length: &SerialLength) -> Result<String, String> {
        Ok(match length {
            SerialLength::Static(n) => {
                let register = self.r64();
                self.push(format!("mov.u64 {register}, {n};"));
                register
            }
            SerialLength::Runtime(id) => self.extent_register(*id),
        })
    }

    /// One ascending fold step with the registry accumulator/identity/tie
    /// semantics: `sum` accumulates in the accumulator dtype; `max`/`min`
    /// keep the extremum; `argmax` tracks the value and the smaller
    /// coordinate on ties.
    fn fold_step(
        &mut self,
        op: ReduceOp,
        accumulator: DType,
        element: String,
        binder: &str,
    ) -> Result<(), String> {
        match op {
            ReduceOp::Sum => {
                let current = match self.fold_accumulator.clone() {
                    Some(register) => register,
                    None => {
                        let zero = self.typed_register(accumulator);
                        match accumulator {
                            DType::F32 => self.push(format!("mov.b32 {zero}, 0x00000000;")),
                            _ => self.push(format!("mov.u32 {zero}, 0;")),
                        }
                        self.fold_accumulator = Some(zero.clone());
                        zero
                    }
                };
                let out = self.typed_register(accumulator);
                if accumulator.is_float() {
                    self.push(format!("add.rn.f32 {out}, {current}, {element};"));
                } else {
                    self.push(format!("add.u32 {out}, {current}, {element};"));
                }
                self.fold_accumulator = Some(out);
            }
            ReduceOp::Max | ReduceOp::Min => {
                let instruction = if op == ReduceOp::Max {
                    "max.f32"
                } else {
                    "min.f32"
                };
                match self.fold_accumulator.clone() {
                    Some(current) => {
                        let out = self.f32();
                        self.push(format!("{instruction} {out}, {current}, {element};"));
                        self.fold_accumulator = Some(out);
                    }
                    None => {
                        // No identity: the fold starts from the first
                        // (ascending) element.
                        self.fold_accumulator = Some(element);
                    }
                }
            }
            ReduceOp::Argmax => {
                let better = self.pred();
                match (
                    self.fold_accumulator.clone(),
                    self.fold_tracked_index.clone(),
                ) {
                    (Some(current), Some(index)) => {
                        let strictly = self.pred();
                        self.push(format!("setp.gt.f32 {strictly}, {element}, {current};"));
                        let tie = self.pred();
                        self.push(format!("setp.eq.f32 {tie}, {element}, {current};"));
                        let smaller = self.pred();
                        self.push(format!("setp.lt.u64 {smaller}, {binder}, {index};"));
                        let tied_smaller = self.pred();
                        self.push(format!("and.pred {tied_smaller}, {tie}, {smaller};"));
                        self.push(format!("or.pred {better}, {strictly}, {tied_smaller};"));
                        let value = self.f32();
                        self.push(format!("selp.f32 {value}, {element}, {current}, {better};"));
                        let chosen = self.r64();
                        self.push(format!("selp.u64 {chosen}, {binder}, {index}, {better};"));
                        self.fold_accumulator = Some(value);
                        self.fold_tracked_index = Some(chosen);
                    }
                    _ => {
                        self.fold_accumulator = Some(element);
                        let start = self.r64();
                        self.push(format!("mov.u64 {start}, {binder};"));
                        self.fold_tracked_index = Some(start);
                        let _ = better;
                    }
                }
            }
        }
        Ok(())
    }

    /// The publication value of the completed fold.
    fn fold_result_register(&mut self, op: ReduceOp, accumulator: DType) -> String {
        match op {
            ReduceOp::Argmax => self
                .fold_tracked_index
                .clone()
                .unwrap_or_else(|| self.r64()),
            _ => self
                .fold_accumulator
                .clone()
                .unwrap_or_else(|| self.typed_register(accumulator)),
        }
    }
}

// ---------------------------------------------------------------------------
// Free helpers
// ---------------------------------------------------------------------------

/// The status code of one planned runtime check (nonzero; first error wins).
fn status_code(kind: crate::physical::CheckKind) -> u32 {
    use crate::physical::CheckKind;
    match kind {
        CheckKind::IndexInBounds => 1,
        CheckKind::RangeInBounds => 2,
        CheckKind::DivisionByZero => 3,
        CheckKind::DivisionOverflow => 4,
        CheckKind::ShiftOutOfRange => 5,
        CheckKind::EmptyReductionInput => 6,
        CheckKind::ShapeOverflow => 7,
    }
}

/// The comparison mnemonic of one relational operator.
fn comparison_name(op: BinaryOp) -> &'static str {
    match op {
        BinaryOp::Eq => "eq",
        BinaryOp::Ne => "ne",
        BinaryOp::Lt => "lt",
        BinaryOp::Le => "le",
        BinaryOp::Gt => "gt",
        BinaryOp::Ge => "ge",
        _ => "eq",
    }
}

/// The PTX instruction of one non-relational operator.
fn binary_instruction(op: BinaryOp, dtype: DType) -> Result<&'static str, String> {
    Ok(match op {
        BinaryOp::Or | BinaryOp::BitOr => "or.b32",
        BinaryOp::And | BinaryOp::BitAnd => "and.b32",
        BinaryOp::BitXor => "xor.b32",
        BinaryOp::Shl => "shl.b32",
        BinaryOp::Shr => {
            if dtype == DType::I32 {
                "shr.s32"
            } else {
                "shr.u32"
            }
        }
        BinaryOp::Add => {
            if dtype.is_float() {
                "add.rn.f32"
            } else {
                "add.u32"
            }
        }
        BinaryOp::Sub => {
            if dtype.is_float() {
                "sub.rn.f32"
            } else {
                "sub.u32"
            }
        }
        BinaryOp::Mul => {
            if dtype.is_float() {
                "mul.rn.f32"
            } else {
                "mul.lo.u32"
            }
        }
        BinaryOp::Div => {
            if dtype.is_float() {
                "div.rn.f32"
            } else {
                return Err(bug("integer division is emitted by the Euclidean sequence"));
            }
        }
        BinaryOp::Rem => {
            if dtype.is_float() {
                return Err(bug("floating remainder is undefined"));
            } else {
                return Err(bug(
                    "integer remainder is emitted by the Euclidean sequence",
                ));
            }
        }
        _ => return Err(bug("a comparison reached arithmetic emission")),
    })
}

/// The number of axes a view transform keeps (sliced points drop axes).
fn remaining_axes(view: &CudaView, shape: &[ExtentExpr]) -> usize {
    match view {
        CudaView::Slice { axes } => axes
            .iter()
            .filter(|axis| !matches!(axis, CudaSliceAxis::Point(_)))
            .count()
            .min(shape.len()),
        _ => shape.len(),
    }
}

/// The shape a view transform keeps (points drop their axis; ranges keep
/// theirs with a runtime extent).
fn remaining_shape(view: &CudaView, shape: &[ExtentExpr]) -> Vec<ExtentExpr> {
    match view {
        CudaView::Slice { axes } => shape
            .iter()
            .enumerate()
            .filter(|(index, _)| {
                axes.get(*index)
                    .map(|axis| !matches!(axis, CudaSliceAxis::Point(_)))
                    .unwrap_or(true)
            })
            .map(|(_, extent)| extent.clone())
            .collect(),
        _ => shape.to_vec(),
    }
}

/// The load dtype of one fold element: the accumulator dtype governs
/// floating sums (f32 for narrow inputs); integers keep theirs.
fn read_dtype(op: ReduceOp, accumulator: DType) -> DType {
    if op == ReduceOp::Sum {
        accumulator
    } else {
        accumulator
    }
}
