//! MSL printer for lowered functions.
//!
//! Storage legality is derived from IR by `storage`; explicit selections are
//! validated against those domains. The diagnostic default still contains a
//! placement policy and is not an optimized Tuned IR path.
//! Load realization is explicit, with borrowing admitted by shared lifetime rules.
//! Dtype conversions preserve the checked numerical publication boundaries.

use crate::collective::{
    Implementation as CollectiveImplementation, Site as CollectiveSite, StorageSpace,
};
use crate::memory::{AllocationId, BarrierPurpose, BarrierSite, MemorySpace, Purpose};
use crate::storage::StorageDecision;
use seismic_lang::abi::ScalarParameter;
use seismic_lang::ast::{AssignOp, BinaryOp, UnaryOp};
use seismic_lang::ir::*;
use seismic_lang::lowered_ir::LoweredIr;
use seismic_lang::repr;
use seismic_lang::sym::{Atom, Sym};
use seismic_lang::types::{DType, Elem, Ty};
use seismic_realization::{
    BufferSpec,
    dispatch::{GroupDispatch, TileDeclaration, TilePlacement},
};
use std::collections::{HashMap, HashSet};

use crate::execution::{self, Config, Execution, SUBGROUP};
use crate::terminal::{Expression as TE, Space as TSpa, Statement as TS, Type as TT};

#[derive(Clone, Debug, PartialEq)]
pub struct Emitted {
    pub source: String,
    pub terminal: crate::terminal::Program,
    pub launches: Vec<Launch>,
    pub buffers: Vec<BufferSpec>,
    pub scalars: Vec<ScalarParameter>,
    /// Scratch the realization needs and the caller did not supply: bytes per buffer, in
    /// the order they follow the caller's buffers. A split reduction's partial states live
    /// here, so no kernel has to declare them.
    pub scratch: Vec<usize>,
    /// Compiler-owned invocation status, bound and checked by every execution path.
    pub status_slot: Option<usize>,
    /// Bound buffer pairs must be disjoint; exact alias is admitted only when
    /// both views use the same dense element type and coordinates.
    pub alias_pairs: Vec<(usize, usize, bool)>,
}

impl Emitted {
    pub fn scalar_layout(&self) -> Result<seismic_lang::abi::ScalarLayout, String> {
        seismic_lang::abi::ScalarLayout::natural(&self.scalars)
    }
    pub fn encode_scalars(&self, values: &[f64]) -> Result<Vec<u8>, String> {
        self.scalar_layout()?.encode(values)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Launch {
    pub kernel: String,
    pub threadgroups: u64,
    pub threads_per_threadgroup: u64,
    /// This launch reads what an earlier one wrote, so the two must not overlap. Sequential
    /// `parallel` phases and a split's merge are the cases; the emitter knows which.
    pub after_barrier: bool,
    pub dispatch: Option<GroupDispatch>,
    /// Tile arrays declared through the tile materializer, not a complete native
    /// private-storage or register account. Native optimization can remove them.
    pub tiles: Vec<TileDeclaration>,
    pub declared_threadgroup_bytes: u64,
}

/// One axis of a tile: static capacity and a C expression for the runtime extent.
#[derive(Clone, Debug, PartialEq)]
struct Dim {
    cap: i64,
    ext: String,
    value: TE,
}

impl Dim {
    fn is_static(&self) -> bool {
        self.ext == self.cap.to_string()
    }
}

/// Checked coordinates and retained axes of an indexed logical view. Address
/// construction and shape-only queries consume the same evaluated geometry.
struct ViewGeometry {
    starts: Vec<Sym>,
    axes: Vec<usize>,
    shape: Vec<Sym>,
}

#[derive(Clone, Debug)]
enum Realization {
    /// Captured logical tile geometry with no addressable element storage.
    Geometry {
        dims: Vec<Dim>,
        strides: Vec<Sym>,
    },
    Scalar {
        name: String,
    },
    Index {
        name: String,
    },
    Param {
        name: String,
        shape: Vec<i64>,
        elem: Elem,
    },
    View {
        space: TSpa,
        param: String,
        elem: Elem,
        offset: Sym,
        strides: Vec<Sym>,
        shape: Vec<Sym>,
    },
    Replicated {
        name: String,
        dims: Vec<Dim>,
        dtype: DType,
    },
    Distributed {
        name: String,
        dims: Vec<Dim>,
        dtype: DType,
        slots: i64,
    },
    Shared {
        name: String,
        dims: Vec<Dim>,
        dtype: DType,
    },
    Frag {
        name: String,
    },
}

struct SplitTail {
    phase: usize,
    kernel: String,
    carried: Vec<VarId>,
    body: Vec<Stmt>,
}

struct Printer<'a> {
    terminal: crate::terminal::Program,
    expressions: HashMap<String, TE>,
    local_planes: HashMap<String, (TSpa, DType, u64)>,
    f: &'a LoweredIr,
    execution: &'a Execution,
    cfg: &'a Config,
    out: String,
    real: HashMap<VarId, Realization>,
    /// atom or emitted symbol -> C name
    names: HashMap<String, String>,
    indent: usize,
    counter: usize,
    owned_ctx: Vec<(VarId, Vec<String>, Option<String>)>,
    buffers: Vec<BufferSpec>,
    scalars: Vec<ScalarParameter>,
    shared_decls: Vec<(String, u64)>,
    tile_declarations: Vec<TileDeclaration>,
    memory_launch: usize,
    emitted_barriers: HashSet<BarrierSite>,
    emitted_collectives: HashSet<CollectiveSite>,
    current_operation: Option<OperationId>,
    collective_ordinal: usize,
    /// Simdgroups per threadgroup for the kernel being emitted.
    simdgroups: i64,
}

pub fn emit(f: &LoweredIr) -> Result<Emitted, String> {
    emit_with(f, Config::default())
}

pub fn emit_with(f: &LoweredIr, cfg: Config) -> Result<Emitted, String> {
    emit_execution(&execution::prepare(f, cfg)?)
}

/// Resolve storage before printing the selected execution.
pub fn emit_storage_selected(
    f: &LoweredIr,
    cfg: Config,
    select: &mut dyn FnMut(&StorageDecision) -> Result<TilePlacement, String>,
) -> Result<Emitted, String> {
    emit_execution(&execution::prepare_storage_selected(f, cfg, select)?)
}

/// Print an already transformed execution with resolved materialized-value placements.
/// Allocation lifetimes and intrinsic scheduling remain incomplete boundaries.
pub fn emit_execution(execution: &Execution) -> Result<Emitted, String> {
    Ok(prepare_execution(execution)?.clone())
}
/// One immutable emission implementation is retained with each selected execution.
/// Native compilation and accounting consume this same result.
pub fn prepare_execution(execution: &Execution) -> Result<&Emitted, String> {
    execution
        .emission
        .get_or_init(|| build_execution(execution))
        .as_ref()
        .map_err(Clone::clone)
}
fn build_execution(execution: &Execution) -> Result<Emitted, String> {
    let f = &execution.function;
    let cfg = &execution.config;
    let mut emitted = Printer {
        terminal: Default::default(),
        expressions: HashMap::new(),
        local_planes: HashMap::new(),
        f,
        execution,
        cfg,
        out: String::new(),
        real: HashMap::new(),
        names: HashMap::new(),
        indent: 0,
        counter: 0,
        owned_ctx: Vec::new(),
        buffers: Vec::new(),
        scalars: Vec::new(),
        shared_decls: Vec::new(),
        tile_declarations: Vec::new(),
        memory_launch: 0,
        emitted_barriers: HashSet::new(),
        emitted_collectives: HashSet::new(),
        current_operation: None,
        collective_ordinal: 0,
        simdgroups: cfg.sg_per_tg,
    }
    .emit()?;
    let planned = execution.memory.launches();
    if emitted.launches.len() != planned.len() {
        return Err("emitted launch count disagrees with allocation plan".into());
    }
    for (launch, plan) in emitted.launches.iter().zip(planned) {
        if launch
            .tiles
            .iter()
            .ne(plan.arrays.iter().map(|a| &a.declaration))
            || launch.declared_threadgroup_bytes != plan.shared_bytes_per_group
        {
            let at = launch
                .tiles
                .iter()
                .zip(&plan.arrays)
                .position(|(a, b)| a != &b.declaration)
                .unwrap_or(launch.tiles.len().min(plan.arrays.len()));
            return Err(format!(
                "{}: emitted tile arrays disagree with the allocation plan at {at}: emitted {:?}, planned {:?}; counts {} vs {}",
                launch.kernel,
                launch.tiles.get(at),
                plan.arrays.get(at).map(|a| &a.declaration),
                launch.tiles.len(),
                plan.arrays.len()
            ));
        }
    }
    for (at, (a, ad)) in execution.partition_parameters.iter().enumerate() {
        for (b, bd) in &execution.partition_parameters[at + 1..] {
            let find = |id: usize| {
                emitted
                    .buffers
                    .iter()
                    .position(|slot| slot.parameter == f.vars[id].name && slot.plane.is_empty())
                    .ok_or("partition parameter missing from ABI")
            };
            emitted.alias_pairs.push((find(*a)?, find(*b)?, ad == bd));
        }
    }
    emitted.alias_pairs.extend_from_slice(
        seismic_realization::InvocationConditions::from_lowered(execution.source())?.alias_pairs(),
    );
    emitted.alias_pairs.sort_unstable();
    emitted.alias_pairs.dedup();
    Ok(emitted)
}

fn ctype(d: DType) -> &'static str {
    match d {
        DType::F32 => "float",
        DType::BF16 => "bfloat",
        DType::F16 => "half",
        DType::I32 => "int",
        DType::U32 => "uint",
        DType::Bool => "bool",
    }
}

fn scalar_dtype(t: &Ty) -> Option<DType> {
    match t {
        Ty::Scalar(d) => Some(*d),
        _ => None,
    }
}

impl Printer<'_> {
    fn line(&mut self, text: &str) {
        self.target(crate::terminal::Statement::Unmapped(text.into()));
    }
    fn target(&mut self, statement: crate::terminal::Statement) {
        match &statement {
            TS::Let { name, ty, .. } => {
                self.expressions
                    .insert(name.clone(), TE::variable(name, *ty));
            }
            TS::For { name, .. } => {
                self.expressions
                    .insert(name.clone(), TE::variable(name, TT::I32));
            }
            _ => {}
        }
        let text = statement.render();
        if self.terminal.launches.len() <= self.memory_launch {
            self.terminal
                .launches
                .resize_with(self.memory_launch + 1, Vec::new);
        }
        self.terminal.launches[self.memory_launch].push(crate::terminal::Site {
            operation: self.current_operation,
            statement,
        });
        self.source_line(&text);
    }
    fn source_line(&mut self, text: &str) {
        for _ in 0..self.indent {
            self.out.push_str("  ");
        }
        self.out.push_str(text);
        self.out.push('\n');
    }

    fn fresh(&mut self, base: &str) -> String {
        self.counter += 1;
        let name = format!("{base}_{}", self.counter);
        self.names.insert(name.clone(), name.clone());
        name
    }

    /// C expression of a symbolic integer under the current names.
    fn sym(&self, s: &Sym) -> Result<String, String> {
        self.target_sym(s).map(|expression| expression.render())
    }

    /// Static capacity of a symbolic extent: piece atoms take their capacity.
    fn cap(&self, s: &Sym) -> Result<i64, String> {
        self.execution.storage.capacity(s)
    }

    fn dim(&self, s: &Sym) -> Result<Dim, String> {
        let cap = self.cap(s)?;
        let ext = if s.as_constant().is_some() {
            cap.to_string()
        } else {
            self.sym(s)?
        };
        Ok(Dim {
            cap,
            ext,
            value: self.target_sym(s)?,
        })
    }

    /// All value identities are established before printing.
    fn vars(&self) -> &[Var] {
        &self.f.vars
    }

    fn dims(&self, shape: &[Sym]) -> Result<Vec<Dim>, String> {
        shape.iter().map(|s| self.dim(s)).collect()
    }

    fn emit(mut self) -> Result<Emitted, String> {
        let mut header = String::new();
        header.push_str("#include <metal_stdlib>\n#include <metal_simdgroup_matrix>\n#pragma clang fp contract(off)\nusing namespace metal;\n\n");
        header.push_str(&crate::support::render(self.execution.support()));
        let mut index = 0usize;
        let mut params_sig: Vec<String> = Vec::new();
        for (i, (name, ty)) in self.f.params.iter().enumerate() {
            match ty {
                Ty::Tensor(s) => {
                    let shape: Vec<i64> = s
                        .shape
                        .iter()
                        .map(|d| {
                            d.as_constant().ok_or_else(|| {
                                format!("parameter `{name}` has a non-concrete shape")
                            })
                        })
                        .collect::<Result<_, _>>()?;
                    let elements = shape.iter().try_fold(1usize, |n, d| {
                        usize::try_from(*d)
                            .ok()
                            .and_then(|d| n.checked_mul(d))
                            .ok_or("invalid Metal tensor size")
                    })?;
                    let plane_bytes = |count: usize, width: u32| {
                        count
                            .checked_mul(width as usize)
                            .ok_or("Metal buffer size overflow")
                    };
                    match &s.elem {
                        Elem::Dtype(d) => {
                            params_sig
                                .push(format!("device {}* {name} [[buffer({index})]]", ctype(*d)));
                            self.buffers.push(BufferSpec {
                                parameter: name.clone(),
                                plane: "".into(),
                                bytes: plane_bytes(elements, d.bytes())?,
                                alignment: d.bytes() as usize,
                            });
                            index += 1;
                        }
                        Elem::Repr(r) => {
                            let rep = repr::lookup(r).ok_or("unknown Metal representation")?;
                            if shape
                                .last()
                                .is_none_or(|n| *n % i64::from(rep.storage_group()) != 0)
                            {
                                return Err(
                                    "Metal packed parameter requires complete storage groups"
                                        .into(),
                                );
                            }
                            for plane in rep.planes() {
                                let dtype = plane.dtype();
                                let bytes = usize::try_from(
                                    plane
                                        .bytes(elements as u64)
                                        .ok_or("packed plane size overflow")?,
                                )
                                .map_err(|_| "packed plane size exceeds usize")?;
                                params_sig.push(format!(
                                    "device const {}* {name}_{} [[buffer({index})]]",
                                    ctype(dtype),
                                    plane.name
                                ));
                                self.buffers.push(BufferSpec {
                                    parameter: name.clone(),
                                    plane: plane.name.into(),
                                    bytes,
                                    alignment: dtype.bytes() as usize,
                                });
                                index += 1;
                            }
                        }
                        Elem::Param(p) => {
                            return Err(format!(
                                "parameter `{name}` has unresolved element type `{p}`"
                            ));
                        }
                    }
                    self.real.insert(
                        i,
                        Realization::Param {
                            name: name.clone(),
                            shape,
                            elem: s.elem.clone(),
                        },
                    );
                }
                Ty::Scalar(d) => {
                    self.scalars
                        .push(ScalarParameter::from_lowered(self.f, name, *d)?);
                    self.real.insert(
                        i,
                        Realization::Scalar {
                            name: format!("sc.{name}"),
                        },
                    );
                    if let VarKind::Index(Atom::Param(atom)) = &self.vars()[i].kind {
                        self.names.insert(atom.clone(), format!("sc.{name}"));
                    }
                }
                other => {
                    return Err(format!(
                        "parameter `{name}` of type {other} is not supported in a kernel signature"
                    ));
                }
            }
        }
        // Scratch the split needs, declared on every kernel so one binding list serves all.
        for scratch in self.execution.memory.scratch() {
            params_sig.push(format!(
                "device {}* split_{} [[buffer({})]]",
                ctype(scratch.dtype),
                scratch.index,
                index + scratch.index
            ));
        }
        index += self.execution.memory.scratch().len();
        if !self.scalars.is_empty() {
            header.push_str("struct Scalars {\n");
            for parameter in &self.scalars {
                let (n, d) = (&parameter.name, &parameter.dtype);
                header.push_str(&format!("  {} {n};\n", ctype(*d)));
            }
            header.push_str("};\n\n");
            params_sig.push(format!("constant Scalars& sc [[buffer({index})]]"));
        }
        params_sig.push("uint3 tg_pos [[threadgroup_position_in_grid]]".into());
        params_sig.push("uint sg_id [[simdgroup_index_in_threadgroup]]".into());
        params_sig.push("uint lane [[thread_index_in_simdgroup]]".into());

        let status_slot = index + usize::from(!self.scalars.is_empty());
        params_sig.push(format!(
            "device atomic_uint* seismic_status [[buffer({status_slot})]]"
        ));
        let mut launches = Vec::new();
        let body = &self.f.body;
        for (k, stmt) in body.iter().enumerate() {
            let mut split_tails = Vec::new();
            let phase = &self.execution.phases[k];
            let StmtKind::Parallel { vars, body, .. } = &stmt.kind else {
                unreachable!()
            };
            let split_here = phase.split.is_some();
            if let Some(split) = &phase.split {
                let VarKind::Index(Atom::Param(atom)) = &self.f.vars[split.part].kind else {
                    unreachable!()
                };
                self.names.insert(atom.clone(), "part".into());
                self.real.insert(
                    split.part,
                    Realization::Index {
                        name: "part".into(),
                    },
                );
            }
            let kernel = format!("{}_{k}", self.f.name);
            let mut kernel_out = String::new();
            std::mem::swap(&mut self.out, &mut kernel_out);
            self.indent = 1;
            let sg_per_tg = phase.dispatch.items_per_group as i64;
            self.simdgroups = sg_per_tg;
            self.emit_storage(&phase.dispatch)?;
            self.emit_prologue(vars, &phase.dispatch)?;
            if split_here {
                // Each part streams its slice and publishes its carried state to scratch.
                // A second launch folds the parts together with the loop body's own merge
                // rule and runs the tail, so the kernel text never mentions either.
                let split = phase.split.as_ref().unwrap();
                self.block(&body[..split.loop_at])?;
                self.block(&split.validation_bindings)?;
                for view in &split.original_views {
                    self.view_of(view)?;
                }
                let carried = split.carried.clone();
                self.block(&body[split.loop_at..split.loop_at + 1])?;
                self.publish_partials(k, &carried, "item")?;
                split_tails.push(SplitTail {
                    phase: k,
                    kernel: kernel.clone(),
                    carried,
                    body: body[split.loop_at + 1..].to_vec(),
                });
                self.block(&[])?;
            } else {
                self.block(body)?;
            }
            self.target(TS::Return(None));
            self.indent = 0;
            std::mem::swap(&mut self.out, &mut kernel_out);
            self.out.push_str(&format!(
                "kernel void {kernel}(\n    {}\n) {{\n",
                params_sig.join(",\n    ")
            ));
            // Resource fit: a realization whose threadgroup memory exceeds the device's is
            // not a candidate. The model must not offer it, so it is an error here.
            let shared_bytes = self.shared_decls.iter().try_fold(0u64, |sum, (_, bytes)| {
                sum.checked_add(*bytes).ok_or("shared storage sum overflow")
            })?;
            if shared_bytes > self.cfg.max_threadgroup_bytes as u64 {
                return Err(format!(
                    "this realization needs {shared_bytes} bytes of threadgroup memory, over this device's {}",
                    self.cfg.max_threadgroup_bytes
                ));
            }
            for (d, _) in self.shared_decls.drain(..) {
                self.out.push_str(&format!("  {d}\n"));
            }
            self.out.push_str(&kernel_out);
            self.out.push_str("}\n\n");
            let after_barrier = self.execution.memory.launches()[launches.len()]
                .predecessor
                .is_some();
            launches.push(Launch {
                kernel,
                threadgroups: phase.dispatch.groups,
                threads_per_threadgroup: phase.dispatch.threads_per_group,
                after_barrier,
                dispatch: Some(phase.dispatch.clone()),
                tiles: self.finish_memory()?,
                declared_threadgroup_bytes: shared_bytes,
            });
            // Complete this phase before a subsequent phase can observe its output.
            for SplitTail {
                phase: k,
                kernel: first,
                carried,
                body: tail,
            } in split_tails
            {
                let StmtKind::Parallel { vars, .. } = &self.f.body[k].kind else {
                    unreachable!()
                };
                let dispatch = self.execution.phases[k].merge_dispatch.as_ref().unwrap();
                let kernel = format!("{first}_merge");
                let mut kernel_out = String::new();
                std::mem::swap(&mut self.out, &mut kernel_out);
                self.indent = 1;
                let sg_per_tg = dispatch.items_per_group as i64;
                self.simdgroups = sg_per_tg;
                self.emit_storage(dispatch)?;
                self.emit_prologue(vars, dispatch)?;
                self.merge_partials(
                    k,
                    &carried,
                    &self.execution.phases[k].split.as_ref().unwrap().merges,
                    self.f.body[k]
                        .id
                        .ok_or("merge phase has no operation identity")?,
                )?;
                self.block(&tail)?;
                self.target(TS::Return(None));
                self.indent = 0;
                std::mem::swap(&mut self.out, &mut kernel_out);
                self.out.push_str(&format!(
                    "kernel void {kernel}(\n    {}\n) {{\n",
                    params_sig.join(",\n    ")
                ));
                let shared_bytes = self.shared_decls.iter().try_fold(0u64, |sum, (_, bytes)| {
                    sum.checked_add(*bytes).ok_or("shared storage sum overflow")
                })?;
                if shared_bytes > self.cfg.max_threadgroup_bytes as u64 {
                    return Err("split merge exceeds threadgroup storage limit".into());
                }
                for (d, _) in self.shared_decls.drain(..) {
                    self.out.push_str(&format!("  {d}\n"));
                }
                self.out.push_str(&kernel_out);
                self.out.push_str("}\n\n");
                let after_barrier = self.execution.memory.launches()[launches.len()]
                    .predecessor
                    .is_some();
                launches.push(Launch {
                    kernel,
                    threadgroups: dispatch.groups,
                    threads_per_threadgroup: dispatch.threads_per_group,
                    after_barrier,
                    dispatch: Some(dispatch.clone()),
                    tiles: self.finish_memory()?,
                    declared_threadgroup_bytes: shared_bytes,
                });
            }
        }
        Ok(Emitted {
            terminal: self.terminal,
            source: header + &self.out,
            launches,
            buffers: self.buffers,
            scalars: self.scalars,
            scratch: self
                .execution
                .memory
                .scratch()
                .iter()
                .map(|s| s.bytes)
                .collect(),
            status_slot: Some(status_slot),
            alias_pairs: Vec::new(),
        })
    }

    fn emit_prologue(&mut self, vars: &[VarId], dispatch: &GroupDispatch) -> Result<(), String> {
        use crate::support::{Helper, LaunchOperation as O, LaunchValue as V};
        use crate::terminal::{Expression as E, Statement as S, Type as T};
        let program = self.execution.memory.launches()[self.memory_launch]
            .prologue
            .instantiate(dispatch)?;
        if vars.len() != program.coordinates.len() {
            return Err("parallel variables do not match launch coordinates".into());
        }
        let value = |v: V| match v {
            V::Group => E::variable("tg_pos.x", T::U32),
            V::Subgroup => E::variable("sg_id", T::U32),
            V::Constant(n) => E::Integer(i64::from(n), T::U32),
            V::Result(n) => E::variable(format!("seismic_entry_{n}"), T::U32),
        };
        for (n, step) in program.steps.iter().enumerate() {
            let expression = match step.operation {
                O::Add(a, b) => E::binary(BinaryOp::Add, value(a), value(b), T::U32),
                O::Multiply(a, b) => E::binary(BinaryOp::Mul, value(a), value(b), T::U32),
                O::Divide(a, b) => E::binary(BinaryOp::Div, value(a), value(b), T::U32),
                O::Remainder(a, b) => E::binary(BinaryOp::Rem, value(a), value(b), T::U32),
                O::SignedIndex(a) => value(a).cast(T::I32),
                O::Live(a, b) => E::Helper(Helper::WorkItemLive, vec![value(a), value(b)], T::Bool),
            };
            self.target(S::Let {
                name: format!("seismic_entry_{n}"),
                ty: expression.ty(),
                value: expression,
            });
            if n == program.guard {
                self.target(S::ReturnIf(E::Unary(
                    UnaryOp::Not,
                    Box::new(E::variable(format!("seismic_entry_{n}"), T::Bool)),
                    T::Bool,
                )));
            }
        }
        self.target(S::Let {
            name: "item".into(),
            ty: T::U32,
            value: value(program.item),
        });
        if let Some(part) = program.part {
            self.target(S::Let {
                name: "part".into(),
                ty: T::I32,
                value: value(part),
            });
        }
        for (&v, coordinate) in vars.iter().zip(program.coordinates) {
            let name = self.index_name(v);
            self.target(S::Let {
                name,
                ty: T::I32,
                value: value(coordinate).cast(T::I32),
            });
        }
        Ok(())
    }

    fn index_name(&mut self, v: VarId) -> String {
        let var = &self.vars()[v];
        let VarKind::Index(Atom::Param(atom)) = &var.kind else {
            panic!("not an index")
        };
        let name = format!("{}_{}", sanitize(&var.name), v);
        self.names.insert(atom.clone(), name.clone());
        self.names.insert(name.clone(), name.clone());
        self.real
            .insert(v, Realization::Index { name: name.clone() });
        name
    }

    fn block(&mut self, stmts: &[Stmt]) -> Result<(), String> {
        for s in stmts {
            self.stmt(s)?;
        }
        Ok(())
    }

    fn scoped(&mut self, body: impl FnOnce(&mut Self) -> Result<(), String>) -> Result<(), String> {
        // These maps describe lexical bindings, not mutable storage contents.
        // Writes to an enclosing scalar or tile still execute, while a child's
        // captured dimensions and temporary declarations cannot escape its scope.
        let bindings = (
            self.names.clone(),
            self.expressions.clone(),
            self.real.clone(),
        );
        let result = body(self);
        (self.names, self.expressions, self.real) = bindings;
        result
    }

    fn stmt(&mut self, s: &Stmt) -> Result<(), String> {
        let previous = (self.current_operation, self.collective_ordinal);
        self.current_operation = s.id;
        self.collective_ordinal = 0;
        let result = if matches!(
            s.kind,
            StmtKind::Range { .. }
                | StmtKind::Lanes { .. }
                | StmtKind::LoadLoop { .. }
                | StmtKind::Owned { .. }
        ) {
            self.scoped(|printer| printer.stmt_inner(s))
        } else {
            self.stmt_inner(s)
        };
        (self.current_operation, self.collective_ordinal) = previous;
        result
    }
    fn stmt_inner(&mut self, s: &Stmt) -> Result<(), String> {
        match &s.kind {
            StmtKind::Reduction(_) => {
                Err("selected structured reduction reached printing without materialization".into())
            }
            StmtKind::Parallel { .. } => Err("nested `parallel` is not supported".into()),
            StmtKind::Range { var, lo, hi, body } => {
                let name = self.index_name(*var);
                let lo = self.target_sym(lo)?;
                let hi = self.target_sym(hi)?;
                self.target(crate::terminal::Statement::For {
                    name,
                    start: lo,
                    end: hi,
                    step: 1,
                });
                self.indent += 1;
                self.block(body)?;
                self.indent -= 1;
                self.target(crate::terminal::Statement::End);
                Ok(())
            }
            StmtKind::Lanes {
                var,
                extent,
                width,
                body,
            } => {
                let name = self.index_name(*var);
                let run = SUBGROUP
                    .checked_mul(*width)
                    .filter(|n| *n > 0)
                    .ok_or("invalid lane run extent")?;
                let extent = self.target_sym(extent)?.cast(TT::I64);
                let divisor = TE::Integer(run, TT::I64);
                let runs = TE::binary(
                    BinaryOp::Add,
                    TE::binary(BinaryOp::Div, extent.clone(), divisor.clone(), TT::I64),
                    TE::binary(
                        BinaryOp::Ne,
                        TE::binary(BinaryOp::Rem, extent.clone(), divisor, TT::I64),
                        TE::integer(0),
                        TT::Bool,
                    )
                    .cast(TT::I64),
                    TT::I64,
                )
                .cast(TT::I32);
                let t = self.fresh("t");
                let u = self.fresh("u");
                self.target(TS::For {
                    name: t.clone(),
                    start: TE::integer(0),
                    end: runs,
                    step: 1,
                });
                self.indent += 1;
                self.target(TS::For {
                    name: u.clone(),
                    start: TE::integer(0),
                    end: TE::integer(*width),
                    step: 1,
                });
                self.indent += 1;
                let coordinate = TE::binary(
                    BinaryOp::Add,
                    TE::binary(
                        BinaryOp::Add,
                        TE::binary(
                            BinaryOp::Mul,
                            TE::variable(t, TT::I32).cast(TT::I64),
                            TE::Integer(run, TT::I64),
                            TT::I64,
                        ),
                        TE::binary(
                            BinaryOp::Mul,
                            TE::variable("lane", TT::U32).cast(TT::I64),
                            TE::Integer(*width, TT::I64),
                            TT::I64,
                        ),
                        TT::I64,
                    ),
                    TE::variable(u, TT::I32).cast(TT::I64),
                    TT::I64,
                );
                self.target(TS::If(TE::binary(
                    BinaryOp::Lt,
                    coordinate.clone(),
                    extent,
                    TT::Bool,
                )));
                self.indent += 1;
                self.target(TS::Let {
                    name,
                    ty: TT::I32,
                    value: coordinate.cast(TT::I32),
                });
                self.block(body)?;
                self.indent -= 1;
                self.target(TS::End);
                self.indent -= 1;
                self.target(crate::terminal::Statement::End);
                self.indent -= 1;
                self.target(crate::terminal::Statement::End);
                let mut writes = HashSet::new();
                for statement in body {
                    seismic_lang::rewrite::writes(statement, &mut writes);
                }
                let mut writes = writes.into_iter().collect::<Vec<_>>();
                writes.sort_unstable();
                for variable in writes {
                    self.barrier(BarrierSite {
                        operation: s.id.ok_or("lane domain has no identity")?,
                        variable,
                        purpose: BarrierPurpose::Lanes,
                    })?;
                }
                Ok(())
            }
            StmtKind::LoadLoop {
                vars,
                offset: stream_offset,
                views,
                domain,
                axes,
                piece,
                capacity,
                modes,
                body,
            } => {
                let operation = s.id.ok_or("stream has no operation identity")?;
                let modes = modes
                    .as_ref()
                    .filter(|m| m.len() == vars.len())
                    .ok_or("unresolved stream loads reached Metal emission")?;
                if axes.len() != views.len() || vars.len() != views.len() {
                    return Err("stream transfer binding geometry differs".into());
                }
                let domain_shape = self.view_shape(&domain.view)?;
                let domain_extent = domain_shape
                    .get(domain.axis)
                    .ok_or("stream domain axis exceeds rank")?
                    .clone();
                let realized: Vec<Realization> = views
                    .iter()
                    .map(|v| self.view_of(v))
                    .collect::<Result<_, _>>()?;
                for (view, &axis) in realized.iter().zip(axes) {
                    let Realization::View { shape, .. } = view else {
                        unreachable!()
                    };
                    let extent = shape.get(axis).ok_or("stream transfer axis exceeds rank")?;
                    if extent != &domain_extent {
                        self.target(TS::Evaluate(TE::Helper(
                            crate::support::Helper::Validate,
                            vec![TE::binary(
                                BinaryOp::Eq,
                                self.target_sym(extent)?,
                                self.target_sym(&domain_extent)?,
                                TT::Bool,
                            )],
                            TT::Bool,
                        )));
                    }
                }
                match capacity {
                    None => {
                        if let Some(var) = stream_offset {
                            let name = self.index_name(*var);
                            self.target(TS::Let {
                                name,
                                ty: TT::I32,
                                value: TE::integer(0),
                            });
                        }
                        if domain_extent.as_constant() == Some(0) {
                            return Ok(());
                        }
                        for ((v, r), mode) in vars.iter().zip(realized).zip(modes) {
                            self.bind_stream_load(*v, r, *mode, operation)?;
                        }
                        self.block(body)
                    }
                    Some(cap) => {
                        if *cap <= 0 || *cap > i64::from(i32::MAX) {
                            return Err("Metal stream capacity must fit a positive i32".into());
                        }
                        let Atom::Param(pname) = piece else {
                            unreachable!()
                        };
                        let ext = self.target_sym(&domain_extent)?.cast(TT::I32);
                        let chunk = self.fresh("chunk");
                        let pe = self.fresh("pe");
                        self.names.insert(pname.clone(), pe.clone());
                        let chunks = TE::binary(
                            BinaryOp::Add,
                            TE::binary(BinaryOp::Div, ext.clone(), TE::integer(*cap), TT::I32),
                            TE::binary(
                                BinaryOp::Ne,
                                TE::binary(BinaryOp::Rem, ext.clone(), TE::integer(*cap), TT::I32),
                                TE::integer(0),
                                TT::Bool,
                            )
                            .cast(TT::I32),
                            TT::I32,
                        );
                        self.target(TS::For {
                            name: chunk.clone(),
                            start: TE::integer(0),
                            end: chunks,
                            step: 1,
                        });
                        self.indent += 1;
                        if let Some(var) = stream_offset {
                            let name = self.index_name(*var);
                            self.target(TS::Let {
                                name,
                                ty: TT::I32,
                                value: TE::binary(
                                    BinaryOp::Mul,
                                    TE::variable(&chunk, TT::I32),
                                    TE::integer(*cap),
                                    TT::I32,
                                ),
                            });
                        }
                        let remaining = TE::binary(
                            BinaryOp::Sub,
                            ext,
                            TE::binary(
                                BinaryOp::Mul,
                                TE::variable(&chunk, TT::I32),
                                TE::integer(*cap),
                                TT::I32,
                            ),
                            TT::I32,
                        );
                        self.target(TS::Let {
                            name: pe,
                            ty: TT::I32,
                            value: TE::Select(
                                Box::new(TE::binary(
                                    BinaryOp::Lt,
                                    remaining.clone(),
                                    TE::integer(*cap),
                                    TT::Bool,
                                )),
                                Box::new(remaining),
                                Box::new(TE::integer(*cap)),
                            ),
                        });
                        for (((v, r), mode), axis) in vars.iter().zip(realized).zip(modes).zip(axes)
                        {
                            let Realization::View {
                                space,
                                param,
                                elem,
                                offset,
                                strides,
                                mut shape,
                            } = r
                            else {
                                unreachable!()
                            };
                            let offset = offset.add(
                                &Sym::param(&chunk)
                                    .mul(&Sym::constant(*cap))
                                    .mul(&strides[*axis]),
                            );
                            shape[*axis] = Sym::atom(piece.clone());
                            self.bind_stream_load(
                                *v,
                                Realization::View {
                                    space,
                                    param,
                                    elem,
                                    offset,
                                    strides,
                                    shape,
                                },
                                *mode,
                                operation,
                            )?;
                        }
                        self.block(body)?;
                        self.indent -= 1;
                        self.target(crate::terminal::Statement::End);
                        Ok(())
                    }
                }
            }
            StmtKind::Owned { vars, tile, body } => {
                let ExprKind::Var(tv) = tile.kind else {
                    return self.owned_view(
                        vars,
                        tile,
                        body,
                        s.id.ok_or("owned domain has no identity")?,
                    );
                };
                let real = self
                    .real
                    .get(&tv)
                    .cloned()
                    .ok_or("owned() over an unrealized tile")?;
                let names: Vec<String> = vars.iter().map(|v| self.index_name(*v)).collect();
                match real {
                    Realization::Replicated { dims, .. } | Realization::Geometry { dims, .. } => {
                        for (n, d) in names.iter().zip(&dims) {
                            self.target(TS::For {
                                name: n.clone(),
                                start: TE::integer(0),
                                end: d.value.clone(),
                                step: 1,
                            });
                            self.indent += 1;
                        }
                        self.owned_ctx.push((tv, names.clone(), None));
                        self.block(body)?;
                        self.owned_ctx.pop();
                        for _ in &dims {
                            self.indent -= 1;
                            self.target(crate::terminal::Statement::End);
                        }
                        Ok(())
                    }
                    Realization::Shared { dims, .. } => {
                        // Threadgroup memory: elements are spread over the lanes in the same
                        // order a distributed tile uses, then a barrier publishes them.
                        let n_cap: i64 = dims.iter().map(|d| d.cap).product();
                        let slots = n_cap / SUBGROUP + i64::from(n_cap % SUBGROUP != 0);
                        let j = self.fresh("slot");
                        let e = self.fresh("e");
                        self.target(TS::For {
                            name: j.clone(),
                            start: TE::integer(0),
                            end: TE::integer(slots),
                            step: 1,
                        });
                        self.indent += 1;
                        self.target(TS::Let {
                            name: e.clone(),
                            ty: TT::I32,
                            value: TE::binary(
                                BinaryOp::Add,
                                TE::variable("lane", TT::U32).cast(TT::I32),
                                TE::binary(
                                    BinaryOp::Mul,
                                    TE::integer(SUBGROUP),
                                    TE::variable(j.clone(), TT::I32),
                                    TT::I32,
                                ),
                                TT::I32,
                            ),
                        });
                        let guard = self.distributed_guard(&e, &dims, &names);
                        self.target(TS::If(guard));
                        self.indent += 1;
                        self.owned_ctx.push((tv, names.clone(), Some(j.clone())));
                        self.block(body)?;
                        self.owned_ctx.pop();
                        self.indent -= 1;
                        self.target(crate::terminal::Statement::End);
                        self.indent -= 1;
                        self.target(crate::terminal::Statement::End);
                        Ok(())
                    }
                    Realization::Distributed { dims, slots, .. } => {
                        let j = self.fresh("slot");
                        let e = self.fresh("e");
                        self.target(TS::For {
                            name: j.clone(),
                            start: TE::integer(0),
                            end: TE::integer(slots),
                            step: 1,
                        });
                        self.indent += 1;
                        self.target(TS::Let {
                            name: e.clone(),
                            ty: TT::I32,
                            value: TE::binary(
                                BinaryOp::Add,
                                TE::variable("lane", TT::U32).cast(TT::I32),
                                TE::binary(
                                    BinaryOp::Mul,
                                    TE::integer(SUBGROUP),
                                    TE::variable(j.clone(), TT::I32),
                                    TT::I32,
                                ),
                                TT::I32,
                            ),
                        });
                        let guard = self.distributed_guard(&e, &dims, &names);
                        self.target(TS::If(guard));
                        self.indent += 1;
                        self.owned_ctx.push((tv, names.clone(), Some(j.clone())));
                        self.block(body)?;
                        self.owned_ctx.pop();
                        self.indent -= 1;
                        self.target(crate::terminal::Statement::End);
                        self.indent -= 1;
                        self.target(crate::terminal::Statement::End);
                        Ok(())
                    }
                    other => Err(format!("owned() over {other:?} is not supported")),
                }?;
                self.barrier(BarrierSite {
                    operation: s.id.ok_or("owned domain has no identity")?,
                    variable: tv,
                    purpose: BarrierPurpose::Owned,
                })
            }
            StmtKind::If { cond, then, els } => {
                let c = self.target_expr(cond)?;
                self.target(crate::terminal::Statement::If(c));
                self.indent += 1;
                self.scoped(|printer| printer.block(then))?;
                self.indent -= 1;
                if els.is_empty() {
                    self.target(crate::terminal::Statement::End);
                } else {
                    self.target(crate::terminal::Statement::Else);
                    self.indent += 1;
                    self.scoped(|printer| printer.block(els))?;
                    self.indent -= 1;
                    self.target(crate::terminal::Statement::End);
                }
                Ok(())
            }
            StmtKind::Assign { target, op, value } => self.assign(
                target,
                *op,
                value,
                s.id.ok_or("emission requires normalized operation identities")?,
            ),
            StmtKind::Expr(e) => match &e.kind {
                ExprKind::Builtin {
                    name: Builtin::Store,
                    args,
                } => self.store(&args[0], &args[1]),
                ExprKind::Builtin {
                    name: Builtin::Atomic,
                    ..
                } => Err("atomic is not yet supported on Metal".into()),
                ExprKind::Intrinsic { op: name, args } => {
                    self.intrinsic_stmt(name, args, s.id.ok_or("intrinsic has no identity")?)
                }
                _ => {
                    let s = self.target_expr(e)?;
                    self.target(crate::terminal::Statement::Evaluate(s));
                    Ok(())
                }
            },
        }
    }

    /// Declares the index variables of a distributed element and returns the validity guard.
    fn distributed_guard(&mut self, e: &str, dims: &[Dim], names: &[String]) -> TE {
        let n_cap: i64 = dims.iter().map(|d| d.cap).product();
        if n_cap == 0 {
            for name in names {
                self.target(TS::Let {
                    name: name.clone(),
                    ty: TT::I32,
                    value: TE::integer(0),
                });
            }
            return TE::Integer(0, TT::Bool);
        }
        let mut stride = n_cap;
        let mut guard = TE::binary(
            BinaryOp::Lt,
            TE::variable(e, TT::I32),
            TE::integer(n_cap),
            TT::Bool,
        );
        for (name, d) in names.iter().zip(dims) {
            stride /= d.cap;
            self.target(TS::Let {
                name: name.clone(),
                ty: TT::I32,
                value: TE::binary(
                    BinaryOp::Rem,
                    TE::binary(
                        BinaryOp::Div,
                        TE::variable(e, TT::I32),
                        TE::integer(stride),
                        TT::I32,
                    ),
                    TE::integer(d.cap),
                    TT::I32,
                ),
            });
            if !d.is_static() {
                guard = TE::binary(
                    BinaryOp::And,
                    guard,
                    TE::binary(
                        BinaryOp::Lt,
                        TE::variable(name, TT::I32),
                        d.value.clone(),
                        TT::Bool,
                    ),
                    TT::Bool,
                );
            }
        }
        guard
    }

    fn view_of(&mut self, e: &Expr) -> Result<Realization, String> {
        match &e.kind {
            ExprKind::Builtin {
                name: Builtin::Reshape,
                args,
            } => {
                let Realization::View {
                    space,
                    param,
                    elem,
                    offset,
                    strides,
                    shape,
                } = self.view_of(&args[0])?
                else {
                    return Err("reshape requires a view".into());
                };
                let target =
                    e.ty.shaped()
                        .ok_or("reshape requires shaped result")?
                        .shape
                        .clone();
                let strides = Self::reshape_geometry(&shape, &strides, &target)?;
                Ok(Realization::View {
                    space,
                    param,
                    elem,
                    offset,
                    strides,
                    shape: target,
                })
            }

            ExprKind::Var(v) => match self.real.get(v).cloned() {
                Some(Realization::Param { name, shape, elem }) => {
                    let strides = row_major_syms(&shape);
                    Ok(Realization::View {
                        space: TSpa::Device,
                        param: name,
                        elem,
                        offset: Sym::constant(0),
                        strides,
                        shape: shape.iter().map(|d| Sym::constant(*d)).collect(),
                    })
                }
                Some(Realization::Replicated { name, dims, dtype })
                | Some(Realization::Shared { name, dims, dtype }) => {
                    let space = if matches!(self.real.get(v), Some(Realization::Shared { .. })) {
                        TSpa::Threadgroup
                    } else {
                        TSpa::Private
                    };
                    Ok(Realization::View {
                        space,
                        param: name,
                        elem: Elem::Dtype(dtype),
                        offset: Sym::constant(0),
                        strides: row_major_syms(&dims.iter().map(|d| d.cap).collect::<Vec<_>>()),
                        shape: self.realized_shape(&dims),
                    })
                }
                Some(r @ Realization::View { .. }) => Ok(r),
                other => Err(format!("not a tensor view: {other:?}")),
            },
            ExprKind::Index { base, indices } => {
                let Realization::View {
                    space,
                    param,
                    elem,
                    offset,
                    strides,
                    shape,
                } = self.view_of(base)?
                else {
                    unreachable!()
                };
                let geometry = self.view_geometry(e, indices, &shape)?;
                let off = geometry
                    .starts
                    .iter()
                    .zip(&strides)
                    .fold(offset, |off, (start, stride)| off.add(&start.mul(stride)));
                Ok(Realization::View {
                    space,
                    param,
                    elem,
                    offset: off,
                    strides: geometry.axes.iter().map(|&a| strides[a].clone()).collect(),
                    shape: geometry.shape,
                })
            }
            ExprKind::Transpose(inner) => {
                let Realization::View {
                    space,
                    param,
                    elem,
                    offset,
                    strides,
                    shape,
                } = self.view_of(inner)?
                else {
                    unreachable!()
                };
                Ok(Realization::View {
                    space,
                    param,
                    elem,
                    offset,
                    strides: vec![strides[1].clone(), strides[0].clone()],
                    shape: vec![shape[1].clone(), shape[0].clone()],
                })
            }
            _ => Err("expression is not a tensor view".into()),
        }
    }

    fn realized_shape(&mut self, dims: &[Dim]) -> Vec<Sym> {
        dims.iter()
            .map(|d| {
                if d.is_static() {
                    Sym::constant(d.cap)
                } else {
                    // Retain the captured dimension, not a checked atom that a
                    // later view evaluation can bind to a different occurrence.
                    let name = self.fresh("view_extent");
                    self.expressions.insert(name.clone(), d.value.clone());
                    Sym::param(&name)
                }
            })
            .collect()
    }

    fn view_geometry(
        &mut self,
        view: &Expr,
        indices: &[Index],
        shape: &[Sym],
    ) -> Result<ViewGeometry, String> {
        let result = view.ty.shaped().ok_or("indexed view has no result shape")?;
        let mut geometry = ViewGeometry {
            starts: Vec::new(),
            axes: Vec::new(),
            shape: Vec::new(),
        };
        for (axis, extent) in shape.iter().enumerate() {
            match indices.get(axis) {
                Some(Index::Point(point)) => {
                    let point = self.int_value(point)?;
                    geometry.starts.push(self.checked_index(&point, extent)?);
                }
                Some(Index::Slice { start, end }) => {
                    let (safe, length) = self.slice_bounds(start.as_ref(), end.as_ref(), extent)?;
                    let checked = result
                        .shape
                        .get(geometry.shape.len())
                        .ok_or("indexed view rank mismatch")?;
                    let atoms = checked.atoms();
                    if let [Atom::Param(atom)] = atoms.as_slice() {
                        if *checked == Sym::param(atom) {
                            // The checked shape names the current evaluation.
                            // Captured bounds are symbolic too; a sibling branch
                            // must bind its own extent before allocating a tile.
                            // Existing realizations retain their captured Dim.
                            self.names.insert(atom.clone(), self.sym(&length)?);
                        }
                    }
                    geometry.starts.push(safe);
                    geometry.axes.push(axis);
                    // Each evaluated view retains its own extent, including in
                    // sibling branches that share a checked dynamic atom.
                    geometry.shape.push(if checked.as_constant().is_some() {
                        checked.clone()
                    } else {
                        length
                    });
                }
                None => {
                    geometry.starts.push(Sym::constant(0));
                    geometry.axes.push(axis);
                    geometry.shape.push(extent.clone());
                }
            }
        }
        Ok(geometry)
    }

    fn view_shape(&mut self, view: &Expr) -> Result<Vec<Sym>, String> {
        self.geometry_of(view).map(|(shape, _)| shape)
    }

    fn reshape_geometry(
        shape: &[Sym],
        strides: &[Sym],
        target: &[Sym],
    ) -> Result<Vec<Sym>, String> {
        let resolve = |dims: &[Sym]| {
            dims.iter()
                .map(|s| {
                    s.as_constant().ok_or_else(|| {
                        "reshape requires statically resolved extents and strides".to_string()
                    })
                })
                .collect::<Result<Vec<_>, _>>()
        };
        seismic_lang::layout::reshape_strides(
            &resolve(shape)?,
            &resolve(strides)?,
            &resolve(target)?,
        )
        .map(|strides| strides.into_iter().map(Sym::constant).collect())
    }

    /// Logical view evaluation retains layout checks and endpoint reads even
    /// when the underlying element data has no remaining consumers.
    fn geometry_of(&mut self, view: &Expr) -> Result<(Vec<Sym>, Vec<Sym>), String> {
        match &view.kind {
            ExprKind::Var(v) => match self.real.get(v).cloned() {
                Some(Realization::Param { shape, .. }) => {
                    let strides = row_major_syms(&shape);
                    Ok((shape.into_iter().map(Sym::constant).collect(), strides))
                }
                Some(Realization::View { shape, strides, .. }) => Ok((shape, strides)),
                Some(Realization::Geometry { dims, strides }) => {
                    Ok((self.realized_shape(&dims), strides))
                }
                Some(Realization::Replicated { dims, .. })
                | Some(Realization::Distributed { dims, .. })
                | Some(Realization::Shared { dims, .. }) => {
                    let strides = row_major_syms(&dims.iter().map(|d| d.cap).collect::<Vec<_>>());
                    Ok((self.realized_shape(&dims), strides))
                }
                _ => Err("extent requires a realized shaped value".into()),
            },
            ExprKind::TileAlloc { shape, .. } => {
                let dims = self.dims(shape)?;
                let strides = row_major_syms(&dims.iter().map(|d| d.cap).collect::<Vec<_>>());
                Ok((self.realized_shape(&dims), strides))
            }
            ExprKind::Load { view: source, .. } => {
                let (shape, _) = self.geometry_of(source)?;
                let capacities = view
                    .ty
                    .shaped()
                    .ok_or("load has no geometry")?
                    .shape
                    .iter()
                    .map(|dim| self.cap(dim))
                    .collect::<Result<Vec<_>, _>>()?;
                Ok((shape, row_major_syms(&capacities)))
            }
            ExprKind::Index { base, indices } => {
                let (shape, strides) = self.geometry_of(base)?;
                let geometry = self.view_geometry(view, indices, &shape)?;
                Ok((
                    geometry.shape,
                    geometry
                        .axes
                        .iter()
                        .map(|&axis| strides[axis].clone())
                        .collect(),
                ))
            }
            ExprKind::Transpose(base) => {
                let (mut shape, mut strides) = self.geometry_of(base)?;
                if shape.len() != 2 {
                    return Err("transpose requires rank two".into());
                }
                shape.swap(0, 1);
                strides.swap(0, 1);
                Ok((shape, strides))
            }
            ExprKind::Builtin {
                name: Builtin::Reshape,
                args,
            } => {
                let (shape, strides) = self.geometry_of(&args[0])?;
                let target = &view.ty.shaped().ok_or("reshape has no shape")?.shape;
                let strides = Self::reshape_geometry(&shape, &strides, target)?;
                Ok((target.clone(), strides))
            }
            _ => Err("extent requires a shaped view".into()),
        }
    }

    /// An integer-valued expression as a symbol: static syms pass through; dynamic values are
    /// evaluated into a named C variable that the symbol refers to.
    fn int_value(&mut self, e: &Expr) -> Result<Sym, String> {
        if let Some(s) = &e.sym {
            if !seismic_lang::effects::can_substitute_symbolic_value(e) {
                let value = self.target_expr(e)?;
                self.target(TS::Evaluate(value));
            }
            // Retain checked coordinate equality for participant ownership,
            // even when evaluating the expression also establishes metadata.
            return Ok(s.clone());
        }
        let value = self.target_expr(e)?;
        let name = self.fresh("iv");
        self.target(TS::Let {
            name: name.clone(),
            ty: TT::I32,
            value,
        });
        Ok(Sym::param(&name))
    }

    /// Source data-dependent slices clamp into their parent. Symbolic bounds
    /// carry checked source conditions and retain their validity checks here.
    /// Both shaped views and scalar accesses compose these same coordinates.
    fn slice_bounds(
        &mut self,
        start: Option<&Expr>,
        end: Option<&Expr>,
        extent: &Sym,
    ) -> Result<(Sym, Sym), String> {
        let dynamic =
            start.is_some_and(|e| e.sym.is_none()) || end.is_some_and(|e| e.sym.is_none());
        let start = start
            .map(|e| self.int_value(e))
            .transpose()?
            .unwrap_or(Sym::constant(0));
        let end = end
            .map(|e| self.int_value(e))
            .transpose()?
            .unwrap_or_else(|| extent.clone());
        let begin_value = self.target_sym(&start)?.cast(TT::I64);
        let end_value = self.target_sym(&end)?.cast(TT::I64);
        let extent_value = self.target_sym(extent)?.cast(TT::I64);
        let safe = self.fresh("slice_start");
        let length = self.fresh("slice_extent");
        if dynamic {
            let clamp = |value: TE, upper: TE| {
                TE::Select(
                    Box::new(TE::binary(
                        BinaryOp::Lt,
                        value.clone(),
                        TE::Integer(0, TT::I64),
                        TT::Bool,
                    )),
                    Box::new(TE::Integer(0, TT::I64)),
                    Box::new(TE::Select(
                        Box::new(TE::binary(
                            BinaryOp::Gt,
                            value.clone(),
                            upper.clone(),
                            TT::Bool,
                        )),
                        Box::new(upper),
                        Box::new(value),
                    )),
                )
            };
            let stop = self.fresh("slice_end");
            self.target(TS::Let {
                name: stop.clone(),
                ty: TT::I64,
                value: clamp(end_value, extent_value),
            });
            self.target(TS::Let {
                name: safe.clone(),
                ty: TT::I64,
                value: clamp(begin_value, TE::variable(&stop, TT::I64)),
            });
            self.target(TS::Let {
                name: length.clone(),
                ty: TT::I32,
                value: TE::binary(
                    BinaryOp::Sub,
                    TE::variable(stop, TT::I64),
                    TE::variable(&safe, TT::I64),
                    TT::I64,
                )
                .cast(TT::I32),
            });
        } else {
            let valid = self.fresh("slice_valid");
            self.target(TS::Let {
                name: valid.clone(),
                ty: TT::Bool,
                value: TE::Helper(
                    crate::support::Helper::SliceValid,
                    vec![begin_value.clone(), end_value.clone(), extent_value],
                    TT::Bool,
                ),
            });
            self.target(TS::Let {
                name: safe.clone(),
                ty: TT::I64,
                value: TE::Helper(
                    crate::support::Helper::SliceStart,
                    vec![TE::variable(&valid, TT::Bool), begin_value.clone()],
                    TT::I64,
                ),
            });
            self.target(TS::Let {
                name: length.clone(),
                ty: TT::I32,
                value: TE::Helper(
                    crate::support::Helper::SliceExtent,
                    vec![TE::variable(valid, TT::Bool), begin_value, end_value],
                    TT::I32,
                ),
            });
        }
        self.names.insert(safe.clone(), safe.clone());
        self.names.insert(length.clone(), length.clone());
        Ok((Sym::param(&safe), Sym::param(&length)))
    }

    /// Compose a logical view's coordinates into its existing allocation. Each
    /// intermediate slice is checked before its offset is used.
    fn tile_coordinates(
        &mut self,
        view: &Expr,
        points: Vec<Sym>,
    ) -> Result<(VarId, Vec<Sym>), String> {
        match &view.kind {
            ExprKind::Var(v) => Ok((*v, points)),
            ExprKind::Transpose(base) => {
                if points.len() != 2 {
                    return Err("transpose coordinate rank is not two".into());
                }
                self.tile_coordinates(base, vec![points[1].clone(), points[0].clone()])
            }
            ExprKind::Builtin {
                name: Builtin::Reshape,
                args,
            } => {
                let base = args.first().ok_or("reshape has no source")?;
                let shape = &view.ty.shaped().ok_or("reshape has no result shape")?.shape;
                if points.len() != shape.len() {
                    return Err("reshape coordinate rank differs".into());
                }
                let mut flat = Sym::constant(0);
                for (point, extent) in points.iter().zip(shape) {
                    flat = flat.mul(extent).add(&self.checked_index(point, extent)?);
                }
                let source_shape = &base.ty.shaped().ok_or("reshape has no source shape")?.shape;
                let mut stride = Sym::constant(1);
                let mut source = Vec::with_capacity(source_shape.len());
                for extent in source_shape.iter().rev() {
                    source.push(flat.quot(&stride).rem(extent));
                    stride = stride.mul(extent);
                }
                source.reverse();
                self.tile_coordinates(base, source)
            }
            ExprKind::Index { base, indices } => {
                let shape = &base.ty.shaped().ok_or("indexed tile has no shape")?.shape;
                let mut local = points.into_iter();
                let mut composed = Vec::new();
                for (axis, extent) in shape.iter().enumerate() {
                    composed.push(match indices.get(axis) {
                        Some(Index::Point(point)) => {
                            let point = self.int_value(point)?;
                            self.checked_index(&point, extent)?
                        }
                        Some(Index::Slice { start, end }) if start.is_some() || end.is_some() => {
                            let (safe, length) =
                                self.slice_bounds(start.as_ref(), end.as_ref(), extent)?;
                            let point = local
                                .next()
                                .ok_or("tile view coordinate rank is too small")?;
                            safe.add(&self.checked_index(&point, &length)?)
                        }
                        _ => {
                            let point = local
                                .next()
                                .ok_or("tile view coordinate rank is too small")?;
                            self.checked_index(&point, extent)?
                        }
                    });
                }
                if local.next().is_some() {
                    return Err("tile view coordinate rank is too large".into());
                }
                self.tile_coordinates(base, composed)
            }
            _ => Err("tile coordinate view is not supported".into()),
        }
    }
    fn indexed_value(&mut self, base: &Expr, indices: &[Index]) -> Result<TE, String> {
        let points = indices
            .iter()
            .map(|index| match index {
                Index::Point(point) => self.int_value(point),
                _ => Err("scalar tile read contains a slice".into()),
            })
            .collect::<Result<Vec<_>, _>>()?;
        let (root, points) = self.tile_coordinates(base, points)?;
        let indices = points
            .into_iter()
            .map(|point| {
                Index::Point(Expr {
                    kind: ExprKind::ShapeParam(point.to_string()),
                    ty: Ty::Scalar(DType::I32),
                    sym: Some(point),
                    span: base.span,
                })
            })
            .collect::<Vec<_>>();
        self.target_tile_element(root, &indices)
    }
    fn owned_view(
        &mut self,
        vars: &[VarId],
        tile: &Expr,
        body: &[Stmt],
        operation: OperationId,
    ) -> Result<(), String> {
        let root = crate::storage::tile_root(tile).ok_or("owned view has no allocation")?;
        let shared = match self.real.get(&root) {
            Some(Realization::Replicated { .. } | Realization::Geometry { .. }) => false,
            Some(Realization::Shared { .. }) => true,
            _ => return Err("owned view requires addressable selected tile storage".into()),
        };
        let shape = self.view_shape(tile)?;
        let shaped = tile.ty.shaped().ok_or("owned view has no shape")?;
        let dims = shape
            .iter()
            .zip(&shaped.shape)
            .map(|(actual, checked)| {
                Ok(Dim {
                    cap: self.cap(checked)?,
                    ext: self.sym(actual)?,
                    value: self.target_sym(actual)?,
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        let count = dims
            .iter()
            .try_fold(1i64, |n, d| n.checked_mul(d.cap))
            .ok_or("owned view capacity overflow")?;
        if dims.len() != vars.len() {
            return Err("owned view coordinate rank differs".into());
        }
        let flat = self.fresh("view_element");
        self.target(TS::For {
            name: flat.clone(),
            start: if shared {
                TE::variable("lane", TT::U32).cast(TT::I32)
            } else {
                TE::integer(0)
            },
            end: TE::integer(count),
            step: if shared { SUBGROUP } else { 1 },
        });
        self.indent += 1;
        let names = vars.iter().map(|v| self.index_name(*v)).collect::<Vec<_>>();
        let guard = self.distributed_guard(&flat, &dims, &names);
        self.target(TS::If(guard));
        self.indent += 1;
        self.block(body)?;
        self.indent -= 1;
        self.target(TS::End);
        self.indent -= 1;
        self.target(TS::End);
        self.barrier(BarrierSite {
            operation,
            variable: root,
            purpose: BarrierPurpose::Owned,
        })
    }

    fn emit_storage(&mut self, dispatch: &GroupDispatch) -> Result<(), String> {
        let slots = self.execution.memory.launches()[self.memory_launch]
            .slots
            .clone();
        for declaration in slots {
            let layout = declaration.layout(dispatch)?;
            if declaration.placement == TilePlacement::GroupShared {
                self.shared_decls.push((
                    format!(
                        "threadgroup {} {}[{}];",
                        ctype(declaration.dtype),
                        declaration.symbol,
                        layout.shared_elements_per_item * dispatch.items_per_group
                    ),
                    layout.shared_bytes_per_group,
                ));
            } else {
                self.target(TS::Array {
                    name: declaration.symbol,
                    ty: declaration.dtype.into(),
                    elements: layout.private_elements_per_lane,
                });
            }
        }
        Ok(())
    }
    fn allocation(&mut self, id: AllocationId) -> Result<TileDeclaration, String> {
        let launch = self
            .execution
            .memory
            .launches()
            .get(self.memory_launch)
            .ok_or("allocation launch is missing")?;
        let allocation = launch
            .arrays
            .get(self.tile_declarations.len())
            .ok_or("unplanned tile allocation reached emission")?;
        if allocation.id != id {
            return Err(format!(
                "allocation site mismatch: requested {id:?}, planned {:?}",
                allocation.id
            ));
        }
        let declaration = allocation.declaration.clone();
        self.tile_declarations.push(declaration.clone());
        self.barrier(BarrierSite {
            operation: id.operation,
            variable: id.variable,
            purpose: BarrierPurpose::Reuse(id.purpose),
        })?;
        Ok(declaration)
    }
    fn bind_allocation(&mut self, id: AllocationId, name: &str) -> Result<(), String> {
        let launch = &self.execution.memory.launches()[self.memory_launch];
        let allocation = launch
            .arrays
            .iter()
            .find(|a| a.id == id)
            .ok_or("bound allocation is absent")?;
        let slot = &launch.slots[allocation.slot];
        let shared = slot.placement == TilePlacement::GroupShared;
        let index = if shared {
            TE::binary(
                BinaryOp::Mul,
                TE::variable("sg_id", TT::U32),
                TE::Integer(slot.capacity.max(1) as i64, TT::U32),
                TT::U32,
            )
        } else {
            TE::Integer(0, TT::U32)
        };
        self.target(TS::Pointer {
            name: name.into(),
            base: slot.symbol.clone(),
            index,
            space: if shared {
                TSpa::Threadgroup
            } else {
                TSpa::Private
            },
            ty: slot.dtype.into(),
        });
        Ok(())
    }

    fn barrier(&mut self, site: BarrierSite) -> Result<(), String> {
        let launch = self
            .execution
            .memory
            .launches()
            .get(self.memory_launch)
            .ok_or("memory launch is missing")?;
        if let Some(barrier) = launch.barriers.get(&site) {
            if !self.emitted_barriers.insert(site) {
                return Err("duplicate emitted memory barrier".into());
            }
            match barrier.memory {
                MemorySpace::Threadgroup => self.target(TS::Barrier),
            }
        }
        Ok(())
    }

    fn collective(
        &mut self,
        operation: seismic_lang::intrinsics::Operation,
    ) -> Result<CollectiveImplementation, String> {
        let site = CollectiveSite {
            operation: self
                .current_operation
                .ok_or("collective emission has no operation identity")?,
            ordinal: self.collective_ordinal,
        };
        self.collective_ordinal += 1;
        let instruction = self
            .execution
            .memory
            .launches()
            .get(self.memory_launch)
            .and_then(|launch| launch.collectives.get(&site))
            .ok_or("unplanned collective reached emission")?;
        if instruction.implementation.operation() != operation {
            return Err("collective operation differs from the prepared implementation".into());
        }
        if !self.emitted_collectives.insert(site) {
            return Err("duplicate collective emission".into());
        }
        Ok(instruction.implementation.clone())
    }
    fn finish_memory(&mut self) -> Result<Vec<TileDeclaration>, String> {
        let launch = self
            .execution
            .memory
            .launches()
            .get(self.memory_launch)
            .ok_or("allocation launch is missing")?;
        if self.tile_declarations.len() != launch.arrays.len() {
            return Err("emission omitted planned tile allocations".into());
        }
        if self.emitted_barriers.len() != launch.barriers.len() {
            return Err("emission omitted planned memory barriers".into());
        }
        if self.emitted_collectives.len() != launch.collectives.len() {
            return Err("emission omitted prepared collective implementations".into());
        }
        self.emitted_collectives.clear();
        self.emitted_barriers.clear();
        self.memory_launch += 1;
        Ok(std::mem::take(&mut self.tile_declarations))
    }

    fn declare_tile(
        &mut self,
        v: VarId,
        shape: &[Sym],
        dtype: DType,
        operation: OperationId,
        purpose: Purpose,
    ) -> Result<Realization, String> {
        let dims = self.dims(shape)?;
        self.declare_tile_dims(v, dims, dtype, operation, purpose)
    }

    fn declare_tile_dims(
        &mut self,
        v: VarId,
        dims: Vec<Dim>,
        dtype: DType,
        operation: OperationId,
        purpose: Purpose,
    ) -> Result<Realization, String> {
        let n_cap = dims.iter().try_fold(1i64, |capacity, dim| {
            if dim.cap < 0 {
                return Err("negative tile extent");
            }
            capacity
                .checked_mul(dim.cap)
                .ok_or("tile capacity overflow")
        })?;
        let name = self.fresh(&format!("{}_{}", sanitize(&self.vars()[v].name), v));
        let declaration = self.allocation(AllocationId {
            operation,
            variable: v,
            purpose,
        })?;
        if declaration.capacity != n_cap as u64 || declaration.dtype != dtype {
            return Err(format!(
                "emitted tile {v} disagrees with its selected storage contract"
            ));
        }
        let placement = declaration.placement.clone();
        let dispatch = GroupDispatch::new(
            1,
            SUBGROUP as u64,
            self.simdgroups
                .try_into()
                .map_err(|_| "negative simdgroup count")?,
        )?;
        let layout = declaration.layout(&dispatch)?;
        self.bind_allocation(
            AllocationId {
                operation,
                variable: v,
                purpose,
            },
            &name,
        )?;
        let r = if placement == TilePlacement::GroupShared {
            Realization::Shared { name, dims, dtype }
        } else if placement == TilePlacement::Replicated {
            Realization::Replicated { name, dims, dtype }
        } else {
            let slots = i64::try_from(layout.private_elements_per_lane)
                .map_err(|_| "private tile extent overflow")?;
            Realization::Distributed {
                name,
                dims,
                dtype,
                slots,
            }
        };
        self.real.insert(v, r.clone());
        Ok(r)
    }

    fn coerce(&self, text: String, from: &Ty, to: DType) -> String {
        match scalar_dtype(from) {
            Some(d) if d != to => format!("{}({text})", ctype(to)),
            _ => text,
        }
    }
    fn arithmetic(op: BinaryOp, left: TE, right: TE, ty: TT) -> TE {
        // Seismic integer arithmetic wraps; signed overflow in MSL is undefined.
        // Keep the bit conversion and unsigned operation visible to accounting.
        if ty == TT::I32 && matches!(op, BinaryOp::Add | BinaryOp::Sub | BinaryOp::Mul) {
            TE::Bitcast(
                TT::I32,
                Box::new(TE::binary(
                    op,
                    TE::Bitcast(TT::U32, Box::new(left)),
                    TE::Bitcast(TT::U32, Box::new(right)),
                    TT::U32,
                )),
            )
        } else {
            TE::binary(op, left, right, ty)
        }
    }

    fn assign(
        &mut self,
        target: &Expr,
        op: AssignOp,
        value: &Expr,
        operation: OperationId,
    ) -> Result<(), String> {
        match &target.kind {
            ExprKind::Var(v) => {
                if matches!(target.ty, Ty::Tile(_)) && !self.execution.storage.requires_data(*v) {
                    if op != AssignOp::Assign {
                        return Err("compound geometry-only tile assignment".into());
                    }
                    let (shape, _) = self.geometry_of(value)?;
                    let checked = &target
                        .ty
                        .shaped()
                        .ok_or("geometry binding has no shape")?
                        .shape;
                    let dims = shape
                        .iter()
                        .zip(checked)
                        .map(|(actual, checked)| {
                            Ok(Dim {
                                cap: self.cap(checked)?,
                                ext: self.sym(actual)?,
                                value: self.target_sym(actual)?,
                            })
                        })
                        .collect::<Result<Vec<_>, String>>()?;
                    if let Some(Realization::Geometry { dims: previous, .. }) =
                        self.real.get(v).cloned()
                    {
                        if dims.len() != previous.len() {
                            return Err("geometry assignment rank mismatch".into());
                        }
                        for (source, destination) in dims.iter().zip(previous) {
                            if source.value != destination.value {
                                self.target(TS::Evaluate(TE::Helper(
                                    crate::support::Helper::Validate,
                                    vec![TE::binary(
                                        BinaryOp::Eq,
                                        source.value.clone(),
                                        destination.value,
                                        TT::Bool,
                                    )],
                                    TT::Bool,
                                )));
                            }
                        }
                    } else {
                        let strides =
                            row_major_syms(&dims.iter().map(|d| d.cap).collect::<Vec<_>>());
                        self.real
                            .insert(*v, Realization::Geometry { dims, strides });
                    }
                    return Ok(());
                }
                match &value.kind {
                    ExprKind::TileAlloc { shape, dtype } => {
                        let Elem::Dtype(dtype) = dtype else {
                            return Err("unresolved or packed local tile dtype".into());
                        };
                        let previous = self.real.get(v).cloned();
                        let allocated =
                            self.declare_tile(*v, shape, *dtype, operation, Purpose::Value)?;
                        if let Some(destination) = previous {
                            let dimensions = |value: &Realization| match value {
                                Realization::Replicated { dims, .. }
                                | Realization::Distributed { dims, .. }
                                | Realization::Shared { dims, .. } => Some(dims.clone()),
                                _ => None,
                            };
                            let source = dimensions(&allocated)
                                .ok_or("tile allocation has no owned dimensions")?;
                            let target = dimensions(&destination)
                                .ok_or("tile allocation reassignment needs owned storage")?;
                            if source.len() != target.len() {
                                return Err("tile allocation reassignment rank mismatch".into());
                            }
                            for (source, target) in source.iter().zip(&target) {
                                if source.value != target.value {
                                    self.target(TS::Evaluate(TE::Helper(
                                        crate::support::Helper::Validate,
                                        vec![TE::binary(
                                            BinaryOp::Eq,
                                            source.value.clone(),
                                            target.value.clone(),
                                            TT::Bool,
                                        )],
                                        TT::Bool,
                                    )));
                                }
                            }
                            // The selected allocation is still emitted and
                            // accounted. It has no initialized values to copy;
                            // subsequent writes update the enclosing binding.
                            // Placement/barriers use this same VarId, and its
                            // lifetime includes every use across both sites.
                            self.real.insert(*v, destination);
                        }
                        return Ok(());
                    }
                    ExprKind::Intrinsic { op: name, .. }
                        if *name == seismic_lang::intrinsics::Operation::Matrix =>
                    {
                        let CollectiveImplementation::Declare { fragment, layout } =
                            self.collective(*name)?
                        else {
                            return Err("matrix declaration implementation mismatch".into());
                        };
                        if fragment != *v {
                            return Err(
                                "matrix declaration binding differs from prepared storage".into()
                            );
                        }
                        self.frag_decl(*v, layout)?;
                        return Ok(());
                    }
                    ExprKind::Builtin {
                        name: Builtin::Reduce,
                        args,
                    } => return self.reduce_into(*v, args, operation),
                    ExprKind::Load { view, mode } => {
                        return self.load_into(*v, view, *mode, operation);
                    }
                    _ => {}
                }
                match &target.ty {
                    Ty::Scalar(d) => {
                        let val = self.target_expr(value)?.cast((*d).into());
                        let exists = self.real.contains_key(v);
                        let name = match self.real.get(v) {
                            Some(Realization::Scalar { name }) => name.clone(),
                            _ => format!("{}_{}", sanitize(&self.vars()[*v].name), v),
                        };
                        if !exists {
                            if let VarKind::Index(Atom::Param(atom)) = &self.vars()[*v].kind {
                                self.names.insert(atom.clone(), name.clone());
                            }
                            self.real
                                .insert(*v, Realization::Scalar { name: name.clone() });
                            self.target(crate::terminal::Statement::Let {
                                name,
                                ty: (*d).into(),
                                value: val,
                            });
                        } else {
                            let val = match op {
                                AssignOp::Assign => val,
                                other => Self::arithmetic(
                                    match other {
                                        AssignOp::Add => BinaryOp::Add,
                                        AssignOp::Sub => BinaryOp::Sub,
                                        AssignOp::Mul => BinaryOp::Mul,
                                        _ => unreachable!(),
                                    },
                                    crate::terminal::Expression::variable(
                                        name.clone(),
                                        (*d).into(),
                                    ),
                                    val,
                                    (*d).into(),
                                ),
                            };
                            self.target(crate::terminal::Statement::Assign { name, value: val });
                        }
                        Ok(())
                    }
                    Ty::Tile(shaped) => {
                        if matches!(shaped.elem, Elem::Repr(_)) {
                            if op != AssignOp::Assign {
                                return Err("packed arithmetic assignment is not defined".into());
                            }
                            let source = self.view_of(value)?;
                            let previous = self.real.remove(v);
                            self.snapshot_packets(
                                *v,
                                source,
                                operation,
                                if previous.is_some() {
                                    BarrierPurpose::Snapshot(Purpose::Value)
                                } else {
                                    BarrierPurpose::Copy
                                },
                            )?;
                            if let Some(destination) = previous {
                                let source =
                                    self.real.get(v).cloned().ok_or("packet copy missing")?;
                                self.copy_packets(
                                    &destination,
                                    &source,
                                    *v,
                                    operation,
                                    BarrierPurpose::Copy,
                                )?;
                                self.real.insert(*v, destination);
                            }
                            return Ok(());
                        }
                        let dtype = shaped
                            .elem
                            .read_dtype()
                            .ok_or("unresolved tile assignment type")?;
                        // Establish runtime view extents while the old source
                        // binding is still live, before allocating its snapshot.
                        let source = match value.kind {
                            ExprKind::Var(src) => self
                                .real
                                .get(&src)
                                .cloned()
                                .ok_or("unrealized source tile")?,
                            _ => self.view_of(value)?,
                        };
                        let previous = self.real.get(v).cloned();
                        let temporary =
                            self.declare_tile(*v, &shaped.shape, dtype, operation, Purpose::Value)?;
                        if let Some(destination) = &previous {
                            self.real.insert(*v, destination.clone());
                        }
                        self.copy_tile(
                            &temporary,
                            &source,
                            AssignOp::Assign,
                            BarrierSite {
                                operation,
                                variable: *v,
                                purpose: if previous.is_some() {
                                    BarrierPurpose::Snapshot(Purpose::Value)
                                } else {
                                    BarrierPurpose::Copy
                                },
                            },
                        )?;
                        if let Some(destination) = previous {
                            self.copy_tile(
                                &destination,
                                &temporary,
                                op,
                                BarrierSite {
                                    operation,
                                    variable: *v,
                                    purpose: BarrierPurpose::Copy,
                                },
                            )?;
                        } else if op != AssignOp::Assign {
                            return Err(
                                "compound tile assignment requires initialized storage".into()
                            );
                        }
                        Ok(())
                    }
                    Ty::Tensor(_) => {
                        if self.real.contains_key(v) {
                            return Err(
                                "tensor view rebinding needs explicit control-flow alias analysis"
                                    .into(),
                            );
                        }
                        let view = self
                            .view_of(value)
                            .map_err(|e| format!("view binding {}: {e}", self.vars()[*v].name))?;
                        self.real.insert(*v, view);
                        Ok(())
                    }
                    other => Err(format!("assignment to {other}")),
                }
            }
            ExprKind::Index { base, indices } => {
                let lhs = self.indexed_value(base, indices)?;
                let Ty::Scalar(dtype) = target.ty else {
                    unreachable!()
                };
                let value = self.target_expr(value)?.cast(dtype.into());
                let value = match op {
                    AssignOp::Assign => value,
                    op => Self::arithmetic(
                        match op {
                            AssignOp::Add => BinaryOp::Add,
                            AssignOp::Sub => BinaryOp::Sub,
                            AssignOp::Mul => BinaryOp::Mul,
                            _ => unreachable!(),
                        },
                        lhs.clone().cast(dtype.into()),
                        value,
                        dtype.into(),
                    ),
                };
                let TE::Read {
                    name,
                    index,
                    space,
                    ty,
                } = lhs
                else {
                    return Err("assignment needs owned tile storage".into());
                };
                self.target(TS::Write {
                    name,
                    index: *index,
                    space,
                    ty,
                    value: value.cast(ty),
                });
                Ok(())
            }
            _ => Err("unsupported assignment target".into()),
        }
    }

    fn copy_tile(
        &mut self,
        destination: &Realization,
        source: &Realization,
        op: AssignOp,
        site: BarrierSite,
    ) -> Result<(), String> {
        let (name, dims, dtype, distributed, shared) = match destination {
            Realization::Replicated { name, dims, dtype } => (name, dims, *dtype, false, false),
            Realization::Distributed {
                name, dims, dtype, ..
            } => (name, dims, *dtype, true, false),
            Realization::Shared { name, dims, dtype } => (name, dims, *dtype, true, true),
            _ => return Err("tile value needs owned destination storage".into()),
        };
        let count: i64 = dims.iter().map(|d| d.cap).product();
        let index = self.fresh("copy");
        let flat = self.fresh("element");
        let slots = if distributed {
            (count + SUBGROUP - 1) / SUBGROUP
        } else {
            count
        };
        self.target(TS::For {
            name: index.clone(),
            start: TE::integer(0),
            end: TE::integer(slots),
            step: 1,
        });
        self.indent += 1;
        self.target(TS::Let {
            name: flat.clone(),
            ty: TT::I32,
            value: if distributed {
                TE::binary(
                    BinaryOp::Add,
                    TE::variable("lane", TT::U32).cast(TT::I32),
                    TE::binary(
                        BinaryOp::Mul,
                        TE::integer(SUBGROUP),
                        TE::variable(&index, TT::I32),
                        TT::I32,
                    ),
                    TT::I32,
                )
            } else {
                TE::variable(&index, TT::I32)
            },
        });
        let strides = row_major_syms(&dims.iter().map(|d| d.cap).collect::<Vec<_>>());
        let (guard, _) = self.unflatten(&flat, dims, &strides, &Sym::constant(0));
        self.target(TS::If(guard));
        self.indent += 1;
        let value = match source {
            Realization::Replicated { name, dtype, .. } => TE::Read {
                name: name.clone(),
                index: Box::new(TE::variable(&flat, TT::I32)),
                space: TSpa::Private,
                ty: (*dtype).into(),
            },
            Realization::Shared { name, dtype, .. } => TE::Read {
                name: name.clone(),
                index: Box::new(TE::variable(&flat, TT::I32)),
                space: TSpa::Threadgroup,
                ty: (*dtype).into(),
            },
            Realization::Distributed { name, dtype, .. } => {
                let at = if distributed {
                    TE::variable(&index, TT::I32)
                } else {
                    TE::binary(
                        BinaryOp::Div,
                        TE::variable(&flat, TT::I32),
                        TE::integer(SUBGROUP),
                        TT::I32,
                    )
                };
                let read = TE::Read {
                    name: name.clone(),
                    index: Box::new(at),
                    space: TSpa::Private,
                    ty: (*dtype).into(),
                };
                if distributed {
                    read
                } else {
                    let transfer = match dtype {
                        DType::F16 | DType::BF16 => TT::F32,
                        DType::Bool => TT::U32,
                        _ => (*dtype).into(),
                    };
                    TE::Builtin(
                        "simd_shuffle".into(),
                        vec![
                            read.cast(transfer),
                            TE::binary(
                                BinaryOp::Rem,
                                TE::variable(&flat, TT::I32),
                                TE::integer(SUBGROUP),
                                TT::I32,
                            )
                            .cast(TT::U32),
                        ],
                        transfer,
                    )
                    .cast((*dtype).into())
                }
            }
            Realization::View {
                space,
                param,
                elem,
                offset,
                strides,
                ..
            } => {
                let (_, offset) = self.unflatten(&flat, dims, strides, offset);
                self.target_view_read(*space, param, &offset, elem)?
            }
            _ => return Err("unsupported source tile storage".into()),
        }
        .cast(dtype.into());
        let offset = TE::variable(
            if distributed && !shared {
                &index
            } else {
                &flat
            },
            TT::I32,
        );
        let space = if shared {
            TSpa::Threadgroup
        } else {
            TSpa::Private
        };
        let value = match op {
            AssignOp::Assign => value,
            op => Self::arithmetic(
                match op {
                    AssignOp::Add => BinaryOp::Add,
                    AssignOp::Sub => BinaryOp::Sub,
                    AssignOp::Mul => BinaryOp::Mul,
                    _ => unreachable!(),
                },
                TE::Read {
                    name: name.clone(),
                    index: Box::new(offset.clone()),
                    space,
                    ty: dtype.into(),
                },
                value,
                dtype.into(),
            ),
        };
        self.target(TS::Write {
            name: name.clone(),
            index: offset,
            space,
            ty: dtype.into(),
            value,
        });
        self.indent -= 1;
        self.target(TS::End);
        self.indent -= 1;
        self.target(TS::End);
        self.barrier(site)
    }

    fn tile_element(&mut self, tv: VarId, indices: &[Index]) -> Result<String, String> {
        Ok(self.target_tile_element(tv, indices)?.render())
    }
    fn target_tile_element(&mut self, tv: VarId, indices: &[Index]) -> Result<TE, String> {
        let real = self.real.get(&tv).cloned().ok_or("tile is not realized")?;
        let mut points = Vec::new();
        for index in indices {
            let Index::Point(p) = index else {
                return Err("slice of tile element".into());
            };
            points.push(self.int_value(p)?);
        }
        match real {
            r @ (Realization::Replicated { .. } | Realization::Shared { .. }) => {
                let shared = matches!(r, Realization::Shared { .. });
                let (name, dims, dtype) = match r {
                    Realization::Replicated { name, dims, dtype }
                    | Realization::Shared { name, dims, dtype } => (name, dims, dtype),
                    _ => unreachable!(),
                };
                let mut flat = TE::Integer(0, TT::I64);
                for (point, dim) in points.iter().zip(dims) {
                    let index = TE::Helper(
                        crate::support::Helper::Index,
                        vec![
                            self.target_sym(point)?.cast(TT::I64),
                            dim.value.cast(TT::I64),
                        ],
                        TT::I64,
                    );
                    flat = TE::binary(
                        BinaryOp::Add,
                        TE::binary(BinaryOp::Mul, flat, TE::Integer(dim.cap, TT::I64), TT::I64),
                        index,
                        TT::I64,
                    );
                }
                Ok(TE::Read {
                    name,
                    index: Box::new(flat),
                    space: if shared {
                        TSpa::Threadgroup
                    } else {
                        TSpa::Private
                    },
                    ty: dtype.into(),
                })
            }
            Realization::Distributed {
                name, dims, dtype, ..
            } => {
                for (ov, names, slot) in self.owned_ctx.iter().rev() {
                    let same_shape = match self.real.get(ov) {
                        Some(Realization::Distributed { dims: d, .. })
                        | Some(Realization::Shared { dims: d, .. }) => *d == dims && slot.is_some(),
                        _ => false,
                    };
                    let same_index = points.iter().zip(names).all(
                        |(p, n)| matches!(self.atom_of(p),Some(a) if self.names.get(&a)==Some(n)),
                    );
                    if same_shape && same_index {
                        return Ok(TE::Read {
                            name,
                            index: Box::new(TE::variable(slot.clone().unwrap(), TT::I32)),
                            space: TSpa::Private,
                            ty: dtype.into(),
                        });
                    }
                }
                Err(format!(
                    "distributed tile {} is read outside its owner",
                    self.vars()[tv].name
                ))
            }
            Realization::View { .. } | Realization::Param { .. } => {
                let (space, ptr, off, elem) = self.view_element(tv, &points)?;
                self.target_view_read(space, &ptr, &off, &elem)
            }
            _ => Err("cannot index selected realization".into()),
        }
    }
    fn target_raw_read(&self, ptr: &str, index: TE, dtype: DType) -> Result<TE, String> {
        if let Some((space, actual, count)) = self.local_planes.get(ptr) {
            if *actual != dtype {
                return Err("packet plane read dtype differs from allocation".into());
            }
            let index = TE::Helper(
                crate::support::Helper::Index,
                vec![index.cast(TT::I64), TE::Integer(*count as i64, TT::I64)],
                TT::I64,
            );
            return Ok(TE::Read {
                name: ptr.into(),
                index: Box::new(index),
                space: *space,
                ty: dtype.into(),
            });
        }

        let slot = self
            .buffers
            .iter()
            .find(|b| {
                if b.plane.is_empty() {
                    b.parameter == ptr
                } else {
                    format!("{}_{}", b.parameter, b.plane) == ptr
                }
            })
            .ok_or("missing device storage contract")?;
        Ok(TE::Helper(
            crate::support::Helper::Read,
            vec![
                TE::variable(ptr, TT::U64),
                index.cast(TT::I64),
                TE::Integer((slot.bytes / dtype.bytes() as usize) as i64, TT::U64),
            ],
            dtype.into(),
        ))
    }
    fn target_packed(
        &self,
        ptr: &str,
        entry: TE,
        bits: u32,
        interpretation: &repr::CodeInterpretation,
    ) -> Result<TE, String> {
        if bits == 0 || bits > 32 {
            return Err("packed field width must fit one code word".into());
        }
        let bit = TE::binary(
            BinaryOp::Mul,
            entry.cast(TT::U64),
            TE::Integer(i64::from(bits), TT::U64),
            TT::U64,
        );
        let word = TE::binary(
            BinaryOp::Div,
            bit.clone(),
            TE::Integer(32, TT::U64),
            TT::U64,
        );
        let shift = TE::binary(BinaryOp::Rem, bit, TE::Integer(32, TT::U64), TT::U64).cast(TT::U32);
        let low = TE::binary(
            BinaryOp::Shr,
            self.target_raw_read(ptr, word.clone(), DType::U32)?,
            shift.clone(),
            TT::U32,
        );
        let crossing = TE::binary(
            BinaryOp::Gt,
            TE::binary(
                BinaryOp::Add,
                shift.clone(),
                TE::Integer(i64::from(bits), TT::U32),
                TT::U32,
            ),
            TE::Integer(32, TT::U32),
            TT::Bool,
        );
        let high = TE::binary(
            BinaryOp::Shl,
            self.target_raw_read(
                ptr,
                TE::binary(BinaryOp::Add, word, TE::Integer(1, TT::U64), TT::U64),
                DType::U32,
            )?,
            TE::binary(BinaryOp::Sub, TE::Integer(32, TT::U32), shift, TT::U32),
            TT::U32,
        );
        let joined = TE::binary(
            BinaryOp::BitOr,
            low,
            TE::Select(
                Box::new(crossing),
                Box::new(high),
                Box::new(TE::Integer(0, TT::U32)),
            ),
            TT::U32,
        );
        let raw = TE::binary(
            BinaryOp::BitAnd,
            joined,
            TE::Integer(
                i64::from(if bits == 32 {
                    u32::MAX
                } else {
                    (1u32 << bits) - 1
                }),
                TT::U32,
            ),
            TT::U32,
        );
        Ok(match interpretation {
            repr::CodeInterpretation::Unsigned => raw,
            repr::CodeInterpretation::Offset(zero) => TE::binary(
                BinaryOp::Sub,
                raw.cast(TT::I32),
                TE::integer(i64::from(*zero)),
                TT::I32,
            ),
            repr::CodeInterpretation::TwosComplement => TE::binary(
                BinaryOp::Shr,
                TE::binary(
                    BinaryOp::Shl,
                    raw,
                    TE::Integer(i64::from(32 - bits), TT::U32),
                    TT::U32,
                )
                .cast(TT::I32),
                TE::integer(i64::from(32 - bits)),
                TT::I32,
            ),
            repr::CodeInterpretation::Table(table) => {
                let mut value = TE::integer(i64::from(*table.last().ok_or("empty code table")?));
                for (i, n) in table.iter().enumerate().rev().skip(1) {
                    value = TE::Select(
                        Box::new(TE::binary(
                            BinaryOp::Eq,
                            raw.clone(),
                            TE::Integer(i as i64, TT::U32),
                            TT::Bool,
                        )),
                        Box::new(TE::integer(i64::from(*n))),
                        Box::new(value),
                    );
                }
                value
            }
        })
    }
    fn target_coefficient(
        &self,
        ptr: &str,
        logical: TE,
        coefficient: &repr::Coefficient,
    ) -> Result<TE, String> {
        Ok(match coefficient {
            repr::Coefficient::Direct { plane } => self.target_raw_read(
                &format!("{ptr}_{}", plane.name),
                TE::binary(
                    BinaryOp::Div,
                    logical,
                    TE::Integer(i64::from(plane.group), TT::I64),
                    TT::I64,
                ),
                plane.dtype(),
            )?,
            repr::Coefficient::Product {
                factor,
                coefficients,
                field,
                sign,
            } => {
                let factor_value = self
                    .target_raw_read(
                        &format!("{ptr}_{}", factor.name),
                        TE::binary(
                            BinaryOp::Div,
                            logical.clone(),
                            TE::Integer(i64::from(factor.group), TT::I64),
                            TT::I64,
                        ),
                        factor.dtype(),
                    )?
                    .cast(TT::F32);
                let entry = TE::binary(
                    BinaryOp::Add,
                    TE::binary(
                        BinaryOp::Mul,
                        TE::binary(
                            BinaryOp::Div,
                            logical,
                            TE::Integer(i64::from(coefficients.group), TT::I64),
                            TT::I64,
                        ),
                        TE::Integer(i64::from(coefficients.fields), TT::I64),
                        TT::I64,
                    ),
                    TE::Integer(i64::from(*field), TT::I64),
                    TT::I64,
                );
                let repr::PlaneEncoding::Packed {
                    bits,
                    interpretation,
                } = &coefficients.encoding
                else {
                    return Err("hierarchical coefficient field is not packed".into());
                };
                let value = self
                    .target_packed(
                        &format!("{ptr}_{}", coefficients.name),
                        entry,
                        *bits,
                        interpretation,
                    )?
                    .cast(TT::F32);
                let product = TE::binary(BinaryOp::Mul, factor_value, value, TT::F32);
                if *sign == 1 {
                    product
                } else {
                    TE::binary(
                        BinaryOp::Mul,
                        product,
                        TE::Float(f64::from(*sign).to_bits(), TT::F32),
                        TT::F32,
                    )
                }
            }
        })
    }
    fn target_read_elem(&self, ptr: &str, off: &Sym, elem: &Elem) -> Result<TE, String> {
        let logical = self.target_sym(off)?.cast(TT::I64);
        match elem {
            Elem::Dtype(dtype) => self.target_raw_read(ptr, logical, *dtype),
            Elem::Repr(name) => {
                let r = repr::lookup(name).ok_or("unknown representation")?;
                let code = self
                    .target_packed(&format!("{ptr}_words"), logical.clone(), r.bits, &r.code)?
                    .cast(TT::F32);
                let scale = self
                    .target_coefficient(
                        ptr,
                        logical.clone(),
                        &r.coefficient(false).ok_or("missing scale")?,
                    )?
                    .cast(TT::F32);
                let bias = match r.coefficient(true) {
                    Some(c) => self.target_coefficient(ptr, logical, &c)?.cast(TT::F32),
                    None => TE::Float(0f64.to_bits(), TT::F32),
                };
                Ok(TE::Builtin("fma".into(), vec![code, scale, bias], TT::F32))
            }
            Elem::Param(_) => Err("unresolved packed element".into()),
        }
    }

    fn target_view_read(
        &self,
        space: TSpa,
        ptr: &str,
        off: &Sym,
        elem: &Elem,
    ) -> Result<TE, String> {
        if space == TSpa::Device || matches!(elem, Elem::Repr(_)) {
            return self.target_read_elem(ptr, off, elem);
        }
        let Elem::Dtype(dtype) = elem else {
            return Err("packed local view has no selected representation".into());
        };
        Ok(TE::Read {
            name: ptr.into(),
            index: Box::new(self.target_sym(off)?.cast(TT::I64)),
            space,
            ty: (*dtype).into(),
        })
    }
    fn atom_of(&self, s: &Sym) -> Option<String> {
        if let [Atom::Param(p)] = s.atoms().as_slice() {
            if *s == Sym::param(p) {
                return Some(p.clone());
            }
        }
        None
    }

    fn view_element(
        &mut self,
        tv: VarId,
        points: &[Sym],
    ) -> Result<(TSpa, String, Sym, Elem), String> {
        let r = self.real.get(&tv).cloned().unwrap();
        let (space, param, elem, offset, strides, shape) = match r {
            Realization::View {
                space,
                param,
                elem,
                offset,
                strides,
                shape,
            } => (space, param, elem, offset, strides, shape),
            Realization::Param { name, shape, elem } => (
                TSpa::Device,
                name,
                elem,
                Sym::constant(0),
                row_major_syms(&shape),
                shape.iter().map(|n| Sym::constant(*n)).collect(),
            ),
            _ => unreachable!(),
        };
        let mut off = offset;
        for ((p, s), extent) in points.iter().zip(&strides).zip(&shape) {
            let p = self.checked_index(p, extent)?;
            off = off.add(&p.mul(s));
        }
        Ok((space, param, off, elem))
    }

    fn checked_index(&mut self, index: &Sym, extent: &Sym) -> Result<Sym, String> {
        let name = self.fresh("checked_index");
        let value = TE::Helper(
            crate::support::Helper::Index,
            vec![
                self.target_sym(index)?.cast(TT::I64),
                self.target_sym(extent)?.cast(TT::I64),
            ],
            TT::I64,
        );
        // Index validity is an effect of constructing the view, including when
        // only its shape is consumed or a later data traversal is empty.
        self.target(TS::Let {
            name: name.clone(),
            ty: TT::I64,
            value,
        });
        self.names.insert(name.clone(), name.clone());
        self.expressions
            .insert(name.clone(), TE::variable(&name, TT::I64));
        Ok(Sym::param(&name))
    }

    fn load_into(
        &mut self,
        v: VarId,
        view: &Expr,
        mode: LoadMode,
        operation: OperationId,
    ) -> Result<(), String> {
        let realized = self
            .view_of(view)
            .map_err(|e| format!("load into {}: {e}", self.vars()[v].name))?;
        if mode == LoadMode::Borrow {
            if self.real.contains_key(&v) {
                return Err("borrowed load must define fresh tile storage".into());
            }
            self.real.insert(v, realized);
            return Ok(());
        }
        let previous = self.real.remove(&v);
        self.snapshot_into(v, realized, operation, Purpose::Value)?;
        if let Some(destination) = previous {
            let source = self
                .real
                .get(&v)
                .cloned()
                .ok_or("load snapshot is missing")?;
            if matches!(
                source,
                Realization::View {
                    elem: Elem::Repr(_),
                    ..
                }
            ) {
                self.copy_packets(&destination, &source, v, operation, BarrierPurpose::Copy)?;
            } else {
                self.copy_tile(
                    &destination,
                    &source,
                    AssignOp::Assign,
                    BarrierSite {
                        operation,
                        variable: v,
                        purpose: BarrierPurpose::Copy,
                    },
                )?;
            }
            self.real.insert(v, destination);
        }
        Ok(())
    }

    fn bind_stream_load(
        &mut self,
        v: VarId,
        realized: Realization,
        mode: LoadMode,
        operation: OperationId,
    ) -> Result<(), String> {
        if mode == LoadMode::Materialize {
            self.snapshot_into(v, realized, operation, Purpose::Value)
        } else {
            self.real.insert(v, realized);
            Ok(())
        }
    }

    fn snapshot_packets(
        &mut self,
        v: VarId,
        source: Realization,
        operation: OperationId,
        publication: BarrierPurpose,
    ) -> Result<(), String> {
        let layout = self
            .execution
            .storage
            .packets(v)
            .ok_or("packed value has no selected plane storage")?
            .clone();
        let base = self.fresh(&format!("{}_packets", self.vars()[v].name));
        let mut space = None;
        for (plane, part) in layout.planes.iter().enumerate() {
            let id = AllocationId {
                operation,
                variable: v,
                purpose: Purpose::PacketPlane(plane),
            };
            let declaration = self.allocation(id)?;
            if declaration.capacity != part.elements || declaration.dtype != part.plane.dtype() {
                return Err("packet allocation differs from representation geometry".into());
            }
            let actual = match declaration.placement {
                TilePlacement::Replicated => TSpa::Private,
                TilePlacement::GroupShared => TSpa::Threadgroup,
                _ => return Err("encoded planes require addressable placement".into()),
            };
            if space.is_some_and(|s| s != actual) {
                return Err("packet planes have inconsistent address spaces".into());
            }
            space = Some(actual);
            let name = format!("{base}_{}", part.plane.name);
            self.bind_allocation(id, &name)?;
            self.local_planes
                .insert(name, (actual, part.plane.dtype(), part.elements));
        }
        let prefix = self.fresh("packet_prefix");
        self.names.insert(prefix.clone(), prefix.clone());
        self.target(TS::Let {
            name: prefix.clone(),
            ty: TT::I32,
            value: TE::integer(0),
        });
        let Realization::View { elem, shape, .. } = &source else {
            return Err("encoded snapshot requires a shaped packet view".into());
        };
        let destination = Realization::View {
            space: space.ok_or("representation has no physical planes")?,
            param: base,
            elem: elem.clone(),
            offset: Sym::param(&prefix),
            strides: layout
                .strides
                .iter()
                .map(|&n| {
                    i64::try_from(n)
                        .map(Sym::constant)
                        .map_err(|_| "packet stride exceeds the target index domain")
                })
                .collect::<Result<_, _>>()?,
            shape: shape.clone(),
        };
        self.copy_packets(&destination, &source, v, operation, publication)?;
        self.real.insert(v, destination);
        Ok(())
    }
    fn copy_packets(
        &mut self,
        destination: &Realization,
        source: &Realization,
        v: VarId,
        operation: OperationId,
        publication: BarrierPurpose,
    ) -> Result<(), String> {
        let Realization::View {
            space,
            param,
            offset: destination_offset,
            ..
        } = destination
        else {
            return Err("encoded destination is not a packet view".into());
        };
        let Realization::View {
            param: input,
            elem: Elem::Repr(representation),
            offset,
            strides,
            shape,
            ..
        } = source
        else {
            return Err("encoded source is not a packet view".into());
        };
        let representation = repr::lookup(representation).ok_or("unknown packet representation")?;
        let group = i64::from(representation.storage_group());
        if strides.len() != shape.len() || shape.is_empty() {
            return Err("encoded snapshot has inconsistent row metadata".into());
        }
        if strides.last().and_then(Sym::as_constant) != Some(1)
            || strides[..strides.len() - 1]
                .iter()
                .any(|s| s.as_constant().is_none_or(|n| n < 0 || n % group != 0))
        {
            return Err("encoded snapshot requires contiguous packet rows".into());
        }
        let layout = self
            .execution
            .storage
            .packets(v)
            .ok_or("encoded copy has no selected layout")?
            .clone();
        let prefix = self.fresh("source_prefix");
        self.names.insert(prefix.clone(), prefix.clone());
        self.target(TS::Let {
            name: prefix.clone(),
            ty: TT::I32,
            value: self.target_sym(&offset.rem(&Sym::constant(group)))?,
        });
        let Some(destination_prefix) = self.atom_of(destination_offset) else {
            return Err("encoded destination prefix is not mutable storage".into());
        };
        let destination_prefix = self
            .names
            .get(&destination_prefix)
            .ok_or("encoded destination prefix is unbound")?
            .clone();
        let source_origin = offset.sub(&Sym::param(&prefix));
        let Ty::Tile(tile) = &self.vars()[v].ty else {
            return Err("encoded copy target is not tile".into());
        };
        if tile.shape.len() != shape.len() || tile.elem != Elem::Repr(representation.name.into()) {
            return Err("encoded snapshot differs from the target type".into());
        }
        let dims = tile
            .shape
            .iter()
            .zip(shape)
            .map(|(bound, extent)| {
                Ok(Dim {
                    cap: self.cap(bound)?,
                    ext: self.sym(extent)?,
                    value: self.target_sym(extent)?,
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        let (width, outer) = dims.split_last().ok_or("packed row rank is zero")?;
        let rows = outer
            .iter()
            .try_fold(1i64, |n, d| n.checked_mul(d.cap))
            .ok_or("packed row capacity overflow")?;
        if rows > i64::from(i32::MAX) - SUBGROUP {
            return Err("packet rows exceed the target loop index domain".into());
        }
        for part in &layout.planes {
            if part.elements > i32::MAX as u64 - SUBGROUP as u64 {
                return Err("packet storage exceeds the target loop index domain".into());
            }
            let row = self.fresh("packet_row");
            self.names.insert(row.clone(), row.clone());
            self.target(TS::For {
                name: row.clone(),
                start: TE::integer(0),
                end: TE::integer(rows),
                step: 1,
            });
            self.indent += 1;
            let (guard, origin) =
                self.unflatten(&row, outer, &strides[..strides.len() - 1], &source_origin);
            self.target(TS::If(guard));
            self.indent += 1;
            let integer = |n| TE::Integer(n, TT::I64);
            let extent = TE::binary(
                BinaryOp::Add,
                TE::variable(&prefix, TT::I32).cast(TT::I64),
                width.value.clone().cast(TT::I64),
                TT::I64,
            );
            let groups = TE::binary(
                BinaryOp::Div,
                TE::binary(
                    BinaryOp::Add,
                    extent,
                    integer(i64::from(part.plane.group) - 1),
                    TT::I64,
                ),
                integer(i64::from(part.plane.group)),
                TT::I64,
            );
            let bits = TE::binary(
                BinaryOp::Mul,
                groups,
                integer(i64::from(part.plane.fields) * i64::from(part.plane.entry_bits())),
                TT::I64,
            );
            let storage_bits = i64::from(part.plane.dtype().bytes()) * 8;
            let count = TE::Select(
                Box::new(TE::binary(
                    BinaryOp::Eq,
                    width.value.clone(),
                    TE::integer(0),
                    TT::Bool,
                )),
                Box::new(integer(0)),
                Box::new(TE::binary(
                    BinaryOp::Div,
                    TE::binary(BinaryOp::Add, bits, integer(storage_bits - 1), TT::I64),
                    integer(storage_bits),
                    TT::I64,
                )),
            );
            let index = self.fresh("packet_copy");
            let shared = *space == TSpa::Threadgroup;
            self.target(TS::For {
                name: index.clone(),
                start: if shared {
                    TE::variable("lane", TT::U32).cast(TT::I32)
                } else {
                    TE::integer(0)
                },
                end: count,
                step: if shared { SUBGROUP } else { 1 },
            });
            self.indent += 1;
            let origin = origin
                .quot(&Sym::constant(i64::from(part.plane.group)))
                .scale(i64::from(part.plane.fields) * i64::from(part.plane.entry_bits()))
                .quot(&Sym::constant(storage_bits));
            let source_index = TE::binary(
                BinaryOp::Add,
                self.target_sym(&origin)?.cast(TT::I64),
                TE::variable(&index, TT::I32).cast(TT::I64),
                TT::I64,
            );
            let value = self.target_raw_read(
                &format!("{input}_{}", part.plane.name),
                source_index,
                part.plane.dtype(),
            )?;
            let destination_index = TE::binary(
                BinaryOp::Add,
                TE::binary(
                    BinaryOp::Mul,
                    TE::variable(&row, TT::I32),
                    TE::integer(part.elements_per_row as i64),
                    TT::I32,
                ),
                TE::variable(index, TT::I32),
                TT::I32,
            );
            self.target(TS::Write {
                name: format!("{param}_{}", part.plane.name),
                index: destination_index,
                space: *space,
                ty: part.plane.dtype().into(),
                value,
            });
            self.indent -= 1;
            self.target(TS::End);
            self.indent -= 1;
            self.target(TS::End);
            self.indent -= 1;
            self.target(TS::End);
        }
        self.target(TS::Assign {
            name: destination_prefix,
            value: TE::variable(prefix, TT::I32),
        });
        self.barrier(BarrierSite {
            operation,
            variable: v,
            purpose: publication,
        })
    }

    fn snapshot_into(
        &mut self,
        v: VarId,
        realized: Realization,
        operation: OperationId,
        purpose: Purpose,
    ) -> Result<(), String> {
        let Realization::View {
            space,
            param,
            elem,
            offset,
            strides,
            shape,
        } = realized
        else {
            return Err("snapshot load requires a tensor view".into());
        };
        if purpose == Purpose::Value && matches!(elem, Elem::Repr(_)) {
            return self.snapshot_packets(
                v,
                Realization::View {
                    space,
                    param,
                    elem,
                    offset,
                    strides,
                    shape,
                },
                operation,
                BarrierPurpose::Snapshot(purpose),
            );
        }
        let dtype = match &elem {
            Elem::Dtype(d) => *d,
            _ => DType::F32,
        };

        // View extents describe this invocation; the selected tile type carries
        // allocation capacities. Split slices can have a dynamic extent even
        // when the unsplit value has a static capacity.
        let Ty::Tile(tile) = &self.vars()[v].ty else {
            return Err("snapshot destination is not a tile".into());
        };
        if tile.shape.len() != shape.len() {
            return Err("snapshot rank disagrees with selected tile".into());
        }
        let dims = tile
            .shape
            .iter()
            .zip(&shape)
            .map(|(bound, extent)| {
                let cap = self.cap(bound)?;
                if extent.as_constant().is_some_and(|n| n < 0 || n > cap) {
                    return Err("snapshot extent exceeds its selected capacity".to_string());
                }
                Ok(Dim {
                    cap,
                    ext: self.sym(extent)?,
                    value: self.target_sym(extent)?,
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        let r = self.declare_tile_dims(v, dims, dtype, operation, purpose)?;
        match r {
            r @ (Realization::Replicated { .. } | Realization::Shared { .. }) => {
                let shared = matches!(r, Realization::Shared { .. });
                let (name, dims) = match r {
                    Realization::Replicated { name, dims, .. }
                    | Realization::Shared { name, dims, .. } => (name, dims),
                    _ => unreachable!(),
                };
                let n_cap: i64 = dims.iter().map(|d| d.cap).product();
                let c = self.fresh("c");
                let first = if shared {
                    TE::variable("lane", TT::U32).cast(TT::I32)
                } else {
                    TE::integer(0)
                };
                let step = if shared { SUBGROUP } else { 1 };
                self.target(TS::For {
                    name: c.clone(),
                    start: first,
                    end: TE::integer(n_cap),
                    step,
                });
                self.indent += 1;
                let (guard, off) = self.unflatten(&c, &dims, &strides, &offset);
                self.target(TS::If(guard));
                self.indent += 1;
                let value = self
                    .target_view_read(space, &param, &off, &elem)?
                    .cast(dtype.into());
                self.target(TS::Write {
                    name,
                    index: TE::variable(c, TT::I32),
                    space: if shared {
                        TSpa::Threadgroup
                    } else {
                        TSpa::Private
                    },
                    ty: dtype.into(),
                    value,
                });
                self.indent -= 1;
                self.target(crate::terminal::Statement::End);
                self.indent -= 1;
                self.target(crate::terminal::Statement::End);
            }
            Realization::Distributed {
                name, dims, slots, ..
            } => {
                let j = self.fresh("slot");
                let e = self.fresh("e");
                self.target(TS::For {
                    name: j.clone(),
                    start: TE::integer(0),
                    end: TE::integer(slots),
                    step: 1,
                });
                self.indent += 1;
                self.target(TS::Let {
                    name: e.clone(),
                    ty: TT::I32,
                    value: TE::binary(
                        BinaryOp::Add,
                        TE::variable("lane", TT::U32).cast(TT::I32),
                        TE::binary(
                            BinaryOp::Mul,
                            TE::integer(SUBGROUP),
                            TE::variable(j.clone(), TT::I32),
                            TT::I32,
                        ),
                        TT::I32,
                    ),
                });
                let (guard, off) = self.unflatten(&e, &dims, &strides, &offset);
                self.target(TS::If(guard));
                self.indent += 1;
                let value = self
                    .target_view_read(space, &param, &off, &elem)?
                    .cast(dtype.into());
                self.target(TS::Write {
                    name,
                    index: TE::variable(j, TT::I32),
                    space: TSpa::Private,
                    ty: dtype.into(),
                    value,
                });
                self.indent -= 1;
                self.target(crate::terminal::Statement::End);
                self.indent -= 1;
                self.target(crate::terminal::Statement::End);
            }
            _ => unreachable!(),
        }
        self.barrier(BarrierSite {
            operation,
            variable: v,
            purpose: BarrierPurpose::Snapshot(purpose),
        })
    }

    /// Decompose a capacity-flat element counter into indices; returns the validity guard and
    /// the element offset through the view's strides.
    fn unflatten(&mut self, flat: &str, dims: &[Dim], strides: &[Sym], offset: &Sym) -> (TE, Sym) {
        let n_cap: i64 = dims.iter().map(|d| d.cap).product();
        if n_cap == 0 {
            return (TE::Integer(0, TT::Bool), offset.clone());
        }
        let mut stride = n_cap;
        let mut guard = TE::binary(
            BinaryOp::Lt,
            TE::variable(flat, TT::I32),
            TE::integer(n_cap),
            TT::Bool,
        );
        let mut off = offset.clone();
        for (k, d) in dims.iter().enumerate() {
            stride /= d.cap;
            let idx = self.fresh("ix");
            self.target(TS::Let {
                name: idx.clone(),
                ty: TT::I32,
                value: TE::binary(
                    BinaryOp::Rem,
                    TE::binary(
                        BinaryOp::Div,
                        TE::variable(flat, TT::I32),
                        TE::integer(stride),
                        TT::I32,
                    ),
                    TE::integer(d.cap),
                    TT::I32,
                ),
            });
            if !d.is_static() {
                guard = TE::binary(
                    BinaryOp::And,
                    guard,
                    TE::binary(
                        BinaryOp::Lt,
                        TE::variable(idx.clone(), TT::I32),
                        d.value.clone(),
                        TT::Bool,
                    ),
                    TT::Bool,
                );
            }
            off = off.add(&Sym::param(&idx).mul(&strides[k]));
        }
        (guard, off)
    }

    fn store(&mut self, tile: &Expr, view: &Expr) -> Result<(), String> {
        let real = match tile.kind {
            ExprKind::Var(v) => self.real.get(&v).cloned(),
            _ => None,
        };
        let direct = !matches!(
            real,
            Some(
                Realization::Replicated { .. }
                    | Realization::Shared { .. }
                    | Realization::Distributed { .. }
            )
        );
        let Realization::View {
            space,
            param,
            elem,
            offset,
            strides,
            shape,
        } = self.view_of(view)?
        else {
            unreachable!()
        };
        if space != TSpa::Device {
            return Err("store destination is not tensor memory".into());
        }
        let Elem::Dtype(dtype) = elem else {
            return Err("store into packed tensor".into());
        };
        let count = self
            .buffers
            .iter()
            .find(|b| b.parameter == param && b.plane.is_empty())
            .ok_or("missing store storage")?
            .bytes
            / dtype.bytes() as usize;
        let (name, dims, from, space, slots) = match real {
            Some(Realization::Replicated { name, dims, dtype }) => {
                (name, dims, dtype, TSpa::Private, None)
            }
            Some(Realization::Shared { name, dims, dtype }) => {
                (name, dims, dtype, TSpa::Threadgroup, None)
            }
            Some(Realization::Distributed {
                name,
                dims,
                dtype,
                slots,
            }) => (name, dims, dtype, TSpa::Private, Some(slots)),
            _ => {
                let shape = tile.ty.shaped().ok_or("store source is not shaped")?;
                (
                    String::new(),
                    self.dims(&shape.shape)?,
                    shape
                        .elem
                        .read_dtype()
                        .ok_or("store source dtype is unknown")?,
                    TSpa::Private,
                    None,
                )
            }
        };
        if dims.len() != shape.len() {
            return Err("store rank mismatch".into());
        }
        let mut valid = TE::Integer(1, TT::Bool);
        for (d, e) in dims.iter().zip(&shape) {
            valid = TE::binary(
                BinaryOp::And,
                valid,
                TE::binary(
                    BinaryOp::Eq,
                    d.value.clone().cast(TT::I64),
                    self.target_sym(e)?.cast(TT::I64),
                    TT::Bool,
                ),
                TT::Bool,
            );
        }
        self.target(TS::Evaluate(TE::Helper(
            crate::support::Helper::Validate,
            vec![valid.clone()],
            TT::Bool,
        )));
        let index = self.fresh("store_slot");
        let flat;
        if let Some(slots) = slots {
            self.target(TS::For {
                name: index.clone(),
                start: TE::integer(0),
                end: TE::integer(slots),
                step: 1,
            });
            self.indent += 1;
            flat = self.fresh("store_element");
            self.target(TS::Let {
                name: flat.clone(),
                ty: TT::I32,
                value: TE::binary(
                    BinaryOp::Add,
                    TE::variable("lane", TT::U32).cast(TT::I32),
                    TE::binary(
                        BinaryOp::Mul,
                        TE::integer(SUBGROUP),
                        TE::variable(index.clone(), TT::I32),
                        TT::I32,
                    ),
                    TT::I32,
                ),
            });
        } else {
            self.target(TS::For {
                name: index.clone(),
                start: TE::variable("lane", TT::U32).cast(TT::I32),
                end: TE::integer(dims.iter().map(|d| d.cap).product()),
                step: SUBGROUP,
            });
            self.indent += 1;
            flat = index.clone();
        }
        let (guard, off) = self.unflatten(&flat, &dims, &strides, &offset);
        self.target(TS::If(TE::binary(BinaryOp::And, guard, valid, TT::Bool)));
        self.indent += 1;
        let value = if direct {
            let mut stride = dims.iter().map(|d| d.cap).product::<i64>();
            let mut indices = Vec::new();
            for dim in &dims {
                let coordinate = self.fresh("store_coordinate");
                let value = if dim.cap == 0 {
                    TE::integer(0)
                } else {
                    stride /= dim.cap;
                    TE::binary(
                        BinaryOp::Rem,
                        TE::binary(
                            BinaryOp::Div,
                            TE::variable(&flat, TT::I32),
                            TE::integer(stride.max(1)),
                            TT::I32,
                        ),
                        TE::integer(dim.cap),
                        TT::I32,
                    )
                };
                self.target(TS::Let {
                    name: coordinate.clone(),
                    ty: TT::I32,
                    value,
                });
                self.names.insert(coordinate.clone(), coordinate.clone());
                indices.push(Index::Point(Expr {
                    kind: ExprKind::ShapeParam(coordinate.clone()),
                    ty: Ty::Scalar(DType::I32),
                    sym: Some(Sym::param(&coordinate)),
                    span: tile.span,
                }));
            }
            if let ExprKind::Accessor { base, name } = &tile.kind {
                self.target_accessor(base, name, &indices)?
            } else {
                self.indexed_value(tile, &indices)?
            }
        } else {
            TE::Read {
                name,
                index: Box::new(TE::variable(index, TT::I32)),
                space,
                ty: from.into(),
            }
        }
        .cast(dtype.into());
        self.target(TS::Evaluate(TE::Helper(
            crate::support::Helper::Write,
            vec![
                TE::variable(param, TT::U64),
                self.target_sym(&off)?.cast(TT::I64),
                TE::Integer(count as i64, TT::U64),
                value,
            ],
            TT::Bool,
        )));
        self.indent -= 1;
        self.target(TS::End);
        self.indent -= 1;
        self.target(TS::End);
        Ok(())
    }

    fn reduce_into(
        &mut self,
        v: VarId,
        args: &[Expr],
        operation: OperationId,
    ) -> Result<(), String> {
        let previous = self.real.remove(&v);
        self.reduce_new(v, args, operation)?;
        if let Some(destination) = previous {
            let source = self
                .real
                .get(&v)
                .cloned()
                .ok_or("missing reduction result")?;
            match (&destination, &source) {
                (Realization::Scalar { name: to }, Realization::Scalar { name: from }) => {
                    self.target(TS::Assign {
                        name: to.clone(),
                        value: TE::variable(
                            from.clone(),
                            scalar_dtype(&self.vars()[v].ty)
                                .ok_or("scalar reduction type is missing")?
                                .into(),
                        ),
                    });
                }
                _ => self.copy_tile(
                    &destination,
                    &source,
                    AssignOp::Assign,
                    BarrierSite {
                        operation,
                        variable: v,
                        purpose: BarrierPurpose::Copy,
                    },
                )?,
            }
            if let (VarKind::Index(Atom::Param(atom)), Realization::Scalar { name }) =
                (&self.vars()[v].kind, &destination)
            {
                self.names.insert(atom.clone(), name.clone());
            }
            self.real.insert(v, destination);
        }
        Ok(())
    }

    fn reduce_new(
        &mut self,
        v: VarId,
        args: &[Expr],
        operation: OperationId,
    ) -> Result<(), String> {
        let ExprKind::Var(tv) = args[0].kind else {
            return Err("reduce of a non-variable tile".into());
        };
        let ExprKind::Int(axis) = args[1].kind else {
            unreachable!()
        };
        let ExprKind::Int(op) = args[2].kind else {
            unreachable!()
        };
        let axis = axis as usize;
        if op == 3 {
            return self.argmax_into(v, tv, axis, operation);
        }
        let selected = self
            .execution
            .reductions
            .get(crate::reduction::Site {
                output: v,
                operation,
            })?
            .clone();
        let mut src = self
            .real
            .get(&tv)
            .cloned()
            .ok_or("reduce of an unrealized tile")?;
        if selected.decision.input != tv
            || selected.decision.materialize_input != matches!(src, Realization::View { .. })
        {
            return Err("reduction input disagrees with its selected ownership".into());
        }
        if selected.decision.materialize_input {
            // A borrowed stream remains a tile value. Materialize it when this
            // reduction realization needs owned lane storage.
            let encoded = matches!(
                src,
                Realization::View {
                    elem: Elem::Repr(_),
                    ..
                }
            )
            .then(|| src.clone());
            self.snapshot_into(tv, src, operation, Purpose::ReductionInput)?;
            src = self
                .real
                .get(&tv)
                .cloned()
                .ok_or("reduction snapshot is missing")?;
            if let Some(original) = encoded {
                self.real.insert(tv, original);
            }
        }
        let (dims, dtype) = match &src {
            Realization::Replicated { dims, dtype, .. }
            | Realization::Distributed { dims, dtype, .. }
            | Realization::Shared { dims, dtype, .. } => (dims.clone(), *dtype),
            other => return Err(format!("reduce of {other:?}")),
        };
        let contract = seismic_lang::reduction::Contract::new(
            ReduceOp::from_tag(op).ok_or("unknown reduction operation")?,
            dtype,
            matches!(args.get(3).map(|e| &e.kind), Some(ExprKind::Bool(true))),
        );
        if contract != selected.decision.contract {
            return Err("fold numerical contract disagrees with selected execution".into());
        }
        let init = if dtype.is_float() {
            TE::Float(contract.identity().value().to_bits(), dtype.into())
        } else {
            TE::Integer(contract.identity().value() as i64, dtype.into())
        };
        let combine = |acc: TE, x: TE| -> TE {
            use seismic_lang::reduction::Combination;
            let ty = TT::from(dtype);
            match contract.combination() {
                Combination::LogicalOr => TE::binary(BinaryOp::Or, acc, x, TT::Bool),
                Combination::LogicalAnd => TE::binary(BinaryOp::And, acc, x, TT::Bool),
                Combination::SaturatingAdd => {
                    let wide = if dtype == DType::I32 {
                        TT::I64
                    } else {
                        TT::U64
                    };
                    let sum = TE::binary(BinaryOp::Add, acc.cast(wide), x.cast(wide), wide);
                    let maximum = TE::Integer(
                        if dtype == DType::I32 {
                            i64::from(i32::MAX)
                        } else {
                            i64::from(u32::MAX)
                        },
                        wide,
                    );
                    let upper = TE::Select(
                        Box::new(TE::binary(
                            BinaryOp::Gt,
                            sum.clone(),
                            maximum.clone(),
                            TT::Bool,
                        )),
                        Box::new(maximum),
                        Box::new(sum),
                    );
                    if dtype == DType::I32 {
                        let minimum = TE::Integer(i64::from(i32::MIN), wide);
                        TE::Select(
                            Box::new(TE::binary(
                                BinaryOp::Lt,
                                upper.clone(),
                                minimum.clone(),
                                TT::Bool,
                            )),
                            Box::new(minimum),
                            Box::new(upper),
                        )
                        .cast(ty)
                    } else {
                        upper.cast(ty)
                    }
                }
                Combination::FloatingAdd => {
                    TE::binary(BinaryOp::Add, acc.cast(TT::F32), x.cast(TT::F32), TT::F32).cast(ty)
                }
                Combination::Maximum | Combination::Minimum => {
                    let arithmetic = if matches!(dtype, DType::F16 | DType::BF16) {
                        TT::F32
                    } else {
                        ty
                    };
                    TE::Builtin(
                        if contract.combination() == Combination::Maximum {
                            "max"
                        } else {
                            "min"
                        }
                        .into(),
                        vec![acc.cast(arithmetic), x.cast(arithmetic)],
                        arithmetic,
                    )
                    .cast(ty)
                }
                Combination::FirstMaximum => unreachable!(),
            }
        };
        // Narrow collective arithmetic is not admitted yet: ordered scalar
        // publication is a legal realization even when reassociation is allowed.
        let ordered = contract.ordered
            || matches!(dtype, DType::BF16 | DType::F16)
            || contract.combination() == seismic_lang::reduction::Combination::SaturatingAdd;
        let mut out_dims = dims.clone();
        out_dims.remove(axis);
        let placement = match src {
            Realization::Replicated { .. } => TilePlacement::Replicated,
            Realization::Shared { .. } => TilePlacement::GroupShared,
            Realization::Distributed { .. } => TilePlacement::Distributed,
            _ => unreachable!(),
        };
        if (selected.decision.contract.operation == ReduceOp::Argmax)
            || selected.decision.input_placement != Some(placement.clone())
        {
            return Err("fold input placement disagrees with its selected contract".into());
        }
        let reduction = crate::reduction::ReductionDomain::new(
            &dims.iter().map(|d| d.cap).collect::<Vec<_>>(),
            axis,
            dtype,
            ordered,
            placement.clone(),
            SUBGROUP as u64,
        )?;
        use crate::reduction::Algorithm;
        let out_cap = reduction.output_capacity() as i64;
        let scalar_result = reduction.scalar_output();
        let inner_cap = reduction.inner_capacity() as i64;
        let axis_cap = reduction.axis_capacity() as i64;
        let axis_ext = dims[axis].value.clone().cast(TT::I32);
        let algorithm = selected.algorithm;
        if reduction != selected.decision.domain {
            return Err("emitted reduction geometry disagrees with its selected domain".into());
        }
        if let Some(expected) = selected.output {
            let declaration = self.allocation(AllocationId {
                operation,
                variable: v,
                purpose: Purpose::Value,
            })?;
            if declaration != expected {
                return Err("reduction output disagrees with allocation plan".into());
            }
        }
        let output_slots = reduction.output_slots(algorithm)?;
        let name = self.fresh("reduced");
        if scalar_result {
            self.real
                .insert(v, Realization::Scalar { name: name.clone() });
            self.target(TS::Let {
                name: name.clone(),
                ty: dtype.into(),
                value: init.clone(),
            });
        } else {
            self.bind_allocation(
                AllocationId {
                    operation,
                    variable: v,
                    purpose: Purpose::Value,
                },
                &name,
            )?;
            self.real.insert(
                v,
                if algorithm == Algorithm::LaneLocal {
                    Realization::Distributed {
                        name: name.clone(),
                        dims: out_dims.clone(),
                        dtype,
                        slots: output_slots as i64,
                    }
                } else {
                    Realization::Replicated {
                        name: name.clone(),
                        dims: out_dims.clone(),
                        dtype,
                    }
                },
            );
        }
        if out_cap == 0 {
            return Ok(());
        }
        let read = |source: &Realization, index: TE| -> TE {
            match source {
                Realization::Replicated { name, dtype, .. }
                | Realization::Distributed { name, dtype, .. } => TE::Read {
                    name: name.clone(),
                    index: Box::new(index),
                    space: TSpa::Private,
                    ty: (*dtype).into(),
                },
                Realization::Shared { name, dtype, .. } => TE::Read {
                    name: name.clone(),
                    index: Box::new(index),
                    space: TSpa::Threadgroup,
                    ty: (*dtype).into(),
                },
                _ => unreachable!(),
            }
        };
        let element = |output: TE, k: TE| {
            TE::binary(
                BinaryOp::Add,
                TE::binary(
                    BinaryOp::Mul,
                    TE::binary(
                        BinaryOp::Add,
                        TE::binary(
                            BinaryOp::Mul,
                            TE::binary(
                                BinaryOp::Div,
                                output.clone(),
                                TE::integer(inner_cap),
                                TT::I32,
                            ),
                            TE::integer(axis_cap),
                            TT::I32,
                        ),
                        k,
                        TT::I32,
                    ),
                    TE::integer(inner_cap),
                    TT::I32,
                ),
                TE::binary(BinaryOp::Rem, output, TE::integer(inner_cap), TT::I32),
                TT::I32,
            )
        };
        let o = self.fresh("output");
        let slot = self.fresh("slot");
        let acc = self.fresh("acc");
        if algorithm == Algorithm::LaneLocal {
            self.target(TS::For {
                name: slot.clone(),
                start: TE::integer(0),
                end: TE::integer(output_slots as i64),
                step: 1,
            });
            self.indent += 1;
            self.target(TS::Let {
                name: o.clone(),
                ty: TT::I32,
                value: TE::binary(
                    BinaryOp::Add,
                    TE::variable("lane", TT::U32).cast(TT::I32),
                    TE::binary(
                        BinaryOp::Mul,
                        TE::integer(SUBGROUP),
                        TE::variable(&slot, TT::I32),
                        TT::I32,
                    ),
                    TT::I32,
                ),
            });
        } else {
            self.target(TS::For {
                name: o.clone(),
                start: TE::integer(0),
                end: TE::integer(out_cap),
                step: 1,
            });
            self.indent += 1;
        }
        self.target(TS::Let {
            name: acc.clone(),
            ty: dtype.into(),
            value: init,
        });
        let k = self.fresh("axis");
        let j = self.fresh("source_slot");
        if placement != TilePlacement::Distributed
            || algorithm == Algorithm::Ordered
            || algorithm == Algorithm::LaneLocal
        {
            self.target(TS::For {
                name: k.clone(),
                start: TE::integer(0),
                end: axis_ext.clone(),
                step: 1,
            });
            self.indent += 1;
            if algorithm == Algorithm::LaneLocal {
                self.target(TS::If(TE::binary(
                    BinaryOp::Lt,
                    TE::variable(&o, TT::I32),
                    TE::integer(out_cap),
                    TT::Bool,
                )));
                self.indent += 1;
            }
            let e = element(TE::variable(&o, TT::I32), TE::variable(&k, TT::I32));
            let x = if placement == TilePlacement::Distributed {
                let x = read(
                    &src,
                    TE::binary(BinaryOp::Div, e.clone(), TE::integer(SUBGROUP), TT::I32),
                );
                if algorithm == Algorithm::LaneLocal {
                    x
                } else {
                    let transfer = match dtype {
                        DType::F16 | DType::BF16 => TT::F32,
                        DType::Bool => TT::U32,
                        _ => dtype.into(),
                    };
                    TE::Builtin(
                        "simd_shuffle".into(),
                        vec![
                            x.cast(transfer),
                            TE::binary(BinaryOp::Rem, e, TE::integer(SUBGROUP), TT::I32)
                                .cast(TT::U32),
                        ],
                        transfer,
                    )
                    .cast(dtype.into())
                }
            } else {
                read(&src, e)
            };
            self.target(TS::Assign {
                name: acc.clone(),
                value: combine(TE::variable(&acc, dtype.into()), x),
            });
            if algorithm == Algorithm::LaneLocal {
                self.indent -= 1;
                self.target(TS::End);
            }
            self.indent -= 1;
            self.target(TS::End);
        } else {
            let Realization::Distributed { slots, .. } = &src else {
                unreachable!()
            };
            self.target(TS::For {
                name: j.clone(),
                start: TE::integer(0),
                end: TE::integer(*slots),
                step: 1,
            });
            self.indent += 1;
            let e = TE::binary(
                BinaryOp::Add,
                TE::variable("lane", TT::U32).cast(TT::I32),
                TE::binary(
                    BinaryOp::Mul,
                    TE::integer(SUBGROUP),
                    TE::variable(&j, TT::I32),
                    TT::I32,
                ),
                TT::I32,
            );
            let axis = TE::binary(
                BinaryOp::Rem,
                TE::binary(BinaryOp::Div, e.clone(), TE::integer(inner_cap), TT::I32),
                TE::integer(axis_cap),
                TT::I32,
            );
            let output = TE::binary(
                BinaryOp::Add,
                TE::binary(
                    BinaryOp::Mul,
                    TE::binary(
                        BinaryOp::Div,
                        TE::binary(BinaryOp::Div, e.clone(), TE::integer(inner_cap), TT::I32),
                        TE::integer(axis_cap),
                        TT::I32,
                    ),
                    TE::integer(inner_cap),
                    TT::I32,
                ),
                TE::binary(BinaryOp::Rem, e.clone(), TE::integer(inner_cap), TT::I32),
                TT::I32,
            );
            let guard = TE::binary(
                BinaryOp::And,
                TE::binary(
                    BinaryOp::And,
                    TE::binary(
                        BinaryOp::Lt,
                        e,
                        TE::integer(reduction.input_capacity() as i64),
                        TT::Bool,
                    ),
                    TE::binary(BinaryOp::Lt, axis, axis_ext, TT::Bool),
                    TT::Bool,
                ),
                TE::binary(BinaryOp::Eq, output, TE::variable(&o, TT::I32), TT::Bool),
                TT::Bool,
            );
            self.target(TS::If(guard));
            self.indent += 1;
            self.target(TS::Assign {
                name: acc.clone(),
                value: combine(
                    TE::variable(&acc, dtype.into()),
                    read(&src, TE::variable(&j, TT::I32)),
                ),
            });
            self.indent -= 1;
            self.target(TS::End);
            self.indent -= 1;
            self.target(TS::End);
            self.target(TS::Assign {
                name: acc.clone(),
                value: TE::Builtin(
                    match op {
                        0 => "simd_sum",
                        1 => "simd_max",
                        2 => "simd_min",
                        _ => unreachable!(),
                    }
                    .into(),
                    vec![TE::variable(&acc, dtype.into())],
                    dtype.into(),
                ),
            });
        }
        if scalar_result {
            self.target(TS::Assign {
                name,
                value: TE::variable(acc, dtype.into()),
            });
        } else {
            self.target(TS::Write {
                name,
                index: TE::variable(
                    if algorithm == Algorithm::LaneLocal {
                        slot
                    } else {
                        o
                    },
                    TT::I32,
                ),
                space: TSpa::Private,
                ty: dtype.into(),
                value: TE::variable(acc, dtype.into()),
            });
        }
        self.indent -= 1;
        self.target(TS::End);
        Ok(())
    }

    /// `argmax` along an axis: the index of the largest value, ties to the smaller index.
    fn argmax_into(
        &mut self,
        v: VarId,
        tv: VarId,
        axis: usize,
        operation: OperationId,
    ) -> Result<(), String> {
        let src = self
            .real
            .get(&tv)
            .cloned()
            .ok_or("reduce of an unrealized tile")?;
        let (dims, dtype) = match &src {
            Realization::Replicated { dims, dtype, .. }
            | Realization::Distributed { dims, dtype, .. }
            | Realization::Shared { dims, dtype, .. } => (dims.clone(), *dtype),
            Realization::View { shape, elem, .. } => {
                let dims = shape
                    .iter()
                    .map(|s| self.dim(s))
                    .collect::<Result<Vec<_>, _>>()?;
                (dims, elem.read_dtype().unwrap_or(DType::F32))
            }
            other => return Err(format!("argmax of {other:?}")),
        };
        let selected = self
            .execution
            .reductions
            .get(crate::reduction::Site {
                output: v,
                operation,
            })?
            .clone();
        let placement = match &src {
            Realization::Replicated { .. } => Some(TilePlacement::Replicated),
            Realization::Distributed { .. } => Some(TilePlacement::Distributed),
            Realization::Shared { .. } => Some(TilePlacement::GroupShared),
            Realization::View { .. } => None,
            _ => unreachable!(),
        };
        let domain = crate::reduction::ReductionDomain::argmax(
            &dims.iter().map(|d| d.cap).collect::<Vec<_>>(),
            axis,
            dtype,
            placement.clone(),
            selected.decision.full_lanes,
            SUBGROUP as u64,
        )?;
        if selected.decision.contract.operation != ReduceOp::Argmax
            || selected.decision.contract.input != dtype
            || selected.decision.input != tv
            || selected.decision.input_placement != placement
            || selected.decision.domain != domain
        {
            return Err("argmax disagrees with its selected input/domain".into());
        }
        let algorithm = selected.algorithm;
        use crate::reduction::Algorithm;
        let mut out_dims = dims.clone();
        out_dims.remove(axis);
        let out_cap = domain.output_capacity() as i64;
        domain.output_slots(algorithm)?;
        let scalar_result = domain.scalar_output();
        if let Some(expected) = selected.output {
            let declaration = self.allocation(AllocationId {
                operation,
                variable: v,
                purpose: Purpose::Value,
            })?;
            if declaration != expected {
                return Err("reduction output disagrees with allocation plan".into());
            }
        }
        let name = self.fresh("argmax");
        if let VarKind::Index(Atom::Param(atom)) = &self.vars()[v].kind {
            self.names.insert(atom.clone(), name.clone());
        }
        if scalar_result {
            self.real
                .insert(v, Realization::Scalar { name: name.clone() });
            self.line(&format!("int {name};"));
        } else {
            self.real.insert(
                v,
                Realization::Replicated {
                    name: name.clone(),
                    dims: out_dims.clone(),
                    dtype: DType::I32,
                },
            );
            self.bind_allocation(
                AllocationId {
                    operation,
                    variable: v,
                    purpose: Purpose::Value,
                },
                &name,
            )?;
        }
        if out_cap == 0 {
            return Ok(());
        }
        let inner_cap = domain.inner_capacity() as i64;
        let axis_cap = domain.axis_capacity() as i64;
        let axis_ext = dims[axis].ext.clone();
        let initial = reduction_identity(selected.decision.contract.identity());
        let o = self.fresh("o");
        let best = self.fresh("best");
        let at = self.fresh("at");
        self.line(&format!("for (int {o} = 0; {o} < {out_cap}; ++{o}) {{"));
        self.indent += 1;
        self.line(&format!(
            "{} {best} = {}({initial}); int {at} = 0x7fffffff;",
            ctype(dtype),
            ctype(dtype)
        ));
        let tie_valid = if matches!(dtype, DType::F32 | DType::F16 | DType::BF16) {
            format!("{at} != 0x7fffffff && ")
        } else {
            String::new()
        };
        match &src {
            Realization::Replicated { name: sn, .. } | Realization::Shared { name: sn, .. } => {
                let k = self.fresh("k");
                self.line(&format!("for (int {k} = 0; {k} < {axis_ext}; ++{k}) {{"));
                self.indent += 1;
                self.line(&format!("const int e = (({o} / {inner_cap}) * {axis_cap} + {k}) * {inner_cap} + ({o} % {inner_cap});"));
                self.line(&format!("if ({sn}[e] > {best} || ({sn}[e] == {best} && {tie_valid}{k} < {at})) {{ {best} = {sn}[e]; {at} = {k}; }}"));
                self.indent -= 1;
                self.target(crate::terminal::Statement::End);
            }
            Realization::Distributed { name: sn, .. } if algorithm == Algorithm::Ordered => {
                let k = self.fresh("k");
                self.line(&format!("for (int {k} = 0; {k} < {axis_ext}; ++{k}) {{"));
                self.indent += 1;
                self.line(&format!("const int e = (({o} / {inner_cap}) * {axis_cap} + {k}) * {inner_cap} + ({o} % {inner_cap});"));
                let read = if matches!(dtype, DType::F16 | DType::BF16) {
                    format!(
                        "{}(simd_shuffle(float({sn}[e / {SUBGROUP}]), uint(e % {SUBGROUP})))",
                        ctype(dtype)
                    )
                } else if dtype == DType::Bool {
                    format!("bool(simd_shuffle(uint({sn}[e / {SUBGROUP}]), uint(e % {SUBGROUP})))")
                } else {
                    format!("simd_shuffle({sn}[e / {SUBGROUP}], uint(e % {SUBGROUP}))")
                };
                let x = self.fresh("x");
                self.line(&format!("const {} {x} = {read};", ctype(dtype)));
                self.line(&format!("if ({x} > {best} || ({x} == {best} && {tie_valid}{k} < {at})) {{ {best} = {x}; {at} = {k}; }}"));
                self.indent -= 1;
                self.target(crate::terminal::Statement::End);
            }
            Realization::Distributed {
                name: sn, slots, ..
            } => {
                let n_cap: i64 = dims.iter().map(|d| d.cap).product();
                let j = self.fresh("slot");
                self.target(TS::For {
                    name: j.clone(),
                    start: TE::integer(0),
                    end: TE::integer(*slots),
                    step: 1,
                });
                self.indent += 1;
                self.line(&format!("const int e = int(lane) + {SUBGROUP} * {j};"));
                self.line(&format!("const int k = (e / {inner_cap}) % {axis_cap};"));
                self.line(&format!("if (e < {n_cap} && k < {axis_ext} && ((e / {inner_cap}) / {axis_cap}) * {inner_cap} + (e % {inner_cap}) == {o} && ({sn}[{j}] > {best} || ({sn}[{j}] == {best} && {tie_valid}k < {at}))) {{ {best} = {sn}[{j}]; {at} = k; }}"));
                self.indent -= 1;
                self.target(crate::terminal::Statement::End);
                self.line(&format!("{{ const {} m = simd_max({best}); {at} = simd_min({best} == m ? {at} : 0x7fffffff); }}", ctype(dtype)));
            }
            Realization::View { .. } => {
                // Lanes stride through the reduced axis of the device view, then combine.
                let k = self.fresh("k");
                self.names.insert(k.clone(), k.clone());
                let mut points = Vec::new();
                for (d, _) in dims.iter().enumerate() {
                    if d == axis {
                        points.push(Sym::param(&k));
                    } else {
                        let outer: i64 = dims[d + 1..]
                            .iter()
                            .filter(|_| true)
                            .enumerate()
                            .map(|(i, dd)| if d + 1 + i == axis { 1 } else { dd.cap })
                            .product();
                        let c = self.fresh("c");
                        self.names.insert(c.clone(), c.clone());
                        self.line(&format!(
                            "const int {c} = ({o} / {outer}) % {};",
                            dims[d].cap
                        ));
                        points.push(Sym::param(&c));
                    }
                }
                let (start, step) = if algorithm == Algorithm::Collective {
                    ("int(lane)", SUBGROUP)
                } else {
                    ("0", 1)
                };
                self.line(&format!(
                    "for (int {k} = {start}; {k} < {axis_ext}; {k} += {step}) {{"
                ));
                self.indent += 1;
                let (space, ptr, off, elem) = self.view_element(tv, &points)?;
                let val = self.target_view_read(space, &ptr, &off, &elem)?.render();
                let x = self.fresh("x");
                self.line(&format!("const {} {x} = {val};", ctype(dtype)));
                self.line(&format!("if ({x} > {best} || ({x} == {best} && {tie_valid}{k} < {at})) {{ {best} = {x}; {at} = {k}; }}"));
                self.indent -= 1;
                self.target(crate::terminal::Statement::End);
                if algorithm == Algorithm::Collective {
                    self.line(&format!("{{ const {} m = simd_max({best}); {at} = simd_min({best} == m ? {at} : 0x7fffffff); }}", ctype(dtype)));
                }
            }
            _ => unreachable!(),
        }
        // An all-NaN row has no improving candidate; reference semantics
        // select zero. Nonempty axes are established by the checked contract.
        self.line(&format!("if ({at} == 0x7fffffff) {at} = 0;"));
        if scalar_result {
            self.line(&format!("{name} = {at};"));
        } else {
            self.line(&format!("{name}[{o}] = {at};"));
        }
        self.indent -= 1;
        self.target(crate::terminal::Statement::End);
        Ok(())
    }

    /// Write this part's carried state to compiler-allocated scratch, one region per
    /// (item, part), using the memory plan's buffer identity and layout.
    fn publish_partials(
        &mut self,
        phase: usize,
        carried: &[VarId],
        row: &str,
    ) -> Result<(), String> {
        let scratch: Vec<_> = self
            .execution
            .memory
            .scratch()
            .iter()
            .filter(|s| s.phase == phase)
            .cloned()
            .collect();
        if scratch.len() != carried.len() {
            return Err("split scratch plan does not match carried values".into());
        }
        for (v, allocation) in carried.iter().zip(scratch) {
            let (name, cap) = match self.real.get(v) {
                Some(Realization::Replicated { name, dims, .. })
                | Some(Realization::Shared { name, dims, .. }) => (
                    name.clone(),
                    dims.iter().map(|d| d.cap).product::<i64>().max(1),
                ),
                Some(Realization::Distributed {
                    name, dims, slots, ..
                }) => {
                    let _ = slots;
                    (
                        name.clone(),
                        dims.iter().map(|d| d.cap).product::<i64>().max(1),
                    )
                }
                other => {
                    return Err(format!(
                        "carried tile is {other:?}; splitting cannot publish it"
                    ));
                }
            };
            if allocation.variable != *v
                || allocation.elements_per_item != cap as u64
                || allocation.dtype != DType::F32
            {
                return Err("published partial value disagrees with scratch plan".into());
            }
            let buf = format!("split_{}", allocation.index);
            let parts = allocation.parts;
            let i = self.fresh("i");
            self.line(&format!("if (lane == 0) for (int {i} = 0; {i} < {cap}; ++{i}) {buf}[(uint({row}) * {parts} + uint(part)) * {cap} + uint({i})] = {name}[{i}];"));
        }
        Ok(())
    }

    /// Fold the parts of a split reduction, reading each part's published state and
    /// applying the streaming body's own merge rule, then leave the result in the carried
    /// tiles so the kernel's tail runs unchanged.
    fn merge_partials(
        &mut self,
        phase: usize,
        carried: &[VarId],
        merges: &[execution::Merge],
        operation: OperationId,
    ) -> Result<(), String> {
        let scratch: Vec<_> = self
            .execution
            .memory
            .scratch()
            .iter()
            .filter(|s| s.phase == phase)
            .cloned()
            .collect();
        if carried.len() != scratch.len() || carried.len() != merges.len() {
            return Err("split handoff does not match its proven merge rules".into());
        }
        for ((v, allocation), merge) in carried.iter().zip(scratch).zip(merges) {
            if allocation.variable != *v {
                return Err("merge value disagrees with scratch plan".into());
            }
            let buf = format!("split_{}", allocation.index);
            let parts = allocation.parts;
            let cap = allocation.elements_per_item;
            let Ty::Tile(shaped) = self.vars()[*v].ty.clone() else {
                return Err("carried state is not a tile".into());
            };
            let r = self.declare_tile(*v, &shaped.shape, DType::F32, operation, Purpose::Merge)?;
            let (name, shared) = match r {
                Realization::Replicated { name, .. } | Realization::Distributed { name, .. } => {
                    (name, false)
                }
                Realization::Shared { name, .. } => (name, true),
                _ => return Err("merge needs materialized state".into()),
            };
            if cap != 1 {
                return Err("scalar merge state has non-scalar capacity".into());
            }
            if shared {
                self.line("if (lane == 0) {");
                self.indent += 1;
            }
            self.line(&format!("{name}[0] = {buf}[item * {parts}];"));
            let p = self.fresh("part");
            match merge {
                execution::Merge::Sum => self.line(&format!("for (int {p} = 1; {p} < {parts}; ++{p}) {name}[0] += {buf}[item * {parts} + uint({p})];")),
            }
            if shared {
                self.indent -= 1;
                self.target(crate::terminal::Statement::End);
            }
            self.barrier(BarrierSite {
                operation,
                variable: *v,
                purpose: BarrierPurpose::Merge,
            })?;
        }
        Ok(())
    }

    fn intrinsic_stmt(
        &mut self,
        name: &seismic_lang::intrinsics::Operation,
        args: &[Expr],
        operation: OperationId,
    ) -> Result<(), String> {
        let implementation = self.collective(*name)?;
        match name {
            seismic_lang::intrinsics::Operation::MatrixLoad
            | seismic_lang::intrinsics::Operation::MatrixLoadTranspose
            | seismic_lang::intrinsics::Operation::MatrixStore => {
                let ExprKind::Var(fv) = args[0].kind else {
                    return Err("fragment must be a variable".into());
                };
                let Some(Realization::Frag { name: frag }) = self.real.get(&fv).cloned() else {
                    return Err("unrealized fragment".into());
                };
                let (planned_fragment, layout, memory, want_t) = match &implementation {
                    CollectiveImplementation::Load {
                        fragment,
                        memory,
                        transpose,
                        layout,
                    } => (*fragment, *layout, memory, *transpose),
                    CollectiveImplementation::Store {
                        fragment,
                        memory,
                        layout,
                    } => (*fragment, *layout, memory, false),
                    _ => return Err("matrix transfer implementation mismatch".into()),
                };
                if planned_fragment != fv
                    || memory.operand != args[1]
                    || memory.row != args[2]
                    || memory.column != args[3]
                {
                    return Err("matrix transfer operands differ from prepared execution".into());
                }
                let row = self.int_value(&memory.row)?;
                let col = self.int_value(&memory.column)?;
                let operand = match args[1].kind {
                    ExprKind::Var(tv) => self
                        .real
                        .get(&tv)
                        .cloned()
                        .ok_or("unrealized intrinsic operand")?,
                    _ => self
                        .view_of(&args[1])
                        .map_err(|e| format!("matrix operand: {e}"))?,
                };
                // (pointer, offset of the block origin, leading dimension, memory holds the transpose)
                let (ptr, off, ld, col_major, space) = match operand {
                    Realization::Shared { name, dims, dtype } => {
                        if memory.space != StorageSpace::Threadgroup || memory.dtype != dtype {
                            return Err(
                                "matrix shared operand disagrees with selected memory contract"
                                    .into(),
                            );
                        }
                        let ld = Sym::constant(dims[1].cap);
                        (name, row.mul(&ld).add(&col), ld, false, TSpa::Threadgroup)
                    }
                    Realization::View {
                        space,
                        param,
                        elem,
                        offset,
                        strides,
                        ..
                    } => {
                        let expected = match space {
                            TSpa::Device => StorageSpace::Device,
                            TSpa::Threadgroup => StorageSpace::Threadgroup,
                            _ => return Err("matrix view needs device or shared storage".into()),
                        };
                        if memory.space != expected || elem != Elem::Dtype(memory.dtype) {
                            return Err(
                                "matrix device operand disagrees with selected memory contract"
                                    .into(),
                            );
                        }
                        if *name == seismic_lang::intrinsics::Operation::MatrixStore {
                            return Err("fragment store requires owned shared tile storage".into());
                        }
                        if !matches!(elem, Elem::Dtype(_)) {
                            return Err("simdgroup atoms need a dense operand".into());
                        }
                        if strides[1].as_constant() == Some(1) {
                            (
                                param,
                                offset.add(&row.mul(&strides[0])).add(&col),
                                strides[0].clone(),
                                false,
                                space,
                            )
                        } else if strides[0].as_constant() == Some(1) {
                            (
                                param,
                                offset.add(&col.mul(&strides[1])).add(&row),
                                strides[1].clone(),
                                true,
                                space,
                            )
                        } else {
                            return Err("simdgroup atoms need a unit stride along one axis".into());
                        }
                    }
                    other => {
                        return Err(format!(
                            "simdgroup operand {other:?} is not in threadgroup or device memory"
                        ));
                    }
                };
                let offset = self.target_sym(&off)?;
                let leading = self.target_sym(&ld)?;
                let transpose = want_t != col_major;
                match name {
                    seismic_lang::intrinsics::Operation::MatrixLoad
                    | seismic_lang::intrinsics::Operation::MatrixLoadTranspose => {
                        self.target(TS::MatrixLoad {
                            fragment: frag,
                            layout,
                            base: ptr,
                            offset,
                            leading,
                            space,
                            transpose,
                        })
                    }
                    _ => {
                        if col_major {
                            return Err(
                                "simdgroup_store into a column-major view is not supported".into(),
                            );
                        }
                        self.target(TS::MatrixStore {
                            fragment: frag,
                            layout,
                            base: ptr,
                            offset,
                            leading,
                            space,
                        });
                        let ExprKind::Var(destination) = args[1].kind else {
                            return Err("fragment store has no tile binding".into());
                        };
                        self.barrier(BarrierSite {
                            operation,
                            variable: destination,
                            purpose: BarrierPurpose::IntrinsicStore,
                        })?;
                    }
                }
                Ok(())
            }
            seismic_lang::intrinsics::Operation::MatrixMultiplyAccumulate => {
                let CollectiveImplementation::MultiplyAccumulate { fragments, layouts } =
                    &implementation
                else {
                    return Err("matrix multiply implementation mismatch".into());
                };
                if args
                    .iter()
                    .zip(fragments)
                    .any(|(arg, id)| !matches!(arg.kind,ExprKind::Var(v) if v==*id))
                {
                    return Err("matrix multiply operands differ from prepared execution".into());
                }
                let names: Vec<String> = args
                    .iter()
                    .map(|a| match a.kind {
                        ExprKind::Var(v) => match self.real.get(&v) {
                            Some(Realization::Frag { name }) => Ok(name.clone()),
                            _ => Err("unrealized fragment".to_string()),
                        },
                        _ => Err("fragment must be a variable".to_string()),
                    })
                    .collect::<Result<_, _>>()?;
                self.target(TS::MatrixMultiplyAccumulate {
                    fragments: names
                        .try_into()
                        .map_err(|_| "matrix multiply operand count changed")?,
                    layouts: *layouts,
                });
                Ok(())
            }
            other @ (seismic_lang::intrinsics::Operation::SimdSum
            | seismic_lang::intrinsics::Operation::SimdMax
            | seismic_lang::intrinsics::Operation::SimdMin
            | seismic_lang::intrinsics::Operation::LaneIndex
            | seismic_lang::intrinsics::Operation::ShuffleIndex
            | seismic_lang::intrinsics::Operation::Matrix) => {
                Err(format!("intrinsic `{other}` is not a statement"))
            }
        }
    }

    fn frag_decl(
        &mut self,
        v: VarId,
        layout: crate::collective::FragmentLayout,
    ) -> Result<(), String> {
        let name = format!("{}_{}", sanitize(&self.vars()[v].name), v);
        let allocation = self.execution.memory.launches()[self.memory_launch]
            .fragments
            .iter()
            .find(|a| a.variable == v)
            .ok_or("unplanned fragment storage")?;
        if allocation.layout != layout {
            return Err("fragment storage differs from selected layout".into());
        }
        layout.metal_type()?;
        self.target(TS::Fragment {
            name: name.clone(),
            layout,
        });
        self.real.insert(v, Realization::Frag { name });
        Ok(())
    }

    // Generic reads are checked as F32 even when specialization binds a narrow
    // storage type. Preserve the checked expression type before native overload
    // resolution or arithmetic, including when a load borrows device storage.
    fn read_publication(&self, value: String, ty: &Ty) -> String {
        match scalar_dtype(ty) {
            Some(dtype) => format!("{}({value})", ctype(dtype)),
            None => value,
        }
    }

    fn target_expr(&mut self, e: &Expr) -> Result<crate::terminal::Expression, String> {
        use crate::terminal::{Expression as E, Type as T};
        let ty = scalar_dtype(&e.ty).map(T::from).unwrap_or(T::I32);
        Ok(match &e.kind {
            ExprKind::Int(n) => E::Integer(*n, ty),
            ExprKind::Float(n) => E::Float(n.to_bits(), ty),
            ExprKind::Bool(v) => E::Integer(i64::from(*v), T::Bool),
            ExprKind::Var(v) => match self.real.get(v) {
                Some(Realization::Scalar { name }) if *v < self.f.params.len() => E::Parameter {
                    name: name.clone(),
                    ty,
                },
                Some(Realization::Scalar { name }) | Some(Realization::Index { name }) => {
                    E::variable(name.clone(), ty)
                }
                _ => E::Unmapped(self.expr(e)?, ty),
            },
            ExprKind::ShapeParam(_) => {
                self.target_sym(e.sym.as_ref().ok_or("shape value lacks symbol")?)?
            }
            ExprKind::Builtin {
                name: Builtin::Extent,
                args,
            } => self.target_extent(args)?,
            ExprKind::Intrinsic { op, args }
                if matches!(
                    op,
                    seismic_lang::intrinsics::Operation::LaneIndex
                        | seismic_lang::intrinsics::Operation::ShuffleIndex
                        | seismic_lang::intrinsics::Operation::SimdSum
                        | seismic_lang::intrinsics::Operation::SimdMax
                        | seismic_lang::intrinsics::Operation::SimdMin
                ) =>
            {
                let values = args
                    .iter()
                    .map(|a| self.target_expr(a))
                    .collect::<Result<Vec<_>, _>>()?;
                let implementation = self.collective(*op)?;
                match implementation {
                    CollectiveImplementation::ParticipantIndex => {
                        E::variable("lane", T::U32).cast(T::I32)
                    }
                    CollectiveImplementation::Exchange { dtype } => {
                        let [value, index]: [E; 2] = values
                            .try_into()
                            .map_err(|_| "shuffle operand count changed")?;
                        let lane = E::Helper(
                            crate::support::Helper::Index,
                            vec![index.cast(T::I64), E::Integer(SUBGROUP, T::I64)],
                            T::I64,
                        )
                        .cast(T::U32);
                        E::Builtin(
                            "simd_shuffle".into(),
                            vec![value.cast(T::F32), lane],
                            T::F32,
                        )
                        .cast(dtype.into())
                    }
                    CollectiveImplementation::Reduction { .. } => E::Builtin(
                        implementation
                            .metal_builtin()
                            .ok_or("collective builtin missing")?
                            .into(),
                        values,
                        ty,
                    ),
                    _ => return Err("scalar intrinsic implementation changed".into()),
                }
            }
            ExprKind::Index { base, indices } if matches!(base.kind, ExprKind::Accessor { .. }) => {
                let ExprKind::Accessor { base, name } = &base.kind else {
                    unreachable!()
                };
                self.target_accessor(base, name, indices)?.cast(ty)
            }
            ExprKind::Index { base, indices } => self.indexed_value(base, indices)?.cast(ty),
            ExprKind::Unary {
                op: UnaryOp::Neg,
                expr,
            } if ty == TT::I32 => Self::arithmetic(
                BinaryOp::Sub,
                TE::integer(0),
                self.target_expr(expr)?,
                TT::I32,
            ),
            ExprKind::Unary { op, expr } => E::Unary(*op, Box::new(self.target_expr(expr)?), ty),
            ExprKind::Cast { dtype, expr } => self.target_expr(expr)?.cast((*dtype).into()),
            ExprKind::Binary { op, lhs, rhs }
                if scalar_dtype(&lhs.ty).is_some_and(|t| t.is_int())
                    && matches!(
                        op,
                        BinaryOp::Div | BinaryOp::Rem | BinaryOp::Shl | BinaryOp::Shr
                    ) =>
            {
                let left = self.target_expr(lhs)?;
                let right = self.target_expr(rhs)?;
                let unsigned = scalar_dtype(&lhs.ty) == Some(DType::U32);
                let helper = match op {
                    BinaryOp::Shl | BinaryOp::Shr => {
                        if unsigned {
                            crate::support::Helper::ShiftUnsigned
                        } else {
                            crate::support::Helper::ShiftSigned
                        }
                    }
                    _ => {
                        if unsigned {
                            crate::support::Helper::DivideUnsigned
                        } else {
                            crate::support::Helper::DivideSigned
                        }
                    }
                };
                let integer = if unsigned { TT::U32 } else { TT::I32 };
                let arguments = if matches!(op, BinaryOp::Shl | BinaryOp::Shr) {
                    vec![
                        left.cast(integer),
                        right.cast(TT::I64),
                        TE::Integer(i64::from(*op == BinaryOp::Shl), TT::Bool),
                    ]
                } else {
                    vec![
                        left.cast(integer),
                        right.cast(integer),
                        TE::Integer(i64::from(*op == BinaryOp::Rem), TT::Bool),
                    ]
                };
                E::Helper(helper, arguments, integer)
            }
            ExprKind::Binary { op, lhs, rhs }
                if !matches!(
                    op,
                    BinaryOp::Div | BinaryOp::Rem | BinaryOp::Shl | BinaryOp::Shr
                ) || !scalar_dtype(&lhs.ty).is_some_and(|t| t.is_int()) =>
            {
                let left = self.target_expr(lhs)?;
                let right = self.target_expr(rhs)?;
                Self::arithmetic(*op, left, right, ty)
            }
            ExprKind::Builtin { name, args }
                if matches!(
                    name,
                    Builtin::Fma
                        | Builtin::Exp
                        | Builtin::ExpFast
                        | Builtin::Rsqrt
                        | Builtin::Sqrt
                        | Builtin::Log
                        | Builtin::Sin
                        | Builtin::Cos
                        | Builtin::Abs
                        | Builtin::Max
                        | Builtin::Min
                ) =>
            {
                let name = match name {
                    Builtin::Fma => "fma",
                    Builtin::Exp => "exp",
                    Builtin::ExpFast => "fast::exp",
                    Builtin::Rsqrt => "rsqrt",
                    Builtin::Sqrt => "sqrt",
                    Builtin::Log => "log",
                    Builtin::Sin => "sin",
                    Builtin::Cos => "cos",
                    Builtin::Abs => "abs",
                    Builtin::Max => "max",
                    Builtin::Min => "min",
                    _ => unreachable!(),
                };
                E::Builtin(
                    name.into(),
                    args.iter()
                        .map(|a| self.target_expr(a))
                        .collect::<Result<_, _>>()?,
                    ty,
                )
            }
            _ => E::Unmapped(self.expr(e)?, ty),
        })
    }
    /// Establish view metadata and guards before reading its logical extent.
    /// A snapshot's data may have been eliminated while its shape is still used.
    fn target_extent(&mut self, args: &[Expr]) -> Result<TE, String> {
        let view = args.first().ok_or("extent has no view")?;
        let axis = args
            .get(1)
            .and_then(|e| e.sym.as_ref())
            .and_then(Sym::as_constant)
            .and_then(|n| usize::try_from(n).ok())
            .ok_or("extent axis must be a nonnegative constant")?;
        let shape = self.view_shape(view)?;
        if !seismic_lang::effects::can_substitute_symbolic_value(&args[1]) {
            let value = self.target_expr(&args[1])?;
            self.target(TS::Evaluate(value));
        }
        self.target_sym(shape.get(axis).ok_or("extent axis outside view")?)
    }

    fn target_sym(&self, s: &Sym) -> Result<crate::terminal::Expression, String> {
        use crate::terminal::{Expression as E, Type as T};
        fn atom(p: &Printer<'_>, a: &Atom) -> Result<E, String> {
            Ok(match a {
                Atom::Param(name) => match p.expressions.get(name) {
                    Some(e) => e.clone(),
                    None => {
                        let actual = p
                            .names
                            .get(name)
                            .ok_or_else(|| format!("unbound symbol {name}"))?;
                        p.expressions
                            .get(actual)
                            .cloned()
                            .unwrap_or_else(|| E::variable(actual, T::I32))
                    }
                },
                Atom::Quot(a, b) => {
                    E::binary(BinaryOp::Div, p.target_sym(a)?, p.target_sym(b)?, T::I32)
                }
                Atom::Rem(a, b) => {
                    E::binary(BinaryOp::Rem, p.target_sym(a)?, p.target_sym(b)?, T::I32)
                }
            })
        }
        let mut sum: Option<E> = None;
        for (term, coefficient) in s.monomials() {
            let mut product: Option<E> = None;
            for (a, power) in term {
                for _ in 0..*power {
                    let value = atom(self, a)?;
                    product = Some(match product {
                        None => value,
                        Some(p) => {
                            let ty = T::arithmetic(p.ty(), value.ty());
                            E::binary(BinaryOp::Mul, p, value, ty)
                        }
                    });
                }
            }
            let value = match product {
                None => E::integer(coefficient),
                Some(p) if coefficient == 1 => p,
                Some(p) => {
                    let c = E::integer(coefficient);
                    let ty = T::arithmetic(c.ty(), p.ty());
                    E::binary(BinaryOp::Mul, c, p, ty)
                }
            };
            sum = Some(match sum {
                None => value,
                Some(v) => {
                    let ty = T::arithmetic(v.ty(), value.ty());
                    E::binary(BinaryOp::Add, v, value, ty)
                }
            });
        }
        Ok(sum.unwrap_or_else(|| E::integer(0)))
    }

    fn expr(&mut self, e: &Expr) -> Result<String, String> {
        match &e.kind {
            ExprKind::Load { .. } => {
                Err("selected load requires an explicit IR value binding".into())
            }

            ExprKind::Int(v) => Ok(match e.ty {
                Ty::Scalar(d) if d.is_float() => format!("{}f", *v as f64),
                _ => v.to_string(),
            }),
            ExprKind::ShapeParam(_) => self.sym(e.sym.as_ref().unwrap()),
            ExprKind::Float(v) => Ok(if v.is_infinite() {
                if *v > 0.0 {
                    "INFINITY".into()
                } else {
                    "-INFINITY".into()
                }
            } else {
                format!("{v:?}f")
            }),
            ExprKind::Bool(b) => Ok(b.to_string()),
            ExprKind::Var(v) => match self.real.get(v).cloned() {
                Some(Realization::Scalar { name })
                | Some(Realization::Index { name })
                | Some(Realization::Frag { name }) => Ok(name),
                Some(other) => Err(format!(
                    "`{}` ({other:?}) used as a value",
                    self.vars()[*v].name
                )),
                None => match e.sym.as_ref() {
                    Some(s) => self.sym(s),
                    None => Err(format!("`{}` is not realized", self.vars()[*v].name)),
                },
            },
            ExprKind::Index { base, indices } => {
                let ExprKind::Var(tv) = base.kind else {
                    // Transposition changes coordinates, not storage ownership.
                    // Recurse so materialized and distributed tiles retain their
                    // ordinary checked element reads just like borrowed views.
                    if let ExprKind::Transpose(inner) = &base.kind {
                        if indices.len() != 2 {
                            return Err("transpose element read requires two indices".into());
                        }
                        let mut read = e.clone();
                        read.kind = ExprKind::Index {
                            base: inner.clone(),
                            indices: vec![indices[1].clone(), indices[0].clone()],
                        };
                        return self.expr(&read);
                    }
                    if let ExprKind::Accessor { base: inner, name } = &base.kind {
                        let value = self.accessor_element(inner, name, indices)?;
                        return Ok(self.read_publication(value, &e.ty));
                    }
                    // A view expression (transpose or slice of a view): read through it.
                    let Realization::View {
                        space,
                        param,
                        elem,
                        offset,
                        strides,
                        shape,
                    } = self.view_of(base)?
                    else {
                        unreachable!()
                    };
                    let mut off = offset;
                    for ((i, st), extent) in indices.iter().zip(&strides).zip(&shape) {
                        let Index::Point(p) = i else {
                            return Err("slice in an element read".into());
                        };
                        let v = self.int_value(p)?;
                        let v = self.checked_index(&v, extent)?;
                        off = off.add(&v.mul(st));
                    }
                    let value = self.target_view_read(space, &param, &off, &elem)?.render();
                    return Ok(self.read_publication(value, &e.ty));
                };
                let value = self.tile_element(tv, indices)?;
                Ok(self.read_publication(value, &e.ty))
            }
            ExprKind::Accessor { .. } => Err("a packet accessor must be indexed".into()),
            ExprKind::Intrinsic { op: name, args } => match name {
                seismic_lang::intrinsics::Operation::LaneIndex
                | seismic_lang::intrinsics::Operation::ShuffleIndex => {
                    self.target_expr(e).map(|e| e.render())
                }
                seismic_lang::intrinsics::Operation::SimdSum
                | seismic_lang::intrinsics::Operation::SimdMax
                | seismic_lang::intrinsics::Operation::SimdMin => {
                    let a = self.expr(&args[0])?;
                    let implementation = self.collective(*name)?;
                    let builtin = implementation
                        .metal_builtin()
                        .ok_or("scalar collective has no builtin")?;
                    Ok(format!("{builtin}({a})"))
                }
                seismic_lang::intrinsics::Operation::Matrix => {
                    Err("simdgroup_matrix must be assigned to a variable".into())
                }
                other @ (seismic_lang::intrinsics::Operation::MatrixLoad
                | seismic_lang::intrinsics::Operation::MatrixLoadTranspose
                | seismic_lang::intrinsics::Operation::MatrixStore
                | seismic_lang::intrinsics::Operation::MatrixMultiplyAccumulate) => {
                    Err(format!("intrinsic `{other}` in expression position"))
                }
            },
            ExprKind::Builtin {
                name: Builtin::Reduce,
                ..
            } => Err("reduction expression reached emission without an IR binding".into()),
            ExprKind::Builtin {
                name: Builtin::Extent,
                args,
            } => self.target_extent(args).map(|e| e.render()),
            ExprKind::Builtin { name, args } => {
                let result = scalar_dtype(&e.ty);
                let mut a: Vec<String> = Vec::new();
                for x in args {
                    let t = self.expr(x)?;
                    a.push(match result {
                        Some(d) if !matches!(name, Builtin::Extent) => self.coerce(t, &x.ty, d),
                        _ => t,
                    });
                }
                Ok(match name {
                    Builtin::Fma => format!("fma({}, {}, {})", a[0], a[1], a[2]),
                    Builtin::Exp => format!("exp({})", a[0]),
                    Builtin::ExpFast => format!("fast::exp({})", a[0]),
                    Builtin::Rsqrt => format!("rsqrt({})", a[0]),
                    Builtin::Sqrt => format!("sqrt({})", a[0]),
                    Builtin::Log => format!("log({})", a[0]),
                    Builtin::Sin => format!("sin({})", a[0]),
                    Builtin::Cos => format!("cos({})", a[0]),
                    Builtin::Abs => format!("abs({})", a[0]),
                    Builtin::Max => format!("max({}, {})", a[0], a[1]),
                    Builtin::Min => format!("min({}, {})", a[0], a[1]),
                    Builtin::Extent => self.sym(e.sym.as_ref().unwrap())?,
                    Builtin::Reshape
                    | Builtin::Load
                    | Builtin::Store
                    | Builtin::Atomic
                    | Builtin::Reduce => return Err(format!("{name:?} is a statement")),
                })
            }
            ExprKind::Unary { op, expr } => {
                let x = self.expr(expr)?;
                Ok(match op {
                    UnaryOp::Neg => format!("(-{x})"),
                    UnaryOp::Not => format!("(!{x})"),
                    UnaryOp::BitNot => format!("(~{x})"),
                })
            }
            ExprKind::Binary { op, lhs, rhs } => {
                let l = self.expr(lhs)?;
                let r = self.expr(rhs)?;
                let is_cmp = matches!(
                    op,
                    BinaryOp::Eq
                        | BinaryOp::Ne
                        | BinaryOp::Lt
                        | BinaryOp::Le
                        | BinaryOp::Gt
                        | BinaryOp::Ge
                );
                let is_shift = matches!(op, BinaryOp::Shl | BinaryOp::Shr);
                let (l, r) = if is_shift {
                    (l, r)
                } else {
                    let target = if is_cmp {
                        match (scalar_dtype(&lhs.ty), scalar_dtype(&rhs.ty)) {
                            (Some(a), Some(b)) => DType::promote(a, b),
                            _ => None,
                        }
                    } else {
                        scalar_dtype(&e.ty)
                    };
                    match target {
                        Some(d) => (self.coerce(l, &lhs.ty, d), self.coerce(r, &rhs.ty, d)),
                        None => (l, r),
                    }
                };
                if is_shift {
                    let ty = ctype(scalar_dtype(&lhs.ty).ok_or("shift operand type missing")?);
                    return Ok(format!(
                        "seismic_shift({ty}({l}),long({r}),{},seismic_status)",
                        if *op == BinaryOp::Shl {
                            "true"
                        } else {
                            "false"
                        }
                    ));
                }
                if matches!(op, BinaryOp::Div | BinaryOp::Rem) {
                    if let Some(dtype) = scalar_dtype(&e.ty).filter(|d| d.is_int()) {
                        let ty = if dtype == DType::U32 { "uint" } else { "int" };
                        return Ok(format!(
                            "seismic_integer_division({ty}({l}), {ty}({r}), {}, seismic_status)",
                            if *op == BinaryOp::Rem {
                                "true"
                            } else {
                                "false"
                            }
                        ));
                    }
                }
                // Logical operands are evaluated eagerly, matching the portable
                // interpreter and typed scalar realization. Use `if` to guard work.
                if matches!(op, BinaryOp::And | BinaryOp::Or) {
                    return Ok(format!(
                        "bool(({l}) {} ({r}))",
                        if *op == BinaryOp::And { "&" } else { "|" }
                    ));
                }
                let o = match op {
                    BinaryOp::And | BinaryOp::Or => unreachable!(),
                    other => other.text(),
                };
                Ok(format!("({l} {o} {r})"))
            }
            ExprKind::Cast { dtype, expr } => {
                let x = self.expr(expr)?;
                Ok(format!("{}({x})", ctype(*dtype)))
            }
            ExprKind::Tuple(_) => Err("tuple in expression position".into()),
            ExprKind::TileAlloc { .. } => {
                Err("tile allocation must be assigned to a variable".into())
            }
            ExprKind::Transpose(_) => Err("transpose in expression position".into()),
            ExprKind::Lanes { .. } => Err("lanes in expression position".into()),
            ExprKind::Call { .. } => Err("call in expression position after inlining".into()),
        }
    }

    fn accessor_element(
        &mut self,
        base: &Expr,
        name: &str,
        indices: &[Index],
    ) -> Result<String, String> {
        Ok(self.target_accessor(base, name, indices)?.render())
    }
    fn target_accessor(
        &mut self,
        base: &Expr,
        name: &str,
        indices: &[Index],
    ) -> Result<TE, String> {
        let ExprKind::Var(v) = base.kind else {
            return Err("packet accessor requires variable".into());
        };
        let Some(Realization::View {
            space: _,
            param,
            elem,
            offset,
            strides,
            shape,
        }) = self.real.get(&v).cloned()
        else {
            return Err("packet accessor needs borrowed device view".into());
        };
        let Elem::Repr(r) = elem else {
            return Err("packet accessor requires packed view".into());
        };
        let r = repr::lookup(&r).ok_or("unknown representation")?;
        if indices.len() != strides.len() {
            return Err("packet accessor rank mismatch".into());
        }
        let coefficient = if name == "scale" || name == "bias" {
            r.coefficient(name == "bias")
        } else {
            None
        };
        let plane = if coefficient.is_none() {
            Some(r.plane(name).ok_or("unknown physical plane")?)
        } else {
            None
        };
        let physical = |s: &Sym| -> Sym {
            let p = plane.as_ref().unwrap();
            match p.encoding {
                repr::PlaneEncoding::Dense(_) => s
                    .quot(&Sym::constant(i64::from(p.group)))
                    .scale(i64::from(p.fields)),
                repr::PlaneEncoding::Packed { bits, .. } => s
                    .scale(i64::from(p.fields) * i64::from(bits))
                    .quot(&Sym::constant(i64::from(p.group) * 32)),
            }
        };
        let mut off = if coefficient.is_some() {
            offset
        } else {
            physical(&offset)
        };
        for (axis, index) in indices.iter().enumerate() {
            let Index::Point(e) = index else {
                return Err("packet accessor requires point coordinates".into());
            };
            let point = self.int_value(e)?;
            let last = axis + 1 == strides.len();
            let extent = if last {
                if coefficient.is_some() {
                    r.groups_extent(&shape[axis])
                } else {
                    plane.as_ref().unwrap().extent(&shape[axis])
                }
            } else {
                shape[axis].clone()
            };
            let point = self.checked_index(&point, &extent)?;
            let stride = if coefficient.is_some() {
                if last {
                    Sym::constant(i64::from(r.group))
                } else {
                    strides[axis].clone()
                }
            } else if last {
                Sym::constant(1)
            } else {
                physical(&strides[axis])
            };
            off = off.add(&point.mul(&stride));
        }
        let offset = self.target_sym(&off)?.cast(TT::I64);
        match coefficient {
            Some(c) => self.target_coefficient(&param, offset, &c),
            None => {
                let plane = plane.unwrap();
                self.target_raw_read(&format!("{param}_{}", plane.name), offset, plane.dtype())
            }
        }
    }
}

fn sanitize(name: &str) -> String {
    name.chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect()
}

fn row_major_syms(shape: &[i64]) -> Vec<Sym> {
    let mut strides = vec![1i64; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides.into_iter().map(Sym::constant).collect()
}

fn reduction_identity(identity: seismic_lang::reduction::Identity) -> &'static str {
    use seismic_lang::reduction::Identity;
    match identity {
        Identity::Zero => "0",
        Identity::One => "1",
        Identity::NegativeInfinity => "-INFINITY",
        Identity::PositiveInfinity => "INFINITY",
        Identity::MinI32 => "(-2147483647 - 1)",
        Identity::MaxI32 => "2147483647",
        Identity::MaxU32 => "0xffffffffu",
    }
}

#[cfg(test)]
mod allocation_tests {
    use super::*;

    #[test]
    fn rejects_wrong_allocation_site_even_when_declarations_match() {
        let program = seismic_lang::program::compile(&[seismic_lang::program::SourceFile {
            path: "allocation_identity.seismic.portable".into(),
            scope: seismic_lang::Scope::Portable,
            text: "fn evaluate(x: tensor[6] f32, out: tensor[6] f32):\n  a = load(x)\n  a = load(x)\n  store(a,out)\n".into(),
        }], &[]).unwrap();
        let lowered =
            seismic_lang::lower::lower(&program, "evaluate", "metal", &Default::default()).unwrap();
        let mut execution = execution::prepare(
            &lowered,
            Config {
                loads: seismic_realization::LoadStrategy::Materialize,
                ..Default::default()
            },
        )
        .unwrap();
        emit_execution(&execution).unwrap();
        let arrays = &execution.memory.launches()[0].arrays;
        assert_eq!(arrays.len(), 2);
        assert_eq!(arrays[0].declaration, arrays[1].declaration);
        let first = arrays[0].id.operation;
        let second = arrays[1].id.operation;
        let StmtKind::Parallel { body, .. } = &mut execution.function.body[0].kind else {
            panic!()
        };
        let statement = body
            .iter_mut()
            .find(|statement| statement.id == Some(first))
            .unwrap();
        statement.id = Some(second);
        execution.emission = Default::default();
        assert!(
            emit_execution(&execution)
                .unwrap_err()
                .contains("allocation site mismatch")
        );
    }

    #[test]
    fn rejects_a_missing_publication_site() {
        let program = seismic_lang::program::compile(&[seismic_lang::program::SourceFile {
            path: "publication_identity.seismic.portable".into(),
            scope: seismic_lang::Scope::Portable,
            text: "fn evaluate(out: tensor[6] f32):\n  a = tile[6] f32\n  for i in owned(a): a[i] = 1.0\n  store(a,out)\n".into(),
        }], &[]).unwrap();
        let lowered =
            seismic_lang::lower::lower(&program, "evaluate", "metal", &Default::default()).unwrap();
        let mut execution =
            execution::prepare_storage_selected(&lowered, Config::default(), &mut |_| {
                Ok(TilePlacement::GroupShared)
            })
            .unwrap();
        assert_eq!(execution.memory.launches()[0].barriers.len(), 1);
        emit_execution(&execution).unwrap();
        let StmtKind::Parallel { body, .. } = &mut execution.function.body[0].kind else {
            panic!()
        };
        let owned = body
            .iter_mut()
            .find(|statement| matches!(statement.kind, StmtKind::Owned { .. }))
            .unwrap();
        owned.id = Some(OperationId(usize::MAX));
        execution.emission = Default::default();
        assert!(
            emit_execution(&execution)
                .unwrap_err()
                .contains("omitted planned memory barriers")
        );
    }
}
