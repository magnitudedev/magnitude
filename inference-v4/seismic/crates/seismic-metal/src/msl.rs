//! MSL printer for lowered functions.
//!
//! Storage legality is derived from IR by `storage`; explicit selections are
//! validated against those domains. The diagnostic default still contains a
//! placement policy and is not an optimized Tuned IR path.
//! Load realization is explicit, with borrowing admitted by shared lifetime rules.
//! Dtype conversions preserve the checked numerical publication boundaries.

use seismic_lang::abi::ScalarParameter;
use seismic_lang::ast::{AssignOp, BinaryOp, UnaryOp};
use seismic_lang::ir::*;
use seismic_lang::lowered_ir::LoweredIr;
use crate::storage::StorageDecision;
use seismic_realization::memory::{AllocationId, Purpose, BarrierSite, BarrierPurpose, MemorySpace};
use seismic_lang::repr;
use seismic_lang::sym::{Atom, Sym};
use seismic_lang::types::{DType, Elem, Ty};
use seismic_realization::{BufferSpec, dispatch::{GroupDispatch, TileDeclaration, TilePlacement}};
use std::collections::{HashMap, HashSet};

use crate::execution::{self, Config, Execution, SUBGROUP};

#[derive(Clone, Debug)]
pub struct Emitted {
    pub source: String,
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

#[derive(Clone, Debug)]
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
}

impl Dim {
    fn is_static(&self) -> bool {
        self.ext == self.cap.to_string()
    }
}

#[derive(Clone, Debug)]
enum Realization {
    Scalar { name: String ,
    },
    Index { name: String ,
    },
    Param { name: String, shape: Vec<i64>, elem: Elem ,
    },
    View { param: String, elem: Elem, offset: Sym, strides: Vec<Sym>, shape: Vec<Sym> ,
    },
    Replicated { name: String, dims: Vec<Dim>, dtype: DType ,
    },
    Distributed { name: String, dims: Vec<Dim>, dtype: DType, slots: i64 ,
    },
    Shared { name: String, dims: Vec<Dim>, dtype: DType ,
    },
    Frag { name: String ,
    },
}


struct SplitTail {
    phase: usize,
    kernel: String,
    carried: Vec<VarId>,
    body: Vec<Stmt>,
}

struct Printer<'a> {
    f: &'a LoweredIr,
    execution: &'a Execution,
    cfg: &'a Config,
    out: String,
    real: HashMap<VarId, Realization>,
    /// atom or emitted symbol -> C name
    names: HashMap<String, String>,
    /// piece atom -> capacity
    pieces: HashMap<String, i64>,
    indent: usize,
    counter: usize,
    owned_ctx: Vec<(VarId, Vec<String>, Option<String>)>,
    buffers: Vec<BufferSpec>,
    scalars: Vec<ScalarParameter>,
    shared_decls: Vec<(String, u64)>,
    tile_declarations: Vec<TileDeclaration>,
    memory_launch: usize,
    emitted_barriers: HashSet<BarrierSite>,
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
    f: &LoweredIr, cfg: Config,
    select: &mut dyn FnMut(&StorageDecision) -> Result<TilePlacement, String>,
) -> Result<Emitted, String> {
    emit_execution(&execution::prepare_storage_selected(f, cfg, select)?)
}

/// Print an already transformed execution with resolved materialized-value placements.
/// Allocation lifetimes and intrinsic scheduling remain incomplete boundaries.
pub fn emit_execution(execution: &Execution) -> Result<Emitted, String> {
    let f = &execution.function;
    let cfg = &execution.config;
    let mut emitted = Printer { f, execution, cfg, out: String::new(),
        real: HashMap::new(), names: HashMap::new(), pieces: HashMap::new(), indent: 0, counter: 0,
        owned_ctx: Vec::new(), buffers: Vec::new(), scalars: Vec::new(), shared_decls: Vec::new(),
        tile_declarations: Vec::new(), memory_launch: 0, emitted_barriers: HashSet::new(), simdgroups: cfg.sg_per_tg,
    }.emit()?;
    let planned = execution.memory.launches();
    if emitted.launches.len() != planned.len() { return Err("emitted launch count disagrees with allocation plan".into()); }
    for (launch, plan) in emitted.launches.iter().zip(planned) {
        if launch.tiles.iter().ne(plan.arrays.iter().map(|a| &a.declaration))
            || launch.declared_threadgroup_bytes != plan.shared_bytes_per_group {
            let at = launch.tiles.iter().zip(&plan.arrays).position(|(a,b)| a != &b.declaration).unwrap_or(launch.tiles.len().min(plan.arrays.len()));
            return Err(format!("{}: emitted tile arrays disagree with the allocation plan at {at}: emitted {:?}, planned {:?}; counts {} vs {}",launch.kernel,launch.tiles.get(at),plan.arrays.get(at).map(|a|&a.declaration),launch.tiles.len(),plan.arrays.len()));
        }
    }
    for (at, (a, ad)) in execution.partition_parameters.iter().enumerate() {
        for (b, bd) in &execution.partition_parameters[at + 1..] {
            let find = |id: usize| emitted.buffers.iter().position(|slot| slot.parameter == f.vars[id].name && slot.plane.is_empty())
                .ok_or("partition parameter missing from ABI");
            emitted.alias_pairs.push((find(*a)?, find(*b)?, ad == bd));
        }
    }
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
        sym_to_c(s, &self.names)
    }

    /// Static capacity of a symbolic extent: piece atoms take their capacity.
    fn cap(&self, s: &Sym) -> Result<i64, String> {
        let mut e = s.clone();
        for a in s.atoms() {
            if let Atom::Param(p) = &a {
                if let Some(c) = self.pieces.get(p) {
                    e = e.subst(&a, &Sym::constant(*c));
                }
            }
        }
        e.as_constant().ok_or_else(|| format!("extent `{s}` has no static capacity; only piece extents may be dynamic in a tile shape"))
    }

    fn dim(&self, s: &Sym) -> Result<Dim, String> {
        let cap = self.cap(s)?;
        let ext = if s.as_constant().is_some() { cap.to_string() } else { self.sym(s)? };
        Ok(Dim { cap, ext })
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
        header.push_str(r#"
inline uint seismic_shift(uint a, long b, bool left, device atomic_uint* status) {
    if (b < 0 || b >= 32) { atomic_store_explicit(status, 1u, memory_order_relaxed); return 0; }
    return left ? a << uint(b) : a >> uint(b);
}
inline int seismic_shift(int a, long b, bool left, device atomic_uint* status) {
    if (b < 0 || b >= 32) { atomic_store_explicit(status, 1u, memory_order_relaxed); return 0; }
    return left ? as_type<int>(as_type<uint>(a) << uint(b)) : a >> uint(b);
}
inline int seismic_integer_division(int a, int b, bool remainder, device atomic_uint* status) {
    if (b == 0 || (a == (-2147483647 - 1) && b == -1)) { atomic_store_explicit(status, 1u, memory_order_relaxed); return 0; }
    long q = long(a) / long(b), r = long(a) % long(b);
    if (r < 0) { q += b < 0 ? 1 : -1; r += b < 0 ? -long(b) : long(b); }
    return int(remainder ? r : q);
}
inline uint seismic_integer_division(uint a, uint b, bool remainder, device atomic_uint* status) {
    if (b == 0) { atomic_store_explicit(status, 1u, memory_order_relaxed); return 0; }
    return remainder ? a % b : a / b;
}
inline long seismic_index(long index, long extent, device atomic_uint* status) {
    if (index < 0 || index >= extent) {
        atomic_store_explicit(status, 1u, memory_order_relaxed);
        return 0;
    }
    return index;
}
template<typename T> inline T seismic_read(device const T* pointer, long index, ulong count, device atomic_uint* status) {
    if (index < 0 || ulong(index) >= count) {
        atomic_store_explicit(status, 1u, memory_order_relaxed);
        return T(0);
    }
    return pointer[index];
}
template<typename T> inline void seismic_write(device T* pointer, long index, ulong count, T value, device atomic_uint* status) {
    if (index < 0 || ulong(index) >= count) {
        atomic_store_explicit(status, 1u, memory_order_relaxed);
        return;
    }
    pointer[index] = value;
}
"#);
        let mut index = 0usize;
        let mut params_sig: Vec<String> = Vec::new();
        for (i, (name, ty)) in self.f.params.iter().enumerate() {
            match ty {
                Ty::Tensor(s) => {
                    let shape: Vec<i64> = s.shape.iter().map(|d| {
                            d.as_constant().ok_or_else(|| {
                                format!("parameter `{name}` has a non-concrete shape")})}).collect::<Result<_, _>>()?;
                    let elements = shape.iter().try_fold(1usize, |n, d| {
                        usize::try_from(*d).ok().and_then(|d| n.checked_mul(d)).ok_or("invalid Metal tensor size")
                    })?;
                    let plane_bytes = |count: usize, width: u32| {
                        count.checked_mul(width as usize).ok_or("Metal buffer size overflow")};
                    match &s.elem {
                        Elem::Dtype(d) => {
                            params_sig.push(format!("device {}* {name} [[buffer({index})]]", ctype(*d)));
                            self.buffers.push(BufferSpec { parameter: name.clone(), plane: "".into(), bytes: plane_bytes(elements, d.bytes())?, alignment: d.bytes() as usize ,
                            });
                            index += 1;
                        }
                        Elem::Repr(r) => {
                            let rep = repr::lookup(r).ok_or("unknown Metal representation")?;
                            if shape.last().is_none_or(|n| *n % i64::from(rep.group) != 0) { return Err("Metal packed parameter requires complete groups".into()); }
                            params_sig.push(format!("device const uint* {name}_words [[buffer({index})]]"));
                            self.buffers.push(BufferSpec { parameter: name.clone(), plane: "words".into(), bytes: plane_bytes(elements / rep.codes_per_word() as usize, 4)?, alignment: 4 ,
                            });
                            index += 1;
                            params_sig.push(format!("device const {}* {name}_scale [[buffer({index})]]", ctype(rep.coefficient)));
                            self.buffers.push(BufferSpec { parameter: name.clone(), plane: "scale".into(), bytes: plane_bytes(elements / rep.group as usize, rep.coefficient.bytes(),
                                )?, alignment: rep.coefficient.bytes() as usize ,
                            });
                            index += 1;
                            if rep.has_bias {
                                params_sig.push(format!("device const {}* {name}_bias [[buffer({index})]]", ctype(rep.coefficient)));
                                self.buffers.push(BufferSpec { parameter: name.clone(), plane: "bias".into(), bytes: plane_bytes(elements / rep.group as usize, rep.coefficient.bytes(),
                                    )?, alignment: rep.coefficient.bytes() as usize ,
                                });
                                index += 1;
                            }
                        }
                        Elem::Param(p) => {
                            return Err(format!("parameter `{name}` has unresolved element type `{p}`"))}
                    }
                    self.real.insert(i, Realization::Param { name: name.clone(), shape, elem: s.elem.clone() ,
                        },
                    );
                }
                Ty::Scalar(d) => {
                    self.scalars.push(ScalarParameter::from_lowered(self.f, name, *d)?);
                    self.real.insert(i, Realization::Scalar { name: format!("sc.{name}") ,
                        },
                    );
                    if let VarKind::Index(Atom::Param(atom)) = &self.vars()[i].kind {
                        self.names.insert(atom.clone(), format!("sc.{name}"));
                    }
                }
                other => {
                    return Err(format!("parameter `{name}` of type {other} is not supported in a kernel signature"))}
            }
        }
        // Scratch the split needs, declared on every kernel so one binding list serves all.
        for scratch in self.execution.memory.scratch() {
            params_sig.push(format!("device {}* split_{} [[buffer({})]]", ctype(scratch.dtype), scratch.index, index + scratch.index));
        }
        index += self.execution.memory.scratch().len();
        if !self.scalars.is_empty() {
            header.push_str("struct Scalars {\n");
            for parameter in &self.scalars {
                let (n,d)=(&parameter.name,&parameter.dtype);
                header.push_str(&format!("  {} {n};\n", ctype(*d)));
            }
            header.push_str("};\n\n");
            params_sig.push(format!("constant Scalars& sc [[buffer({index})]]"));
        }
        params_sig.push("uint3 tg_pos [[threadgroup_position_in_grid]]".into());
        params_sig.push("uint sg_id [[simdgroup_index_in_threadgroup]]".into());
        params_sig.push("uint lane [[thread_index_in_simdgroup]]".into());

        let status_slot = index + usize::from(!self.scalars.is_empty());
        params_sig.push(format!("device atomic_uint* seismic_status [[buffer({status_slot})]]"));
        let mut launches = Vec::new();
        let body = &self.f.body;
        for (k, stmt) in body.iter().enumerate() {
            let mut split_tails = Vec::new();
            let phase = &self.execution.phases[k];
            let StmtKind::Parallel { vars, body, .. } = &stmt.kind else { unreachable!() };
            let split_here = phase.split.is_some();
            let parts = phase.parts;
            let items = phase.dispatch.work_items;
            if let Some(split) = &phase.split {
                let VarKind::Index(Atom::Param(atom)) = &self.f.vars[split.part].kind else { unreachable!() };
                self.names.insert(atom.clone(), "part".into());
                self.real.insert(split.part, Realization::Index { name: "part".into() });
            }
            let kernel = format!("{}_{k}", self.f.name);
            let mut kernel_out = String::new();
            std::mem::swap(&mut self.out, &mut kernel_out);
            self.indent = 1;
            let sg_per_tg = phase.dispatch.items_per_group as i64;
            self.simdgroups = sg_per_tg;
            self.line(&format!("const uint slot = tg_pos.x * {sg_per_tg} + sg_id;"));
            self.line(&format!("if (slot >= {items}) return;"));
            if split_here {
                // The part varies fastest so neighbouring items read neighbouring slices.
                self.line(&format!("const int part = int(slot % {parts});"));
                self.line(&format!("const uint item = slot / {parts};"));
            } else {
                self.line("const uint item = slot;");
            }
            self.bind_work_indices(vars, &phase.mapping)?;
            if split_here {
                // Each part streams its slice and publishes its carried state to scratch.
                // A second launch folds the parts together with the loop body's own merge
                // rule and runs the tail, so the kernel text never mentions either.
                let split = phase.split.as_ref().unwrap();
                self.block(&body[..split.loop_at])?;
                self.block(&split.validation_bindings)?;
                for view in &split.original_views { self.view_of(view)?; }
                let carried = split.carried.clone();
                self.block(&body[split.loop_at..split.loop_at + 1])?;
                self.publish_partials(k, &carried, "item")?;
                split_tails.push(SplitTail { phase: k, kernel: kernel.clone(), carried,
                    body: body[split.loop_at + 1..].to_vec() });
                self.block(&[])?;
            } else {
                self.block(body)?;
            }
            self.indent = 0;
            std::mem::swap(&mut self.out, &mut kernel_out);
            self.out.push_str(&format!("kernel void {kernel}(\n    {}\n) {{\n", params_sig.join(",\n    ")));
            // Resource fit: a realization whose threadgroup memory exceeds the device's is
            // not a candidate. The model must not offer it, so it is an error here.
            let shared_bytes = self.shared_decls.iter().try_fold(0u64,|sum,(_,bytes)|sum.checked_add(*bytes).ok_or("shared storage sum overflow"))?;
            if shared_bytes > self.cfg.max_threadgroup_bytes as u64 {
                return Err(format!("this realization needs {shared_bytes} bytes of threadgroup memory, over this device's {}", self.cfg.max_threadgroup_bytes));
            }
            for (d,_) in self.shared_decls.drain(..) {
                self.out.push_str(&format!("  {d}\n"));
            }
            self.out.push_str(&kernel_out);
            self.out.push_str("}\n\n");
            let after_barrier = self.execution.memory.launches()[launches.len()].predecessor.is_some();
            launches.push(Launch { kernel, threadgroups: phase.dispatch.groups, threads_per_threadgroup: phase.dispatch.threads_per_group, after_barrier,
                dispatch: Some(phase.dispatch.clone()),
                tiles: self.finish_memory()?,declared_threadgroup_bytes:shared_bytes,
            });
        // Complete this phase before a subsequent phase can observe its output.
        for SplitTail { phase: k, kernel: first, carried, body: tail } in split_tails {
            let StmtKind::Parallel { vars, .. } = &self.f.body[k].kind else { unreachable!() };
            let dispatch = self.execution.phases[k].merge_dispatch.as_ref().unwrap();
            let items = dispatch.work_items as i64;
            let kernel = format!("{first}_merge");
            let mut kernel_out = String::new();
            std::mem::swap(&mut self.out, &mut kernel_out);
            self.indent = 1;
            let sg_per_tg = dispatch.items_per_group as i64;
            self.simdgroups = sg_per_tg;
            self.line(&format!("const uint item = tg_pos.x * {sg_per_tg} + sg_id;"));
            self.line(&format!("if (item >= {items}) return;"));
            self.bind_work_indices(vars, &self.execution.phases[k].mapping)?;
            self.merge_partials(k, &carried, &self.execution.phases[k].split.as_ref().unwrap().merges, self.f.body[k].id.ok_or("merge phase has no operation identity")?)?;
            self.block(&tail)?;
            self.indent = 0;
            std::mem::swap(&mut self.out, &mut kernel_out);
            self.out.push_str(&format!("kernel void {kernel}(\n    {}\n) {{\n", params_sig.join(",\n    ")));
            let shared_bytes = self.shared_decls.iter().try_fold(0u64,|sum,(_,bytes)|sum.checked_add(*bytes).ok_or("shared storage sum overflow"))?;
            if shared_bytes > self.cfg.max_threadgroup_bytes as u64 {return Err("split merge exceeds threadgroup storage limit".into());}
            for (d,_) in self.shared_decls.drain(..) {
                self.out.push_str(&format!("  {d}\n"));
            }
            self.out.push_str(&kernel_out);
            self.out.push_str("}\n\n");
            let after_barrier = self.execution.memory.launches()[launches.len()].predecessor.is_some();
            launches.push(Launch { kernel, threadgroups: dispatch.groups, threads_per_threadgroup: dispatch.threads_per_group, after_barrier,
                dispatch: Some(dispatch.clone()),
                tiles: self.finish_memory()?,declared_threadgroup_bytes:shared_bytes,
            });
        }
        }
        Ok(Emitted { source: header + &self.out, launches, buffers: self.buffers, scalars: self.scalars, scratch: self.execution.memory.scratch().iter().map(|s| s.bytes).collect(), status_slot: Some(status_slot), alias_pairs: Vec::new(),
        })
    }

    fn bind_work_indices(&mut self, vars: &[VarId], mapping: &seismic_realization::dispatch::WorkMapping) -> Result<(), String> {
        if vars.len() != mapping.axes().len() { return Err("parallel variables do not match the work mapping".into()); }
        for (v, axis) in vars.iter().zip(mapping.axes()) {
            let name = self.index_name(*v);
            if mapping.work_items() == 0 {
                // The launch has no work. Keep its source well-defined as well.
                self.line(&format!("const int {name} = 0;"));
            } else {
                let (stride, extent, step) = (axis.stride, axis.extent, axis.step);
                self.line(&format!("const int {name} = int(((item / {stride}u) % {extent}u) * {step}u);"));
            }
        }
        Ok(())
    }

    fn index_name(&mut self, v: VarId) -> String {
        let var = &self.vars()[v];
        let VarKind::Index(Atom::Param(atom)) = &var.kind else { panic!("not an index") };
        let name = format!("{}_{}", sanitize(&var.name), v);
        self.names.insert(atom.clone(), name.clone());
        self.names.insert(name.clone(), name.clone());
        self.real.insert(v, Realization::Index { name: name.clone() });
        name
    }

    fn block(&mut self, stmts: &[Stmt]) -> Result<(), String> {
        for s in stmts {
            self.stmt(s)?;
        }
        Ok(())
    }

    fn stmt(&mut self, s: &Stmt) -> Result<(), String> {
        match &s.kind {
            StmtKind::Parallel { .. } => Err("nested `parallel` is not supported".into()),
            StmtKind::Range { var, lo, hi, body } => {
                let name = self.index_name(*var);
                let lo = self.sym(lo)?;
                let hi = self.sym(hi)?;
                self.line(&format!("for (int {name} = {lo}; {name} < {hi}; ++{name}) {{"));
                self.indent += 1;
                self.block(body)?;
                self.indent -= 1;
                self.line("}");
                Ok(())
            }
            StmtKind::Lanes { var, extent, width, body ,
            } => {
                let name = self.index_name(*var);
                // The extent may be a piece with a runtime value; the residual guarantees it is
                // a multiple of the run, so every lane takes the same number of runs.
                let runs = match extent.as_constant() {
                    Some(e) => (e / (SUBGROUP * width)).to_string(),
                    None => format!("({}) / {}", self.sym(extent)?, SUBGROUP * width),
                };
                let run = SUBGROUP * width;
                let t = self.fresh("t");
                let u = self.fresh("u");
                self.line(&format!("for (int {t} = 0; {t} < {runs}; ++{t}) {{"));
                self.indent += 1;
                self.line(&format!("for (int {u} = 0; {u} < {width}; ++{u}) {{"));
                self.indent += 1;
                self.line(&format!("const int {name} = {t} * {run} + int(lane) * {width} + {u};"));
                self.block(body)?;
                self.indent -= 1;
                self.line("}");
                self.indent -= 1;
                self.line("}");
                Ok(())
            }
            StmtKind::LoadLoop { vars, views, axis, piece, capacity, modes, body ,
            } => {
                let operation = s.id.ok_or("stream has no operation identity")?;
                let modes = modes.as_ref().filter(|m| m.len() == vars.len()).ok_or("unresolved stream loads reached Metal emission")?;
                let realized: Vec<Realization> = views.iter().map(|v| self.view_of(v)).collect::<Result<_, _>>()?;
                match capacity {
                    None => {
                        if let Some(Realization::View { shape, .. }) = realized.first() {
                            if shape.get(*axis).and_then(Sym::as_constant) == Some(0) { return Ok(()); }
                        }
                        for ((v, r), mode) in vars.iter().zip(realized).zip(modes) {
                            self.bind_stream_load(*v, r, *mode, operation)?;
                        }
                        self.block(body)
                    }
                    Some(cap) => {
                        if *cap <= 0 || *cap > i64::from(i32::MAX) { return Err("Metal stream capacity must fit a positive i32".into()); }
                        let Atom::Param(pname) = piece else { unreachable!() };
                        let Realization::View { shape, .. } = &realized[0] else { unreachable!() };
                        let ext = self.sym(&shape[*axis])?;
                        let chunk = self.fresh("chunk");
                        let pe = self.fresh("pe");
                        self.names.insert(pname.clone(), pe.clone());
                        self.pieces.insert(pname.clone(), *cap);
                        self.line(&format!("for (int {chunk} = 0; {chunk} < ({ext} / {cap} + int({ext} % {cap} != 0)); ++{chunk}) {{"));
                        self.indent += 1;
                        self.line(&format!("const int {pe} = min({cap}, {ext} - {chunk} * {cap});"));
                        for ((v, r), mode) in vars.iter().zip(realized).zip(modes) {
                            let Realization::View { param, elem, offset, strides, mut shape ,
                            } = r else { unreachable!() };
                            let offset = offset.add(&Sym::param(&chunk).mul(&Sym::constant(*cap)).mul(&strides[*axis]),
                            );
                            shape[*axis] = Sym::atom(piece.clone());
                            self.bind_stream_load(*v, Realization::View { param, elem, offset, strides, shape ,
                                }, *mode, operation,
                            )?;
                        }
                        self.block(body)?;
                        self.indent -= 1;
                        self.line("}");
                        Ok(())
                    }
                }
            }
            StmtKind::Owned { vars, tile, body } => {
                let ExprKind::Var(tv) = tile.kind else { return Err("owned() over a non-variable tile is not supported".into()) ;
                };
                let real = self.real.get(&tv).cloned().ok_or("owned() over an unrealized tile")?;
                let names: Vec<String> = vars.iter().map(|v| self.index_name(*v)).collect();
                match real {
                    Realization::Replicated { dims, .. } => {
                        for (n, d) in names.iter().zip(&dims) {
                            self.line(&format!("for (int {n} = 0; {n} < {}; ++{n}) {{", d.ext));
                            self.indent += 1;
                        }
                        self.owned_ctx.push((tv, names.clone(), None));
                        self.block(body)?;
                        self.owned_ctx.pop();
                        for _ in &dims {
                            self.indent -= 1;
                            self.line("}");
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
                        self.line(&format!("for (int {j} = 0; {j} < {slots}; ++{j}) {{"));
                        self.indent += 1;
                        self.line(&format!("const int {e} = int(lane) + {SUBGROUP} * {j};"));
                        let guard = self.distributed_guard(&e, &dims, &names);
                        self.line(&format!("if ({guard}) {{"));
                        self.indent += 1;
                        self.owned_ctx.push((tv, names.clone(), Some(j.clone())));
                        self.block(body)?;
                        self.owned_ctx.pop();
                        self.indent -= 1;
                        self.line("}");
                        self.indent -= 1;
                        self.line("}");
                        Ok(())
                    }
                    Realization::Distributed { dims, slots, .. } => {
                        let j = self.fresh("slot");
                        let e = self.fresh("e");
                        self.line(&format!("for (int {j} = 0; {j} < {slots}; ++{j}) {{"));
                        self.indent += 1;
                        self.line(&format!("const int {e} = int(lane) + {SUBGROUP} * {j};"));
                        let guard = self.distributed_guard(&e, &dims, &names);
                        self.line(&format!("if ({guard}) {{"));
                        self.indent += 1;
                        self.owned_ctx.push((tv, names.clone(), Some(j.clone())));
                        self.block(body)?;
                        self.owned_ctx.pop();
                        self.indent -= 1;
                        self.line("}");
                        self.indent -= 1;
                        self.line("}");
                        Ok(())
                    }
                    other => Err(format!("owned() over {other:?} is not supported")),
                }?;
                self.barrier(BarrierSite { operation: s.id.ok_or("owned domain has no identity")?, variable: tv, purpose: BarrierPurpose::Owned })
            }
            StmtKind::If { cond, then, els } => {
                let c = self.expr(cond)?;
                self.line(&format!("if ({c}) {{"));
                self.indent += 1;
                self.block(then)?;
                self.indent -= 1;
                if els.is_empty() {
                    self.line("}");
                } else {
                    self.line("} else {");
                    self.indent += 1;
                    self.block(els)?;
                    self.indent -= 1;
                    self.line("}");
                }
                Ok(())
            }
            StmtKind::Assign { target, op, value } => self.assign(target, *op, value, s.id.ok_or("emission requires normalized operation identities")?),
            StmtKind::Expr(e) => match &e.kind {
                ExprKind::Builtin { name: Builtin::Store, args ,
                } => self.store(&args[0], &args[1]),
                ExprKind::Builtin { name: Builtin::Atomic, .. } => Err("atomic is not yet supported on Metal".into()),
                ExprKind::Intrinsic { op: name, args } => self.intrinsic_stmt(name, args, s.id.ok_or("intrinsic has no identity")?),
                _ => {
                    let s = self.expr(e)?;
                    self.line(&format!("{s};"));
                    Ok(())
                }
            },
        }
    }

    /// Declares the index variables of a distributed element and returns the validity guard.
    fn distributed_guard(&mut self, e: &str, dims: &[Dim], names: &[String]) -> String {
        let n_cap: i64 = dims.iter().map(|d| d.cap).product();
        if n_cap == 0 {
            for name in names { self.line(&format!("const int {name} = 0;")); }
            return "false".into();
        }
        let mut stride = n_cap;
        let mut guards = vec![format!("{e} < {n_cap}")];
        for (nm, d) in names.iter().zip(dims) {
            stride /= d.cap;
            self.line(&format!("const int {nm} = ({e} / {stride}) % {};", d.cap));
            if !d.is_static() {
                guards.push(format!("{nm} < {}", d.ext));
            }
        }
        guards.join(" && ")
    }

    fn view_of(&mut self, e: &Expr) -> Result<Realization, String> {
        match &e.kind {
            ExprKind::Builtin { name: Builtin::Reshape, args ,
            } => {
                let Realization::View { param,elem,offset,strides,shape ,
                } = self.view_of(&args[0])? else { return Err("reshape requires a view".into()); };
                let resolve = |dims: &[Sym]| {
                    dims.iter().map(|s| {
                            s.as_constant().ok_or_else(|| {
                                "reshape requires statically resolved extents and strides".to_string()})}).collect::<Result<Vec<_>,_>>()};
                let target = e.ty.shaped().ok_or("reshape requires shaped result")?.shape.clone();
                let strides = seismic_lang::layout::reshape_strides(&resolve(&shape)?,&resolve(&strides)?,&resolve(&target)?,
                )?.into_iter().map(Sym::constant).collect();
                Ok(Realization::View{param,elem,offset,strides,shape:target,
                })
            }

            ExprKind::Var(v) => match self.real.get(v).cloned() {
                Some(Realization::Param { name, shape, elem }) => {
                    let strides = row_major_syms(&shape);
                    Ok(Realization::View { param: name, elem, offset: Sym::constant(0), strides, shape: shape.iter().map(|d| Sym::constant(*d)).collect() ,
                    })
                }
                Some(r @ Realization::View { .. }) => Ok(r),
                other => Err(format!("not a tensor view: {other:?}")),
            },
            ExprKind::Index { base, indices } => {
                let Realization::View { param, elem, offset, strides, shape ,
                } = self.view_of(base)? else { unreachable!() };
                let Ty::Tensor(result) = &e.ty else { return Err("indexing a tensor view must yield a view".into()) ;
                };
                let mut off = offset;
                let mut ns = Vec::new();
                let mut nst = Vec::new();
                let mut out_axis = 0;
                for (axis, ext) in shape.iter().enumerate() {
                    if axis < indices.len() {
                        match &indices[axis] {
                            Index::Point(p) => {
                                let s = self.int_value(p)?;
                                let s = self.checked_index(&s, ext)?;
                                off = off.add(&s.mul(&strides[axis]));
                            }
                            Index::Slice { start, end } => {
                                let s = match start {
                                    Some(x) => self.int_value(x)?,
                                    None => Sym::constant(0),
                                };
                                let en = match end { Some(x) => self.int_value(x)?, None => ext.clone() ,
                                };
                                let s_c = self.sym(&s)?;
                                let en_c = self.sym(&en)?;
                                let ext_c = self.sym(ext)?;
                                let valid = self.fresh("slice_valid");
                                self.line(&format!("const bool {valid} = long({s_c}) >= 0 && long({s_c}) <= long({en_c}) && long({en_c}) <= long({ext_c});"));
                                self.line(&format!("if (!{valid}) atomic_store_explicit(seismic_status, 1u, memory_order_relaxed);"));
                                let safe = self.fresh("slice_start");
                                self.line(&format!("const long {safe} = {valid} ? long({s_c}) : 0;"));
                                self.names.insert(safe.clone(), safe.clone());
                                let result_ext = result.shape[out_axis].clone();
                                if result_ext.as_constant().is_none() && !result_ext.atoms().iter().all(|a| matches!(a, Atom::Param(p) if self.names.contains_key(p))) {
                                    let atoms = result_ext.atoms();
                                    let [Atom::Param(dyn_atom)] = atoms.as_slice() else { return Err("unexpected slice extent form".into()) };
                                    let dyn_atom = dyn_atom.clone();
                                    let name = self.fresh("dyn");
                                    self.line(&format!("const int {name} = {valid} ? int(long({en_c}) - long({s_c})) : 0;"));
                                    self.names.insert(dyn_atom, name);
                                }
                                off = off.add(&Sym::param(&safe).mul(&strides[axis]));
                                ns.push(result_ext);
                                nst.push(strides[axis].clone());
                                out_axis += 1;
                            }
                        }
                    } else {
                        ns.push(ext.clone());
                        nst.push(strides[axis].clone());
                        out_axis += 1;
                    }
                }
                Ok(Realization::View { param, elem, offset: off, strides: nst, shape: ns ,
                })
            }
            ExprKind::Transpose(inner) => {
                let Realization::View { param, elem, offset, strides, shape ,
                } = self.view_of(inner)? else { unreachable!() };
                Ok(Realization::View { param, elem, offset, strides: vec![strides[1].clone(), strides[0].clone()], shape: vec![shape[1].clone(), shape[0].clone()] ,
                })
            }
            _ => Err("expression is not a tensor view".into()),
        }
    }

    /// An integer-valued expression as a symbol: static syms pass through; dynamic values are
    /// evaluated into a named C variable that the symbol refers to.
    fn int_value(&mut self, e: &Expr) -> Result<Sym, String> {
        if let Some(s) = &e.sym {
            return Ok(s.clone());
        }
        let text = self.expr(e)?;
        let name = self.fresh("iv");
        self.line(&format!("const int {name} = {text};"));
        Ok(Sym::param(&name))
    }

    fn allocation(&mut self, id: AllocationId) -> Result<TileDeclaration, String> {
        let launch = self.execution.memory.launches().get(self.memory_launch)
            .ok_or("allocation launch is missing")?;
        let allocation = launch.arrays.get(self.tile_declarations.len())
            .ok_or("unplanned tile allocation reached emission")?;
        if allocation.id != id {
            return Err(format!("allocation site mismatch: requested {id:?}, planned {:?}", allocation.id));
        }
        let declaration = allocation.declaration.clone();
        self.tile_declarations.push(declaration.clone());
        Ok(declaration)
    }

    fn barrier(&mut self, site: BarrierSite) -> Result<(), String> {
        let launch = self.execution.memory.launches().get(self.memory_launch)
            .ok_or("memory launch is missing")?;
        if let Some(barrier) = launch.barriers.get(&site) {
            if !self.emitted_barriers.insert(site) { return Err("duplicate emitted memory barrier".into()); }
            self.line(match barrier.memory {
                MemorySpace::Threadgroup => "simdgroup_barrier(mem_flags::mem_threadgroup);",
            });
        }
        Ok(())
    }

    fn finish_memory(&mut self) -> Result<Vec<TileDeclaration>, String> {
        let launch = self.execution.memory.launches().get(self.memory_launch)
            .ok_or("allocation launch is missing")?;
        if self.tile_declarations.len() != launch.arrays.len() {
            return Err("emission omitted planned tile allocations".into());
        }
        if self.emitted_barriers.len() != launch.barriers.len() {
            return Err("emission omitted planned memory barriers".into());
        }
        self.emitted_barriers.clear();
        self.memory_launch += 1;
        Ok(std::mem::take(&mut self.tile_declarations))
    }

    fn declare_tile(&mut self, v: VarId, shape: &[Sym], dtype: DType, operation: OperationId, purpose: Purpose,
    ) -> Result<Realization, String> {
        let dims = self.dims(shape)?;
        self.declare_tile_dims(v, dims, dtype, operation, purpose)
    }

    fn declare_tile_dims(&mut self, v: VarId, dims: Vec<Dim>, dtype: DType, operation: OperationId, purpose: Purpose,
    ) -> Result<Realization, String> {
        let n_cap = dims.iter().try_fold(1i64, |capacity, dim| {
            if dim.cap < 0 { return Err("negative tile extent"); }
            capacity.checked_mul(dim.cap).ok_or("tile capacity overflow")
        })?;
        let name = self.fresh(&format!("{}_{}", sanitize(&self.vars()[v].name), v));
        let declaration = self.allocation(AllocationId { operation, variable: v, purpose })?;
        if declaration.capacity != n_cap as u64 || declaration.dtype != dtype {
            return Err(format!("emitted tile {v} disagrees with its selected storage contract"));
        }
        let placement = declaration.placement.clone();
        let dispatch = GroupDispatch::new(1, SUBGROUP as u64,
            self.simdgroups.try_into().map_err(|_|"negative simdgroup count")?)?;
        let layout = declaration.layout(&dispatch)?;
        let r = if placement == TilePlacement::GroupShared {
            let sg = self.simdgroups;
            self.shared_decls.push((format!("threadgroup {} {name}[{sg}][{}];", ctype(dtype), layout.shared_elements_per_item),layout.shared_bytes_per_group));
            Realization::Shared { name: format!("{name}[sg_id]"), dims, dtype ,
            }
        } else if placement == TilePlacement::Replicated {
            self.line(&format!("{} {name}[{}];", ctype(dtype), layout.private_elements_per_lane));
            Realization::Replicated { name, dims, dtype }
        } else {
            let slots = i64::try_from(layout.private_elements_per_lane).map_err(|_|"private tile extent overflow")?;
            self.line(&format!("{} {name}[{slots}];", ctype(dtype)));
            Realization::Distributed { name, dims, dtype, slots ,
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

    fn assign(&mut self, target: &Expr, op: AssignOp, value: &Expr, operation: OperationId) -> Result<(), String> {
        match &target.kind {
            ExprKind::Var(v) => {
                match &value.kind {
                    ExprKind::TileAlloc { shape, dtype } => {
                        let Elem::Dtype(dtype) = dtype else {return Err("unresolved or packed local tile dtype".into())};
                        self.declare_tile(*v, shape, *dtype, operation, Purpose::Value)?;
                        return Ok(());
                    }
                    ExprKind::Intrinsic { op: name, args } if *name == seismic_lang::intrinsics::Operation::Matrix => {
                        let Ty::Scalar(d) = args[0].ty else { unreachable!() };
                        self.frag_decl(*v, d);
                        return Ok(());
                    }
                    ExprKind::Builtin { name: Builtin::Reduce, args ,
                    } => return self.reduce_into(*v, args, operation),
                    ExprKind::Load { view, mode } => return self.load_into(*v, view, *mode, operation),
                    _ => {}
                }
                match &target.ty {
                    Ty::Scalar(d) => {
                        let val = self.expr(value)?;
                        let val = self.coerce(val, &value.ty, *d);
                        let exists = self.real.contains_key(v);
                        let name = match self.real.get(v) {
                            Some(Realization::Scalar { name }) => name.clone(),
                            _ => format!("{}_{}", sanitize(&self.vars()[*v].name), v),
                        };
                        if !exists {
                            self.real.insert(*v, Realization::Scalar { name: name.clone() });
                            self.line(&format!("{} {name} = {val};", ctype(*d)));
                        } else {
                            self.line(&format!("{name} {} {val};", assign_text(op)));
                        }
                        Ok(())
                    }
                    Ty::Tile(shaped) => {
                        let ExprKind::Var(src) = value.kind else { return Err("tile assignment from a non-variable".into()) ;
                        };
                        let source = self.real.get(&src).cloned().ok_or("unrealized source tile")?;
                        if !self.real.contains_key(v) {
                            let dtype = shaped.elem.read_dtype().ok_or("unresolved tile assignment type")?;
                            self.declare_tile(*v, &shaped.shape, dtype, operation, Purpose::Value)?;
                        }
                        let destination = self.real.get(v).cloned().ok_or("unrealized destination tile")?;
                        self.copy_tile(&destination, &source, op, BarrierSite { operation, variable: *v, purpose: BarrierPurpose::Copy })
                    }
                    Ty::Tensor(_) => {
                        if self.real.contains_key(v) { return Err("tensor view rebinding needs explicit control-flow alias analysis".into(),
                            ); }
                        let view = self.view_of(value)?;
                        self.real.insert(*v,view);
                        Ok(())
                    }
                    other => Err(format!("assignment to {other}")),
                }
            }
            ExprKind::Index { base, indices } => {
                let ExprKind::Var(tv) = base.kind else { return Err("element assignment to a non-variable".into()) ;
                };
                let lhs = self.tile_element(tv, indices)?;
                let Ty::Scalar(d) = target.ty else { unreachable!() };
                let val = self.expr(value)?;
                let val = self.coerce(val, &value.ty, d);
                // Generic indexing retains its checked computation dtype, while
                // publication uses the concrete tile storage dtype after binding.
                let storage = base.ty.shaped().and_then(|s| match s.elem {
                    Elem::Dtype(dtype) => Some(dtype), _ => None,
                }).ok_or("element assignment requires concrete dense storage")?;
                let result = match op {
                    AssignOp::Assign => val,
                    AssignOp::Add | AssignOp::Sub | AssignOp::Mul => {
                        let operator = match op { AssignOp::Add => "+", AssignOp::Sub => "-", _ => "*" };
                        format!("({}({lhs}) {operator} ({val}))", ctype(d))
                    }
                };
                self.line(&format!("{lhs} = {}({result});", ctype(storage)));
                Ok(())
            }
            _ => Err("unsupported assignment target".into()),
        }
    }

    fn copy_tile(&mut self, destination: &Realization, source: &Realization, op: AssignOp, site: BarrierSite,
    ) -> Result<(), String> {
        let (name, dims, dtype, distributed, shared) = match destination {
            Realization::Replicated { name, dims, dtype } => (name, dims, *dtype, false, false),
            Realization::Distributed { name, dims, dtype, .. } => (name, dims, *dtype, true, false),
            Realization::Shared { name, dims, dtype } => (name, dims, *dtype, true, true),
            _ => return Err("tile value needs owned destination storage".into()),
        };
        let count: i64 = dims.iter().map(|d| d.cap).product();
        let index = self.fresh("copy");
        let flat = self.fresh("element");
        let slots = if distributed { (count + SUBGROUP - 1) / SUBGROUP } else { count };
        self.line(&format!("for (int {index} = 0; {index} < {slots}; ++{index}) {{"));
        self.indent += 1;
        self.line(&format!("const int {flat} = {};", if distributed { format!("int(lane) + {SUBGROUP} * {index}") } else { index.clone() }));
        let strides = row_major_syms(&dims.iter().map(|d| d.cap).collect::<Vec<_>>());
        let (guard, _) = self.unflatten(&flat, dims, &strides, &Sym::constant(0));
        self.line(&format!("if ({guard}) {{"));
        self.indent += 1;
        let (value, from) = match source {
            Realization::Replicated { name, dtype, .. } | Realization::Shared { name, dtype, .. } => (format!("{name}[{flat}]"), *dtype),
            Realization::Distributed { name, dtype, .. } => {
                let value = if distributed { format!("{name}[{index}]") } else { format!("simd_shuffle({name}[({flat}) / {SUBGROUP}], uint(({flat}) % {SUBGROUP}))") };
                (value, *dtype)
            }
            Realization::View { param, elem, offset, strides, .. } => {
                let (_, offset) = self.unflatten(&flat, dims, strides, offset);
                (self.read_elem(param, &offset, elem)?, elem.read_dtype().ok_or("unresolved source element")?,
                )
            }
            _ => return Err("unsupported source tile storage".into()),
        };
        let value = self.coerce(value, &Ty::Scalar(from), dtype);
        let offset = if distributed && !shared { &index } else { &flat };
        self.line(&format!("{name}[{offset}] {} {value};", assign_text(op)));
        self.indent -= 1; self.line("}");
        self.indent -= 1; self.line("}");
        self.barrier(site)
    }

    fn tile_element(&mut self, tv: VarId, indices: &[Index]) -> Result<String, String> {
        let real = self.real.get(&tv).cloned().ok_or_else(|| format!("tile `{}` is not realized", self.vars()[tv].name))?;
        let mut points = Vec::new();
        for i in indices {
            match i {
                Index::Point(p) => points.push(self.int_value(p)?),
                _ => return Err("slice of a tile element".into()),
            }
        }
        match real {
            Realization::Replicated { name, dims, .. } | Realization::Shared { name, dims, .. } => {
                let mut checked = Vec::new();
                for (point, dim) in points.iter().zip(&dims) {
                    let name = self.fresh("tile_index");
                    let point = self.sym(point)?;
                    self.names.insert(name.clone(),format!("seismic_index(long({point}), long({}), seismic_status)",dim.ext),
                    );
                    checked.push(Sym::param(&name));
                }
                let caps: Vec<i64> = dims.iter().map(|d| d.cap).collect();
                let flat = flat_index(&checked, &caps);
                Ok(format!("{name}[{}]", self.sym(&flat)?))
            }
            Realization::Distributed { name, dims, .. } => {
                for (ov, names, slot) in self.owned_ctx.iter().rev() {
                    let same_shape = match self.real.get(ov) {
                        Some(Realization::Distributed { dims: d, .. }) | Some(Realization::Shared { dims: d, .. }) => *d == dims && slot.is_some(),
                        _ => false,
                    };
                    let same_index = points.iter().zip(names).all(|(p, n)| matches!(self.atom_of(p), Some(a) if self.names.get(&a) == Some(n)));
                    if same_shape && same_index {
                        return Ok(format!("{name}[{}]", slot.clone().unwrap()));
                    }
                }
                Err(format!("element of distributed tile `{}` is read outside its owner; only same-index reads inside `owned` are realizable in this version", self.vars()[tv].name))
            }
            Realization::View { .. } | Realization::Param { .. } => {
                let (ptr, off, elem) = self.view_element(tv, &points)?;
                self.read_elem(&ptr, &off, &elem)
            }
            other => Err(format!("cannot index {other:?}")),
        }
    }

    fn atom_of(&self, s: &Sym) -> Option<String> {
        if let [Atom::Param(p)] = s.atoms().as_slice() {
            if *s == Sym::param(p) {
                return Some(p.clone());
            }
        }
        None
    }

    fn view_element(&mut self, tv: VarId, points: &[Sym]) -> Result<(String, Sym, Elem), String> {
        let r = self.real.get(&tv).cloned().unwrap();
        let (param, elem, offset, strides, shape) = match r {
            Realization::View { param, elem, offset, strides, shape ,
            } => (param, elem, offset, strides, shape),
            Realization::Param { name, shape, elem } => (name, elem, Sym::constant(0), row_major_syms(&shape), shape.iter().map(|n| Sym::constant(*n)).collect(),
            ),
            _ => unreachable!(),
        };
        let mut off = offset;
        for ((p, s), extent) in points.iter().zip(&strides).zip(&shape) {
            let p = self.checked_index(p, extent)?;
            off = off.add(&p.mul(s));
        }
        Ok((param, off, elem))
    }

    fn checked_read(&self, pointer: &str, offset: &str, bytes: u32) -> Result<String, String> {
        let slot = self.buffers.iter().find(|slot| {
            let name = if slot.plane.is_empty() { slot.parameter.clone() } else { format!("{}_{}", slot.parameter, slot.plane) };
            name == pointer
        }).ok_or_else(|| format!("missing device storage contract for `{pointer}`"))?;
        Ok(format!("seismic_read({pointer}, long({offset}), {}ul, seismic_status)", slot.bytes / bytes as usize))
    }
    fn checked_index(&mut self, index: &Sym, extent: &Sym) -> Result<Sym, String> {
        let name = self.fresh("checked_index");
        let i = self.sym(index)?;
        let n = self.sym(extent)?;
        self.names.insert(name.clone(), format!("seismic_index(long({i}), long({n}), seismic_status)"),
        );
        Ok(Sym::param(&name))
    }
    /// Raw element read: the element's own type; packed elements decode to float.
    fn read_elem(&self, ptr: &str, off: &Sym, elem: &Elem) -> Result<String, String> {
        let off_c = self.sym(off)?;
        match elem {
            Elem::Dtype(dtype) => self.checked_read(ptr, &off_c, dtype.bytes()),
            Elem::Repr(r) => {
                let rep = repr::lookup(r).unwrap();
                let cpw = rep.codes_per_word();
                let mask = (1u32 << rep.bits) - 1;
                let words = self.checked_read(&format!("{ptr}_words"), &format!("({off_c}) / {cpw}"), 4)?;
                let scale = self.checked_read(&format!("{ptr}_scale"), &format!("({off_c}) / {}", rep.group), rep.coefficient.bytes(),
                )?;
                let bias = if rep.has_bias { self.checked_read(&format!("{ptr}_bias"), &format!("({off_c}) / {}", rep.group), rep.coefficient.bytes())? } else { "0.0f".into() };
                let raw=format!("(({words} >> ((({off_c}) % {cpw}) * {})) & {mask}u)",rep.bits);
                let code=match rep.code {
                    repr::CodeInterpretation::Unsigned=>raw,
                    repr::CodeInterpretation::Offset(zero)=>format!("(int({raw}) - {zero})"),
                    repr::CodeInterpretation::TwosComplement=>format!("(int(uint({raw}) << {}) >> {})",32-rep.bits,32-rep.bits),
                    repr::CodeInterpretation::Table(table)=>{
                        let mut value=table.last().ok_or("empty code table")?.to_string();
                        for (i,n) in table.iter().enumerate().rev().skip(1) {value=format!("({raw} == {i}u ? {n} : {value})");}
                        value
                    }
                };
                Ok(format!("fma(float({code}), float({scale}), float({bias}))"))
            }
            Elem::Param(p) => Err(format!("unresolved element type `{p}`")),
        }
    }

    fn load_into(&mut self, v: VarId, view: &Expr, mode: LoadMode, operation: OperationId) -> Result<(), String> {
        let realized = self.view_of(view)?;
        if mode == LoadMode::Borrow {
            if self.real.contains_key(&v) { return Err("borrowed load must define fresh tile storage".into()); }
            self.real.insert(v, realized);
            return Ok(());
        }
        let previous = self.real.remove(&v);
        self.snapshot_into(v, realized, operation, Purpose::Value)?;
        if let Some(destination) = previous {
            let source = self.real.get(&v).cloned().ok_or("load snapshot is missing")?;
            self.copy_tile(&destination, &source, AssignOp::Assign, BarrierSite { operation, variable: v, purpose: BarrierPurpose::Copy })?;
            self.real.insert(v, destination);
        }
        Ok(())
    }

    fn bind_stream_load(&mut self, v: VarId, realized: Realization, mode: LoadMode, operation: OperationId,
    ) -> Result<(), String> {
        if mode == LoadMode::Materialize {
            self.snapshot_into(v, realized, operation, Purpose::Value)
        } else {
            self.real.insert(v, realized);
            Ok(())
        }
    }

    fn snapshot_into(&mut self, v: VarId, realized: Realization, operation: OperationId, purpose: Purpose) -> Result<(), String> {
        let Realization::View { param, elem, offset, strides, shape ,
        } = realized else { return Err("snapshot load requires a tensor view".into()) ;
        };
        let dtype = match &elem {
            Elem::Dtype(d) => *d,
            _ => DType::F32,
        };
        let from = match &elem {
            Elem::Dtype(d) => Ty::Scalar(*d),
            _ => Ty::Scalar(DType::F32),
        };
        // View extents describe this invocation; the selected tile type carries
        // allocation capacities. Split slices can have a dynamic extent even
        // when the unsplit value has a static capacity.
        let Ty::Tile(tile) = &self.vars()[v].ty else { return Err("snapshot destination is not a tile".into()); };
        if tile.shape.len() != shape.len() { return Err("snapshot rank disagrees with selected tile".into()); }
        let dims = tile.shape.iter().zip(&shape).map(|(bound, extent)| {
            let cap = self.cap(bound)?;
            if extent.as_constant().is_some_and(|n| n < 0 || n > cap) {
                return Err("snapshot extent exceeds its selected capacity".to_string());
            }
            Ok(Dim { cap, ext: self.sym(extent)? })
        }).collect::<Result<Vec<_>, String>>()?;
        let r = self.declare_tile_dims(v, dims, dtype, operation, purpose)?;
        match r {
            r @ (Realization::Replicated { .. } | Realization::Shared { .. }) => {
                let shared = matches!(r, Realization::Shared { .. });
                let (name, dims) = match r { Realization::Replicated { name, dims, .. } | Realization::Shared { name, dims, .. } => (name, dims), _ => unreachable!() ,
                };
                let n_cap: i64 = dims.iter().map(|d| d.cap).product();
                let c = self.fresh("c");
                let first = if shared { "int(lane)" } else { "0" };
                let step = if shared { SUBGROUP } else { 1 };
                self.line(&format!("for (int {c} = {first}; {c} < {n_cap}; {c} += {step}) {{"));
                self.indent += 1;
                let (guard, off) = self.unflatten(&c, &dims, &strides, &offset);
                self.line(&format!("if ({guard}) {{"));
                self.indent += 1;
                let rd = self.read_elem(&param, &off, &elem)?;
                let rd = self.coerce(rd, &from, dtype);
                self.line(&format!("{name}[{c}] = {rd};"));
                self.indent -= 1;
                self.line("}");
                self.indent -= 1;
                self.line("}");

            }
            Realization::Distributed { name, dims, slots, .. } => {
                let j = self.fresh("slot");
                let e = self.fresh("e");
                self.line(&format!("for (int {j} = 0; {j} < {slots}; ++{j}) {{"));
                self.indent += 1;
                self.line(&format!("const int {e} = int(lane) + {SUBGROUP} * {j};"));
                let (guard, off) = self.unflatten(&e, &dims, &strides, &offset);
                self.line(&format!("if ({guard}) {{"));
                self.indent += 1;
                let rd = self.read_elem(&param, &off, &elem)?;
                let rd = self.coerce(rd, &from, dtype);
                self.line(&format!("{name}[{j}] = {rd};"));
                self.indent -= 1;
                self.line("}");
                self.indent -= 1;
                self.line("}");
            }
            _ => unreachable!(),
        }
        self.barrier(BarrierSite { operation, variable: v, purpose: BarrierPurpose::Snapshot(purpose) })
    }

    /// Decompose a capacity-flat element counter into indices; returns the validity guard and
    /// the element offset through the view's strides.
    fn unflatten(&mut self, flat: &str, dims: &[Dim], strides: &[Sym], offset: &Sym,
    ) -> (String, Sym) {
        let n_cap: i64 = dims.iter().map(|d| d.cap).product();
        if n_cap == 0 { return ("false".into(), offset.clone()); }
        let mut stride_c = n_cap;
        let mut guards = vec![format!("{flat} < {n_cap}")];
        let mut off = offset.clone();
        for (k, d) in dims.iter().enumerate() {
            stride_c /= d.cap;
            let idx = self.fresh("ix");
            self.line(&format!("const int {idx} = ({flat} / {stride_c}) % {};", d.cap));
            if !d.is_static() {
                guards.push(format!("{idx} < {}", d.ext));
            }
            off = off.add(&Sym::param(&idx).mul(&strides[k]));
        }
        (guards.join(" && "), off)
    }

    fn store(&mut self, tile: &Expr, view: &Expr) -> Result<(), String> {
        let ExprKind::Var(tv) = tile.kind else { return Err("store of a non-variable tile".into()) ;
        };
        let real = self.real.get(&tv).cloned().ok_or("store of an unrealized tile")?;
        let Realization::View { param, elem, offset, strides, shape ,
        } = self.view_of(view)? else { unreachable!() };
        let Elem::Dtype(d) = elem else { return Err("store into a packed tensor".into()) ;
        };
        let count = self.buffers.iter().find(|slot| slot.parameter == param && slot.plane.is_empty()).ok_or("missing store storage contract")?.bytes / d.bytes() as usize;
        let dims = match &real {
            Realization::Replicated { dims, .. } | Realization::Shared { dims, .. } | Realization::Distributed { dims, .. } => dims,
            _ => return Err("store requires an owned tile".into()),
        };
        if dims.len() != shape.len() { return Err("store rank mismatch".into()); }
        let valid = dims.iter().zip(&shape).map(|(dim,extent)| Ok(format!("long({}) == long({})", dim.ext, self.sym(extent)?))).collect::<Result<Vec<_>,String>>()?.join(" && ");
        let valid = if valid.is_empty() { "true".into() } else { valid };
        self.line(&format!("if (!({valid})) atomic_store_explicit(seismic_status, 1u, memory_order_relaxed);"));
        match real {
            Realization::Replicated { name, dims, dtype } | Realization::Shared { name, dims, dtype } => {
                let n_cap: i64 = dims.iter().map(|d| d.cap).product();
                let c = self.fresh("c");
                self.line(&format!("for (int {c} = int(lane); {c} < {n_cap}; {c} += {SUBGROUP}) {{"));
                self.indent += 1;
                let (guard, off) = self.unflatten(&c, &dims, &strides, &offset);
                let off_c = self.sym(&off)?;
                let val = self.coerce(format!("{name}[{c}]"), &Ty::Scalar(dtype), d);
                self.line(&format!("if (({guard}) && ({valid})) seismic_write({param}, long({off_c}), {count}ul, {val}, seismic_status);"));
                self.indent -= 1;
                self.line("}");
            }
            Realization::Distributed { name, dims, dtype, slots ,
            } => {
                let j = self.fresh("slot");
                let e = self.fresh("e");
                self.line(&format!("for (int {j} = 0; {j} < {slots}; ++{j}) {{"));
                self.indent += 1;
                self.line(&format!("const int {e} = int(lane) + {SUBGROUP} * {j};"));
                let (guard, off) = self.unflatten(&e, &dims, &strides, &offset);
                let off_c = self.sym(&off)?;
                let val = self.coerce(format!("{name}[{j}]"), &Ty::Scalar(dtype), d);
                self.line(&format!("if (({guard}) && ({valid})) seismic_write({param}, long({off_c}), {count}ul, {val}, seismic_status);"));
                self.indent -= 1;
                self.line("}");
            }
            other => return Err(format!("store of {other:?}")),
        }
        Ok(())
    }

    fn reduce_into(&mut self, v: VarId, args: &[Expr], operation: OperationId) -> Result<(), String> {
        let previous = self.real.remove(&v);
        self.reduce_new(v, args, operation)?;
        if let Some(destination) = previous {
            let source = self.real.get(&v).cloned().ok_or("missing reduction result")?;
            match (&destination, &source) {
                (Realization::Scalar { name: to }, Realization::Scalar { name: from }) => {
                    self.line(&format!("{to} = {from};"));
                }
                _ => self.copy_tile(&destination, &source, AssignOp::Assign, BarrierSite { operation, variable: v, purpose: BarrierPurpose::Copy })?,
            }
            if let (VarKind::Index(Atom::Param(atom)), Realization::Scalar { name }) = (&self.vars()[v].kind, &destination) {
                self.names.insert(atom.clone(), name.clone());
            }
            self.real.insert(v, destination);
        }
        Ok(())
    }

    fn reduce_new(&mut self, v: VarId, args: &[Expr], operation: OperationId) -> Result<(), String> {
        let ExprKind::Var(tv) = args[0].kind else { return Err("reduce of a non-variable tile".into()) ;
        };
        let ExprKind::Int(axis) = args[1].kind else { unreachable!() };
        let ExprKind::Int(op) = args[2].kind else { unreachable!() };
        let axis = axis as usize;
        if op == 3 {
            return self.argmax_into(v, tv, axis, operation);
        }
        let (opname, simd) = match op {
            0 => ("+", "simd_sum"),
            1 => ("max", "simd_max"),
            2 => ("min", "simd_min"),
            _ => unreachable!(),
        };
        let selected = self.execution.reductions.get(crate::reduction::Site { output: v, operation })?.clone();
        let mut src = self.real.get(&tv).cloned().ok_or("reduce of an unrealized tile")?;
        if selected.decision.input != tv || selected.decision.materialize_input != matches!(src, Realization::View { .. }) {
            return Err("reduction input disagrees with its selected ownership".into());
        }
        if selected.decision.materialize_input {
            // A borrowed stream remains a tile value. Materialize it when this
            // reduction realization needs owned lane storage.
            self.snapshot_into(tv, src, operation, Purpose::ReductionInput)?;
            src = self.real.get(&tv).cloned().ok_or("reduction snapshot is missing")?;
        }
        let (dims, dtype) = match &src {
            Realization::Replicated { dims, dtype, .. } | Realization::Distributed { dims, dtype, .. } | Realization::Shared { dims, dtype, .. } => (dims.clone(), *dtype),
            other => return Err(format!("reduce of {other:?}")),
        };
        let contract = seismic_lang::reduction::Contract::new(ReduceOp::from_tag(op).ok_or("unknown reduction operation")?, dtype,
            matches!(args.get(3).map(|e| &e.kind),Some(ExprKind::Bool(true))));
        if contract != selected.decision.contract { return Err("fold numerical contract disagrees with selected execution".into()); }
        let init = format!("{}({})", ctype(dtype), reduction_identity(contract.identity()));
        let combine=|acc:&str,x:&str| -> String {
            if dtype == DType::Bool { return format!("({acc} {} {x})", if op == 2 { "&&" } else { "||" }); }
            if dtype == DType::I32 && op == 0 { return format!("int(clamp(long({acc}) + long({x}), long(-2147483647) - 1L, 2147483647L))"); }
            if dtype == DType::U32 && op == 0 { return format!("uint(min(ulong({acc}) + ulong({x}), 4294967295UL))"); }
            if matches!(dtype,DType::BF16|DType::F16) {
                let inner=if opname=="+" {format!("float({acc}) + float({x})")}else{format!("{opname}(float({acc}), float({x}))")};
                format!("{}({inner})",ctype(dtype))
            } else if opname=="+" {format!("{acc} + {x}")}else{format!("{opname}({acc}, {x})")}
        };
        // Narrow collective arithmetic is not admitted yet: ordered scalar
        // publication is a legal realization even when reassociation is allowed.
        let ordered = contract.ordered || matches!(dtype,DType::BF16|DType::F16) || contract.combination() == seismic_lang::reduction::Combination::SaturatingAdd;
        let mut out_dims = dims.clone();
        out_dims.remove(axis);
        let placement = match src {
            Realization::Replicated { .. } => TilePlacement::Replicated,
            Realization::Shared { .. } => TilePlacement::GroupShared,
            Realization::Distributed { .. } => TilePlacement::Distributed,
            _ => unreachable!(),
        };
        if (selected.decision.contract.operation == ReduceOp::Argmax) || selected.decision.input_placement != Some(placement.clone()) {
            return Err("fold input placement disagrees with its selected contract".into());
        }
        let reduction = crate::reduction::ReductionDomain::new(&dims.iter().map(|d| d.cap).collect::<Vec<_>>(), axis,
            dtype, ordered, placement.clone(), SUBGROUP as u64)?;
        use crate::reduction::Algorithm;
        let out_cap = reduction.output_capacity() as i64;
        let scalar_result = reduction.scalar_output();
        let inner_cap = reduction.inner_capacity() as i64;
        let axis_cap = reduction.axis_capacity() as i64;
        let axis_ext = dims[axis].ext.clone();
        let algorithm = selected.algorithm;
        if reduction != selected.decision.domain {
            return Err("emitted reduction geometry disagrees with its selected domain".into());
        }
        if let Some(expected) = selected.output {
            let declaration = self.allocation(AllocationId { operation, variable: v, purpose: Purpose::Value })?;
            if declaration != expected { return Err("reduction output disagrees with allocation plan".into()); }
        }
        let output_slots = reduction.output_slots(algorithm)?;
        let name = self.fresh("reduced");
        if algorithm == Algorithm::LaneLocal {
            let Realization::Distributed { name: sn, .. } = &src else { unreachable!() };
            let slots = output_slots as i64;
            self.real.insert(v, Realization::Distributed { name: name.clone(), dims: out_dims.clone(), dtype, slots ,
                },
            );
            self.line(&format!("{} {name}[{slots}];", ctype(dtype)));
            let j = self.fresh("slot");
            let o = self.fresh("o");
            let k = self.fresh("k");
            let acc = self.fresh("acc");
            self.line(&format!("for (int {j} = 0; {j} < {slots}; ++{j}) {{"));
            self.indent += 1;
            self.line(&format!("const int {o} = int(lane) + {SUBGROUP} * {j};"));
            self.line(&format!("{} {acc} = {init};", ctype(dtype)));
            self.line(&format!("for (int {k} = 0; {k} < {axis_ext}; ++{k}) {{"));
            self.indent += 1;
            self.line(&format!("const int e = (({o} / {inner_cap}) * {axis_cap} + {k}) * {inner_cap} + ({o} % {inner_cap});"));
            self.line(&format!("if ({o} < {out_cap}) {acc} = {};", combine(&acc, &format!("{sn}[e / {SUBGROUP}]"))));
            self.indent -= 1;
            self.line("}");
            self.line(&format!("{name}[{j}] = {acc};"));
            self.indent -= 1;
            self.line("}");
            return Ok(());
        }
        if scalar_result {
            self.real.insert(v, Realization::Scalar { name: name.clone() });
            self.line(&format!("{} {name};", ctype(dtype)));
        } else {
            self.real.insert(v, Realization::Replicated { name: name.clone(), dims: out_dims.clone(), dtype ,
                },
            );
            self.line(&format!("{} {name}[{output_slots}];", ctype(dtype)));
        }
        if out_cap == 0 { return Ok(()); }
        let o = self.fresh("o");
        self.line(&format!("for (int {o} = 0; {o} < {out_cap}; ++{o}) {{"));
        self.indent += 1;
        let acc = self.fresh("acc");
        self.line(&format!("{} {acc} = {init};", ctype(dtype)));
        // A source element e = ((o / inner) * axis_cap + k) * inner + (o % inner).
        match &src {
            Realization::Replicated { name: sn, .. } | Realization::Shared { name: sn, .. } => {
                let k = self.fresh("k");
                self.line(&format!("for (int {k} = 0; {k} < {axis_ext}; ++{k}) {{"));
                self.indent += 1;
                self.line(&format!("const int e = (({o} / {inner_cap}) * {axis_cap} + {k}) * {inner_cap} + ({o} % {inner_cap});"));
                let x = self.coerce(format!("{sn}[e]"), &Ty::Scalar(dtype), dtype);
                self.line(&format!("{acc} = {};", combine(&acc, &x)));
                self.indent -= 1;
                self.line("}");
            }
            Realization::Distributed { name:sn,.. } if algorithm == Algorithm::Ordered => {
                let k=self.fresh("k");
                self.line(&format!("for (int {k}=0; {k}<{axis_ext}; ++{k}) {{"));self.indent+=1;
                self.line(&format!("const int e = (({o} / {inner_cap}) * {axis_cap} + {k}) * {inner_cap} + ({o} % {inner_cap});"));
                let value=if matches!(dtype,DType::BF16|DType::F16) {format!("{}(simd_shuffle(float({sn}[e / {SUBGROUP}]), uint(e % {SUBGROUP})))",ctype(dtype))}else if dtype == DType::Bool {format!("bool(simd_shuffle(uint({sn}[e / {SUBGROUP}]), uint(e % {SUBGROUP})))")}else{format!("simd_shuffle({sn}[e / {SUBGROUP}], uint(e % {SUBGROUP}))")};
                self.line(&format!("{acc} = {};",combine(&acc,&value)));self.indent-=1;self.line("}");
            }
            Realization::Distributed { name: sn, slots, .. } => {
                let n_cap = reduction.input_capacity();
                let j = self.fresh("slot");
                self.line(&format!("for (int {j} = 0; {j} < {slots}; ++{j}) {{"));
                self.indent += 1;
                self.line(&format!("const int e = int(lane) + {SUBGROUP} * {j};"));
                self.line(&format!("const int k = (e / {inner_cap}) % {axis_cap};"));
                self.line(&format!("if (e < {n_cap} && k < {axis_ext} && ((e / {inner_cap}) / {axis_cap}) * {inner_cap} + (e % {inner_cap}) == {o}) {acc} = {};", combine(&acc, &format!("{sn}[{j}]"))));
                self.indent -= 1;
                self.line("}");
                self.line(&format!("{acc} = {simd}({acc});"));
            }
            _ => unreachable!(),
        }
        if scalar_result {
            self.line(&format!("{name} = {acc};"));
        } else {
            self.line(&format!("{name}[{o}] = {acc};"));
        }
        self.indent -= 1;
        self.line("}");
        Ok(())
    }

    /// `argmax` along an axis: the index of the largest value, ties to the smaller index.
    fn argmax_into(&mut self, v: VarId, tv: VarId, axis: usize, operation: OperationId) -> Result<(), String> {
        let src = self.real.get(&tv).cloned().ok_or("reduce of an unrealized tile")?;
        let (dims, dtype) = match &src {
            Realization::Replicated { dims, dtype, .. } | Realization::Distributed { dims, dtype, .. } | Realization::Shared { dims, dtype, .. } => (dims.clone(), *dtype),
            Realization::View { shape, elem, .. } => {
                let dims = shape.iter().map(|s| self.dim(s)).collect::<Result<Vec<_>, _>>()?;
                (dims, elem.read_dtype().unwrap_or(DType::F32))
            }
            other => return Err(format!("argmax of {other:?}")),
        };
        let selected = self.execution.reductions.get(crate::reduction::Site { output: v, operation })?.clone();
        let placement = match &src {
            Realization::Replicated { .. } => Some(TilePlacement::Replicated),
            Realization::Distributed { .. } => Some(TilePlacement::Distributed),
            Realization::Shared { .. } => Some(TilePlacement::GroupShared),
            Realization::View { .. } => None,
            _ => unreachable!(),
        };
        let domain = crate::reduction::ReductionDomain::argmax(&dims.iter().map(|d| d.cap).collect::<Vec<_>>(), axis,
            dtype, placement.clone(), selected.decision.full_lanes, SUBGROUP as u64)?;
        if selected.decision.contract.operation != ReduceOp::Argmax || selected.decision.contract.input != dtype || selected.decision.input != tv || selected.decision.input_placement != placement || selected.decision.domain != domain {
            return Err("argmax disagrees with its selected input/domain".into());
        }
        let algorithm = selected.algorithm;
        use crate::reduction::Algorithm;
        let mut out_dims = dims.clone();
        out_dims.remove(axis);
        let out_cap = domain.output_capacity() as i64;
        let slots = domain.output_slots(algorithm)?;
        let scalar_result = domain.scalar_output();
        if let Some(expected) = selected.output {
            let declaration = self.allocation(AllocationId { operation, variable: v, purpose: Purpose::Value })?;
            if declaration != expected { return Err("reduction output disagrees with allocation plan".into()); }
        }
        let name = self.fresh("argmax");
        if let VarKind::Index(Atom::Param(atom)) = &self.vars()[v].kind {
            self.names.insert(atom.clone(), name.clone());
        }
        if scalar_result {
            self.real.insert(v, Realization::Scalar { name: name.clone() });
            self.line(&format!("int {name};"));
        } else {
            self.real.insert(v, Realization::Replicated { name: name.clone(), dims: out_dims.clone(), dtype: DType::I32 ,
                },
            );
            self.line(&format!("int {name}[{slots}];"));
        }
        if out_cap == 0 { return Ok(()); }
        let inner_cap = domain.inner_capacity() as i64;
        let axis_cap = domain.axis_capacity() as i64;
        let axis_ext = dims[axis].ext.clone();
        let initial = reduction_identity(selected.decision.contract.identity());
        let o = self.fresh("o");
        let best = self.fresh("best");
        let at = self.fresh("at");
        self.line(&format!("for (int {o} = 0; {o} < {out_cap}; ++{o}) {{"));
        self.indent += 1;
        self.line(&format!("{} {best} = {}({initial}); int {at} = 0x7fffffff;", ctype(dtype), ctype(dtype)));
        let tie_valid = if matches!(dtype, DType::F32 | DType::F16 | DType::BF16) { format!("{at} != 0x7fffffff && ") } else { String::new() };
        match &src {
            Realization::Replicated { name: sn, .. } | Realization::Shared { name: sn, .. } => {
                let k = self.fresh("k");
                self.line(&format!("for (int {k} = 0; {k} < {axis_ext}; ++{k}) {{"));
                self.indent += 1;
                self.line(&format!("const int e = (({o} / {inner_cap}) * {axis_cap} + {k}) * {inner_cap} + ({o} % {inner_cap});"));
                self.line(&format!("if ({sn}[e] > {best} || ({sn}[e] == {best} && {tie_valid}{k} < {at})) {{ {best} = {sn}[e]; {at} = {k}; }}"));
                self.indent -= 1;
                self.line("}");
            }
            Realization::Distributed { name: sn, .. } if algorithm == Algorithm::Ordered => {
                let k = self.fresh("k");
                self.line(&format!("for (int {k} = 0; {k} < {axis_ext}; ++{k}) {{"));
                self.indent += 1;
                self.line(&format!("const int e = (({o} / {inner_cap}) * {axis_cap} + {k}) * {inner_cap} + ({o} % {inner_cap});"));
                let read = if matches!(dtype, DType::F16 | DType::BF16) {
                    format!("{}(simd_shuffle(float({sn}[e / {SUBGROUP}]), uint(e % {SUBGROUP})))", ctype(dtype))
                } else if dtype == DType::Bool { format!("bool(simd_shuffle(uint({sn}[e / {SUBGROUP}]), uint(e % {SUBGROUP})))") }
                else { format!("simd_shuffle({sn}[e / {SUBGROUP}], uint(e % {SUBGROUP}))") };
                let x = self.fresh("x");
                self.line(&format!("const {} {x} = {read};", ctype(dtype)));
                self.line(&format!("if ({x} > {best} || ({x} == {best} && {tie_valid}{k} < {at})) {{ {best} = {x}; {at} = {k}; }}"));
                self.indent -= 1;
                self.line("}");
            }
            Realization::Distributed { name: sn, slots, .. } => {
                let n_cap: i64 = dims.iter().map(|d| d.cap).product();
                let j = self.fresh("slot");
                self.line(&format!("for (int {j} = 0; {j} < {slots}; ++{j}) {{"));
                self.indent += 1;
                self.line(&format!("const int e = int(lane) + {SUBGROUP} * {j};"));
                self.line(&format!("const int k = (e / {inner_cap}) % {axis_cap};"));
                self.line(&format!("if (e < {n_cap} && k < {axis_ext} && ((e / {inner_cap}) / {axis_cap}) * {inner_cap} + (e % {inner_cap}) == {o} && ({sn}[{j}] > {best} || ({sn}[{j}] == {best} && {tie_valid}k < {at}))) {{ {best} = {sn}[{j}]; {at} = k; }}"));
                self.indent -= 1;
                self.line("}");
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
                        let outer: i64 = dims[d + 1..].iter().filter(|_| true).enumerate().map(|(i, dd)| if d + 1 + i == axis { 1 } else { dd.cap }).product();
                        let c = self.fresh("c");
                        self.names.insert(c.clone(), c.clone());
                        self.line(&format!("const int {c} = ({o} / {outer}) % {};", dims[d].cap));
                        points.push(Sym::param(&c));
                    }
                }
                let (ptr, off, elem) = self.view_element(tv, &points)?;
                let val = self.read_elem(&ptr, &off, &elem)?;
                let (start, step) = if algorithm == Algorithm::Collective { ("int(lane)", SUBGROUP) } else { ("0", 1) };
                self.line(&format!("for (int {k} = {start}; {k} < {axis_ext}; {k} += {step}) {{"));
                self.indent += 1;
                let x = self.fresh("x");
                self.line(&format!("const {} {x} = {val};", ctype(dtype)));
                self.line(&format!("if ({x} > {best} || ({x} == {best} && {tie_valid}{k} < {at})) {{ {best} = {x}; {at} = {k}; }}"));
                self.indent -= 1;
                self.line("}");
                if algorithm == Algorithm::Collective { self.line(&format!("{{ const {} m = simd_max({best}); {at} = simd_min({best} == m ? {at} : 0x7fffffff); }}", ctype(dtype))); }
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
        self.line("}");
        Ok(())
    }

    /// Write this part's carried state to compiler-allocated scratch, one region per
    /// (item, part), using the memory plan's buffer identity and layout.
    fn publish_partials(&mut self, phase: usize, carried: &[VarId], row: &str,
    ) -> Result<(), String> {
        let scratch: Vec<_> = self.execution.memory.scratch().iter().filter(|s| s.phase == phase).cloned().collect();
        if scratch.len() != carried.len() { return Err("split scratch plan does not match carried values".into()); }
        for (v, allocation) in carried.iter().zip(scratch) {
            let (name, cap) = match self.real.get(v) {
                Some(Realization::Replicated { name, dims, .. }) | Some(Realization::Shared { name, dims, .. }) => (name.clone(), dims.iter().map(|d| d.cap).product::<i64>().max(1),
                )
                ,
                Some(Realization::Distributed { name, dims, slots, .. }) => {
                    let _ = slots;
                    (name.clone(), dims.iter().map(|d| d.cap).product::<i64>().max(1),
                    )
                }
                other => {
                    return Err(format!("carried tile is {other:?}; splitting cannot publish it"))}
            };
            if allocation.variable != *v || allocation.elements_per_item != cap as u64 || allocation.dtype != DType::F32 {
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
    fn merge_partials(&mut self, phase: usize, carried: &[VarId],
        merges: &[execution::Merge], operation: OperationId,
    ) -> Result<(), String> {
        let scratch: Vec<_> = self.execution.memory.scratch().iter().filter(|s| s.phase == phase).cloned().collect();
        if carried.len() != scratch.len() || carried.len() != merges.len() {
            return Err("split handoff does not match its proven merge rules".into());
        }
        for ((v, allocation), merge) in carried.iter().zip(scratch).zip(merges) {
            if allocation.variable != *v { return Err("merge value disagrees with scratch plan".into()); }
            let buf = format!("split_{}", allocation.index);
            let parts = allocation.parts;
            let cap = allocation.elements_per_item;
            let Ty::Tile(shaped) = self.vars()[*v].ty.clone() else { return Err("carried state is not a tile".into()); };
            let r = self.declare_tile(*v, &shaped.shape, DType::F32, operation, Purpose::Merge)?;
            let (name, shared) = match r {
                Realization::Replicated { name, .. } | Realization::Distributed { name, .. } => (name, false),
                Realization::Shared { name, .. } => (name, true),
                _ => return Err("merge needs materialized state".into()),
            };
            if cap != 1 { return Err("scalar merge state has non-scalar capacity".into()); }
            if shared { self.line("if (lane == 0) {"); self.indent += 1; }
            self.line(&format!("{name}[0] = {buf}[item * {parts}];"));
            let p = self.fresh("part");
            match merge {
                execution::Merge::Sum => self.line(&format!("for (int {p} = 1; {p} < {parts}; ++{p}) {name}[0] += {buf}[item * {parts} + uint({p})];")),
            }
            if shared {
                self.indent -= 1; self.line("}");
            }
            self.barrier(BarrierSite { operation, variable: *v, purpose: BarrierPurpose::Merge })?;
        }
        Ok(())
    }

    fn intrinsic_stmt(&mut self, name: &seismic_lang::intrinsics::Operation, args: &[Expr], operation: OperationId) -> Result<(), String> {
        match name {
            seismic_lang::intrinsics::Operation::MatrixLoad | seismic_lang::intrinsics::Operation::MatrixLoadTranspose | seismic_lang::intrinsics::Operation::MatrixStore => {
                let ExprKind::Var(fv) = args[0].kind else { return Err("fragment must be a variable".into()) ;
                };
                let Some(Realization::Frag { name: frag }) = self.real.get(&fv).cloned() else { return Err("unrealized fragment".into()) ;
                };
                let row = self.int_value(&args[2])?;
                let col = self.int_value(&args[3])?;
                let operand = match args[1].kind {
                    ExprKind::Var(tv) => self.real.get(&tv).cloned().ok_or("unrealized intrinsic operand")?,
                    _ => self.view_of(&args[1])?,
                };
                // (pointer, offset of the block origin, leading dimension, memory holds the transpose)
                let (ptr, off, ld, col_major) = match operand {
                    Realization::Shared { name, dims, .. } => {
                        let ld = Sym::constant(dims[1].cap);
                        (name, row.mul(&ld).add(&col), ld, false)
                    }
                    Realization::View { param, elem, offset, strides, .. } => {
                        if *name == seismic_lang::intrinsics::Operation::MatrixStore { return Err("fragment store requires owned shared tile storage".into()); }
                        if !matches!(elem, Elem::Dtype(_)) {
                            return Err("simdgroup atoms need a dense operand".into());
                        }
                        if strides[1].as_constant() == Some(1) {
                            (param, offset.add(&row.mul(&strides[0])).add(&col), strides[0].clone(), false,
                            )
                        } else if strides[0].as_constant() == Some(1) {
                            (param, offset.add(&col.mul(&strides[1])).add(&row), strides[1].clone(), true,
                            )
                        } else {
                            return Err("simdgroup atoms need a unit stride along one axis".into());
                        }
                    }
                    other => {
                        return Err(format!("simdgroup operand {other:?} is not in threadgroup or device memory"))}
                };
                let off_c = self.sym(&off)?;
                let ld_c = self.sym(&ld)?;
                let want_t = *name == seismic_lang::intrinsics::Operation::MatrixLoadTranspose;
                let transpose = want_t != col_major;
                match name {
                    seismic_lang::intrinsics::Operation::MatrixLoad | seismic_lang::intrinsics::Operation::MatrixLoadTranspose => self.line(&format!("simdgroup_load({frag}, {ptr} + ({off_c}), {ld_c}, ulong2(0, 0), {transpose});")),
                    _ => {
                        if col_major {
                            return Err("simdgroup_store into a column-major view is not supported".into());
                        }
                        self.line(&format!("simdgroup_store({frag}, {ptr} + ({off_c}), {ld_c});"));
                        let ExprKind::Var(destination) = args[1].kind else { return Err("fragment store has no tile binding".into()); };
                        self.barrier(BarrierSite { operation, variable: destination, purpose: BarrierPurpose::IntrinsicStore })?;
                    }
                }
                Ok(())
            }
            seismic_lang::intrinsics::Operation::MatrixMultiplyAccumulate => {
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
                self.line(&format!("simdgroup_multiply_accumulate({}, {}, {}, {});", names[0], names[1], names[2], names[3]));
                Ok(())
            }
            other @ (seismic_lang::intrinsics::Operation::SimdSum
                | seismic_lang::intrinsics::Operation::SimdMax
                | seismic_lang::intrinsics::Operation::SimdMin
                | seismic_lang::intrinsics::Operation::Matrix) => Err(format!("intrinsic `{other}` is not a statement")),
        }
    }

    fn frag_decl(&mut self, v: VarId, dtype: DType) -> String {
        let name = format!("{}_{}", sanitize(&self.vars()[v].name), v);
        let elem = match dtype {
            DType::BF16 => "bfloat",
            DType::F16 => "half",
            DType::F32 => "float",
            other => panic!("no simdgroup matrix of {}", other.name()),
        };
        self.line(&format!("simdgroup_{elem}8x8 {name};"));
        self.real.insert(v, Realization::Frag { name: name.clone() });
        name
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

    fn expr(&mut self, e: &Expr) -> Result<String, String> {
        match &e.kind {
            ExprKind::Load { .. } => Err("selected load requires an explicit IR value binding".into()),

            ExprKind::Int(v) => Ok(match e.ty {
                Ty::Scalar(d) if d.is_float() => format!("{}f", *v as f64),
                _ => v.to_string(),
            }),
            ExprKind::ShapeParam(_) => self.sym(e.sym.as_ref().unwrap()),
            ExprKind::Float(v) => Ok(if v.is_infinite() { if *v > 0.0 { "INFINITY".into() } else { "-INFINITY".into() } } else { format!("{v:?}f") }),
            ExprKind::Bool(b) => Ok(b.to_string()),
            ExprKind::Var(v) => match self.real.get(v).cloned() {
                Some(Realization::Scalar { name }) | Some(Realization::Index { name }) | Some(Realization::Frag { name }) => Ok(name),
                Some(other) => Err(format!("`{}` ({other:?}) used as a value", self.vars()[*v].name)),
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
                    let Realization::View { param, elem, offset, strides, shape ,
                    } = self.view_of(base)? else { unreachable!() };
                    let mut off = offset;
                    for ((i, st), extent) in indices.iter().zip(&strides).zip(&shape) {
                        let Index::Point(p) = i else { return Err("slice in an element read".into()) ;
                        };
                        let v = self.int_value(p)?;
                        let v = self.checked_index(&v, extent)?;
                        off = off.add(&v.mul(st));
                    }
                    let value = self.read_elem(&param, &off, &elem)?;
                    return Ok(self.read_publication(value, &e.ty));
                };
                let value = self.tile_element(tv, indices)?;
                Ok(self.read_publication(value, &e.ty))
            }
            ExprKind::Accessor { .. } => Err("a packet accessor must be indexed".into()),
            ExprKind::Intrinsic { op: name, args } => match name {
                seismic_lang::intrinsics::Operation::SimdSum | seismic_lang::intrinsics::Operation::SimdMax | seismic_lang::intrinsics::Operation::SimdMin => {
                    let a = self.expr(&args[0])?;
                    Ok(format!("{name}({a})"))
                }
                seismic_lang::intrinsics::Operation::Matrix => Err("simdgroup_matrix must be assigned to a variable".into()),
                other @ (seismic_lang::intrinsics::Operation::MatrixLoad
                    | seismic_lang::intrinsics::Operation::MatrixLoadTranspose
                    | seismic_lang::intrinsics::Operation::MatrixStore
                    | seismic_lang::intrinsics::Operation::MatrixMultiplyAccumulate) => Err(format!("intrinsic `{other}` in expression position")),
            },
            ExprKind::Builtin { name: Builtin::Reduce, .. } => Err("reduction expression reached emission without an IR binding".into()),
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
                    Builtin::Reshape | Builtin::Load | Builtin::Store | Builtin::Atomic | Builtin::Reduce => return Err(format!("{name:?} is a statement")),
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
                let is_cmp = matches!(op, BinaryOp::Eq | BinaryOp::Ne | BinaryOp::Lt | BinaryOp::Le | BinaryOp::Gt | BinaryOp::Ge);
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
                    let ty=ctype(scalar_dtype(&lhs.ty).ok_or("shift operand type missing")?);
                    return Ok(format!("seismic_shift({ty}({l}),long({r}),{},seismic_status)",if *op==BinaryOp::Shl {"true"}else{"false"}));
                }
                if matches!(op, BinaryOp::Div | BinaryOp::Rem) {
                    if let Some(dtype) = scalar_dtype(&e.ty).filter(|d| d.is_int()) {
                        let ty = if dtype == DType::U32 { "uint" } else { "int" };
                        return Ok(format!("seismic_integer_division({ty}({l}), {ty}({r}), {}, seismic_status)", if *op == BinaryOp::Rem { "true" } else { "false" }));
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
                Err("tile allocation must be assigned to a variable".into())}
            ExprKind::Transpose(_) => Err("transpose in expression position".into()),
            ExprKind::Lanes { .. } => Err("lanes in expression position".into()),
            ExprKind::Call { .. } => Err("call in expression position after inlining".into()),
        }
    }

    fn accessor_element(&mut self, base: &Expr, name: &str, indices: &[Index],
    ) -> Result<String, String> {
        let ExprKind::Var(tv) = base.kind else { return Err("accessor on a non-variable".into()) ;
        };
        let Some(Realization::View { param, elem, offset, strides, shape ,
        }) = self.real.get(&tv).cloned() else {
            return Err("packet accessors need a global view in this version".into());
        };
        let Elem::Repr(r) = elem else { return Err("accessor on a dense view".into()) ;
        };
        let rep = repr::lookup(&r).unwrap();
        if indices.len() != strides.len() {
            return Err(format!("accessor takes {} point indices", strides.len()));
        }
        let (ptr, div) = match name {
            "words" => (format!("{param}_words"), rep.codes_per_word() as i64),
            "scale" => (format!("{param}_scale"), rep.group as i64),
            _ => (format!("{param}_bias"), rep.group as i64),
        };
        // Leading axes keep their strides in packets; the packet axis is indexed directly.
        let mut off = offset.quot(&Sym::constant(div));
        for (axis, idx) in indices.iter().enumerate() {
            let Index::Point(p) = idx else { return Err("accessor indices must be points".into()) ;
            };
            let i = self.int_value(p)?;
            let extent = if axis + 1 == strides.len() { shape[axis].quot(&Sym::constant(div)) } else { shape[axis].clone() };
            let i = self.checked_index(&i, &extent)?;
            if axis + 1 == strides.len() {
                off = off.add(&i);
            } else {
                off = off.add(&i.mul(&strides[axis].quot(&Sym::constant(div))));
            }
        }
        let off_c = self.sym(&off)?;
        self.checked_read(&ptr, &off_c, if name == "words" { 4 } else { rep.coefficient.bytes() },
        )
    }
}

fn assign_text(op: AssignOp) -> &'static str {
    match op {
        AssignOp::Assign => "=",
        AssignOp::Add => "+=",
        AssignOp::Sub => "-=",
        AssignOp::Mul => "*=",
    }
}

fn sanitize(name: &str) -> String {
    name.chars().map(|c| if c.is_ascii_alphanumeric() { c } else { '_' }).collect()
}

fn row_major_syms(shape: &[i64]) -> Vec<Sym> {
    let mut strides = vec![1i64; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides.into_iter().map(Sym::constant).collect()
}

fn flat_index(points: &[Sym], caps: &[i64]) -> Sym {
    let mut flat = Sym::constant(0);
    for (p, d) in points.iter().zip(caps) {
        flat = flat.mul(&Sym::constant(*d)).add(p);
    }
    flat
}

/// Print a symbolic integer as a C expression; atoms map to their C names.
pub fn sym_to_c(s: &Sym, names: &HashMap<String, String>) -> Result<String, String> {
    if let Some(c) = s.as_constant() {
        return Ok(c.to_string());
    }
    let text = s.to_string();
    let mut out = String::new();
    let mut ident = String::new();
    let flush = |ident: &mut String, out: &mut String| -> Result<(), String> {
        if ident.is_empty() {
            return Ok(());
        }
        if ident.chars().all(|c| c.is_ascii_digit()) {
            out.push_str(ident);
        } else if let Some(n) = names.get(ident.as_str()) {
            out.push_str(n);
        } else {
            return Err(format!("unbound symbol `{ident}` in `{text}`"));
        }
        ident.clear();
        Ok(())
    };
    for ch in text.chars() {
        if ch.is_ascii_alphanumeric() || ch == '_' || ch == '#' {
            ident.push(ch);
        } else {
            flush(&mut ident, &mut out)?;
            out.push(ch);
        }
    }
    flush(&mut ident, &mut out)?;
    Ok(format!("({out})"))
}

fn reduction_identity(identity: seismic_lang::reduction::Identity) -> &'static str {
    use seismic_lang::reduction::Identity;
    match identity {
        Identity::Zero => "0", Identity::One => "1",
        Identity::NegativeInfinity => "-INFINITY", Identity::PositiveInfinity => "INFINITY",
        Identity::MinI32 => "(-2147483647 - 1)", Identity::MaxI32 => "2147483647", Identity::MaxU32 => "0xffffffffu",
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
        let lowered = seismic_lang::lower::lower(&program, "evaluate", "metal", &Default::default()).unwrap();
        let mut execution = execution::prepare(&lowered, Config {
            loads: seismic_realization::LoadStrategy::Materialize,
            ..Default::default()
        }).unwrap();
        emit_execution(&execution).unwrap();
        let arrays = &execution.memory.launches()[0].arrays;
        assert_eq!(arrays.len(), 2);
        assert_eq!(arrays[0].declaration, arrays[1].declaration);
        let first = arrays[0].id.operation;
        let second = arrays[1].id.operation;
        let StmtKind::Parallel { body, .. } = &mut execution.function.body[0].kind else { panic!() };
        let statement = body.iter_mut().find(|statement| statement.id == Some(first)).unwrap();
        statement.id = Some(second);
        assert!(emit_execution(&execution).unwrap_err().contains("allocation site mismatch"));
    }

    #[test]
    fn rejects_a_missing_publication_site() {
        let program = seismic_lang::program::compile(&[seismic_lang::program::SourceFile {
            path: "publication_identity.seismic.portable".into(),
            scope: seismic_lang::Scope::Portable,
            text: "fn evaluate(out: tensor[6] f32):\n  a = tile[6] f32\n  for i in owned(a): a[i] = 1.0\n  store(a,out)\n".into(),
        }], &[]).unwrap();
        let lowered = seismic_lang::lower::lower(&program, "evaluate", "metal", &Default::default()).unwrap();
        let mut execution = execution::prepare_storage_selected(&lowered, Config::default(),
            &mut |_| Ok(TilePlacement::GroupShared)).unwrap();
        assert_eq!(execution.memory.launches()[0].barriers.len(), 1);
        emit_execution(&execution).unwrap();
        let StmtKind::Parallel { body, .. } = &mut execution.function.body[0].kind else { panic!() };
        let owned = body.iter_mut().find(|statement| matches!(statement.kind, StmtKind::Owned { .. })).unwrap();
        owned.id = Some(OperationId(usize::MAX));
        assert!(emit_execution(&execution).unwrap_err().contains("omitted planned memory barriers"));
    }

}
