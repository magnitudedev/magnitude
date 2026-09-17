//! MSL printer for lowered functions.
//!
//! Realization on Metal, this version:
//! - every top-level `parallel` block is one compute kernel; a work item is one simdgroup;
//!   `sg_per_tg` simdgroups form a threadgroup;
//! - a tile with at most 32 elements of capacity is *replicated*: every lane holds all of it
//!   and computes it uniformly; a larger tile is *distributed*: element `e` lives in lane
//!   `e % 32`, slot `e / 32`; a tile passed to simdgroup intrinsics lives in threadgroup memory;
//! - a tile bound by a `load(..., over=axis)` loop is a *global view*: reads index device memory.
//!   Over a dynamic extent the loop runs in pieces of a static capacity with a runtime extent;
//! - a `load(view)` tile is materialized into its register realization;
//! - dtype conversions are inserted exactly where the checker's types change.

use seismic_lang::abi::ScalarParameter;
use seismic_lang::ast::{AssignOp, BinaryOp, UnaryOp};
use seismic_lang::hir::*;
use seismic_lang::lower::Lowered;
use seismic_lang::repr;
use seismic_lang::sym::{Atom, Sym};
use seismic_lang::types::{DType, Elem, Ty};
use seismic_realization::{BufferSpec, dispatch::{GroupDispatch, TileDeclaration, TilePlacement}};
use std::collections::HashMap;

pub const SUBGROUP: i64 = 32;

/// Realization choices the model will close; defaults for now.
#[derive(Clone, Debug)]
pub struct Config {
    pub sg_per_tg: i64,
    /// Piece capacity for static streaming extents; `None` streams whole axes.
    pub piece: Option<i64>,
    /// How many consecutive values of the innermost `parallel` index one work item covers.
    /// A free integer of the realization: kernels never name it.
    pub per_item: i64,
    /// How many simdgroups of one threadgroup share a splittable streamed range, each
    /// taking a slice and merging through threadgroup memory. A free integer of the
    /// realization: kernels never name it. 1 leaves the range whole.
    pub split: i64,
    /// Explicit independent pointwise tile extent per work item. Requires a
    /// checked partition proof and runtime overlap validation; no implicit policy.
    pub tile_piece: Option<i64>,
    /// Device capacity facts, queried. The performance model will weigh
    /// parallelism against reuse; it is carried so no part of the compiler invents it.
    pub max_threads_per_threadgroup: i64,
    pub max_threadgroup_bytes: i64,
}

impl Default for Config {
    fn default() -> Config {
        Config { sg_per_tg: 4, piece: None, per_item: 1, split: 1, tile_piece: None, max_threads_per_threadgroup: 1024, max_threadgroup_bytes: 32768 ,
        }
    }
}

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

struct Printer<'a> {
    f: &'a Lowered,
    cfg: Config,
    out: String,
    /// Tiles read at indices other than their own inside an `owned` loop, or outside any.
    cross_read: std::collections::HashSet<VarId>,
    /// Tiles written element-wise or by an intrinsic store.
    written: std::collections::HashSet<VarId>,
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
    /// Simdgroups per threadgroup for the kernel being emitted.
    simdgroups: i64,
    /// The variable table, extended by rewrites that introduce variables.
    extra_vars: Vec<Var>,
    /// Byte sizes of compiler-allocated scratch buffers.
    scratch: Vec<usize>,
}

pub fn emit(f: &Lowered) -> Result<Emitted, String> {
    emit_with(f, Config::default())
}

pub fn emit_with(f: &Lowered, cfg: Config) -> Result<Emitted, String> {
    if cfg.sg_per_tg <= 0 || cfg.per_item <= 0 || cfg.split <= 0
        || cfg.sg_per_tg.checked_mul(SUBGROUP).is_none_or(|n| n > cfg.max_threads_per_threadgroup)
        || cfg.max_threadgroup_bytes < 0 {
        return Err("invalid Metal candidate or device resource limits".into());
    }
    if cfg.per_item > 1 && cfg.split > 1 {
        return Err("combined widening and splitting is not realized".into());
    }
    // A serial root is one logical work item. Its existence must not require
    // authors to wrap natural sequential code in a synthetic parallel loop.
    let serial_root;
    let f = if !f.body.iter().any(|s| matches!(s.kind, StmtKind::Parallel { .. })) {
        serial_root = Lowered { body: vec![Stmt { kind: StmtKind::Parallel {
            vars: Vec::new(), extents: Vec::new(), body: f.body.clone(),
        }, span: f.body.first().map(|s| s.span).unwrap_or_default() ,
            }], ..f.clone() };
        &serial_root
    } else { f };
    // An explicit candidate either realizes its declared choices or is rejected.
    // Silently shrinking widening/splitting would invalidate accounting, cache
    // identities and any comparison with the requested realization.
    if let Some(piece) = cfg.tile_piece {
        if cfg.per_item != 1 || cfg.split != 1 {
            return Err("pointwise partition cannot combine with widening or reduction splitting".into());
        }
        let partitioned = seismic_lang::partition::pointwise(f,piece)?;
        let mut emitted = emit_exact(&partitioned.function,cfg)?;
        for (at,(a,ad)) in partitioned.parameters.iter().enumerate() {
            for (b,bd) in &partitioned.parameters[at+1..] {
                let find = |id:usize| emitted.buffers.iter().position(|slot|slot.parameter == f.vars[id].name && slot.plane.is_empty()).ok_or("partition parameter missing from ABI");
                emitted.alias_pairs.push((find(*a)?,find(*b)?,ad==bd));
            }
        }
        Ok(emitted)
    } else { emit_exact(f, cfg) }
}

fn emit_exact(f: &Lowered, cfg: Config) -> Result<Emitted, String> {
    let (cross_read, written) = analyze_usage(f);
    let cfg_sg = cfg.sg_per_tg;
    Printer { f, cfg, out: String::new(), cross_read, written, real: HashMap::new(), names: HashMap::new(), pieces: HashMap::new(), indent: 0, counter: 0, owned_ctx: Vec::new(), buffers: Vec::new(), scalars: Vec::new(), shared_decls: Vec::new(), tile_declarations: Vec::new(), simdgroups: cfg_sg, extra_vars: Vec::new(), scratch: Vec::new() ,
    }.emit()
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

impl<'a> Printer<'a> {
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

    /// The variables in scope: the function's, plus any a rewrite introduced.
    fn vars(&self) -> &[Var] {
        if self.extra_vars.is_empty() {
            &self.f.vars
        } else {
            &self.extra_vars
        }
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
        let scratch_count = if self.cfg.split > 1 {
            seismic_lang::split::splittable(&self.f.body, &self.vars()).iter().map(|sp| sp.carried.len()).sum::<usize>()
        } else {
            0
        };
        for n in 0..scratch_count {
            params_sig.push(format!("device float* split_{n} [[buffer({})]]", index + n));
        }
        index += scratch_count;
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
        let mut split_tails: Vec<(usize, String, Vec<seismic_lang::hir::VarId>, Vec<(String, usize, i64)>, i64, i64, Vec<Stmt>,
        )> = Vec::new();
        let mut body = self.f.body.clone();
        // A streamed range that carries state is the only sequential work a `parallel` block
        // has. When the model splits it, the simdgroups of one threadgroup each take a slice
        // and merge afterwards, so the item count is unchanged and no scratch is needed.
        // The parts of a split are the simdgroups of one threadgroup, so a Metal threadgroup's
        // limit bounds it: at most 1024 threads, and every part must be a real simdgroup.
        let splits = if self.cfg.split > 1 { seismic_lang::split::splittable(&body, &self.vars()) } else { Vec::new() };
        if self.cfg.split > 1 && splits.is_empty() {
            return Err("requested split has no legal streamed reduction".into());
        }
        let mut split_state: Vec<(usize, Vec<seismic_lang::hir::VarId>)> = Vec::new();
        for sp in &splits {
            let atom = Atom::Param(format!("part#{}", self.counter));
            self.counter += 1;
            let part = self.vars().len() + split_state.len();
            {
                let StmtKind::Parallel { body: block, .. } = &body[sp.stmt].kind else { unreachable!() };
                let StmtKind::LoadLoop { body: loop_body, .. } = &block[sp.loop_at].kind else { unreachable!() };
                let names = seismic_lang::rewrite::namer(&self.vars());
                seismic_lang::rewrite::check_split(loop_body, &sp.carried, &names).map_err(|e| e.to_string())?;
                
            }
            seismic_lang::split::narrow_range(sp, &mut body, part, &atom, self.cfg.split)?;
            self.names.insert(match &atom { Atom::Param(p) => p.clone(), _ => unreachable!() ,
                }, "part".to_string(),
            );
            split_state.push((sp.stmt, sp.carried.clone()));
        }
        for (k, stmt) in body.iter().enumerate() {
            let StmtKind::Parallel { vars, extents, body ,
            } = &stmt.kind else {
                return Err("every top-level statement of a kernel must be a `parallel` block".into(),
                );
            };
            let extents: Vec<i64> = extents.iter().map(|e| e.as_constant().ok_or("parallel extent is not concrete")).collect::<Result<_, _>>()?;
            // One work item covers `per_item` consecutive values of the innermost index, so
            // the weight rows it reads are contiguous and its activation reads are shared.
            // The declared widening must divide the domain; tail widening is
            // a different realization and cannot silently replace this choice.
            let inner = *extents.last().unwrap_or(&1);
            let splits_here = split_state.iter().any(|(at, _)| *at == k);
            if inner % self.cfg.per_item != 0 {
                return Err(format!("requested widening {} does not divide parallel extent {inner}",self.cfg.per_item));
            }
            let per_item = self.cfg.per_item;
            let base_items: i64 = extents.iter().product::<i64>() / per_item;
            // A split gives every part its own work item, so the machine fills with
            // threadgroups rather than with simdgroups of one threadgroup.
            let split_here = splits_here;
            let parts = if split_here { self.cfg.split } else { 1 };
            let items: i64 = base_items * parts;
            let kernel = format!("{}_{k}", self.f.name);
            let mut kernel_out = String::new();
            std::mem::swap(&mut self.out, &mut kernel_out);
            self.indent = 1;
            let sg_per_tg = self.cfg.sg_per_tg;
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
            // Each index's own extent, with the innermost reduced by the run one item covers.
            // The strides come from these reduced extents, so the two rewrites compose.
            let last = vars.len().saturating_sub(1);
            let reduced: Vec<i64> = extents.iter().enumerate().map(|(i, e)| if i == last { (e / per_item).max(1) } else { *e }).collect();
            let mut stride: i64 = reduced.iter().product();
            let mut inner_name = String::new();
            for (i, (v, e)) in vars.iter().zip(&reduced).enumerate() {
                let e = *e;
                stride /= e.max(1);
                let name = self.index_name(*v);
                if i == last && per_item > 1 {
                    // The innermost index becomes the base of this item's run.
                    inner_name = name.clone();
                    self.line(&format!("const int {name}_base = ((item / {stride}) % {e}) * {per_item};"));
                } else {
                    self.line(&format!("const int {name} = (item / {stride}) % {e};"));
                }
            }
            if per_item > 1 {
                // One item covers several values of the innermost index. The body is rewritten
                // so the statements that depend on that index exist once per covered value,
                // each with its own state, while everything else stays single. Work the covered
                // values share, such as reading the activation row they all multiply against,
                // is therefore done once instead of once per value.
                let inner_atom = match &self.vars()[vars[last]].kind {
                    VarKind::Index(a) => a.clone(),
                    _ => unreachable!(),
                };
                let base = Expr {
                    kind: ExprKind::Var(vars[last]),
                    ty: Ty::Scalar(DType::I32),
                    sym: Some(Sym::atom(inner_atom.clone())),
                    span: stmt.span,
                };
                let mut vars_mut: Vec<Var> = self.vars().to_vec();
                let widened = seismic_lang::widen::apply(body, vars[last], &inner_atom, per_item, &mut vars_mut, &base,
                );
                self.extra_vars = vars_mut;
                self.line(&format!("const int {inner_name} = {inner_name}_base;"));
                let mut terms: Vec<String> = Vec::new();
                let mut place = 1i64;
                for (v, e) in vars.iter().zip(&extents).rev() {
                    let n = self.index_name(*v);
                    terms.push(if place == 1 { format!("uint({n})") } else { format!("uint({n}) * {place}") });
                    place *= e;
                }
                self.line(&format!("const uint tuple = {};", terms.join(" + ")));
                self.block(&widened)?;
            } else if split_here {
                self.line("const uint tuple = item;");
                // Each part streams its slice and publishes its carried state to scratch.
                // A second launch folds the parts together with the loop body's own merge
                // rule and runs the tail, so the kernel text never mentions either.
                let at = split_state.iter().position(|(a, _)| *a == k).unwrap();
                let carried = split_state[at].1.clone();
                self.block(&body[..splits[at].loop_at + 1])?;
                let handoff = self.publish_partials(&carried, base_items * per_item, parts, "tuple")?;
                split_tails.push((k, kernel.clone(), carried, handoff, base_items, parts, body[splits[at].loop_at + 1..].to_vec(),
                ));
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
            let after_barrier = !launches.is_empty();
            launches.push(Launch { kernel, threadgroups: ((items + sg_per_tg - 1) / sg_per_tg) as u64, threads_per_threadgroup: (sg_per_tg * SUBGROUP) as u64, after_barrier,
                dispatch: Some(GroupDispatch::new(items as u64,SUBGROUP as u64,sg_per_tg as u64)?),
                tiles: std::mem::take(&mut self.tile_declarations),declared_threadgroup_bytes:shared_bytes,
            });
        }
        // A split reduction's second launch: fold the parts and run the tail.
        for (k, first, carried, handoff, _items, parts, tail) in split_tails.clone() {
            let StmtKind::Parallel { vars, extents, .. } = &body[k].kind else { unreachable!() };
            let extents: Vec<i64> = extents.iter().map(|e| e.as_constant().unwrap()).collect();
            // One item per index tuple: the merge does not itself use multiplicity.
            let items: i64 = extents.iter().product();
            let kernel = format!("{first}_merge");
            let mut kernel_out = String::new();
            std::mem::swap(&mut self.out, &mut kernel_out);
            self.indent = 1;
            let sg_per_tg = self.cfg.sg_per_tg;
            self.simdgroups = sg_per_tg;
            self.line(&format!("const uint item = tg_pos.x * {sg_per_tg} + sg_id;"));
            self.line(&format!("if (item >= {items}) return;"));
            let mut stride = items;
            for (v, e) in vars.iter().zip(&extents) {
                stride /= e;
                let name = self.index_name(*v);
                self.line(&format!("const int {name} = (item / {stride}) % {e};"));
            }
            self.merge_partials(&carried, &handoff, parts)?;
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
            launches.push(Launch { kernel, threadgroups: ((items + sg_per_tg - 1) / sg_per_tg) as u64, threads_per_threadgroup: (sg_per_tg * SUBGROUP) as u64, after_barrier: true,
                dispatch: Some(GroupDispatch::new(items as u64,SUBGROUP as u64,sg_per_tg as u64)?),
                tiles: std::mem::take(&mut self.tile_declarations),declared_threadgroup_bytes:shared_bytes,
            });
        }
        Ok(Emitted { source: header + &self.out, launches, buffers: self.buffers, scalars: self.scalars, scratch: self.scratch, status_slot: Some(status_slot), alias_pairs: Vec::new(),
        })
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
                if matches!((lo.as_constant(),hi.as_constant()),(Some(lo),Some(hi)) if lo>=hi) {
                    return Ok(());
                }
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
            StmtKind::LoadLoop { vars, views, axis, piece, capacity, body ,
            } => {
                let realized: Vec<Realization> = views.iter().map(|v| self.view_of(v)).collect::<Result<_, _>>()?;
                match capacity {
                    None => {
                        if let Some(Realization::View { shape, .. }) = realized.first() {
                            if shape.get(*axis).and_then(Sym::as_constant) == Some(0) { return Ok(()); }
                        }
                        for (v, r) in vars.iter().zip(realized) {
                            self.bind_stream_load(*v, r, body)?;
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
                        for (v, r) in vars.iter().zip(realized) {
                            let Realization::View { param, elem, offset, strides, mut shape ,
                            } = r else { unreachable!() };
                            let offset = offset.add(&Sym::param(&chunk).mul(&Sym::constant(*cap)).mul(&strides[*axis]),
                            );
                            shape[*axis] = Sym::atom(piece.clone());
                            self.bind_stream_load(*v, Realization::View { param, elem, offset, strides, shape ,
                                }, body,
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
                        let slots = (n_cap + SUBGROUP - 1) / SUBGROUP;
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
                        self.line("simdgroup_barrier(mem_flags::mem_threadgroup);");
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
                }
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
            StmtKind::Assign { target, op, value } => self.assign(target, *op, value),
            StmtKind::Expr(e) => match &e.kind {
                ExprKind::Builtin { name: Builtin::Store, args ,
                } => self.store(&args[0], &args[1]),
                ExprKind::Builtin { name: Builtin::Atomic, .. } => Err("atomic is not yet supported on Metal".into()),
                ExprKind::Intrinsic { name, args } => self.intrinsic_stmt(name, args),
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

    fn declare_tile(&mut self, v: VarId, shape: &[Sym], dtype: DType,
    ) -> Result<Realization, String> {
        let dims = self.dims(shape)?;
        let n_cap: i64 = dims.iter().map(|d| d.cap).product();
        let name = format!("{}_{}", sanitize(&self.vars()[v].name), v);
        let needs_shared = uses_intrinsic(&self.f.body, v) || (self.cross_read.contains(&v) && n_cap > SUBGROUP);
        let r = if needs_shared {
            let sg = self.simdgroups;
            let bytes = u64::try_from(n_cap.max(1)).map_err(|_|"negative tile capacity")?.checked_mul(dtype.bytes() as u64).and_then(|n|n.checked_mul(sg as u64)).ok_or("shared tile storage overflow")?;
            self.shared_decls.push((format!("threadgroup {} {name}[{sg}][{}];", ctype(dtype), n_cap.max(1)),bytes));
            Realization::Shared { name: format!("{name}[sg_id]"), dims, dtype ,
            }
        } else if n_cap <= SUBGROUP {
            self.line(&format!("{} {name}[{}];", ctype(dtype), n_cap.max(1)));
            Realization::Replicated { name, dims, dtype }
        } else {
            let slots = (n_cap + SUBGROUP - 1) / SUBGROUP;
            self.line(&format!("{} {name}[{slots}];", ctype(dtype)));
            Realization::Distributed { name, dims, dtype, slots ,
            }
        };
        self.tile_declarations.push(TileDeclaration {
            symbol: self.vars()[v].name.clone(), dtype, capacity:u64::try_from(n_cap).map_err(|_|"negative tile capacity")?,
            placement:match &r {Realization::Replicated {..}=>TilePlacement::Replicated,Realization::Distributed {..}=>TilePlacement::Distributed,Realization::Shared {..}=>TilePlacement::GroupShared,_=>unreachable!()},
        });
        self.real.insert(v, r.clone());
        Ok(r)
    }

    fn coerce(&self, text: String, from: &Ty, to: DType) -> String {
        match scalar_dtype(from) {
            Some(d) if d != to => format!("{}({text})", ctype(to)),
            _ => text,
        }
    }

    fn assign(&mut self, target: &Expr, op: AssignOp, value: &Expr) -> Result<(), String> {
        match &target.kind {
            ExprKind::Var(v) => {
                match &value.kind {
                    ExprKind::TileAlloc { shape, dtype } => {
                        let Elem::Dtype(dtype) = dtype else {return Err("unresolved or packed local tile dtype".into())};
                        self.declare_tile(*v, shape, *dtype)?;
                        return Ok(());
                    }
                    ExprKind::Intrinsic { name, args } if name == "simdgroup_matrix" => {
                        let Ty::Scalar(d) = args[0].ty else { unreachable!() };
                        self.frag_decl(*v, d);
                        return Ok(());
                    }
                    ExprKind::Builtin { name: Builtin::Reduce, args ,
                    } => return self.reduce_into(*v, args),
                    ExprKind::Builtin { name: Builtin::Load, args ,
                    } => return self.load_into(*v, &args[0]),
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
                            self.declare_tile(*v, &shaped.shape, dtype)?;
                        }
                        let destination = self.real.get(v).cloned().ok_or("unrealized destination tile")?;
                        self.copy_tile(&destination, &source, op)
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
                self.line(&format!("{lhs} {} {val};", assign_text(op)));
                Ok(())
            }
            _ => Err("unsupported assignment target".into()),
        }
    }

    fn copy_tile(&mut self, destination: &Realization, source: &Realization, op: AssignOp,
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
        if shared { self.line("simdgroup_barrier(mem_flags::mem_threadgroup);"); }
        Ok(())
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

    fn load_into(&mut self, v: VarId, view: &Expr) -> Result<(), String> {
        let realized = self.view_of(view)?;
        let Realization::View { shape, .. } = &realized else { unreachable!() };
        let n_cap: i64 = self.dims(&shape)?.iter().map(|d| d.cap).product();
        if self.cross_read.contains(&v) && !self.written.contains(&v) && n_cap > SUBGROUP && seismic_lang::effects::load_can_borrow(&self.f.body, v, "metal") {
            // Read-only tile with a store-free lifetime: its snapshot can borrow device memory.
            self.real.insert(v, realized);
            return Ok(());
        }
        self.snapshot_into(v, realized)
    }

    fn bind_stream_load(&mut self, v: VarId, realized: Realization, body: &[Stmt],
    ) -> Result<(), String> {
        if body.iter().any(|s| {
            seismic_lang::effects::tensor_effect(s, "metal") || seismic_lang::effects::tile_mutated(s, v, "metal")}) {
            self.snapshot_into(v, realized)
        } else {
            self.real.insert(v, realized);
            Ok(())
        }
    }

    fn snapshot_into(&mut self, v: VarId, realized: Realization) -> Result<(), String> {
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
        let r = self.declare_tile(v, &shape, dtype)?;
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
                if shared { self.line("simdgroup_barrier(mem_flags::mem_threadgroup);"); }
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
        Ok(())
    }

    /// Decompose a capacity-flat element counter into indices; returns the validity guard and
    /// the element offset through the view's strides.
    fn unflatten(&mut self, flat: &str, dims: &[Dim], strides: &[Sym], offset: &Sym,
    ) -> (String, Sym) {
        let n_cap: i64 = dims.iter().map(|d| d.cap).product();
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

    fn reduce_into(&mut self, v: VarId, args: &[Expr]) -> Result<(), String> {
        let previous = self.real.remove(&v);
        self.reduce_new(v, args)?;
        if let Some(destination) = previous {
            let source = self.real.get(&v).cloned().ok_or("missing reduction result")?;
            match (&destination, &source) {
                (Realization::Scalar { name: to }, Realization::Scalar { name: from }) => {
                    self.line(&format!("{to} = {from};"));
                }
                _ => self.copy_tile(&destination, &source, AssignOp::Assign)?,
            }
            if let (VarKind::Index(Atom::Param(atom)), Realization::Scalar { name }) = (&self.vars()[v].kind, &destination) {
                self.names.insert(atom.clone(), name.clone());
            }
            self.real.insert(v, destination);
        }
        Ok(())
    }

    fn reduce_new(&mut self, v: VarId, args: &[Expr]) -> Result<(), String> {
        let ExprKind::Var(tv) = args[0].kind else { return Err("reduce of a non-variable tile".into()) ;
        };
        let ExprKind::Int(axis) = args[1].kind else { unreachable!() };
        let ExprKind::Int(op) = args[2].kind else { unreachable!() };
        let axis = axis as usize;
        if op == 3 {
            return self.argmax_into(v, tv, axis);
        }
        let (opname, simd, init) = match op {
            0 => ("+", "simd_sum", "0.0f"),
            1 => ("max", "simd_max", "-INFINITY"),
            2 => ("min", "simd_min", "INFINITY"),
            _ => unreachable!(),
        };
        let mut src = self.real.get(&tv).cloned().ok_or("reduce of an unrealized tile")?;
        if matches!(src, Realization::View { .. }) {
            // A borrowed stream remains a tile value. Materialize it when this
            // reduction realization needs owned lane storage.
            self.snapshot_into(tv, src)?;
            src = self.real.get(&tv).cloned().ok_or("reduction snapshot is missing")?;
        }
        let (dims, dtype) = match &src {
            Realization::Replicated { dims, dtype, .. } | Realization::Distributed { dims, dtype, .. } | Realization::Shared { dims, dtype, .. } => (dims.clone(), *dtype),
            other => return Err(format!("reduce of {other:?}")),
        };
        let init = match (dtype,op) {
            (DType::I32,1) => "(-2147483647 - 1)", (DType::I32,2) => "2147483647",
            (DType::U32,1) => "0u", (DType::U32,2) => "0xffffffffu", _ => init,
        };
        let init=format!("{}({init})",ctype(dtype));
        let combine=|acc:&str,x:&str| -> String {
            if matches!(dtype,DType::BF16|DType::F16) {
                let inner=if opname=="+" {format!("float({acc}) + float({x})")}else{format!("{opname}(float({acc}), float({x}))")};
                format!("{}({inner})",ctype(dtype))
            } else if opname=="+" {format!("{acc} + {x}")}else{format!("{opname}({acc}, {x})")}
        };
        // Narrow collective arithmetic is not admitted yet: ordered scalar
        // publication is a legal realization even when reassociation is allowed.
        let ordered=matches!(args.get(3).map(|e|&e.kind),Some(ExprKind::Bool(true))) || matches!(dtype,DType::BF16|DType::F16);
        let mut out_dims = dims.clone();
        out_dims.remove(axis);
        let out_cap: i64 = out_dims.iter().map(|d| d.cap).product::<i64>().max(1);
        let scalar_result = out_dims.is_empty();
        let name = self.fresh("reduced");
        let inner_cap: i64 = dims[axis + 1..].iter().map(|d| d.cap).product();
        let axis_cap = dims[axis].cap;
        let axis_ext = dims[axis].ext.clone();
        if !scalar_result && out_cap > SUBGROUP {
            // Large result: only realizable when the reduced axis does not cross lanes, i.e. every
            // source element of one output lives in that output's lane.
            let Realization::Distributed { name: sn, .. } = &src else {
                return Err("large reductions of replicated or shared tiles are not yet supported on Metal".into(),
                );
            };
            if inner_cap % SUBGROUP != 0 {
                return Err(format!("reduction over axis {axis} crosses lanes (inner extent {inner_cap} is not a multiple of {SUBGROUP}); not yet supported on Metal"));
            }
            let slots = (out_cap + SUBGROUP - 1) / SUBGROUP;
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
            self.line(&format!("{} {name}[{out_cap}];", ctype(dtype)));
        }
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
            Realization::Distributed { name:sn,.. } if ordered => {
                let k=self.fresh("k");
                self.line(&format!("for (int {k}=0; {k}<{axis_ext}; ++{k}) {{"));self.indent+=1;
                self.line(&format!("const int e = (({o} / {inner_cap}) * {axis_cap} + {k}) * {inner_cap} + ({o} % {inner_cap});"));
                let value=if matches!(dtype,DType::BF16|DType::F16) {format!("{}(simd_shuffle(float({sn}[e / {SUBGROUP}]), uint(e % {SUBGROUP})))",ctype(dtype))}else{format!("simd_shuffle({sn}[e / {SUBGROUP}], uint(e % {SUBGROUP}))")};
                self.line(&format!("{acc} = {};",combine(&acc,&value)));self.indent-=1;self.line("}");
            }
            Realization::Distributed { name: sn, slots, .. } => {
                let n_cap: i64 = dims.iter().map(|d| d.cap).product();
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
    fn argmax_into(&mut self, v: VarId, tv: VarId, axis: usize) -> Result<(), String> {
        let src = self.real.get(&tv).cloned().ok_or("reduce of an unrealized tile")?;
        let (dims, dtype) = match &src {
            Realization::Replicated { dims, dtype, .. } | Realization::Distributed { dims, dtype, .. } | Realization::Shared { dims, dtype, .. } => (dims.clone(), *dtype),
            Realization::View { shape, elem, .. } => {
                let dims = shape.iter().map(|s| self.dim(s)).collect::<Result<Vec<_>, _>>()?;
                (dims, elem.read_dtype().unwrap_or(DType::F32))
            }
            other => return Err(format!("argmax of {other:?}")),
        };
        let mut out_dims = dims.clone();
        out_dims.remove(axis);
        let out_cap: i64 = out_dims.iter().map(|d| d.cap).product::<i64>().max(1);
        if out_cap > SUBGROUP {
            return Err("argmax results larger than a subgroup are not yet supported on Metal".into(),
            );
        }
        let scalar_result = out_dims.is_empty();
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
            self.line(&format!("int {name}[{out_cap}];"));
        }
        let inner_cap: i64 = dims[axis + 1..].iter().map(|d| d.cap).product();
        let axis_cap = dims[axis].cap;
        let axis_ext = dims[axis].ext.clone();
        if axis_cap < 1 || axis_cap > i64::from(i32::MAX) {return Err("Metal argmax axis exceeds its signed loop-index domain".into());}
        let initial=match dtype {DType::I32=>"(-2147483647 - 1)",DType::U32=>"0u",_=>"-INFINITY",
        };
        let o = self.fresh("o");
        let best = self.fresh("best");
        let at = self.fresh("at");
        self.line(&format!("for (int {o} = 0; {o} < {out_cap}; ++{o}) {{"));
        self.indent += 1;
        self.line(&format!("{} {best} = {}({initial}); int {at} = 0x7fffffff;", ctype(dtype), ctype(dtype)));
        match &src {
            Realization::Replicated { name: sn, .. } | Realization::Shared { name: sn, .. } => {
                let k = self.fresh("k");
                self.line(&format!("for (int {k} = 0; {k} < {axis_ext}; ++{k}) {{"));
                self.indent += 1;
                self.line(&format!("const int e = (({o} / {inner_cap}) * {axis_cap} + {k}) * {inner_cap} + ({o} % {inner_cap});"));
                self.line(&format!("if ({sn}[e] > {best} || ({sn}[e] == {best} && {k} < {at})) {{ {best} = {sn}[e]; {at} = {k}; }}"));
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
                self.line(&format!("if (e < {n_cap} && k < {axis_ext} && ((e / {inner_cap}) / {axis_cap}) * {inner_cap} + (e % {inner_cap}) == {o} && ({sn}[{j}] > {best} || ({sn}[{j}] == {best} && k < {at}))) {{ {best} = {sn}[{j}]; {at} = k; }}"));
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
                self.line(&format!("for (int {k} = int(lane); {k} < {axis_ext}; {k} += {SUBGROUP}) {{"));
                self.indent += 1;
                let x = self.fresh("x");
                self.line(&format!("const {} {x} = {val};", ctype(dtype)));
                self.line(&format!("if ({x} > {best} || ({x} == {best} && {k} < {at})) {{ {best} = {x}; {at} = {k}; }}"));
                self.indent -= 1;
                self.line("}");
                self.line(&format!("{{ const {} m = simd_max({best}); {at} = simd_min({best} == m ? {at} : 0x7fffffff); }}", ctype(dtype)));
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
    /// (item, part). Returns, per carried tile, its scratch buffer name, its index among the
    /// scratch buffers, and its element count.
    fn publish_partials(&mut self, carried: &[VarId], items: i64, parts: i64, row: &str,
    ) -> Result<Vec<(String, usize, i64)>, String> {
        let mut out = Vec::new();
        for v in carried {
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
            let buf = format!("split_{}", self.scratch.len());
            let index = self.scratch.len();
            self.scratch.push((items * parts * cap * 4) as usize);
            let i = self.fresh("i");
            self.line(&format!("for (int {i} = 0; {i} < {cap}; ++{i}) {buf}[(uint({row}) * {parts} + uint(part)) * {cap} + uint({i})] = {name}[{i}];"));
            out.push((buf, index, cap));
        }
        Ok(out)
    }

    /// Fold the parts of a split reduction, reading each part's published state and
    /// applying the streaming body's own merge rule, then leave the result in the carried
    /// tiles so the kernel's tail runs unchanged.
    fn merge_partials(&mut self, carried: &[VarId], handoff: &[(String, usize, i64)], parts: i64,
    ) -> Result<(), String> {
        if carried.len() != 3 || handoff.len() != 3 {
            return Err(format!("splitting a streamed range needs three carried tiles (maximum, denominator, accumulator); this loop carries {}", carried.len()));
        }
        // Re-declare the carried tiles in this kernel and seed them from part 0.
        let mut names = Vec::new();
        for (v, (buf, _, cap)) in carried.iter().zip(handoff) {
            let Ty::Tile(shaped) = self.vars()[*v].ty.clone() else { return Err("carried state is not a tile".into()) ;
            };
            let dtype = shaped.elem.read_dtype().unwrap_or(DType::F32);
            let r = self.declare_tile(*v, &shaped.shape, dtype)?;
            let name = match &r {
                Realization::Replicated { name, .. } | Realization::Shared { name, .. } | Realization::Distributed { name, .. } => name.clone(),
                other => return Err(format!("merged tile is {other:?}")),
            };
            let i = self.fresh("i");
            self.line(&format!("for (int {i} = 0; {i} < {cap}; ++{i}) {name}[{i}] = {buf}[(item * {parts}) * {cap} + uint({i})];"));
            names.push((name, *cap, buf.clone()));
        }
        let (m_name, m_cap, m_buf) = names[0].clone();
        let (l_name, _, l_buf) = names[1].clone();
        let (acc_name, acc_cap, acc_buf) = names[2].clone();
        let width = acc_cap / m_cap.max(1);
        let p = self.fresh("p");
        let i = self.fresh("i");
        let j = self.fresh("j");
        self.line(&format!("for (int {p} = 1; {p} < {parts}; ++{p}) {{"));
        self.indent += 1;
        self.line(&format!("const uint base = (item * {parts} + uint({p}));"));
        self.line(&format!("for (int {i} = 0; {i} < {m_cap}; ++{i}) {{"));
        self.indent += 1;
        self.line(&format!("const float om = {m_buf}[base * {m_cap} + uint({i})];"));
        self.line(&format!("const float m2 = max({m_name}[{i}], om);"));
        self.line(&format!("const float a = exp({m_name}[{i}] - m2);"));
        self.line(&format!("const float b = exp(om - m2);"));
        self.line(&format!("{l_name}[{i}] = {l_name}[{i}] * a + {l_buf}[base * {m_cap} + uint({i})] * b;"));
        self.line(&format!("for (int {j} = 0; {j} < {width}; ++{j}) {{"));
        self.indent += 1;
        self.line(&format!("const int e = {i} * {width} + {j};"));
        self.line(&format!("{acc_name}[e] = {acc_name}[e] * a + {acc_buf}[base * {acc_cap} + uint(e)] * b;"));
        self.indent -= 1;
        self.line("}");
        self.line(&format!("{m_name}[{i}] = m2;"));
        self.indent -= 1;
        self.line("}");
        self.indent -= 1;
        self.line("}");
        Ok(())
    }

    fn intrinsic_stmt(&mut self, name: &str, args: &[Expr]) -> Result<(), String> {
        match name {
            "simdgroup_load" | "simdgroup_load_t" | "simdgroup_store" => {
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
                let want_t = name == "simdgroup_load_t";
                let transpose = want_t != col_major;
                match name {
                    "simdgroup_load" | "simdgroup_load_t" => self.line(&format!("simdgroup_load({frag}, {ptr} + ({off_c}), {ld_c}, ulong2(0, 0), {transpose});")),
                    _ => {
                        if col_major {
                            return Err("simdgroup_store into a column-major view is not supported".into());
                        }
                        self.line(&format!("simdgroup_store({frag}, {ptr} + ({off_c}), {ld_c});"));
                        self.line("simdgroup_barrier(mem_flags::mem_threadgroup);");
                    }
                }
                Ok(())
            }
            "simdgroup_multiply_accumulate" => {
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
            other => Err(format!("intrinsic `{other}` is not a statement")),
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
            ExprKind::Intrinsic { name, args } => match name.as_str() {
                "simd_sum" | "simd_max" | "simd_min" => {
                    let a = self.expr(&args[0])?;
                    Ok(format!("{name}({a})"))
                }
                "simdgroup_matrix" => Err("simdgroup_matrix must be assigned to a variable".into()),
                other => Err(format!("intrinsic `{other}` in expression position")),
            },
            ExprKind::Builtin { name: Builtin::Reduce, args ,
            } => {
                if !matches!(e.ty, Ty::Scalar(_)) { return Err("tile-valued reduction needs a value binding".into()); }
                if self.extra_vars.is_empty() { self.extra_vars = self.f.vars.clone(); }
                let id = self.extra_vars.len();
                self.extra_vars.push(Var { name: "reduction_value".into(), ty: e.ty.clone(), span: e.span, kind: VarKind::Local ,
                });
                self.reduce_into(id, args)?;
                match self.real.get(&id) {
                    Some(Realization::Scalar { name }) => Ok(name.clone()),
                    _ => Err("scalar reduction did not produce a scalar".into()),
                }
            }
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

fn uses_intrinsic(stmts: &[Stmt], v: VarId) -> bool {
    fn in_expr(e: &Expr, v: VarId) -> bool {
        match &e.kind {
            ExprKind::Intrinsic { args, .. } => args.iter().any(|a| matches!(a.kind, ExprKind::Var(x) if x == v)),
            _ => false,
        }
    }
    stmts.iter().any(|s| match &s.kind {
        StmtKind::Parallel { body, .. } | StmtKind::LoadLoop { body, .. } | StmtKind::Owned { body, .. } | StmtKind::Range { body, .. } | StmtKind::Lanes { body, .. } => uses_intrinsic(body, v),
        StmtKind::If { then, els, .. } => uses_intrinsic(then, v) || uses_intrinsic(els, v),
        StmtKind::Expr(e) => in_expr(e, v),
        StmtKind::Assign { value, .. } => in_expr(value, v),
    })
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

/// Which tiles are read outside their owner, and which are written element-wise.
fn analyze_usage(f: &Lowered,
) -> (std::collections::HashSet<VarId>, std::collections::HashSet<VarId>,
) {
    use std::collections::HashSet;
    let mut cross = HashSet::new();
    let mut written = HashSet::new();
    // Stack of enclosing owned loops: (tile var, index atoms).
    fn atoms_of(vars: &[Var], ids: &[VarId]) -> Vec<String> {
        ids.iter()
            .map(|v| match &vars[*v].kind {
                VarKind::Index(Atom::Param(p)) => p.clone(),
                _ => String::new(),
            })
            .collect()
    }
    fn same_shape(vars: &[Var], a: VarId, b: VarId) -> bool {
        match (&vars[a].ty, &vars[b].ty) {
            (Ty::Tile(x), Ty::Tile(y)) => x.shape == y.shape,
            _ => false,
        }
    }
    fn visit_expr(e: &Expr, vars: &[Var], owned: &[(VarId, Vec<String>)], cross: &mut HashSet<VarId>,
    ) {
        match &e.kind {
            ExprKind::Accessor { base, .. } => {
                // Packet coordinates differ from decoded element ownership.
                // Preserve a proven read-only packed view when lowering uses them.
                if let ExprKind::Var(v)=base.kind {cross.insert(v);}
                visit_expr(base,vars,owned,cross);
            }
            ExprKind::Index { base, indices } => {
                // Ownership is expressed in the underlying tile's coordinates.
                // Analyze the same transposed read that emission will perform;
                // otherwise materialization may incorrectly assign private lanes.
                if let ExprKind::Transpose(inner) = &base.kind {
                    if indices.len() == 2 {
                        let mut read = e.clone();
                        read.kind = ExprKind::Index { base: inner.clone(),
                            indices: vec![indices[1].clone(), indices[0].clone()] };
                        visit_expr(&read, vars, owned, cross);
                        return;
                    }
                }

                if let ExprKind::Var(t) = base.kind {
                    if matches!(vars[t].ty, Ty::Tile(_)) {
                        let own = owned.last().map(|(ov, atoms)| {
                            same_shape(vars, *ov, t)
                                && indices.len() == atoms.len()
                                && indices.iter().zip(atoms).all(|(i, a)| matches!(i, Index::Point(p) if p.sym.as_ref().map(|s| *s == Sym::param(a)).unwrap_or(false)))
                        }).unwrap_or(false);
                        if !own {
                            cross.insert(t);
                        }
                    }
                }
                for i in indices {
                    match i {
                        Index::Point(p) => visit_expr(p, vars, owned, cross),
                        Index::Slice { start, end } => {
                            if let Some(x) = start { visit_expr(x, vars, owned, cross) }
                            if let Some(x) = end { visit_expr(x, vars, owned, cross) }
                        }
                    }
                }
                visit_expr(base, vars, owned, cross);
            }
            ExprKind::Transpose(x) | ExprKind::Lanes { base: x, .. } | ExprKind::Unary { expr: x, .. } | ExprKind::Cast { expr: x, .. } => visit_expr(x, vars, owned, cross),
            ExprKind::Binary { lhs, rhs, .. } => {
                visit_expr(lhs, vars, owned, cross);
                visit_expr(rhs, vars, owned, cross);
            }
            ExprKind::Builtin { args, .. } | ExprKind::Intrinsic { args, .. } | ExprKind::Call { args, .. } | ExprKind::Tuple(args) => {
                for a in args {
                    visit_expr(a, vars, owned, cross);
                }
            }
            _ => {}
        }
    }
    fn visit(stmts: &[Stmt], vars: &[Var], owned: &mut Vec<(VarId, Vec<String>)>, cross: &mut HashSet<VarId>, written: &mut HashSet<VarId>,
    ) {
        for s in stmts {
            match &s.kind {
                StmtKind::Owned { vars: ids, tile, body ,
                } => {
                    if let ExprKind::Var(t) = tile.kind {
                        owned.push((t, atoms_of(vars, ids)));
                        visit(body, vars, owned, cross, written);
                        owned.pop();
                    }
                }
                StmtKind::Parallel { body, .. } | StmtKind::LoadLoop { body, .. } | StmtKind::Range { body, .. } | StmtKind::Lanes { body, .. } => visit(body, vars, owned, cross, written),
                StmtKind::If { cond, then, els } => {
                    visit_expr(cond, vars, owned, cross);
                    visit(then, vars, owned, cross, written);
                    visit(els, vars, owned, cross, written);
                }
                StmtKind::Assign { target, value, .. } => {
                    if let ExprKind::Index { base, indices } = &target.kind {
                        if let ExprKind::Var(t) = base.kind {
                            written.insert(t);
                        }
                        for i in indices {
                            if let Index::Point(p) = i {
                                visit_expr(p, vars, owned, cross);
                            }
                        }
                    }
                    if let ExprKind::Var(t) = target.kind {
                        if matches!(vars[t].ty, Ty::Tile(_)) && !matches!(value.kind, ExprKind::TileAlloc { .. } | ExprKind::Builtin { .. }) {
                            written.insert(t);
                        }
                    }
                    visit_expr(value, vars, owned, cross);
                }
                StmtKind::Expr(e) => {
                    if let ExprKind::Intrinsic { name, args } = &e.kind {
                        if name == "simdgroup_store" {
                            if let ExprKind::Var(t) = args[1].kind {
                                written.insert(t);
                            }
                        }
                    }
                    visit_expr(e, vars, owned, cross);
                }
            }
        }
    }
    visit(&f.body, &f.vars, &mut Vec::new(), &mut cross, &mut written);
    (cross, written)
}
