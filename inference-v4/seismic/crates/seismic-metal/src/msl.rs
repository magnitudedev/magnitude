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
use seismic_lang::abi::ScalarParameter;
use seismic_lang::exec::ir::*;
use seismic_lang::exec::lowered_ir::LoweredIr;
use seismic_lang::exec::types::Ty;
use seismic_lang::repr;
use seismic_lang::sym::{Atom, Sym};
use seismic_lang::syntax::ast::{AssignOp, BinaryOp, UnaryOp};
use seismic_lang::types::{DType, Elem};
use seismic_realization::{
    BufferRole, BufferSpec,
    dispatch::{GroupDispatch, TileDeclaration, TilePlacement},
};
use std::collections::{HashMap, HashSet};

use crate::execution::{Config, Execution, SUBGROUP};
use crate::terminal::{Expression as TE, Space as TSpa, Statement as TS, Type as TT};

#[derive(Clone, Debug, PartialEq)]
pub struct Emitted {
    pub source: String,
    pub terminal: crate::terminal::Program,
    pub launches: Vec<Launch>,
    pub buffers: Vec<BufferSpec>,
    pub scalars: Vec<ScalarParameter>,
    /// Scratch the realization needs and the caller did not supply: bytes per buffer, in
    /// the order they follow the caller's buffers. Values retained across launches live
    /// here, so no kernel has to declare them.
    pub scratch: Vec<usize>,
    /// Named compiler-owned bindings, in the same order as scratch allocations.
    pub scratch_bindings: Vec<BufferSpec>,
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
    /// The kernel's buffer table: position `i` is its `[[buffer(i)]]`. A kernel declares
    /// only the invocation resources its body references, numbered densely from 0.
    pub bindings: Vec<Binding>,
}

/// One invocation resource, by its position in the entry's invocation ABI.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Binding {
    /// `Emitted::buffers[n]`: a caller-supplied parameter plane.
    Buffer(usize),
    /// `Emitted::scratch[n]`: compiler-owned storage.
    Scratch(usize),
    /// The scalar parameter block.
    Scalars,
    /// The invocation status word.
    Status,
}

/// A thread's index among all threads of its threadgroup: the lane of the outer owner of a
/// launch whose SIMD groups are inner owners.
const GROUP_LANE: &str = "tg_lane";

/// Metal admits buffer indices 0..=30 in one compute function.
pub const MAX_KERNEL_BUFFERS: usize = 31;

/// A resource a kernel may declare: its MSL declaration without the buffer attribute, the
/// identifier its body names it by, and its invocation binding.
struct Resource {
    declaration: String,
    identifier: String,
    binding: Binding,
}

/// The signature of one kernel: the resources whose identifier occurs in `body`, in
/// invocation order, then the thread coordinates. The status word is always bound: the
/// runtime reads it after every invocation.
fn kernel_signature(kernel: &str, resources: &[Resource], body: &[&str]) -> Result<(String, Vec<Binding>), String> {
    let mut identifiers = std::collections::HashSet::new();
    for text in body {
        identifiers.extend(text.split(|c: char| !(c.is_ascii_alphanumeric() || c == '_')).filter(|token| !token.is_empty()));
    }
    let mut parameters = Vec::new();
    let mut bindings = Vec::new();
    for resource in resources {
        if resource.binding == Binding::Status || identifiers.contains(resource.identifier.as_str()) {
            parameters.push(format!("{} [[buffer({})]]", resource.declaration, bindings.len()));
            bindings.push(resource.binding);
        }
    }
    if bindings.len() > MAX_KERNEL_BUFFERS {
        return Err(format!("launch `{kernel}` references {} buffers; one Metal kernel binds at most {MAX_KERNEL_BUFFERS}", bindings.len()));
    }
    parameters.extend(["uint3 tg_pos [[threadgroup_position_in_grid]]", "uint sg_id [[simdgroup_index_in_threadgroup]]", "uint lane [[thread_index_in_simdgroup]]"].map(String::from));
    // The outer owner of a launch with inner owner regions addresses every thread of its group.
    if identifiers.contains(GROUP_LANE) {
        parameters.push(format!("uint {GROUP_LANE} [[thread_index_in_threadgroup]]"));
    }
    Ok((parameters.join(",\n    "), bindings))
}

/// One axis of a tile: static capacity and a C expression for the runtime extent.
#[derive(Clone, Debug, PartialEq)]
struct Dim {
    /// Finite construction envelope, never a selected physical stride.
    cap: i64,
    physical: Sym,
    physical_value: TE,
    ext: String,
    value: TE,
    /// The extent `value` renders: the bound of an owned coordinate over this axis.
    actual: Sym,
}

impl Dim {
    fn is_static(&self) -> bool {
        self.value == self.physical_value
    }
}

/// Checked coordinates and retained axes of an indexed logical view. Address
/// construction and shape-only queries consume the same evaluated geometry.
struct ViewGeometry {
    starts: Vec<Sym>,
    axes: Vec<usize>,
    shape: Vec<Sym>,
}

#[derive(Clone, Debug, PartialEq)]
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
        slots: TE,
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

#[derive(Clone)]
struct MatrixView {
    name: String,
    space: TSpa,
    dtype: DType,
    offset: TE,
    row_stride: TE,
    column_stride: TE,
    rows: TE,
    columns: TE,
}

impl MatrixView {
    fn index(&self, row: TE, column: TE) -> TE {
        TE::binary(
            BinaryOp::Add,
            self.offset.clone(),
            TE::binary(
                BinaryOp::Add,
                TE::binary(BinaryOp::Mul, row, self.row_stride.clone(), TT::I32),
                TE::binary(
                    BinaryOp::Mul,
                    column,
                    self.column_stride.clone(),
                    TT::I32,
                ),
                TT::I32,
            ),
            TT::I32,
        )
    }

    fn read(&self, row: TE, column: TE) -> TE {
        TE::Read {
            name: self.name.clone(),
            index: Box::new(self.index(row, column).cast(TT::I64)),
            space: self.space,
            ty: self.dtype.into(),
        }
    }

    fn leading_and_transpose(&self) -> Result<(TE, bool), String> {
        if self.column_stride == TE::integer(1) {
            Ok((self.row_stride.clone(), false))
        } else if self.row_stride == TE::integer(1) {
            Ok((self.column_stride.clone(), true))
        } else {
            Err("logical matrix atoms require a unit stride along one operand axis".into())
        }
    }
}

struct Printer<'a> {
    terminal: crate::terminal::Program,
    terminal_indents: Vec<Vec<usize>>,
    expressions: HashMap<String, TE>,
    local_planes: HashMap<String, (TSpa, DType, TE)>,
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
    lane_domains: Vec<(Sym, i64)>,
    /// Lexical stack of integer atoms whose inclusive range holds by construction of the
    /// emitted code (loop indices, owned coordinates, slice starts and lengths, checked
    /// indices); `None` shadows an outer binding of an atom whose range is unknown. Only
    /// these atoms take part in the symbolic bounds proofs of `proves_index`.
    ranges: Vec<(Atom, Option<(Sym, Sym)>)>,
    /// Slices already evaluated in an enclosing scope: `(start, end, extent)` to the names
    /// of their checked start and length.
    slices: Vec<((Sym, Sym, Sym), (String, String))>,
    /// Scalar locals with exactly one assignment statement: immutable within one execution
    /// of the block that assigns them, so their name is their symbolic value.
    single_assignment: HashSet<VarId>,
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
    /// Tiles placed once per threadgroup (`TilePlacement::GroupWide`).
    group_wide: HashSet<VarId>,
    /// Piece coordinates of the inner owner of this thread's SIMD group, in binder order:
    /// the trailing work axes of a launch with inner owner regions.
    owner_coordinates: Vec<TE>,
    /// Emission is inside an inner owner region.
    inner_owner: bool,
}

/// `i32` scalar locals that exactly one statement assigns.
fn single_assignment(f: &LoweredIr) -> HashSet<VarId> {
    fn visit(stmts: &[Stmt], counts: &mut HashMap<VarId, usize>) {
        for s in stmts {
            match &s.kind {
                StmtKind::Parallel { body, .. }
                | StmtKind::LoadLoop { body, .. }
                | StmtKind::Owned { body, .. }
                | StmtKind::Range { body, .. }
                | StmtKind::Lanes { body, .. } => visit(body, counts),
                StmtKind::If { then, els, .. } => {
                    visit(then, counts);
                    visit(els, counts);
                }
                StmtKind::Assign { target, .. } => {
                    if let ExprKind::Var(v) = target.kind {
                        *counts.entry(v).or_default() += 1;
                    }
                }
                StmtKind::Expr(_) => {}
            }
        }
    }
    let mut counts = HashMap::new();
    visit(&f.body, &mut counts);
    counts
        .into_iter()
        .filter(|(v, count)| *count == 1 && matches!(f.vars[*v].kind, VarKind::Local) && matches!(f.vars[*v].ty, Ty::Scalar(DType::I32)))
        .map(|(v, _)| v)
        .collect()
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
        .terminal
        .emission
        .get_or_init(|| build_execution(execution))
        .as_ref()
        .map_err(Clone::clone)
}
fn build_execution(execution: &Execution) -> Result<Emitted, String> {
    let f = &execution.function;
    let phase_indices = execution.phases.iter()
        .map(|_| Vec::new())
        .collect::<Vec<_>>();
    seismic_lang::exec::verify::executable_phases(f, &phase_indices)?;
    let cfg = &execution.config;
    let mut emitted = Printer {
        terminal: Default::default(),
        terminal_indents: Vec::new(),
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
        lane_domains: Vec::new(),
        ranges: Vec::new(),
        slices: Vec::new(),
        single_assignment: single_assignment(f),
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
        group_wide: HashSet::new(),
        owner_coordinates: Vec::new(),
        inner_owner: false,
    }
    .emit()?;
    emitted.terminal.validate_typed()?;
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

/// Canonical native name of a typed scalar binding. Retained source predicates
/// use the same identity as ordinary scalar emission; callers never rediscover
/// parameters by scanning or parsing emitted source text.
pub(crate) fn variable_symbol(variable: &seismic_lang::exec::ir::Var, id: VarId) -> String {
    format!("{}_{}", sanitize(&variable.name), id)
}

fn scalar_dtype(t: &Ty) -> Option<DType> {
    match t {
        Ty::Scalar(d) => Some(*d),
        _ => None,
    }
}

impl Printer<'_> {
    fn target(&mut self, statement: crate::terminal::Statement) {
        let statement = statement.realized();
        if statement.is_dead() {
            return;
        }
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
        if self.terminal.launches.len() <= self.memory_launch {
            self.terminal
                .launches
                .resize_with(self.memory_launch + 1, Vec::new);
            self.terminal_indents
                .resize_with(self.memory_launch + 1, Vec::new);
        }
        self.terminal_indents[self.memory_launch].push(self.indent);
        self.terminal.launches[self.memory_launch].push(crate::terminal::Site {
            operation: self.current_operation,
            statement,
        });
    }

    fn realize_kernel(&mut self) -> Result<(), String> {
        let mut removed = self.terminal.realize_launch(self.memory_launch);
        let synchronized = crate::terminal::synchronize::coalesce(
            &mut self.terminal.launches[self.memory_launch],
        )?;
        let transferred = self
            .execution
            .transfers
            .iter()
            .any(|s| s.choice.launch == self.memory_launch && s.width != 1);
        if transferred {
            crate::terminal::transfer::apply(
                &mut self.terminal.launches[self.memory_launch],
                self.memory_launch,
                &self.execution.transfers,
            )?;
            removed |= self.terminal.realize_launch(self.memory_launch);
        }
        let selected = self
            .execution
            .traversals
            .iter()
            .any(|s| s.choice.launch == self.memory_launch && s.width != 1);
        if selected {
            crate::terminal::traversal::apply(
                &mut self.terminal.launches[self.memory_launch],
                self.memory_launch,
                &self.execution.traversals,
            )?;
            removed |= self.terminal.realize_launch(self.memory_launch);
        }
        if selected || transferred || synchronized || removed {
            let mut indent = 1usize;
            self.terminal_indents[self.memory_launch] = self.terminal.launches[self.memory_launch]
                .iter()
                .map(|site| {
                    if matches!(site.statement, TS::End | TS::Else) {
                        indent = indent.saturating_sub(1);
                    }
                    let current = indent;
                    if matches!(
                        site.statement,
                        TS::For { .. } | TS::If(_) | TS::Scope | TS::Else
                    ) {
                        indent += 1;
                    }
                    current
                })
                .collect();
        }
        self.out.clear();
        for (site, indent) in self.terminal.launches[self.memory_launch]
            .iter()
            .zip(&self.terminal_indents[self.memory_launch])
        {
            self.out.push_str(&"  ".repeat(*indent));
            self.out.push_str(&site.statement.render());
            self.out.push('\n');
        }
        Ok(())
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

    /// Whether every atom of `s` is a plain parameter whose range holds by construction.
    /// Quotient and remainder atoms are excluded: their emitted arithmetic truncates.
    fn ranged(&self, s: &Sym) -> bool {
        let ranges = self.live_ranges();
        s.atoms().iter().all(|atom| matches!(atom, Atom::Param(_)) && ranges.contains_key(atom))
    }

    /// The innermost binding of every atom, without shadowed atoms and without ranges that
    /// mention an atom that is no longer ranged.
    fn live_ranges(&self) -> std::collections::BTreeMap<Atom, (Sym, Sym)> {
        let mut innermost = std::collections::BTreeMap::new();
        for (atom, range) in &self.ranges {
            innermost.insert(atom.clone(), range.clone());
        }
        let mut live: std::collections::BTreeMap<Atom, (Sym, Sym)> = innermost.into_iter().filter_map(|(atom, range)| Some((atom, range?))).collect();
        loop {
            let stale: Vec<Atom> = live
                .iter()
                .filter(|(_, (lo, hi))| lo.atoms().iter().chain(hi.atoms().iter()).any(|a| !matches!(a, Atom::Param(_)) || !live.contains_key(a)))
                .map(|(atom, _)| atom.clone())
                .collect();
            if stale.is_empty() {
                return live;
            }
            for atom in stale {
                live.remove(&atom);
            }
        }
    }

    /// Bind `atom` to the inclusive range `[lo, hi]` until the enclosing scope is left. A
    /// range over atoms that are not themselves ranged shadows any outer binding instead.
    fn bind_range(&mut self, atom: Atom, lo: Sym, hi: Sym) {
        // The prover takes every atom as nonnegative: a range enters only with a proven
        // nonnegative lower end (`0 <= lo < lo + 1`).
        let admitted = self.ranged(&hi) && self.proves_index(&lo, &lo.add(&Sym::constant(1)));
        self.ranges.push((atom, admitted.then_some((lo, hi))));
    }

    /// Symbolic proof that `0 <= index < extent` from the ranges that hold by construction.
    /// The prover treats every atom as nonnegative, so only ranged atoms are admitted.
    fn proves_index(&self, index: &Sym, extent: &Sym) -> bool {
        let ranges = self.live_ranges();
        let ranged = |s: &Sym| s.atoms().iter().all(|atom| matches!(atom, Atom::Param(_)) && ranges.contains_key(atom));
        if !ranged(index) || !ranged(extent) {
            return false;
        }
        let mut facts = seismic_lang::sym::Facts::new();
        for (atom, (lo, hi)) in &ranges {
            facts.set_range(atom.clone(), lo.clone(), hi.clone());
        }
        let prover = seismic_lang::sym::Prover::new(&facts);
        prover.nonneg(index) && prover.lt(index, extent)
    }

    /// Owned coordinates lie in `[0, extent)` of their axis: the replicated loops run to the
    /// extent, the cooperative guard rejects every other element.
    fn bind_owned(&mut self, vars: &[VarId], dims: &[Dim]) {
        for (var, dim) in vars.iter().zip(dims) {
            if let VarKind::Index(atom) = &self.vars()[*var].kind {
                let extent = if self.ranged(&dim.actual) { dim.actual.clone() } else { Sym::constant(dim.cap) };
                self.bind_range(atom.clone(), Sym::constant(0), extent.sub(&Sym::constant(1)));
            }
        }
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
            physical: self.execution.storage.capacity_expression(s),
            physical_value: self.target_sym(&self.execution.storage.capacity_expression(s))?,
            ext,
            value: self.target_sym(s)?,
            actual: s.clone(),
        })
    }

    fn physical_count(dims: &[Dim]) -> TE {
        dims.iter().fold(TE::integer(1), |count, dim| TE::binary(BinaryOp::Mul, count, dim.physical_value.clone(), TT::I32))
    }
    fn physical_strides(dims: &[Dim]) -> Vec<Sym> {
        let mut strides = vec![Sym::constant(1); dims.len()];
        let mut stride = Sym::constant(1);
        for index in (0..dims.len()).rev() { strides[index] = stride.clone(); stride = stride.mul(&dims[index].physical); }
        strides
    }
    fn nonzero_divisor(value: TE) -> TE {
        TE::Select(Box::new(TE::binary(BinaryOp::Lt, value.clone(), TE::integer(1), TT::Bool)), Box::new(TE::integer(1)), Box::new(value))
    }
    fn physical_slots(dims: &[Dim]) -> TE {
        Self::participant_slots(dims, SUBGROUP)
    }
    /// Elements each of `lanes` participants visits under the interleaved cover.
    fn participant_slots(dims: &[Dim], lanes: i64) -> TE {
        let count = Self::physical_count(dims);
        TE::binary(BinaryOp::Add, TE::binary(BinaryOp::Div, count.clone(), TE::integer(lanes), TT::I32),
            TE::binary(BinaryOp::Ne, TE::binary(BinaryOp::Rem, count, TE::integer(lanes), TT::I32), TE::integer(0), TT::Bool).cast(TT::I32), TT::I32)
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
        header.push_str(crate::terminal::VALUE_SELECTION_SUPPORT);
        header.push_str(&crate::support::render(self.execution.support()));
        let mut index = 0usize;
        let mut resources: Vec<Resource> = Vec::new();
        for (i, (name, ty)) in self.f.params.iter().enumerate() {
            // Source ABI names are semantic identities and are not required to be
            // valid (or collision-free after sanitization) in MSL.  In particular,
            // owned results are named `$return.N`.  Native names are deliberately
            // ordinal so backend spelling can never leak into the source ABI.
            let native_name = native_parameter_name(i);
            let variable = self
                .f
                .vars
                .iter()
                .position(|v| matches!(v.kind, VarKind::Param(parameter) if parameter == i))
                .or_else(|| (i < self.execution.source().params.len()).then_some(i))
                .ok_or("parameter has no variable binding")?;
            if let Some(retained) = self.execution.retained().iter().find(|r| r.parameter == i) {
                let Ty::Tensor(shaped) = ty else {
                    return Err("retained binding must be a tensor".into());
                };
                if shaped.elem != Elem::Dtype(retained.dtype) {
                    return Err("retained binding dtype disagrees with phase plan".into());
                }
                self.real.insert(
                    variable,
                    Realization::Param {
                        name: name.clone(),
                        shape: shaped
                            .shape
                            .iter()
                            .map(|d| d.as_constant().ok_or("retained capacity must be static"))
                            .collect::<Result<_, _>>()?,
                        elem: shaped.elem.clone(),
                    },
                );
                continue;
            }
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
                            resources.push(Resource { declaration: format!("device {}* {native_name}", ctype(*d)), identifier: native_name.clone(), binding: Binding::Buffer(index) });
                            self.buffers.push(BufferSpec {
                                parameter: name.clone(),
                                plane: "".into(),
                                role: self
                                    .f
                                    .result_bindings
                                    .iter()
                                    .find(|binding| binding.parameter == i)
                                    .map_or(BufferRole::Parameter, |binding| BufferRole::Result {
                                        path: binding.path.clone(),
                                    }),
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
                                resources.push(Resource {
                                    declaration: format!("device const {}* {native_name}_{}", ctype(dtype), plane.name),
                                    identifier: format!("{native_name}_{}", plane.name),
                                    binding: Binding::Buffer(index),
                                });
                                self.buffers.push(BufferSpec {
                                    parameter: name.clone(),
                                    plane: plane.name.into(),
                                    role: self
                                        .f
                                        .result_bindings
                                        .iter()
                                        .find(|binding| binding.parameter == i)
                                        .map_or(BufferRole::Parameter, |binding| BufferRole::Result {
                                            path: binding.path.clone(),
                                        }),
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
                        variable,
                        Realization::Param {
                            name: native_name,
                            shape,
                            elem: s.elem.clone(),
                        },
                    );
                }
                Ty::Scalar(d) => {
                    self.scalars
                        .push(ScalarParameter::from_lowered(self.f, name, *d)?);
                    self.real.insert(
                        variable,
                        Realization::Scalar {
                            name: format!("sc.{name}"),
                        },
                    );
                    if let VarKind::Index(Atom::Param(atom)) = &self.vars()[variable].kind {
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
        // Every launch shares one invocation ABI: source bindings followed by
        // compiler-owned split and retained-value storage.
        let scratch_bindings: Vec<_> = self
            .execution
            .memory
            .scratch()
            .iter()
            .map(|scratch| {
                let name = scratch
                    .parameter
                    .map(|parameter| self.f.params[parameter].0.clone())
                    .unwrap_or_else(|| format!("split_{}", scratch.index));
                BufferSpec {
                    parameter: name,
                    plane: String::new(),
                    role: BufferRole::Internal,
                    bytes: scratch.bytes,
                    alignment: scratch.dtype.bytes() as usize,
                }
            })
            .collect();
        for (scratch, binding) in self
            .execution
            .memory
            .scratch()
            .iter()
            .zip(&scratch_bindings)
        {
            resources.push(Resource {
                declaration: format!("device {}* {}", ctype(scratch.dtype), binding.parameter),
                identifier: binding.parameter.clone(),
                binding: Binding::Scratch(scratch.index),
            });
        }
        index += self.execution.memory.scratch().len();
        if !self.scalars.is_empty() {
            header.push_str("struct Scalars {\n");
            for parameter in &self.scalars {
                let (n, d) = (&parameter.name, &parameter.dtype);
                header.push_str(&format!("  {} {n};\n", ctype(*d)));
            }
            header.push_str("};\n\n");
            resources.push(Resource { declaration: "constant Scalars& sc".into(), identifier: "sc".into(), binding: Binding::Scalars });
        }
        // Position of the status word in the invocation ABI (after buffers, scratch, scalars).
        let status_slot = index + usize::from(!self.scalars.is_empty());
        resources.push(Resource { declaration: "device atomic_uint* seismic_status".into(), identifier: "seismic_status".into(), binding: Binding::Status });
        let mut launches = Vec::new();
        let body = &self.f.body;
        let parameter_realizations = self.real.clone();
        let parameter_names = self.names.clone();
        let parameter_expressions = self.expressions.clone();
        for (k, stmt) in body.iter().enumerate() {
            self.real = parameter_realizations.clone();
            self.names = parameter_names.clone();
            self.expressions = parameter_expressions.clone();
            self.local_planes.clear();
            let phase = &self.execution.phases[k];
            let StmtKind::Parallel { vars, extents, body } = &stmt.kind else {
                unreachable!()
            };
            // A live work item owns one piece coordinate below each extent of the launch.
            self.ranges.clear();
            self.slices.clear();
            for (var, extent) in vars.iter().zip(extents) {
                if let VarKind::Index(atom) = &self.vars()[*var].kind {
                    self.bind_range(atom.clone(), Sym::constant(0), extent.sub(&Sym::constant(1)));
                }
            }
            let kernel = format!("{}_{k}", self.f.name);
            let mut kernel_out = String::new();
            std::mem::swap(&mut self.out, &mut kernel_out);
            self.indent = 1;
            let sg_per_tg = phase.dispatch.items_per_group as i64;
            self.simdgroups = sg_per_tg;
            self.emit_storage(&phase.dispatch)?;
            self.emit_prologue(vars, &phase.dispatch)?;
            self.block(body)?;
            self.target(TS::Return(None));
            self.realize_kernel()?;
            self.indent = 0;
            std::mem::swap(&mut self.out, &mut kernel_out);
            let declared: Vec<&str> = self.shared_decls.iter().map(|(d, _)| d.as_str()).chain([kernel_out.as_str()]).collect();
            let (signature, bindings) = kernel_signature(&kernel, &resources, &declared)?;
            self.out.push_str(&format!("kernel void {kernel}(\n    {signature}\n) {{\n"));
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
                bindings,
            });
        }
        Ok(Emitted {
            terminal: self.terminal,
            scratch_bindings,
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
        // Work axes beyond the launch binders are the inner owners of one launch piece.
        if vars.len() > program.coordinates.len() {
            return Err("parallel variables do not match launch coordinates".into());
        }
        let value = |v: V| match v {
            V::Group => E::variable("tg_pos.x", T::U32),
            V::Subgroup => E::variable("sg_id", T::U32),
            V::Constant(n) => E::Integer(i64::from(n), T::U32),
            V::Result(n) => E::variable(format!("seismic_entry_{n}"), match program.steps[n].operation {
                O::SignedIndex(_) => T::I32,
                O::Live(..) => T::Bool,
                _ => T::U32,
            }),
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
        self.owner_coordinates = program.coordinates[vars.len()..].iter().map(|coordinate| value(*coordinate).cast(T::I32)).collect();
        if !self.owner_coordinates.is_empty() {
            let threads = i64::try_from(dispatch.threads_per_group).map_err(|_| "threadgroup thread count exceeds the signed range")?;
            self.target(S::Participants { name: GROUP_LANE.into(), count: threads });
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

    /// Lane of this thread among the participants of the statement being emitted, and their
    /// count: the threads of the threadgroup for a threadgroup-wide tile of the outer owner,
    /// else the lanes of the SIMD group.
    fn participants(&self, tile: VarId) -> (TE, i64) {
        if self.group_wide.contains(&tile) && !self.inner_owner {
            (TE::variable(GROUP_LANE, TT::U32), SUBGROUP * self.simdgroups)
        } else {
            (TE::variable("lane", TT::U32), SUBGROUP)
        }
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
        // Ranges and evaluated slices are lexical: names declared in this block end with it.
        let scope = (self.ranges.len(), self.slices.len());
        for s in stmts {
            self.stmt(s)?;
        }
        self.ranges.truncate(scope.0);
        self.slices.truncate(scope.1);
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
            StmtKind::Parallel { vars, extents, body } => {
                // An inner owner region: this thread's SIMD group runs the visit whose piece
                // coordinates are the trailing work axes of the launch. Every thread of the
                // threadgroup then meets at the region's completion barrier.
                if self.inner_owner || vars.len() != self.owner_coordinates.len() {
                    return Err("an inner owner region does not match the owner axes of its launch".into());
                }
                self.target(TS::Scope);
                self.indent += 1;
                let bound = self.ranges.len();
                for ((var, extent), coordinate) in vars.iter().zip(extents).zip(self.owner_coordinates.clone()) {
                    let name = self.index_name(*var);
                    if let VarKind::Index(atom) = &self.vars()[*var].kind {
                        self.bind_range(atom.clone(), Sym::constant(0), extent.sub(&Sym::constant(1)));
                    }
                    self.target(TS::Let { name, ty: TT::I32, value: coordinate });
                }
                self.inner_owner = true;
                let emitted = self.block(body);
                self.inner_owner = false;
                self.ranges.truncate(bound);
                emitted?;
                self.indent -= 1;
                self.target(TS::End);
                self.barrier(BarrierSite {
                    operation: s.id.ok_or("inner owner region has no identity")?,
                    variable: *vars.first().ok_or("inner owner region has no binder")?,
                    purpose: BarrierPurpose::Owners,
                })
            }
            StmtKind::Range { var, lo, hi, body, .. } => {
                let name = self.index_name(*var);
                let bound = self.ranges.len();
                if let VarKind::Index(atom) = &self.vars()[*var].kind {
                    self.bind_range(atom.clone(), lo.clone(), hi.sub(&Sym::constant(1)));
                }
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
                self.ranges.truncate(bound);
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
                let lane_domain = (extent.clone(), run);
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
                self.lane_domains.push(lane_domain);
                let result = self.block(body);
                self.lane_domains.pop();
                result?;
                self.indent -= 1;
                self.target(TS::End);
                self.indent -= 1;
                self.target(crate::terminal::Statement::End);
                self.indent -= 1;
                self.target(crate::terminal::Statement::End);
                let mut writes = HashSet::new();
                for statement in body {
                    seismic_lang::exec::writes::writes(statement, &mut writes);
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
                // Evaluate sources in order before checking equality or entering
                // the loop, even if no piece or element transfer will execute.
                let realized: Vec<Realization> = vars
                    .iter()
                    .zip(views)
                    .map(|(variable, view)| {
                        if self.execution.storage.requires_data(*variable) {
                            self.view_of(view)
                        } else {
                            let (shape, _) = self.geometry_of(view)?;
                            self.geometry_snapshot(
                                &shape,
                                &view
                                    .ty
                                    .shaped()
                                    .ok_or("stream source has no geometry")?
                                    .shape,
                            )
                        }
                    })
                    .collect::<Result<_, _>>()?;
                for (view, &axis) in realized.iter().zip(axes) {
                    let shape = match view {
                        Realization::View { shape, .. } => shape.clone(),
                        Realization::Geometry { dims, .. } => self.realized_shape(dims),
                        _ => return Err("stream transfer requires view geometry".into()),
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
                            let realized = match r {
                                Realization::Geometry { mut dims, .. } => {
                                    dims[*axis] = self.dim(&Sym::atom(piece.clone()))?;
                                    let strides = Self::physical_strides(&dims);
                                    Realization::Geometry { dims, strides }
                                }
                                Realization::View {
                                    space,
                                    param,
                                    elem,
                                    offset,
                                    strides,
                                    mut shape,
                                } => {
                                    let offset = offset.add(
                                        &Sym::param(&chunk)
                                            .mul(&Sym::constant(*cap))
                                            .mul(&strides[*axis]),
                                    );
                                    shape[*axis] = Sym::atom(piece.clone());
                                    Realization::View {
                                        space,
                                        param,
                                        elem,
                                        offset,
                                        strides,
                                        shape,
                                    }
                                }
                                _ => return Err("stream transfer requires view geometry".into()),
                            };
                            self.bind_stream_load(*v, realized, *mode, operation)?;
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
                if matches!(
                    real,
                    Realization::View { .. } | Realization::Geometry { .. }
                ) {
                    return self.owned_view(
                        vars,
                        tile,
                        body,
                        s.id.ok_or("owned domain has no identity")?,
                    );
                }
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
                        let bound = self.ranges.len();
                        self.bind_owned(vars, &dims);
                        self.owned_ctx.push((tv, names.clone(), None));
                        self.block(body)?;
                        self.owned_ctx.pop();
                        self.ranges.truncate(bound);
                        for _ in &dims {
                            self.indent -= 1;
                            self.target(crate::terminal::Statement::End);
                        }
                        Ok(())
                    }
                    Realization::Shared { dims, .. } => {
                        // Threadgroup memory: elements are spread over the lanes in the same
                        // order a distributed tile uses, then a barrier publishes them. A
                        // threadgroup-wide tile of the outer owner is spread over every
                        // thread of the threadgroup.
                        let (lane, lanes) = self.participants(tv);
                        let slots = Self::participant_slots(&dims, lanes);
                        let j = self.fresh("slot");
                        let e = self.fresh("e");
                        self.target(TS::For {
                            name: j.clone(),
                            start: TE::integer(0),
                            end: slots.clone(),
                            step: 1,
                        });
                        self.indent += 1;
                        self.target(TS::Let {
                            name: e.clone(),
                            ty: TT::I32,
                            value: TE::binary(
                                BinaryOp::Add,
                                lane.cast(TT::I32),
                                TE::binary(
                                    BinaryOp::Mul,
                                    TE::integer(lanes),
                                    TE::variable(j.clone(), TT::I32),
                                    TT::I32,
                                ),
                                TT::I32,
                            ),
                        });
                        let guard = self.distributed_guard(&e, &dims, &names);
                        self.target(TS::If(guard));
                        self.indent += 1;
                        let bound = self.ranges.len();
                        self.bind_owned(vars, &dims);
                        self.owned_ctx.push((tv, names.clone(), Some(j.clone())));
                        self.block(body)?;
                        self.owned_ctx.pop();
                        self.ranges.truncate(bound);
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
                            end: slots.clone(),
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
                        let bound = self.ranges.len();
                        self.bind_owned(vars, &dims);
                        self.owned_ctx.push((tv, names.clone(), Some(j.clone())));
                        self.block(body)?;
                        self.owned_ctx.pop();
                        self.ranges.truncate(bound);
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

        let mut guard = TE::binary(
            BinaryOp::Lt,
            TE::variable(e, TT::I32),
            Self::physical_count(dims),
            TT::Bool,
        );
        for (axis, (name, d)) in names.iter().zip(dims).enumerate() {
            let stride = Self::nonzero_divisor(Self::physical_count(&dims[axis + 1..]));
            self.target(TS::Let {
                name: name.clone(),
                ty: TT::I32,
                value: TE::binary(
                    BinaryOp::Rem,
                    TE::binary(
                        BinaryOp::Div,
                        TE::variable(e, TT::I32),
                        stride,
                        TT::I32,
                    ),
                    Self::nonzero_divisor(d.physical_value.clone()),
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
                self.reshape_dimensions(args)?;
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
                        strides: Self::physical_strides(&dims),
                        shape: self.realized_shape(&dims),
                    })
                }
                Some(r @ Realization::View { .. }) => Ok(r),
                other => Err(format!(
                    "not a tensor view in {}: variable {v:?} {:?}, realization {other:?}",
                    self.f.name,
                    self.vars().get(*v)
                )),
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
                    d.physical.clone()
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
                            self.bind_range(Atom::Param(atom.clone()), length.clone(), length.clone());
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

    /// A logical tile's captured geometry owns contiguous snapshot layout.
    fn geometry_snapshot(&self, shape: &[Sym], checked: &[Sym]) -> Result<Realization, String> {
        if shape.len() != checked.len() {
            return Err("geometry snapshot rank mismatch".into());
        }
        let dims = shape
            .iter()
            .zip(checked)
            .map(|(actual, checked)| {
                Ok(Dim {
                    cap: self.cap(checked)?,
                    physical: self.execution.storage.capacity_expression(checked),
                    physical_value: self.target_sym(&self.execution.storage.capacity_expression(checked))?,
                    ext: self.sym(actual)?,
                    value: self.target_sym(actual)?,
                    actual: actual.clone(),
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        let strides = Self::physical_strides(&dims);
        Ok(Realization::Geometry { dims, strides })
    }

    // Call after realizing the source view, preserving source argument order.
    // Stored view realizations already captured these evaluations and reach no
    // reshape expression when their later metadata or elements are queried.
    fn reshape_dimensions(&mut self, args: &[Expr]) -> Result<(), String> {
        for dimension in args.iter().skip(1) {
            self.int_value(dimension)?;
        }
        Ok(())
    }

    fn reshape_geometry(
        shape: &[Sym],
        strides: &[Sym],
        target: &[Sym],
    ) -> Result<Vec<Sym>, String> {
        let row_major = |dimensions: &[Sym]| {
            let mut result = vec![Sym::constant(1); dimensions.len()];
            let mut stride = Sym::constant(1);
            for axis in (0..dimensions.len()).rev() { result[axis] = stride.clone(); stride = stride.mul(&dimensions[axis]); }
            result
        };
        let product = |dimensions: &[Sym]| dimensions.iter().fold(Sym::constant(1), |count, extent| count.mul(extent));
        if shape.len() == strides.len() && product(shape) == product(target) {
            let contiguous = row_major(shape);
            if shape.iter().zip(strides).zip(&contiguous).all(|((extent, stride), expected)|
                extent.as_constant() == Some(1) || stride == expected) {
                return Ok(row_major(target));
            }
            if shape == target { return Ok(strides.to_vec()); }
        }
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
                    let strides = Self::physical_strides(&dims);
                    Ok((self.realized_shape(&dims), strides))
                }
                _ => Err("extent requires a realized shaped value".into()),
            },
            ExprKind::TileAlloc { shape, .. } => {
                let dims = self.dims(shape)?;
                let strides = Self::physical_strides(&dims);
                Ok((self.realized_shape(&dims), strides))
            }
            ExprKind::Load { view: source, .. } => {
                let (shape, _) = self.geometry_of(source)?;
                let dimensions = self.dims(&view.ty.shaped().ok_or("load has no geometry")?.shape)?;
                Ok((shape, Self::physical_strides(&dimensions)))
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
                self.reshape_dimensions(args)?;
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
            if !seismic_lang::exec::effects::can_substitute_symbolic_value(e) {
                let value = self.target_expr(e)?;
                self.target(TS::Evaluate(value));
            }
            // Retain checked coordinate equality for participant ownership,
            // even when evaluating the expression also establishes metadata.
            return Ok(s.clone());
        }
        // A scalar assigned by one statement is its own symbolic value: equal uses name
        // equal coordinates (one evaluated slice, comparable bounds).
        if let ExprKind::Var(v) = e.kind {
            if let (true, Some(Realization::Scalar { name })) = (self.single_assignment.contains(&v), self.real.get(&v)) {
                let name = name.clone();
                self.names.insert(name.clone(), name.clone());
                return Ok(Sym::param(&name));
            }
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
        // A slice evaluated in an enclosing scope is the same checked start and length.
        let key = (start.clone(), end.clone(), extent.clone());
        if let Some((_, (safe, length))) = self.slices.iter().find(|(evaluated, _)| *evaluated == key) {
            return Ok((Sym::param(safe), Sym::param(length)));
        }
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
        // Clamped or validated (an invalid slice is empty at start zero):
        // `0 <= start`, `0 <= length` and `start + length <= extent`.
        self.bind_range(Atom::Param(safe.clone()), Sym::constant(0), extent.clone());
        self.bind_range(Atom::Param(length.clone()), Sym::constant(0), extent.sub(&Sym::param(&safe)));
        self.slices.push((key, (safe.clone(), length.clone())));
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
        if matches!(base.ty, Ty::Tensor(_)) {
            // Addressable tensor views evaluate their complete geometry before
            // point indices. Reuse it directly so reshape/slice guards are not
            // dropped or evaluated again by logical tile-coordinate mapping.
            let Realization::View {
                space,
                param,
                elem,
                mut offset,
                strides,
                shape,
            } = self.view_of(base)?
            else {
                return Err("tensor element read requires a realized view".into());
            };
            if indices.len() != shape.len() {
                return Err("tensor element coordinate rank differs".into());
            }
            for ((index, stride), extent) in indices.iter().zip(&strides).zip(&shape) {
                let Index::Point(point) = index else {
                    return Err("scalar tensor read contains a slice".into());
                };
                let point = self.int_value(point)?;
                let point = self.checked_index(&point, extent)?;
                offset = offset.add(&point.mul(stride));
            }
            // Dense and packed tensor reads keep their existing typed decoder;
            // accessor expressions are handled by target_expr before this path.
            return self.target_view_read(space, &param, &offset, &elem);
        }
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
            Some(Realization::Replicated { .. }) => false,
            Some(Realization::Shared { .. }) => true,
            Some(Realization::Geometry { .. } | Realization::View { .. }) => {
                self.execution.storage.owned_cooperative(root)
            }
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
                    physical: self.execution.storage.capacity_expression(checked),
                    physical_value: self.target_sym(&self.execution.storage.capacity_expression(checked))?,
                    ext: self.sym(actual)?,
                    value: self.target_sym(actual)?,
                    actual: actual.clone(),
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        dims.iter().try_fold(1i64, |count, dimension| count.checked_mul(dimension.cap))
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
            end: Self::physical_count(&dims),
            step: if shared { SUBGROUP } else { 1 },
        });
        self.indent += 1;
        let names = vars.iter().map(|v| self.index_name(*v)).collect::<Vec<_>>();
        let guard = self.distributed_guard(&flat, &dims, &names);
        self.target(TS::If(guard));
        self.indent += 1;
        let slot = if shared {
            let slot = self.fresh("view_slot");
            self.target(TS::Let { name: slot.clone(), ty: TT::I32,
                value: TE::binary(BinaryOp::Div, TE::variable(&flat, TT::I32), TE::integer(SUBGROUP), TT::I32) });
            Some(slot)
        } else { None };
        let bound = self.ranges.len();
        self.bind_owned(vars, &dims);
        self.owned_ctx.push((root, names, slot));
        let result = self.block(body);
        self.owned_ctx.pop();
        self.ranges.truncate(bound);
        result?;
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
            if declaration.placement.group_memory() {
                // An item-owned array holds one share per item of the group; a group-wide
                // array exists once.
                let shares = if declaration.placement == TilePlacement::GroupWide { 1 } else { dispatch.items_per_group };
                self.shared_decls.push((
                    format!(
                        "threadgroup {} {}[{}];",
                        ctype(declaration.dtype),
                        declaration.symbol,
                        (layout.shared_elements_per_item * shares).to_string()
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
        let shared = slot.placement.group_memory();
        let index = if slot.placement == TilePlacement::GroupShared {
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
                MemorySpace::Group => self.target(TS::GroupBarrier),
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
        let held = |native| seismic_realization::dispatch::local_storage_dtype(dtype, native);
        if declaration.capacity != n_cap as u64 || (declaration.dtype != held(false) && declaration.dtype != held(true)) {
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
        declaration.layout(&dispatch)?;
        self.bind_allocation(
            AllocationId {
                operation,
                variable: v,
                purpose,
            },
            &name,
        )?;
        let r = if placement.group_memory() {
            if placement == TilePlacement::GroupWide {
                self.group_wide.insert(v);
            }
            Realization::Shared { name, dims, dtype }
        } else if placement == TilePlacement::Replicated {
            Realization::Replicated { name, dims, dtype }
        } else {
            let slots = Self::physical_slots(&dims);
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
                    let snapshot = self.geometry_snapshot(&shape, checked)?;
                    let Realization::Geometry { dims, .. } = &snapshot else {
                        unreachable!()
                    };
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
                        self.real.insert(*v, snapshot);
                    }
                    return Ok(());
                }
                match &value.kind {
                    ExprKind::Intrinsic { op: name, args } if name.produces_owned_result() => {
                        if op != AssignOp::Assign {
                            return Err("logical matrix result requires plain assignment".into());
                        }
                        return self.logical_matrix_into(*v, *name, args, operation);
                    }
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
                            _ => variable_symbol(&self.vars()[*v], *v),
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
                match lhs {
                    TE::Read {
                        name,
                        index,
                        space,
                        ty,
                    } => self.target(TS::Write {
                        name,
                        index: *index,
                        space,
                        ty,
                        value: value.cast(ty),
                    }),
                    TE::Helper(crate::support::Helper::Read, mut args, ty) => {
                        args.push(value.cast(ty));
                        self.target(TS::If(TE::binary(
                            BinaryOp::Eq,
                            TE::variable("lane", TT::U32),
                            TE::Integer(0, TT::U32),
                            TT::Bool,
                        )));
                        self.indent += 1;
                        self.target(TS::Evaluate(TE::Helper(
                            crate::support::Helper::Write,
                            args,
                            TT::Bool,
                        )));
                        self.indent -= 1;
                        self.target(TS::End);
                    }
                    _ => return Err("assignment needs addressable scalar storage".into()),
                }
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
        let index = self.fresh("copy");
        let flat = self.fresh("element");
        let slots = if distributed { Self::physical_slots(dims) } else { Self::physical_count(dims) };
        self.target(TS::For {
            name: index.clone(),
            start: TE::integer(0),
            end: slots.clone(),
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
        let strides = Self::physical_strides(&dims);
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
                    let index = if self.proves_index(point, &dim.actual) {
                        self.target_sym(point)?.cast(TT::I64)
                    } else {
                        TE::Helper(
                            crate::support::Helper::Index,
                            vec![
                                self.target_sym(point)?.cast(TT::I64),
                                dim.value.cast(TT::I64),
                            ],
                            TT::I64,
                        )
                    };
                    flat = TE::binary(
                        BinaryOp::Add,
                        TE::binary(BinaryOp::Mul, flat, dim.physical_value.clone().cast(TT::I64), TT::I64),
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
                let slot = self.owned_ctx.iter().rev().find_map(|(ov, names, slot)| {
                    let same_shape = match self.real.get(ov) {
                        Some(Realization::Distributed { dims: d, .. })
                        | Some(Realization::Shared { dims: d, .. })
                        | Some(Realization::Geometry { dims: d, .. }) => d.len() == dims.len()
                            && d.iter().zip(&dims).all(|(a, b)| a.physical == b.physical),
                        Some(Realization::View { .. }) => self.vars()[*ov].ty.shaped().is_some_and(|shape|
                            shape.shape.len() == dims.len() && shape.shape.iter().zip(&dims).all(|(a, b)|
                                self.execution.storage.capacity_expression(a) == b.physical)),
                        _ => false,
                    };
                    let same_index = points.len() == names.len() && points.len() == dims.len() && points.iter().zip(names).all(
                        |(p, n)| matches!(self.atom_of(p),Some(a) if self.names.get(&a)==Some(n)),
                    );
                    (same_shape && same_index).then(|| slot.clone()).flatten()
                });
                if let Some(slot) = slot {
                    // Ownership is determined by physical strides, not by the
                    // names used to capture runtime extents. Keep logical
                    // bounds checks even when a matching lane owns the slot.
                    for (point, dim) in points.iter().zip(&dims) {
                        self.target(TS::Evaluate(TE::Helper(crate::support::Helper::Index,
                            vec![self.target_sym(point)?.cast(TT::I64), dim.value.clone().cast(TT::I64)], TT::I64)));
                    }
                    return Ok(TE::Read { name, index: Box::new(TE::variable(slot, TT::I32)),
                        space: TSpa::Private, ty: dtype.into() });
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
    fn device_elements(&self, name: &str, dtype: DType) -> Result<usize, String> {
        if let Some(slot) = self.buffers.iter().find(|b| {
            let Some(parameter) = self.f.params.iter().position(|(logical, _)| logical == &b.parameter) else {
                return false;
            };
            let native = native_parameter_name(parameter);
            if b.plane.is_empty() {
                native == name
            } else {
                format!("{native}_{}", b.plane) == name
            }
        }) {
            return Ok(slot.bytes / dtype.bytes() as usize);
        }
        self.execution
            .retained()
            .iter()
            .find(|r| r.name == name && r.dtype == dtype)
            .map(|r| r.bytes / dtype.bytes() as usize)
            .ok_or_else(|| format!("missing device storage contract for `{name}`"))
    }

    fn target_raw_read(&self, ptr: &str, index: TE, dtype: DType) -> Result<TE, String> {
        if let Some((space, actual, count)) = self.local_planes.get(ptr) {
            if *actual != dtype {
                return Err("packet plane read dtype differs from allocation".into());
            }
            let index = TE::Helper(
                crate::support::Helper::Index,
                vec![index.cast(TT::I64), count.clone().cast(TT::I64)],
                TT::I64,
            );
            return Ok(TE::Read {
                name: ptr.into(),
                index: Box::new(index),
                space: *space,
                ty: dtype.into(),
            });
        }

        let count = self.device_elements(ptr, dtype)?;
        Ok(TE::Helper(
            crate::support::Helper::Read,
            vec![
                TE::variable(ptr, TT::U64),
                index.cast(TT::I64),
                TE::Integer(count as i64, TT::U64),
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
            // An address proven inside the bound storage needs no checked read.
            Elem::Dtype(dtype) if !self.local_planes.contains_key(ptr) && self.proves_index(off, &Sym::constant(i64::try_from(self.device_elements(ptr, *dtype)?).map_err(|_| "device storage exceeds the index range")?)) => {
                Ok(TE::Read { name: ptr.into(), index: Box::new(logical), space: TSpa::Device, ty: (*dtype).into() })
            }
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
        // Valid by construction of the enclosing loops and slices: nothing to check, and the
        // coordinate keeps its symbolic relation to them for the address proof of the read.
        if self.proves_index(index, extent) {
            return Ok(index.clone());
        }
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
        // The helper yields a coordinate of the axis, or zero after recording the failure. Over
        // a statically nonempty axis either is within it.
        if extent.as_constant().is_some_and(|n| n >= 1) {
            self.bind_range(Atom::Param(name.clone()), Sym::constant(0), extent.sub(&Sym::constant(1)));
        }
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
            // Load normalization proves this occurrence's complete lifetime.
            // A subsequent checked Borrow ends the old reference binding;
            // it does not publish into the previously borrowed backing.
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
        if matches!(realized, Realization::Geometry { .. }) || mode == LoadMode::Borrow {
            self.real.insert(v, realized);
            Ok(())
        } else {
            self.snapshot_into(v, realized, operation, Purpose::Value)
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
            if declaration.capacity != part.elements || declaration.dtype != seismic_realization::dispatch::local_storage_dtype(part.plane.dtype(), false) {
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
            let physical = self.execution.storage.physical(v).and_then(|physical| physical.packets.as_ref())
                .and_then(|packet| packet.planes.get(plane)).ok_or("encoded plane has no physical extent")?;
            self.local_planes.insert(name, (actual, part.plane.dtype(), self.target_sym(&physical.elements)?));
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
            strides: self.execution.storage.physical(v).and_then(|physical| physical.packets.as_ref())
                .ok_or("encoded snapshot has no physical row layout")?.strides.clone(),
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
        if strides.last().and_then(Sym::as_constant) != Some(1) {
            return Err("encoded snapshot requires contiguous packet rows".into());
        }
        for stride in &strides[..strides.len() - 1] {
            if stride.as_constant().is_none_or(|value| value < 0 || value % group != 0) {
                return Err("encoded snapshot requires contiguous packet rows".into());
            }
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
                    physical: self.execution.storage.capacity_expression(bound),
                    physical_value: self.target_sym(&self.execution.storage.capacity_expression(bound))?,
                    ext: self.sym(extent)?,
                    value: self.target_sym(extent)?,
                    actual: extent.clone(),
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
                end: Self::physical_count(outer),
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
                    self.target_sym(&self.execution.storage.physical(v).and_then(|physical| physical.packets.as_ref())
                        .and_then(|packet| packet.planes.iter().find(|plane| plane.plane.name == part.plane.name))
                        .ok_or("encoded snapshot has no retained plane geometry")?.elements_per_row)?,
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
                    physical: self.execution.storage.capacity_expression(bound),
                    physical_value: self.target_sym(&self.execution.storage.capacity_expression(bound))?,
                    ext: self.sym(extent)?,
                    value: self.target_sym(extent)?,
                    actual: extent.clone(),
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
                    end: Self::physical_count(&dims),
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
                    end: slots.clone(),
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

        let mut guard = TE::binary(
            BinaryOp::Lt,
            TE::variable(flat, TT::I32),
            Self::physical_count(dims),
            TT::Bool,
        );
        let mut off = offset.clone();
        for (k, d) in dims.iter().enumerate() {
            let stride = Self::nonzero_divisor(Self::physical_count(&dims[k + 1..]));
            let idx = self.fresh("ix");
            self.target(TS::Let {
                name: idx.clone(),
                ty: TT::I32,
                value: TE::binary(
                    BinaryOp::Rem,
                    TE::binary(
                        BinaryOp::Div,
                        TE::variable(flat, TT::I32),
                        stride,
                        TT::I32,
                    ),
                    Self::nonzero_divisor(d.physical_value.clone()),
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
        let count = self.device_elements(&param, dtype)?;
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
                end: slots.clone(),
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
            // A threadgroup-wide tile of the outer owner is published by every thread of the
            // threadgroup; any other tile by the lanes of its SIMD group.
            let (lane, lanes) = match tile.kind {
                ExprKind::Var(v) => self.participants(v),
                _ => (TE::variable("lane", TT::U32), SUBGROUP),
            };
            self.target(TS::For {
                name: index.clone(),
                start: lane.cast(TT::I32),
                end: Self::physical_count(&dims),
                step: lanes,
            });
            self.indent += 1;
            flat = index.clone();
        }
        let (guard, off) = self.unflatten(&flat, &dims, &strides, &offset);
        self.target(TS::If(TE::binary(BinaryOp::And, guard, valid, TT::Bool)));
        self.indent += 1;
        let value = if direct {
            let mut indices = Vec::new();
            for (axis, dim) in dims.iter().enumerate() {
                let coordinate = self.fresh("store_coordinate");
                let value = if dim.cap == 0 {
                    TE::integer(0)
                } else {
                    TE::binary(
                        BinaryOp::Rem,
                        TE::binary(
                            BinaryOp::Div,
                            TE::variable(&flat, TT::I32),
                            Self::nonzero_divisor(Self::physical_count(&dims[axis + 1..])),
                            TT::I32,
                        ),
                        Self::nonzero_divisor(dim.physical_value.clone()),
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
        let Some(()) = self.reduce_new(v, args, operation)? else {
            if let Some(previous) = previous { self.real.insert(v, previous); }
            return Ok(());
        };
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
    ) -> Result<Option<()>, String> {
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
        let selected = self.execution.reductions.get(crate::reduction::Site { output: v, operation })?.clone();
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
        let contract = seismic_lang::exec::reduction::Contract::new(
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
            use seismic_lang::exec::reduction::Combination;
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
            || contract.combination() == seismic_lang::exec::reduction::Combination::SaturatingAdd;
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
        let inner_cap = Self::nonzero_divisor(Self::physical_count(&dims[axis + 1..]));
        let axis_cap = Self::nonzero_divisor(dims[axis].physical_value.clone());
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
        reduction.output_slots(algorithm)?;
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
                        slots: Self::physical_slots(&out_dims),
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
            return Ok(Some(()));
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
                                inner_cap.clone(),
                                TT::I32,
                            ),
                            axis_cap.clone(),
                            TT::I32,
                        ),
                        k,
                        TT::I32,
                    ),
                    inner_cap.clone(),
                    TT::I32,
                ),
                TE::binary(BinaryOp::Rem, output, inner_cap.clone(), TT::I32),
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
                end: Self::physical_slots(&out_dims),
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
                end: Self::physical_count(&out_dims),
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
                    Self::physical_count(&out_dims),
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
                end: slots.clone(),
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
                TE::binary(BinaryOp::Div, e.clone(), inner_cap.clone(), TT::I32),
                axis_cap.clone(),
                TT::I32,
            );
            let output = TE::binary(
                BinaryOp::Add,
                TE::binary(
                    BinaryOp::Mul,
                    TE::binary(
                        BinaryOp::Div,
                        TE::binary(BinaryOp::Div, e.clone(), inner_cap.clone(), TT::I32),
                        axis_cap.clone(),
                        TT::I32,
                    ),
                    inner_cap.clone(),
                    TT::I32,
                ),
                TE::binary(BinaryOp::Rem, e.clone(), inner_cap.clone(), TT::I32),
                TT::I32,
            );
            // A rank-1 input has one output and its element index is the axis coordinate.
            let guard = if dims.len() == 1 {
                TE::binary(
                    BinaryOp::And,
                    TE::binary(BinaryOp::Lt, e.clone(), Self::physical_count(&dims), TT::Bool),
                    TE::binary(BinaryOp::Lt, e, axis_ext, TT::Bool),
                    TT::Bool,
                )
            } else {
                TE::binary(
                    BinaryOp::And,
                    TE::binary(
                        BinaryOp::And,
                        TE::binary(
                            BinaryOp::Lt,
                            e,
                            Self::physical_count(&dims),
                            TT::Bool,
                        ),
                        TE::binary(BinaryOp::Lt, axis, axis_ext, TT::Bool),
                        TT::Bool,
                    ),
                    TE::binary(BinaryOp::Eq, output, TE::variable(&o, TT::I32), TT::Bool),
                    TT::Bool,
                )
            };
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
        Ok(Some(()))
    }

    /// `argmax` along an axis: the index of the largest value, ties to the smaller index.
    fn argmax_into(
        &mut self,
        v: VarId,
        tv: VarId,
        axis: usize,
        operation: OperationId,
    ) -> Result<Option<()>, String> {
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
        let selected = self.execution.reductions.get(crate::reduction::Site { output: v, operation })?.clone();
        let placement = match &src {
            Realization::Replicated { .. } => Some(TilePlacement::Replicated),
            Realization::Distributed { .. } => Some(TilePlacement::Distributed),
            Realization::Shared { .. } => Some(TilePlacement::GroupShared),
            Realization::View { .. } => None,
            _ => unreachable!(),
        };
        let domain = crate::reduction::ReductionDomain::argmax(&dims.iter().map(|dim| dim.cap).collect::<Vec<_>>(), axis, dtype,
            placement.clone(), selected.decision.full_lanes, SUBGROUP as u64)?;
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
            self.target(TS::Let {
                name: name.clone(),
                ty: TT::I32,
                value: TE::integer(0),
            });
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
            return Ok(Some(()));
        }
        let inner_cap = Self::nonzero_divisor(Self::physical_count(&dims[axis + 1..]));
        let axis_cap = Self::nonzero_divisor(dims[axis].physical_value.clone());
        let axis_ext = dims[axis].value.clone();
        let identity = selected.decision.contract.identity().value();
        let initial = if dtype.is_float() {
            TE::Float(identity.to_bits(), dtype.into())
        } else {
            TE::Integer(identity as i64, dtype.into())
        };
        let o = self.fresh("o");
        let best = self.fresh("best");
        let at = self.fresh("at");
        let integer = |op, a, b| TE::binary(op, a, b, TT::I32);
        let compare = |op, a, b| TE::binary(op, a, b, TT::Bool);
        let flatten = |k: TE| {
            integer(
                BinaryOp::Add,
                integer(
                    BinaryOp::Mul,
                    integer(
                        BinaryOp::Add,
                        integer(
                            BinaryOp::Mul,
                            integer(
                                BinaryOp::Div,
                                TE::variable(&o, TT::I32),
                                inner_cap.clone(),
                            ),
                            axis_cap.clone(),
                        ),
                        k,
                    ),
                    inner_cap.clone(),
                ),
                integer(
                    BinaryOp::Rem,
                    TE::variable(&o, TT::I32),
                    inner_cap.clone(),
                ),
            )
        };
        self.target(TS::For {
            name: o.clone(),
            start: TE::integer(0),
            end: Self::physical_count(&out_dims),
            step: 1,
        });
        self.indent += 1;
        self.target(TS::Let {
            name: best.clone(),
            ty: dtype.into(),
            value: initial,
        });
        self.target(TS::Let {
            name: at.clone(),
            ty: TT::I32,
            value: TE::integer(i64::from(i32::MAX)),
        });
        match &src {
            Realization::Replicated { name: sn, .. }
            | Realization::Shared { name: sn, .. }
            | Realization::Distributed { name: sn, .. }
                if algorithm == Algorithm::Ordered =>
            {
                let k = self.fresh("k");
                self.target(TS::For {
                    name: k.clone(),
                    start: TE::integer(0),
                    end: axis_ext.clone(),
                    step: 1,
                });
                self.indent += 1;
                let e = self.fresh("element");
                self.target(TS::Let {
                    name: e.clone(),
                    ty: TT::I32,
                    value: flatten(TE::variable(&k, TT::I32)),
                });
                let distributed = matches!(src, Realization::Distributed { .. });
                let index = if distributed {
                    integer(
                        BinaryOp::Div,
                        TE::variable(&e, TT::I32),
                        TE::integer(SUBGROUP),
                    )
                } else {
                    TE::variable(&e, TT::I32)
                };
                let value = TE::Read {
                    name: sn.clone(),
                    index: Box::new(index),
                    space: if matches!(src, Realization::Shared { .. }) {
                        TSpa::Threadgroup
                    } else {
                        TSpa::Private
                    },
                    ty: dtype.into(),
                };
                let value = if distributed {
                    let transport = match dtype {
                        DType::F16 | DType::BF16 => TT::F32,
                        DType::Bool => TT::U32,
                        _ => dtype.into(),
                    };
                    TE::Builtin(
                        "simd_shuffle".into(),
                        vec![
                            value.cast(transport),
                            integer(
                                BinaryOp::Rem,
                                TE::variable(&e, TT::I32),
                                TE::integer(SUBGROUP),
                            )
                            .cast(TT::U32),
                        ],
                        transport,
                    )
                    .cast(dtype.into())
                } else {
                    value
                };
                self.argmax_candidate(&best, &at, value, TE::variable(&k, TT::I32), dtype);
                self.indent -= 1;
                self.target(TS::End);
            }
            Realization::Distributed {
                name: sn, slots, ..
            } => {
                let j = self.fresh("slot");
                self.target(TS::For {
                    name: j.clone(),
                    start: TE::integer(0),
                    end: slots.clone(),
                    step: 1,
                });
                self.indent += 1;
                let e = self.fresh("element");
                let k = self.fresh("k");
                self.target(TS::Let {
                    name: e.clone(),
                    ty: TT::I32,
                    value: integer(
                        BinaryOp::Add,
                        TE::variable("lane", TT::U32).cast(TT::I32),
                        integer(
                            BinaryOp::Mul,
                            TE::integer(SUBGROUP),
                            TE::variable(&j, TT::I32),
                        ),
                    ),
                });
                self.target(TS::Let {
                    name: k.clone(),
                    ty: TT::I32,
                    value: integer(
                        BinaryOp::Rem,
                        integer(
                            BinaryOp::Div,
                            TE::variable(&e, TT::I32),
                            inner_cap.clone(),
                        ),
                        axis_cap.clone(),
                    ),
                });
                let output = integer(
                    BinaryOp::Add,
                    integer(
                        BinaryOp::Mul,
                        integer(
                            BinaryOp::Div,
                            integer(
                                BinaryOp::Div,
                                TE::variable(&e, TT::I32),
                                inner_cap.clone(),
                            ),
                            axis_cap.clone(),
                        ),
                        inner_cap.clone(),
                    ),
                    integer(
                        BinaryOp::Rem,
                        TE::variable(&e, TT::I32),
                        inner_cap.clone(),
                    ),
                );
                let valid = Self::and(
                    compare(BinaryOp::Lt, TE::variable(&e, TT::I32), Self::physical_count(&dims)),
                    Self::and(
                        compare(BinaryOp::Lt, TE::variable(&k, TT::I32), axis_ext.clone()),
                        compare(BinaryOp::Eq, output, TE::variable(&o, TT::I32)),
                    ),
                );
                self.target(TS::If(valid));
                self.indent += 1;
                self.argmax_candidate(
                    &best,
                    &at,
                    TE::Read {
                        name: sn.clone(),
                        index: Box::new(TE::variable(&j, TT::I32)),
                        space: TSpa::Private,
                        ty: dtype.into(),
                    },
                    TE::variable(&k, TT::I32),
                    dtype,
                );
                self.indent -= 1;
                self.target(TS::End);
                self.indent -= 1;
                self.target(TS::End);
                self.argmax_collective(&best, &at, dtype);
            }
            Realization::View { .. } => {
                let k = self.fresh("k");
                self.names.insert(k.clone(), k.clone());
                let mut points = Vec::new();
                for (d, dim) in dims.iter().enumerate() {
                    if d == axis {
                        points.push(Sym::param(&k));
                    } else {
                        let outer = dims.iter().enumerate().skip(d + 1).filter(|(index, _)| *index != axis)
                            .fold(TE::integer(1), |count, (_, dim)| integer(BinaryOp::Mul, count, dim.physical_value.clone()));
                        let c = self.fresh("c");
                        self.names.insert(c.clone(), c.clone());
                        self.target(TS::Let {
                            name: c.clone(),
                            ty: TT::I32,
                            value: integer(
                                BinaryOp::Rem,
                                integer(
                                    BinaryOp::Div,
                                    TE::variable(&o, TT::I32),
                                    Self::nonzero_divisor(outer),
                                ),
                                Self::nonzero_divisor(dim.physical_value.clone()),
                            ),
                        });
                        points.push(Sym::param(&c));
                    }
                }
                let collective = algorithm == Algorithm::Collective;
                self.target(TS::For {
                    name: k.clone(),
                    start: if collective {
                        TE::variable("lane", TT::U32).cast(TT::I32)
                    } else {
                        TE::integer(0)
                    },
                    end: axis_ext.clone(),
                    step: if collective { SUBGROUP } else { 1 },
                });
                self.indent += 1;
                let (space, ptr, off, elem) = self.view_element(tv, &points)?;
                let value = self
                    .target_view_read(space, &ptr, &off, &elem)?
                    .cast(dtype.into());
                self.argmax_candidate(&best, &at, value, TE::variable(&k, TT::I32), dtype);
                self.indent -= 1;
                self.target(TS::End);
                if collective {
                    self.argmax_collective(&best, &at, dtype);
                }
            }
            _ => return Err("argmax algorithm disagrees with input placement".into()),
        }
        // All-NaN and all-negative-infinity rows preserve the reference's index zero.
        self.target(TS::Assign {
            name: at.clone(),
            value: TE::Select(
                Box::new(compare(
                    BinaryOp::Eq,
                    TE::variable(&at, TT::I32),
                    TE::integer(i64::from(i32::MAX)),
                )),
                Box::new(TE::integer(0)),
                Box::new(TE::variable(&at, TT::I32)),
            ),
        });
        if scalar_result {
            self.target(TS::Assign {
                name,
                value: TE::variable(at, TT::I32),
            });
        } else {
            self.target(TS::Write {
                name,
                index: TE::variable(o, TT::I32),
                space: TSpa::Private,
                ty: TT::I32,
                value: TE::variable(at, TT::I32),
            });
        }
        self.indent -= 1;
        self.target(TS::End);
        Ok(Some(()))
    }

    fn and(left: TE, right: TE) -> TE {
        TE::ShortCircuit {
            or: false,
            left: Box::new(left),
            right: Box::new(right),
        }
    }

    fn argmax_candidate(&mut self, best: &str, at: &str, value: TE, index: TE, dtype: DType) {
        let x = self.fresh("candidate");
        self.target(TS::Let {
            name: x.clone(),
            ty: dtype.into(),
            value,
        });
        let value = TE::variable(&x, dtype.into());
        let compare = |op, a, b| TE::binary(op, a, b, TT::Bool);
        let earlier = compare(BinaryOp::Lt, index.clone(), TE::variable(at, TT::I32));
        let earlier = if dtype.is_float() {
            Self::and(
                compare(
                    BinaryOp::Ne,
                    TE::variable(at, TT::I32),
                    TE::integer(i64::from(i32::MAX)),
                ),
                earlier,
            )
        } else {
            earlier
        };
        let condition = TE::ShortCircuit {
            or: true,
            left: Box::new(compare(
                BinaryOp::Gt,
                value.clone(),
                TE::variable(best, dtype.into()),
            )),
            right: Box::new(Self::and(
                compare(
                    BinaryOp::Eq,
                    value.clone(),
                    TE::variable(best, dtype.into()),
                ),
                earlier,
            )),
        };
        self.target(TS::If(condition));
        self.indent += 1;
        self.target(TS::Assign {
            name: best.into(),
            value,
        });
        self.target(TS::Assign {
            name: at.into(),
            value: index,
        });
        self.indent -= 1;
        self.target(TS::End);
    }

    fn argmax_collective(&mut self, best: &str, at: &str, dtype: DType) {
        let maximum = self.fresh("maximum");
        let transport = match dtype {
            DType::F16 | DType::BF16 => TT::F32,
            DType::Bool => TT::U32,
            _ => dtype.into(),
        };
        self.target(TS::Let {
            name: maximum.clone(),
            ty: dtype.into(),
            value: TE::Builtin(
                "simd_max".into(),
                vec![TE::variable(best, dtype.into()).cast(transport)],
                transport,
            )
            .cast(dtype.into()),
        });
        self.target(TS::Assign {
            name: at.into(),
            value: TE::Builtin(
                "simd_min".into(),
                vec![TE::Select(
                    Box::new(TE::binary(
                        BinaryOp::Eq,
                        TE::variable(best, dtype.into()),
                        TE::variable(maximum, dtype.into()),
                        TT::Bool,
                    )),
                    Box::new(TE::variable(at, TT::I32)),
                    Box::new(TE::integer(i64::from(i32::MAX))),
                )],
                TT::I32,
            ),
        });
    }

    fn matrix_view(&mut self, operand: &Expr) -> Result<MatrixView, String> {
        let Realization::View {
            space,
            param,
            elem,
            offset,
            strides,
            shape,
        } = self.view_of(operand)?
        else {
            return Err("logical matrix operand has no addressable view".into());
        };
        if !matches!(space, TSpa::Device | TSpa::Threadgroup)
            || strides.len() != 2
            || shape.len() != 2
        {
            return Err("logical matrix operand must be a rank-two device or threadgroup view".into());
        }
        let Elem::Dtype(dtype) = elem else {
            return Err("logical matrix operand must have dense elements".into());
        };
        Ok(MatrixView {
            name: param,
            space,
            dtype,
            offset: self.target_sym(&offset)?.cast(TT::I32),
            row_stride: self.target_sym(&strides[0])?.cast(TT::I32),
            column_stride: self.target_sym(&strides[1])?.cast(TT::I32),
            rows: self.target_sym(&shape[0])?.cast(TT::I32),
            columns: self.target_sym(&shape[1])?.cast(TT::I32),
        })
    }

    /// Realize one logical product with an explicit numerical order: full interior 8x8x8
    /// atoms use native SIMD-group MMA in ascending K-block order, publishing each accumulator
    /// block at its logical dtype; K and output-edge tails then continue with ascending scalar
    /// `fma` operations. This is a backend-intrinsic numerical effect, not exact equivalence to
    /// the interpreter's all-scalar ascending-K reference.
    fn logical_matrix_into(
        &mut self,
        destination: VarId,
        operation: seismic_lang::intrinsics::Operation,
        args: &[Expr],
        site: OperationId,
    ) -> Result<(), String> {
        let CollectiveImplementation::LogicalMatrix {
            operation: planned,
            destination: planned_destination,
            plan,
        } = self.collective(operation)?
        else {
            return Err("logical matrix collective implementation mismatch".into());
        };
        if planned != operation || planned_destination != destination {
            return Err("logical matrix destination differs from its prepared implementation".into());
        }
        let Ty::Tile(result) = &self.vars()[destination].ty else {
            return Err("logical matrix destination is not an owned tile".into());
        };
        let Elem::Dtype(dtype) = result.elem else {
            return Err("logical matrix destination has unresolved elements".into());
        };
        let result_shape = result.shape.clone();
        if dtype != plan.accumulation_dtype {
            return Err("logical matrix destination dtype differs from its plan".into());
        }
        let realized = self.declare_tile(
            destination,
            &result_shape,
            dtype,
            site,
            Purpose::Value,
        )?;
        let Realization::Shared { name, dims, .. } = realized else {
            return Err("logical matrix destination must use compiler-owned threadgroup storage".into());
        };
        if dims.len() != 2 {
            return Err("logical matrix destination storage is not rank two".into());
        }
        let output = MatrixView {
            name,
            space: TSpa::Threadgroup,
            dtype,
            offset: TE::integer(0),
            row_stride: dims[1].physical_value.clone().cast(TT::I32),
            column_stride: TE::integer(1),
            rows: dims[0].value.clone().cast(TT::I32),
            columns: dims[1].value.clone().cast(TT::I32),
        };
        let left = self.matrix_view(&args[0])?;
        let right = self.matrix_view(&args[1])?;
        let accumulator = if operation == seismic_lang::intrinsics::Operation::MatrixMatmulAdd {
            Some(self.matrix_view(&args[2])?)
        } else {
            None
        };
        if left.dtype != plan.left_dtype
            || right.dtype != plan.right_dtype
            || accumulator.as_ref().is_some_and(|view| view.dtype != dtype)
        {
            return Err("logical matrix operand dtypes differ from the selected plan".into());
        }

        let rows = output.rows.clone();
        let columns = output.columns.clone();
        let inner = left.columns.clone();
        let total = TE::binary(BinaryOp::Mul, rows.clone(), columns.clone(), TT::I32);
        let flat = self.fresh("matrix_element");
        let row = self.fresh("matrix_row");
        let column = self.fresh("matrix_column");
        self.target(TS::For {
            name: flat.clone(),
            start: TE::variable(GROUP_LANE, TT::U32).cast(TT::I32),
            end: total.clone(),
            step: SUBGROUP * self.simdgroups,
        });
        self.indent += 1;
        self.target(TS::Let {
            name: row.clone(),
            ty: TT::I32,
            value: TE::binary(
                BinaryOp::Div,
                TE::variable(&flat, TT::I32),
                columns.clone(),
                TT::I32,
            ),
        });
        self.target(TS::Let {
            name: column.clone(),
            ty: TT::I32,
            value: TE::binary(
                BinaryOp::Rem,
                TE::variable(&flat, TT::I32),
                columns.clone(),
                TT::I32,
            ),
        });
        let initial = accumulator.as_ref().map_or_else(
            || TE::Float(0f64.to_bits(), dtype.into()),
            |view| {
                view.read(
                    TE::variable(&row, TT::I32),
                    TE::variable(&column, TT::I32),
                )
                .cast(dtype.into())
            },
        );
        self.target(TS::Write {
            name: output.name.clone(),
            index: output.index(
                TE::variable(&row, TT::I32),
                TE::variable(&column, TT::I32),
            ),
            space: TSpa::Threadgroup,
            ty: dtype.into(),
            value: initial,
        });
        self.indent -= 1;
        self.target(TS::End);
        self.barrier(BarrierSite {
            operation: site,
            variable: destination,
            purpose: BarrierPurpose::MatrixInitialize,
        })?;

        let eight = TE::integer(8);
        let full = |extent: TE| {
            TE::binary(
                BinaryOp::Mul,
                TE::binary(BinaryOp::Div, extent, eight.clone(), TT::I32),
                eight.clone(),
                TT::I32,
            )
        };
        let full_rows = full(rows.clone());
        let full_columns = full(columns.clone());
        let full_inner = full(inner.clone());
        let block = self.fresh("matrix_block");
        let block_row = self.fresh("matrix_block_row");
        let block_column = self.fresh("matrix_block_column");
        let block_inner = self.fresh("matrix_block_inner");
        let column_blocks = TE::binary(
            BinaryOp::Div,
            full_columns.clone(),
            eight.clone(),
            TT::I32,
        );
        // Keep the emitted body well-formed even when there are no complete column atoms. The
        // trip count remains zero, but native compilers may still diagnose a literal division by
        // zero in an unreachable loop body.
        let column_block_divisor = TE::Builtin(
            "max".into(),
            vec![column_blocks.clone(), TE::integer(1)],
            TT::I32,
        );
        let block_count = TE::binary(
            BinaryOp::Mul,
            TE::binary(
                BinaryOp::Div,
                full_rows.clone(),
                eight.clone(),
                TT::I32,
            ),
            column_blocks.clone(),
            TT::I32,
        );
        self.target(TS::For {
            name: block.clone(),
            start: TE::variable("sg_id", TT::U32).cast(TT::I32),
            end: block_count,
            step: self.simdgroups,
        });
        self.indent += 1;
        self.target(TS::Let {
            name: block_row.clone(),
            ty: TT::I32,
            value: TE::binary(
                BinaryOp::Mul,
                TE::binary(
                    BinaryOp::Div,
                    TE::variable(&block, TT::I32),
                    column_block_divisor.clone(),
                    TT::I32,
                ),
                eight.clone(),
                TT::I32,
            ),
        });
        self.target(TS::Let {
            name: block_column.clone(),
            ty: TT::I32,
            value: TE::binary(
                BinaryOp::Mul,
                TE::binary(
                    BinaryOp::Rem,
                    TE::variable(&block, TT::I32),
                    column_block_divisor,
                    TT::I32,
                ),
                eight,
                TT::I32,
            ),
        });
        let accumulator_fragment = self.fresh("matrix_accumulator");
        let left_fragment = self.fresh("matrix_left");
        let right_fragment = self.fresh("matrix_right");
        let accumulator_layout = crate::collective::FragmentLayout::metal(dtype)?;
        let left_layout = crate::collective::FragmentLayout::metal(left.dtype)?;
        let right_layout = crate::collective::FragmentLayout::metal(right.dtype)?;
        self.target(TS::Fragment {
            name: accumulator_fragment.clone(),
            layout: accumulator_layout,
        });
        self.target(TS::Fragment {
            name: left_fragment.clone(),
            layout: left_layout,
        });
        self.target(TS::Fragment {
            name: right_fragment.clone(),
            layout: right_layout,
        });
        self.target(TS::MatrixLoad {
            fragment: accumulator_fragment.clone(),
            layout: accumulator_layout,
            base: output.name.clone(),
            offset: output.index(
                TE::variable(&block_row, TT::I32),
                TE::variable(&block_column, TT::I32),
            ),
            leading: output.row_stride.clone(),
            space: TSpa::Threadgroup,
            transpose: false,
        });
        self.target(TS::For {
            name: block_inner.clone(),
            start: TE::integer(0),
            end: full_inner.clone(),
            step: 8,
        });
        self.indent += 1;
        let (left_leading, left_transpose) = left.leading_and_transpose()?;
        let (right_leading, right_transpose) = right.leading_and_transpose()?;
        self.target(TS::MatrixLoad {
            fragment: left_fragment.clone(),
            layout: left_layout,
            base: left.name.clone(),
            offset: left.index(
                TE::variable(&block_row, TT::I32),
                TE::variable(&block_inner, TT::I32),
            ),
            leading: left_leading,
            space: left.space,
            transpose: left_transpose,
        });
        self.target(TS::MatrixLoad {
            fragment: right_fragment.clone(),
            layout: right_layout,
            base: right.name.clone(),
            offset: right.index(
                TE::variable(&block_inner, TT::I32),
                TE::variable(&block_column, TT::I32),
            ),
            leading: right_leading,
            space: right.space,
            transpose: right_transpose,
        });
        self.target(TS::MatrixMultiplyAccumulate {
            fragments: [
                accumulator_fragment.clone(),
                left_fragment,
                right_fragment,
                accumulator_fragment.clone(),
            ],
            layouts: [
                accumulator_layout,
                left_layout,
                right_layout,
                accumulator_layout,
            ],
        });
        self.indent -= 1;
        self.target(TS::End);
        self.target(TS::MatrixStore {
            fragment: accumulator_fragment,
            layout: accumulator_layout,
            base: output.name.clone(),
            offset: output.index(
                TE::variable(&block_row, TT::I32),
                TE::variable(&block_column, TT::I32),
            ),
            leading: output.row_stride.clone(),
            space: TSpa::Threadgroup,
        });
        self.indent -= 1;
        self.target(TS::End);
        self.barrier(BarrierSite {
            operation: site,
            variable: destination,
            purpose: BarrierPurpose::MatrixAtoms,
        })?;

        let tail_flat = self.fresh("matrix_tail_element");
        let tail_row = self.fresh("matrix_tail_row");
        let tail_column = self.fresh("matrix_tail_column");
        let tail_inner = self.fresh("matrix_tail_inner");
        let sum = self.fresh("matrix_tail_sum");
        self.target(TS::For {
            name: tail_flat.clone(),
            start: TE::variable(GROUP_LANE, TT::U32).cast(TT::I32),
            end: total,
            step: SUBGROUP * self.simdgroups,
        });
        self.indent += 1;
        self.target(TS::Let {
            name: tail_row.clone(),
            ty: TT::I32,
            value: TE::binary(
                BinaryOp::Div,
                TE::variable(&tail_flat, TT::I32),
                columns.clone(),
                TT::I32,
            ),
        });
        self.target(TS::Let {
            name: tail_column.clone(),
            ty: TT::I32,
            value: TE::binary(
                BinaryOp::Rem,
                TE::variable(&tail_flat, TT::I32),
                columns,
                TT::I32,
            ),
        });
        self.target(TS::Let {
            name: sum.clone(),
            ty: dtype.into(),
            value: output.read(
                TE::variable(&tail_row, TT::I32),
                TE::variable(&tail_column, TT::I32),
            ),
        });
        let interior = TE::binary(
            BinaryOp::And,
            TE::binary(
                BinaryOp::Lt,
                TE::variable(&tail_row, TT::I32),
                full_rows,
                TT::Bool,
            ),
            TE::binary(
                BinaryOp::Lt,
                TE::variable(&tail_column, TT::I32),
                full_columns,
                TT::Bool,
            ),
            TT::Bool,
        );
        self.target(TS::For {
            name: tail_inner.clone(),
            start: TE::Select(
                Box::new(interior),
                Box::new(full_inner),
                Box::new(TE::integer(0)),
            ),
            end: inner,
            step: 1,
        });
        self.indent += 1;
        let product = TE::Builtin(
            "fma".into(),
            vec![
                left.read(
                    TE::variable(&tail_row, TT::I32),
                    TE::variable(&tail_inner, TT::I32),
                )
                .cast(dtype.into()),
                right
                    .read(
                        TE::variable(&tail_inner, TT::I32),
                        TE::variable(&tail_column, TT::I32),
                    )
                    .cast(dtype.into()),
                TE::variable(&sum, dtype.into()),
            ],
            dtype.into(),
        );
        self.target(TS::Assign {
            name: sum.clone(),
            value: product,
        });
        self.indent -= 1;
        self.target(TS::End);
        self.target(TS::Write {
            name: output.name.clone(),
            index: output.index(
                TE::variable(&tail_row, TT::I32),
                TE::variable(&tail_column, TT::I32),
            ),
            space: TSpa::Threadgroup,
            ty: dtype.into(),
            value: TE::variable(sum, dtype.into()),
        });
        self.indent -= 1;
        self.target(TS::End);
        self.barrier(BarrierSite {
            operation: site,
            variable: destination,
            purpose: BarrierPurpose::IntrinsicStore,
        })
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
                        let ld = dims[1].physical.clone();
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
                        if *name == seismic_lang::intrinsics::Operation::MatrixStore && space != TSpa::Threadgroup {
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
                        let destination = crate::storage::tile_root(&args[1]).ok_or("fragment store has no tile binding")?;
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
            seismic_lang::intrinsics::Operation::MatrixMatmul
            | seismic_lang::intrinsics::Operation::MatrixMatmulAdd => Err(
                "logical matrix intrinsic reached Metal source emission without a realization"
                    .into(),
            ),
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

    fn target_expr(&mut self, e: &Expr) -> Result<crate::terminal::Expression, String> {
        use crate::terminal::{Expression as E, Type as T};
        let ty = scalar_dtype(&e.ty).map(T::from).unwrap_or(T::I32);
        Ok(match &e.kind {
            ExprKind::Int(n) => E::Integer(*n, ty),
            ExprKind::Float(n) => E::Float(n.to_bits(), ty),
            ExprKind::Bool(v) => E::Integer(i64::from(*v), T::Bool),
            ExprKind::Var(v) => match self.real.get(v) {
                Some(Realization::Scalar { name }) if *v < self.execution.source().params.len() => {
                    E::Parameter {
                        name: name.clone(),
                        ty,
                    }
                }
                Some(Realization::Scalar { name }) | Some(Realization::Index { name }) => {
                    E::variable(name.clone(), ty)
                }
                None if e.sym.is_some() => self.target_sym(e.sym.as_ref().unwrap())?.cast(ty),
                other => {
                    return Err(format!(
                        "compiler error: scalar `{}` has no typed Metal realization: {other:?}",
                        self.vars()[*v].name
                    ));
                }
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
            ExprKind::Builtin { name: Builtin::Select, args } => {
                let [condition, yes, no] = args.as_slice() else { return Err("eager value selection arity".into()); };
                let condition = self.target_expr(condition)?;
                let yes = self.target_expr(yes)?;
                let no = self.target_expr(no)?;
                if condition.ty() != TT::Bool || yes.ty() != no.ty() || yes.ty() != ty {
                    return Err("eager value selection type mismatch".into());
                }
                E::EagerSelect(Box::new(condition), Box::new(yes), Box::new(no))
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
                        .map(|a| self.target_expr(a).map(|value| value.cast(ty)))
                        .collect::<Result<_, _>>()?,
                    ty,
                )
            }
            other => {
                return Err(format!(
                    "compiler error: expression has no typed Metal realization: {other:?}"
                ));
            }
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
        if !seismic_lang::exec::effects::can_substitute_symbolic_value(&args[1]) {
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

fn native_parameter_name(ordinal: usize) -> String {
    format!("seismic_param_{ordinal}")
}

fn row_major_syms(shape: &[i64]) -> Vec<Sym> {
    let mut strides = vec![1i64; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides.into_iter().map(Sym::constant).collect()
}
