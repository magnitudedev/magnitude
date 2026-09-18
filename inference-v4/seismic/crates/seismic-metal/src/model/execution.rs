//! Conditional timing of the retained MSL statement implementation. This is a
//! pooled-device, source-order machine; it is not a claim about native scheduling.
use crate::terminal::{Expression, Primitive, Statement, Type};
use seismic_accounting::{
    authority::ModelRelationship,
    schedule::{self, Model, Operation, Reservation, Resource, Timebase},
    workload::{DerivationError, DerivationLimit, DerivationLimits, ScalarWorkload},
};
use seismic_lang::{
    abi::ScalarLayout,
    ast::{BinaryOp, UnaryOp},
};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Units {
    PerLane(u64),
    PerSubgroup(u64),
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Service {
    pub resource: usize,
    pub offset: u64,
    pub duration: u64,
    pub units: Units,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Timing {
    pub primitive: Primitive,
    pub latency: u64,
    pub services: Vec<Service>,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Hardware {
    pub identity: String,
    pub timebase: Timebase,
    pub resources: Vec<Resource>,
    /// Explicit pooled resident capacities; no device-name occupancy guesses.
    pub resident_groups: u64,
    pub resident_shared_bytes: u64,
    pub timings: Vec<Timing>,
}
impl Hardware {
    pub fn validate(&self) -> Result<(), String> {
        if self.identity.is_empty()
            || self.resident_groups == 0
            || self.timebase.seconds_numerator == 0
            || self.timebase.seconds_denominator == 0
        {
            return Err(
                "Metal model needs an identity, positive resident capacity and timebase".into(),
            );
        }
        let mut names = BTreeSet::new();
        for r in &self.resources {
            if r.capacity == 0 || r.name.is_empty() || !names.insert(&r.name) {
                return Err(
                    "Metal model resources require unique names and positive capacities".into(),
                );
            }
        }
        for (i, t) in self.timings.iter().enumerate() {
            if self.timings[..i].iter().any(|p| p.primitive == t.primitive) {
                return Err("duplicate Metal primitive timing".into());
            }
            for s in &t.services {
                let units = match s.units {
                    Units::PerLane(n) | Units::PerSubgroup(n) => n,
                };
                if s.resource >= self.resources.len()
                    || s.duration == 0
                    || units == 0
                    || s.offset
                        .checked_add(s.duration)
                        .is_none_or(|end| end > t.latency)
                {
                    return Err("invalid Metal primitive service interval".into());
                }
            }
            if t.latency > 0 && t.services.is_empty() {
                return Err("positive Metal primitive timing must declare its service".into());
            }
        }
        Ok(())
    }
}
pub fn execution(
    execution: &crate::execution::Execution,
    hardware: &Hardware,
    workload: &ScalarWorkload,
    limits: DerivationLimits,
) -> Result<Model, DerivationError> {
    hardware.validate()?;
    let emitted = crate::msl::prepare_execution(execution)?;
    validate_workload(&emitted, workload)?;
    let mut state = Derivation {
        hardware,
        limits,
        visits: 0,
        model: Model {
            relationship: ModelRelationship::hypothetical_execution(),
            identity: format!(
                "{}:{}:{}",
                execution.function().name,
                workload.identity,
                hardware.identity
            ),
            timebase: hardware.timebase.clone(),
            resources: hardware.resources.clone(),
            operations: Vec::new(),
            lifetimes: Vec::new(),
            static_orders: Vec::new(),
            unmapped: Vec::new(),
        },
        last: None,
        scope: String::new(),
        env: BTreeMap::new(),
        active: u32::MAX,
        alive: u32::MAX,
        returned: [None; 32],
    };
    let groups_resource = state.model.resources.len();
    state.model.resources.push(Resource {
        name: "Metal resident threadgroups".into(),
        capacity: hardware.resident_groups,
        unit: schedule::CapacityUnit::Slots,
    });
    let shared_resource = if hardware.resident_shared_bytes > 0 {
        let i = state.model.resources.len();
        state.model.resources.push(Resource {
            name: "Metal resident shared bytes".into(),
            capacity: hardware.resident_shared_bytes,
            unit: schedule::CapacityUnit::Bytes,
        });
        Some(i)
    } else {
        None
    };
    // Incomplete terminal mappings cannot supply a feasible execution. Keeping
    // an explicit gap also prevents any diagnostic source from becoming a zero cost.
    if emitted.terminal.launches().len() != emitted.launches.len() {
        state
            .model
            .unmapped
            .push("launch has no retained terminal implementation".into());
        return Ok(state.model);
    }
    let scalar_values = canonical_scalars(&emitted, workload)?;
    let mut predecessor = None;
    for (launch, (metadata, body)) in emitted
        .launches
        .iter()
        .zip(emitted.terminal.launches())
        .enumerate()
    {
        state.scope = format!("launch {launch}");
        state.last = predecessor;
        let start = state.issue(Primitive::Launch, 1)?;
        let dispatch = metadata
            .dispatch
            .as_ref()
            .ok_or("Metal target launch has no dispatch")?;
        let mut groups = Vec::new();
        for group in 0..dispatch.groups {
            state.scope = format!("launch {launch} group {group}");
            state.last = Some(start);
            let begin = state.issue(Primitive::Group, 1)?;
            let mut subgroups = Vec::new();
            for subgroup in 0..dispatch.items_per_group {
                state.scope = format!("launch {launch} group {group} subgroup {subgroup}");
                state.last = Some(begin);
                state.env = scalar_values.clone();
                state.active = u32::MAX;
                state.alive = u32::MAX;
                state.returned = [None; 32];
                state
                    .env
                    .insert("lane".into(), std::array::from_fn(|i| Some(i as u64)));
                for b in &emitted.buffers {
                    let name = if b.plane.is_empty() {
                        b.parameter.clone()
                    } else {
                        format!("{}_{}", b.parameter, b.plane)
                    };
                    state.env.insert(name, [None; 32]);
                }
                state.env.insert("tg_pos.x".into(), [Some(group); 32]);
                state.env.insert("sg_id".into(), [Some(subgroup); 32]);
                if let Err(gap) = state.block(body, 0, body.len())? {
                    state.model.unmapped.push(format!("{}: {gap}", state.scope));
                }
                subgroups.push(state.last.unwrap_or(begin));
            }
            let end = state.join(subgroups)?;
            state.model.lifetimes.push(schedule::Lifetime {
                resource: groups_resource,
                units: 1,
                begin: schedule::Event {
                    operation: begin,
                    point: schedule::Point::Start,
                },
                end: schedule::Event {
                    operation: end,
                    point: schedule::Point::Completion,
                },
            });
            if metadata.declared_threadgroup_bytes > 0 {
                if let Some(resource) = shared_resource {
                    state.model.lifetimes.push(schedule::Lifetime {
                        resource,
                        units: metadata.declared_threadgroup_bytes,
                        begin: schedule::Event {
                            operation: begin,
                            point: schedule::Point::Start,
                        },
                        end: schedule::Event {
                            operation: end,
                            point: schedule::Point::Completion,
                        },
                    });
                } else {
                    state
                        .model
                        .unmapped
                        .push("shared allocation has no pooled resident capacity".into());
                }
            }
            groups.push(end);
        }
        predecessor = Some(state.join(groups)?);
    }
    state.model.unmapped.sort();
    state.model.unmapped.dedup();
    Ok(state.model)
}
fn canonical_scalars(
    emitted: &crate::msl::Emitted,
    workload: &ScalarWorkload,
) -> Result<BTreeMap<String, Values>, String> {
    let layout = ScalarLayout::words(&emitted.scalars)?;
    layout.validate_bytes(&workload.scalars)?;
    let mut result = BTreeMap::new();
    for (i, p) in emitted.scalars.iter().enumerate() {
        let mut bytes = [0; 8];
        bytes.copy_from_slice(&workload.scalars[i * 8..i * 8 + 8]);
        result.insert(
            format!("sc.{}", p.name),
            [Some(u64::from_le_bytes(bytes)); 32],
        );
    }
    Ok(result)
}
fn validate_workload(emitted: &crate::msl::Emitted, w: &ScalarWorkload) -> Result<(), String> {
    if w.identity.is_empty() || w.buffers.len() != emitted.buffers.len() {
        return Err("Metal workload identity or buffer count differs from target ABI".into());
    }
    let mut allocations = BTreeMap::new();
    for a in &w.allocations {
        if a.alignment == 0
            || !a.alignment.is_power_of_two()
            || allocations.insert(a.id, a).is_some()
            || a.known_bytes.keys().any(|offset| *offset >= a.bytes)
        {
            return Err("invalid Metal workload allocation".into());
        }
    }
    for (b, spec) in w.buffers.iter().zip(&emitted.buffers) {
        let a = allocations
            .get(&b.allocation)
            .ok_or("unknown Metal allocation")?;
        if b.bytes < spec.bytes as u64
            || b.offset
                .checked_add(b.bytes)
                .is_none_or(|end| end > a.bytes)
            || a.alignment < spec.alignment as u64
            || b.offset % (spec.alignment as u64) != 0
        {
            return Err("Metal workload buffer violates target ABI".into());
        }
    }
    for &(a, b, exact) in &emitted.alias_pairs {
        let (left_bytes, right_bytes) = (
            emitted.buffers[a].bytes as u64,
            emitted.buffers[b].bytes as u64,
        );
        let (a, b) = (&w.buffers[a], &w.buffers[b]);
        if a.allocation == b.allocation
            && left_bytes != 0
            && right_bytes != 0
            && a.offset < b.offset + right_bytes
            && b.offset < a.offset + left_bytes
            && !(exact && a.offset == b.offset && left_bytes == right_bytes)
        {
            return Err("Metal workload alias violates source or partition admission".into());
        }
    }
    Ok(())
}
type Values = [Option<u64>; 32];
struct Derivation<'a> {
    hardware: &'a Hardware,
    limits: DerivationLimits,
    visits: u64,
    model: Model,
    last: Option<usize>,
    scope: String,
    env: BTreeMap<String, Values>,
    active: u32,
    alive: u32,
    returned: Values,
}
impl Derivation<'_> {
    fn issue(&mut self, primitive: Primitive, lanes: u64) -> Result<usize, DerivationError> {
        self.visits = self
            .visits
            .checked_add(1)
            .ok_or("Metal visit count overflow")?;
        if self.visits > self.limits.instructions {
            return Err(DerivationError::Exhausted(DerivationLimit::Instructions(
                self.limits.instructions,
            )));
        }
        if self.model.operations.len() >= self.limits.operations {
            return Err(DerivationError::Exhausted(DerivationLimit::Operations(
                self.limits.operations,
            )));
        }
        let timing = self
            .hardware
            .timings
            .iter()
            .find(|t| t.primitive == primitive);
        let (latency, reservations) = if let Some(t) = timing {
            (
                t.latency,
                t.services
                    .iter()
                    .map(|s| {
                        Ok(Reservation {
                            resource: s.resource,
                            offset: s.offset,
                            duration: s.duration,
                            units: match s.units {
                                Units::PerLane(n) => {
                                    n.checked_mul(lanes).ok_or("Metal service count overflow")?
                                }
                                Units::PerSubgroup(n) => n,
                            },
                        })
                    })
                    .collect::<Result<Vec<_>, String>>()?,
            )
        } else {
            self.model
                .unmapped
                .push(format!("Metal primitive {primitive:?}"));
            (0, Vec::new())
        };
        let i = self.model.operations.len();
        self.model.operations.push(Operation {
            name: format!("{} instruction {i}: {primitive:?}", self.scope),
            predecessors: self.last.into_iter().collect(),
            start_predecessors: Vec::new(),
            latency,
            reservations,
        });
        self.last = Some(i);
        Ok(i)
    }
    fn join(&mut self, predecessors: Vec<usize>) -> Result<usize, DerivationError> {
        if self.model.operations.len() >= self.limits.operations {
            return Err(DerivationError::Exhausted(DerivationLimit::Operations(
                self.limits.operations,
            )));
        }
        let i = self.model.operations.len();
        self.model.operations.push(Operation {
            name: format!("completion {i}"),
            predecessors,
            start_predecessors: Vec::new(),
            latency: 0,
            reservations: Vec::new(),
        });
        self.last = Some(i);
        Ok(i)
    }
    fn expr(&mut self, e: &Expression) -> Result<Result<Values, String>, DerivationError> {
        macro_rules! value {
            ($e:expr) => {
                match self.expr($e)? {
                    Ok(v) => v,
                    Err(g) => return Ok(Err(g)),
                }
            };
        }
        use Expression as E;
        let lanes = u64::from(self.active.count_ones());
        if lanes == 0 {
            return Ok(Ok([None; 32]));
        }
        let result = match e {
            E::Integer(n, _) => [Some(*n as u64); 32],
            E::Float(bits, t) => {
                [if *t == Type::F32 {
                    Some((f64::from_bits(*bits) as f32).to_bits() as u64)
                } else {
                    None
                }; 32]
            }
            E::Parameter { name, ty } => {
                self.issue(
                    Primitive::Read {
                        space: crate::terminal::Space::Constant,
                        ty: *ty,
                    },
                    lanes,
                )?;
                self.env
                    .get(name)
                    .copied()
                    .ok_or("unbound Metal scalar ABI field")?
            }
            E::Variable(name, _) => match self.env.get(name) {
                Some(v) => *v,
                None => return Ok(Err(format!("unresolved target value {name}"))),
            },
            E::Unmapped(_, _) => return Ok(Err("unmapped target expression".into())),
            E::Binary(op, a, b, _) => {
                let (a, b) = (value!(a), value!(b));
                self.issue(
                    Primitive::Binary {
                        operation: *op,
                        ty: a_type(e),
                    },
                    lanes,
                )?;
                std::array::from_fn(|i| {
                    a[i].zip(b[i])
                        .and_then(|(a, b)| integer_binary(*op, a, b, a_type(e)))
                })
            }
            E::Unary(op, a, ty) => {
                let a = value!(a);
                self.issue(
                    Primitive::Unary {
                        operation: *op,
                        ty: *ty,
                    },
                    lanes,
                )?;
                a.map(|a| {
                    a.and_then(|a| {
                        if matches!(ty, Type::F16 | Type::BF16 | Type::F32) {
                            None
                        } else {
                            match op {
                                UnaryOp::Not => Some(u64::from(a == 0)),
                                UnaryOp::BitNot => Some(!a),
                                UnaryOp::Neg => {
                                    if matches!(ty, Type::I32) {
                                        (a as i32).checked_neg().map(|n| n as u32 as u64)
                                    } else if matches!(ty, Type::I64) {
                                        (a as i64).checked_neg().map(|n| n as u64)
                                    } else {
                                        Some(a.wrapping_neg())
                                    }
                                }
                            }
                        }
                    })
                })
            }
            E::Cast(ty, a) => {
                let from = a.ty();
                let a = value!(a);
                self.issue(Primitive::Cast { from, to: *ty }, lanes)?;
                a.map(|v| v.and_then(|v| integer_cast(v, from, *ty)))
            }
            E::Bitcast(ty, a) => {
                let from = a.ty();
                let a = value!(a);
                self.issue(Primitive::Bitcast { from, to: *ty }, lanes)?;
                a
            }
            E::ShortCircuit { or, left, right } => {
                let left = value!(left);
                let outer = self.active;
                let yes = match self.predicate(left) {
                    Ok(v) => v,
                    Err(g) => return Ok(Err(g)),
                };
                self.issue(Primitive::Branch, lanes)?;
                self.active = if *or { outer & !yes } else { yes };
                let right = value!(right);
                self.active = outer;
                std::array::from_fn(|i| {
                    if (*or && yes & (1 << i) != 0) || (!*or && yes & (1 << i) == 0) {
                        Some(u64::from(*or))
                    } else {
                        right[i].map(|v| u64::from(v != 0))
                    }
                })
            }
            E::Select(c, a, b) => {
                let c = value!(c);
                let outer = self.active;
                let mut yes = 0;
                let mut no = 0;
                for (i, c) in c.iter().enumerate() {
                    if outer & (1 << i) != 0 {
                        match c {
                            Some(0) => no |= 1 << i,
                            Some(_) => yes |= 1 << i,
                            None => return Ok(Err("unknown target select predicate".into())),
                        }
                    }
                }
                self.active = yes;
                let a = value!(a);
                self.active = no;
                let b = value!(b);
                self.active = outer;
                self.issue(Primitive::Select, lanes)?;
                std::array::from_fn(|i| if yes & (1 << i) != 0 { a[i] } else { b[i] })
            }
            E::Builtin(name, args, ty) => {
                for a in args {
                    value!(a);
                }
                self.issue(
                    Primitive::Builtin {
                        name: name.clone(),
                        inputs: args.iter().map(E::ty).collect(),
                        result: *ty,
                    },
                    lanes,
                )?;
                [None; 32]
            }
            E::Helper(helper, args, ty) => {
                let definition = crate::support::Definition::new(*helper);
                let values = args
                    .iter()
                    .map(|arg| self.expr(arg))
                    .collect::<Result<Vec<_>, _>>()?;
                let mut bindings = BTreeMap::new();
                for ((name, _), value) in definition.parameters.iter().zip(values) {
                    match value {
                        Ok(v) => {
                            bindings.insert((*name).to_string(), v);
                        }
                        Err(g) => return Ok(Err(g)),
                    }
                }
                let element = if *helper == crate::support::Helper::Write {
                    args.get(3).map(E::ty).unwrap_or(*ty)
                } else {
                    *ty
                };
                let body = helper_body(&definition, element)?;
                let saved = std::mem::replace(&mut self.env, bindings);
                let outer = (self.active, self.alive, self.returned);
                self.alive = self.active;
                self.returned = [None; 32];
                let result = self.block(&body, 0, body.len())?;
                let values = self.returned;
                self.env = saved;
                (self.active, self.alive, self.returned) = outer;
                return Ok(result.map(|_| values));
            }
            E::Read {
                index, space, ty, ..
            } => {
                value!(index);
                self.issue(
                    Primitive::Read {
                        space: *space,
                        ty: *ty,
                    },
                    lanes,
                )?;
                [None; 32]
            }
        };
        Ok(Ok(result.map(|v| v.map(|v| mask(v, e.ty())))))
    }
    fn assign(&mut self, name: &str, values: Values) {
        let old = self.env.entry(name.into()).or_insert([None; 32]);
        for i in 0..32 {
            if self.active & (1 << i) != 0 {
                old[i] = values[i];
            }
        }
    }
    fn predicate(&self, values: Values) -> Result<u32, String> {
        let mut yes = 0;
        for (i, v) in values.iter().enumerate() {
            if self.active & (1 << i) != 0 {
                match v {
                    Some(0) => {}
                    Some(_) => yes |= 1 << i,
                    None => return Err("unknown target control predicate".into()),
                }
            }
        }
        Ok(yes)
    }
    fn block(
        &mut self,
        body: &[crate::terminal::Site],
        start: usize,
        end: usize,
    ) -> Result<Result<(), String>, DerivationError> {
        let mut at = start;
        while at < end && self.active != 0 {
            macro_rules! value {
                ($e:expr) => {
                    match self.expr($e)? {
                        Ok(v) => v,
                        Err(g) => return Ok(Err(g)),
                    }
                };
            }
            let lanes = u64::from(self.active.count_ones());
            match &body[at].statement {
                Statement::Let { name, value, .. } | Statement::Assign { name, value } => {
                    let v = value!(value);
                    self.assign(name, v);
                }
                Statement::Evaluate(e) => {
                    value!(e);
                }
                Statement::ReturnIf(e) => {
                    let values = value!(e);
                    let yes = match self.predicate(values) {
                        Ok(v) => v,
                        Err(g) => return Ok(Err(g)),
                    };
                    self.issue(Primitive::Branch, lanes)?;
                    if yes != 0 {
                        self.issue(Primitive::Return, u64::from(yes.count_ones()))?;
                    }
                    self.alive &= !yes;
                    self.active &= !yes;
                }
                Statement::Return(e) => {
                    let values = if let Some(e) = e {
                        value!(e)
                    } else {
                        [None; 32]
                    };
                    self.issue(Primitive::Return, lanes)?;
                    for (i, v) in values.iter().enumerate() {
                        if self.active & (1 << i) != 0 {
                            self.returned[i] = *v;
                        }
                    }
                    self.alive &= !self.active;
                    self.active = 0;
                }
                Statement::FailureStatus => {
                    return Ok(Err("workload can execute target validity failure".into()));
                }
                Statement::For {
                    name,
                    start,
                    end: upper,
                    step,
                } => {
                    let (close, alternative) = matching_end(body, at, end)?;
                    if alternative.is_some() {
                        return Err("else on target loop".into());
                    }
                    if *step <= 0 {
                        return Err("nonpositive target loop step".into());
                    }
                    let outer = self.active;
                    let mut n = value!(start);
                    self.assign(name, n);
                    loop {
                        let limit = value!(upper);
                        let mut live = 0;
                        for i in 0..32 {
                            if self.active & (1 << i) != 0 {
                                let (Some(n), Some(limit)) = (n[i], limit[i]) else {
                                    return Ok(Err("unknown target loop domain".into()));
                                };
                                if (n as i32) < (limit as i32) {
                                    live |= 1 << i;
                                }
                            }
                        }
                        self.issue(
                            Primitive::Binary {
                                operation: BinaryOp::Lt,
                                ty: Type::I32,
                            },
                            u64::from(self.active.count_ones()),
                        )?;
                        self.issue(Primitive::Branch, u64::from(self.active.count_ones()))?;
                        self.active = live;
                        if live == 0 {
                            break;
                        }
                        if let Err(g) = self.block(body, at + 1, close)? {
                            return Ok(Err(g));
                        }
                        if self.active == 0 {
                            break;
                        }
                        self.issue(
                            Primitive::Binary {
                                operation: BinaryOp::Add,
                                ty: Type::I32,
                            },
                            u64::from(self.active.count_ones()),
                        )?;
                        for i in 0..32 {
                            if self.active & (1 << i) != 0 {
                                n[i] = n[i]
                                    .and_then(|n| (n as i32).checked_add(*step as i32))
                                    .map(|n| n as u32 as u64);
                            }
                        }
                        self.assign(name, n);
                    }
                    self.active = outer & self.alive;
                    at = close;
                }
                Statement::If(condition) => {
                    let (close, alternative) = matching_end(body, at, end)?;
                    let condition = value!(condition);
                    let yes = match self.predicate(condition) {
                        Ok(v) => v,
                        Err(g) => return Ok(Err(g)),
                    };
                    let outer = self.active;
                    self.issue(Primitive::Branch, lanes)?;
                    self.active = yes;
                    if let Err(g) = self.block(body, at + 1, alternative.unwrap_or(close))? {
                        return Ok(Err(g));
                    }
                    self.active = (outer & !yes) & self.alive;
                    if let Some(other) = alternative {
                        if let Err(g) = self.block(body, other + 1, close)? {
                            return Ok(Err(g));
                        }
                    }
                    self.active = outer & self.alive;
                    at = close;
                }
                Statement::Pointer {
                    name,
                    index,
                    space,
                    ty,
                    ..
                } => {
                    value!(index);
                    self.issue(
                        Primitive::Address {
                            space: *space,
                            ty: *ty,
                        },
                        lanes,
                    )?;
                    self.assign(name, [None; 32]);
                }
                Statement::Array { .. } | Statement::Fragment { .. } => {}
                Statement::MatrixLoad {
                    layout,
                    offset,
                    leading,
                    space,
                    transpose,
                    ..
                } => {
                    if self.active != u32::MAX {
                        return Ok(Err("matrix load requires all subgroup participants".into()));
                    }
                    value!(offset);
                    value!(leading);
                    self.issue(
                        Primitive::Address {
                            space: *space,
                            ty: layout.dtype.into(),
                        },
                        lanes,
                    )?;
                    self.issue(
                        Primitive::MatrixLoad {
                            layout: *layout,
                            space: *space,
                            transpose: *transpose,
                        },
                        lanes,
                    )?;
                }
                Statement::MatrixStore {
                    layout,
                    offset,
                    leading,
                    space,
                    ..
                } => {
                    if self.active != u32::MAX {
                        return Ok(Err("matrix store requires all subgroup participants".into()));
                    }
                    value!(offset);
                    value!(leading);
                    self.issue(
                        Primitive::Address {
                            space: *space,
                            ty: layout.dtype.into(),
                        },
                        lanes,
                    )?;
                    self.issue(
                        Primitive::MatrixStore {
                            layout: *layout,
                            space: *space,
                        },
                        lanes,
                    )?;
                }
                Statement::MatrixMultiplyAccumulate { layouts, .. } => {
                    if self.active != u32::MAX {
                        return Ok(Err(
                            "matrix arithmetic requires all subgroup participants".into()
                        ));
                    }
                    self.issue(
                        Primitive::MatrixMultiplyAccumulate { layouts: *layouts },
                        lanes,
                    )?;
                }
                Statement::Barrier => {
                    if self.active != u32::MAX {
                        return Err("partial target barrier participation".into());
                    }
                    self.issue(Primitive::Barrier, lanes)?;
                }
                Statement::End | Statement::Else => return Err("unmatched target scope".into()),
                Statement::Unmapped(_) => {
                    return Ok(Err(format!(
                        "unmapped target statement at {:?}",
                        body[at].operation
                    )));
                }
                Statement::Write {
                    index,
                    space,
                    ty,
                    value,
                    ..
                } => {
                    value!(index);
                    value!(value);
                    self.issue(
                        Primitive::Write {
                            space: *space,
                            ty: *ty,
                        },
                        lanes,
                    )?;
                }
            }
            at += 1;
        }
        Ok(Ok(()))
    }
}

fn matching_end(
    body: &[crate::terminal::Site],
    at: usize,
    end: usize,
) -> Result<(usize, Option<usize>), String> {
    let mut depth = 0;
    let mut alternative = None;
    for (i, s) in body.iter().enumerate().take(end).skip(at + 1) {
        match s.statement {
            Statement::If(_) | Statement::For { .. } => depth += 1,
            Statement::End if depth == 0 => return Ok((i, alternative)),
            Statement::End => depth -= 1,
            Statement::Else if depth == 0 => alternative = Some(i),
            _ => {}
        }
    }
    Err("unterminated typed target scope".into())
}
fn a_type(e: &Expression) -> Type {
    match e {
        Expression::Binary(_, a, _, _) => a.ty(),
        _ => e.ty(),
    }
}
fn mask(v: u64, t: Type) -> u64 {
    match t {
        Type::Bool => u64::from(v != 0),
        Type::I32 | Type::U32 | Type::F32 => v & u32::MAX as u64,
        Type::F16 | Type::BF16 => v & u16::MAX as u64,
        _ => v,
    }
}
fn integer_cast(v: u64, from: Type, to: Type) -> Option<u64> {
    if matches!(from, Type::F16 | Type::BF16 | Type::F32)
        || matches!(to, Type::F16 | Type::BF16 | Type::F32)
    {
        return if from == to { Some(v) } else { None };
    }
    let v = match from {
        Type::I32 => (v as i32 as i64) as u64,
        _ => v,
    };
    Some(mask(v, to))
}
fn integer_binary(op: BinaryOp, a: u64, b: u64, t: Type) -> Option<u64> {
    if matches!(t, Type::F16 | Type::BF16 | Type::F32) {
        return None;
    }
    use BinaryOp::*;
    let signed = matches!(t, Type::I32 | Type::I64);
    let ai = if t == Type::I32 {
        a as i32 as i64
    } else {
        a as i64
    };
    let bi = if t == Type::I32 {
        b as i32 as i64
    } else {
        b as i64
    };
    Some(match op {
        Add => {
            if signed {
                ai.checked_add(bi)
                    .filter(|v| t != Type::I32 || i32::try_from(*v).is_ok())? as u64
            } else {
                a.wrapping_add(b)
            }
        }
        Sub => {
            if signed {
                ai.checked_sub(bi)
                    .filter(|v| t != Type::I32 || i32::try_from(*v).is_ok())? as u64
            } else {
                a.wrapping_sub(b)
            }
        }
        Mul => {
            if signed {
                ai.checked_mul(bi)
                    .filter(|v| t != Type::I32 || i32::try_from(*v).is_ok())? as u64
            } else {
                a.wrapping_mul(b)
            }
        }
        Div if b != 0 => {
            if signed {
                ai.checked_div(bi)? as u64
            } else {
                a / b
            }
        }
        Rem if b != 0 => {
            if signed {
                ai.checked_rem(bi)? as u64
            } else {
                a % b
            }
        }
        Eq => u64::from(a == b),
        Ne => u64::from(a != b),
        Lt => u64::from(if signed { ai < bi } else { a < b }),
        Le => u64::from(if signed { ai <= bi } else { a <= b }),
        Gt => u64::from(if signed { ai > bi } else { a > b }),
        Ge => u64::from(if signed { ai >= bi } else { a >= b }),
        And => u64::from(a != 0 && b != 0),
        Or => u64::from(a != 0 || b != 0),
        BitAnd => a & b,
        BitOr => a | b,
        BitXor => a ^ b,
        Shl if b < if matches!(t, Type::I64 | Type::U64) {
            64
        } else {
            32
        } =>
        {
            a << b
        }
        Shr if b < if matches!(t, Type::I64 | Type::U64) {
            64
        } else {
            32
        } =>
        {
            if signed {
                (ai >> b) as u64
            } else {
                a >> b
            }
        }
        _ => return None,
    })
}

fn helper_body(
    definition: &crate::support::Definition,
    element: Type,
) -> Result<Vec<crate::terminal::Site>, String> {
    use crate::support::{Binary as B, Expression as E, Statement as S, Type as T};
    fn ty(t: T, element: Type) -> Type {
        match t {
            T::Bool => Type::Bool,
            T::I32 => Type::I32,
            T::U32 => Type::U32,
            T::I64 => Type::I64,
            T::U64 | T::DevicePointer | T::StatusPointer => Type::U64,
            T::Element => element,
            T::Void => Type::Bool,
        }
    }
    fn expr(e: &E, types: &BTreeMap<&str, Type>, element: Type) -> Result<Expression, String> {
        Ok(match e {
            E::Value(name) => {
                Expression::variable(*name, *types.get(name).ok_or("helper value has no type")?)
            }
            E::Integer(n) => Expression::Integer(
                *n,
                if i32::try_from(*n).is_ok() {
                    Type::I32
                } else {
                    Type::I64
                },
            ),
            E::Bool(v) => Expression::Integer(i64::from(*v), Type::Bool),
            E::Cast(t, e) => expr(e, types, element)?.cast(ty(*t, element)),
            E::Bitcast(t, e) => {
                Expression::Bitcast(ty(*t, element), Box::new(expr(e, types, element)?))
            }
            E::Negate(e) => {
                let e = expr(e, types, element)?;
                let t = e.ty();
                Expression::Unary(UnaryOp::Neg, Box::new(e), t)
            }
            E::Select(c, a, b) => Expression::Select(
                Box::new(expr(c, types, element)?),
                Box::new(expr(a, types, element)?),
                Box::new(expr(b, types, element)?),
            ),
            E::Binary(op, a, b) if matches!(op, B::And | B::Or) => Expression::ShortCircuit {
                or: *op == B::Or,
                left: Box::new(expr(a, types, element)?),
                right: Box::new(expr(b, types, element)?),
            },
            E::Binary(op, a, b) => {
                let a = expr(a, types, element)?;
                let b = expr(b, types, element)?;
                let (op, comparison) = match op {
                    B::Add => (BinaryOp::Add, false),
                    B::Subtract => (BinaryOp::Sub, false),
                    B::Divide => (BinaryOp::Div, false),
                    B::Remainder => (BinaryOp::Rem, false),
                    B::ShiftLeft => (BinaryOp::Shl, false),
                    B::ShiftRight => (BinaryOp::Shr, false),
                    B::Less => (BinaryOp::Lt, true),
                    B::GreaterEqual => (BinaryOp::Ge, true),
                    B::LessEqual => (BinaryOp::Le, true),
                    B::Equal => (BinaryOp::Eq, true),
                    B::And => (BinaryOp::And, true),
                    B::Or => (BinaryOp::Or, true),
                };
                let t = if comparison { Type::Bool } else { a.ty() };
                Expression::binary(op, a, b, t)
            }
            E::Read { pointer, index } => {
                let E::Value(name) = &**pointer else {
                    return Err("unsupported helper pointer expression".into());
                };
                Expression::Read {
                    name: (*name).into(),
                    index: Box::new(expr(index, types, element)?),
                    space: crate::terminal::Space::Device,
                    ty: element,
                }
            }
        })
    }
    fn statements(
        body: &[S],
        types: &mut BTreeMap<&'static str, Type>,
        element: Type,
        out: &mut Vec<crate::terminal::Site>,
    ) -> Result<(), String> {
        let push = |out: &mut Vec<crate::terminal::Site>, statement| {
            out.push(crate::terminal::Site {
                operation: None,
                statement,
            })
        };
        for s in body {
            match s {
                S::Let { name, ty: t, value } => {
                    let value = expr(value, types, element)?;
                    let t = ty(*t, element);
                    types.insert(name, t);
                    push(
                        out,
                        Statement::Let {
                            name: (*name).into(),
                            ty: t,
                            value,
                        },
                    );
                }
                S::Assign { name, value } => push(
                    out,
                    Statement::Assign {
                        name: (*name).into(),
                        value: expr(value, types, element)?,
                    },
                ),
                S::If { condition, body } => {
                    push(out, Statement::If(expr(condition, types, element)?));
                    statements(body, types, element, out)?;
                    push(out, Statement::End);
                }
                S::Return(value) => push(
                    out,
                    Statement::Return(
                        value
                            .as_ref()
                            .map(|v| expr(v, types, element))
                            .transpose()?,
                    ),
                ),
                S::FailureStatus => push(out, Statement::FailureStatus),
                S::Write {
                    pointer,
                    index,
                    value,
                } => {
                    let E::Value(name) = pointer else {
                        return Err("unsupported helper write pointer".into());
                    };
                    push(
                        out,
                        Statement::Write {
                            name: (*name).into(),
                            index: expr(index, types, element)?,
                            space: crate::terminal::Space::Device,
                            ty: element,
                            value: expr(value, types, element)?,
                        },
                    );
                }
            }
        }
        Ok(())
    }
    let mut types = definition
        .parameters
        .iter()
        .map(|(n, t)| (*n, ty(*t, element)))
        .collect();
    let mut out = Vec::new();
    statements(&definition.body, &mut types, element, &mut out)?;
    Ok(out)
}

#[derive(Clone, Debug)]
pub struct Requirements {
    pub primitives: Vec<Primitive>,
    pub unmapped: Vec<String>,
}
/// Resource keys come from the same terminal nodes that render the selected MSL.
/// This inventories static mappings, not dynamic work or a per-kernel timing table.
pub fn requirements(execution: &crate::execution::Execution) -> Result<Requirements, String> {
    let emitted = crate::msl::prepare_execution(execution)?;
    let mut out = Requirements {
        primitives: vec![Primitive::Launch, Primitive::Group],
        unmapped: Vec::new(),
    };
    fn add(out: &mut Requirements, p: Primitive) {
        if !out.primitives.contains(&p) {
            out.primitives.push(p);
        }
    }
    fn expression(e: &Expression, out: &mut Requirements) -> Result<(), String> {
        use Expression as E;
        match e {
            E::Integer(..) | E::Float(..) | E::Variable(..) => {}
            E::Parameter { ty, .. } => add(
                out,
                Primitive::Read {
                    space: crate::terminal::Space::Constant,
                    ty: *ty,
                },
            ),
            E::Unmapped(..) => out.unmapped.push("terminal expression".into()),
            E::Binary(op, a, b, _) => {
                expression(a, out)?;
                expression(b, out)?;
                add(
                    out,
                    Primitive::Binary {
                        operation: *op,
                        ty: a.ty(),
                    },
                );
            }
            E::Unary(op, a, t) => {
                expression(a, out)?;
                add(
                    out,
                    Primitive::Unary {
                        operation: *op,
                        ty: *t,
                    },
                );
            }
            E::Cast(t, a) => {
                expression(a, out)?;
                add(
                    out,
                    Primitive::Cast {
                        from: a.ty(),
                        to: *t,
                    },
                );
            }
            E::Bitcast(t, a) => {
                expression(a, out)?;
                add(
                    out,
                    Primitive::Bitcast {
                        from: a.ty(),
                        to: *t,
                    },
                );
            }
            E::ShortCircuit { left, right, .. } => {
                expression(left, out)?;
                expression(right, out)?;
                add(out, Primitive::Branch);
            }
            E::Select(c, a, b) => {
                for e in [&**c, &**a, &**b] {
                    expression(e, out)?;
                }
                add(out, Primitive::Select);
            }
            E::Builtin(name, args, result) => {
                for a in args {
                    expression(a, out)?;
                }
                add(
                    out,
                    Primitive::Builtin {
                        name: name.clone(),
                        inputs: args.iter().map(E::ty).collect(),
                        result: *result,
                    },
                );
            }
            E::Helper(helper, args, result) => {
                for a in args {
                    expression(a, out)?;
                }
                let definition = crate::support::Definition::new(*helper);
                let element = if *helper == crate::support::Helper::Write {
                    args[3].ty()
                } else {
                    *result
                };
                statements(&helper_body(&definition, element)?, out)?;
            }
            E::Read {
                index, space, ty, ..
            } => {
                expression(index, out)?;
                add(
                    out,
                    Primitive::Read {
                        space: *space,
                        ty: *ty,
                    },
                );
            }
        }
        Ok(())
    }
    fn statements(body: &[crate::terminal::Site], out: &mut Requirements) -> Result<(), String> {
        for site in body {
            match &site.statement {
                Statement::Let { value, .. }
                | Statement::Assign { value, .. }
                | Statement::Evaluate(value) => expression(value, out)?,
                Statement::Write {
                    index,
                    space,
                    ty,
                    value,
                    ..
                } => {
                    expression(index, out)?;
                    expression(value, out)?;
                    add(
                        out,
                        Primitive::Write {
                            space: *space,
                            ty: *ty,
                        },
                    );
                }
                Statement::Pointer {
                    index, space, ty, ..
                } => {
                    expression(index, out)?;
                    add(
                        out,
                        Primitive::Address {
                            space: *space,
                            ty: *ty,
                        },
                    );
                }
                Statement::For { start, end, .. } => {
                    expression(start, out)?;
                    expression(end, out)?;
                    add(
                        out,
                        Primitive::Binary {
                            operation: BinaryOp::Lt,
                            ty: Type::I32,
                        },
                    );
                    add(
                        out,
                        Primitive::Binary {
                            operation: BinaryOp::Add,
                            ty: Type::I32,
                        },
                    );
                    add(out, Primitive::Branch);
                }
                Statement::If(e) | Statement::ReturnIf(e) => {
                    expression(e, out)?;
                    add(out, Primitive::Branch);
                    if matches!(site.statement, Statement::ReturnIf(_)) {
                        add(out, Primitive::Return);
                    }
                }
                Statement::Return(e) => {
                    if let Some(e) = e {
                        expression(e, out)?;
                    }
                    add(out, Primitive::Return);
                }
                Statement::MatrixLoad {
                    layout,
                    offset,
                    leading,
                    space,
                    transpose,
                    ..
                } => {
                    expression(offset, out)?;
                    expression(leading, out)?;
                    add(
                        out,
                        Primitive::Address {
                            space: *space,
                            ty: layout.dtype.into(),
                        },
                    );
                    add(
                        out,
                        Primitive::MatrixLoad {
                            layout: *layout,
                            space: *space,
                            transpose: *transpose,
                        },
                    );
                }
                Statement::MatrixStore {
                    layout,
                    offset,
                    leading,
                    space,
                    ..
                } => {
                    expression(offset, out)?;
                    expression(leading, out)?;
                    add(
                        out,
                        Primitive::Address {
                            space: *space,
                            ty: layout.dtype.into(),
                        },
                    );
                    add(
                        out,
                        Primitive::MatrixStore {
                            layout: *layout,
                            space: *space,
                        },
                    );
                }
                Statement::MatrixMultiplyAccumulate { layouts, .. } => add(
                    out,
                    Primitive::MatrixMultiplyAccumulate { layouts: *layouts },
                ),
                Statement::Barrier => add(out, Primitive::Barrier),
                Statement::Unmapped(_) => {
                    out.unmapped.push(format!("statement {:?}", site.operation))
                }
                Statement::Array { .. }
                | Statement::Fragment { .. }
                | Statement::Else
                | Statement::End
                | Statement::FailureStatus => {}
            }
        }
        Ok(())
    }
    for launch in emitted.terminal.launches() {
        statements(launch, &mut out)?;
    }
    out.unmapped.sort();
    out.unmapped.dedup();
    Ok(out)
}
