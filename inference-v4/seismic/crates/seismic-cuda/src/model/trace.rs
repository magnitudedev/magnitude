//! Abstract evaluation of terminal PTX, with symbolic allocation addresses.
//! Unknown data stays unknown. Trace-shaping predicates/addresses must be known.
use super::*;
use ptx::{
    Address, AddressBase, Binary, Comparison, DataType, Item, Operand, Operation, ParameterRole,
    Space, Unary,
};

pub(super) struct Trace {
    pub events: Vec<Event>,
    pub instructions: u64,
}
#[derive(Clone, Debug, PartialEq, Eq)]
enum Value {
    Bits(u64),
    Pointer(Allocation, u64),
    Unknown,
}
impl Value {
    fn bits(&self) -> Result<u64, String> {
        if let Self::Bits(v) = self {
            Ok(*v)
        } else {
            Err("CUDA trace needs a known integer or predicate".into())
        }
    }
    fn predicate(&self) -> Result<bool, String> {
        match self.bits()? {
            0 => Ok(false),
            1 => Ok(true),
            _ => Err("invalid PTX predicate value".into()),
        }
    }
}
#[derive(Clone)]
struct Cell {
    value: Value,
    writer: Option<usize>,
}
struct Lane {
    linear: u64,
    thread: u32,
    pc: Option<usize>,
    registers: Vec<Option<Cell>>,
    parameters: Vec<Option<Cell>>,
    control: usize,
    status: Option<u32>,
}
struct Storage {
    bytes: u64,
    alignment: u64,
    known: BTreeMap<u64, u8>,
    pointers: BTreeMap<u64, Value>,
}
struct Memory {
    allocations: BTreeMap<Allocation, Storage>,
    accesses: Vec<(Access, usize)>,
}

impl Memory {
    fn new(
        execution: &Execution,
        workload: &ScalarWorkload,
        hardware: &CudaHardware,
        limits: DerivationLimits,
    ) -> Result<Self, String> {
        let program = execution.program();
        if workload.identity.is_empty()
            || workload.buffers.len() != program.buffers.len()
            || workload.allocations.len() > limits.operations
            || workload.scalars.len() > limits.operations
        {
            return Err("invalid CUDA workload shape or budget".into());
        }
        seismic_lang::abi::ScalarLayout::words(&program.scalars)?
            .validate_bytes(&workload.scalars)?;
        let mut allocations = BTreeMap::new();
        let mut known_count = 0usize;
        for a in &workload.allocations {
            known_count = known_count
                .checked_add(a.known_bytes.len())
                .ok_or("CUDA binding budget overflow")?;
            if known_count > limits.operations
                || !a.alignment.is_power_of_two()
                || a.known_bytes.keys().any(|&i| i >= a.bytes)
                || allocations
                    .insert(
                        Allocation::External(a.id),
                        Storage {
                            bytes: a.bytes,
                            alignment: a.alignment,
                            known: a.known_bytes.clone(),
                            pointers: BTreeMap::new(),
                        },
                    )
                    .is_some()
            {
                return Err(
                    "invalid CUDA allocation identity, alignment, byte domain or budget".into(),
                );
            }
        }
        let mut pointers = BTreeMap::new();
        for (i, (binding, spec)) in workload.buffers.iter().zip(&program.buffers).enumerate() {
            let a = allocations
                .get(&Allocation::External(binding.allocation))
                .ok_or("CUDA buffer refers to missing allocation")?;
            let alignment =
                u64::try_from(spec.alignment).map_err(|_| "buffer alignment overflow")?;
            if !alignment.is_power_of_two()
                || binding.bytes < spec.bytes as u64
                || binding
                    .offset
                    .checked_add(binding.bytes)
                    .is_none_or(|n| n > a.bytes)
                || a.alignment < alignment
                || binding.offset % alignment != 0
            {
                return Err("CUDA buffer size or alignment violates typed ABI".into());
            }
            pointers.insert(
                (i as u64).checked_mul(8).ok_or("buffer table overflow")?,
                Value::Pointer(Allocation::External(binding.allocation), binding.offset),
            );
        }
        allocations.insert(
            Allocation::BufferTable,
            Storage {
                bytes: execution.storage().buffer_table_bytes as u64,
                alignment: hardware.internal_alignment,
                known: BTreeMap::new(),
                pointers,
            },
        );
        allocations.insert(
            Allocation::Scalars,
            Storage {
                bytes: workload.scalars.len() as u64,
                alignment: hardware.internal_alignment,
                known: workload
                    .scalars
                    .iter()
                    .enumerate()
                    .map(|(i, &v)| (i as u64, v))
                    .collect(),
                pointers: BTreeMap::new(),
            },
        );
        allocations.insert(
            Allocation::Scratch,
            Storage {
                bytes: execution.storage().scratch_bytes as u64,
                alignment: hardware.internal_alignment,
                known: BTreeMap::new(),
                pointers: BTreeMap::new(),
            },
        );
        allocations.insert(
            Allocation::Statuses,
            Storage {
                bytes: execution.storage().status_bytes as u64,
                alignment: hardware.internal_alignment,
                known: BTreeMap::new(),
                pointers: BTreeMap::new(),
            },
        );
        Ok(Self {
            allocations,
            accesses: Vec::new(),
        })
    }
    fn access(&self, value: &Value, bytes: u32, lane: u64, write: bool) -> Result<Access, String> {
        let Value::Pointer(allocation, offset) = value else {
            return Err("CUDA memory address is not a known allocation-relative pointer".into());
        };
        let storage = self
            .allocations
            .get(allocation)
            .ok_or("unknown CUDA storage")?;
        if bytes == 0
            || offset
                .checked_add(u64::from(bytes))
                .is_none_or(|end| end > storage.bytes)
            || offset % u64::from(bytes) != 0
            || storage.alignment < u64::from(bytes)
        {
            return Err("PTX memory access exceeds storage or natural alignment".into());
        }
        if write && matches!(allocation, Allocation::BufferTable | Allocation::Scalars) {
            return Err("CUDA kernel writes immutable ABI input storage".into());
        }
        Ok(Access {
            lane,
            allocation: allocation.clone(),
            offset: *offset,
            bytes,
            alignment: storage.alignment,
            write,
        })
    }
    fn dependencies(&self, access: &Access) -> Result<Vec<usize>, String> {
        let mut dependencies = Vec::new();
        for (prior, event) in &self.accesses {
            if access.allocation == prior.allocation
                && (access.write || prior.write)
                && access.offset < prior.offset + u64::from(prior.bytes)
                && prior.offset < access.offset + u64::from(access.bytes)
            {
                if prior.lane != access.lane {
                    return Err("unsynchronized cross-lane conflicting CUDA memory access".into());
                }
                dependencies.push(*event);
            }
        }
        Ok(dependencies)
    }
    fn load(&self, a: &Access) -> Value {
        let storage = &self.allocations[&a.allocation];
        if a.bytes == 8 {
            if let Some(p) = storage.pointers.get(&a.offset) {
                return p.clone();
            }
        }
        let mut value = 0u64;
        for byte in 0..a.bytes {
            let Some(&b) = storage.known.get(&(a.offset + u64::from(byte))) else {
                return Value::Unknown;
            };
            value |= u64::from(b) << (byte * 8);
        }
        Value::Bits(value)
    }
    fn store(&mut self, a: &Access, value: Value) -> Result<(), String> {
        let storage = self.allocations.get_mut(&a.allocation).unwrap();
        // A partial write invalidates every overlapping saved pointer word.
        storage
            .pointers
            .retain(|&offset, _| offset >= a.offset + u64::from(a.bytes) || offset + 8 <= a.offset);
        for byte in 0..a.bytes {
            let offset = a.offset + u64::from(byte);
            if let Value::Bits(value) = value {
                storage
                    .known
                    .insert(offset, ((value >> (8 * byte)) & 255) as u8);
            } else {
                storage.known.remove(&offset);
            }
        }
        if let Value::Pointer(..) = value {
            if a.bytes != 8 {
                return Err("partial PTX pointer store is not modeled".into());
            }
            storage.pointers.insert(a.offset, value);
        }
        Ok(())
    }
}

fn next(plan: &ptx::TargetPlan, mut pc: usize) -> Result<usize, String> {
    while matches!(plan.body().get(pc), Some(Item::Label(_))) {
        pc += 1;
    }
    if !matches!(plan.body().get(pc), Some(Item::Instruction(_))) {
        return Err("PTX control falls off the entry body".into());
    }
    Ok(pc)
}
fn lifecycle(
    events: &mut Vec<Event>,
    kind: Lifecycle,
    block: Option<u64>,
    warp: Option<u64>,
    predecessors: Vec<usize>,
    limits: DerivationLimits,
) -> Result<usize, String> {
    if events.len() >= limits.operations {
        return Err("CUDA event derivation budget exceeded".into());
    }
    let i = events.len();
    events.push(Event {
        requirement: Requirement::Lifecycle(kind),
        block,
        warp,
        position: None,
        origin: None,
        issued_lanes: Vec::new(),
        active_lanes: Vec::new(),
        accesses: Vec::new(),
        predecessors,
        start_predecessors: Vec::new(),
    });
    Ok(i)
}

pub(super) fn derive(
    execution: &Execution,
    hardware: &CudaHardware,
    workload: &ScalarWorkload,
    limits: DerivationLimits,
) -> Result<Trace, String> {
    let plan = execution.target_plan();
    let d = execution.dispatch();
    let state_size = plan
        .registers()
        .len()
        .checked_add(plan.parameters().len())
        .and_then(|n| n.checked_mul(hardware.warp_width as usize))
        .ok_or("CUDA trace state overflow")?;
    if state_size as u64 > limits.instructions
        || plan.body().len() > limits.operations
        || d.dispatched_lanes() > limits.instructions
    {
        return Err("CUDA trace state exceeds derivation budget".into());
    }
    let labels = plan
        .body()
        .iter()
        .enumerate()
        .filter_map(|(i, item)| {
            if let Item::Label(l) = item {
                Some((*l, i))
            } else {
                None
            }
        })
        .collect::<BTreeMap<_, _>>();
    let mut memory = Memory::new(execution, workload, hardware, limits)?;
    let mut events = Vec::new();
    let launch = lifecycle(
        &mut events,
        Lifecycle::Launch,
        None,
        None,
        Vec::new(),
        limits,
    )?;
    let mut block_ends = Vec::new();
    let mut instructions = 0u64;
    let warps = d.threads_per_group.div_ceil(u64::from(hardware.warp_width));
    for block in 0..d.groups {
        let admission = lifecycle(
            &mut events,
            Lifecycle::BlockAdmission,
            Some(block),
            None,
            vec![launch],
            limits,
        )?;
        let mut warp_ends = Vec::new();
        for local_warp in 0..warps {
            let warp = block
                .checked_mul(warps)
                .and_then(|n| n.checked_add(local_warp))
                .ok_or("warp index overflow")?;
            let start = lifecycle(
                &mut events,
                Lifecycle::WarpStart,
                Some(block),
                Some(warp),
                vec![admission],
                limits,
            )?;
            let first = local_warp * u64::from(hardware.warp_width);
            let last = (first + u64::from(hardware.warp_width)).min(d.threads_per_group);
            let mut lanes = Vec::new();
            for thread in first..last {
                let linear = block * d.threads_per_group + thread;
                let mut parameters = Vec::new();
                for parameter in plan.parameters() {
                    let value = match parameter.role {
                        ParameterRole::Buffers => Some(Value::Pointer(Allocation::BufferTable, 0)),
                        ParameterRole::Scalars => Some(Value::Pointer(Allocation::Scalars, 0)),
                        ParameterRole::Scratch => Some(Value::Pointer(Allocation::Scratch, 0)),
                        ParameterRole::Statuses => Some(Value::Pointer(Allocation::Statuses, 0)),
                        ParameterRole::CallArgument(_) | ParameterRole::CallResult(_) => None,
                    };
                    parameters.push(value.map(|value| Cell {
                        value,
                        writer: None,
                    }));
                }
                lanes.push(Lane {
                    linear,
                    thread: thread as u32,
                    pc: Some(next(plan, 0)?),
                    registers: vec![None; plan.registers().len()],
                    parameters,
                    control: start,
                    status: None,
                });
            }
            let mut previous = start;
            let mut warp_events = Vec::new();
            while let Some(pc) = lanes.iter().filter_map(|l| l.pc).min() {
                let Item::Instruction(instruction) = &plan.body()[pc] else {
                    unreachable!()
                };
                let cohort = lanes
                    .iter()
                    .enumerate()
                    .filter_map(|(i, l)| (l.pc == Some(pc)).then_some(i))
                    .collect::<Vec<_>>();
                instructions = instructions
                    .checked_add(cohort.len() as u64)
                    .ok_or("CUDA instruction count overflow")?;
                if instructions > limits.instructions || events.len() >= limits.operations {
                    return Err("CUDA instruction or event derivation budget exceeded".into());
                }
                let event_id = events.len();
                let mut event = Event {
                    requirement: Requirement::Instruction(instruction.operation.primitive()),
                    block: Some(block),
                    warp: Some(warp),
                    position: Some(pc),
                    origin: Some(instruction.origin),
                    issued_lanes: Vec::new(),
                    active_lanes: Vec::new(),
                    accesses: Vec::new(),
                    predecessors: Vec::new(),
                    start_predecessors: vec![previous],
                };
                let mut active = Vec::new();
                for &i in &cohort {
                    let lane = &lanes[i];
                    event.issued_lanes.push(lane.thread - first as u32);
                    event.predecessors.push(lane.control);
                    let enabled = if let Some(p) = instruction.predicate {
                        let cell = lane.registers[p.register.0]
                            .as_ref()
                            .ok_or("undefined PTX predicate")?;
                        event.predecessors.extend(cell.writer);
                        cell.value.predicate()? != p.inverted
                    } else {
                        true
                    };
                    if enabled {
                        active.push(i);
                        event.active_lanes.push(lane.thread - first as u32);
                    }
                }
                // A call's body is a separate, explicitly complete opaque-body event.
                // Its memory/private/control internals are conditions of its mapping,
                // not omitted or charged as one primitive by the trace.
                let body_id = matches!(instruction.operation, Operation::Call { .. })
                    .then_some(event_id + 1)
                    .filter(|_| !active.is_empty());
                for &i in &active {
                    let lane = &mut lanes[i];
                    let effects = instruction.effects();
                    for r in effects
                        .register_reads
                        .iter()
                        .chain(&effects.register_writes)
                    {
                        if let Some(cell) = &lane.registers[r.0] {
                            event.predecessors.extend(cell.writer);
                        } else if effects.register_reads.contains(r) {
                            return Err("PTX reads an undefined register".into());
                        }
                    }
                    for p in effects
                        .parameter_reads
                        .iter()
                        .chain(&effects.parameter_writes)
                    {
                        if let Some(cell) = &lane.parameters[p.0] {
                            event.predecessors.extend(cell.writer);
                        } else if effects.parameter_reads.contains(p) {
                            return Err("PTX reads an undefined call parameter".into());
                        }
                    }
                    execute(
                        instruction,
                        lane,
                        block,
                        d.threads_per_group,
                        plan,
                        &labels,
                        &mut memory,
                        &mut event,
                        event_id,
                        body_id,
                    )?;
                }
                for &i in &cohort {
                    if !active.contains(&i) {
                        lanes[i].pc = Some(next(plan, pc + 1)?);
                    }
                }
                event.predecessors.sort_unstable();
                event.predecessors.dedup();
                for access in &event.accesses {
                    memory.accesses.push((access.clone(), event_id));
                }
                events.push(event);
                warp_events.push(event_id);
                previous = event_id;
                if let Some(body_id) = body_id {
                    if events.len() >= limits.operations {
                        return Err("CUDA helper body event exceeds budget".into());
                    }
                    let Operation::Call { function, .. } = instruction.operation else {
                        unreachable!()
                    };
                    let call = &events[event_id];
                    let body = Event {
                        requirement: Requirement::WholeBody(ptx::Helper::for_function(function)),
                        block: Some(block),
                        warp: Some(warp),
                        position: Some(pc),
                        origin: Some(instruction.origin),
                        issued_lanes: call.active_lanes.clone(),
                        active_lanes: call.active_lanes.clone(),
                        accesses: Vec::new(),
                        predecessors: vec![event_id],
                        start_predecessors: vec![event_id],
                    };
                    events.push(body);
                    warp_events.push(body_id);
                    previous = body_id;
                }
            }
            for lane in &lanes {
                if lane.linear < d.work_items && lane.status != Some(0) {
                    return Err("CUDA trace returns without successful invocation status".into());
                }
                if lane.linear >= d.work_items && lane.status.is_some() {
                    return Err("padded CUDA lane wrote invocation status".into());
                }
            }
            warp_ends.push(lifecycle(
                &mut events,
                Lifecycle::WarpCompletion,
                Some(block),
                Some(warp),
                warp_events,
                limits,
            )?);
        }
        block_ends.push(lifecycle(
            &mut events,
            Lifecycle::BlockCompletion,
            Some(block),
            None,
            warp_ends,
            limits,
        )?);
    }
    if block_ends.is_empty() {
        block_ends.push(launch);
    }
    lifecycle(
        &mut events,
        Lifecycle::Completion,
        None,
        None,
        block_ends,
        limits,
    )?;
    Ok(Trace {
        events,
        instructions,
    })
}

fn operand(op: Operand, lane: &Lane, block: u64, width: u64) -> Result<Value, String> {
    Ok(match op {
        Operand::Register(r) => lane.registers[r.0]
            .as_ref()
            .ok_or("undefined PTX operand")?
            .value
            .clone(),
        Operand::Signed(v) => Value::Bits(v as u64),
        Operand::Unsigned(v) => Value::Bits(v),
        Operand::Float32Bits(v) => Value::Bits(u64::from(v)),
        Operand::Special(s) => Value::Bits(match s {
            ptx::SpecialRegister::BlockIndexX => block,
            ptx::SpecialRegister::BlockWidthX => width,
            ptx::SpecialRegister::ThreadIndexX => u64::from(lane.thread),
        }),
    })
}
fn address(a: Address, lane: &Lane) -> Result<Value, String> {
    let AddressBase::Register(r) = a.base else {
        return Err("parameter address used as a global pointer".into());
    };
    let Value::Pointer(allocation, offset) = &lane.registers[r.0]
        .as_ref()
        .ok_or("undefined address register")?
        .value
    else {
        return Err("unknown CUDA memory address".into());
    };
    let offset = offset
        .checked_add_signed(i64::from(a.offset))
        .ok_or("CUDA pointer offset overflow")?;
    Ok(Value::Pointer(allocation.clone(), offset))
}
fn parameter_access(lane: &Lane, parameter: ptx::ParameterId, bytes: u32, write: bool) -> Access {
    Access {
        lane: lane.linear,
        allocation: Allocation::Parameter {
            lane: lane.linear,
            parameter: parameter.0,
        },
        offset: 0,
        bytes,
        alignment: u64::from(bytes),
        write,
    }
}

fn execute(
    instruction: &ptx::Instruction,
    lane: &mut Lane,
    block: u64,
    width: u64,
    plan: &ptx::TargetPlan,
    labels: &BTreeMap<ptx::Label, usize>,
    memory: &mut Memory,
    event: &mut Event,
    event_id: usize,
    body_id: Option<usize>,
) -> Result<(), String> {
    let pc = lane.pc.unwrap();
    let mut destination = None;
    let mut value = Value::Unknown;
    let mut advance = true;
    match instruction.operation {
        Operation::Unary {
            operation,
            data_type,
            destination: d,
            source,
            ..
        } => {
            destination = Some(d);
            value = unary(operation, data_type, operand(source, lane, block, width)?)?;
        }
        Operation::Binary {
            operation,
            data_type,
            destination: d,
            lhs,
            rhs,
            ..
        } => {
            destination = Some(d);
            value = binary(
                operation,
                data_type,
                operand(lhs, lane, block, width)?,
                operand(rhs, lane, block, width)?,
            )?;
        }
        Operation::Convert {
            destination_type,
            source_type,
            destination: d,
            source,
            ..
        } => {
            destination = Some(d);
            value = convert(
                destination_type,
                source_type,
                operand(source, lane, block, width)?,
            )?;
        }
        Operation::Compare {
            comparison,
            data_type,
            destination: d,
            lhs,
            rhs,
        } => {
            destination = Some(d);
            value = compare(
                comparison,
                data_type,
                operand(lhs, lane, block, width)?,
                operand(rhs, lane, block, width)?,
            )?;
        }
        Operation::Select {
            destination: d,
            when_true,
            when_false,
            predicate,
            ..
        } => {
            destination = Some(d);
            value = match &lane.registers[predicate.0]
                .as_ref()
                .ok_or("undefined select predicate")?
                .value
            {
                Value::Unknown => Value::Unknown,
                p => operand(
                    if p.predicate()? {
                        when_true
                    } else {
                        when_false
                    },
                    lane,
                    block,
                    width,
                )?,
            };
        }
        Operation::Fma { destination: d, .. } => {
            destination = Some(d);
        }
        Operation::Load {
            space,
            data_type,
            destination: d,
            address: at,
        } => {
            destination = Some(d);
            if space == Space::Parameter {
                let AddressBase::Parameter(p) = at.base else {
                    return Err("PTX parameter load requires declaration".into());
                };
                if at.offset != 0 {
                    return Err("partial parameter addressing is not modeled".into());
                }
                value = lane.parameters[p.0]
                    .as_ref()
                    .ok_or("undefined parameter value")?
                    .value
                    .clone();
                event
                    .accesses
                    .push(parameter_access(lane, p, data_type.bits() / 8, false));
            } else {
                let a = memory.access(
                    &address(at, lane)?,
                    data_type.bits() / 8,
                    lane.linear,
                    false,
                )?;
                event.predecessors.extend(memory.dependencies(&a)?);
                value = memory.load(&a);
                event.accesses.push(a);
            }
            value = truncate(value, data_type);
        }
        Operation::Store {
            space,
            data_type,
            address: at,
            value: v,
        } => {
            let value = truncate(operand(v, lane, block, width)?, data_type);
            if space == Space::Parameter {
                let AddressBase::Parameter(p) = at.base else {
                    return Err("PTX parameter store requires declaration".into());
                };
                if at.offset != 0 {
                    return Err("partial parameter addressing is not modeled".into());
                }
                lane.parameters[p.0] = Some(Cell {
                    value,
                    writer: Some(event_id),
                });
                event
                    .accesses
                    .push(parameter_access(lane, p, data_type.bits() / 8, true));
            } else {
                let a =
                    memory.access(&address(at, lane)?, data_type.bits() / 8, lane.linear, true)?;
                if a.allocation == Allocation::Statuses {
                    if a.bytes != 4
                        || a.offset != lane.linear.checked_mul(4).ok_or("status index overflow")?
                    {
                        return Err("CUDA lane writes another invocation status".into());
                    }
                    lane.status =
                        Some(u32::try_from(value.bits()?).map_err(|_| "invalid CUDA status")?);
                }
                event.predecessors.extend(memory.dependencies(&a)?);
                // Also check other lanes of this same instruction before mutation.
                for prior in &event.accesses {
                    if prior.lane != a.lane
                        && prior.allocation == a.allocation
                        && prior.offset < a.offset + u64::from(a.bytes)
                        && a.offset < prior.offset + u64::from(prior.bytes)
                    {
                        return Err("unsynchronized CUDA cohort writes alias".into());
                    }
                }
                memory.store(&a, value)?;
                event.accesses.push(a);
            }
        }
        Operation::Branch { target } => {
            lane.pc = Some(next(
                plan,
                *labels.get(&target).ok_or("unknown PTX branch label")?,
            )?);
            lane.control = event_id;
            advance = false;
        }
        Operation::Return => {
            lane.pc = None;
            advance = false;
        }
        Operation::Call { result, .. } => {
            let ready = body_id.ok_or("active call missing helper body")?;
            lane.parameters[result.0] = Some(Cell {
                value: Value::Unknown,
                writer: Some(ready),
            });
            lane.control = ready;
        }
    }
    if let Some(d) = destination {
        lane.registers[d.0] = Some(Cell {
            value,
            writer: Some(event_id),
        });
    }
    if advance {
        lane.pc = Some(next(plan, pc + 1)?);
    }
    if matches!(
        instruction.operation,
        Operation::Branch { .. } | Operation::Return
    ) {
        lane.control = event_id;
    }
    Ok(())
}

fn mask(t: DataType) -> u64 {
    if t.bits() == 64 {
        u64::MAX
    } else {
        (1u64 << t.bits()) - 1
    }
}
fn signed(v: u64, t: DataType) -> i64 {
    if t.bits() == 64 {
        v as i64
    } else {
        ((v << (64 - t.bits())) as i64) >> (64 - t.bits())
    }
}
fn is_signed(t: DataType) -> bool {
    matches!(t, DataType::S32 | DataType::S64)
}
fn truncate(v: Value, t: DataType) -> Value {
    match v {
        Value::Bits(v) => Value::Bits(v & mask(t)),
        p @ Value::Pointer(..) if t.bits() == 64 => p,
        _ => Value::Unknown,
    }
}
fn unary(op: Unary, t: DataType, a: Value) -> Result<Value, String> {
    if op == Unary::Move {
        return Ok(truncate(a, t));
    }
    let Value::Bits(v) = a else {
        return Ok(Value::Unknown);
    };
    Ok(match (op, t) {
        (Unary::Negate, DataType::F32) => Value::Bits(v ^ (1 << 31)),
        (Unary::Absolute, DataType::F32) => Value::Bits(v & 0x7fff_ffff),
        (Unary::Negate, _) => Value::Bits((0u64.wrapping_sub(v)) & mask(t)),
        (Unary::Absolute, _) => Value::Bits(signed(v, t).unsigned_abs() & mask(t)),
        (Unary::Sqrt, _) => Value::Unknown,
        (Unary::Move, _) => unreachable!(),
    })
}
fn binary(op: Binary, t: DataType, a: Value, b: Value) -> Result<Value, String> {
    if let (Value::Pointer(allocation, offset), Value::Bits(delta)) = (&a, &b) {
        if t.bits() == 64 && matches!(op, Binary::Add | Binary::Subtract) {
            let delta = *delta as i64;
            let offset = if op == Binary::Add {
                offset.checked_add_signed(delta)
            } else {
                delta
                    .checked_neg()
                    .and_then(|d| offset.checked_add_signed(d))
            }
            .ok_or("PTX symbolic pointer arithmetic overflow")?;
            return Ok(Value::Pointer(allocation.clone(), offset));
        }
    }
    if matches!(a, Value::Pointer(..)) || matches!(b, Value::Pointer(..)) {
        return Err("unsupported PTX pointer arithmetic".into());
    }
    let (Value::Bits(a), Value::Bits(b)) = (a, b) else {
        return Ok(Value::Unknown);
    };
    if t == DataType::F32 {
        return Ok(Value::Unknown);
    }
    let a = a & mask(t);
    let b = b & mask(t);
    let wide = matches!(op, Binary::Multiply(ptx::Multiply::Wide));
    let out = match op {
        Binary::Add => a.wrapping_add(b),
        Binary::Subtract => a.wrapping_sub(b),
        Binary::Multiply(_) => {
            if wide && is_signed(t) {
                (signed(a, t) as i128).wrapping_mul(signed(b, t) as i128) as u64
            } else {
                a.wrapping_mul(b)
            }
        }
        Binary::Divide | Binary::Remainder => {
            if b == 0 {
                return Err("undefined PTX integer division by zero".into());
            }
            if is_signed(t) {
                let a = signed(a, t);
                let b = signed(b, t);
                // PTX rem explicitly leaves negative-operand behavior machine
                // dependent. The selected hardware timing does not establish a
                // rounding rule, so do not substitute the host's signed `%`.
                // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#integer-arithmetic-instructions-rem
                if op == Binary::Remainder && (a < 0 || b < 0) {
                    return Ok(Value::Unknown);
                }
                if a == (-(1i128 << (t.bits() - 1))) as i64 && b == -1 {
                    return Err("overflowing PTX signed division".into());
                }
                if op == Binary::Divide {
                    (a / b) as u64
                } else {
                    (a % b) as u64
                }
            } else if op == Binary::Divide {
                a / b
            } else {
                a % b
            }
        }
        Binary::And => a & b,
        Binary::Or => a | b,
        Binary::Xor => a ^ b,
        Binary::ShiftLeft => {
            if b >= u64::from(t.bits()) {
                0
            } else {
                a << b
            }
        }
        Binary::ShiftRight => {
            if is_signed(t) {
                (signed(a, t) >> (b.min(u64::from(t.bits() - 1)))) as u64
            } else if b >= u64::from(t.bits()) {
                0
            } else {
                a >> b
            }
        }
        Binary::MinimumNaN | Binary::MaximumNaN => {
            return Err("invalid integer PTX floating minimum/maximum".into());
        }
    };
    Ok(Value::Bits(if wide { out } else { out & mask(t) }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn signed_negative_remainder_does_not_choose_a_machine_rounding_rule() {
        for ty in [DataType::S32, DataType::S64] {
            let minimum = -(1i128 << (ty.bits() - 1));
            for (a, b) in [(-7, 3), (7, -3), (-7, -3), (minimum, -1)] {
                let result = binary(
                    Binary::Remainder,
                    ty,
                    Value::Bits(a as u64),
                    Value::Bits(b as u64),
                )
                .unwrap();
                assert_eq!(result, Value::Unknown);
                let condition = compare(Comparison::Equal, ty, result, Value::Bits(0)).unwrap();
                assert!(condition.predicate().is_err());
            }
        }
    }

    #[test]
    fn nonnegative_and_unsigned_remainders_remain_exact() {
        for ty in [DataType::S32, DataType::S64, DataType::U32, DataType::U64] {
            assert_eq!(
                binary(Binary::Remainder, ty, Value::Bits(7), Value::Bits(3)).unwrap(),
                Value::Bits(1),
            );
        }
        assert_eq!(
            binary(
                Binary::Remainder,
                DataType::U64,
                Value::Bits(u64::MAX),
                Value::Bits(2),
            )
            .unwrap(),
            Value::Bits(1),
        );
    }
}
fn convert(to: DataType, from: DataType, value: Value) -> Result<Value, String> {
    let Value::Bits(value) = value else {
        return Ok(Value::Unknown);
    };
    if from == DataType::F32 || to == DataType::F32 {
        return Ok(Value::Unknown);
    }
    Ok(Value::Bits(
        (if is_signed(from) {
            signed(value, from) as u64
        } else {
            value & mask(from)
        }) & mask(to),
    ))
}
fn compare(c: Comparison, t: DataType, a: Value, b: Value) -> Result<Value, String> {
    let (Value::Bits(a), Value::Bits(b)) = (a, b) else {
        return Ok(Value::Unknown);
    };
    let (less, equal, greater, unordered) = if t == DataType::F32 {
        let a = f32::from_bits(a as u32);
        let b = f32::from_bits(b as u32);
        (a < b, a == b, a > b, a.is_nan() || b.is_nan())
    } else if is_signed(t) {
        let a = signed(a, t);
        let b = signed(b, t);
        (a < b, a == b, a > b, false)
    } else {
        let a = a & mask(t);
        let b = b & mask(t);
        (a < b, a == b, a > b, false)
    };
    let value = match c {
        Comparison::Equal => equal,
        Comparison::NotEqual => !equal && !unordered,
        Comparison::Less => less,
        Comparison::LessEqual => less || equal,
        Comparison::Greater => greater,
        Comparison::GreaterEqual => greater || equal,
        Comparison::Number => !unordered,
        Comparison::NaN => unordered,
        Comparison::NotEqualOrUnordered => !equal,
        Comparison::EqualOrUnordered => equal || unordered,
        Comparison::LessOrUnordered => less || unordered,
        Comparison::LessEqualOrUnordered => less || equal || unordered,
        Comparison::GreaterOrUnordered => greater || unordered,
        Comparison::GreaterEqualOrUnordered => greater || equal || unordered,
    };
    Ok(Value::Bits(u64::from(value)))
}
