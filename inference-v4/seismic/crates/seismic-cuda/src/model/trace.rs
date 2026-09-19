//! Abstract evaluation of terminal PTX, with symbolic allocation addresses.
//! Unknown data stays unknown, including checked varying integer inputs.
//! Runtime trace-shaping predicates and addresses must be established by the
//! workload domain. Compile-time predicates retain guarded terminal regions.
use super::*;
mod family;
pub(super) use family::{derive as derive_family, cohort};
use ptx::{
    Address, AddressBase, Binary, Comparison, DataType, Item, Operand, Operation, ParameterRole,
    Space, Unary,
};

pub(super) struct Trace {
    pub events: Vec<Event>,
    pub guards: Vec<Guard>,
    pub instructions: u64,
    pub external_values: BTreeMap<u64, BTreeMap<u64, u8>>,
    pub external_symbolic: SymbolicMemory,
}
pub(super) type Guard = BTreeMap<usize, bool>;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) enum Value {
    Bits(u64),
    Pointer(Allocation, Affine),
    Integer(Affine),
    Choice { parameter: usize, yes: Box<Value>, no: Box<Value> },
    Unknown,
}
impl Value {
    fn select(parameter: usize, yes: Value, no: Value) -> Value {
        if yes == no { yes } else { Self::Choice { parameter, yes: Box::new(yes), no: Box::new(no) } }
    }
    fn resolve(&self, guard: &Guard) -> Value {
        match self {
            Self::Choice { parameter, yes, no } => match guard.get(parameter) {
                Some(true) => yes.resolve(guard), Some(false) => no.resolve(guard),
                None => Self::select(*parameter, yes.resolve(guard), no.resolve(guard)),
            },
            value => value.clone(),
        }
    }
    fn bits(&self) -> Result<u64, DerivationError> {
        match self {
            Self::Bits(v) => Ok(*v),
            Self::Integer(value) => value.exact().map(|v| v as u64).ok_or_else(|| {
                DerivationError::Unsupported(
                    "CUDA trace needs a uniform integer or predicate".into(),
                )
            }),
            Self::Choice { .. } | Self::Unknown => Err(DerivationError::Unsupported(
                "CUDA trace needs an input-independent integer or predicate".into(),
            )),
            Self::Pointer(..) => {
                Err("PTX uses an allocation pointer as an integer or predicate".into())
            }
        }
    }
    fn predicate(&self) -> Result<bool, DerivationError> {
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
    writer: Vec<usize>,
}
#[derive(Clone)]
struct Lane {
    linear: u64,
    thread: u32,
    pc: Option<usize>,
    registers: Vec<Option<Cell>>,
    parameters: Vec<Option<Cell>>,
    control: Vec<usize>,
    status: Option<u32>,
}
pub(super) type SymbolicMemory = BTreeMap<u64, BTreeMap<(Affine, u32), Value>>;
#[derive(Clone)]
struct Storage {
    bytes: u64,
    alignment: u64,
    known: BTreeMap<u64, u8>,
    pointers: BTreeMap<u64, Value>,
    symbolic: BTreeMap<(Affine, u32), Value>,
}
#[derive(Clone)]
struct Memory {
    allocations: BTreeMap<Allocation, Storage>,
    accesses: Vec<(Access, usize, Guard)>,
}

impl Memory {
    fn new(
        execution: &Execution,
        workload: &ScalarWorkload,
        hardware: &CudaHardware,
        limits: DerivationLimits,
    ) -> Result<Self, DerivationError> {
        Self::from_program(execution.program(),execution.storage(),workload,hardware,limits)
    }
    fn from_program(
        program:&seismic_realization::ScalarProgram,
        storage:&crate::execution::InvocationStorage,
        workload:&ScalarWorkload,
        hardware:&CudaHardware,
        limits:DerivationLimits,
    )->Result<Self,DerivationError> {
        workload.validate()?;
        if workload.identity.is_empty() || workload.buffers.len() != program.buffers.len() {
            return Err("invalid CUDA workload shape".into());
        }
        program.conditions.validate_aliases(&program.buffers, |i| {
            let binding = &workload.buffers[i];
            (binding.allocation, binding.offset)
        })?;
        if workload.allocations.len() > limits.operations
            || workload.scalars.len() > limits.operations
        {
            return Err(DerivationError::Exhausted(DerivationLimit::Operations(
                limits.operations,
            )));
        }
        seismic_lang::abi::ScalarLayout::words(&program.scalars)?
            .validate_bytes(&workload.scalars)?;
        let mut allocations = BTreeMap::new();
        let mut known_count = 0usize;
        for a in &workload.allocations {
            known_count = known_count
                .checked_add(a.known_bytes.len())
                .ok_or("CUDA binding budget overflow")?;
            if known_count > limits.operations {
                return Err(DerivationError::Exhausted(DerivationLimit::Operations(
                    limits.operations,
                )));
            }
            if !a.alignment.is_power_of_two()
                || a.known_bytes.keys().any(|&i| i >= a.bytes)
                || allocations
                    .insert(
                        Allocation::External(a.id),
                        Storage {
                            bytes: a.bytes,
                            alignment: a.alignment,
                            known: a.known_bytes.clone(),
                            pointers: BTreeMap::new(),
                            symbolic: BTreeMap::new(),
                        },
                    )
                    .is_some()
            {
                return Err("invalid CUDA allocation identity, alignment or byte domain".into());
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
                Value::Pointer(
                    Allocation::External(binding.allocation),
                    binding.offset.into(),
                ),
            );
        }
        allocations.insert(
            Allocation::BufferTable,
            Storage {
                bytes: storage.buffer_table_bytes as u64,
                alignment: hardware.internal_alignment,
                known: BTreeMap::new(),
                pointers,
                symbolic: BTreeMap::new(),
            },
        );
        // The canonical bytes of a scalar domain encode its minimum only for
        // workload identity. They are not an exact input. Erase those bytes so
        // every permitted value follows the same traced instructions and memory
        // accesses; numerical data may vary, but unresolved control or addresses
        // cannot accidentally be modeled using the minimum as a representative.
        let mut scalar_known: BTreeMap<_, _> = workload
            .scalars
            .iter()
            .enumerate()
            .map(|(i, &value)| (i as u64, value))
            .collect();
        for domain in &workload.integer_domains {
            if let seismic_accounting::workload::IntegerInput::Scalar { slot } = domain.input {
                let offset = u64::try_from(slot)
                    .map_err(|_| "CUDA scalar domain slot overflow")?
                    .checked_mul(8)
                    .ok_or("CUDA scalar domain offset overflow")?;
                for byte in 0..u64::from(domain.bytes) {
                    scalar_known.remove(&(offset + byte));
                }
            }
        }
        allocations.insert(
            Allocation::Scalars,
            Storage {
                bytes: workload.scalars.len() as u64,
                alignment: hardware.internal_alignment,
                known: scalar_known,
                pointers: BTreeMap::new(),
                symbolic: BTreeMap::new(),
            },
        );
        allocations.insert(
            Allocation::Scratch,
            Storage {
                bytes: storage.scratch_bytes as u64,
                alignment: hardware.internal_alignment,
                known: BTreeMap::new(),
                pointers: BTreeMap::new(),
                symbolic: BTreeMap::new(),
            },
        );
        allocations.insert(
            Allocation::Statuses,
            Storage {
                bytes: storage.status_bytes as u64,
                alignment: hardware.internal_alignment,
                known: BTreeMap::new(),
                pointers: BTreeMap::new(),
                symbolic: BTreeMap::new(),
            },
        );
        for domain in &workload.integer_domains {
            let (allocation, offset) = match domain.input {
                seismic_accounting::workload::IntegerInput::Scalar { slot } => {
                    (Allocation::Scalars, (slot as u64) * 8)
                }
                seismic_accounting::workload::IntegerInput::Allocation { allocation, offset } => {
                    (Allocation::External(allocation), offset)
                }
            };
            let expression = Affine::domain(domain.input, domain.range)
                .ok_or("CUDA integer domain expression overflow")?;
            allocations
                .get_mut(&allocation)
                .ok_or("CUDA integer domain storage is missing")?
                .symbolic
                .insert(
                    (offset.into(), u32::from(domain.bytes)),
                    Value::Integer(expression),
                );
        }
        Ok(Self {
            allocations,
            accesses: Vec::new(),
        })
    }
    fn access(
        &self,
        value: &Value,
        bytes: u32,
        lane: u64,
        write: bool,
    ) -> Result<Access, DerivationError> {
        let Value::Pointer(allocation, offset) = value else {
            return Err("CUDA memory address is not an allocation-relative pointer".into());
        };
        let storage = self
            .allocations
            .get(allocation)
            .ok_or("unknown CUDA storage")?;
        let (lo, hi) = offset.bounds().ok_or("CUDA affine address overflow")?;
        if bytes == 0
            || lo < 0
            || hi
                .checked_add(i128::from(bytes))
                .is_none_or(|end| end > i128::from(storage.bytes))
            || !offset.aligned(u64::from(bytes))
            || storage.alignment < u64::from(bytes)
        {
            return Err(DerivationError::Unsupported(
                "CUDA input domain does not establish in-bounds aligned memory accesses".into(),
            ));
        }
        if write && matches!(allocation, Allocation::BufferTable | Allocation::Scalars) {
            return Err("CUDA kernel writes immutable ABI input storage".into());
        }
        Ok(Access {
            lane,
            allocation: allocation.clone(),
            offset: offset.clone(),
            bytes,
            alignment: storage.alignment,
            write,
        })
    }
    fn dependencies(&self, access: &Access, guard: &Guard) -> Result<Vec<usize>, DerivationError> {
        let mut dependencies = Vec::new();
        for (prior, event, before) in &self.accesses {
            if before.iter().any(|(parameter, value)| guard.get(parameter).is_some_and(|other| other != value)) { continue; }
            if access.allocation != prior.allocation || !(access.write || prior.write) {
                continue;
            }
            match access.offset.disjoint(
                u64::from(access.bytes),
                &prior.offset,
                u64::from(prior.bytes),
            ) {
                Some(true) => {}
                Some(false) if prior.lane == access.lane => dependencies.push(*event),
                Some(false) => {
                    return Err("unsynchronized cross-lane conflicting CUDA memory access".into());
                }
                None => return Err(DerivationError::Unsupported(
                    "CUDA memory dependence or cross-lane conflict varies across the input domain"
                        .into(),
                )),
            }
        }
        Ok(dependencies)
    }
    fn load(&self, access: &Access) -> Value {
        let storage = &self.allocations[&access.allocation];
        if let Some(value) = storage.symbolic.get(&(access.offset.clone(), access.bytes)) {
            return value.clone();
        }
        let Some(offset) = access.offset.exact().and_then(|n| u64::try_from(n).ok()) else {
            return Value::Unknown;
        };
        if access.bytes == 8 {
            if let Some(pointer) = storage.pointers.get(&offset) {
                return pointer.clone();
            }
        }
        let mut value = 0u64;
        for byte in 0..access.bytes {
            let Some(&b) = storage.known.get(&(offset + u64::from(byte))) else {
                return Value::Unknown;
            };
            value |= u64::from(b) << (byte * 8);
        }
        Value::Bits(value)
    }
    fn store(&mut self, access: &Access, value: Value) -> Result<(), DerivationError> {
        let storage = self.allocations.get_mut(&access.allocation).unwrap();
        // A varying write invalidates every potentially overlapping fact. A
        // retained exact symbolic address/value can then be read through that
        // same address; it is not installed as known bytes at a sampled offset.
        storage.pointers.retain(|&offset, _| {
            access
                .offset
                .disjoint(u64::from(access.bytes), &offset.into(), 8)
                == Some(true)
        });
        storage.known.retain(|&offset, _| {
            access
                .offset
                .disjoint(u64::from(access.bytes), &offset.into(), 1)
                == Some(true)
        });
        storage.symbolic.retain(|(offset, bytes), _| {
            access
                .offset
                .disjoint(u64::from(access.bytes), offset, u64::from(*bytes))
                == Some(true)
        });
        if let (Some(offset), Value::Bits(value)) = (
            access.offset.exact().and_then(|n| u64::try_from(n).ok()),
            &value,
        ) {
            for byte in 0..access.bytes {
                storage.known.insert(
                    offset + u64::from(byte),
                    ((value >> (8 * byte)) & 255) as u8,
                );
            }
        }
        if !matches!(value, Value::Unknown) {
            storage
                .symbolic
                .insert((access.offset.clone(), access.bytes), value);
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
) -> Result<usize, DerivationError> {
    if events.len() >= limits.operations {
        return Err(DerivationError::Exhausted(DerivationLimit::Operations(
            limits.operations,
        )));
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
    inherited: &SymbolicMemory,
) -> Result<Trace, DerivationError> {
    derive_family(execution, hardware, workload, limits, inherited, &BTreeMap::new(), &BTreeMap::new())
}

fn operand(op: Operand, lane: &Lane, block: u64, width: u64) -> Result<Value, DerivationError> {
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
            ptx::SpecialRegister::LaneIndex => u64::from(lane.thread % 32),
            ptx::SpecialRegister::BlockIndexX => block,
            ptx::SpecialRegister::BlockWidthX => width,
            ptx::SpecialRegister::ThreadIndexX => u64::from(lane.thread),
        }),
    })
}
fn address(a: Address, lane: &Lane) -> Result<Value, DerivationError> {
    let AddressBase::Register(r) = a.base else {
        return Err("parameter address used as a global pointer".into());
    };
    let value = &lane.registers[r.0]
        .as_ref()
        .ok_or("undefined address register")?
        .value;
    let (allocation, offset) = match value {
        Value::Pointer(allocation, offset) => (allocation, offset),
        Value::Choice { .. } | Value::Unknown => {
            return Err(DerivationError::Unsupported(
                "CUDA trace needs an input-independent memory address".into(),
            ));
        }
        Value::Bits(_) | Value::Integer(_) => {
            return Err("PTX global address has no allocation provenance".into());
        }
    };
    let offset = offset
        .add(&Affine::constant(i128::from(a.offset)))
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
        offset: 0u64.into(),
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
    guard: &Guard,
) -> Result<(), DerivationError> {
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
            let predicate = &lane.registers[predicate.0].as_ref().ok_or("undefined select predicate")?.value;
            value = select(predicate, operand(when_true,lane,block,width)?, operand(when_false,lane,block,width)?)?;
        }
        Operation::Shuffle { .. } => return Err("shuffle needs participant-wide evaluation".into()),
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
                event.predecessors.extend(memory.dependencies(&a, guard)?);
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
                    writer: vec![event_id],
                });
                event
                    .accesses
                    .push(parameter_access(lane, p, data_type.bits() / 8, true));
            } else {
                let a =
                    memory.access(&address(at, lane)?, data_type.bits() / 8, lane.linear, true)?;
                if a.allocation == Allocation::Statuses {
                    if a.bytes != 4
                        || a.offset
                            != lane
                                .linear
                                .checked_mul(4)
                                .ok_or("status index overflow")?
                                .into()
                    {
                        return Err("CUDA lane writes another invocation status".into());
                    }
                    lane.status =
                        Some(u32::try_from(value.bits()?).map_err(|_| "invalid CUDA status")?);
                }
                event.predecessors.extend(memory.dependencies(&a, guard)?);
                // Also check other lanes of this same instruction before mutation.
                for prior in &event.accesses {
                    if prior.lane == a.lane || prior.allocation != a.allocation {
                        continue;
                    }
                    match prior.offset.disjoint(
                        u64::from(prior.bytes),
                        &a.offset,
                        u64::from(a.bytes),
                    ) {
                        Some(true) => {}
                        Some(false) => return Err("unsynchronized CUDA cohort writes alias".into()),
                        None => {
                            return Err(DerivationError::Unsupported(
                                "CUDA cohort write conflicts vary across the input domain".into(),
                            ));
                        }
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
            lane.control = vec![event_id];
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
                writer: vec![ready],
            });
            lane.control = vec![ready];
        }
    }
    if let Some(d) = destination {
        lane.registers[d.0] = Some(Cell {
            value,
            writer: vec![event_id],
        });
    }
    if advance {
        lane.pc = Some(next(plan, pc + 1)?);
    }
    if matches!(
        instruction.operation,
        Operation::Branch { .. } | Operation::Return
    ) {
        lane.control = vec![event_id];
    }
    Ok(())
}

fn select(predicate: &Value, yes: Value, no: Value) -> Result<Value, DerivationError> {
    if yes==no {return Ok(yes);}
    match predicate {
        Value::Unknown=>Ok(Value::Unknown),
        Value::Choice {parameter,yes:a,no:b}=>{
            let yes_guard=Guard::from([(*parameter,true)]);
            let no_guard=Guard::from([(*parameter,false)]);
            Ok(Value::select(*parameter,select(a,yes.resolve(&yes_guard),no.resolve(&yes_guard))?,select(b,yes.resolve(&no_guard),no.resolve(&no_guard))?))
        },
        value=>Ok(if value.predicate()? {yes} else {no}),
    }
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
    if let Value::Choice { parameter, yes, no } = v {
        return Value::select(parameter, truncate(*yes, t), truncate(*no, t));
    }
    match v {
        Value::Bits(v) => Value::Bits(v & mask(t)),
        p @ Value::Pointer(..) if t.bits() == 64 => p,
        Value::Integer(expression) if t != DataType::F32 => {
            affine_value(expression.interpreted(t.bits(), is_signed(t)))
        }
        _ => Value::Unknown,
    }
}
fn affine_value(expression: Option<Affine>) -> Value {
    match expression {
        Some(expression) => expression
            .exact()
            .map_or(Value::Integer(expression), |n| Value::Bits(n as u64)),
        None => Value::Unknown,
    }
}
fn integer(value: &Value, ty: DataType) -> Option<Affine> {
    if ty == DataType::F32 {
        return None;
    }
    match value {
        Value::Bits(bits) => Some(Affine::constant(if is_signed(ty) {
            i128::from(signed(*bits, ty))
        } else {
            i128::from(*bits & mask(ty))
        })),
        Value::Integer(expression) => expression.interpreted(ty.bits(), is_signed(ty)),
        _ => None,
    }
}
fn symbolic_binary(op: Binary, t: DataType, a: &Value, b: &Value) -> Option<Affine> {
    let (a, b) = (integer(a, t)?, integer(b, t)?);
    let result = match op {
        Binary::Add => a.add(&b),
        Binary::Subtract => a.sub(&b),
        Binary::Multiply(_) => {
            if let Some(n) = a.exact() {
                b.scale(n)
            } else {
                a.scale(b.exact()?)
            }
        }
        Binary::Divide => {
            let divisor = u64::try_from(b.exact()?).ok()?;
            // Signed truncation agrees with floor for nonnegative values, or
            // an exactly divisible affine expression.
            let quotient = a.quotient(divisor)?;
            if is_signed(t) && a.bounds()?.0 < 0 && a.remainder(divisor)?.exact() != Some(0) {
                return None;
            }
            Some(quotient)
        }
        Binary::Remainder => {
            if a.bounds()?.0 < 0 {
                return None;
            }
            a.remainder(u64::try_from(b.exact()?).ok()?)
        }
        Binary::ShiftLeft => {
            let shift = u32::try_from(b.exact()?).ok()?;
            if shift >= t.bits() {
                Some(Affine::constant(0))
            } else {
                a.scale(1i128.checked_shl(shift)?)
            }
        }
        Binary::ShiftRight => {
            let shift = u32::try_from(b.exact()?).ok()?;
            if shift >= t.bits() && !is_signed(t) {
                Some(Affine::constant(0))
            } else {
                a.quotient(1u64.checked_shl(shift.min(t.bits() - 1))?)
            }
        }
        Binary::And => {
            let mask = b.exact().or_else(|| a.exact())?;
            let other = if b.exact().is_some() { &a } else { &b };
            let divisor = u64::try_from(mask.checked_add(1)?).ok()?;
            if !divisor.is_power_of_two() {
                return None;
            }
            other.remainder(divisor)
        }
        Binary::Or if a.exact() == Some(0) => Some(b),
        Binary::Or if b.exact() == Some(0) || a == b => Some(a),
        Binary::Xor if a == b => Some(Affine::constant(0)),
        Binary::Xor if a.exact() == Some(0) => Some(b),
        Binary::Xor if b.exact() == Some(0) => Some(a),
        _ => None,
    }?;
    let bits = if matches!(op, Binary::Multiply(ptx::Multiply::Wide)) {
        t.bits().checked_mul(2)?
    } else {
        t.bits()
    };
    result.interpreted(bits, is_signed(t))
}
fn unary(op: Unary, t: DataType, a: Value) -> Result<Value, DerivationError> {
    if let Value::Choice { parameter, yes, no } = a {
        return Ok(Value::select(parameter, unary(op,t,*yes)?, unary(op,t,*no)?));
    }
    if op == Unary::Move {
        return Ok(truncate(a, t));
    }
    if matches!(a, Value::Integer(_)) {
        let result = integer(&a, t)
            .and_then(|a| match op {
                Unary::Negate => a.scale(-1),
                Unary::Absolute => {
                    let (lo, hi) = a.bounds()?;
                    if lo >= 0 {
                        Some(a)
                    } else if hi <= 0 {
                        a.scale(-1)
                    } else {
                        None
                    }
                }
                _ => None,
            })
            .and_then(|a| a.interpreted(t.bits(), is_signed(t)));
        return Ok(affine_value(result));
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
fn binary(op: Binary, t: DataType, a: Value, b: Value) -> Result<Value, DerivationError> {
    if let Value::Choice { parameter, yes, no } = a {
        let mut yes_guard = Guard::new(); yes_guard.insert(parameter, true);
        let mut no_guard = Guard::new(); no_guard.insert(parameter, false);
        return Ok(Value::select(parameter, binary(op,t,*yes,b.resolve(&yes_guard))?, binary(op,t,*no,b.resolve(&no_guard))?));
    }
    if let Value::Choice { parameter, yes, no } = b {
        return Ok(Value::select(parameter, binary(op,t,a.clone(),*yes)?, binary(op,t,a,*no)?));
    }
    if let Value::Pointer(allocation, offset) = &a {
        if t.bits() == 64 && matches!(op, Binary::Add | Binary::Subtract) {
            let delta = integer(&b, DataType::S64).ok_or_else(|| {
                DerivationError::Unsupported(
                    "CUDA pointer arithmetic needs a representable input domain".into(),
                )
            })?;
            let address = if op == Binary::Add {
                offset.add(&delta)
            } else {
                offset.sub(&delta)
            }
            .ok_or("CUDA affine pointer arithmetic overflow")?;
            return Ok(Value::Pointer(allocation.clone(), address));
        }
    }
    if matches!(a, Value::Pointer(..)) || matches!(b, Value::Pointer(..)) {
        return Err("unsupported PTX pointer arithmetic".into());
    }
    if matches!(a, Value::Integer(_)) || matches!(b, Value::Integer(_)) {
        return Ok(affine_value(symbolic_binary(op, t, &a, &b)));
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
fn convert(to: DataType, from: DataType, value: Value) -> Result<Value, DerivationError> {
    if let Value::Choice { parameter, yes, no } = value {
        return Ok(Value::select(parameter, convert(to,from,*yes)?, convert(to,from,*no)?));
    }
    if matches!(value, Value::Integer(_)) {
        return Ok(affine_value(integer(&value, from).and_then(|v| {
            if to == DataType::F32 {
                None
            } else {
                v.interpreted(to.bits(), is_signed(to))
            }
        })));
    }
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
fn compare(c: Comparison, t: DataType, a: Value, b: Value) -> Result<Value, DerivationError> {
    if let Value::Choice { parameter, yes, no } = a {
        let mut yes_guard = Guard::new(); yes_guard.insert(parameter, true);
        let mut no_guard = Guard::new(); no_guard.insert(parameter, false);
        return Ok(Value::select(parameter, compare(c,t,*yes,b.resolve(&yes_guard))?, compare(c,t,*no,b.resolve(&no_guard))?));
    }
    if let Value::Choice { parameter, yes, no } = b {
        return Ok(Value::select(parameter, compare(c,t,a.clone(),*yes)?, compare(c,t,a,*no)?));
    }
    if matches!(a, Value::Integer(_)) || matches!(b, Value::Integer(_)) {
        let result = (|| {
            let (lo, hi) = integer(&a, t)?.sub(&integer(&b, t)?)?.bounds()?;
            let equal = if lo == 0 && hi == 0 {
                Some(true)
            } else if hi < 0 || lo > 0 {
                Some(false)
            } else {
                None
            };
            let less = if hi < 0 {
                Some(true)
            } else if lo >= 0 {
                Some(false)
            } else {
                None
            };
            let greater = if lo > 0 {
                Some(true)
            } else if hi <= 0 {
                Some(false)
            } else {
                None
            };
            match c {
                Comparison::Equal | Comparison::EqualOrUnordered => equal,
                Comparison::NotEqual | Comparison::NotEqualOrUnordered => equal.map(|x| !x),
                Comparison::Less | Comparison::LessOrUnordered => less,
                Comparison::LessEqual | Comparison::LessEqualOrUnordered => greater.map(|x| !x),
                Comparison::Greater | Comparison::GreaterOrUnordered => greater,
                Comparison::GreaterEqual | Comparison::GreaterEqualOrUnordered => less.map(|x| !x),
                Comparison::Number => Some(true),
                Comparison::NaN => Some(false),
            }
        })();
        return Ok(result.map_or(Value::Unknown, |b| Value::Bits(u64::from(b))));
    }
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
