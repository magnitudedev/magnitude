//! Exact derivation for the **direct scalar execution form**.
//!
//! This is a conditional discrete-time model, not a hardware theorem. Its form
//! executes the prepared SSA instructions without speculation, elimination,
//! fusion, or register allocation. Every operation and dependency is derived from
//! that execution. External inputs describe hardware instruction timing and shared
//! service capacity; they cannot introduce execution steps or helper bodies.
//!
//! Unlike multiplying block counts, instantiating control flow preserves phi
//! transfers, loop-carried values, memory versions, and the selected branch. The
//! current finite-instance derivation is bounded by an explicit caller budget;
//! it never truncates an execution and calls the truncated graph complete.
use crate::schedule::{Model, Operation, Resource, Timebase};
use crate::workload::{DerivationError, DerivationLimit, DerivationLimits, ScalarWorkload};
use cranelift_codegen::ir::{
    self, Block, BlockCall, Inst, InstructionData as Data, Opcode, Type, Value,
    condcodes::{FloatCC, IntCC},
    types,
};
use seismic_realization::{
    Dispatch, MathFunction, ScalarProgram,
    execution::MemoryObject,
    graph::{Graph, Instruction},
};
use std::collections::{BTreeMap, BTreeSet, HashMap};

/// Typed primitive identity. Immediates and predicates participate in the
/// identity because their implementations need not have identical resource use.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Primitive {
    pub kind: PrimitiveKind,
    pub inputs: Vec<Type>,
    pub outputs: Vec<Type>,
    pub memory: Option<MemoryClass>,
    pub modifier: Modifier,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MemoryClass {
    Tensor,
    PrivateScratch,
    BufferTable,
    ScalarArguments,
}

/// Reusable hardware timing signature. Facts apply to an opcode/type/storage
/// family, not separately to every kernel's constants.
/// Exact encoding specializations are explicit and must form disjoint domains.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PrimitivePattern {
    pub kind: PrimitiveKind,
    pub inputs: Vec<Type>,
    pub outputs: Vec<Type>,
    pub memory: Option<MemoryClass>,
    pub modifier: ModifierPattern,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ModifierPattern {
    Any,
    Exact(Modifier),
}
impl Primitive {
    pub fn signature(&self) -> PrimitivePattern {
        PrimitivePattern {
            kind: self.kind.clone(),
            inputs: self.inputs.clone(),
            outputs: self.outputs.clone(),
            memory: self.memory.clone(),
            modifier: ModifierPattern::Any,
        }
    }
}
impl PrimitivePattern {
    fn matches(&self, primitive: &Primitive) -> bool {
        self.kind == primitive.kind
            && self.inputs == primitive.inputs
            && self.outputs == primitive.outputs
            && self.memory == primitive.memory
            && match &self.modifier {
                ModifierPattern::Any => true,
                ModifierPattern::Exact(value) => value == &primitive.modifier,
            }
    }
    fn overlaps(&self, other: &Self) -> bool {
        self.kind == other.kind
            && self.inputs == other.inputs
            && self.outputs == other.outputs
            && self.memory == other.memory
            && match (&self.modifier, &other.modifier) {
                (ModifierPattern::Exact(a), ModifierPattern::Exact(b)) => a == b,
                _ => true,
            }
    }
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PrimitiveKind {
    Instruction(Opcode),
    Math(MathFunction),
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Modifier {
    None,
    Integer(i64),
    Float32(u32),
    IntCompare(IntCC),
    IntCompareImmediate(IntCC, i64),
    FloatCompare(FloatCC),
    Memory { bytes: u32, offset: i32 },
}

/// Hardware timing/service facts for one actual typed instruction. This cannot
/// describe a replacement algorithm, sequence, dependency graph, or helper body.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PrimitiveTiming {
    pub primitive: PrimitivePattern,
    pub latency: u64,
    pub services: Vec<crate::schedule::Reservation>,
}

/// Hardware facts and explicit instruction-preserving assumptions. All program
/// work, ordering, addresses and multiplicities come from ScalarProgram itself.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ScalarHardware {
    pub identity: String,
    pub scope: Scope,
    pub timebase: Timebase,
    pub resources: Vec<Resource>,
    pub timings: Vec<PrimitiveTiming>,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Scope {
    /// Conditional instruction-preserving scalar form. This variant does not
    /// assert admission to any physical device or downstream native compiler.
    HypotheticalDirectScalarV1,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Origin {
    Instruction {
        invocation: u64,
        instruction: Inst,
        occurrence: u64,
    },
    EdgeTransfer {
        invocation: u64,
        terminator: Inst,
        occurrence: u64,
        destination: Block,
    },
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Access {
    pub invocation: u64,
    pub instruction: Inst,
    pub occurrence: u64,
    pub allocation: AllocationIdentity,
    pub offset: u64,
    pub bytes: u32,
    pub write: bool,
    pub completion: usize,
}
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum AllocationIdentity {
    External(u64),
    Scratch(u64),
    BufferTable,
    Scalars,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DerivedModel {
    pub model: Model,
    /// Source order used to derive instance identities. The selected order comes
    /// from `schedule::static_order::orders` and must be materialized before emission.
    pub order: seismic_realization::scheduling::Order,
    pub origins: Vec<Origin>,
    pub accesses: Vec<Access>,
    pub instructions: u64,
    /// Actual static primitives, including untaken paths. Missing hardware timing
    /// remains explicit in the analysis instead of supplying invented work.
    pub requirements: Vec<Primitive>,
}

/// Exhaustive classification of the scalar compiler's admitted vocabulary. New
/// SSA operations must be deliberately admitted here; opaque calls/effects and
/// unfamiliar opcodes cannot silently receive zero resource use.
pub fn requirements(program: &ScalarProgram) -> Result<Vec<Primitive>, String> {
    cranelift_codegen::verify_function(
        &program.function,
        &cranelift_codegen::settings::Flags::new(cranelift_codegen::settings::builder()),
    )
    .map_err(|error| format!("invalid scalar execution: {error}"))?;
    let graph = Graph::scalar(program);
    if !graph.unavailable.is_empty() {
        return Err(format!(
            "scalar model has opaque effects: {:?}",
            graph.unavailable
        ));
    }
    let mut result = Vec::new();
    for instruction in &graph.instructions {
        let p = primitive(instruction)?;
        if !result.contains(&p) {
            result.push(p);
        }
    }
    Ok(result)
}

fn primitive(instruction: &Instruction) -> Result<Primitive, String> {
    use Opcode as O;
    let op = instruction.opcode;
    match op {
        O::Iconst
        | O::F32const
        | O::Iadd
        | O::IaddImm
        | O::Isub
        | O::Imul
        | O::ImulImm
        | O::Udiv
        | O::UdivImm
        | O::Urem
        | O::UremImm
        | O::Sdiv
        | O::Srem
        | O::Band
        | O::BandImm
        | O::Bor
        | O::BorImm
        | O::Bxor
        | O::BxorImm
        | O::Bnot
        | O::Ishl
        | O::IshlImm
        | O::Ushr
        | O::UshrImm
        | O::Sshr
        | O::SshrImm
        | O::Ineg
        | O::Icmp
        | O::IcmpImm
        | O::Fcmp
        | O::Select
        | O::Sextend
        | O::Uextend
        | O::Ireduce
        | O::Bitcast
        | O::Fadd
        | O::Fsub
        | O::Fmul
        | O::Fdiv
        | O::Fma
        | O::Fmin
        | O::Fmax
        | O::Fneg
        | O::Fabs
        | O::Sqrt
        | O::FcvtFromSint
        | O::FcvtFromUint
        | O::FcvtToSintSat
        | O::FcvtToUintSat
        | O::Load
        | O::Store
        | O::Jump
        | O::Brif
        | O::Return => {}
        O::Call if instruction.primitive.is_some() => {}
        _ => {
            return Err(format!(
                "{}: scalar primitive {op} is not admitted",
                instruction.id
            ));
        }
    }
    let inputs = if op == O::Jump {
        Vec::new()
    } else if op == O::Brif {
        instruction.inputs.iter().take(1).map(|v| v.ty).collect()
    } else {
        instruction.inputs.iter().map(|v| v.ty).collect()
    };
    let outputs: Vec<_> = instruction.outputs.iter().map(|(_, ty)| *ty).collect();
    for ty in inputs.iter().chain(&outputs) {
        if !matches!(
            *ty,
            types::I8 | types::I16 | types::I32 | types::I64 | types::F32
        ) {
            return Err(format!(
                "{}: scalar model does not admit {ty}",
                instruction.id
            ));
        }
    }
    let modifier = match instruction.encoding {
        Data::UnaryImm { imm, .. } | Data::BinaryImm64 { imm, .. } => Modifier::Integer(imm.bits()),
        Data::UnaryIeee32 { imm, .. } => Modifier::Float32(imm.bits()),
        Data::IntCompare { cond, .. } => Modifier::IntCompare(cond),
        Data::IntCompareImm { cond, imm, .. } => Modifier::IntCompareImmediate(cond, imm.bits()),
        Data::FloatCompare { cond, .. } => Modifier::FloatCompare(cond),
        _ => Modifier::None,
    };
    let modifier = if let Some(access) = &instruction.memory {
        Modifier::Memory {
            bytes: access.bytes,
            offset: access.offset,
        }
    } else {
        modifier
    };
    let memory = instruction
        .memory
        .as_ref()
        .map(|access| match access.object.as_ref() {
            Some(MemoryObject::Buffer { .. }) => Ok(MemoryClass::Tensor),
            Some(MemoryObject::PrivateScratch) => Ok(MemoryClass::PrivateScratch),
            Some(MemoryObject::BufferTable) => Ok(MemoryClass::BufferTable),
            Some(MemoryObject::ScalarArguments) => Ok(MemoryClass::ScalarArguments),
            None => Err(format!(
                "{}: storage class of scalar access is unresolved",
                instruction.id
            )),
        })
        .transpose()?;
    Ok(Primitive {
        kind: instruction
            .primitive
            .map(PrimitiveKind::Math)
            .unwrap_or(PrimitiveKind::Instruction(op)),
        inputs,
        outputs,
        memory,
        modifier,
    })
}

impl ScalarHardware {
    pub fn validate(&self) -> Result<(), String> {
        if self.identity.is_empty()
            || self.timebase.seconds_numerator == 0
            || self.timebase.seconds_denominator == 0
        {
            return Err("scalar hardware requires an identity and positive exact timebase".into());
        }
        let mut names = BTreeSet::new();
        for resource in &self.resources {
            if resource.name.is_empty() || resource.capacity == 0 || !names.insert(&resource.name) {
                return Err("scalar resources require unique names and positive capacities".into());
            }
        }
        for (i, timing) in self.timings.iter().enumerate() {
            if !matches!(timing.primitive.kind, PrimitiveKind::Instruction(_)) {
                return Err(
                    "a helper call cannot be supplied as a hardware instruction timing".into(),
                );
            }
            if self.timings[..i]
                .iter()
                .any(|p| p.primitive.overlaps(&timing.primitive))
            {
                return Err("overlapping scalar hardware timing domains".into());
            }
            if timing.services.is_empty() {
                return Err("instruction timing has no hardware service fact".into());
            }
            let mut events = BTreeMap::<usize, Vec<(u64, bool, u64)>>::new();
            for service in &timing.services {
                let resource = self
                    .resources
                    .get(service.resource)
                    .ok_or("unknown hardware service resource")?;
                let end = service
                    .offset
                    .checked_add(service.duration)
                    .ok_or("hardware service interval overflow")?;
                if service.units == 0
                    || service.duration == 0
                    || service.units > resource.capacity
                    || end > timing.latency
                {
                    return Err("invalid scalar hardware service interval".into());
                }
                events.entry(service.resource).or_default().extend([
                    (service.offset, true, service.units),
                    (end, false, service.units),
                ]);
            }
            for (resource, mut events) in events {
                events.sort_unstable(); // half-open intervals release before acquire
                let mut occupied = 0u64;
                for (_, acquire, units) in events {
                    occupied = if acquire {
                        occupied
                            .checked_add(units)
                            .ok_or("hardware service demand overflow")?
                    } else {
                        occupied
                            .checked_sub(units)
                            .ok_or("invalid hardware service release")?
                    };
                    if occupied > self.resources[resource].capacity {
                        return Err("one instruction exceeds hardware service capacity".into());
                    }
                }
            }
        }
        Ok(())
    }
    fn timing(&self, primitive: &Primitive) -> Option<&PrimitiveTiming> {
        self.timings.iter().find(|i| i.primitive.matches(primitive))
    }
}

pub fn derive_scalar(
    program: &ScalarProgram,
    hardware: &ScalarHardware,
    workload: &ScalarWorkload,
    limits: DerivationLimits,
) -> Result<DerivedModel, DerivationError> {
    let required = requirements(program)?;
    hardware.validate()?;
    let memory = Memory::new(program, workload)?;
    let graph = Graph::scalar(program);
    let static_orders = seismic_realization::scheduling::Space::new(program)?
        .constraints()
        .into_iter()
        .map(|block| crate::schedule::static_order::Constraint {
            block: block.block,
            instructions: block.instructions,
            predecessors: block.predecessors,
            visits: Vec::new(),
        })
        .collect();
    let mut derivation = Derivation {
        program,
        graph: &graph,
        hardware,
        limits,
        memory,
        output: DerivedModel {
            order: seismic_realization::scheduling::Order::current(program),
            model: Model {
                relationship: crate::authority::ModelRelationship::hypothetical_execution(),
                identity: format!(
                    "direct-scalar-v1/{}/{}",
                    hardware.identity, workload.identity
                ),
                timebase: hardware.timebase.clone(),
                resources: hardware.resources.clone(),
                operations: Vec::new(),
                lifetimes: Vec::new(),
                static_orders,
                unmapped: Vec::new(),
            },
            origins: Vec::new(),
            accesses: Vec::new(),
            instructions: 0,
            requirements: required,
        },
    };
    if program.dispatch == Dispatch::Sequential && program.work_items != 1 {
        return Err("sequential scalar ABI requires exactly one invocation".into());
    }
    for invocation in 0..program.work_items {
        derivation.invocation(invocation)?;
    }
    Ok(derivation.output)
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum Datum {
    Bits(u64),
    Pointer {
        allocation: AllocationIdentity,
        offset: i64,
    },
    Unknown,
}
#[derive(Clone, Debug)]
struct Binding {
    datum: Datum,
    ready: Vec<usize>,
}
struct Memory {
    capacities: BTreeMap<AllocationIdentity, u64>,
    bytes: BTreeMap<(AllocationIdentity, u64), Option<u8>>,
    pointers: Vec<Datum>,
    scratch_bytes: u64,
}
impl Memory {
    fn new(program: &ScalarProgram, workload: &ScalarWorkload) -> Result<Self, String> {
        workload.validate()?;
        if !workload.integer_domains.is_empty() {
            return Err("scalar accounting does not yet support varying integer input domains".into());
        }
        if workload.identity.is_empty() {
            return Err("scalar workload requires an identity".into());
        }
        seismic_lang::abi::ScalarLayout::words(&program.scalars)?
            .validate_bytes(&workload.scalars)?;
        if workload.buffers.len() != program.buffers.len() {
            return Err("scalar workload buffer count mismatch".into());
        }
        program.conditions.validate_aliases(&program.buffers, |i| {
            let binding = &workload.buffers[i];
            (binding.allocation, binding.offset)
        })?;
        let mut capacities = BTreeMap::new();
        let mut bytes = BTreeMap::new();
        for allocation in &workload.allocations {
            let id = AllocationIdentity::External(allocation.id);
            if !allocation.alignment.is_power_of_two()
                || capacities.insert(id.clone(), allocation.bytes).is_some()
            {
                return Err(
                    "workload allocations require unique identities and power-of-two alignment"
                        .into(),
                );
            }
            for (&offset, &byte) in &allocation.known_bytes {
                if offset >= allocation.bytes {
                    return Err("known workload byte exceeds allocation".into());
                }
                bytes.insert((id.clone(), offset), Some(byte));
            }
        }
        let mut pointers = Vec::new();
        for (spec, binding) in program.buffers.iter().zip(&workload.buffers) {
            let allocation = workload
                .allocations
                .iter()
                .find(|a| a.id == binding.allocation)
                .ok_or("workload buffer names an absent allocation")?;
            let width = u64::try_from(spec.bytes).map_err(|_| "buffer size exceeds model range")?;
            if binding.bytes < width
                || binding
                    .offset
                    .checked_add(binding.bytes)
                    .is_none_or(|end| end > allocation.bytes)
                || spec.alignment == 0
                || allocation.alignment < spec.alignment as u64
                || !binding.offset.is_multiple_of(spec.alignment as u64)
            {
                return Err("workload buffer violates its allocation size or alignment".into());
            }
            pointers.push(Datum::Pointer {
                allocation: AllocationIdentity::External(binding.allocation),
                offset: i64::try_from(binding.offset)
                    .map_err(|_| "buffer offset exceeds scalar address range")?,
            });
        }
        capacities.insert(
            AllocationIdentity::BufferTable,
            (program.buffers.len() as u64)
                .checked_mul(8)
                .ok_or("buffer table overflow")?,
        );
        capacities.insert(AllocationIdentity::Scalars, workload.scalars.len() as u64);
        for (i, &byte) in workload.scalars.iter().enumerate() {
            bytes.insert((AllocationIdentity::Scalars, i as u64), Some(byte));
        }
        Ok(Self {
            capacities,
            bytes,
            pointers,
            scratch_bytes: program.scratch_bytes as u64,
        })
    }
    fn address(
        &self,
        datum: &Datum,
        displacement: i32,
        width: u32,
    ) -> Result<(AllocationIdentity, u64), String> {
        let Datum::Pointer { allocation, offset } = datum else {
            return Err("memory address depends on an unresolved workload value".into());
        };
        let offset = offset
            .checked_add(i64::from(displacement))
            .ok_or("memory address overflow")?;
        let offset = u64::try_from(offset).map_err(|_| "negative scalar memory address")?;
        let capacity = self
            .capacities
            .get(allocation)
            .ok_or("unknown scalar allocation")?;
        if offset
            .checked_add(u64::from(width))
            .is_none_or(|end| end > *capacity)
        {
            return Err("scalar workload executes an out-of-bounds access".into());
        }
        Ok((allocation.clone(), offset))
    }
    fn load(
        &self,
        allocation: &AllocationIdentity,
        offset: u64,
        width: u32,
    ) -> Result<Datum, String> {
        if *allocation == AllocationIdentity::BufferTable {
            if width != 8 || !offset.is_multiple_of(8) {
                return Err("invalid scalar buffer-table access".into());
            }
            return self
                .pointers
                .get((offset / 8) as usize)
                .cloned()
                .ok_or("buffer-table index out of bounds".into());
        }
        let mut result = 0;
        let mut known = true;
        for i in 0..width {
            match self.bytes.get(&(allocation.clone(), offset + u64::from(i))) {
                Some(Some(byte)) => result |= u64::from(*byte) << (8 * i),
                Some(None) => known = false,
                None if matches!(allocation, AllocationIdentity::Scratch(_)) => {
                    return Err("scalar execution reads uninitialized private scratch".into());
                }
                None => known = false,
            }
        }
        Ok(if known {
            Datum::Bits(result)
        } else {
            Datum::Unknown
        })
    }
    fn store(
        &mut self,
        allocation: &AllocationIdentity,
        offset: u64,
        width: u32,
        datum: &Datum,
    ) -> Result<(), String> {
        if matches!(
            allocation,
            AllocationIdentity::BufferTable | AllocationIdentity::Scalars
        ) {
            return Err("scalar execution mutates a read-only ABI allocation".into());
        }
        if matches!(datum, Datum::Pointer { .. }) {
            return Err("scalar data store cannot publish an ABI pointer".into());
        }
        for i in 0..width {
            let byte = if let Datum::Bits(bits) = datum {
                Some((bits >> (8 * i)) as u8)
            } else {
                None
            };
            self.bytes
                .insert((allocation.clone(), offset + u64::from(i)), byte);
        }
        Ok(())
    }
}

struct Derivation<'a> {
    program: &'a ScalarProgram,
    graph: &'a Graph,
    hardware: &'a ScalarHardware,
    limits: DerivationLimits,
    memory: Memory,
    output: DerivedModel,
}
struct Instance {
    completion: usize,
    roots: Vec<usize>,
}

impl Derivation<'_> {
    fn append(&mut self, operation: Operation, origin: Origin) -> Result<usize, DerivationError> {
        if self.output.model.operations.len() == self.limits.operations {
            return Err(DerivationError::Exhausted(DerivationLimit::Operations(
                self.limits.operations,
            )));
        }
        let id = self.output.model.operations.len();
        self.output.model.operations.push(operation);
        self.output.origins.push(origin);
        Ok(id)
    }
    fn instantiate(
        &mut self,
        primitive: &Primitive,
        dependencies: &[usize],
        origin: Origin,
    ) -> Result<Instance, DerivationError> {
        let timing = self.hardware.timing(primitive);
        let (latency, reservations) = if let Some(timing) = timing {
            (timing.latency, timing.services.clone())
        } else {
            let missing = format!("resource timing unavailable for actual primitive {primitive:?}");
            if !self.output.model.unmapped.contains(&missing) {
                self.output.model.unmapped.push(missing);
            }
            // This retains known execution/dependency facts, not a zero-cost
            // assertion. Incomplete timing blocks a complete objective result.
            (0, Vec::new())
        };
        let id = self.output.model.operations.len();
        let completion = self.append(
            Operation {
                name: format!("instruction{id}"),
                predecessors: dependencies.to_vec(),
                start_predecessors: Vec::new(),
                latency,
                reservations,
            },
            origin,
        )?;
        Ok(Instance {
            completion,
            roots: vec![completion],
        })
    }

    fn invocation(&mut self, invocation: u64) -> Result<(), DerivationError> {
        let function = &self.program.function;
        let entry = function
            .layout
            .entry_block()
            .ok_or("empty scalar program")?;
        let params = function.dfg.block_params(entry);
        let expected = if self.program.dispatch == Dispatch::ParallelRoot {
            4
        } else {
            3
        };
        if params.len() != expected
            || params
                .iter()
                .any(|v| function.dfg.value_type(*v) != types::I64)
        {
            return Err("scalar program does not match the admitted invocation ABI".into());
        }
        self.memory.capacities.insert(
            AllocationIdentity::Scratch(invocation),
            self.memory.scratch_bytes,
        );
        let mut values = HashMap::new();
        for (v, allocation) in params.iter().take(3).zip([
            AllocationIdentity::BufferTable,
            AllocationIdentity::Scalars,
            AllocationIdentity::Scratch(invocation),
        ]) {
            values.insert(
                *v,
                Binding {
                    datum: Datum::Pointer {
                        allocation,
                        offset: 0,
                    },
                    ready: Vec::new(),
                },
            );
        }
        if params.len() == 4 {
            values.insert(
                params[3],
                Binding {
                    datum: Datum::Bits(invocation),
                    ready: Vec::new(),
                },
            );
        }
        let mut block = entry;
        let mut gate = Vec::new();
        let mut completed = Vec::new();
        let mut occurrences = HashMap::<Inst, u64>::new();
        let mut block_occurrences = HashMap::new();
        loop {
            let basic = self
                .graph
                .blocks
                .iter()
                .find(|b| b.id == block)
                .ok_or("absent scalar CFG block")?;
            let mut edge = None;
            let block_occurrence = block_occurrences.entry(block).or_insert(0u64);
            let mut visit = crate::schedule::static_order::Visit {
                invocation,
                occurrence: *block_occurrence,
                roots: Vec::new(),
            };
            *block_occurrence += 1;
            for &index in &basic.instructions {
                if self.output.instructions == self.limits.instructions {
                    return Err(DerivationError::Exhausted(DerivationLimit::Instructions(
                        self.limits.instructions,
                    )));
                }
                self.output.instructions += 1;
                let instruction = &self.graph.instructions[index];
                let occurrence = *occurrences.entry(instruction.id).or_default();
                *occurrences.get_mut(&instruction.id).unwrap() += 1;
                let input_values: Vec<_> = if instruction.opcode == Opcode::Jump {
                    Vec::new()
                } else if instruction.opcode == Opcode::Brif {
                    instruction.inputs.iter().take(1).collect()
                } else {
                    instruction.inputs.iter().collect()
                };
                let inputs = input_values
                    .iter()
                    .map(|input| {
                        values
                            .get(&function.dfg.resolve_aliases(input.value))
                            .cloned()
                            .ok_or_else(|| {
                                format!(
                                    "{}: value {} is not defined on this execution path",
                                    instruction.id, input.value
                                )
                            })
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let mut dependencies: BTreeSet<usize> = gate.iter().copied().collect();
                dependencies.extend(inputs.iter().flat_map(|v| v.ready.iter().copied()));
                let mut memory = None;
                if let Some(access) = &instruction.memory {
                    let address = values
                        .get(&function.dfg.resolve_aliases(access.address))
                        .ok_or("undefined scalar memory address")?;
                    let (allocation, offset) =
                        self.memory
                            .address(&address.datum, access.offset, access.bytes)?;
                    for before in &self.output.accesses {
                        if before.allocation == allocation
                            && (before.write || access.write)
                            && before.offset < offset + u64::from(access.bytes)
                            && offset < before.offset + u64::from(before.bytes)
                        {
                            if before.invocation != invocation {
                                return Err("parallel scalar invocations have conflicting aliased accesses; this direct form has no inter-invocation ordering".into());
                            }
                            dependencies.insert(before.completion);
                        }
                    }
                    memory = Some((allocation, offset, access.bytes, access.write));
                }
                if instruction.opcode == Opcode::Return {
                    dependencies.extend(completed.iter().copied());
                }
                let instance = self.instantiate(
                    &primitive(instruction)?,
                    &dependencies.into_iter().collect::<Vec<_>>(),
                    Origin::Instruction {
                        invocation,
                        instruction: instruction.id,
                        occurrence,
                    },
                )?;
                let completion = instance.completion;
                visit.roots.push(instance.roots);
                completed.push(completion);
                let datum = if let Some((allocation, offset, width, write)) = memory {
                    let datum = if write {
                        self.memory
                            .store(&allocation, offset, width, &inputs[0].datum)?;
                        Datum::Unknown
                    } else {
                        self.memory.load(&allocation, offset, width)?
                    };
                    self.output.accesses.push(Access {
                        invocation,
                        instruction: instruction.id,
                        occurrence,
                        allocation,
                        offset,
                        bytes: width,
                        write,
                        completion,
                    });
                    datum
                } else {
                    evaluate(
                        instruction,
                        &inputs.iter().map(|v| &v.datum).collect::<Vec<_>>(),
                    )?
                };
                if instruction.outputs.len() > 1 {
                    return Err("scalar primitive has multiple unmodeled results".into());
                }
                for &(value, _) in &instruction.outputs {
                    values.insert(
                        value,
                        Binding {
                            datum: datum.clone(),
                            ready: vec![completion],
                        },
                    );
                }
                match instruction.encoding {
                    Data::Jump { destination, .. } => {
                        edge = Some((destination, instruction.id, occurrence, completion));
                        break;
                    }
                    Data::Brif { blocks, .. } => {
                        let Datum::Bits(condition) = inputs[0].datum else {
                            return Err(format!(
                                "{}: workload does not determine a data-dependent branch",
                                instruction.id
                            )
                            .into());
                        };
                        edge = Some((
                            blocks[usize::from(condition == 0)],
                            instruction.id,
                            occurrence,
                            completion,
                        ));
                        break;
                    }
                    Data::MultiAry {
                        opcode: Opcode::Return,
                        ..
                    } => {
                        if inputs.len() != 1 || inputs[0].datum != Datum::Bits(0) {
                            return Err(
                                "workload does not complete the scalar kernel successfully".into(),
                            );
                        }
                        self.output
                            .model
                            .static_orders
                            .iter_mut()
                            .find(|c| c.block == block)
                            .expect("all source blocks have order constraints")
                            .visits
                            .push(visit);
                        return Ok(());
                    }
                    _ => {}
                }
            }
            self.output
                .model
                .static_orders
                .iter_mut()
                .find(|c| c.block == block)
                .expect("all source blocks have order constraints")
                .visits
                .push(visit);
            let (destination, terminator, occurrence, completion) =
                edge.ok_or("scalar block has no admitted terminator")?;
            let (next, assignments, next_gate) = self.transfer(
                invocation,
                terminator,
                occurrence,
                completion,
                destination,
                &values,
            )?;
            // Collect every source before assigning any parameter; this preserves
            // swaps and loop-carried cycles in simultaneous phi transfers.
            for (parameter, binding) in assignments {
                values.insert(parameter, binding);
            }
            block = next;
            gate = vec![next_gate];
            completed.push(next_gate);
        }
    }

    fn transfer(
        &mut self,
        invocation: u64,
        terminator: Inst,
        occurrence: u64,
        completion: usize,
        edge: BlockCall,
        values: &HashMap<Value, Binding>,
    ) -> Result<(Block, Vec<(Value, Binding)>, usize), DerivationError> {
        let f = &self.program.function;
        let destination = edge.block(&f.dfg.value_lists);
        let mut arguments = Vec::new();
        for argument in edge.args(&f.dfg.value_lists) {
            let ir::BlockArg::Value(value) = argument else {
                return Err("exception CFG transfers are not admitted".into());
            };
            let value = f.dfg.resolve_aliases(value);
            arguments.push(
                values
                    .get(&value)
                    .cloned()
                    .ok_or("undefined scalar edge argument")?,
            );
        }
        let parameters = f.dfg.block_params(destination);
        if arguments.len() != parameters.len() {
            return Err("scalar edge arity mismatch".into());
        }
        let ready = if arguments.is_empty() {
            completion
        } else {
            let mut dependencies = BTreeSet::from([completion]);
            dependencies.extend(arguments.iter().flat_map(|a| a.ready.iter().copied()));
            let id = self.output.model.operations.len();
            self.append(
                Operation {
                    name: format!("transfer{id}"),
                    predecessors: dependencies.into_iter().collect(),
                    start_predecessors: Vec::new(),
                    latency: 0,
                    reservations: Vec::new(),
                },
                Origin::EdgeTransfer {
                    invocation,
                    terminator,
                    occurrence,
                    destination,
                },
            )?
        };
        Ok((
            destination,
            parameters
                .iter()
                .copied()
                .zip(arguments.into_iter().map(|mut a| {
                    a.ready = vec![ready];
                    a
                }))
                .collect(),
            ready,
        ))
    }
}

/// Propagation is only for workload/control specialization. Unknown tensor data
/// stays unknown while its actual operation remains in the analysis. Missing
/// timing remains explicit. Known-value propagation never removes instructions.
fn evaluate(instruction: &Instruction, inputs: &[&Datum]) -> Result<Datum, String> {
    use Opcode as O;
    let op = instruction.opcode;
    let output = instruction
        .outputs
        .first()
        .map(|(_, ty)| *ty)
        .unwrap_or(types::I64);
    let bits = |v: &Datum| {
        if let Datum::Bits(bits) = v {
            Some(*bits)
        } else {
            None
        }
    };
    let known = |value: u64| -> Result<Datum, String> {
        let bits_type = if output == types::F32 {
            types::I32
        } else {
            output
        };
        seismic_realization::integer::unsigned(value, bits_type)
            .map(Datum::Bits)
            .ok_or_else(|| format!("unsupported scalar bit width: {output}"))
    };
    match instruction.encoding {
        Data::UnaryIeee32 { imm, .. } => return known(u64::from(imm.bits())),
        Data::FloatCompare { cond, .. } => {
            return Ok(bits(inputs[0])
                .zip(bits(inputs[1]))
                .map(|(a, b)| {
                    let (a, b) = (f32::from_bits(a as u32), f32::from_bits(b as u32));
                    let ordered = !a.is_nan() && !b.is_nan();
                    let value = match cond {
                        FloatCC::Ordered => ordered,
                        FloatCC::Unordered => !ordered,
                        FloatCC::Equal => a == b,
                        FloatCC::NotEqual => a != b,
                        FloatCC::OrderedNotEqual => ordered && a != b,
                        FloatCC::UnorderedOrEqual => !ordered || a == b,
                        FloatCC::LessThan => a < b,
                        FloatCC::LessThanOrEqual => a <= b,
                        FloatCC::GreaterThan => a > b,
                        FloatCC::GreaterThanOrEqual => a >= b,
                        FloatCC::UnorderedOrLessThan => !ordered || a < b,
                        FloatCC::UnorderedOrLessThanOrEqual => !ordered || a <= b,
                        FloatCC::UnorderedOrGreaterThan => !ordered || a > b,
                        FloatCC::UnorderedOrGreaterThanOrEqual => !ordered || a >= b,
                    };
                    Datum::Bits(u64::from(value))
                })
                .unwrap_or(Datum::Unknown));
        }
        _ => {}
    }
    if matches!(op, O::Jump | O::Brif | O::Return | O::Call) {
        return Ok(Datum::Unknown);
    }
    if op == O::Select {
        return Ok(match bits(inputs[0]) {
            Some(condition) => inputs[if condition != 0 { 1 } else { 2 }].clone(),
            None if inputs[1] == inputs[2] => inputs[1].clone(),
            None => Datum::Unknown,
        });
    }
    let immediate = match instruction.encoding {
        Data::BinaryImm64 { imm, .. } => Some(imm.bits() as u64),
        _ => None,
    };
    let a = inputs.first().copied().unwrap_or(&Datum::Unknown);
    let b = immediate
        .map(Datum::Bits)
        .or_else(|| inputs.get(1).map(|x| (*x).clone()));
    if matches!(op, O::Iadd | O::IaddImm) {
        let add = |pointer: &Datum, delta: &Datum| -> Result<Option<Datum>, String> {
            if let (Datum::Pointer { allocation, offset }, Datum::Bits(delta)) = (pointer, delta) {
                return Ok(Some(Datum::Pointer {
                    allocation: allocation.clone(),
                    offset: offset
                        .checked_add(*delta as i64)
                        .ok_or("scalar pointer overflow")?,
                }));
            }
            Ok(None)
        };
        if let Some(b) = &b {
            if let Some(pointer) = add(a, b)?.or(add(b, a)?) {
                return Ok(pointer);
            }
        }
    }
    if op == O::Bitcast {
        return Ok(a.clone());
    }
    match seismic_realization::integer::evaluate(&instruction.encoding, output, |index| {
        let input = instruction.inputs.get(index)?;
        Some((input.ty, inputs.get(index).and_then(|value| bits(value))))
    }) {
        seismic_realization::integer::Evaluation::Exact(value) => return Ok(Datum::Bits(value)),
        seismic_realization::integer::Evaluation::Unknown { may_trap: false } => {
            return Ok(Datum::Unknown);
        }
        seismic_realization::integer::Evaluation::Unknown { may_trap: true } => {
            return Err("workload does not establish the integer operation's validity".into());
        }
        seismic_realization::integer::Evaluation::Trap(trap) => {
            return Err(match trap {
                seismic_realization::integer::Trap::DivisionByZero => {
                    "scalar workload divides by zero"
                }
                seismic_realization::integer::Trap::SignedDivisionOverflow => {
                    "scalar workload has invalid signed division"
                }
            }
            .into());
        }
        seismic_realization::integer::Evaluation::Unsupported => {}
    }
    let Some(av) = bits(a) else {
        return Ok(Datum::Unknown);
    };
    if matches!(op, O::FcvtFromSint | O::FcvtFromUint) {
        let f = if op == O::FcvtFromSint {
            seismic_realization::integer::signed(av, instruction.inputs[0].ty)
                .ok_or("invalid scalar integer conversion width")? as f32
        } else {
            av as f32
        };
        return known(u64::from(f.to_bits()));
    }
    if matches!(op, O::Fneg | O::Fabs) {
        return known(if op == O::Fneg {
            av ^ 0x8000_0000
        } else {
            av & 0x7fff_ffff
        });
    }
    // Transcendentals, sqrt, saturating float conversions, and floating arithmetic
    // can carry backend-specific rounding/NaN rules. They retain unknown values
    // rather than using host arithmetic as an unstated native semantic axiom.
    if matches!(
        op,
        O::Sqrt
            | O::FcvtToSintSat
            | O::FcvtToUintSat
            | O::Fadd
            | O::Fsub
            | O::Fmul
            | O::Fdiv
            | O::Fma
            | O::Fmin
            | O::Fmax
    ) {
        return Ok(Datum::Unknown);
    }
    Err(format!(
        "scalar value propagation lacks admitted opcode {op}"
    ))
}
