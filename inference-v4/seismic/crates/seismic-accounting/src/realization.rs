//! Consumption of the concrete scalar SSA realization. This is prior to native
//! optimization: instruction instances and requested bytes, not issued machine
//! instructions, cache transactions, or DRAM traffic.
use crate::quantity::Count;
use cranelift_codegen::ir::{self, Value};
use seismic_realization::{
    ScalarProgram,
    execution::{MemoryObject, Multiplicity},
};
use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};

#[derive(Clone, Debug)]
pub struct InstructionTerm {
    pub block: String,
    pub instruction: String,
    pub opcode: String,
    pub primitive: Option<seismic_realization::MathFunction>,
    pub operand_types: Vec<String>,
    pub result_types: Vec<String>,
    /// Whole-dispatch instances, conditional on successful validity guards.
    pub count: Count,
}
#[derive(Clone, Debug)]
pub struct RequestedTraffic {
    pub reads: Count,
    pub writes: Count,
}
impl Default for RequestedTraffic {
    fn default() -> Self {
        Self {
            reads: Count::Exact(0),
            writes: Count::Exact(0),
        }
    }
}
#[derive(Clone, Debug)]
pub struct ScalarAccount {
    pub instructions: Vec<InstructionTerm>,
    pub traffic: BTreeMap<MemoryObject, RequestedTraffic>,
    pub scratch_bytes_per_invocation: u64,
    pub invocation_count: u64,
    pub scratch_bytes_per_dispatch: Count,
    pub assumptions: Vec<String>,
    pub unavailable: Vec<String>,
}

/// Analysis traverses the emitted program, not its iteration space. Constant
/// ranges compose symbolically. Data-dependent counts stay bounded or unavailable.
pub fn scalar(program: &ScalarProgram) -> ScalarAccount {
    let graph = seismic_realization::graph::Graph::scalar(program);
    let mut analysis = Analysis {
        program,
        integers: IntegerFacts::derive(&program.function, &graph),
        multiplicities: HashMap::new(),
    };
    let mut out = ScalarAccount {
        instructions:Vec::new(),traffic:BTreeMap::new(),
        scratch_bytes_per_invocation:program.scratch_bytes as u64,
        invocation_count:program.work_items,
        scratch_bytes_per_dispatch:Count::Exact(program.scratch_bytes as u64).scale(program.work_items),
        assumptions:vec!["all runtime validity guards pass; guard-failure executions are outside this account".into(),"counts describe scalar SSA before native optimization; memory bytes are requested accesses to named storage, not physical transactions".into()],
        unavailable:Vec::new(),
    };
    out.unavailable.extend(graph.unavailable.iter().cloned());
    for block in &graph.blocks {
        let executions = block
            .executions
            .as_ref()
            .map(|m| analysis.multiplicity(m))
            .unwrap_or_else(|| Count::unknown(format!("missing execution domain for {}", block.id)))
            .scale(program.work_items);
        for &index in &block.instructions {
            let instruction = &graph.instructions[index];
            out.instructions.push(InstructionTerm {
                block: block.id.to_string(),
                instruction: instruction.id.to_string(),
                opcode: instruction.opcode.to_string(),
                primitive: instruction.primitive,
                operand_types: instruction
                    .inputs
                    .iter()
                    .map(|v| v.ty.to_string())
                    .collect(),
                result_types: instruction
                    .outputs
                    .iter()
                    .map(|(_, ty)| ty.to_string())
                    .collect(),
                count: executions.clone(),
            });
            if let Some(access) = &instruction.memory {
                if let Some(root) = &access.object {
                    let traffic = out.traffic.entry(root.clone()).or_default();
                    let count = executions.scale(u64::from(access.bytes));
                    if access.write {
                        traffic.writes = traffic.writes.add(&count);
                    } else {
                        traffic.reads = traffic.reads.add(&count);
                    }
                } else if executions != Count::Exact(0) {
                    out.unavailable.push(format!(
                        "{}: memory object unresolved for {}",
                        instruction.id, access.address
                    ));
                }
            }
        }
    }
    if out.instructions.iter().any(|i| i.count.bounds().is_none()) {
        out.unavailable
            .push("one or more execution domains need runtime or dependent-loop facts".into());
    }
    out
}
/// Finite residue lattice: Bottom has no established incoming execution yet;
/// Known(k,v) says every incoming value equals v modulo 2^k. Known(0,0) is
/// unconstrained. Joins only lose bits, so loops terminate after at most 65
/// changes per integer value without enumerating source iterations.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Residue {
    Bottom,
    Known { bits: u32, value: u64 },
}
impl Residue {
    fn known(bits: u32, value: u64) -> Self {
        let mask = if bits == 64 {
            u64::MAX
        } else {
            (1u64 << bits) - 1
        };
        Self::Known {
            bits,
            value: value & mask,
        }
    }
    fn join(self, other: Self) -> Self {
        match (self, other) {
            (Self::Bottom, x) | (x, Self::Bottom) => x,
            (Self::Known { bits: a, value: x }, Self::Known { bits: b, value: y }) => {
                Self::known(a.min(b).min((x ^ y).trailing_zeros()), x)
            }
        }
    }
    fn low(self, needed: u32) -> Option<u64> {
        match self {
            Self::Known { bits, value } if bits >= needed => {
                let mask = if needed == 64 {
                    u64::MAX
                } else {
                    (1u64 << needed) - 1
                };
                Some(value & mask)
            }
            _ => None,
        }
    }
}
struct IntegerFacts {
    values: HashMap<Value, Residue>,
}
impl IntegerFacts {
    fn derive(function: &ir::Function, graph: &seismic_realization::graph::Graph) -> Self {
        enum Definition {
            Incoming(Vec<Value>),
            Instruction(usize),
            External,
        }
        let canonical = |v| function.dfg.resolve_aliases(v);
        let mut incoming: HashMap<Value, Vec<Value>> = HashMap::new();
        // Including all structural edges is conservative: no feasible incoming
        // execution is omitted, even when branch feasibility is unknown.
        for edge in &graph.edges {
            for &(source, target) in &edge.arguments {
                incoming
                    .entry(canonical(target))
                    .or_default()
                    .push(canonical(source));
            }
        }
        let mut definitions = HashMap::new();
        for block in &graph.blocks {
            for &(value, _) in &block.parameters {
                let value = canonical(value);
                let definition = if Some(block.id) == function.layout.entry_block() {
                    Definition::External
                } else {
                    match incoming.remove(&value) {
                        Some(inputs) => Definition::Incoming(inputs),
                        None => Definition::External,
                    }
                };
                definitions.insert(value, definition);
            }
        }
        for (index, instruction) in graph.instructions.iter().enumerate() {
            for &(value, _) in &instruction.outputs {
                definitions.insert(canonical(value), Definition::Instruction(index));
            }
        }
        let mut users: HashMap<Value, Vec<Value>> = HashMap::new();
        for (&value, definition) in &definitions {
            let inputs = match definition {
                Definition::Incoming(inputs) => inputs.clone(),
                Definition::Instruction(index) => graph.instructions[*index]
                    .inputs
                    .iter()
                    .map(|i| canonical(i.value))
                    .collect(),
                Definition::External => Vec::new(),
            };
            for input in inputs {
                users.entry(input).or_default().push(value);
            }
        }
        let mut values = definitions
            .keys()
            .map(|&v| (v, Residue::Bottom))
            .collect::<HashMap<_, _>>();
        let mut order = definitions.keys().copied().collect::<Vec<_>>();
        order.sort_unstable_by_key(|v| v.as_u32());
        let mut pending = order.iter().copied().collect::<HashSet<_>>();
        let mut queue = VecDeque::from(order);
        loop {
            while let Some(value) = queue.pop_front() {
                pending.remove(&value);
                let ty = function.dfg.value_type(value);
                let get = |v| {
                    values
                        .get(&canonical(v))
                        .copied()
                        .unwrap_or(Residue::known(0, 0))
                };
                let next = if seismic_realization::integer::mask(ty).is_none() {
                    Residue::known(0, 0)
                } else {
                    match &definitions[&value] {
                        Definition::External => Residue::known(0, 0),
                        Definition::Incoming(inputs) => {
                            inputs.iter().fold(Residue::Bottom, |state, &v| {
                                state.join(if function.dfg.value_type(v) == ty {
                                    get(v)
                                } else {
                                    Residue::known(0, 0)
                                })
                            })
                        }
                        Definition::Instruction(index) => {
                            let instruction = &graph.instructions[*index];
                            Self::operation(
                                &instruction.encoding,
                                ty,
                                &instruction
                                    .inputs
                                    .iter()
                                    .map(|i| (i.ty, get(i.value)))
                                    .collect::<Vec<_>>(),
                            )
                        }
                    }
                };
                let next = values[&value].join(next);
                if next != values[&value] {
                    values.insert(value, next);
                    if let Some(dependents) = users.get(&value) {
                        for &dependent in dependents {
                            if pending.insert(dependent) {
                                queue.push_back(dependent);
                            }
                        }
                    }
                }
            }
            // Cycles without an anchored value confer no facts. Seal remaining
            // bottom values as unknown and propagate that loss to their users;
            // no optimistic, unproved cyclic assumption escapes the analysis.
            let unresolved = values
                .iter()
                .filter_map(|(&v, &s)| (s == Residue::Bottom).then_some(v))
                .collect::<Vec<_>>();
            if unresolved.is_empty() {
                break;
            }
            for value in unresolved {
                values.insert(value, Residue::known(0, 0));
                if let Some(dependents) = users.get(&value) {
                    for &dependent in dependents {
                        if pending.insert(dependent) {
                            queue.push_back(dependent);
                        }
                    }
                }
            }
        }
        Self { values }
    }
    fn operation(
        data: &ir::InstructionData,
        ty: ir::Type,
        inputs: &[(ir::Type, Residue)],
    ) -> Residue {
        use seismic_realization::integer::{self, Evaluation};
        // Bottom is not an assumed zero. A cyclic instruction waits for an
        // anchored incoming value before it can establish any invariant.
        if inputs.iter().any(|(_, state)| *state == Residue::Bottom) {
            return Residue::Bottom;
        }
        let exact = |index: usize| {
            let &(ty, state) = inputs.get(index)?;
            Some((ty, state.low(ty.bits() as u32)))
        };
        if let Evaluation::Exact(bits) = integer::evaluate(data, ty, exact) {
            return Residue::known(ty.bits() as u32, bits);
        }
        if data.opcode() == ir::Opcode::Select
            && inputs.len() == 3
            && inputs[0].0 == ir::types::I8
            && inputs[1].0 == ty
            && inputs[2].0 == ty
        {
            return match inputs[0].1.low(8) {
                Some(0) => inputs[2].1,
                Some(_) => inputs[1].1,
                None => inputs[1].1.join(inputs[2].1),
            };
        }
        // For a positive power-of-two unsigned divisor, only those low bits
        // determine the entire remainder. Zero and unknown divisors confer no
        // fact; the instruction's validity/trap contract is unchanged.
        let divisor = match data {
            ir::InstructionData::BinaryImm64 {
                opcode: ir::Opcode::UremImm,
                imm,
                ..
            } => integer::unsigned(imm.bits() as u64, ty),
            ir::InstructionData::Binary {
                opcode: ir::Opcode::Urem,
                ..
            } => exact(1).and_then(|(_, v)| v),
            _ => None,
        };
        if let Some(divisor) = divisor.filter(|d| d.is_power_of_two()) {
            if divisor == 1 {
                return Residue::known(ty.bits() as u32, 0);
            }
            if let Some((_, state)) = inputs.first() {
                if let Some(value) = state.low(divisor.trailing_zeros()) {
                    return Residue::known(ty.bits() as u32, value);
                }
            }
        }
        for bits in (1..=ty.bits() as u32).rev() {
            if let Some(value) = integer::low_bits(data, ty, bits, |index, needed| {
                let &(ty, state) = inputs.get(index)?;
                Some((ty, state.low(needed)))
            }) {
                return Residue::known(bits, value);
            }
        }
        Residue::known(0, 0)
    }
}
struct Analysis<'a> {
    program: &'a ScalarProgram,
    integers: IntegerFacts,
    multiplicities: HashMap<usize, Count>,
}
impl Analysis<'_> {
    fn multiplicity(&mut self, m: &std::sync::Arc<Multiplicity>) -> Count {
        let mut cache = std::mem::take(&mut self.multiplicities);
        let count =
            crate::multiplicity::evaluate(m, &mut cache, &mut |value| self.constant(*value));
        self.multiplicities = cache;
        count
    }
    fn constant(&self, value: Value) -> Option<i64> {
        let value = self.program.function.dfg.resolve_aliases(value);
        let ty = self.program.function.dfg.value_type(value);
        let bits = self.integers.values.get(&value)?.low(ty.bits() as u32)?;
        seismic_realization::integer::signed(bits, ty)
    }
}
/// Source-ordered scalar phases. Shared tensor identities remain the same across
/// phase accounts; callers must not sum their interface regions as new obligations.
#[derive(Clone, Debug)]
pub struct SequenceAccount {
    pub name: String,
    pub phases: Vec<PhaseAccount>,
    pub completion_edges: Vec<(usize, usize)>,
    /// Logical scratch when all phase allocations are retained (the current CUDA
    /// sequence runtime). This is not a physical transaction or register estimate.
    pub retained_scratch_bytes: Count,
}
#[derive(Clone, Debug)]
pub struct PhaseAccount {
    pub source_statement: usize,
    pub account: ScalarAccount,
}
pub fn sequence(sequence: &seismic_realization::ScalarSequence) -> SequenceAccount {
    let phases = sequence
        .phases
        .iter()
        .map(|phase| PhaseAccount {
            source_statement: phase.source_statement,
            account: scalar(&phase.program),
        })
        .collect::<Vec<_>>();
    let retained_scratch_bytes = phases.iter().fold(Count::Exact(0), |sum, phase| {
        sum.add(&phase.account.scratch_bytes_per_dispatch)
    });
    let completion_edges = (1..phases.len()).map(|index| (index - 1, index)).collect();
    SequenceAccount {
        name: sequence.name.clone(),
        phases,
        completion_edges,
        retained_scratch_bytes,
    }
}

#[cfg(test)]
mod integer_facts_tests {
    use super::*;
    use cranelift_codegen::{
        cursor::{Cursor, FuncCursor},
        ir::{AbiParam, InstBuilder, condcodes::IntCC, types},
    };

    fn loop_program(step: i64, alternate_entry: bool) -> (ScalarProgram, Value, Value) {
        let mut function = ir::Function::new();
        function.signature.params.push(AbiParam::new(types::I64));
        function.signature.returns.push(AbiParam::new(types::I64));
        let entry = function.dfg.make_block();
        let head = function.dfg.make_block();
        let body = function.dfg.make_block();
        let done = function.dfg.make_block();
        for block in [entry, head, body, done] {
            function.layout.append_block(block);
        }
        let argument = function.dfg.append_block_param(entry, types::I64);
        let counter = function.dfg.append_block_param(head, types::I64);
        let carried = function.dfg.append_block_param(head, types::I64);
        let mut cursor = FuncCursor::new(&mut function);
        cursor.goto_bottom(entry);
        let zero = cursor.ins().iconst(types::I64, 0);
        let aligned = cursor.ins().imul_imm(argument, 64);
        let initial = if alternate_entry {
            let condition = cursor.ins().icmp_imm(IntCC::Equal, argument, 0);
            let odd = cursor.ins().iconst(types::I64, 1);
            cursor.ins().select(condition, aligned, odd)
        } else {
            aligned
        };
        cursor.ins().jump(head, &[zero.into(), initial.into()]);
        cursor.goto_bottom(head);
        let remainder = cursor.ins().urem_imm(carried, 64);
        let run = cursor.ins().icmp_imm(IntCC::UnsignedLessThan, counter, 4);
        cursor.ins().brif(run, body, &[], done, &[]);
        cursor.goto_bottom(body);
        let next = cursor.ins().iadd_imm(carried, step);
        let iteration = cursor.ins().iadd_imm(counter, 1);
        cursor.ins().jump(head, &[iteration.into(), next.into()]);
        cursor.goto_bottom(done);
        cursor.ins().return_(&[remainder]);
        drop(cursor);
        cranelift_codegen::verify_function(
            &function,
            &cranelift_codegen::settings::Flags::new(cranelift_codegen::settings::builder()),
        )
        .unwrap();
        (
            ScalarProgram {
                conditions: Default::default(),
                function,
                buffers: Vec::new(),
                public_buffer_count: 0,
                scalars: Vec::new(),
                scratch_bytes: 0,
                imports: Vec::new(),
                backend_calls: Vec::new(),
                participation: seismic_realization::dispatch::Participation::Thread,
                work_items: 1,
                dispatch: seismic_realization::Dispatch::Sequential,
                loads: Vec::new(),
                execution: Default::default(),
            },
            carried,
            remainder,
        )
    }
    #[test]
    fn anchored_loop_residues_follow_every_backedge_without_unrolling() {
        for step in [0, 64, -64, 320] {
            let (program, carried, remainder) = loop_program(step, false);
            let graph = seismic_realization::graph::Graph::scalar(&program);
            let facts = IntegerFacts::derive(&program.function, &graph);
            assert_eq!(facts.values[&carried].low(6), Some(0));
            assert_eq!(facts.values[&carried].low(64), None);
            assert_eq!(facts.values[&remainder].low(64), Some(0));
        }
    }
    #[test]
    fn differing_initial_or_backedge_residues_cannot_claim_alignment() {
        for (step, alternate) in [(1, false), (2, false), (64, true)] {
            let (program, carried, remainder) = loop_program(step, alternate);
            let graph = seismic_realization::graph::Graph::scalar(&program);
            let facts = IntegerFacts::derive(&program.function, &graph);
            assert_eq!(facts.values[&carried].low(6), None);
            assert_eq!(facts.values[&remainder].low(64), None);
        }
    }
    #[test]
    fn zero_divisor_and_unanchored_bottom_never_supply_a_count() {
        let (mut program, _, _) = loop_program(64, false);
        let mut data = program.function.dfg.insts[program
            .function
            .layout
            .block_insts(program.function.layout.blocks().nth(1).unwrap())
            .next()
            .unwrap()]
        .clone();
        let ir::InstructionData::BinaryImm64 { imm, .. } = &mut data else {
            panic!("expected remainder")
        };
        *imm = ir::immediates::Imm64::new(0);
        assert_eq!(
            IntegerFacts::operation(&data, types::I64, &[(types::I64, Residue::known(64, 0))])
                .low(64),
            None
        );
        assert_eq!(
            IntegerFacts::operation(&data, types::I64, &[(types::I64, Residue::Bottom)]),
            Residue::Bottom
        );
        let unanchored = program.function.dfg.make_block();
        program.function.layout.append_block(unanchored);
        let value = program
            .function
            .dfg
            .append_block_param(unanchored, types::I64);
        let mut cursor = FuncCursor::new(&mut program.function);
        cursor.goto_bottom(unanchored);
        cursor.ins().jump(unanchored, &[value.into()]);
        drop(cursor);
        cranelift_codegen::verify_function(
            &program.function,
            &cranelift_codegen::settings::Flags::new(cranelift_codegen::settings::builder()),
        )
        .unwrap();
        let graph = seismic_realization::graph::Graph::scalar(&program);
        let facts = IntegerFacts::derive(&program.function, &graph);
        assert_eq!(facts.values[&value], Residue::known(0, 0));
    }
}
