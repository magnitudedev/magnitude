//! Consumption of the concrete scalar SSA realization. This is prior to native
//! optimization: instruction instances and requested bytes, not issued machine
//! instructions, cache transactions, or DRAM traffic.
use crate::quantity::Count;
use cranelift_codegen::ir::{self, Value, ValueDef};
use seismic_realization::{
    ScalarProgram,
    execution::{MemoryObject, Multiplicity},
};
use std::collections::{BTreeMap, HashMap};

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
    let mut analysis = Analysis {
        program,
        constants: HashMap::new(),
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
    let graph = seismic_realization::graph::Graph::scalar(program);
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
struct Analysis<'a> {
    program: &'a ScalarProgram,
    constants: HashMap<Value, Option<i64>>,
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
    fn constant(&mut self, value: Value) -> Option<i64> {
        let value = self.program.function.dfg.resolve_aliases(value);
        if let Some(n) = self.constants.get(&value) {
            return *n;
        }
        self.constants.insert(value, None);
        let f = &self.program.function;
        let ty = f.dfg.value_type(value);
        // Count reports remain partial: unsupported widths, runtime operands and
        // traps are unavailable facts, never invented execution counts.
        let result = if matches!(ty, ir::types::I64 | ir::types::I8) {
            match f.dfg.value_def(value) {
                ValueDef::Result(inst, _) => {
                    let data = &f.dfg.insts[inst];
                    let args = f.dfg.inst_args(inst);
                    match seismic_realization::integer::evaluate(data, ty, |index| {
                        let argument = *args.get(index)?;
                        Some((
                            f.dfg.value_type(argument),
                            self.constant(argument).map(|n| n as u64),
                        ))
                    }) {
                        seismic_realization::integer::Evaluation::Exact(bits) => {
                            seismic_realization::integer::signed(bits, ty)
                        }
                        _ => None,
                    }
                }
                _ => None,
            }
        } else {
            None
        };
        self.constants.insert(value, result);
        result
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
