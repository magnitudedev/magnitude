//! Dependency and memory structure of the exact scalar program consumed by native
//! emitters. Control-flow edges and block arguments remain explicit: flattening a
//! loop or a branch into an unordered operation count would lose dependencies.
use crate::{
    MathFunction, ScalarProgram,
    execution::{MemoryObject, Multiplicity},
};
use cranelift_codegen::ir::{
    Block, BlockArg, Inst, InstructionData as Data, Opcode, Type, Value, ValueDef,
};
use std::{collections::HashMap, sync::Arc};

#[derive(Clone, Debug)]
pub struct Input {
    pub value: Value,
    pub ty: Type,
    pub definition: ValueDef,
}
#[derive(Clone, Debug)]
pub struct Instruction {
    pub id: Inst,
    pub block: Block,
    pub opcode: Opcode,
    /// Original immediates/modifiers; pooled handles remain scoped to the source program.
    pub encoding: Data,
    pub inputs: Vec<Input>,
    pub outputs: Vec<(Value, Type)>,
    pub primitive: Option<MathFunction>,
    pub memory: Option<MemoryAccess>,
    pub opaque_effect: bool,
}
#[derive(Clone, Debug)]
pub struct MemoryAccess {
    pub object: Option<MemoryObject>,
    pub address: Value,
    pub offset: i32,
    pub bytes: u32,
    pub write: bool,
}
impl MemoryAccess {
    /// External tensor parameters may alias even when their names differ. The
    /// invocation ABI's internal argument tables and scratch are separate owners.
    pub fn may_overlap(&self, other: &Self) -> bool {
        if let (Some(a), Some(b)) = (&self.object, &other.object) {
            if a != b
                && !matches!(
                    (a, b),
                    (MemoryObject::Buffer { .. }, MemoryObject::Buffer { .. })
                )
            {
                return false;
            }
        }
        if self.address == other.address {
            let a = i64::from(self.offset)..i64::from(self.offset) + i64::from(self.bytes);
            let b = i64::from(other.offset)..i64::from(other.offset) + i64::from(other.bytes);
            return a.start < b.end && b.start < a.end;
        }
        true
    }
}
#[derive(Clone, Debug)]
pub struct BasicBlock {
    pub id: Block,
    pub parameters: Vec<(Value, Type)>,
    pub instructions: Vec<usize>,
    pub executions: Option<Arc<Multiplicity>>,
}
#[derive(Clone, Debug)]
pub struct Edge {
    pub from: Block,
    pub terminator: Inst,
    pub destination: Block,
    /// Simultaneous value transfer, including loop-carried values on backedges.
    pub arguments: Vec<(Value, Value)>,
}
#[derive(Clone, Debug)]
pub struct Graph {
    pub blocks: Vec<BasicBlock>,
    pub instructions: Vec<Instruction>,
    pub edges: Vec<Edge>,
    /// Required ordering within a block. Cross-block ordering also requires the
    /// control-flow edges; this list alone is not a whole-program schedule.
    pub memory_order: Vec<(Inst, Inst)>,
    pub unavailable: Vec<String>,
}

impl Graph {
    pub fn scalar(program: &ScalarProgram) -> Self {
        let f = &program.function;
        let mut roots = Roots {
            program,
            cache: HashMap::new(),
        };
        let mut graph = Self {
            blocks: Vec::new(),
            instructions: Vec::new(),
            edges: Vec::new(),
            memory_order: Vec::new(),
            unavailable: Vec::new(),
        };
        for block in f.layout.blocks() {
            let mut basic = BasicBlock {
                id: block,
                parameters: f
                    .dfg
                    .block_params(block)
                    .iter()
                    .map(|v| (*v, f.dfg.value_type(*v)))
                    .collect(),
                instructions: Vec::new(),
                executions: program.execution.blocks.get(&block).cloned(),
            };
            for inst in f.layout.block_insts(block) {
                let data = &f.dfg.insts[inst];
                let primitive = match data {
                    Data::Call { func_ref, .. } => program
                        .imports
                        .iter()
                        .find(|(id, _)| id == func_ref)
                        .map(|(_, p)| *p),
                    _ => None,
                };
                let memory = match data {
                    Data::Load { arg, offset, .. } => Some(MemoryAccess {
                        object: roots.get(*arg),
                        address: f.dfg.resolve_aliases(*arg),
                        offset: (*offset).into(),
                        bytes: f.dfg.value_type(f.dfg.inst_results(inst)[0]).bytes(),
                        write: false,
                    }),
                    Data::Store { args, offset, .. } => Some(MemoryAccess {
                        object: roots.get(args[1]),
                        address: f.dfg.resolve_aliases(args[1]),
                        offset: (*offset).into(),
                        bytes: f.dfg.value_type(args[0]).bytes(),
                        write: true,
                    }),
                    _ => None,
                };
                let opcode = data.opcode();
                let opaque_effect = (opcode.is_call() && primitive.is_none())
                    || ((opcode.can_load() || opcode.can_store())
                        && memory.is_none()
                        && !opcode.is_call())
                    || (opcode.other_side_effects()
                        && !opcode.is_call()
                        && !opcode.is_branch()
                        && opcode != Opcode::Return
                        && memory.is_none());
                if opaque_effect {
                    graph
                        .unavailable
                        .push(format!("{inst}: external effects are not admitted"));
                }
                let instruction = Instruction {
                    id: inst,
                    block,
                    opcode: data.opcode(),
                    encoding: *data,
                    inputs: f
                        .dfg
                        .inst_args(inst)
                        .iter()
                        .map(|v| {
                            let value = f.dfg.resolve_aliases(*v);
                            Input {
                                value,
                                ty: f.dfg.value_type(value),
                                definition: f.dfg.value_def(value),
                            }
                        })
                        .collect(),
                    outputs: f
                        .dfg
                        .inst_results(inst)
                        .iter()
                        .map(|v| (*v, f.dfg.value_type(*v)))
                        .collect(),
                    primitive,
                    memory,
                    opaque_effect,
                };
                for &previous in &basic.instructions {
                    let before: &Instruction = &graph.instructions[previous];
                    let ordered = if before.opaque_effect || instruction.opaque_effect {
                        before.memory.is_some()
                            || instruction.memory.is_some()
                            || (before.opaque_effect && instruction.opaque_effect)
                    } else if let (Some(a), Some(b)) = (&before.memory, &instruction.memory) {
                        (a.write || b.write) && a.may_overlap(b)
                    } else {
                        false
                    };
                    if ordered {
                        graph.memory_order.push((before.id, inst));
                    }
                }
                for edge in data.branch_destination(&f.dfg.jump_tables, &f.dfg.exception_tables) {
                    let destination = edge.block(&f.dfg.value_lists);
                    let mut arguments = Vec::new();
                    for (source, target) in edge
                        .args(&f.dfg.value_lists)
                        .zip(f.dfg.block_params(destination))
                    {
                        if let BlockArg::Value(source) = source {
                            arguments.push((f.dfg.resolve_aliases(source), *target));
                        } else {
                            graph
                                .unavailable
                                .push(format!("{inst}: exception edge values are not admitted"));
                        }
                    }
                    graph.edges.push(Edge {
                        from: block,
                        terminator: inst,
                        destination,
                        arguments,
                    });
                }
                basic.instructions.push(graph.instructions.len());
                graph.instructions.push(instruction);
            }
            graph.blocks.push(basic);
        }
        graph
    }
}

struct Roots<'a> {
    program: &'a ScalarProgram,
    cache: HashMap<Value, Option<MemoryObject>>,
}
impl Roots<'_> {
    fn get(&mut self, value: Value) -> Option<MemoryObject> {
        let f = &self.program.function;
        let value = f.dfg.resolve_aliases(value);
        if let Some(root) = self.program.execution.memory_roots.get(&value) {
            return Some(root.clone());
        }
        if let Some(root) = self.cache.get(&value) {
            return root.clone();
        }
        self.cache.insert(value, None);
        let result = match f.dfg.value_def(value) {
            ValueDef::Result(inst, _) => match f.dfg.insts[inst] {
                Data::BinaryImm64 {
                    opcode: Opcode::IaddImm,
                    arg,
                    ..
                } => self.get(arg),
                Data::Binary {
                    opcode: Opcode::Iadd,
                    args,
                } => match (self.get(args[0]), self.get(args[1])) {
                    (Some(root), None) | (None, Some(root)) => Some(root),
                    _ => None,
                },
                _ => None,
            },
            _ => None,
        };
        self.cache.insert(value, result.clone());
        result
    }
}
