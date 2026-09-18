//! Static instruction-order choices shared by accounting and native emission.
//!
//! A choice changes only the order of existing instructions inside each basic
//! block. It never unrolls loops, changes control flow, drops guards, substitutes
//! values, or assigns hardware issue times. Every loop visit uses the same order.
use crate::{
    graph::{Graph, Instruction},
    ScalarProgram,
};
use cranelift_codegen::ir::{Block, Inst, InstructionData, ValueDef};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BlockOrder {
    pub block: Block,
    pub instructions: Vec<Inst>,
}
/// A proposal, never unchecked authority. Both materialization and independent
/// correspondence checking reconstruct the legal dependency domain.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Order {
    pub blocks: Vec<BlockOrder>,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Domain {
    pub block: Block,
    pub position: usize,
    pub alternatives: Vec<Inst>,
}
pub enum Expansion {
    Choice(Domain),
    Order { order: Order, consumed: usize },
}
/// Shared legal static precedence consumed by the joint resource scheduler.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BlockConstraints {
    pub block: Block,
    pub instructions: Vec<Inst>,
    pub predecessors: Vec<(usize, usize)>,
}
#[derive(Clone, Debug)]
struct BlockDomain {
    block: Block,
    instructions: Vec<Inst>,
    predecessors: BTreeMap<Inst, BTreeSet<Inst>>,
}

/// Lazy enumeration of every topological order in the declared block-preserving
/// form. Forced singleton domains are propagated; enumeration order supplies no
/// performance preference or preferred subset.
pub struct Space {
    blocks: Vec<BlockDomain>,
}
impl Space {
    pub fn new(program: &ScalarProgram) -> Result<Self, String> {
        cranelift_codegen::verify_function(
            &program.function,
            &cranelift_codegen::settings::Flags::new(cranelift_codegen::settings::builder()),
        )
        .map_err(|error| format!("invalid scalar scheduling input: {error}"))?;
        let graph = Graph::scalar(program);
        if !graph.unavailable.is_empty() {
            return Err(format!(
                "static scheduling has unmodeled effects: {:?}",
                graph.unavailable
            ));
        }
        let mut blocks = Vec::new();
        for block in &graph.blocks {
            let instructions: Vec<_> = block
                .instructions
                .iter()
                .map(|&i| &graph.instructions[i])
                .collect();
            let mut predecessors: BTreeMap<Inst, BTreeSet<Inst>> = instructions
                .iter()
                .map(|instruction| (instruction.id, BTreeSet::new()))
                .collect();
            for (position, instruction) in instructions.iter().enumerate() {
                for input in &instruction.inputs {
                    match input.definition {
                        ValueDef::Result(definition, _)
                            if predecessors.contains_key(&definition) =>
                        {
                            predecessors
                                .get_mut(&instruction.id)
                                .unwrap()
                                .insert(definition);
                        }
                        ValueDef::Union(_, _) => {
                            return Err("static scheduling requires resolved SSA values".into())
                        }
                        _ => {}
                    }
                }
                // Preserve observable trapping/effect order as well as ordinary
                // alias hazards. A valid reordering cannot move an observable
                // write before an earlier operation that may fail.
                for earlier in &instructions[..position] {
                    if (may_trap(earlier) && observable(instruction))
                        || (may_trap(instruction) && observable(earlier))
                    {
                        predecessors
                            .get_mut(&instruction.id)
                            .unwrap()
                            .insert(earlier.id);
                    }
                }
            }
            for &(before, after) in &graph.memory_order {
                if let Some(dependencies) = predecessors.get_mut(&after) {
                    dependencies.insert(before);
                }
            }
            let terminator = instructions
                .last()
                .ok_or("scalar block has no terminator")?;
            if !terminator.opcode.is_terminator() {
                return Err("scalar block does not end in a terminator".into());
            }
            for instruction in &instructions[..instructions.len() - 1] {
                predecessors
                    .get_mut(&terminator.id)
                    .unwrap()
                    .insert(instruction.id);
            }
            blocks.push(BlockDomain {
                block: block.id,
                instructions: instructions.iter().map(|i| i.id).collect(),
                predecessors,
            });
        }
        Ok(Self { blocks })
    }

    pub fn constraints(&self) -> Vec<BlockConstraints> {
        self.blocks
            .iter()
            .map(|block| {
                let indices: BTreeMap<_, _> = block
                    .instructions
                    .iter()
                    .copied()
                    .enumerate()
                    .map(|(index, instruction)| (instruction, index))
                    .collect();
                BlockConstraints {
                    block: block.block,
                    instructions: block.instructions.clone(),
                    predecessors: block
                        .predecessors
                        .iter()
                        .flat_map(|(after, before)| {
                            before
                                .iter()
                                .map(|before| (indices[before], indices[after]))
                        })
                        .collect(),
                }
            })
            .collect()
    }

    pub fn expand(&self, prefix: &[usize]) -> Result<Expansion, String> {
        let mut consumed = 0;
        let mut blocks = Vec::new();
        for block in &self.blocks {
            let mut selected = BTreeSet::new();
            let mut instructions = Vec::new();
            while instructions.len() < block.instructions.len() {
                let alternatives: Vec<_> = block
                    .instructions
                    .iter()
                    .copied()
                    .filter(|instruction| {
                        !selected.contains(instruction)
                            && block.predecessors[instruction].is_subset(&selected)
                    })
                    .collect();
                if alternatives.is_empty() {
                    return Err("cyclic static instruction-order constraints".into());
                }
                let next = if alternatives.len() == 1 {
                    alternatives[0]
                } else {
                    let Some(&index) = prefix.get(consumed) else {
                        return Ok(Expansion::Choice(Domain {
                            block: block.block,
                            position: instructions.len(),
                            alternatives,
                        }));
                    };
                    consumed += 1;
                    *alternatives
                        .get(index)
                        .ok_or("static instruction-order choice is outside its domain")?
                };
                selected.insert(next);
                instructions.push(next);
            }
            blocks.push(BlockOrder {
                block: block.block,
                instructions,
            });
        }
        Ok(Expansion::Order {
            order: Order { blocks },
            consumed,
        })
    }

    pub fn check(&self, order: &Order) -> Result<(), String> {
        if order.blocks.len() != self.blocks.len() {
            return Err("static order changes the block domain".into());
        }
        for (proposed, domain) in order.blocks.iter().zip(&self.blocks) {
            if proposed.block != domain.block
                || proposed.instructions.len() != domain.instructions.len()
            {
                return Err("static order changes block identity or instruction count".into());
            }
            let mut visited = BTreeSet::new();
            for instruction in &proposed.instructions {
                let dependencies = domain
                    .predecessors
                    .get(instruction)
                    .ok_or("static order moves an instruction between blocks")?;
                if !dependencies.is_subset(&visited) {
                    return Err(
                        "static order violates data, effect, trap, or terminator dependencies"
                            .into(),
                    );
                }
                if !visited.insert(*instruction) {
                    return Err("static order repeats an instruction".into());
                }
            }
        }
        Ok(())
    }
}

fn may_trap(instruction: &Instruction) -> bool {
    instruction.opcode.can_trap()
        || match instruction.encoding {
            InstructionData::Load { flags, .. } | InstructionData::Store { flags, .. } => {
                !flags.notrap()
            }
            _ => false,
        }
}
fn observable(instruction: &Instruction) -> bool {
    may_trap(instruction)
        || instruction
            .memory
            .as_ref()
            .is_some_and(|access| access.write)
        || instruction.opaque_effect
}

impl Order {
    pub fn current(program: &ScalarProgram) -> Self {
        Self {
            blocks: program
                .function
                .layout
                .blocks()
                .map(|block| BlockOrder {
                    block,
                    instructions: program.function.layout.block_insts(block).collect(),
                })
                .collect(),
        }
    }
    pub fn check(&self, program: &ScalarProgram) -> Result<(), String> {
        Space::new(program)?.check(self)
    }
}

/// Materialize the selected static order in the exact SSA object subsequently
/// consumed by CPU code generation or CUDA PTX printing. No model timing is
/// interpreted as a machine-cycle insertion, and no native compiler is invoked.
pub fn apply(program: &mut ScalarProgram, order: &Order) -> Result<(), String> {
    order.check(program)?;
    let before = program.function.clone();
    for block in &order.blocks {
        let old: Vec<_> = program.function.layout.block_insts(block.block).collect();
        for instruction in old {
            program.function.layout.remove_inst(instruction);
        }
        for &instruction in &block.instructions {
            program
                .function
                .layout
                .append_inst(instruction, block.block);
        }
    }
    if let Err(error) = cranelift_codegen::verify_function(
        &program.function,
        &cranelift_codegen::settings::Flags::new(cranelift_codegen::settings::builder()),
    ) {
        program.function = before;
        return Err(format!("materialized static order violates SSA: {error}"));
    }
    let mut unchanged = program.function.clone();
    unchanged.layout = before.layout.clone();
    if unchanged != before {
        program.function = before;
        return Err("static scheduling changed something other than instruction layout".into());
    }
    Ok(())
}

/// Independent correspondence check for an externally retained proposal and
/// emitted scalar body. Structural equality covers immediates, call signatures,
/// numerical flags, imports, SSA values, and CFG argument transfers.
pub fn check_materialization(
    before: &ScalarProgram,
    after: &ScalarProgram,
    order: &Order,
) -> Result<(), String> {
    order.check(before)?;
    if Order::current(after) != *order {
        return Err("emitted scalar order differs from its selected plan".into());
    }
    let mut normalized = after.function.clone();
    normalized.layout = before.function.layout.clone();
    if normalized != before.function
        || before.buffers != after.buffers
        || before.scalars != after.scalars
        || before.scratch_bytes != after.scratch_bytes
        || before.imports != after.imports
        || before.work_items != after.work_items
        || before.dispatch != after.dispatch
        || before.loads != after.loads
        || before.execution != after.execution
    {
        return Err(
            "static-order materialization changed the computation, ABI, or execution evidence"
                .into(),
        );
    }
    cranelift_codegen::verify_function(
        &after.function,
        &cranelift_codegen::settings::Flags::new(cranelift_codegen::settings::builder()),
    )
    .map_err(|error| format!("invalid materialized scalar body: {error}"))
}
