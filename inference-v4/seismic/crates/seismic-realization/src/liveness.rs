//! SSA value liveness in the selected, loop-preserving execution form.
//!
//! These are virtual typed values before instruction selection and allocation.
//! Counts and bytes are representation facts, never physical registers, spill
//! bytes, rename pressure, or a promise about an external native optimizer. The
//! analysis is the least fixed point over admitted CFG edges; value-dependent
//! path infeasibility is not assumed.
use crate::{
    ScalarProgram,
    graph::{Graph, Instruction},
};
use cranelift_codegen::ir::{Block, Inst, Opcode, Type, Value};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Point {
    pub instruction: Inst,
    pub before: BTreeSet<Value>,
    pub after: BTreeSet<Value>,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BlockLiveness {
    pub block: Block,
    /// Values needed after simultaneous block-parameter definition.
    pub entry: BTreeSet<Value>,
    pub exit: BTreeSet<Value>,
    pub points: Vec<Point>,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EdgeLiveness {
    pub from: Block,
    pub terminator: Inst,
    pub destination: Block,
    /// Source values live on this particular edge, before parameter transfer.
    pub values: BTreeSet<Value>,
    /// Only transfers whose destination parameter is live are retained.
    pub transfers: Vec<(Value, Value)>,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Liveness {
    pub types: BTreeMap<Value, Type>,
    pub blocks: Vec<BlockLiveness>,
    pub edges: Vec<EdgeLiveness>,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Pressure {
    /// Sum of semantic value widths grouped by type. This does not round values
    /// into an ISA register class or assume that simultaneously live values have
    /// distinct physical locations.
    pub bytes_by_type: Vec<(Type, u64)>,
    pub values: usize,
}
impl Liveness {
    pub fn pressure(&self, values: &BTreeSet<Value>) -> Result<Pressure, String> {
        let mut bytes_by_type: Vec<(Type, u64)> = Vec::new();
        for value in values {
            let ty = *self
                .types
                .get(value)
                .ok_or("liveness references an unknown value")?;
            if let Some((_, bytes)) = bytes_by_type.iter_mut().find(|(kind, _)| *kind == ty) {
                *bytes = bytes
                    .checked_add(u64::from(ty.bytes()))
                    .ok_or("typed live storage overflow")?;
            } else {
                bytes_by_type.push((ty, u64::from(ty.bytes())));
            }
        }
        Ok(Pressure {
            bytes_by_type,
            values: values.len(),
        })
    }
}

pub fn scalar(program: &ScalarProgram) -> Result<Liveness, String> {
    cranelift_codegen::verify_function(
        &program.function,
        &cranelift_codegen::settings::Flags::new(cranelift_codegen::settings::builder()),
    )
    .map_err(|error| format!("invalid scalar liveness input: {error}"))?;
    let graph = Graph::scalar(program);
    if !graph.unavailable.is_empty() {
        return Err("liveness requires admitted scalar effects and CFG edges".into());
    }
    let mut types = BTreeMap::new();
    for block in &graph.blocks {
        types.extend(block.parameters.iter().copied());
    }
    for instruction in &graph.instructions {
        if instruction.opcode.is_branch()
            && !matches!(instruction.opcode, Opcode::Jump | Opcode::Brif)
        {
            return Err("liveness requires an admitted scalar branch".into());
        }
        types.extend(instruction.outputs.iter().copied());
    }
    if types
        .values()
        .any(|ty| ty.is_dynamic_vector() || ty.bytes() == 0)
    {
        return Err("liveness requires fixed nonzero-width value types".into());
    }
    let mut entries: BTreeMap<_, BTreeSet<_>> = graph
        .blocks
        .iter()
        .map(|block| (block.id, BTreeSet::new()))
        .collect();
    // Monotone transfer over a finite value lattice. Loops converge without
    // iteration caps or widening; simultaneous phi transfer is edge-specific.
    loop {
        let mut changed = false;
        for block in graph.blocks.iter().rev() {
            let mut live = exit(&graph, block.id, &entries)?;
            for &index in block.instructions.iter().rev() {
                transfer(&graph.instructions[index], &mut live);
            }
            if live != entries[&block.id] {
                entries.insert(block.id, live);
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    let mut blocks = Vec::new();
    for block in &graph.blocks {
        let exit = exit(&graph, block.id, &entries)?;
        let mut live = exit.clone();
        let mut points = Vec::new();
        for &index in block.instructions.iter().rev() {
            let instruction = &graph.instructions[index];
            let after = live.clone();
            transfer(instruction, &mut live);
            points.push(Point {
                instruction: instruction.id,
                before: live.clone(),
                after,
            });
        }
        points.reverse();
        blocks.push(BlockLiveness {
            block: block.id,
            entry: entries[&block.id].clone(),
            exit,
            points,
        });
    }
    let edges = graph
        .edges
        .iter()
        .map(|edge| {
            let destination = entries
                .get(&edge.destination)
                .ok_or("absent liveness successor")?;
            let mut values = destination.clone();
            let mut transfers = Vec::new();
            // Remove all destination names before inserting any source. Sequential
            // substitution would corrupt loop-carried permutations such as a,b=b,a.
            for &(_, target) in &edge.arguments {
                values.remove(&target);
            }
            for &(source, target) in &edge.arguments {
                if destination.contains(&target) {
                    values.insert(source);
                    transfers.push((source, target));
                }
            }
            Ok(EdgeLiveness {
                from: edge.from,
                terminator: edge.terminator,
                destination: edge.destination,
                values,
                transfers,
            })
        })
        .collect::<Result<Vec<_>, String>>()?;
    Ok(Liveness {
        types,
        blocks,
        edges,
    })
}
fn transfer(instruction: &Instruction, live: &mut BTreeSet<Value>) {
    for &(result, _) in &instruction.outputs {
        live.remove(&result);
    }
    // Branch arguments are consumed by the selected edge's simultaneous transfer,
    // not unconditionally by the branch instruction itself.
    let inputs = match instruction.opcode {
        Opcode::Jump => 0,
        Opcode::Brif => 1,
        _ => instruction.inputs.len(),
    };
    live.extend(
        instruction
            .inputs
            .iter()
            .take(inputs)
            .map(|input| input.value),
    );
}
fn exit(
    graph: &Graph,
    block: Block,
    entries: &BTreeMap<Block, BTreeSet<Value>>,
) -> Result<BTreeSet<Value>, String> {
    let mut result = BTreeSet::new();
    for edge in graph.edges.iter().filter(|edge| edge.from == block) {
        let successor = entries
            .get(&edge.destination)
            .ok_or("absent liveness successor")?;
        let mut live = successor.clone();
        for &(_, target) in &edge.arguments {
            live.remove(&target);
        }
        for &(source, target) in &edge.arguments {
            if successor.contains(&target) {
                live.insert(source);
            }
        }
        result.extend(live);
    }
    Ok(result)
}
