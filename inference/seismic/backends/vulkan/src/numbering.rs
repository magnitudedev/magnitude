//! Value numbering of the analysed copy of a module (`formation::analysed`),
//! the one owner of its value identity: in the entry point, an instruction
//! computing what an instruction dominating it computed is removed, and its
//! result is that instruction's wherever it is named ([`numbered`]). The
//! checks that read the copy (`seal::conditional_subtractions`,
//! `uniformity`) match values by id, so one value is one id.
//!
//! In each function (the entry point, the hardware joins and the
//! summarised functions, `uniformity::summarised`), two instructions compute
//! one value where SPIRV-Tools' value numbering
//! (`ValueNumberTable`, which its `RedundancyElimination` pass applies)
//! has them do so, as it numbers a shader, but for loads (below):
//!
//! - an operation it numbers as a value ([`combinator`], [`glsl_combinator`])
//!   of one opcode, result type and literal operands, whose id operands are
//!   the same values, decorated alike; operand order is kept, as it keeps
//!   it (commutative operations are not normalised);
//! - a load of memory nothing in the launch writes, as the analysis reads
//!   it ([`numbered`]), is such an operation of its pointer: a load of
//!   other memory is a value of its own;
//! - a copy (`OpCopyObject`) is the value it copies, and a phi whose
//!   incoming values, each numbered already in the order of the blocks, are
//!   one value is that value, both where decorated as their (first) operand
//!   is;
//! - where `calls` holds, a call of one function with the same arguments,
//!   where neither its function nor any it calls reads workgroup or device
//!   memory (`uniformity::shared_readers`): a summarised function computes
//!   one value of its arguments and the memory it reads, and only read-only
//!   memory is one memory at both calls. The
//!   copy the contract's value rules read numbers them so, as the inlined
//!   bodies of a straight-line function would be; the copy the execution
//!   model is checked on does not, so its calls are values of their own,
//!   which can only make it refuse more (`uniformity`, "Summarised
//!   functions").
//!
//! This is what the module's ids are. What an invocation holds where some
//! edges of the module are taken by none (`uniformity::Flow::copies`, after
//! constant branches are pruned) is the analysis's, not an identity of
//! values: a phi whose live incoming values agree holds that value there.
//!
//! Each value is numbered once, in the order of the blocks, in one table;
//! then one walk of the dominator tree holds, for each value, the first
//! instruction on the path from the entry computing it, and undoes what a
//! subtree held on leaving it. SPIRV-Tools' pass copies its table of held
//! values into every block, which over a fully inlined kernel of thousands
//! of blocks reaches gigabytes; this holds each value once.

use crate::seal::{self, Instruction, MalformedModule};
use crate::uniformity::{
    shape, shared_readers, Dominance, Place, Program, Shape, OP_COPY_OBJECT, OP_DECORATE, OP_EXT_INST, OP_FUNCTION, OP_FUNCTION_CALL,
    OP_FUNCTION_END, OP_FUNCTION_PARAMETER, OP_LOAD, OP_NAME, OP_PHI, OP_UNDEF,
};
use std::collections::hash_map::Entry;
use std::collections::{HashMap, HashSet};

const OP_EXT_INST_IMPORT: u32 = 11;
const OP_GROUP_DECORATE: u32 = 74;
const OP_DECORATE_ID: u32 = 332;
const OP_DECORATE_STRING: u32 = 5632;

/// The operations SPIRV-Tools numbers as values in a shader (its
/// combinators, `IRContext::AddCombinatorsForCapability`) that the analysis
/// reads (`uniformity::shape`): `OpUndef`; `OpAccessChain`,
/// `OpInBoundsAccessChain`, `OpArrayLength`; dynamic vector extraction and
/// insertion, `OpVectorShuffle`, composite construction, extraction and
/// insertion, `OpCopyObject`, `OpTranspose`; the numeric conversions,
/// `OpQuantizeToF16` and `OpBitcast`; arithmetic, the vector and matrix
/// products, `OpDot` and the extended arithmetic; `OpAny`, `OpAll`,
/// `OpIsNan`, `OpIsInf`; the logical operations, `OpSelect` and the
/// compares; shifts and bit operations; `OpPhi`. `OpLoad` and `OpExtInst`
/// are numbered as [`numbered`] says. Nothing with an effect, nor a
/// variable, call, atomic, image or subgroup operation or barrier, is one.
fn combinator(opcode: u32) -> bool {
    matches!(
        opcode,
        OP_UNDEF | 65 | 66 | 68 | 77..=84 | 109..=116 | 124 | 126..=152 | 154..=157 | 164..=191 | 194..=205 | OP_PHI
    )
}

/// The GLSL.std.450 instructions SPIRV-Tools numbers as values
/// (`IRContext::AddCombinatorsForExtension`): all but `Modf` and `Frexp`,
/// which write through a pointer, and the `InterpolateAt*` instructions.
fn glsl_combinator(instruction: u32) -> bool {
    matches!(instruction, 1..=34 | 36..=50 | 52..=75 | 79..=81)
}

/// The shape of an instruction of an entry point the analysis reads.
fn known(opcode: u32) -> Shape {
    shape(opcode).expect("an entry point the analysis reads holds instructions of a known shape")
}

/// `module`, the analysed copy, with each instruction of a function that
/// computes what an instruction dominating it computed is removed, its
/// result replaced by that instruction's in every id operand, and its names
/// and decorations dropped: the instruction kept is decorated alike, and
/// keeps its own name. Uses of a removed result are dominated by it, so by
/// the one kept: the module stays valid. Where `calls` holds, a call that
/// reads no workgroup or device memory is numbered as an operation of its
/// function and arguments.
///
/// A load is numbered where it reads memory nothing in the launch writes,
/// as the analysis reads it (`uniformity::Place`, read-only or built-in:
/// push-constant, input, uniform-constant and uniform memory but a
/// `BufferBlock`'s), the one test of that. Storage-buffer and physical
/// storage-buffer memory is not, whatever its pointee is decorated: two
/// loads of it are two values, as the analysis takes them.
///
/// A function holding an instruction the analysis does not read is left as
/// it is, its operands never guessed at; where it is the entry point, so is
/// the module (`uniformity::check` refuses it unanalysed). An id a group
/// decoration or a module-scope extended instruction names is never
/// removed: it stays where those name it.
pub fn numbered(module: &[u32], calls: bool) -> Result<Vec<u32>, MalformedModule> {
    if !Program::parse(module)?.unread.is_empty() {
        return Ok(module.to_vec());
    }
    let instructions = seal::instructions(module)?;
    let words_of = |instruction: &Instruction| &module[instruction.start..instruction.start + instruction.words];
    // The GLSL.std.450 set.
    let mut glsl = None;
    // By id: its decorations, each its opcode and its words past the
    // target, in the module's order.
    let mut decorations = HashMap::<u32, Vec<Vec<u32>>>::new();
    let mut pinned = HashSet::new();
    let mut functions = Vec::new();
    let mut in_function = false;
    for instruction in &instructions {
        let words = words_of(instruction);
        match instruction.opcode {
            OP_EXT_INST_IMPORT if seal::literal_string(&words[2..]) == "GLSL.std.450" => glsl = Some(words[1]),
            OP_DECORATE | OP_DECORATE_ID | OP_DECORATE_STRING => {
                let decoration: Vec<u32> =
                    std::iter::once(instruction.opcode).chain(words[2..].iter().copied()).collect();
                decorations.entry(words[1]).or_default().push(decoration);
            }
            OP_GROUP_DECORATE => pinned.extend(words[2..].iter().copied()),
            OP_EXT_INST if !in_function => pinned.extend(known(OP_EXT_INST).positions(words).map(|at| words[at])),
            OP_FUNCTION => {
                in_function = true;
                functions.push(words[2]);
            }
            OP_FUNCTION_END => in_function = false,
            _ => {}
        }
    }
    // Decorated alike whatever their order, as SPIRV-Tools compares them.
    for list in decorations.values_mut() {
        list.sort_unstable();
    }
    let decorations_of = |id: u32| decorations.get(&id).map_or(&[][..], Vec::as_slice);
    let programs = functions
        .into_iter()
        .map(|function| Program::parse_at(module, function).map(|program| (function, program)))
        .collect::<Result<HashMap<_, _>, _>>()?;
    // The functions whose calls read workgroup or device memory.
    let reading = shared_readers(&programs);
    // By result removed: the result of the instruction kept for its value.
    let mut replaced = HashMap::new();
    for program in programs.values() {
        if !program.unread.is_empty() {
            continue;
        }
        // By result of the function: its value, the result of the first
        // instruction (in the order of the blocks) computing it. Any other
        // id is a value of its own; a result not yet numbered has none.
        let mut values = HashMap::<u32, u32>::new();
        let value_of = |values: &HashMap<u32, u32>, id: u32| {
            values.get(&id).copied().or_else(|| (!program.defined.contains_key(&id)).then_some(id))
        };
        // An operation (its words, its result zeroed and its id operands
        // replaced by their values where numbered) with its result's
        // decorations → its value.
        let mut table = HashMap::<(Vec<u32>, &[Vec<u32>]), u32>::new();
        for block in &program.blocks {
            for words in &block.instructions {
                let opcode = words[0] & 0xffff;
                let shape = known(opcode);
                if !shape.result {
                    continue;
                }
                let id = words[2];
                let numbered = !pinned.contains(&id)
                    && match opcode {
                        // A load of memory nothing in the launch writes, as
                        // the analysis reads it.
                        OP_LOAD => matches!(program.place(words[3])?, Place::ReadOnly | Place::BuiltIn(_)),
                        OP_EXT_INST => glsl == Some(words[3]) && glsl_combinator(words[4]),
                        OP_FUNCTION_CALL => calls && !reading.contains(&words[3]),
                        _ => combinator(opcode),
                    };
                let value = if numbered {
                    let own = decorations_of(id);
                    let (sources, step) = match opcode {
                        OP_COPY_OBJECT => (&words[3..], 1),
                        OP_PHI => (&words[3..], 2),
                        _ => (&[][..], 1),
                    };
                    let copied = sources.first().filter(|first| decorations_of(**first) == own).and_then(|first| {
                        let held = value_of(&values, *first)?;
                        sources.iter().step_by(step).all(|source| value_of(&values, *source) == Some(held)).then_some(held)
                    });
                    copied.unwrap_or_else(|| {
                        let mut operation = words.to_vec();
                        operation[2] = 0;
                        for at in shape.positions(words) {
                            operation[at] = value_of(&values, words[at]).unwrap_or(words[at]);
                        }
                        *table.entry((operation, own)).or_insert(id)
                    })
                } else {
                    id
                };
                values.insert(id, value);
            }
        }

        // By value: the result holding it on the path from the entry to the
        // block walked, and the values each block on it took to hold, as the
        // block and where its values start in `log`.
        let mut held = HashMap::new();
        let (mut log, mut path) = (Vec::new(), Vec::<(usize, usize)>::new());
        let dominance = Dominance::new(&mut program.successors());
        for &block in &dominance.preorder {
            while let Some(&(above, start)) = path.last() {
                if dominance.dominates(above, block) {
                    break;
                }
                for value in log.drain(start..) {
                    held.remove(&value);
                }
                path.pop();
            }
            path.push((block, log.len()));
            for words in &program.blocks[block].instructions {
                if !known(words[0] & 0xffff).result {
                    continue;
                }
                let value = values[&words[2]];
                match held.entry(value) {
                    Entry::Occupied(holder) => {
                        replaced.insert(words[2], *holder.get());
                    }
                    Entry::Vacant(vacant) => {
                        vacant.insert(words[2]);
                        log.push(value);
                    }
                }
            }
        }
    }
    if replaced.is_empty() {
        return Ok(module.to_vec());
    }

    let mut numbered = module[..5].to_vec();
    let mut in_function = false;
    for instruction in &instructions {
        let words = words_of(instruction);
        let opcode = instruction.opcode;
        match opcode {
            OP_FUNCTION => in_function = true,
            OP_FUNCTION_END => in_function = false,
            _ => {}
        }
        // A function's own instructions, past its declaration and
        // parameters, whose shape the analysis reads.
        let body = in_function && !matches!(opcode, OP_FUNCTION | OP_FUNCTION_PARAMETER);
        match shape(opcode).filter(|_| body) {
            Some(shape) if shape.result && replaced.contains_key(&words[2]) => {}
            Some(shape) => {
                let start = numbered.len();
                numbered.extend_from_slice(words);
                for at in shape.positions(words) {
                    if let Some(&holder) = replaced.get(&words[at]) {
                        numbered[start + at] = holder;
                    }
                }
            }
            None if matches!(opcode, OP_NAME | OP_DECORATE | OP_DECORATE_ID | OP_DECORATE_STRING)
                && replaced.contains_key(&words[1]) => {}
            None => numbered.extend_from_slice(words),
        }
    }
    Ok(numbered)
}

#[cfg(test)]
mod tests {
    use super::numbered;
    use crate::seal;

    /// An operation computed again where its first computation dominates it
    /// is that one, its result replaced where named and its name dropped;
    /// one computed again in a block its first does not dominate (the other
    /// arm of a selection) is kept.
    #[test]
    fn a_value_computed_again_where_its_first_dominates_is_its_first() {
        fn op(module: &mut Vec<u32>, opcode: u32, operands: &[u32]) {
            module.push(((operands.len() as u32 + 1) << 16) | opcode);
            module.extend_from_slice(operands);
        }
        // "main" and "b", each its terminating null in the word after.
        let (main_name, b_name) = (u32::from_le_bytes(*b"main"), u32::from_le_bytes(*b"b\0\0\0"));
        let (void, signature, int, boolean, truth, seven, main) = (1, 2, 3, 4, 5, 6, 7);
        let (entry, then, otherwise, merge) = (8, 9, 10, 11);
        // `a` and `b` add, `c` and `d` multiply, `seven` by itself.
        let (a, b, c, d, phi) = (12, 13, 14, 15, 16);
        let mut module = vec![0x0723_0203, 0x0001_0300, 0, 17, 0];
        // `OpCapability Shader`, `OpMemoryModel Logical GLSL450`,
        // `OpEntryPoint GLCompute`, `OpExecutionMode LocalSize 1 1 1`.
        op(&mut module, 17, &[1]);
        op(&mut module, 14, &[0, 1]);
        op(&mut module, 15, &[5, main, main_name, 0]);
        op(&mut module, 16, &[main, 17, 1, 1, 1]);
        op(&mut module, 5, &[main, main_name, 0]);
        op(&mut module, 5, &[b, b_name]);
        op(&mut module, 19, &[void]);
        op(&mut module, 33, &[signature, void]);
        op(&mut module, 21, &[int, 32, 0]);
        op(&mut module, 20, &[boolean]);
        op(&mut module, 41, &[boolean, truth]);
        op(&mut module, 43, &[int, seven, 7]);
        op(&mut module, 54, &[void, main, 0, signature]);
        op(&mut module, 248, &[entry]);
        op(&mut module, 128, &[int, a, seven, seven]);
        op(&mut module, 247, &[merge, 0]);
        op(&mut module, 250, &[truth, then, otherwise]);
        op(&mut module, 248, &[then]);
        op(&mut module, 128, &[int, b, seven, seven]);
        op(&mut module, 132, &[int, c, seven, seven]);
        op(&mut module, 249, &[merge]);
        op(&mut module, 248, &[otherwise]);
        op(&mut module, 132, &[int, d, seven, seven]);
        op(&mut module, 249, &[merge]);
        op(&mut module, 248, &[merge]);
        op(&mut module, 245, &[int, phi, b, then, d, otherwise]);
        op(&mut module, 253, &[]);
        op(&mut module, 56, &[]);

        let numbered = numbered(&module, false).expect("module");
        let instructions = seal::instructions(&numbered).expect("module");
        let words = instructions
            .iter()
            .map(|instruction| &numbered[instruction.start..instruction.start + instruction.words])
            .collect::<Vec<_>>();
        let results = words.iter().filter(|words| matches!(words[0] & 0xffff, 128 | 132)).map(|words| words[2]);
        assert_eq!(results.collect::<Vec<_>>(), [a, c, d]);
        assert!(words.iter().any(|words| words[..] == [(7 << 16) | 245, int, phi, a, then, d, otherwise]));
        assert!(!words.iter().any(|words| words[0] & 0xffff == 5 && words[1] == b));
        // Nothing else changes.
        assert_eq!(numbered.len(), module.len() - 5 - 3);
    }
}
