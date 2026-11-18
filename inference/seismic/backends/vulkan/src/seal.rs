//! Seismic's SPIR-V seal pass (§7.2 step 3, §7.3): every floating-point
//! arithmetic result is decorated `NoContraction`, and the entry point runs
//! with RTE rounding and signed-zero/Inf/NaN preservation for fp32, and for
//! fp16 on devices with fp16 arithmetic, plus `DenormPreserve 32` where the
//! device supports it. `RoundingModeRTE 32` is left out where the device's
//! `rounding_rte_32` property does not hold (its `rte_32_division` probe has
//! not shown the mode forming and rounding as declared; the NVIDIA 580
//! driver's compiler traps on fp32 `OpFDiv` under it); there fp32 rounds by
//! the device's default.
//!
//! glslang has no global switch for this, and `precise` is per object.
//!
//! [`contract`] reports the constructs a formed module may not contain.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

/// Version of these passes and of the emitted contract; part of the
/// formation identity and SPIR-V cache key, so a kept module was formed
/// under the current contract.
pub const SEAL_VERSION: u32 = 6;

/// The execution modes a sealed module declares beyond the fixed ones.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Environment {
    /// The device has fp16 arithmetic: `RoundingModeRTE 16` and
    /// `SignedZeroInfNanPreserve 16` are declared.
    pub float16: bool,
    /// `RoundingModeRTE 32` is declared. When false, fp32 add, multiply and
    /// conversion round by the device's default.
    pub rounding_rte_32: bool,
    /// `DenormPreserve 32` is declared.
    pub denorm_preserve_32: bool,
}

const OP_EXTENSION: u32 = 10;
const OP_EXT_INST_IMPORT: u32 = 11;
const OP_EXT_INST: u32 = 12;
/// `OpFmaKHR` and `FMAKHR` of `SPV_KHR_fma`, which the linked SPIRV-Tools
/// predates.
const OP_FMA_KHR: u32 = 4427;
const CAPABILITY_FMA_KHR: u32 = 6030;
const GLSL_STD_450_FMA: u32 = 50;

const OP_CAPABILITY: u32 = 17;
const OP_EXECUTION_MODE: u32 = 16;
const OP_ENTRY_POINT: u32 = 15;
const OP_DECORATE: u32 = 71;

const DECORATION_NO_CONTRACTION: u32 = 42;

const MODE_DENORM_PRESERVE: u32 = 4459;
const MODE_SIGNED_ZERO_INF_NAN_PRESERVE: u32 = 4461;
const MODE_ROUNDING_MODE_RTE: u32 = 4462;
const CAPABILITY_DENORM_PRESERVE: u32 = 4464;
const CAPABILITY_SIGNED_ZERO_INF_NAN_PRESERVE: u32 = 4466;
const CAPABILITY_ROUNDING_MODE_RTE: u32 = 4467;

/// Floating-point arithmetic whose result is contractible: `OpFNegate` is
/// exact; `OpFAdd`, `OpFSub`, `OpFMul`, `OpFDiv`, `OpFRem`, `OpFMod`,
/// `OpVectorTimesScalar`, `OpMatrixTimesScalar`, `OpVectorTimesMatrix`,
/// `OpMatrixTimesVector`, `OpMatrixTimesMatrix`, `OpOuterProduct`, `OpDot`.
const FLOAT_ARITHMETIC: [u32; 13] = [
    129, 131, 133, 136, 140, 141, 142, 143, 144, 145, 146, 147, 148,
];

/// Instructions of the logical-layout sections before the annotations:
/// capabilities, extensions, imports, memory model, entry points, execution
/// modes, and debug instructions (`OpSourceContinued`, `OpSource`,
/// `OpSourceExtension`, `OpName`, `OpMemberName`, `OpString`, `OpLine`,
/// `OpNoLine`, `OpModuleProcessed`, `OpExecutionModeId`), then annotations
/// (`OpDecorate`, `OpMemberDecorate`, `OpDecorationGroup`, `OpGroupDecorate`,
/// `OpGroupMemberDecorate`, `OpDecorateId`, `OpDecorateString`,
/// `OpMemberDecorateString`).
const HEADER_SECTIONS: [u32; 24] = [
    17, 10, 11, 14, 15, 16, 2, 3, 4, 5, 6, 7, 8, 317, 330, 331, 71, 72, 73, 74, 75, 332, 5632, 5633,
];

#[derive(Debug, PartialEq, Eq)]
pub struct MalformedModule(pub String);

struct Instruction {
    start: usize,
    opcode: u32,
    words: usize,
}

fn instructions(module: &[u32]) -> Result<Vec<Instruction>, MalformedModule> {
    if module.len() < 5 || module[0] != 0x0723_0203 {
        return Err(MalformedModule("not a SPIR-V module".into()));
    }
    let mut instructions = Vec::new();
    let mut at = 5;
    while at < module.len() {
        let words = (module[at] >> 16) as usize;
        if words == 0 || at + words > module.len() {
            return Err(MalformedModule(format!(
                "instruction at word {at} has length {words}"
            )));
        }
        instructions.push(Instruction {
            start: at,
            opcode: module[at] & 0xffff,
            words,
        });
        at += words;
    }
    Ok(instructions)
}

/// Seal `module` for Seismic's numerical environment.
pub fn seal(module: &[u32], environment: Environment) -> Result<Vec<u32>, MalformedModule> {
    let instructions = instructions(module)?;
    let words = |instruction: &Instruction| {
        &module[instruction.start..instruction.start + instruction.words]
    };
    let entry = instructions
        .iter()
        .find(|instruction| instruction.opcode == OP_ENTRY_POINT)
        .map(|instruction| words(instruction)[2])
        .ok_or_else(|| MalformedModule("no entry point".into()))?;
    let capabilities = instructions
        .iter()
        .filter(|instruction| instruction.opcode == OP_CAPABILITY)
        .map(|instruction| words(instruction)[1])
        .collect::<Vec<_>>();
    let modes = instructions
        .iter()
        .filter(|instruction| instruction.opcode == OP_EXECUTION_MODE)
        .map(|instruction| words(instruction)[2..].to_vec())
        .collect::<Vec<_>>();
    let decorated = instructions
        .iter()
        .filter(|instruction| {
            instruction.opcode == OP_DECORATE && words(instruction)[2] == DECORATION_NO_CONTRACTION
        })
        .map(|instruction| words(instruction)[1])
        .collect::<HashSet<_>>();

    let mut wanted_modes = Vec::new();
    if environment.float16 {
        wanted_modes.push((MODE_ROUNDING_MODE_RTE, 16));
        wanted_modes.push((MODE_SIGNED_ZERO_INF_NAN_PRESERVE, 16));
    }
    wanted_modes.push((MODE_SIGNED_ZERO_INF_NAN_PRESERVE, 32));
    let mut wanted_capabilities = Vec::new();
    if environment.float16 || environment.rounding_rte_32 {
        wanted_capabilities.push(CAPABILITY_ROUNDING_MODE_RTE);
    }
    wanted_capabilities.push(CAPABILITY_SIGNED_ZERO_INF_NAN_PRESERVE);
    if environment.rounding_rte_32 {
        wanted_modes.push((MODE_ROUNDING_MODE_RTE, 32));
    }
    if environment.denorm_preserve_32 {
        wanted_modes.push((MODE_DENORM_PRESERVE, 32));
        wanted_capabilities.push(CAPABILITY_DENORM_PRESERVE);
    }
    let new_capabilities = wanted_capabilities
        .into_iter()
        .filter(|capability| !capabilities.contains(capability))
        .flat_map(|capability| [(2 << 16) | OP_CAPABILITY, capability])
        .collect::<Vec<_>>();
    let new_modes = wanted_modes
        .into_iter()
        .filter(|(mode, width)| !modes.contains(&vec![*mode, *width]))
        .flat_map(|(mode, width)| [(4 << 16) | OP_EXECUTION_MODE, entry, mode, width])
        .collect::<Vec<_>>();
    let new_decorations = instructions
        .iter()
        .filter(|instruction| FLOAT_ARITHMETIC.contains(&instruction.opcode))
        .map(|instruction| words(instruction)[2])
        .filter(|result| !decorated.contains(result))
        .flat_map(|result| [(3 << 16) | OP_DECORATE, result, DECORATION_NO_CONTRACTION])
        .collect::<Vec<_>>();

    // Capabilities go after the last capability, execution modes after the
    // last entry point or execution mode, decorations before the first
    // instruction past the annotation section.
    let after = |opcodes: &[u32]| {
        instructions
            .iter()
            .filter(|instruction| opcodes.contains(&instruction.opcode))
            .map(|instruction| instruction.start + instruction.words)
            .max()
    };
    let capability_at =
        after(&[OP_CAPABILITY]).ok_or_else(|| MalformedModule("no capability".into()))?;
    let mode_at = after(&[OP_ENTRY_POINT, OP_EXECUTION_MODE]).expect("an entry point was found");
    let decoration_at = instructions
        .iter()
        .find(|instruction| !HEADER_SECTIONS.contains(&instruction.opcode))
        .map_or(module.len(), |instruction| instruction.start);

    let mut sealed = Vec::with_capacity(
        module.len() + new_capabilities.len() + new_modes.len() + new_decorations.len(),
    );
    sealed.extend_from_slice(&module[..capability_at]);
    sealed.extend_from_slice(&new_capabilities);
    sealed.extend_from_slice(&module[capability_at..mode_at]);
    sealed.extend_from_slice(&new_modes);
    sealed.extend_from_slice(&module[mode_at..decoration_at]);
    sealed.extend_from_slice(&new_decorations);
    sealed.extend_from_slice(&module[decoration_at..]);
    Ok(sealed)
}

fn literal_string(words: &[u32]) -> String {
    let bytes = words
        .iter()
        .flat_map(|word| word.to_le_bytes())
        .take_while(|byte| *byte != 0)
        .collect::<Vec<_>>();
    String::from_utf8_lossy(&bytes).into_owned()
}

fn string_words(text: &str) -> Vec<u32> {
    let mut bytes = text.as_bytes().to_vec();
    bytes.resize((bytes.len() / 4 + 1) * 4, 0);
    bytes
        .chunks_exact(4)
        .map(|word| u32::from_le_bytes(word.try_into().expect("four bytes")))
        .collect()
}

/// Bind every `GLSL.std.450 Fma` to `OpFmaKHR` (`SPV_KHR_fma`), which the
/// precision rules require to be one correctly rounded operation. glslang
/// cannot emit it from GLSL, and a driver may split `Fma` into a multiply
/// and an add (RADV next to a `NoContraction` product of the same operands,
/// §16.1). Run after validation: the linked SPIRV-Tools predates the opcode.
pub fn fma_khr(module: &[u32]) -> Result<Vec<u32>, MalformedModule> {
    let instructions = instructions(module)?;
    let words = |instruction: &Instruction| {
        &module[instruction.start..instruction.start + instruction.words]
    };
    let import = instructions
        .iter()
        .find(|instruction| {
            instruction.opcode == OP_EXT_INST_IMPORT
                && literal_string(&words(instruction)[2..]) == "GLSL.std.450"
        })
        .map(|instruction| words(instruction)[1]);
    let capability_at = instructions
        .iter()
        .filter(|instruction| instruction.opcode == OP_CAPABILITY)
        .map(|instruction| instruction.start + instruction.words)
        .max()
        .ok_or_else(|| MalformedModule("no capability".into()))?;
    let mut rewritten = Vec::with_capacity(module.len() + 8);
    rewritten.extend_from_slice(&module[..capability_at]);
    rewritten.extend([(2 << 16) | OP_CAPABILITY, CAPABILITY_FMA_KHR]);
    let name = string_words("SPV_KHR_fma");
    rewritten.push(((1 + name.len() as u32) << 16) | OP_EXTENSION);
    rewritten.extend(name);
    for instruction in instructions
        .iter()
        .filter(|instruction| instruction.start >= capability_at)
    {
        let words = words(instruction);
        if instruction.opcode == OP_EXT_INST
            && Some(words[3]) == import
            && words[4] == GLSL_STD_450_FMA
        {
            // Result type, result id, a, b, c.
            rewritten.extend([
                (6 << 16) | OP_FMA_KHR,
                words[1],
                words[2],
                words[5],
                words[6],
                words[7],
            ]);
        } else {
            rewritten.extend_from_slice(words);
        }
    }
    Ok(rewritten)
}

const OP_NAME: u32 = 5;
const OP_TYPE_INT: u32 = 21;
const OP_TYPE_FLOAT: u32 = 22;
const OP_TYPE_VECTOR: u32 = 23;
const OP_TYPE_ARRAY: u32 = 28;
const OP_TYPE_POINTER: u32 = 32;
const OP_CONSTANT_TRUE: u32 = 41;
const OP_CONSTANT_FALSE: u32 = 42;
const OP_CONSTANT: u32 = 43;
const OP_CONSTANT_COMPOSITE: u32 = 44;
const OP_CONSTANT_NULL: u32 = 46;
/// `OpSpecConstantTrue` through `OpSpecConstantOp`.
const SPEC_CONSTANTS: std::ops::RangeInclusive<u32> = 48..=52;
const OP_FUNCTION: u32 = 54;
const OP_FUNCTION_PARAMETER: u32 = 55;
const OP_FUNCTION_END: u32 = 56;
const OP_FUNCTION_CALL: u32 = 57;
const OP_VARIABLE: u32 = 59;
const OP_STORE: u32 = 62;
const OP_COMPOSITE_CONSTRUCT: u32 = 80;
const OP_COMPOSITE_INSERT: u32 = 82;
const OP_COPY_OBJECT: u32 = 83;
const OP_SELECT: u32 = 169;
const OP_PHI: u32 = 245;
const OP_RETURN_VALUE: u32 = 254;
const OP_TYPE_COOPERATIVE_MATRIX: u32 = 4456;
const OP_COOPERATIVE_MATRIX_MUL_ADD: u32 = 4459;

/// Instructions whose result is a pointer a store can address:
/// `OpFunctionParameter`, `OpVariable`, `OpAccessChain`,
/// `OpInBoundsAccessChain`, `OpPtrAccessChain`, `OpCopyObject`.
const POINTERS: [u32; 6] = [
    OP_FUNCTION_PARAMETER,
    OP_VARIABLE,
    65,
    66,
    67,
    OP_COPY_OBJECT,
];

/// `OpUDiv`, `OpSDiv`, `OpUMod`, `OpSRem`, `OpSMod`.
const INTEGER_DIVISION: [(u32, &str); 5] = [
    (134, "OpUDiv"),
    (135, "OpSDiv"),
    (137, "OpUMod"),
    (138, "OpSRem"),
    (139, "OpSMod"),
];

/// `OpConvertFToU`, `OpConvertFToS`.
const FLOAT_TO_INTEGER: [(u32, &str); 2] = [(109, "OpConvertFToU"), (110, "OpConvertFToS")];

const STORAGE_PRIVATE: u32 = 6;
const STORAGE_FUNCTION: u32 = 7;

/// A construct the emitted contract forbids, with how often and where it
/// occurs.
#[derive(Debug, PartialEq, Eq)]
pub struct Violation(pub String);

/// A constant's value: equal values compare equal whatever their ids.
#[derive(Clone, Debug, PartialEq, Eq)]
enum Value {
    /// A scalar's type and literal words.
    Scalar(u32, Vec<u32>),
    /// A composite's type and constituents.
    Composite(u32, Vec<Value>),
    /// Known by id only: a specialization constant, or the zero of an
    /// array whose length is one.
    Opaque(u32),
}

/// What [`contract`] reads of a module's declarations.
struct Declarations<'a> {
    names: HashMap<u32, String>,
    int_widths: HashMap<u32, u32>,
    float_widths: HashMap<u32, u32>,
    /// Vector type → (component type, count).
    vectors: HashMap<u32, (u32, u32)>,
    /// Array type → (element type, length id).
    arrays: HashMap<u32, (u32, u32)>,
    /// Cooperative-matrix types.
    matrices: HashSet<u32>,
    /// Pointer type → storage class.
    storage: HashMap<u32, u32>,
    /// Pointer value → its type.
    pointer_types: HashMap<u32, u32>,
    /// Constants, copies and constructions, by result id.
    values: HashMap<u32, &'a [u32]>,
    /// Function parameter → (function, position).
    parameters: HashMap<u32, (u32, usize)>,
    /// (Function, position) → the arguments its calls pass there.
    arguments: HashMap<(u32, usize), Vec<u32>>,
}

impl<'a> Declarations<'a> {
    fn of(module: &'a [u32], instructions: &[Instruction]) -> Self {
        let mut declarations = Self {
            names: HashMap::new(),
            int_widths: HashMap::new(),
            float_widths: HashMap::new(),
            vectors: HashMap::new(),
            arrays: HashMap::new(),
            matrices: HashSet::new(),
            storage: HashMap::new(),
            pointer_types: HashMap::new(),
            values: HashMap::new(),
            parameters: HashMap::new(),
            arguments: HashMap::new(),
        };
        let mut function = (0, 0);
        for instruction in instructions {
            let words = &module[instruction.start..instruction.start + instruction.words];
            match instruction.opcode {
                OP_FUNCTION => function = (words[2], 0),
                OP_FUNCTION_PARAMETER => {
                    declarations.parameters.insert(words[2], function);
                    function.1 += 1;
                }
                OP_FUNCTION_CALL => {
                    for (position, argument) in words[4..].iter().enumerate() {
                        declarations
                            .arguments
                            .entry((words[3], position))
                            .or_default()
                            .push(*argument);
                    }
                }
                OP_NAME => {
                    declarations
                        .names
                        .insert(words[1], literal_string(&words[2..]));
                }
                OP_TYPE_INT => {
                    declarations.int_widths.insert(words[1], words[2]);
                }
                OP_TYPE_FLOAT => {
                    declarations.float_widths.insert(words[1], words[2]);
                }
                OP_TYPE_VECTOR => {
                    declarations.vectors.insert(words[1], (words[2], words[3]));
                }
                OP_TYPE_ARRAY => {
                    declarations.arrays.insert(words[1], (words[2], words[3]));
                }
                OP_TYPE_COOPERATIVE_MATRIX => {
                    declarations.matrices.insert(words[1]);
                }
                OP_TYPE_POINTER => {
                    declarations.storage.insert(words[1], words[2]);
                }
                _ => {}
            }
            if POINTERS.contains(&instruction.opcode) {
                declarations.pointer_types.insert(words[2], words[1]);
            }
            if matches!(
                instruction.opcode,
                OP_CONSTANT_TRUE
                    | OP_CONSTANT_FALSE
                    | OP_CONSTANT
                    | OP_CONSTANT_COMPOSITE
                    | OP_CONSTANT_NULL
                    | OP_COMPOSITE_CONSTRUCT
                    | OP_COPY_OBJECT
            ) || SPEC_CONSTANTS.contains(&instruction.opcode)
            {
                declarations.values.insert(words[2], words);
            }
        }
        declarations
    }

    /// The integer width of a scalar or vector type.
    fn int_width(&self, ty: u32) -> Option<u32> {
        let component = self
            .vectors
            .get(&ty)
            .map_or(ty, |(component, _)| *component);
        self.int_widths.get(&component).copied()
    }

    /// The value of `id` if it is a constant, through copies and the
    /// construction of composites from constants.
    fn value(&self, id: u32) -> Option<Value> {
        let words = self.values.get(&id)?;
        match words[0] & 0xffff {
            OP_CONSTANT_TRUE => Some(Value::Scalar(words[1], vec![1])),
            OP_CONSTANT_FALSE => Some(Value::Scalar(words[1], vec![0])),
            OP_CONSTANT => Some(Value::Scalar(words[1], words[3..].to_vec())),
            OP_CONSTANT_NULL => Some(self.zero(words[1])),
            OP_CONSTANT_COMPOSITE | OP_COMPOSITE_CONSTRUCT => words[3..]
                .iter()
                .map(|constituent| self.value(*constituent))
                .collect::<Option<_>>()
                .map(|constituents| Value::Composite(words[1], constituents)),
            OP_COPY_OBJECT => self.value(words[3]),
            _ => Some(Value::Opaque(id)),
        }
    }

    /// The zero of `ty`, as its explicit constants spell it.
    fn zero(&self, ty: u32) -> Value {
        if let Some((component, count)) = self.vectors.get(&ty) {
            return Value::Composite(ty, vec![self.zero(*component); *count as usize]);
        }
        if let Some((element, length)) = self.arrays.get(&ty) {
            let length = self
                .values
                .get(length)
                .filter(|words| words[0] & 0xffff == OP_CONSTANT)
                .map(|words| words[3]);
            return match length {
                Some(length) => Value::Composite(ty, vec![self.zero(*element); length as usize]),
                None => Value::Opaque(ty),
            };
        }
        let width = self
            .int_widths
            .get(&ty)
            .or_else(|| self.float_widths.get(&ty));
        let words = width.map_or(1, |width| width.div_ceil(32));
        Value::Scalar(ty, vec![0; words as usize])
    }

    /// Whether every array within `value` is a splat.
    fn table_free(&self, value: &Value) -> bool {
        match value {
            Value::Composite(ty, constituents) => {
                constituents
                    .iter()
                    .all(|constituent| self.table_free(constituent))
                    && (!self.arrays.contains_key(ty)
                        || constituents.windows(2).all(|pair| pair[0] == pair[1]))
            }
            Value::Scalar(..) | Value::Opaque(_) => true,
        }
    }

    /// Whether `id` holds a constant table: is one, copies one, or is a
    /// parameter some call passes one to.
    fn table(&self, id: u32) -> bool {
        let words = self.values.get(&id);
        if let Some(words) = words.filter(|words| words[0] & 0xffff == OP_COPY_OBJECT) {
            return self.table(words[3]);
        }
        if let Some(parameter) = self.parameters.get(&id) {
            return self
                .arguments
                .get(parameter)
                .is_some_and(|arguments| arguments.iter().any(|argument| self.table(*argument)));
        }
        self.value(id).is_some_and(|value| !self.table_free(&value))
    }

    /// Whether `pointer` addresses per-invocation (Function or Private)
    /// storage.
    fn per_invocation(&self, pointer: u32) -> bool {
        self.pointer_types
            .get(&pointer)
            .and_then(|ty| self.storage.get(ty))
            .is_some_and(|storage| matches!(*storage, STORAGE_FUNCTION | STORAGE_PRIVATE))
    }

    /// Whether `id` is a module-scope constant of a cooperative-matrix type.
    fn matrix_constant(&self, id: u32) -> bool {
        self.values.get(&id).is_some_and(|words| {
            let opcode = words[0] & 0xffff;
            (matches!(opcode, OP_CONSTANT_COMPOSITE | OP_CONSTANT_NULL)
                || SPEC_CONSTANTS.contains(&opcode))
                && self.matrices.contains(&words[1])
        })
    }
}

/// The value operands of `words` (an instruction in a function body) that
/// can carry a whole cooperative matrix: what a stored, initialized,
/// returned, passed, selected, merged, copied, inserted or multiplied-into
/// matrix is read from.
fn matrix_operands(words: &[u32]) -> Vec<u32> {
    match words[0] & 0xffff {
        OP_STORE => vec![words[2]],
        OP_VARIABLE if words.len() > 4 => vec![words[4]],
        OP_RETURN_VALUE => vec![words[1]],
        OP_FUNCTION_CALL => words[4..].to_vec(),
        OP_SELECT => words[4..6].to_vec(),
        OP_PHI => words[3..].iter().step_by(2).copied().collect(),
        OP_COPY_OBJECT => vec![words[3]],
        OP_COMPOSITE_INSERT => words[3..5].to_vec(),
        OP_COOPERATIVE_MATRIX_MUL_ADD => words[3..6].to_vec(),
        _ => Vec::new(),
    }
}

/// The constructs of a formed, cleaned `module` the emitted contract
/// forbids:
///
/// - a 64-bit integer division or remainder, scalar or vector: drivers
///   expand each into a long sequence (Mesa's Intel compiler about 2,300
///   instructions, #159), copied by every unrolled loop around it.
///   Arithmetic on specialization constants is folded at pipeline creation
///   and is exempt;
/// - a constant array that is not a splat (elements equal by value), at
///   any depth of a constant, materialized in Function or Private storage by
///   a variable's initializer or a store, directly, through copies, or
///   through a function parameter: glslang copies a runtime-indexed constant
///   table into per-invocation memory at every use. Vectors, zero arrays
///   and splats are working state and stay legal. Lookup data lives in the
///   device's table buffer;
/// - a float converted to a 64-bit integer, scalar or vector: NVIDIA's
///   compiler (580, GB10) converted 2^32 to 2^33 − 1 (`8f4cee7cf2`);
/// - a cooperative-matrix constant read by more than one function: AMD's
///   LLPC compiler faults in `vkCreateComputePipelines` on it
///   (`seismic/hardware/rdna3-vulkan.md`).
///
/// The contract is syntactic: it does not infer a table a program computes
/// element by element.
pub fn contract(module: &[u32]) -> Result<Vec<Violation>, MalformedModule> {
    let instructions = instructions(module)?;
    let declarations = Declarations::of(module, &instructions);
    let mut found = BTreeMap::<(String, String), usize>::new();
    let name = |function: u32| {
        declarations
            .names
            .get(&function)
            .cloned()
            .unwrap_or_else(|| format!("%{function}"))
    };
    // Cooperative-matrix constant → the functions that read it.
    let mut matrix_readers = BTreeMap::<u32, BTreeSet<u32>>::new();
    let mut function = None;
    for instruction in &instructions {
        let words = &module[instruction.start..instruction.start + instruction.words];
        if let Some(function) = function {
            for operand in matrix_operands(words) {
                if declarations.matrix_constant(operand) {
                    matrix_readers.entry(operand).or_default().insert(function);
                }
            }
        }
        let construct = match instruction.opcode {
            OP_FUNCTION => {
                function = Some(words[2]);
                None
            }
            OP_FUNCTION_END => {
                function = None;
                None
            }
            OP_VARIABLE
                if words.len() > 4
                    && matches!(words[3], STORAGE_FUNCTION | STORAGE_PRIVATE)
                    && declarations.table(words[4]) =>
            {
                Some("constant table initializing per-invocation storage".to_owned())
            }
            OP_STORE if declarations.per_invocation(words[1]) && declarations.table(words[2]) => {
                Some("constant table stored to per-invocation storage".to_owned())
            }
            opcode => INTEGER_DIVISION
                .iter()
                .chain(&FLOAT_TO_INTEGER)
                .find(|(forbidden, _)| *forbidden == opcode)
                .filter(|_| declarations.int_width(words[1]) == Some(64))
                .map(|(_, name)| format!("64-bit {name}")),
        };
        if let Some(construct) = construct {
            let holder = function.map_or_else(|| "module scope".to_owned(), name);
            *found.entry((holder, construct)).or_default() += 1;
        }
    }
    for (constant, readers) in matrix_readers {
        if readers.len() > 1 {
            let holders = readers.into_iter().map(name).collect::<Vec<_>>().join("`, `");
            *found
                .entry((
                    holders,
                    format!("cooperative-matrix constant %{constant} read by more than one function"),
                ))
                .or_default() += 1;
        }
    }
    Ok(found
        .into_iter()
        .map(|((holder, construct), count)| {
            Violation(format!("{count} × {construct} in `{holder}`"))
        })
        .collect())
}

/// Result ids of float arithmetic lacking `NoContraction`, and whether the
/// environment modes of `widths` are present: what a sealed module must not
/// lack (for tests).
pub fn unsealed(module: &[u32], widths: &[u32]) -> Result<(usize, bool), MalformedModule> {
    let instructions = instructions(module)?;
    let words = |instruction: &Instruction| {
        &module[instruction.start..instruction.start + instruction.words]
    };
    let decorated = instructions
        .iter()
        .filter(|instruction| {
            instruction.opcode == OP_DECORATE && words(instruction)[2] == DECORATION_NO_CONTRACTION
        })
        .map(|instruction| words(instruction)[1])
        .collect::<HashSet<_>>();
    let missing = instructions
        .iter()
        .filter(|instruction| FLOAT_ARITHMETIC.contains(&instruction.opcode))
        .filter(|instruction| !decorated.contains(&words(instruction)[2]))
        .count();
    let modes = instructions
        .iter()
        .filter(|instruction| instruction.opcode == OP_EXECUTION_MODE)
        .map(|instruction| (words(instruction)[2], words(instruction)[3]))
        .collect::<Vec<_>>();
    let environment = [MODE_ROUNDING_MODE_RTE, MODE_SIGNED_ZERO_INF_NAN_PRESERVE]
        .iter()
        .all(|mode| widths.iter().all(|width| modes.contains(&(*mode, *width))));
    Ok((missing, environment))
}
