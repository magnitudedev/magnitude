//! Seismic's SPIR-V seal pass (§7.2 step 3, §7.3): every floating-point
//! arithmetic result is decorated `NoContraction`, and the entry point runs
//! with RTE rounding and signed-zero/Inf/NaN preservation for fp32, and for
//! fp16 on devices with fp16 arithmetic, plus `DenormPreserve 32` where the
//! device supports it. `RoundingModeRTE 32` is left out where the driver
//! cannot compile it (NVIDIA proprietary, §16.1); there the device's fp32
//! rounding is probed at open instead.
//!
//! glslang has no global switch for this, and `precise` is per object.

/// Version of these passes; part of the formation identity and SPIR-V cache
/// key.
pub const SEAL_VERSION: u32 = 4;

/// The execution modes a sealed module declares beyond the fixed ones.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Environment {
    /// The device has fp16 arithmetic: `RoundingModeRTE 16` and
    /// `SignedZeroInfNanPreserve 16` are declared.
    pub float16: bool,
    /// `RoundingModeRTE 32` is declared. When false, fp32 add, multiply and
    /// conversion rounding is the device's default, which opening verifies
    /// is round-to-nearest-even.
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
        .collect::<std::collections::HashSet<_>>();

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
        .collect::<std::collections::HashSet<_>>();
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
