//! Formation (§7.2): GLSL → SPIR-V with glslang, dead-code cleanup, the
//! kernel contract, the seal pass and SPIRV-Tools validation
//! ([`compile`]); then, for a module, the `OpFmaKHR` binding of `fma` on
//! devices with `VK_KHR_shader_fma` (§16.1) and one compute pipeline per
//! entry with its workgroup size and group-memory views as specialization
//! constants, full subgroups where the module reads its subgroups
//! ([`uniformity::reads_subgroups`]), a required subgroup size of 32 where
//! the device takes one, and a pipeline cache of its own, whose data is the
//! program's artifact ([`DirectModule::compile`]).

use crate::device::Device;
use crate::device_info::SUBGROUP_WIDTH;
use crate::numbering;
use crate::seal::{self, SEAL_VERSION};
use crate::uniformity;
use std::collections::HashMap;
pub use crate::uniformity::{ADDRESS_TABLE_BLOCK, ARGUMENT_BLOCK, INPUT_BLOCK};
use ash::vk;
use seismic_native_target::{NativeCompilationError, ProgramEntry, ProgramStatistic, StatisticValue};
use spirv_tools::opt::Optimizer;
use spirv_tools::val::Validator;

/// The GLSL compiler this crate links (pinned in its manifest).
pub const GLSLANG_RELEASE: &str = "glslang 0.8.1+275822a";

/// Everything besides the source that determines a formed module: the
/// compiler, target, seal passes and the device info they and the driver
/// consume (§7.5), and what every pipeline on the device is created with
/// ([`created_with`]).
pub(crate) fn formation(device: &Device) -> String {
    format!(
        "{};{};create {}",
        compiler(),
        device.info().formation_identity(device.properties()),
        created_with(device).as_raw()
    )
}

/// What every pipeline on `device` is created with besides what its
/// operation adds: statistics capture, where the device reports statistics.
/// A driver may key a pipeline's cache data by it, so a load creates its
/// pipelines with it as their compile did, and it is part of the formation
/// identity, so no load meets data created without it.
fn created_with(device: &Device) -> vk::PipelineCreateFlags {
    if device.inner.executables.is_some() {
        vk::PipelineCreateFlags::CAPTURE_STATISTICS_KHR
    } else {
        vk::PipelineCreateFlags::empty()
    }
}

/// The part of every formation identity the compiler determines: glslang,
/// the SPIR-V target and the seal passes. Offline formation, which forms no
/// pipeline on a device, is identified by this and the formation target.
pub fn compiler() -> String {
    format!("vulkan;{GLSLANG_RELEASE};spirv 1.6;seal {SEAL_VERSION}")
}

fn toolchain(error: impl std::fmt::Display) -> NativeCompilationError {
    NativeCompilationError::ToolchainFailure(error.to_string())
}

fn malformed(error: seal::MalformedModule) -> NativeCompilationError {
    NativeCompilationError::MalformedToolchainOutput(error.0)
}

/// Form the launch `entry` of `source` for `environment`: its module
/// ([`compile`]), refused where it reads its subgroups and its workgroup
/// does not hold whole ones on every device ([`launch_size`]), counting a
/// read only the module of another device makes.
pub fn form(
    source: &str,
    entry: &ProgramEntry,
    environment: seal::Environment,
) -> Result<Vec<u32>, NativeCompilationError> {
    let (module, reads_subgroups) = compiled(source, &entry.symbol, environment)?;
    launch_size(entry, reads_subgroups)?;
    Ok(module)
}

/// Compile the launch `kernel` of `source` to SPIR-V, cleaned, sealed for
/// `environment` and validated, its 32-lane hardware joins removed
/// (`shipped`). A module holding a construct the emitted contract forbids
/// ([`seal::contract`], and on its `analysed` copy
/// [`seal::conditional_subtractions`]), or breaking the execution model
/// (`uniformity`, checked through [`execution_model`]), is refused, as glslang
/// refuses invalid source. The seal follows the check, which says whether
/// the module relies on the execution model's assumption, which the seal
/// then declares where the device can, and whether it relies on maximal
/// reconvergence, which the check accepts only where `environment` declares
/// it, so the seal declares it exactly where relied on.
pub fn compile(
    source: &str,
    kernel: &str,
    environment: seal::Environment,
) -> Result<Vec<u32>, NativeCompilationError> {
    compiled(source, kernel, environment).map(|(module, _)| module)
}

/// The variable naming a directory where formation writes the analysed copy
/// (`analysed`, or `inlined` where the execution model was checked on that:
/// [`execution_model`]) of each module it refuses for breaking the contract
/// or the execution model, as SPIR-V assembly in `<kernel>-<digest>.spvasm`, one
/// file per module: the module whose ids the refusal names.
pub const ANALYSED_DIRECTORY: &str = "SEISMIC_VULKAN_ANALYSED";

/// `module` as SPIR-V assembly, its ids as numbers.
fn assembly(module: &[u32]) -> Result<String, NativeCompilationError> {
    use spirv_tools::assembler::{Assembler, DisassembleOptions};
    Ok(spirv_tools::assembler::create(Some(target()))
        .disassemble(
            module,
            DisassembleOptions {
                print: false,
                color: false,
                indent: false,
                show_byte_offset: false,
                no_header: true,
                use_friendly_names: false,
                comment: false,
            },
        )
        .map_err(toolchain)?
        .expect("a disassembly not printed is returned"))
}

/// [`compile`]'s module, and whether it reads its subgroups on some device:
/// as formed for `environment`, or through a 32-lane hardware join, which
/// is a subgroup operation on 64 lanes.
fn compiled(
    source: &str,
    kernel: &str,
    environment: seal::Environment,
) -> Result<(Vec<u32>, bool), NativeCompilationError> {
    let cleaned = cleaned(source, kernel)?;
    let (joins, mut violations) = uniformity::hardware_joins(&cleaned).map_err(malformed)?;
    let Analysed { checked, numbered } = analysed(&cleaned, &joins)?;
    violations.extend(seal::contract(&cleaned).map_err(malformed)?);
    violations.extend(seal::conditional_subtractions(&numbered).map_err(malformed)?);
    let (findings, analysed) = execution_model(&cleaned, &joins, checked, environment.maximal_reconvergence)?;
    violations.extend(findings.violations);
    if !violations.is_empty() {
        if let Some(directory) = std::env::var_os(ANALYSED_DIRECTORY) {
            // One file per module: the specializations of a kernel differ.
            let digest = {
                use std::hash::{Hash, Hasher};
                let mut hasher = std::collections::hash_map::DefaultHasher::new();
                analysed.hash(&mut hasher);
                hasher.finish()
            };
            let path = std::path::Path::new(&directory).join(format!("{kernel}-{digest:016x}.spvasm"));
            std::fs::write(&path, assembly(&analysed)?)
                .unwrap_or_else(|error| panic!("{}: {error}", path.display()));
        }
        // Written as glslang writes a diagnostic, so readers that keep a
        // toolchain failure's error lines keep it.
        return Err(toolchain(format!(
            "ERROR: `{kernel}` violates the Vulkan kernel contract: {}",
            violations
                .iter()
                .map(|violation| violation.0.as_str())
                .collect::<Vec<_>>()
                .join("; ")
        )));
    }
    let sealed = seal::seal(
        &cleaned,
        environment,
        findings.relies_on_reconvergence,
        findings.relies_on_maximal_reconvergence,
    )
    .map_err(malformed)?;
    validate(&sealed).map_err(NativeCompilationError::MalformedToolchainOutput)?;
    let reads_subgroups = !joins.is_empty() || uniformity::reads_subgroups(&sealed).map_err(malformed)?;
    let merges_returns =
        findings.relies_on_maximal_reconvergence && environment.merge_maximal_reconvergence_returns;
    Ok((shipped(sealed, &joins, merges_returns)?, reads_subgroups))
}

/// The launch `kernel` of `source` compiled to SPIR-V by glslang and
/// cleaned ([`clean`]).
fn cleaned(source: &str, kernel: &str) -> Result<Vec<u32>, NativeCompilationError> {
    let compiler = glslang::Compiler::acquire().ok_or_else(|| {
        NativeCompilationError::ToolchainFailure("glslang could not be initialized".into())
    })?;
    let text = glslang::ShaderSource::from(source);
    let defines = [("SEISMIC_KERNEL", Some(kernel))];
    let input = glslang::ShaderInput::new(
        &text,
        glslang::ShaderStage::Compute,
        &options(),
        Some(&defines[..]),
        None,
    )
    .map_err(toolchain)?;
    let shader = glslang::Shader::new(compiler, input).map_err(toolchain)?;
    let compiled = shader.compile().map_err(toolchain)?;
    clean(&compiled)
}

/// The glslang options of every formation.
fn options() -> glslang::CompilerOptions {
    glslang::CompilerOptions {
        source_language: glslang::SourceLanguage::GLSL,
        target: glslang::Target::Vulkan {
            version: glslang::VulkanVersion::Vulkan1_3,
            spirv_version: glslang::SpirvVersion::SPIRV1_6,
        },
        version_profile: None,
        messages: glslang::ShaderMessage::DEFAULT,
    }
}

/// glslang's preprocessed text of `source` for the launch `kernel`, as
/// [`compile`] preprocesses it. glslang parses what it preprocesses, so
/// `source` must parse as a compute shader; a preprocessing failure (an
/// active `#error` among them) is a toolchain failure.
pub fn preprocess(source: &str, kernel: &str) -> Result<String, NativeCompilationError> {
    let compiler = glslang::Compiler::acquire().ok_or_else(|| {
        NativeCompilationError::ToolchainFailure("glslang could not be initialized".into())
    })?;
    let text = glslang::ShaderSource::from(source);
    let defines = [("SEISMIC_KERNEL", Some(kernel))];
    let input = glslang::ShaderInput::new(
        &text,
        glslang::ShaderStage::Compute,
        &options(),
        Some(&defines[..]),
        None,
    )
    .map_err(toolchain)?;
    let shader = glslang::Shader::new(compiler, input).map_err(toolchain)?;
    Ok(shader.get_preprocessed_code())
}

fn target() -> spirv_tools::TargetEnv {
    spirv_tools::TargetEnv::Vulkan_1_3
}

fn validator_options() -> spirv_tools::val::ValidatorOptions {
    spirv_tools::val::ValidatorOptions {
        scalar_block_layout: true,
        ..Default::default()
    }
}

/// SPIRV-Tools validation for the Vulkan 1.3 environment.
pub fn validate(module: &[u32]) -> Result<(), String> {
    spirv_tools::val::create(Some(target()))
        .validate(module, Some(validator_options()))
        .map_err(|error| error.to_string())
}

/// Dead-code cleanup only: the driver compiler does all optimization, and
/// the performance passes would rewrite float math.
fn clean(module: &[u32]) -> Result<Vec<u32>, NativeCompilationError> {
    use spirv_tools::opt::Passes;
    optimized(
        module,
        &[
            Passes::AggressiveDCE,
            Passes::DeadVariableElimination,
            Passes::RemoveUnusedInterfaceVariables,
            Passes::EliminateDeadFunctions,
        ],
    )
}

/// The analysed copies of a cleaned module ([`analysed`]).
struct Analysed {
    /// The copy the execution model is checked on (`uniformity::check`): a
    /// call of a summarised function is a value of its own.
    checked: Vec<u32>,
    /// The copy the contract's value rules are checked on
    /// ([`seal::conditional_subtractions`]): the same, but a call of a
    /// summarised function is numbered as an operation of its arguments, so
    /// two calls with one function and arguments, the first dominating the
    /// second, are one value, as the inlined bodies of a straight-line
    /// function are.
    numbered: Vec<u32>,
}

/// The copies of a cleaned module the execution model and the contract's
/// value rules are checked on, never shipped. Each summarised function
/// ([`uniformity::summarised`]), pure, is kept as a function the analysis
/// summarises at its calls, its arguments passed as values ([`by_value`]);
/// every other call but a 32-lane hardware join's (`joins`, which the check
/// reads) is inlined. Then, in every function, returns are merged into
/// breaks from the function's body (after dead branches go, which the merge
/// requires), private variables made local, locals promoted to SSA, and
/// values numbered ([`numbering::numbered`]): a value computed again where
/// it is already held (an operation of the same operands, or a load of
/// read-only memory, dominated by its first) replaced by that, so the
/// checks run over SSA values and one value is one id. A pure function is
/// analysed once per call's argument state instead of once per inlined
/// copy: a helper unrolled callers call many times is one function, not
/// one body per call. None of these passes is held back by a module's
/// extensions, so the copies are the same in every environment.
fn analysed(module: &[u32], joins: &[uniformity::HardwareJoin]) -> Result<Analysed, NativeCompilationError> {
    use spirv_tools::opt::Passes;
    let module = joins_kept(module, joins);
    // Each function specialised to the constants its calls pass, and the
    // branches they decide pruned, before what it does is judged.
    let module = optimized(&specialised(&module, joins).map_err(malformed)?, &[Passes::DeadBranchElim])?;
    let summarised = uniformity::summarised(&module).map_err(malformed)?;
    let inlined = promoted(&by_value(&module, &summarised).map_err(malformed)?)?;
    Ok(Analysed {
        checked: numbering::numbered(&inlined, false).map_err(malformed)?,
        numbered: numbering::numbered(&inlined, true).map_err(malformed)?,
    })
}

/// The copy of a cleaned module the execution model is checked on without
/// summaries: every call but a 32-lane hardware join's (`joins`) inlined,
/// then as [`analysed`] makes its copies. Its check is the exact one
/// [`execution_model`] falls back to.
fn inlined(module: &[u32], joins: &[uniformity::HardwareJoin]) -> Result<Vec<u32>, NativeCompilationError> {
    numbering::numbered(&promoted(&joins_kept(module, joins))?, false).map_err(malformed)
}

/// `module` with its 32-lane hardware joins (`joins`) kept from inlining,
/// so the check reads each call of one.
fn joins_kept(module: &[u32], joins: &[uniformity::HardwareJoin]) -> Vec<u32> {
    let mut module = module.to_vec();
    for join in joins {
        module[join.at + 3] |= uniformity::DONT_INLINE;
    }
    module
}

/// `module` with every call inlined but those of functions marked
/// `DontInline`, after dead branches go; returns merged into breaks, private
/// variables made local and locals promoted to SSA: [`analysed`]'s and
/// [`inlined`]'s copies before numbering.
fn promoted(module: &[u32]) -> Result<Vec<u32>, NativeCompilationError> {
    use spirv_tools::opt::Passes;
    optimized(
        module,
        &[
            Passes::DeadBranchElim,
            Passes::MergeReturn,
            Passes::InlineExhaustive,
            Passes::EliminateDeadFunctions,
            Passes::PrivateToLocal,
            Passes::SSARewrite,
        ],
    )
}

/// The execution model's findings for `cleaned`, a cleaned module whose
/// 32-lane hardware joins are `joins`, in an environment that declares
/// maximal reconvergence or not, and the copy they name: the one function
/// formation checks the execution model through. `summarised`, the copy
/// with summaries ([`analysed`]), is checked first; summaries only
/// over-approximate (`uniformity`, "Summarised functions"), so where that
/// accepts, the module is accepted. Where it refuses, the copy with every
/// call inlined ([`inlined`]) is checked, the exact check, and its verdict
/// stands: a summary never refuses a module. A module the summarised copy
/// cannot be read in but the inlined one can is a fault of the summaries,
/// and panics rather than hide behind the exact check.
fn execution_model(
    cleaned: &[u32],
    joins: &[uniformity::HardwareJoin],
    summarised: Vec<u32>,
    maximal_reconvergence: bool,
) -> Result<(uniformity::Findings, Vec<u32>), NativeCompilationError> {
    let fault = match uniformity::check(&summarised, maximal_reconvergence) {
        Ok(findings) if findings.violations.is_empty() => return Ok((findings, summarised)),
        Ok(_) => None,
        Err(error) => Some(error),
    };
    let inlined = inlined(cleaned, joins)?;
    let findings = uniformity::check(&inlined, maximal_reconvergence).map_err(malformed)?;
    if let Some(error) = fault {
        panic!("invariant violated: the summarised copy cannot be checked ({error:?}) where the inlined one can");
    }
    Ok((findings, inlined))
}

/// `module` with each call passing constant arguments (scalar integers and
/// booleans, not specialization constants: as values, or through a variable
/// of the caller's that holds one, as glslang passes them) calling a copy of
/// its function specialised to them: the copy's body reads the constants
/// where it read those parameters, and each of its conditional branches whose condition
/// they decide (through compares, logical operations, selections, copies,
/// and phis of one value) branches on that constant, which
/// `DeadBranchElim` then prunes, as it prunes a switch on a constant. One
/// copy is made per function and tuple of constant arguments, and its own
/// calls are specialised in turn, so a constant passed down a chain of
/// calls reaches every body on it, as inlining carries it. A function keeps
/// its name in its copies. The entry point, the hardware joins (`joins`), a
/// function named as one, and a function holding an instruction the
/// analysis does not read, are never copied. This serves the analysed copy
/// alone: a summarised function is judged on what the constants its calls
/// pass leave of it (`uniformity`, "Summarised functions").
fn specialised(module: &[u32], joins: &[uniformity::HardwareJoin]) -> Result<Vec<u32>, seal::MalformedModule> {
    const OP_NAME: u32 = 5;
    const OP_LINE: u32 = 8;
    const OP_ENTRY_POINT: u32 = 15;
    const OP_TYPE_BOOL: u32 = 20;
    const OP_TYPE_INT: u32 = 21;
    const OP_TYPE_POINTER: u32 = 32;
    const OP_CONSTANT_TRUE: u32 = 41;
    const OP_CONSTANT_FALSE: u32 = 42;
    const OP_CONSTANT: u32 = 43;
    const OP_FUNCTION: u32 = 54;
    const OP_FUNCTION_PARAMETER: u32 = 55;
    const OP_FUNCTION_END: u32 = 56;
    const OP_FUNCTION_CALL: u32 = 57;
    const OP_VARIABLE: u32 = 59;
    const OP_LOAD: u32 = 61;
    const OP_STORE: u32 = 62;
    const OP_COPY_OBJECT: u32 = 83;
    const OP_LOGICAL_EQUAL: u32 = 164;
    const OP_LOGICAL_NOT_EQUAL: u32 = 165;
    const OP_LOGICAL_OR: u32 = 166;
    const OP_LOGICAL_AND: u32 = 167;
    const OP_LOGICAL_NOT: u32 = 168;
    const OP_SELECT: u32 = 169;
    const OP_PHI: u32 = 245;
    const OP_LABEL: u32 = 248;
    const OP_BRANCH_CONDITIONAL: u32 = 250;
    const OP_SWITCH: u32 = 251;
    const OP_NO_LINE: u32 = 317;
    const STORAGE_FUNCTION: u32 = 7;

    /// What copying reads of the module, and what it adds to it.
    struct Specialising {
        bound: u32,
        /// Scalar integer types → width.
        ints: HashMap<u32, u32>,
        boolean: Option<u32>,
        /// Scalar integer and boolean constants → their type and bits.
        constants: HashMap<u32, (u32, u64)>,
        /// Pointer types.
        pointers: std::collections::HashSet<u32>,
        /// The boolean constants false and true.
        truth: [Option<u32>; 2],
        /// Constants copies need, declared before the first function.
        declared: Vec<u32>,
    }

    impl Specialising {
        fn fresh(&mut self) -> u32 {
            self.bound += 1;
            self.bound - 1
        }

        /// The boolean constant `value`.
        fn truth(&mut self, boolean: u32, value: bool) -> u32 {
            if let Some(id) = self.truth[usize::from(value)] {
                return id;
            }
            let id = self.fresh();
            let opcode = if value { OP_CONSTANT_TRUE } else { OP_CONSTANT_FALSE };
            self.declared.extend([(3 << 16) | opcode, boolean, id]);
            self.truth[usize::from(value)] = Some(id);
            id
        }

        /// A copy of `template`, a function, whose parameters with an
        /// argument (`substitutable`) read it: a value
        /// parameter is the constant, and each load of a pointer parameter
        /// is. None where an instruction's ids cannot all be told (a switch
        /// on a selector of unknown width).
        fn copy(&mut self, template: &[Vec<u32>], arguments: &[Option<u32>]) -> Option<Vec<Vec<u32>>> {
            // Each id the function defines → the copy's; a value parameter
            // with an argument, and a load of a pointer parameter with one,
            // → it. By id: its type.
            let mut renamed = HashMap::new();
            let mut types = HashMap::new();
            // The copy's parameters, in order; the pointer parameters with
            // an argument → it.
            let mut parameters = Vec::new();
            let mut pointed = HashMap::new();
            for words in template {
                let opcode = words[0] & 0xffff;
                let defined = match opcode {
                    OP_LABEL => Some(words[1]),
                    OP_FUNCTION => Some(words[2]),
                    OP_FUNCTION_PARAMETER => {
                        types.insert(words[2], words[1]);
                        let declared = self.fresh();
                        parameters.push(declared);
                        let argument = arguments.get(parameters.len() - 1).copied().flatten();
                        match argument {
                            Some(argument) if self.pointers.contains(&words[1]) => {
                                pointed.insert(words[2], argument);
                                renamed.insert(words[2], declared);
                            }
                            argument => {
                                renamed.insert(words[2], argument.unwrap_or(declared));
                            }
                        }
                        None
                    }
                    OP_FUNCTION_END | OP_LINE | OP_NO_LINE => None,
                    OP_LOAD if pointed.contains_key(&words[3]) => {
                        types.insert(words[2], words[1]);
                        renamed.insert(words[2], pointed[&words[3]]);
                        None
                    }
                    opcode => uniformity::shape(opcode)?.result.then(|| {
                        types.insert(words[2], words[1]);
                        words[2]
                    }),
                };
                if let Some(id) = defined {
                    let fresh = self.fresh();
                    renamed.insert(id, fresh);
                }
            }
            let rename = |id: u32| renamed.get(&id).copied().unwrap_or(id);
            // By id of the copy: its bits, where the constants decide them.
            let mut known = HashMap::<u32, u64>::new();
            let mut copy = Vec::with_capacity(template.len());
            let mut parameter = 0;
            for words in template {
                let opcode = words[0] & 0xffff;
                // A load of a pointer parameter with an argument is the
                // argument.
                if opcode == OP_LOAD && pointed.contains_key(&words[3]) {
                    continue;
                }
                let mut copied = words.clone();
                match opcode {
                    OP_LABEL => copied[1] = rename(words[1]),
                    OP_FUNCTION => copied[2] = rename(words[2]),
                    OP_FUNCTION_PARAMETER => {
                        copied[2] = parameters[parameter];
                        parameter += 1;
                    }
                    OP_FUNCTION_END | OP_LINE | OP_NO_LINE => {}
                    opcode => {
                        let shape = uniformity::shape(opcode).expect("a copied function's shapes are known");
                        if shape.result {
                            copied[2] = rename(words[2]);
                        }
                        for at in shape.positions(words) {
                            copied[at] = rename(words[at]);
                        }
                        match opcode {
                            // The parents.
                            OP_PHI => {
                                for at in (4..words.len()).step_by(2) {
                                    copied[at] = rename(words[at]);
                                }
                            }
                            // Each case: a literal of the selector's width,
                            // then its target.
                            OP_SWITCH => {
                                let width = types
                                    .get(&words[1])
                                    .or_else(|| self.constants.get(&words[1]).map(|(ty, _)| ty))
                                    .and_then(|ty| self.ints.get(ty))?;
                                let case = width.div_ceil(32) as usize + 1;
                                for at in (3..words.len()).skip(case - 1).step_by(case) {
                                    copied[at] = rename(words[at]);
                                }
                            }
                            _ => {}
                        }
                    }
                }
                // What the constants decide.
                let bits = |id: u32| known.get(&id).copied().or_else(|| self.constants.get(&id).map(|(_, bits)| *bits));
                let decided = match opcode {
                    // Compares of integers of the first operand's width.
                    170..=179 => {
                        let width = types
                            .get(&words[3])
                            .or_else(|| self.constants.get(&copied[3]).map(|(ty, _)| ty))
                            .and_then(|ty| self.ints.get(ty))
                            .copied();
                        match (width, bits(copied[3]), bits(copied[4])) {
                            (Some(width), Some(first), Some(second)) => Some(u64::from(compare(opcode, width, first, second))),
                            _ => None,
                        }
                    }
                    OP_LOGICAL_EQUAL | OP_LOGICAL_NOT_EQUAL | OP_LOGICAL_OR | OP_LOGICAL_AND => {
                        match (bits(copied[3]), bits(copied[4])) {
                            (Some(first), Some(second)) => Some(u64::from(match opcode {
                                OP_LOGICAL_EQUAL => first == second,
                                OP_LOGICAL_NOT_EQUAL => first != second,
                                OP_LOGICAL_OR => first != 0 || second != 0,
                                _ => first != 0 && second != 0,
                            })),
                            // False of `&&` and true of `||` decide alone.
                            (Some(0), None) | (None, Some(0)) if opcode == OP_LOGICAL_AND => Some(0),
                            (Some(1), None) | (None, Some(1)) if opcode == OP_LOGICAL_OR => Some(1),
                            _ => None,
                        }
                    }
                    OP_LOGICAL_NOT => bits(copied[3]).map(|value| u64::from(value == 0)),
                    OP_SELECT => bits(copied[3]).and_then(|condition| bits(copied[if condition != 0 { 4 } else { 5 }])),
                    OP_COPY_OBJECT => bits(copied[3]),
                    // A phi whose incoming values are one value, each
                    // decided already.
                    OP_PHI => {
                        let values = copied[3..].iter().step_by(2).map(|id| bits(*id)).collect::<Vec<_>>();
                        values.first().copied().flatten().filter(|first| values.iter().all(|value| *value == Some(*first)))
                    }
                    _ => None,
                };
                let condition = if opcode == OP_BRANCH_CONDITIONAL { bits(copied[1]) } else { None };
                if let Some(value) = decided {
                    known.insert(copied[2], value);
                }
                if let (Some(boolean), Some(condition)) = (self.boolean, condition) {
                    copied[1] = self.truth(boolean, condition != 0);
                }
                copy.push(copied);
            }
            Some(copy)
        }
    }

    /// By parameter of `body`, a function: a constant argument can take its
    /// place. A value parameter's can; a pointer parameter's, where the
    /// function only loads it whole (glslang passes each argument so,
    /// through a variable of the caller's).
    fn substitutable(body: &[Vec<u32>], pointers: &std::collections::HashSet<u32>) -> Vec<bool> {
        let parameters = body
            .iter()
            .filter(|words| words[0] & 0xffff == OP_FUNCTION_PARAMETER)
            .map(|words| (words[2], pointers.contains(&words[1])))
            .collect::<Vec<_>>();
        let mut substitutable = vec![true; parameters.len()];
        let pointer_parameters = parameters
            .iter()
            .enumerate()
            .filter(|(_, (_, pointer))| *pointer)
            .map(|(index, (id, _))| (*id, index))
            .collect::<HashMap<_, _>>();
        for words in body {
            let opcode = words[0] & 0xffff;
            if matches!(opcode, OP_FUNCTION | OP_FUNCTION_PARAMETER | OP_FUNCTION_END | OP_LABEL | OP_LINE | OP_NO_LINE) {
                continue;
            }
            let Some(shape) = uniformity::shape(opcode) else { return vec![false; parameters.len()] };
            for at in shape.positions(words) {
                if let Some(index) = pointer_parameters.get(&words[at]) {
                    if !(opcode == OP_LOAD && at == 3) {
                        substitutable[*index] = false;
                    }
                }
            }
        }
        substitutable
    }

    /// The function variables of `body` that hold one constant wherever a
    /// call reads them, each with it: initialised or stored with it alone,
    /// and otherwise only loaded or passed to one call, which cannot have
    /// written it before (glslang's variable for an argument). Where the call
    /// comes before its store, it reads an undefined value, which any value,
    /// the constant's too, may be.
    fn holding(body: &[Vec<u32>], constants: &HashMap<u32, (u32, u64)>) -> HashMap<u32, u32> {
        // By variable: its one constant, if it has one yet; and the calls
        // passing it.
        let mut held = HashMap::<u32, (Option<u32>, usize)>::new();
        let mut other = std::collections::HashSet::new();
        for words in body {
            if words[0] & 0xffff == OP_VARIABLE && words[3] == STORAGE_FUNCTION {
                match words.get(4) {
                    Some(initializer) if !constants.contains_key(initializer) => {
                        other.insert(words[2]);
                    }
                    initializer => {
                        held.insert(words[2], (initializer.copied(), 0));
                    }
                }
            }
        }
        for words in body {
            let opcode = words[0] & 0xffff;
            if matches!(opcode, OP_FUNCTION | OP_FUNCTION_PARAMETER | OP_FUNCTION_END | OP_LABEL | OP_LINE | OP_NO_LINE | OP_VARIABLE) {
                continue;
            }
            let Some(shape) = uniformity::shape(opcode) else { return HashMap::new() };
            for at in shape.positions(words) {
                let Some((constant, calls)) = held.get_mut(&words[at]) else { continue };
                let kept = match (opcode, at) {
                    (OP_LOAD, 3) => true,
                    (OP_STORE, 1) => {
                        let stored = words[2];
                        let one = constants.contains_key(&stored) && constant.is_none_or(|held| held == stored);
                        *constant = Some(stored);
                        one
                    }
                    (OP_FUNCTION_CALL, _) => {
                        *calls += 1;
                        *calls == 1
                    }
                    _ => false,
                };
                if !kept {
                    other.insert(words[at]);
                }
            }
        }
        held.into_iter()
            .filter(|(variable, _)| !other.contains(variable))
            .filter_map(|(variable, (constant, _))| constant.map(|constant| (variable, constant)))
            .collect()
    }

    /// The compare `opcode` of `first` and `second`, integers of `width`.
    fn compare(opcode: u32, width: u32, first: u64, second: u64) -> bool {
        let shift = 64 - width.clamp(1, 64);
        let unsigned = |value: u64| (value << shift) >> shift;
        let signed = |value: u64| ((value << shift) as i64) >> shift;
        let (first_unsigned, second_unsigned) = (unsigned(first), unsigned(second));
        let (first_signed, second_signed) = (signed(first), signed(second));
        match opcode {
            170 => first_unsigned == second_unsigned,
            171 => first_unsigned != second_unsigned,
            172 => first_unsigned > second_unsigned,
            173 => first_signed > second_signed,
            174 => first_unsigned >= second_unsigned,
            175 => first_signed >= second_signed,
            176 => first_unsigned < second_unsigned,
            177 => first_signed < second_signed,
            178 => first_unsigned <= second_unsigned,
            _ => first_signed <= second_signed,
        }
    }

    let instructions = seal::instructions(module)?;
    let words_of = |instruction: &seal::Instruction| &module[instruction.start..instruction.start + instruction.words];
    let mut specialising = Specialising {
        bound: module[3],
        ints: HashMap::new(),
        boolean: None,
        constants: HashMap::new(),
        pointers: std::collections::HashSet::new(),
        truth: [None, None],
        declared: Vec::new(),
    };
    // Functions never copied: the entry point, the joins, and those named
    // as joins.
    let mut kept = joins.iter().map(|join| join.ids[0]).collect::<std::collections::HashSet<_>>();
    // Each function's instructions, and where the first starts.
    let mut functions: Vec<Vec<Vec<u32>>> = Vec::new();
    let mut first = None;
    for (index, instruction) in instructions.iter().enumerate() {
        let words = words_of(instruction);
        match instruction.opcode {
            OP_FUNCTION => {
                first = first.or(Some(index));
                functions.push(vec![words.to_vec()]);
            }
            _ if first.is_some() => functions.last_mut().expect("a function's instructions follow it").push(words.to_vec()),
            OP_NAME => {
                if seal::literal_string(&words[2..]).starts_with(uniformity::HARDWARE_JOIN) {
                    kept.insert(words[1]);
                }
            }
            OP_ENTRY_POINT => {
                kept.insert(words[2]);
            }
            OP_TYPE_POINTER => {
                specialising.pointers.insert(words[1]);
            }
            OP_TYPE_BOOL => specialising.boolean = Some(words[1]),
            OP_TYPE_INT => {
                specialising.ints.insert(words[1], words[2]);
            }
            OP_CONSTANT_TRUE | OP_CONSTANT_FALSE if Some(words[1]) == specialising.boolean => {
                let value = instruction.opcode == OP_CONSTANT_TRUE;
                specialising.truth[usize::from(value)] = specialising.truth[usize::from(value)].or(Some(words[2]));
                specialising.constants.insert(words[2], (words[1], u64::from(value)));
            }
            OP_CONSTANT if specialising.ints.get(&words[1]).is_some_and(|width| *width <= 64) => {
                let bits = words[3..].iter().rev().fold(0u64, |value, word| value << 32 | u64::from(*word));
                specialising.constants.insert(words[2], (words[1], bits));
            }
            _ => {}
        }
    }
    let Some(first) = first else { return Ok(module.to_vec()) };
    let templates = functions.clone();
    let template_of = templates.iter().map(|body| (body[0][2], body.as_slice())).collect::<HashMap<_, _>>();
    let substitutable = templates
        .iter()
        .map(|body| (body[0][2], substitutable(body, &specialising.pointers)))
        .collect::<HashMap<_, _>>();
    // By function and arguments (each a constant or not): its copy, if it
    // can be made.
    let mut made = HashMap::<(u32, Vec<Option<u32>>), Option<u32>>::new();
    // By function: its copies.
    let mut copies = HashMap::<u32, Vec<u32>>::new();
    let mut at = 0;
    while at < functions.len() {
        let held = holding(&functions[at], &specialising.constants);
        for position in 0..functions[at].len() {
            let words = &functions[at][position];
            if words[0] & 0xffff != OP_FUNCTION_CALL || kept.contains(&words[3]) || !template_of.contains_key(&words[3]) {
                continue;
            }
            let callee = words[3];
            // Each argument's constant: its own, or that of the variable it
            // points to.
            let arguments = words[4..]
                .iter()
                .zip(&substitutable[&callee])
                .map(|(argument, can)| {
                    let constant = if specialising.constants.contains_key(argument) {
                        Some(*argument)
                    } else {
                        held.get(argument).copied()
                    };
                    constant.filter(|_| *can)
                })
                .collect::<Vec<_>>();
            if arguments.iter().all(Option::is_none) {
                continue;
            }
            let key = (callee, arguments);
            let existing = made.get(&key).copied();
            let copy = match existing {
                Some(copy) => copy,
                None => {
                    let copy = specialising.copy(template_of[&callee], &key.1).map(|body| {
                        let id = body[0][2];
                        functions.push(body);
                        copies.entry(callee).or_default().push(id);
                        id
                    });
                    made.insert(key, copy);
                    copy
                }
            };
            if let Some(copy) = copy {
                functions[at][position][3] = copy;
            }
        }
        at += 1;
    }
    if copies.is_empty() {
        return Ok(module.to_vec());
    }
    let mut out = module[..5].to_vec();
    for instruction in &instructions[..first] {
        let words = words_of(instruction);
        out.extend_from_slice(words);
        if instruction.opcode == OP_NAME {
            for copy in copies.get(&words[1]).into_iter().flatten() {
                out.extend([words[0], *copy]);
                out.extend_from_slice(&words[2..]);
            }
        }
    }
    out.extend(specialising.declared);
    out.extend(functions.into_iter().flatten().flatten());
    out[3] = specialising.bound;
    Ok(out)
}

/// What [`by_value`] makes of a summarised function with a pointer
/// parameter.
struct ByValue {
    /// Its new function type.
    ty: u32,
    /// By parameter: the value a pointer parameter becomes.
    values: Vec<Option<u32>>,
    /// Its original parameter types.
    pointers: Vec<u32>,
    /// Where it writes through a pointer parameter: the struct it returns,
    /// whether its own result is void (else the struct's first member), and
    /// the parameters written, each with its pointee, in the struct's order.
    returned: Option<(u32, bool, Vec<(usize, u32)>)>,
    /// Its original result type.
    result: u32,
}

/// `module` with each summarised function (`summarised`: by parameter,
/// whether it is a pointer, and if so whether the function writes through
/// it) kept from inlining (`DontInline`), its pointer parameters passed by
/// value. Each becomes a value the function stores, at its start, to a
/// variable of its own that takes the parameter's place; where the function
/// writes any, it returns a struct of its result and the values those
/// variables end with. Each call loads the pointers it passes and passes
/// the values, and stores the values returned back through them. The call
/// writes no other memory, and no two of its pointers point into one
/// variable (`summarised`), so this is what it did to them, as GLSL defines
/// `in`, `out` and `inout`. SSA then promotes the variables, and the
/// function's operations read its arguments as values, as the inlined body
/// would.
fn by_value(
    module: &[u32],
    summarised: &HashMap<u32, Vec<Option<bool>>>,
) -> Result<Vec<u32>, seal::MalformedModule> {
    const OP_TYPE_VOID: u32 = 19;
    const OP_TYPE_STRUCT: u32 = 30;
    const OP_TYPE_POINTER: u32 = 32;
    const OP_TYPE_FUNCTION: u32 = 33;
    const OP_FUNCTION: u32 = 54;
    const OP_FUNCTION_PARAMETER: u32 = 55;
    const OP_FUNCTION_CALL: u32 = 57;
    const OP_VARIABLE: u32 = 59;
    const OP_LOAD: u32 = 61;
    const OP_STORE: u32 = 62;
    const OP_COMPOSITE_CONSTRUCT: u32 = 80;
    const OP_COMPOSITE_EXTRACT: u32 = 81;
    const OP_LABEL: u32 = 248;
    const OP_RETURN: u32 = 253;
    const OP_RETURN_VALUE: u32 = 254;
    const STORAGE_FUNCTION: u32 = 7;
    fn op(opcode: u32, operands: &[u32]) -> Vec<u32> {
        std::iter::once(((operands.len() as u32 + 1) << 16) | opcode).chain(operands.iter().copied()).collect()
    }
    let instructions = seal::instructions(module)?;
    let words_of = |instruction: &seal::Instruction| &module[instruction.start..instruction.start + instruction.words];
    let mut bound = module[3];
    let mut fresh = || {
        bound += 1;
        bound - 1
    };
    // Pointer type → pointee; function type → its words past its id; void.
    let mut pointees = HashMap::new();
    let mut signatures = HashMap::<Vec<u32>, u32>::new();
    let mut signature_of = HashMap::new();
    let mut voids = std::collections::HashSet::new();
    for instruction in &instructions {
        let words = words_of(instruction);
        match instruction.opcode {
            OP_TYPE_VOID => {
                voids.insert(words[1]);
            }
            OP_TYPE_POINTER => {
                pointees.insert(words[1], words[3]);
            }
            OP_TYPE_FUNCTION => {
                signatures.entry(words[2..].to_vec()).or_insert(words[1]);
                signature_of.insert(words[1], words[2..].to_vec());
            }
            _ => {}
        }
    }
    // By summarised function with a pointer parameter: what it becomes; and
    // the types it needs, declared before the first function.
    let mut retyped = HashMap::<u32, ByValue>::new();
    let mut declared = Vec::new();
    for instruction in &instructions {
        let words = words_of(instruction);
        if instruction.opcode != OP_FUNCTION {
            continue;
        }
        let Some(parameters) = summarised.get(&words[2]) else { continue };
        if parameters.iter().all(Option::is_none) {
            continue;
        }
        let original = signature_of
            .get(&words[4])
            .cloned()
            .ok_or_else(|| seal::MalformedModule(format!("%{} has no function type", words[4])))?;
        let (result, pointers) = (original[0], original[1..].to_vec());
        let void = voids.contains(&result);
        let written = parameters
            .iter()
            .enumerate()
            .filter(|(_, written)| **written == Some(true))
            .map(|(index, _)| (index, pointees[&pointers[index]]))
            .collect::<Vec<_>>();
        let returned = (!written.is_empty()).then(|| {
            let ty = fresh();
            let members = (!void).then_some(result).into_iter().chain(written.iter().map(|(_, pointee)| *pointee));
            declared.extend(op(OP_TYPE_STRUCT, &std::iter::once(ty).chain(members).collect::<Vec<_>>()));
            (ty, void, written)
        });
        let mut signature = vec![returned.as_ref().map_or(result, |(ty, _, _)| *ty)];
        for (index, parameter) in parameters.iter().enumerate() {
            signature.push(if parameter.is_some() { pointees[&pointers[index]] } else { pointers[index] });
        }
        let ty = match signatures.get(&signature) {
            Some(ty) => *ty,
            None => {
                let ty = fresh();
                declared.extend(op(OP_TYPE_FUNCTION, &[&[ty][..], &signature].concat()));
                signatures.insert(signature, ty);
                ty
            }
        };
        let values = parameters.iter().map(|parameter| parameter.map(|_| fresh())).collect();
        retyped.insert(words[2], ByValue { ty, values, pointers, returned, result });
    }

    let mut out = module[..5].to_vec();
    let mut declared = Some(declared);
    // What the function being written becomes; its parameter, while they
    // come; the variables its first block opens with, each with the value
    // stored to it, until stored.
    let mut function: Option<&ByValue> = None;
    let mut parameter = 0;
    let mut copies = Vec::new();
    // By parameter written: the variable taking its place.
    let mut variables = HashMap::new();
    // Copies' stores follow the variables of the first block.
    let mut storing = false;
    for instruction in &instructions {
        let words = words_of(instruction);
        let opcode = instruction.opcode;
        if storing && opcode != OP_VARIABLE {
            for &(pointer, value, _) in &copies {
                out.extend(op(OP_STORE, &[pointer, value]));
            }
            copies.clear();
            storing = false;
        }
        match opcode {
            OP_FUNCTION => {
                if let Some(types) = declared.take() {
                    out.extend(types);
                }
                function = retyped.get(&words[2]);
                parameter = 0;
                variables.clear();
                let mut words = words.to_vec();
                if summarised.contains_key(&words[2]) {
                    words[3] = (words[3] & !0x1) | uniformity::DONT_INLINE;
                }
                if let Some(rewritten) = function {
                    words[1] = rewritten.returned.as_ref().map_or(rewritten.result, |(ty, _, _)| *ty);
                    words[4] = rewritten.ty;
                }
                out.extend(words);
            }
            OP_FUNCTION_PARAMETER => {
                let value = function.and_then(|rewritten| rewritten.values[parameter]);
                match value {
                    Some(value) => {
                        out.extend(op(OP_FUNCTION_PARAMETER, &[pointees[&words[1]], value]));
                        copies.push((words[2], value, words[1]));
                        variables.insert(parameter, words[2]);
                    }
                    None => out.extend_from_slice(words),
                }
                parameter += 1;
            }
            OP_LABEL if !copies.is_empty() && !storing => {
                out.extend_from_slice(words);
                for &(pointer, _, ty) in &copies {
                    out.extend(op(OP_VARIABLE, &[ty, pointer, STORAGE_FUNCTION]));
                }
                storing = true;
            }
            // A function writing a pointer parameter returns its result
            // with what the variables taking their places end with.
            OP_RETURN | OP_RETURN_VALUE
                if function.is_some_and(|rewritten| rewritten.returned.is_some()) =>
            {
                let rewritten = function.expect("a rewritten function is being written");
                let (ty, void, written) = rewritten.returned.as_ref().expect("the function returns its writes");
                let mut members = if *void { Vec::new() } else { vec![words[1]] };
                for (index, pointee) in written {
                    let loaded = fresh();
                    out.extend(op(OP_LOAD, &[*pointee, loaded, variables[index]]));
                    members.push(loaded);
                }
                let returned = fresh();
                out.extend(op(OP_COMPOSITE_CONSTRUCT, &[&[*ty, returned][..], &members].concat()));
                out.extend(op(OP_RETURN_VALUE, &[returned]));
            }
            OP_FUNCTION_CALL => {
                let mut call = words.to_vec();
                let Some(rewritten) = retyped.get(&words[3]) else {
                    out.extend(call);
                    continue;
                };
                for (index, value) in rewritten.values.iter().enumerate() {
                    if value.is_some() {
                        let loaded = fresh();
                        out.extend(op(OP_LOAD, &[pointees[&rewritten.pointers[index]], loaded, words[4 + index]]));
                        call[4 + index] = loaded;
                    }
                }
                let Some((ty, void, written)) = &rewritten.returned else {
                    out.extend(call);
                    continue;
                };
                // A void call's result names nothing: the struct takes it.
                let returned = if *void { words[2] } else { fresh() };
                call[1] = *ty;
                call[2] = returned;
                out.extend(call);
                if !*void {
                    out.extend(op(OP_COMPOSITE_EXTRACT, &[rewritten.result, words[2], returned, 0]));
                }
                for (member, (index, pointee)) in written.iter().enumerate() {
                    let value = fresh();
                    let member = member as u32 + u32::from(!*void);
                    out.extend(op(OP_COMPOSITE_EXTRACT, &[*pointee, value, returned, member]));
                    out.extend(op(OP_STORE, &[words[4 + index], value]));
                }
            }
            _ => out.extend_from_slice(words),
        }
    }
    out[3] = bound;
    Ok(out)
}

/// The module formation ships of `module`, a cleaned and sealed module whose
/// execution model holds: each call of a 32-lane hardware join (`joins`)
/// replaced by its argument, and the join's function, with its names and
/// decorations, dropped, as is a function type left unreferenced, so the
/// join costs nothing and the module is the one an identity macro would
/// give, but for the numbering of its ids. It is rewritten as SPIR-V
/// assembly, where every id is a token, and assembled again with its ids
/// kept and the header's version and generator glslang wrote (drivers key
/// workarounds on the generator), then validated as formation validates.
/// Where `merges_returns` (the module declares `MaximallyReconvergesKHR` and
/// the environment enables `merge_maximal_reconvergence_returns`), early
/// returns are merged into one return block per function in a single optimizer
/// pass (`DeadBranchElim`, which the pass requires, then `MergeReturn`):
/// `vk::DriverId::AMD_PROPRIETARY` (observed on Windows `amdvlk64.dll`
/// 32.0.31019.6049 / driver `26.10.19.06 (LLPC)`) segfaults (`0xC0000005`) in
/// `vkCreateComputePipelines` on `spirv-val`-valid modules declaring
/// `MaximallyReconvergesKHR` with early returns. A module without joins or
/// merged returns is shipped as it is.
fn shipped(
    module: Vec<u32>,
    joins: &[uniformity::HardwareJoin],
    merges_returns: bool,
) -> Result<Vec<u32>, NativeCompilationError> {
    use spirv_tools::assembler::{Assembler, AssemblerOptions};
    const OP_FUNCTION_CALL: u32 = 57;
    let mut shipped = if joins.is_empty() {
        module
    } else {
        let functions = joins.iter().map(|join| join.ids[0]).collect::<std::collections::HashSet<_>>();
        // Each call's result, by the argument it is.
        let mut argument = std::collections::HashMap::new();
        for instruction in seal::instructions(&module).map_err(malformed)? {
            let words = &module[instruction.start..instruction.start + instruction.words];
            if instruction.opcode == OP_FUNCTION_CALL && functions.contains(&words[3]) {
                argument.insert(words[2], words[4]);
            }
        }
        let substitute = |mut id: u32| {
            while let Some(next) = argument.get(&id) {
                id = *next;
            }
            id
        };
        // Ids nothing shipped may name: the joins' own, and the calls'.
        let dropped = joins
            .iter()
            .flat_map(|join| join.ids)
            .chain(argument.keys().copied())
            .collect::<std::collections::HashSet<_>>();
        let text = assembly(&module)?;
        // Each kept line, with the function type it declares.
        let mut kept = Vec::new();
        let mut in_join = false;
        for line in text.lines() {
            let ids = named_ids(line);
            let mut tokens = line.split_whitespace();
            let first = tokens.next().expect("spirv-dis line has an opcode");
            let defined = first.starts_with('%').then(|| ids[0].1);
            let opcode = if defined.is_some() {
                tokens.nth(1).expect("spirv-dis line has an opcode")
            } else {
                first
            };
            if in_join {
                in_join = opcode != "OpFunctionEnd";
                continue;
            }
            if opcode == "OpFunction" && defined.is_some_and(|id| functions.contains(&id)) {
                in_join = true;
                continue;
            }
            if defined.is_some_and(|id| argument.contains_key(&id)) {
                continue;
            }
            if ids.iter().any(|(_, id)| dropped.contains(id)) {
                if matches!(opcode, "OpName" | "OpMemberName" | "OpDecorate" | "OpDecorateId" | "OpDecorateString" | "OpMemberDecorate") {
                    continue;
                }
                assert!(
                    ids.iter().all(|(_, id)| !dropped.contains(id) || argument.contains_key(id)),
                    "a hardware join is named other than by its calls: {line}"
                );
            }
            let mut rewritten = String::with_capacity(line.len());
            let mut at = 0;
            for (span, id) in &ids {
                rewritten.push_str(&line[at..span.start]);
                rewritten.push_str(&format!("%{}", substitute(*id)));
                at = span.end;
            }
            rewritten.push_str(&line[at..]);
            kept.push((rewritten, defined.filter(|_| opcode == "OpTypeFunction")));
        }
        // The ids the kept lines name besides those they declare.
        let named = kept
            .iter()
            .flat_map(|(line, _)| named_ids(line).into_iter().skip(usize::from(line.starts_with('%'))))
            .map(|(_, id)| id)
            .collect::<std::collections::HashSet<_>>();
        let text = kept
            .iter()
            .filter(|(_, function_type)| function_type.is_none_or(|id| named.contains(&id)))
            .map(|(line, _)| format!("{line}\n"))
            .collect::<String>();
        let mut shipped = spirv_tools::assembler::create(Some(target()))
            .assemble(&text, AssemblerOptions { preserve_numeric_ids: true })
            .map(|binary| binary.as_words().to_vec())
            .map_err(toolchain)?;
        shipped[1..3].copy_from_slice(&module[1..3]);
        shipped
    };
    if merges_returns {
        shipped = optimized(
            &shipped,
            &[
                spirv_tools::opt::Passes::DeadBranchElim,
                spirv_tools::opt::Passes::MergeReturn,
            ],
        )?;
    } else if !joins.is_empty() {
        validate(&shipped).map_err(NativeCompilationError::MalformedToolchainOutput)?;
    }
    Ok(shipped)
}

/// The ids a line of SPIR-V assembly without friendly names names outside
/// its strings, with where each stands.
fn named_ids(line: &str) -> Vec<(std::ops::Range<usize>, u32)> {
    let bytes = line.as_bytes();
    let mut ids = Vec::new();
    let (mut at, mut quoted) = (0, false);
    while at < bytes.len() {
        match bytes[at] {
            b'\\' if quoted => at += 1,
            b'"' => quoted = !quoted,
            b'%' if !quoted => {
                let end = (at + 1..bytes.len()).find(|index| !bytes[*index].is_ascii_digit()).unwrap_or(bytes.len());
                let id = line[at + 1..end].parse().expect("assembly without friendly names names each id by its number");
                ids.push((at..end, id));
                at = end;
                continue;
            }
            _ => {}
        }
        at += 1;
    }
    ids
}

/// `module` through SPIRV-Tools `passes`, validated first as formation
/// validates.
fn optimized(module: &[u32], passes: &[spirv_tools::opt::Passes]) -> Result<Vec<u32>, NativeCompilationError> {
    let mut optimizer = spirv_tools::opt::create(Some(target()));
    for pass in passes {
        optimizer.register_pass(*pass);
    }
    let mut messages = Vec::new();
    let options = spirv_tools::opt::Options {
        validator_options: Some(validator_options()),
        preserve_spec_constants: true,
        ..Default::default()
    };
    optimizer
        .optimize(
            module,
            &mut |message: spirv_tools::error::Message| messages.push(message.message),
            Some(options),
        )
        .map(|binary| binary.as_words().to_vec())
        .map_err(|error| {
            NativeCompilationError::MalformedToolchainOutput(format!(
                "{error}: {}",
                messages.join("; ")
            ))
        })
}

/// The lanes of the widest hardware subgroup a device runs a kernel with
/// (GCN's fixed 64).
const WIDEST_SUBGROUP: u32 = 64;

/// The workgroup size of `entry`, whose module reads its subgroups where
/// `reads_subgroups`. A pipeline that reads its subgroups requires them
/// full, so none is ever partial: its X extent is a whole number of
/// hardware subgroups on every device, the widest included, so it forms
/// alike everywhere. One that does not cannot tell how its invocations are
/// grouped, and takes any X extent.
fn launch_size(entry: &ProgramEntry, reads_subgroups: bool) -> Result<[u32; 3], NativeCompilationError> {
    let threads = entry.group_size.ok_or_else(|| {
        NativeCompilationError::MalformedToolchainOutput(format!("`{}` carries no workgroup size", entry.symbol))
    })?;
    if reads_subgroups && threads[0] % WIDEST_SUBGROUP != 0 {
        return Err(toolchain(format!(
            "ERROR: `{}` violates the Vulkan kernel contract: it reads its subgroups, and its workgroup X extent {} is not a multiple of {WIDEST_SUBGROUP}, the widest hardware subgroup",
            entry.symbol, threads[0]
        )));
    }
    Ok(threads)
}

/// The pipelines of one specialization, one per launch in order.
pub struct DirectModule {
    device: Device,
    pipelines: Vec<vk::Pipeline>,
    /// Each pipeline's executables' statistics, as the driver names them;
    /// empty where the device reports none (`DeviceInfo::pipeline_statistics`).
    /// Read once every pipeline is held ([`DirectModule::read_statistics`]).
    statistics: Vec<Vec<ProgramStatistic>>,
}

impl DirectModule {
    /// The pipelines of `entries`, each from its sealed SPIR-V with its
    /// constants (the workgroup size, ids 0..3, then the rest), as the
    /// driver compiled them, and their pipeline cache data, which
    /// [`DirectModule::load`] may create them from without compiling. A
    /// pipeline-creation failure is a toolchain failure. Every pipeline's
    /// statistics are read from the driver, here and on a load alike, so
    /// the artifact carries none.
    pub fn compile(
        device: &Device,
        entries: &[(&[u32], &ProgramEntry)],
    ) -> Result<(Self, Vec<u8>), NativeCompilationError> {
        let (mut module, data) = Self::created(device, entries, &[], vk::PipelineCreateFlags::empty())?
            .expect("a creation that may compile never reports that it would have to");
        module.read_statistics()?;
        Ok((module, data))
    }

    /// The pipelines of `entries` from `pipelines`, the data
    /// [`DirectModule::compile`] made of them on this device and driver,
    /// or `None` where the driver does not serve one of them from it
    /// without compiling (`FAIL_ON_PIPELINE_COMPILE_REQUIRED`): the
    /// pipeline cache is a hint the driver may decline, so a declined load
    /// is a miss, never a failure.
    pub(crate) fn load(
        device: &Device,
        entries: &[(&[u32], &ProgramEntry)],
        pipelines: &[u8],
    ) -> Result<Option<Self>, NativeCompilationError> {
        let Some((mut module, _)) = Self::created(
            device,
            entries,
            pipelines,
            vk::PipelineCreateFlags::FAIL_ON_PIPELINE_COMPILE_REQUIRED,
        )?
        else {
            return Ok(None);
        };
        module.read_statistics()?;
        Ok(Some(module))
    }

    /// The pipelines of `entries`, created in a cache that starts as
    /// `kept`, and what the cache then holds; `None` where `flags` forbid
    /// compiling and a pipeline would need it.
    fn created(
        device: &Device,
        entries: &[(&[u32], &ProgramEntry)],
        kept: &[u8],
        flags: vk::PipelineCreateFlags,
    ) -> Result<Option<(Self, Vec<u8>)>, NativeCompilationError> {
        let mut module = Self {
            device: device.clone(),
            pipelines: Vec::with_capacity(entries.len()),
            statistics: Vec::new(),
        };
        let vulkan = &device.inner.device;
        let cache = unsafe {
            vulkan.create_pipeline_cache(
                &vk::PipelineCacheCreateInfo::default().initial_data(kept),
                None,
            )
        }
        .map_err(|error| toolchain(format!("vkCreatePipelineCache failed: {error}")))?;
        let created = module.create(cache, entries, flags).and_then(|served| {
            served
                .then(|| unsafe { vulkan.get_pipeline_cache_data(cache) })
                .transpose()
                .map_err(|error| toolchain(format!("vkGetPipelineCacheData failed: {error}")))
        });
        unsafe { vulkan.destroy_pipeline_cache(cache, None) };
        Ok(created?.map(|data| (module, data)))
    }

    /// Create every pipeline of `entries` in order; `false`, with the
    /// pipelines created so far held for the module to destroy, at the
    /// first that `flags` forbid compiling and that would need it.
    fn create(
        &mut self,
        cache: vk::PipelineCache,
        entries: &[(&[u32], &ProgramEntry)],
        flags: vk::PipelineCreateFlags,
    ) -> Result<bool, NativeCompilationError> {
        let device = self.device.clone();
        for (words, entry) in entries {
            // The `OpFmaKHR` binding follows validation, which the linked
            // validator does not know.
            let pipeline = if device.properties().fma_32.holds() {
                let bound = seal::fma_khr(words).map_err(malformed)?;
                self.pipeline(cache, &bound, entry, flags)?
            } else {
                self.pipeline(cache, words, entry, flags)?
            };
            let Some(pipeline) = pipeline else {
                return Ok(false);
            };
            self.pipelines.push(pipeline);
        }
        Ok(true)
    }

    fn pipeline(
        &self,
        cache: vk::PipelineCache,
        words: &[u32],
        entry: &ProgramEntry,
        flags: vk::PipelineCreateFlags,
    ) -> Result<Option<vk::Pipeline>, NativeCompilationError> {
        // Whether the module as shipped for this device reads its
        // subgroups; [`form`] also held its launch to whole ones where only
        // another device's module does.
        let full = uniformity::reads_subgroups(words).map_err(malformed)?;
        let threads = launch_size(entry, full)?;
        let inner = &self.device.inner;
        let device = &inner.device;
        let shader = unsafe {
            device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(words), None)
        }
        .map_err(|error| {
            toolchain(format!(
                "vkCreateShaderModule of `{}` failed: {error}",
                entry.symbol
            ))
        })?;
        let values = threads.iter().chain(&entry.constants).collect::<Vec<_>>();
        let entries = (0..values.len() as u32)
            .map(|id| vk::SpecializationMapEntry {
                constant_id: id,
                offset: id * 4,
                size: 4,
            })
            .collect::<Vec<_>>();
        let data = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<u8>>();
        let specialization = vk::SpecializationInfo::default()
            .map_entries(&entries)
            .data(&data);
        let mut subgroup = vk::PipelineShaderStageRequiredSubgroupSizeCreateInfo::default()
            .required_subgroup_size(SUBGROUP_WIDTH);
        let stage = vk::PipelineShaderStageCreateInfo::default()
            .flags(if full {
                vk::PipelineShaderStageCreateFlags::REQUIRE_FULL_SUBGROUPS
            } else {
                vk::PipelineShaderStageCreateFlags::empty()
            })
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(shader)
            .name(c"main")
            .specialization_info(&specialization);
        let stage = if self.device.info().requires_subgroup_size() {
            stage.push_next(&mut subgroup)
        } else {
            stage
        };
        let info = vk::ComputePipelineCreateInfo::default()
            .flags(flags | created_with(&self.device))
            .stage(stage)
            .layout(inner.layout);
        let created = {
            let _creating = crate::device::creating_pipelines();
            unsafe { device.create_compute_pipelines(cache, &[info], None) }
        };
        unsafe { device.destroy_shader_module(shader, None) };
        // `PIPELINE_COMPILE_REQUIRED` is a success code: the pipeline
        // returned with it is null, and nothing was compiled.
        match created {
            Ok(pipelines) if pipelines[0] != vk::Pipeline::null() => Ok(Some(pipelines[0])),
            Ok(_) | Err((_, vk::Result::PIPELINE_COMPILE_REQUIRED)) => Ok(None),
            Err((_, error)) => Err(toolchain(format!(
                "vkCreateComputePipelines of `{}` failed: {error}",
                entry.symbol
            ))),
        }
    }

    pub(crate) fn pipeline_handle(&self, launch: usize) -> vk::Pipeline {
        self.pipelines[launch]
    }

    /// The statistics of launch `launch`'s pipeline, as the driver reports
    /// them.
    pub fn statistics(&self, launch: usize) -> &[ProgramStatistic] {
        &self.statistics[launch]
    }

    /// Read every pipeline's statistics. The module holds its pipelines
    /// first, so a failure to read them destroys them with it.
    fn read_statistics(&mut self) -> Result<(), NativeCompilationError> {
        self.statistics = self
            .pipelines
            .iter()
            .map(|pipeline| self.statistics_of(*pipeline))
            .collect::<Result<_, _>>()?;
        Ok(())
    }

    /// Every statistic of every executable of `pipeline`, named and
    /// described as the driver reports it (an executable's name prefixes
    /// the statistic's where the pipeline has more than one); none where
    /// the device reports no statistics.
    ///
    /// The NVIDIA proprietary driver (595, RTX 3090) reports "Register
    /// Count", "Binary Size", "Stack Size", "Local Memory Size" and "Shared
    /// Memory Size". Two of them are not values: "Local Memory Size" reads
    /// 2^36 for every pipeline (a sentinel) and "Stack Size" reads 0 for
    /// every pipeline. Its spill signal is "Register Count" at the 255
    /// ceiling: the projection GEMM's 64x64 sub-tiles reach it, with binaries
    /// 2-3x larger, and run 6-7x slower than the best tile.
    fn statistics_of(&self, pipeline: vk::Pipeline) -> Result<Vec<ProgramStatistic>, NativeCompilationError> {
        let Some(executables) = &self.device.inner.executables else {
            return Ok(Vec::new());
        };
        let failed = |error| toolchain(format!("pipeline executable statistics: {error}"));
        let text = |raw: Result<&std::ffi::CStr, _>| {
            raw.expect("the driver null-terminates a pipeline executable's text").to_string_lossy().into_owned()
        };
        let properties = unsafe {
            executables.get_pipeline_executable_properties(&vk::PipelineInfoKHR::default().pipeline(pipeline))
        }
        .map_err(failed)?;
        let several = properties.len() > 1;
        let mut statistics = Vec::new();
        for (index, executable) in properties.iter().enumerate() {
            let prefix = if several { format!("{}: ", text(executable.name_as_c_str())) } else { String::new() };
            let index = u32::try_from(index).expect("a pipeline's executables are counted in u32");
            let info = vk::PipelineExecutableInfoKHR::default().pipeline(pipeline).executable_index(index);
            for statistic in unsafe { executables.get_pipeline_executable_statistics(&info) }.map_err(failed)? {
                // SAFETY: `format` names the union's member the driver wrote.
                let value = unsafe {
                    match statistic.format {
                        vk::PipelineExecutableStatisticFormatKHR::BOOL32 => StatisticValue::Bool(statistic.value.b32 == vk::TRUE),
                        vk::PipelineExecutableStatisticFormatKHR::INT64 => StatisticValue::Int(statistic.value.i64),
                        vk::PipelineExecutableStatisticFormatKHR::UINT64 => StatisticValue::Uint(statistic.value.u64),
                        vk::PipelineExecutableStatisticFormatKHR::FLOAT64 => StatisticValue::Float(statistic.value.f64),
                        other => panic!("the driver reported pipeline statistic format {other:?}, which the extension does not define"),
                    }
                };
                statistics.push(ProgramStatistic {
                    name: format!("{prefix}{}", text(statistic.name_as_c_str())),
                    description: text(statistic.description_as_c_str()),
                    value,
                });
            }
        }
        Ok(statistics)
    }
}

impl Drop for DirectModule {
    fn drop(&mut self) {
        // Submissions that use the pipelines keep the module alive until
        // they complete.
        for pipeline in &self.pipelines {
            unsafe { self.device.inner.device.destroy_pipeline(*pipeline, None) };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ENVIRONMENT: seal::Environment = seal::Environment {
        float16: false,
        rounding_rte_32: true,
        denorm_preserve_32: false,
        subgroup_uniform_control_flow: true,
        maximal_reconvergence: false,
        merge_maximal_reconvergence_returns: false,
    };

    /// `body` as the launch `probe` of a source with a u64 and a float
    /// buffer.
    fn source(globals: &str, body: &str) -> String {
        format!(
            "#version 460\n\
             #extension GL_EXT_shader_explicit_arithmetic_types_int64 : require\n\
             #extension GL_KHR_cooperative_matrix : require\n\
             #extension GL_KHR_memory_scope_semantics : require\n\
             #extension GL_KHR_shader_subgroup_arithmetic : require\n\
             layout(local_size_x = 32) in;\n\
             layout(set = 0, binding = 0, std430) buffer Output {{ uint64_t x[]; }} data;\n\
             layout(set = 0, binding = 1, std430) buffer Floats {{ float x[]; }} floats;\n\
             {globals}\n\
             void probe() {{ {body} }}\n\
             void main() {{ SEISMIC_KERNEL(); }}\n"
        )
    }

    /// The module of [`source`].
    fn form(globals: &str, body: &str) -> Result<Vec<u32>, NativeCompilationError> {
        compile(&source(globals, body), "probe", ENVIRONMENT)
    }

    fn refused(globals: &str, body: &str, construct: &str) {
        match form(globals, body) {
            Err(NativeCompilationError::ToolchainFailure(message)) => assert!(
                message.starts_with("ERROR: `probe` violates") && message.contains(construct),
                "{body}: {message}"
            ),
            other => panic!("{body}: expected a contract violation, got {other:?}"),
        }
    }

    fn formed(globals: &str, body: &str) {
        if let Err(error) = form(globals, body) {
            panic!("{body}: {error}");
        }
    }

    const OUT: &str = "data.x[gl_GlobalInvocationID.x]";

    #[test]
    fn a_64_bit_division_is_refused_wherever_it_runs() {
        refused(
            "",
            &format!("{OUT} = data.x[0] / data.x[1];"),
            "64-bit OpUDiv",
        );
        // A literal divisor still divides at run time.
        refused(
            "",
            &format!("{OUT} = data.x[0] / uint64_t(3);"),
            "64-bit OpUDiv",
        );
        refused(
            "",
            &format!("{OUT} = uint64_t(int64_t(data.x[0]) % int64_t(data.x[1]));"),
            "64-bit OpSMod",
        );
        refused(
            "",
            &format!(
                "u64vec2 a = u64vec2(data.x[0], data.x[1]), b = u64vec2(data.x[2], data.x[3]); {OUT} = (a / b).x;"
            ),
            "64-bit OpUDiv",
        );
    }

    #[test]
    fn folded_and_32_bit_division_form() {
        formed("", &format!("{OUT} = uint64_t(123) / uint64_t(3);"));
        formed(
            "",
            &format!("{OUT} = uint64_t(uint(data.x[0]) / uint(data.x[1]) + uint(data.x[2]) % 7u);"),
        );
    }

    #[test]
    fn a_constant_table_in_per_invocation_storage_is_refused() {
        let table = "const uint64_t t[4] = uint64_t[4](3, 5, 7, 11);";
        refused(
            "",
            &format!("{table} {OUT} = t[uint(data.x[0]) & 3u];"),
            "constant table",
        );
        let large = (0..256)
            .map(|value| (value * value + 3).to_string())
            .collect::<Vec<_>>();
        refused(
            "",
            &format!(
                "const uint64_t t[256] = uint64_t[256]({}); {OUT} = t[uint(data.x[0]) & 255u];",
                large.join(", ")
            ),
            "constant table",
        );
        refused(
            "",
            &format!(
                "const uint64_t t[2][2] = uint64_t[2][2](uint64_t[2](3, 5), uint64_t[2](7, 11)); uint i = uint(data.x[0]); {OUT} = t[i & 1u][(i >> 1) & 1u];"
            ),
            "constant table",
        );
        // Rows equal to each other are still tables.
        refused(
            "",
            &format!(
                "const uint64_t t[2][2] = uint64_t[2][2](uint64_t[2](3, 5), uint64_t[2](3, 5)); uint i = uint(data.x[0]); {OUT} = t[i & 1u][(i >> 1) & 1u];"
            ),
            "constant table",
        );
        refused(
            "",
            &format!("{table} uint64_t u[4] = t; {OUT} = u[uint(data.x[0]) & 3u];"),
            "constant table",
        );
        // A table passed to a function is copied into its parameter.
        refused(
            "const uint64_t T[4] = uint64_t[4](3, 5, 7, 11);\n\
             uint64_t pick(const uint64_t t[4], uint i) { return t[i]; }",
            &format!("{OUT} = pick(T, uint(data.x[0]) & 3u);"),
            "constant table",
        );
    }

    #[test]
    fn working_state_and_vectors_form() {
        formed(
            "",
            &format!("uint64_t t[4] = uint64_t[4](0, 0, 0, 0); t[uint(data.x[0]) & 3u] = data.x[1]; {OUT} = t[uint(data.x[2]) & 3u];"),
        );
        formed(
            "",
            &format!("uint64_t t[4] = uint64_t[4](7, 7, 7, 7); t[uint(data.x[1]) & 3u] = data.x[2]; {OUT} = t[uint(data.x[0]) & 3u];"),
        );
        formed(
            "",
            &format!("float t[8]; for (uint i = 0u; i < 8u; i++) t[i] = 0.0; t[uint(data.x[1]) & 7u] = 1.0; {OUT} = uint64_t(uint(t[uint(data.x[0]) & 7u]));"),
        );
        formed(
            "",
            &format!("uvec4 t = uvec4(0); t[uint(data.x[0]) & 3u] = uint(data.x[1]); {OUT} = t[uint(data.x[2]) & 3u];"),
        );
        formed(
            "",
            &format!("u64vec4 t = u64vec4(3, 5, 7, 11); {OUT} = t[uint(data.x[0]) & 3u];"),
        );
        // Constant indexing folds the table away.
        formed(
            "",
            &format!("const uint64_t t[4] = uint64_t[4](3, 5, 7, 11); {OUT} = t[2] + data.x[0];"),
        );
    }

    /// A splat whose equal elements are distinct constants is still a splat:
    /// equality is of values, not ids.
    #[test]
    fn splats_compare_by_value() {
        let module = form(
            "",
            &format!("uint64_t t[4] = uint64_t[4](7, 7, 7, 7); t[uint(data.x[1]) & 3u] = data.x[2]; {OUT} = t[uint(data.x[0]) & 3u];"),
        )
        .expect("a splat forms");
        // Give the composite's first element a fresh constant of equal value.
        let mut at = 5;
        let (mut seven, mut composite) = (None, None);
        while at < module.len() {
            let (words, opcode) = ((module[at] >> 16) as usize, module[at] & 0xffff);
            if opcode == 43 && module[at + 3] == 7 && words == 5 {
                seven = Some(at);
            }
            if opcode == 44 && seven.is_some_and(|seven| module[at + 3] == module[seven + 2]) {
                composite = Some(at);
            }
            at += words;
        }
        let (seven, composite) = (
            seven.expect("the constant 7"),
            composite.expect("the splat"),
        );
        let fresh = module[3];
        let mut patched = module.clone();
        patched[3] += 1;
        patched[composite + 3 + 1] = fresh;
        let copy = [
            module[seven],
            module[seven + 1],
            fresh,
            module[seven + 3],
            module[seven + 4],
        ];
        let after = seven + 5;
        patched.splice(after..after, copy);
        assert_eq!(seal::contract(&patched), Ok(vec![]));
        // An unequal element makes it a table.
        let mut unequal = patched.clone();
        unequal[after + 3] = 8;
        assert_eq!(seal::contract(&unequal).expect("parses").len(), 1);
    }

    #[test]
    fn a_float_converted_to_a_64_bit_integer_is_refused() {
        let value = "uintBitsToFloat(uint(data.x[0]))";
        refused("", &format!("{OUT} = uint64_t({value});"), "64-bit OpConvertFToU");
        refused(
            "",
            &format!("{OUT} = uint64_t(int64_t({value}));"),
            "64-bit OpConvertFToS",
        );
        refused(
            "",
            &format!("{OUT} = u64vec2({value}, 1.0).x;"),
            "64-bit OpConvertFToU",
        );
        formed("", &format!("{OUT} = uint64_t(uint({value}));"));
        formed("", &format!("{OUT} = uint64_t(int({value}));"));
    }

    const ACCUMULATOR: &str = "coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator>";

    #[test]
    fn a_cooperative_matrix_constant_two_functions_read_is_refused() {
        let store = "coopMatStore(c, floats.x, 0, 16, gl_CooperativeMatrixLayoutRowMajor);";
        refused(
            &format!(
                "{ACCUMULATOR} zero_a() {{ return {ACCUMULATOR}(0.0); }}\n\
                 {ACCUMULATOR} zero_b() {{ return {ACCUMULATOR}(0.0); }}"
            ),
            &format!("{ACCUMULATOR} c = data.x[0] == 0ul ? zero_a() : zero_b(); {store}"),
            "cooperative-matrix constant",
        );
        // One function holds the constant; its callers read its result.
        formed(
            &format!(
                "{ACCUMULATOR} zero() {{ return {ACCUMULATOR}(0.0); }}\n\
                 {ACCUMULATOR} cleared() {{ return zero(); }}"
            ),
            &format!("{ACCUMULATOR} c = data.x[0] == 0ul ? zero() : cleared(); {store}"),
        );
    }

    #[test]
    fn a_64_bit_conditional_subtraction_is_a_select() {
        const CONSTRUCT: &str = "64-bit conditional subtraction";
        let operands = "uint64_t x = data.x[0], m = data.x[1];";
        let out = format!("{OUT} = x;");
        refused("", &format!("{operands} if (x >= m) x -= m; {out}"), CONSTRUCT);
        refused("", &format!("{operands} if (m <= x) x -= m; {out}"), CONSTRUCT);
        refused("", &format!("{operands} if (!(x < m)) x -= m; {out}"), CONSTRUCT);
        refused("", &format!("{operands} if (x < m) data.x[2] = 1ul; else x -= m; {out}"), CONSTRUCT);
        // `seismic_sqrt_rn`'s digit step: a subtrahend computed twice is one
        // value.
        refused(
            "",
            &format!("{operands} uint64_t r = data.x[2]; if (x >= r + m) x -= r + m; {out}"),
            CONSTRUCT,
        );
        // One a called function computes twice, each from its own read of
        // a parameter, is one value once the call is inlined.
        refused(REDUCE, &format!("{OUT} = reduce(data.x[0], data.x[1]);"), CONSTRUCT);
        // One a summarised function computes, called twice with one
        // argument, is one value in the copy the value rules read.
        refused(
            "uint64_t twice(uint64_t m) { return m * 2ul; }",
            &format!("{operands} if (x >= twice(m)) x -= twice(m); {out}"),
            CONSTRUCT,
        );
        refused(
            "",
            &format!("int64_t y = int64_t(data.x[0]), n = int64_t(data.x[1]); if (y > n) y -= n; {OUT} = uint64_t(y);"),
            CONSTRUCT,
        );
        // The select chooses between `m` and 0, not between `x` and `x - m`.
        formed("", &format!("{operands} x -= x >= m ? m : 0ul; {out}"));
        // A difference an arm only uses.
        formed("", &format!("{operands} if (x >= m) data.x[2] = x - m; {out}"));
        formed(
            "",
            &format!("uint x = uint(data.x[0]), m = uint(data.x[1]); if (x >= m) x -= m; {OUT} = uint64_t(x);"),
        );
    }

    /// A conditional subtraction whose subtrahend a function computes
    /// twice, each time from its own read of the parameter `m`.
    const REDUCE: &str = "uint64_t reduce(uint64_t x, uint64_t m) { if (x >= m * 2ul) x -= m * 2ul; return x; }";

    /// What a called function computes twice, the first computation
    /// dominating the second, is one id in the analysed copy: [`REDUCE`]'s
    /// two products of `m` are one. `reduce` is pure, so it stays one
    /// function, called, its parameters values, and is not inlined; the
    /// kernel that calls it is.
    #[test]
    fn a_value_a_summarised_function_computes_twice_is_one_id() {
        let cleaned = cleaned(&source(REDUCE, &format!("{OUT} = reduce(data.x[0], data.x[1]);")), "probe")
            .expect("compiles");
        let (joins, _) = uniformity::hardware_joins(&cleaned).expect("module");
        let Analysed { checked, numbered } = analysed(&cleaned, &joins).expect("analysed");
        let count = |module: &[u32], opcode: u32| {
            seal::instructions(module).expect("module").iter().filter(|instruction| instruction.opcode == opcode).count()
        };
        // `OpIMul`.
        assert_eq!(count(&cleaned, 132), 2);
        for module in [&checked, &numbered] {
            assert_eq!(count(module, 132), 1);
            // `OpFunction`: the entry point and `reduce`; `OpFunctionCall`:
            // the entry point's one call; no `OpFunctionParameter` is a
            // pointer, so no `OpLoad` is left in `reduce`.
            assert_eq!((count(module, 54), count(module, 57)), (2, 1));
        }
    }

    /// A pure function's call holds what its body returns of the call's
    /// arguments: a divergent argument steers control as the inlined body
    /// would, and the body's loops and selections count where the call is.
    #[test]
    fn a_summarised_call_steers_as_its_body_would() {
        let pick = "uint pick(uint x) { if (x > 3u) return x; return 7u; }";
        let store = "floats.x[gl_SubgroupID * 32u + gl_SubgroupInvocationID]";
        // A divergent argument makes the result divergent: a subgroup
        // operation under control it steers is refused; under a uniform
        // one's, it forms.
        refused(
            pick,
            &format!("if (pick(gl_SubgroupInvocationID) > 4u) {store} = subgroupAdd(1.0);"),
            "OpGroupNonUniformFAdd inside a divergent selection",
        );
        formed(pick, &format!("if (pick(gl_WorkGroupID.x) > 4u) {store} = subgroupAdd(1.0);"));
        // A loop the argument steers ends lane-varying at the call: a
        // subgroup operation after it needs maximal reconvergence.
        let steps = "uint steps(uint n) { uint s = 0u; for (uint i = 0u; i < n; ++i) s += i * i; return s; }";
        let after = |argument: &str| format!("{store} = float(subgroupAdd(steps({argument})));");
        refused(steps, &after("gl_SubgroupInvocationID"), "OpGroupNonUniformIAdd after a lane-varying loop");
        let maximal = seal::Environment { maximal_reconvergence: true, ..ENVIRONMENT };
        compile(&source(steps, &after("gl_SubgroupInvocationID")), "probe", maximal).expect("forms where declared");
        formed(steps, &after("gl_WorkGroupID.x"));
        // A loop of constant trips is not steered by a divergent argument
        // (the correctly rounded division's neighbour search).
        let nearest = "float nearest(float x) { float best = x; for (int s = -1; s <= 1; s += 2) { \
                       const float c = x + float(s); if (abs(c) < abs(best)) best = c; } return best; }";
        formed(nearest, &format!("{store} = subgroupAdd(nearest(floats.x[gl_SubgroupInvocationID]));"));
    }

    /// The calls left in the analysed copy of [`source`].
    fn calls(globals: &str, body: &str) -> usize {
        let cleaned = cleaned(&source(globals, body), "probe").expect("compiles");
        let (joins, _) = uniformity::hardware_joins(&cleaned).expect("module");
        let Analysed { checked, .. } = analysed(&cleaned, &joins).expect("analysed");
        seal::instructions(&checked).expect("module").iter().filter(|instruction| instruction.opcode == 57).count()
    }

    /// A function that reads workgroup or device memory, and writes none,
    /// is summarised, and its loads hold what the inlined body's would: a
    /// read of memory the launch writes is divergent; one of workgroup memory
    /// at a uniform address is uniform where a barrier orders the caller's
    /// stores before it, and divergent where one of them may race it.
    #[test]
    fn a_summarised_function_reads_memory_as_its_inlined_body_would() {
        let store = "floats.x[gl_SubgroupID * 32u + gl_SubgroupInvocationID]";
        let at = "float at(uint i) { return floats.x[i]; }";
        assert_eq!(calls(at, &format!("{store} = at(0u);")), 1);
        refused(at, &format!("if (at(0u) > 1.0) {store} = subgroupAdd(1.0);"), "inside a divergent selection");
        let shared = "shared float s[32]; float at(uint i) { return s[i]; }";
        let read = format!("if (at(0u) > 0.5) {store} = subgroupAdd(1.0);");
        formed(shared, &format!("s[gl_LocalInvocationIndex] = 1.0; barrier(); {read}"));
        refused(shared, &format!("s[gl_LocalInvocationIndex] = 1.0; {read}"), "inside a divergent selection");
    }

    /// A function is judged on what the constants its calls pass leave of
    /// it: a branch on a constant argument that stores to workgroup memory,
    /// which a call never takes, leaves the function summarised there.
    #[test]
    fn a_constant_argument_prunes_what_a_summarised_function_never_runs() {
        let pick = "shared float s[32]; \
                    float pick(const int kind, uint i) { if (kind == 1) { s[i] = 2.0; return 0.0; } return floats.x[i]; }";
        assert_eq!(calls(pick, "floats.x[gl_GlobalInvocationID.x] = pick(0, 3u);"), 1);
        assert_eq!(calls(pick, "floats.x[gl_GlobalInvocationID.x] = pick(1, 3u);"), 0);
    }

    /// A summary only accepts: where the check with summaries refuses, the
    /// check with every call inlined decides. Here the body's phi merges two
    /// arguments that are one value only through the call (`f(x, x)`), so
    /// with summaries the result is divergent under the body's divergent
    /// selection; inlined, the phi copies `x`, uniform, and the module forms.
    #[test]
    fn a_refusal_with_summaries_is_decided_inlined() {
        let either = "uint either(uint a, uint b, bool c) { uint r; if (c) r = a; else r = b; return r; }";
        let store = "floats.x[gl_SubgroupID * 32u + gl_SubgroupInvocationID]";
        let body = format!(
            "if (either(gl_WorkGroupID.x, gl_WorkGroupID.x, gl_SubgroupInvocationID > 3u) > 4u) {store} = subgroupAdd(1.0);"
        );
        let cleaned = cleaned(&source(either, &body), "probe").expect("compiles");
        let (joins, _) = uniformity::hardware_joins(&cleaned).expect("module");
        let summarised = analysed(&cleaned, &joins).expect("analysed").checked;
        let refusal = uniformity::check(&summarised, false).expect("module").violations;
        assert!(!refusal.is_empty(), "the summary over-approximates the phi");
        let (findings, _) = execution_model(&cleaned, &joins, summarised, false).expect("checked");
        assert!(findings.violations.is_empty(), "{:?}", findings.violations);
        formed(either, &body);
    }

    /// Workgroup memory is followed by byte where an access's address is a
    /// constant offset: invocation 0's store to word 1 does not race every
    /// invocation's read of word 0, which steers a loop or selection holding
    /// a barrier; a store that may touch word 0 does: to word 0 itself, at a
    /// variable offset, or to word 0 beside word 1.
    #[test]
    fn a_store_races_only_reads_of_the_words_it_may_touch() {
        let words = "shared uint w[4];";
        let steered = |store: &str, steer: &str| {
            format!("if (gl_LocalInvocationIndex == 0u) w[0] = 3u; barrier(); if (gl_LocalInvocationIndex == 0u) {{ {store} }} {steer}")
        };
        let looped = "for (uint i = 0u; i < w[0]; ++i) barrier();";
        let selected = "if (w[0] > 2u) barrier();";
        formed(words, &steered("w[1] = 7u;", looped));
        formed(words, &steered("w[1] = 7u;", selected));
        refused(words, &steered("w[0] = 7u;", looped), "OpControlBarrier inside a lane-varying loop");
        refused(words, &steered("w[0] = 7u;", selected), "OpControlBarrier inside a divergent selection");
        refused(words, &steered("w[gl_WorkGroupID.x % 4u] = 7u;", looped), "OpControlBarrier inside a lane-varying loop");
        refused(words, &steered("w[1] = 7u; w[0] = 5u;", looped), "OpControlBarrier inside a lane-varying loop");
    }

    /// A store in one way of a branch the whole workgroup takes alike does
    /// not race a read only the other way reaches in the interval: here the
    /// store's way crosses a barrier before rejoining, so no invocation is
    /// at both. The store races the read where its way reaches the read
    /// without a barrier, and where the branch may part the workgroup (its
    /// condition read beside a store).
    #[test]
    fn a_branch_the_workgroup_takes_alike_parts_its_ways_stores_from_the_others_reads() {
        let words = "shared uint w[4];";
        let parted = |before: &str, way: &str| {
            format!(
                "if (gl_LocalInvocationIndex == 0u) w[1] = 0u; barrier(); {before} \
                 if (w[1] == 0u) {{ if (gl_LocalInvocationIndex == 0u) w[0] = 3u; {way} }} \
                 if (w[0] > 2u) barrier();"
            )
        };
        formed(words, &parted("", "barrier();"));
        refused(words, &parted("", ""), "OpControlBarrier inside a divergent selection");
        refused(
            words,
            &parted("if (gl_LocalInvocationIndex == 1u) w[1] = 5u;", "barrier();"),
            "OpControlBarrier inside a divergent selection",
        );
    }

    /// A subgroup barrier after a lane-varying loop holds its whole subgroup
    /// only where the module declares maximal reconvergence: the check
    /// accepts it only when told the module does, and then says the module
    /// relies on it, so the seal declares it. Inside the loop it is refused
    /// either way. A workgroup barrier there keeps its own rule: legal where
    /// every invocation of the workgroup reaches it, in either mode, and
    /// relying on no reconvergence.
    #[test]
    fn a_subgroup_barrier_after_a_lane_varying_loop_needs_maximal_reconvergence() {
        let looped = |inside: &str, after: &str| {
            format!(
                "float acc = 0.0; for (uint i = gl_SubgroupInvocationID; i < 100u; i += 32u) {{ acc += floats.x[i]; {inside} }} \
                 {after} floats.x[gl_SubgroupID * 32u + gl_SubgroupInvocationID] = acc;"
            )
        };
        let findings = |body: &str, maximal_reconvergence: bool| {
            let cleaned = cleaned(&source("", body), "probe").expect("compiles");
            let (joins, _) = uniformity::hardware_joins(&cleaned).expect("module");
            uniformity::check(&analysed(&cleaned, &joins).expect("analysed").checked, maximal_reconvergence)
                .expect("module")
        };
        let after = looped("", "subgroupBarrier();");
        let relying = findings(&after, true);
        assert!(relying.violations.is_empty() && relying.relies_on_maximal_reconvergence, "{relying:?}");
        let refusal = findings(&after, false).violations;
        assert!(refusal.iter().any(|violation| violation.0.contains("OpControlBarrier of a subgroup after a lane-varying loop")), "{refusal:?}");
        refused("", &after, "OpControlBarrier of a subgroup after a lane-varying loop");
        let maximal = seal::Environment { maximal_reconvergence: true, ..ENVIRONMENT };
        compile(&source("", &after), "probe", maximal).expect("forms where declared");
        let inside = looped("subgroupBarrier();", "");
        for maximal_reconvergence in [false, true] {
            let refusal = findings(&inside, maximal_reconvergence).violations;
            assert!(refusal.iter().any(|violation| violation.0.contains("OpControlBarrier inside a lane-varying loop")), "{refusal:?}");
        }
        let workgroup = looped("", "barrier();");
        for maximal_reconvergence in [false, true] {
            let legal = findings(&workgroup, maximal_reconvergence);
            assert!(legal.violations.is_empty() && !legal.relies_on_maximal_reconvergence, "{legal:?}");
        }
    }

    #[test]
    fn a_launch_reading_its_subgroups_through_a_join_holds_whole_subgroups() {
        let join = "bool seismic_hardware_any(const bool value) { return value; }";
        let body = "if (seismic_hardware_any(gl_WorkGroupID.x > 3u)) data.x[gl_GlobalInvocationID.x] = 1ul;";
        let launch = |x: u32| ProgramEntry { symbol: "probe".into(), group_size: Some([x, 1, 1]), constants: Vec::new() };
        // The module shipped for 32 lanes reads no subgroups, but a 64-lane
        // device's subgroup vote does.
        let module = super::form(&source(join, body), &launch(64), ENVIRONMENT).expect("formed");
        assert!(!uniformity::reads_subgroups(&module).expect("module"));
        match super::form(&source(join, body), &launch(32), ENVIRONMENT) {
            Err(NativeCompilationError::ToolchainFailure(message)) => {
                assert!(message.contains("not a multiple of 64"), "{message}")
            }
            other => panic!("expected the launch refused, got {other:?}"),
        }
        super::form(&source("", "data.x[gl_GlobalInvocationID.x] = 1ul;"), &launch(32), ENVIRONMENT).expect("formed");
    }

    /// A lane-strided loop (`seismic_strided`): the plain loop, each lane
    /// leaving at its own bound, then a subgroup reduction; and the rounds
    /// form, whose trip count is uniform and whose bound is a guard inside.
    fn strided(plain: bool) -> String {
        let end = "gl_NumWorkGroups.x * 7u";
        let head = if plain {
            format!("for (uint i = gl_SubgroupInvocationID; i < {end}; i += 32u)")
        } else {
            format!("for (uint r = 0u, i = gl_SubgroupInvocationID; r < {end}; r += 32u, i += 32u) if (i < {end})")
        };
        format!(
            "float acc = 0.0; {head} acc += floats.x[i]; \
             data.x[gl_SubgroupID * 32u + gl_SubgroupInvocationID] = uint64_t(floatBitsToUint(subgroupAdd(acc)));"
        )
    }

    /// A subgroup operation after a lane-varying loop forms only for a
    /// module that declares maximal reconvergence, and the check says it
    /// relies on it; the rounds form forms either way and relies on none.
    #[test]
    fn a_subgroup_operation_after_a_lane_varying_loop_needs_maximal_reconvergence() {
        let maximal = seal::Environment { maximal_reconvergence: true, ..ENVIRONMENT };
        let findings = |plain: bool, maximal_reconvergence: bool| {
            let cleaned = cleaned(&source("", &strided(plain)), "probe").expect("compiles");
            let (joins, _) = uniformity::hardware_joins(&cleaned).expect("module");
            uniformity::check(&analysed(&cleaned, &joins).expect("analysed").checked, maximal_reconvergence)
                .expect("module")
        };
        compile(&source("", &strided(true)), "probe", maximal).expect("the plain loop forms where declared");
        refused("", &strided(true), "OpGroupNonUniformFAdd after a lane-varying loop");
        let relying = findings(true, true);
        assert!(relying.violations.is_empty() && relying.relies_on_maximal_reconvergence, "{relying:?}");
        for maximal_reconvergence in [false, true] {
            let rounds = findings(false, maximal_reconvergence);
            assert!(rounds.violations.is_empty() && !rounds.relies_on_maximal_reconvergence, "{rounds:?}");
        }
        compile(&source("", &strided(false)), "probe", maximal).expect("the rounds form forms where declared");
        formed("", &strided(false));
    }

    /// A module declaring `MaximallyReconvergesKHR` merges early returns only
    /// when `merge_maximal_reconvergence_returns` is set
    /// (`vk::DriverId::AMD_PROPRIETARY`); when false, the shipped SPIR-V is
    /// bit-for-bit identical to the sealed module (early returns preserved and
    /// no synthetic `OpLoopMerge` added).
    #[test]
    fn a_module_with_maximal_reconvergence_merges_early_returns() {
        const OP_FUNCTION: u32 = 54;
        const OP_FUNCTION_END: u32 = 56;
        const OP_RETURN: u32 = 253;
        const OP_RETURN_VALUE: u32 = 254;
        let max_returns = |module: &[u32]| {
            let (mut current, mut max) = (0usize, 0usize);
            for instruction in seal::instructions(module).expect("valid module") {
                match instruction.opcode {
                    OP_FUNCTION => current = 0,
                    OP_RETURN | OP_RETURN_VALUE => current += 1,
                    OP_FUNCTION_END => max = max.max(current),
                    _ => {}
                }
            }
            max
        };
        let helper = "void maybe_store(uint x) { if (x == 0u) return; data.x[x] = 1ul; }";
        let body = format!("maybe_store(gl_WorkGroupID.x); {}", strided(true));
        let glsl = source(helper, &body);
        let merged_env = seal::Environment {
            maximal_reconvergence: true,
            merge_maximal_reconvergence_returns: true,
            ..ENVIRONMENT
        };
        let merged = compile(&glsl, "probe", merged_env).expect("compiles");
        assert!(
            max_returns(&merged) <= 1,
            "expected at most one return per function when merging returns, found {}",
            max_returns(&merged)
        );
        let unmerged_env = seal::Environment {
            maximal_reconvergence: true,
            merge_maximal_reconvergence_returns: false,
            ..ENVIRONMENT
        };
        let unmerged = compile(&glsl, "probe", unmerged_env).expect("compiles");
        assert!(
            max_returns(&unmerged) > 1,
            "expected early returns preserved when merge_maximal_reconvergence_returns is false, found {}",
            max_returns(&unmerged)
        );
        // Without the workaround, `shipped()` must not rewrite the sealed SPIR-V at all.
        let cleaned = cleaned(&glsl, "probe").expect("cleaned");
        let sealed = seal::seal(&cleaned, unmerged_env, false, true).expect("sealed");
        assert_eq!(
            unmerged, sealed,
            "expected bit-for-bit identical SPIR-V when merge_maximal_reconvergence_returns is false"
        );
    }
}
