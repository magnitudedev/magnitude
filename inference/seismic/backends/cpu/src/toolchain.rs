//! The CPU's program toolchain.
//!
//! A program's source is one kernel function in CLIF, Cranelift's textual
//! IR, after the read-only data objects it addresses: every function it
//! calls is a host symbol it names ([`numeric::host_symbols`]) and every
//! table it reads is a data object of the program, so the text holds no
//! address of the process that emitted it. Compiling parses the text and
//! compiles it with Cranelift for the device's ISA, without
//! position-independent code and with every call and symbol non-colocated:
//! each address the code takes is then an 8-byte absolute (`Abs8`)
//! relocation, the only kind an artifact can hold. The artifact is the code
//! bytes, the data and those relocations; loading copies them into
//! executable memory ([`CoherentCode`]) and patches each relocation. A
//! compile forms its program through that same loader, so a loaded program
//! is exactly the compiled one, and a load runs no Cranelift.

use crate::command::CoherentCode;
use crate::emit::EmittedKernel;
use crate::workers::LaunchEntry;
use crate::{numeric, Cpu};
use cranelift_codegen::binemit::Reloc;
use cranelift_codegen::control::ControlPlane;
use cranelift_codegen::ir::{self, ExternalName};
use cranelift_codegen::isa::OwnedTargetIsa;
use cranelift_codegen::FinalizedRelocTarget;
use seismic_native_target::{
    Artifact, Declined, DeviceDescription, NativeCompilationError, ProgramEntry, ProgramSource,
    ProgramStatistic, Toolchain, ToolchainIdentity,
};
use sha2::{Digest, Sha256};

/// This build's formation code ([`build.rs`](../build.rs)): part of every
/// stored artifact's key, never of tuning's.
pub const FORMATION_BUILD: &str = env!("SEISMIC_FORMATION_BUILD");

/// The symbol of the one function of a kernel's program.
pub(crate) const KERNEL_SYMBOL: &str = "seismic_cpu_kernel";

/// The symbol of a program's data object `index`.
pub(crate) fn data_symbol(index: usize) -> String {
    format!("{DATA_PREFIX}{index}")
}

const DATA_PREFIX: &str = "seismic_data_";

/// Where each data object starts in a program's image, past its code.
const DATA_ALIGNMENT: usize = 16;

pub struct CpuToolchain {
    /// The device's ISA, or why it cannot be built: every compile then fails
    /// with it.
    isa: Result<OwnedTargetIsa, NativeCompilationError>,
    /// Whether the ISA emits branch-target landing pads (aarch64 BTI), so
    /// program memory is guarded where the host enforces it.
    branch_targets: bool,
    identity: ToolchainIdentity,
}

impl CpuToolchain {
    pub fn new(device: &DeviceDescription<Cpu>) -> Self {
        let policy = &device.device_info().codegen;
        let flags = |flags: &std::collections::BTreeMap<String, String>| {
            flags
                .iter()
                .map(|(name, value)| format!("{name}={value}"))
                .collect::<Vec<_>>()
                .join(",")
        };
        Self {
            isa: policy.isa(),
            branch_targets: policy.isa_flags.get("use_bti").is_some_and(|value| value == "true"),
            identity: ToolchainIdentity {
                namespace: "cpu",
                material: format!(
                    "cpu;{};{};{};shared:{};isa:{}",
                    policy.compiler,
                    policy.triple,
                    policy.call_conv,
                    flags(&policy.shared_flags),
                    flags(&policy.isa_flags),
                ),
                stored: format!("build {FORMATION_BUILD}"),
            },
        }
    }
}

/// The source of an emitted kernel's program: one line per data object,
/// `data %<symbol> <hex bytes>`, in order, then the function's CLIF.
pub(crate) fn source(emitted: &EmittedKernel) -> ProgramSource {
    let mut text = String::new();
    for (index, data) in emitted.data.iter().enumerate() {
        text.push_str("data %");
        text.push_str(&data_symbol(index));
        text.push(' ');
        for byte in data {
            text.push_str(&format!("{byte:02x}"));
        }
        text.push('\n');
    }
    text.push_str(&emitted.function.display().to_string());
    ProgramSource {
        text,
        entries: vec![ProgramEntry::named(KERNEL_SYMBOL)],
    }
}

/// The function and data objects of a program's source ([`source`]).
fn parse(text: &str) -> Result<(ir::Function, Vec<Vec<u8>>), NativeCompilationError> {
    let malformed = |reason: String| {
        NativeCompilationError::ToolchainFailure(format!("CPU program source: {reason}"))
    };
    let mut data = Vec::new();
    let mut rest = text;
    while let Some(line) = rest.strip_prefix("data %") {
        let (line, after) = line.split_once('\n').unwrap_or((line, ""));
        let (symbol, hex) = line
            .split_once(' ')
            .ok_or_else(|| malformed(format!("data line `{line}` names no bytes")))?;
        if symbol != data_symbol(data.len()) {
            return Err(malformed(format!(
                "data object {} is named `{symbol}`",
                data.len()
            )));
        }
        if hex.len() % 2 != 0 {
            return Err(malformed(format!("data `{symbol}` holds half a byte")));
        }
        let bytes = (0..hex.len())
            .step_by(2)
            .map(|at| u8::from_str_radix(&hex[at..at + 2], 16))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|error| malformed(format!("data `{symbol}`: {error}")))?;
        data.push(bytes);
        rest = after;
    }
    let mut functions = cranelift_reader::parse_functions(rest)
        .map_err(|error| malformed(format!("CLIF: {error}")))?;
    let function = functions
        .pop()
        .filter(|_| functions.is_empty())
        .ok_or_else(|| malformed("a program holds exactly one function".into()))?;
    Ok((function, data))
}

/// The one entry of a CPU program, as `entries` must request it.
fn single_entry(
    entries: &[ProgramEntry],
    symbol: &str,
) -> Result<(), NativeCompilationError> {
    match entries {
        [entry] if entry.symbol == symbol => Ok(()),
        _ => Err(NativeCompilationError::MalformedToolchainOutput(format!(
            "a CPU program holds the one entry `{symbol}`; requested {:?}",
            entries.iter().map(|entry| &entry.symbol).collect::<Vec<_>>()
        ))),
    }
}

/// A CPU program's artifact: its function's code and frame size, the data
/// objects placed after the code, and the code's relocations. Every
/// relocation is an 8-byte absolute address: no other kind has a
/// representation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CpuArtifact {
    symbol: String,
    frame_size: u32,
    code: Vec<u8>,
    data: Vec<Vec<u8>>,
    relocations: Vec<Relocation>,
}

/// The 8 bytes at `offset` in the code hold `target`'s address plus
/// `addend`.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Relocation {
    offset: u32,
    target: Target,
    addend: i64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum Target {
    /// A host symbol of this crate ([`numeric::host_symbols`]).
    Host(String),
    /// A C library function Cranelift calls for an operation the ISA lacks,
    /// which the process exports.
    Library(String),
    /// The program's data object of this index.
    Data(u32),
}

impl CpuArtifact {
    /// Its bytes, as [`Artifact::decode`] reads them: little-endian `u32`
    /// lengths and counts before their contents.
    fn encode(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        let put = |bytes: &mut Vec<u8>, part: &[u8]| {
            bytes.extend((part.len() as u32).to_le_bytes());
            bytes.extend(part);
        };
        put(&mut bytes, self.symbol.as_bytes());
        bytes.extend(self.frame_size.to_le_bytes());
        put(&mut bytes, &self.code);
        bytes.extend((self.data.len() as u32).to_le_bytes());
        for data in &self.data {
            put(&mut bytes, data);
        }
        bytes.extend((self.relocations.len() as u32).to_le_bytes());
        for relocation in &self.relocations {
            bytes.extend(relocation.offset.to_le_bytes());
            bytes.extend(relocation.addend.to_le_bytes());
            match &relocation.target {
                Target::Host(name) => {
                    bytes.push(0);
                    put(&mut bytes, name.as_bytes());
                }
                Target::Library(name) => {
                    bytes.push(1);
                    put(&mut bytes, name.as_bytes());
                }
                Target::Data(index) => {
                    bytes.push(2);
                    bytes.extend(index.to_le_bytes());
                }
            }
        }
        bytes
    }

    /// Whether every relocation patches 8 bytes within the code and every
    /// data target is one of the program's objects.
    fn well_formed(&self) -> bool {
        !self.code.is_empty()
            && self.relocations.iter().all(|relocation| {
                relocation.offset as usize + 8 <= self.code.len()
                    && match relocation.target {
                        Target::Data(index) => (index as usize) < self.data.len(),
                        Target::Host(_) | Target::Library(_) => true,
                    }
            })
    }
}

/// Reads an encoded artifact front to back.
struct Reader<'a>(&'a [u8]);

impl<'a> Reader<'a> {
    fn take(&mut self, bytes: usize) -> Option<&'a [u8]> {
        let (taken, rest) = self.0.split_at_checked(bytes)?;
        self.0 = rest;
        Some(taken)
    }

    fn u8(&mut self) -> Option<u8> {
        Some(self.take(1)?[0])
    }

    fn u32(&mut self) -> Option<u32> {
        Some(u32::from_le_bytes(self.take(4)?.try_into().ok()?))
    }

    fn i64(&mut self) -> Option<i64> {
        Some(i64::from_le_bytes(self.take(8)?.try_into().ok()?))
    }

    fn part(&mut self) -> Option<&'a [u8]> {
        let bytes = self.u32()? as usize;
        self.take(bytes)
    }

    fn text(&mut self) -> Option<String> {
        String::from_utf8(self.part()?.to_vec()).ok()
    }
}

impl Artifact for CpuArtifact {
    const ENCODING: u32 = 1;

    fn decode(bytes: &[u8], _: Option<seismic_native_target::HeldFile>) -> Option<Self> {
        let mut reader = Reader(bytes);
        let symbol = reader.text()?;
        let frame_size = reader.u32()?;
        let code = reader.part()?.to_vec();
        let data = (0..reader.u32()?)
            .map(|_| reader.part().map(<[u8]>::to_vec))
            .collect::<Option<Vec<_>>>()?;
        let relocations = (0..reader.u32()?)
            .map(|_| {
                let offset = reader.u32()?;
                let addend = reader.i64()?;
                let target = match reader.u8()? {
                    0 => Target::Host(reader.text()?),
                    1 => Target::Library(reader.text()?),
                    2 => Target::Data(reader.u32()?),
                    _ => return None,
                };
                Some(Relocation {
                    offset,
                    target,
                    addend,
                })
            })
            .collect::<Option<Vec<_>>>()?;
        let artifact = Self {
            symbol,
            frame_size,
            code,
            data,
            relocations,
        };
        (reader.0.is_empty() && artifact.well_formed()).then_some(artifact)
    }
}

/// A formed CPU program: its one function, loaded into executable memory.
pub struct CpuProgram {
    entry: LaunchEntry,
    frame_size: u32,
    code_bytes: u64,
    /// The SHA-256 of the function's code before relocation.
    digest: [u8; 32],
    /// The memory `entry` points into.
    _code: CoherentCode,
}

impl CpuProgram {
    pub(crate) fn entry(&self) -> LaunchEntry {
        self.entry
    }

    /// The function's stack frame, in bytes.
    pub(crate) fn frame_size(&self) -> u64 {
        u64::from(self.frame_size)
    }

    pub(crate) fn code_bytes(&self) -> u64 {
        self.code_bytes
    }

    pub(crate) fn digest(&self) -> [u8; 32] {
        self.digest
    }
}

/// The program of `artifact`: its code and data copied into executable
/// memory, each relocation patched to its target's address in this process.
/// A relocation to a host symbol this crate does not define or a library
/// function the process does not export is an invariant violation: every
/// symbol a program names is one of them.
fn load_image(artifact: &CpuArtifact, branch_targets: bool) -> CpuProgram {
    let mut placed = Vec::with_capacity(artifact.data.len());
    let mut bytes = artifact.code.len();
    for data in &artifact.data {
        bytes = bytes.next_multiple_of(DATA_ALIGNMENT);
        placed.push(bytes);
        bytes += data.len();
    }
    let host = numeric::host_symbols();
    let code = CoherentCode::new(bytes, branch_targets, |image, base| {
        image[..artifact.code.len()].copy_from_slice(&artifact.code);
        for (data, &offset) in artifact.data.iter().zip(&placed) {
            image[offset..offset + data.len()].copy_from_slice(data);
        }
        for relocation in &artifact.relocations {
            let address = match &relocation.target {
                Target::Host(name) => host
                    .iter()
                    .find(|(symbol, _)| symbol == name)
                    .map(|(_, address)| *address as usize)
                    .unwrap_or_else(|| {
                        panic!("a CPU program calls `{name}`, which is no host symbol")
                    }),
                Target::Library(name) => crate::library::address(name),
                Target::Data(index) => base + placed[*index as usize],
            };
            let value = (address as u64).wrapping_add_signed(relocation.addend);
            let at = relocation.offset as usize;
            image[at..at + 8].copy_from_slice(&value.to_ne_bytes());
        }
    });
    // SAFETY: the image's code starts with the kernel function, compiled for
    // the launch signature under the device's call convention.
    let entry = unsafe { std::mem::transmute::<*const u8, LaunchEntry>(code.base()) };
    CpuProgram {
        entry,
        frame_size: artifact.frame_size,
        code_bytes: artifact.code.len() as u64,
        digest: Sha256::digest(&artifact.code).into(),
        _code: code,
    }
}

/// The target of a relocation Cranelift emitted into a program whose data
/// objects are `data`.
fn target(name: &ExternalName, data: usize) -> Target {
    match name {
        ExternalName::TestCase(symbol) => {
            let symbol = std::str::from_utf8(symbol.raw())
                .expect("a program names its symbols in UTF-8");
            match symbol.strip_prefix(DATA_PREFIX) {
                Some(index) => {
                    let index = index
                        .parse::<u32>()
                        .ok()
                        .filter(|index| (*index as usize) < data)
                        .unwrap_or_else(|| {
                            panic!("a CPU program addresses `{symbol}`, which it does not define")
                        });
                    Target::Data(index)
                }
                None => Target::Host(symbol.to_owned()),
            }
        }
        ExternalName::LibCall(call) => Target::Library(cranelift_module::default_libcall_names()(*call)),
        ExternalName::User(_) | ExternalName::KnownSymbol(_) => {
            panic!("a CPU program names every symbol it takes; Cranelift emitted {name:?}")
        }
    }
}

impl Toolchain for CpuToolchain {
    type Artifact = CpuArtifact;
    type Program = CpuProgram;
    /// The artifact itself: the compile made it.
    type Pending = CpuArtifact;

    fn identity(&self) -> &ToolchainIdentity {
        &self.identity
    }

    fn compile(
        &self,
        source: &ProgramSource,
    ) -> Result<(CpuArtifact, CpuProgram), NativeCompilationError> {
        let isa = self.isa.as_ref().map_err(Clone::clone)?;
        let (function, data) = parse(&source.text)?;
        let symbol = match &function.name {
            ir::UserFuncName::Testcase(name) => std::str::from_utf8(name.raw())
                .map_err(|error| NativeCompilationError::ToolchainFailure(error.to_string()))?
                .to_owned(),
            ir::UserFuncName::User(_) => {
                return Err(NativeCompilationError::ToolchainFailure(
                    "a CPU program's function is named by its symbol".into(),
                ))
            }
        };
        single_entry(&source.entries, &symbol)?;
        let mut context = cranelift_codegen::Context::for_function(function);
        let compiled = context
            .compile(isa.as_ref(), &mut ControlPlane::default())
            .map_err(|error| {
                NativeCompilationError::ToolchainFailure(format!(
                    "Cranelift compilation: {:?}",
                    error.inner
                ))
            })?;
        if compiled.code_buffer().is_empty() {
            return Err(NativeCompilationError::ToolchainFailure(
                "Cranelift produced an empty native function".into(),
            ));
        }
        // Without position-independent code every address the function
        // takes, of a non-colocated call, library call or symbol, is
        // materialized whole (x86-64 `movabs`, aarch64 a literal load), an
        // `Abs8`; nothing else is relocated outside the function.
        let relocations = compiled
            .buffer
            .relocs()
            .iter()
            .map(|relocation| {
                let FinalizedRelocTarget::ExternalName(name) = &relocation.target else {
                    panic!(
                        "Cranelift relocated a CPU program against its own code: {:?}",
                        relocation.target
                    )
                };
                assert!(
                    relocation.kind == Reloc::Abs8,
                    "Cranelift emitted a {:?} relocation into non-PIC code with non-colocated calls",
                    relocation.kind
                );
                Relocation {
                    offset: relocation.offset,
                    target: target(name, data.len()),
                    addend: relocation.addend,
                }
            })
            .collect();
        let artifact = CpuArtifact {
            symbol,
            frame_size: compiled.frame_size,
            code: compiled.code_buffer().to_vec(),
            data,
            relocations,
        };
        let program = load_image(&artifact, self.branch_targets);
        Ok((artifact, program))
    }

    fn artifact(&self, artifact: CpuArtifact) -> Result<Vec<u8>, NativeCompilationError> {
        Ok(artifact.encode())
    }

    /// An artifact of another entry than the one requested is declined: a
    /// miss, compiled again.
    fn load(&self, artifact: &CpuArtifact, entries: &[ProgramEntry]) -> Result<CpuProgram, Declined> {
        single_entry(entries, &artifact.symbol)?;
        Ok(load_image(artifact, self.branch_targets))
    }

    fn statistics(_: &CpuProgram, _: usize) -> &[ProgramStatistic] {
        &[]
    }
}

/// The programs of this crate's tests, below the runtime and its store:
/// each source compiled, its artifact encoded and decoded, and the program
/// loaded from it, as a later process would form it.
#[cfg(test)]
pub(crate) fn test_programs(device: &DeviceDescription<Cpu>) -> crate::CpuPrograms {
    struct Stored(CpuToolchain);

    impl seismic_native_target::Programs<CpuToolchain> for Stored {
        fn form(
            &self,
            sources: &[ProgramSource],
        ) -> Result<Vec<seismic_native_target::Formed<CpuProgram>>, NativeCompilationError>
        {
            sources
                .iter()
                .map(|source| {
                    let (pending, compiled) = self.0.compile(source)?;
                    let bytes = self.0.artifact(pending)?;
                    let artifact = CpuArtifact::decode(&bytes, None)
                        .expect("an encoded CPU artifact decodes");
                    let loaded = self
                        .0
                        .load(&artifact, &source.entries)
                        .expect("a CPU artifact loads its own entry");
                    assert_eq!(loaded.digest, compiled.digest);
                    let key = seismic_native_target::ArtifactKey::of(&[source.text.as_bytes()]);
                    Ok(seismic_native_target::Formed::new(
                        key,
                        vec![std::sync::Arc::new(seismic_native_target::Compile::new(
                            source.entries.clone(),
                            loaded,
                        ))],
                    ))
                })
                .collect()
        }
    }

    std::sync::Arc::new(Stored(CpuToolchain::new(device)))
}
