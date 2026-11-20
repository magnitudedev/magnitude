//! Build-script source checking, checked-bundle emission, and typed Rust
//! binding generation (spec §14.1).
//!
//! A consumer's `build.rs` names its `.seismic` sources and a module name.
//! Generation parses and checks (with the standard library unless opted
//! out), emits `<module>.seismicbundle` (the checked module and its sources)
//! and `<module>.rs` into `OUT_DIR`, and records source/compiler hashes for
//! cache identity. Generated Rust
//! embeds no target executable: target compilation happens in `for_device`.
//!
//! W9-A owns generation internals. The surface below and the generated
//! code contract are frozen.
//!
//! # Generated code contract
//!
//! `<module>.rs` is `include!`d by the consumer inside `pub mod <module>`:
//!
//! ```text
//! static __BUNDLE: &[u8] = include_bytes!("<OUT_DIR>/<module>.seismicbundle");
//! static __MODULE: seismic::generated::OnceLock<Result<seismic::generated::Module, seismic::CheckedBundleError>> = ...;
//! pub fn module() -> Result<&'static seismic::generated::Module, seismic::CheckedBundleError>;
//! pub const IDENTITY: &str = "<hex digest of sources + compiler semantic version>";
//! pub mod constants {
//!     pub const <NAME>: u64 = <value>; ..                    // every `const NAME = INTEGER`
//!     pub const fn <NAME>(<P>: u64, ..) -> u64; ..           // every `const NAME(P, ..) = EXPR`
//! }
//!
//! pub mod <entry> {                       // one per exported entry (every portable family)
//!     pub struct Args<'a> {               // source parameter order, source names
//!         pub <shared or mutable tensor>: &'a seismic::Tensor,
//!         pub <owned tensor>: seismic::Tensor,
//!         pub <f32|i32|u32|bool scalar>: f32|i32|u32|bool,
//!         pub <index>: u64,
//!         pub <range>: (u64, u64),
//!     }
//!     pub struct Results {                // result leaves by ordinal path
//!         pub value: seismic::Tensor,     // single non-tuple result
//!         pub r0: ..., pub r1_2: ...,     // tuple leaves: `r<path joined by _>`
//!     }
//!     pub struct OutputArgs<'a> {         // tensor result leaves only, in checked order
//!         pub value: &'a mut seismic::Tensor,
//!     }
//!     pub struct Elements { pub <ELEM>: seismic::Element, .. }   // polymorphic entries only
//!     pub struct Entry;                   // impl seismic::Entry
//!     impl seismic::Untuned for Entry {}  // no native implementation declares a tuning parameter
//!     pub fn for_device(charges: &seismic::Charges, options: seismic::PreparationOptions)
//!         -> Result<seismic::Kernel<Entry>, seismic::LoadError>;                 // monomorphic
//!     pub fn for_device_with(charges: &seismic::Charges, options: seismic::PreparationOptions, elements: Elements)
//!         -> Result<seismic::Kernel<Entry>, seismic::LoadError>;                 // polymorphic
//!     pub fn native_for_device(device: &seismic::Device)
//!         -> Result<seismic::NativeKernel<Entry>, seismic::LoadError>;           // when declared
//! }
//! ```
//!
//! Scalar results are `f32|i32|u32|bool|u64|(u64,u64)` by kind. Consumers
//! never see schema internals.

use seismic_lang::checked::SourceError;
use seismic_lang::checked::{
    EntryInfo, NativeImplementation, ParameterSummary, ParameterSummaryKind,
    ResultSummaryKind, SourceSet, TensorAccess,
};
pub use seismic_lang::registry::BackendName;
use seismic_lang::types::DType;
use sha2::{Digest, Sha256};
use std::path::PathBuf;

#[derive(Debug)]
pub enum BuildError {
    Source(SourceError),
    Io(std::io::Error),
    /// `OUT_DIR` or a source path is missing.
    Environment(String),
    /// A direct Metal implementation references a symbol outside the ABI
    /// generated for its checked entry.
    NativeAbi {
        entry: String,
        path: PathBuf,
        line: usize,
        column: usize,
        symbol: String,
    },
    /// Two declarations of the entry would define the same native ABI
    /// symbol (a parameter named `words` and the words buffer, a tensor named
    /// `param_1` and parameter 1), so no asset could address both: the
    /// rendered source's later definition replaces the earlier.
    NativeAbiCollision {
        entry: String,
        symbol: String,
        first: String,
        second: String,
    },
    /// A native asset (or an included file) includes something other than a
    /// library file of its backend inside the build's source roots.
    NativeInclude(seismic_lang::source::NativeIncludeError),
    /// A Metal implementation binds more buffers than Metal's argument
    /// table holds.
    MetalBufferSlots {
        entry: String,
        /// Tensor parameters + tensor results + scratch + words + scalar slots.
        slots: usize,
        limit: usize,
    },
    /// A native asset (or an included file) uses a reserved construct: in
    /// every dialect, a literal buffer index, `Cell<`, or a handle built over
    /// memory; in Vulkan, a construct the generated prefix and suffix own
    /// (`#version`, `#extension`, the workgroup size, push constants, a
    /// `shared` declaration, `main`, an unordered float subgroup sum) or a
    /// handle's field.
    NativeReserved {
        entry: String,
        path: PathBuf,
        line: usize,
        column: usize,
        construct: String,
    },
    /// An authored native file crosses the library boundary
    /// (`boundary`): it names a library internal or a library-visibility
    /// ABI symbol, opens or defines into the library, or pastes an ABI name.
    LibraryBoundary {
        path: PathBuf,
        line: usize,
        column: usize,
        rule: String,
    },
    /// A launch names a kernel function the asset does not define.
    NativeKernelMissing {
        entry: String,
        kernel: String,
    },
    /// A CPU library file or CPU native signature the generated Rust cannot
    /// express.
    CpuNative {
        path: PathBuf,
        reason: String,
    },
    /// Every native source violation of the build, each one of the native
    /// variants above, in the order the sources are checked: one build
    /// names them all.
    Natives(Vec<BuildError>),
}

impl std::fmt::Display for BuildError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Source(e) => write!(f, "{e}"),
            Self::Io(e) => write!(f, "io: {e}"),
            Self::Environment(s) => write!(f, "build environment: {s}"),
            Self::NativeAbi {
                entry,
                path,
                line,
                column,
                symbol,
            } => write!(
                f,
                "native ABI for `{entry}`: {}:{line}:{column}: `{symbol}` is not generated for this entry",
                path.display()
            ),
            Self::NativeAbiCollision {
                entry,
                symbol,
                first,
                second,
            } => write!(
                f,
                "native ABI for `{entry}`: `{symbol}` would name both {first} and {second}; rename one"
            ),
            Self::NativeInclude(e) => write!(f, "{e}"),
            Self::MetalBufferSlots {
                entry,
                slots,
                limit,
            } => write!(
                f,
                "native Metal implementation of `{entry}` binds {slots} buffers; Metal admits {limit}"
            ),
            Self::NativeReserved {
                entry,
                path,
                line,
                column,
                construct,
            } => write!(
                f,
                "native implementation of `{entry}`: {}:{line}:{column}: {construct}",
                path.display()
            ),
            Self::LibraryBoundary {
                path,
                line,
                column,
                rule,
            } => write!(f, "library boundary: {}:{line}:{column}: {rule}", path.display()),
            Self::NativeKernelMissing { entry, kernel } => write!(
                f,
                "native Vulkan implementation of `{entry}` launches `{kernel}`, but its source defines no `void {kernel}()`"
            ),
            Self::CpuNative { path, reason } => {
                write!(f, "CPU native source {}: {reason}", path.display())
            }
            Self::Natives(violations) => {
                write!(f, "{} native source violations:", violations.len())?;
                for violation in violations {
                    write!(f, "\n  {violation}")?;
                }
                Ok(())
            }
        }
    }
}
impl std::error::Error for BuildError {}

/// One module build.
#[derive(Debug)]
pub struct Build {
    module: String,
    sources: Vec<PathBuf>,
    include_std: bool,
    out_dir: Option<PathBuf>,
}

impl Build {
    /// `module` is the Rust module name generated bindings live under.
    pub fn new(module: &str) -> Self {
        Self {
            module: module.to_owned(),
            sources: Vec::new(),
            include_std: true,
            out_dir: None,
        }
    }
    /// A `.seismic` file or a directory searched recursively.
    pub fn source(mut self, path: impl Into<PathBuf>) -> Self {
        self.sources.push(path.into());
        self
    }
    /// Whether the standard library sources are linked into the module.
    pub fn std(mut self, include: bool) -> Self {
        self.include_std = include;
        self
    }
    /// Defaults to `$OUT_DIR`.
    pub fn out_dir(mut self, path: impl Into<PathBuf>) -> Self {
        self.out_dir = Some(path.into());
        self
    }

    /// Checks, emits the bundle and the bindings, and prints
    /// `cargo:rerun-if-changed` for every source.
    pub fn run(self) -> Result<Artifacts, BuildError> {
        internals::run(self)
    }

    pub(crate) fn module(&self) -> &str {
        &self.module
    }
    pub(crate) fn sources(&self) -> &[PathBuf] {
        &self.sources
    }
    pub(crate) fn include_std(&self) -> bool {
        self.include_std
    }
    pub(crate) fn output_directory(&self) -> Option<&PathBuf> {
        self.out_dir.as_ref()
    }
}

/// Where the outputs landed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Artifacts {
    pub bundle: PathBuf,
    pub bindings: PathBuf,
    /// Hex digest of sources plus compiler semantic version.
    pub identity: String,
    /// Every authored native implementation of the module's entries, in
    /// entry order, then backend order. A library can enforce its own
    /// backend coverage policy over these at build time.
    pub natives: Vec<NativeCoverage>,
}

/// One `native <entry> for <backend> from "<asset>"` declaration.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NativeCoverage {
    pub entry: String,
    pub backend: BackendName,
    /// Canonical path of the authored asset.
    pub asset: PathBuf,
}

mod boundary;
mod render_contract;

mod internals {
    use super::*;
    use crate::boundary::{
        violation as boundary_violation, visibility as boundary_visibility, Boundary, Rule, Visibility,
        CONVERSION_FAMILIES, TENSOR_FAMILIES,
    };
    use seismic_lang::checked::{CpuBinding, ElementSummary, NativeElementCoverage, RecordPort};
    use seismic_lang::entry::ResultClass;
    use seismic_lang::registry::{self, RepresentationInfo, SourceDialect};
    use seismic_lang::native_parameter::{
        OwnerReach, element_reach, native_parameter_names, parameter_reach,
    };
    use std::collections::hash_map::Entry;
    use std::collections::{HashMap, HashSet};
    use std::fmt;
    use std::fs;

    /// The native implementations of one entry. Metal and CUDA sources
    /// travel in the checked bundle; a CPU source is compiled into the
    /// generated bindings.
    struct EntryNative<'a> {
        entry: seismic_lang::ids::EntryId,
        cpu: Option<CpuAsset<'a>>,
        /// Whether an implementation of the entry, on any backend, declares
        /// tuning parameters.
        tuned: bool,
    }

    /// A CPU native implementation and its authored source.
    struct CpuAsset<'a> {
        definition: &'a NativeImplementation,
        /// Canonical path of the authored asset, which the bindings include.
        path: PathBuf,
        text: &'a str,
    }

    pub(super) fn run(build: Build) -> Result<Artifacts, BuildError> {
        validate_identifier(build.module())?;
        let output = match build.output_directory() {
            Some(path) => path.clone(),
            None => std::env::var_os("OUT_DIR")
                .map(PathBuf::from)
                .ok_or_else(|| BuildError::Environment("OUT_DIR is not set".to_owned()))?,
        };
        fs::create_dir_all(&output).map_err(BuildError::Io)?;
        let output = output.canonicalize().map_err(BuildError::Io)?;

        let prelude = if build.include_std() {
            seismic_std::sources()
        } else {
            SourceSet::default()
        };
        let loaded = seismic_lang::source::load(build.sources(), prelude).map_err(|e| match e {
            seismic_lang::source::LoadError::Io(e) => BuildError::Io(e),
            seismic_lang::source::LoadError::Source(e) => BuildError::Source(e),
            seismic_lang::source::LoadError::Invalid(e) => BuildError::Environment(e),
            seismic_lang::source::LoadError::NativeInclude(e) => BuildError::NativeInclude(e),
        })?;
        for path in loaded.dependencies() {
            println!("cargo:rerun-if-changed={}", path.display());
        }
        for path in build.sources() {
            println!("cargo:rerun-if-changed={}", path.display());
        }
        // No directory watch is needed: an include resolves from its directive
        // and the including file alone, so a new file cannot change a successful
        // build, and a missing include fails the build, which Cargo reruns.
        let checked = &loaded.module;
        // Captured Metal/CUDA/Vulkan assets carry their includes inlined, so
        // the bundle digest below covers every included file.
        let encoded = seismic_lang::bundle::encode_checked_bundle(checked);
        let native_assets = resolve_native_assets(checked, &loaded.assets, &output)?;
        let cpu_assets = native_assets
            .iter()
            .filter_map(|native| native.cpu.as_ref().map(|cpu| cpu.path.as_path()))
            .collect::<Vec<_>>();
        let cpu_library = if cpu_assets.is_empty() {
            None
        } else {
            let roots =
                seismic_lang::source::source_roots(build.sources()).map_err(
                    |error| match error {
                        seismic_lang::source::LoadError::Io(error) => BuildError::Io(error),
                        other => BuildError::Environment(format!("{other:?}")),
                    },
                )?;
            Some(cpu_library(&roots, &cpu_assets)?)
        };
        // The checked bundle contains the canonical sources, the checked
        // module this build's checker produced (decoded at run time without
        // checking again), bundle format, checker semantic version, registry
        // revision, and semantic hash. Addressing the emitted bundle therefore
        // cannot accidentally reuse generated bindings across a change in any
        // of those inputs.
        let mut identity_hasher = Sha256::new();
        identity_hasher.update(&encoded);
        let bundle_digest: [u8; 32] = identity_hasher.finalize().into();
        let identity = hex(&bundle_digest);
        let bundle = output.join(format!("{}.seismicbundle", build.module()));
        let bindings = output.join(format!("{}.rs", build.module()));
        fs::write(&bundle, encoded).map_err(BuildError::Io)?;
        fs::write(
            &bindings,
            render(
                checked,
                build.module(),
                &identity,
                &native_assets,
                cpu_library.as_ref(),
            )?,
        )
        .map_err(BuildError::Io)?;
        Ok(Artifacts {
            bundle,
            bindings,
            identity,
            natives: native_coverage(checked, &loaded.assets),
        })
    }

    fn native_coverage(
        module: &seismic_lang::checked::CheckedModule,
        captured: &[seismic_lang::source::CapturedAsset],
    ) -> Vec<NativeCoverage> {
        module
            .entries()
            .iter()
            .flat_map(|entry| {
                BackendName::ALL.into_iter().filter_map(move |backend| {
                    captured
                        .iter()
                        .find(|asset| asset.entry == entry.id && asset.backend == backend)
                        .map(|asset| NativeCoverage {
                            entry: entry.name.clone(),
                            backend,
                            asset: asset.asset.path.clone(),
                        })
                })
            })
            .collect()
    }

    fn resolve_native_assets<'a>(
        module: &'a seismic_lang::checked::CheckedModule,
        captured: &[seismic_lang::source::CapturedAsset],
        output: &std::path::Path,
    ) -> Result<Vec<EntryNative<'a>>, BuildError> {
        use seismic_lang::registry::BackendName;
        let mut natives = Vec::new();
        let mut boundary = Boundary::default();
        let mut violations = Vec::new();
        for entry in module.entries() {
            let mut any = false;
            let mut tuned = false;
            let mut cpu = None;
            for backend in BackendName::ALL {
                let Some(definition) = module.native_implementation(entry.id, backend) else {
                    continue;
                };
                any = true;
                tuned |= definition.has_tuning_parameters();
                let source = module
                    .native_asset(entry.id, backend)
                    .expect("shared loader captures native assets")
                    .text;
                // CPU assets are Rust, compiled into the embedding crate from
                // their authored path, so compiler diagnostics and tools point
                // at the authored file.
                let Some(dialect) = backend.source_dialect() else {
                    let files = captured
                        .iter()
                        .find(|asset| asset.entry == entry.id && asset.backend == backend)
                        .expect("shared loader captures every native asset");
                    cpu = Some(CpuAsset {
                        definition,
                        path: files.asset.path.clone(),
                        text: source,
                    });
                    continue;
                };
                let extension = match dialect {
                    SourceDialect::Metal => "metal",
                    SourceDialect::Cuda => "cu",
                    SourceDialect::Vulkan => "comp",
                };
                fs::write(output.join(format!("{}.{extension}", entry.name)), source)
                    .map_err(BuildError::Io)?;
                let files = captured
                    .iter()
                    .find(|asset| asset.entry == entry.id && asset.backend == backend)
                    .expect("shared loader captures every native asset");
                // Included files are validated against the including entry's
                // ABI, exactly like the asset itself. The ABI is independent
                // of the file, so build its symbol set once for this entry.
                let allowed = match native_abi_symbols(entry, definition, dialect, module.constants(), module.constant_functions()) {
                    Ok(allowed) => allowed,
                    Err(error) => {
                        violations.push(error);
                        continue;
                    }
                };
                for file in std::iter::once(&files.asset).chain(&files.includes) {
                    validate_native_abi(entry, &allowed, &file.path, &file.text, &mut violations);
                    boundary.check(&file.path, &file.text, &mut violations);
                    validate_reserved(entry, dialect, &file.path, &file.text, &mut violations);
                }
                match dialect {
                    SourceDialect::Metal => validate_metal_buffer_slots(entry, definition, &mut violations),
                    SourceDialect::Vulkan => validate_vulkan_kernels(entry, definition, source, &mut violations),
                    SourceDialect::Cuda => {}
                }
            }
            if any {
                natives.push(EntryNative {
                    entry: entry.id,
                    cpu,
                    tuned,
                });
            }
        }
        // A library file every including entry checks breaks a reserved or
        // boundary rule once, wherever it is included.
        let mut reported = HashSet::new();
        violations.retain(|violation| match violation {
            BuildError::NativeReserved { path, line, column, construct, .. } => {
                reported.insert((path.clone(), *line, *column, construct.clone()))
            }
            BuildError::LibraryBoundary { path, line, column, rule } => {
                reported.insert((path.clone(), *line, *column, rule.clone()))
            }
            _ => true,
        });
        if !violations.is_empty() {
            return Err(BuildError::Natives(violations));
        }
        Ok(natives)
    }

    /// Every `SEISMIC_*` name `source` spells is generated for the entry, and
    /// of surface visibility.
    fn validate_native_abi(
        entry: &EntryInfo,
        allowed: &HashMap<String, Visibility>,
        path: &std::path::Path,
        source: &str,
        violations: &mut Vec<BuildError>,
    ) {
        let failure = |offset: usize, symbol: &str| {
            let (line, column) = location(source, offset);
            BuildError::NativeAbi {
                entry: entry.name.clone(),
                path: path.to_path_buf(),
                line,
                column,
                symbol: symbol.to_owned(),
            }
        };
        for (symbol, offset) in seismic_identifiers(source) {
            match allowed.get(symbol) {
                None => violations.push(failure(offset, symbol)),
                Some(Visibility::Surface) => {}
                Some(Visibility::Library) => {
                    violations.push(boundary_violation(path, source, offset, Rule::LibrarySymbol(symbol.to_owned())));
                }
            }
        }
    }

    /// Metal's argument table holds 31 buffers. The native ABI binds every
    /// tensor parameter but a host port (which the kernel never accesses),
    /// every tensor result, every scratch buffer, the argument words and the
    /// scalar-result slots; all of them are known from the declaration, so
    /// the limit is checked here rather than at encode.
    fn validate_metal_buffer_slots(
        entry: &EntryInfo,
        definition: &NativeImplementation,
        violations: &mut Vec<BuildError>,
    ) {
        let tensors = entry
            .parameters
            .iter()
            .filter(|parameter| match &parameter.kind {
                ParameterSummaryKind::Tensor { access, .. } => !matches!(access, TensorAccess::Host { .. }),
                _ => false,
            })
            .count()
            + entry
                .results
                .iter()
                .filter(|result| matches!(result.kind, ResultSummaryKind::Tensor { .. }))
                .count();
        let slots = tensors + definition.scratch.len() + 2;
        if slots > METAL_BUFFER_SLOTS {
            violations.push(BuildError::MetalBufferSlots {
                entry: entry.name.clone(),
                slots,
                limit: METAL_BUFFER_SLOTS,
            });
        }
    }

    /// Metal's per-stage buffer argument table size.
    const METAL_BUFFER_SLOTS: usize = 31;

    /// The typed group-memory views of the Vulkan prefix
    /// (`seismic_runtime`'s `VULKAN_SHARED_VIEWS`), as `SEISMIC_SHARED_<view>`.
    const VULKAN_SHARED_VIEWS: [&str; 8] =
        ["F32", "F16", "BF16", "U8", "U16", "U32", "I32", "UVEC4"];

    /// Identifiers a Vulkan asset may not use, with why: the prefix and
    /// suffix own the workgroup size, push constants, group memory and
    /// `main`, and the device library owns the reconvergence mode.
    const VULKAN_RESERVED_IDENTIFIERS: [(&str, &str); 10] = [
        (
            "SEISMIC_HAS_MAXIMAL_RECONVERGENCE",
            "the device library's `seismic_strided*` loops and `seismic_rejoin` own the reconvergence mode",
        ),
        ("local_size_x", "the workgroup size is generated"),
        ("local_size_y", "the workgroup size is generated"),
        ("local_size_z", "the workgroup size is generated"),
        ("local_size_x_id", "the workgroup size is generated"),
        ("local_size_y_id", "the workgroup size is generated"),
        ("local_size_z_id", "the workgroup size is generated"),
        (
            "push_constant",
            "the push constant is the generated argument block address",
        ),
        (
            "shared",
            "group memory is the generated `seismic_shared_*` views of `shared_bytes`",
        ),
        ("main", "`main` is generated and calls the launch's kernel"),
    ];

    /// Why a Vulkan asset may not use the subgroup built-in `identifier`, if
    /// it may not. Kernels are written for the 32-lane logical subgroup,
    /// which on 64-lane hardware is half the hardware subgroup; the device
    /// library maps it to the hardware (`vulkan_prelude.glsl`). A built-in
    /// that names hardware lanes or subgroups, or spans the hardware
    /// subgroup, breaks that mapping on 64-lane hardware. Xor shuffles by a
    /// mask below 32, up and down shuffles whose edge lanes are discarded,
    /// and subgroup barriers stay within a logical subgroup on both widths.
    fn vulkan_subgroup_rule(identifier: &str) -> Option<&'static str> {
        if identifier.starts_with("gl_Subgroup") || identifier == "gl_NumSubgroups" {
            return Some(
                "it names hardware lanes or subgroups; use `SEISMIC_LANE`, `SEISMIC_SUBGROUP`, \
                 `SEISMIC_SUBGROUPS`, `SEISMIC_THREAD` or `SEISMIC_HARDWARE_SUBGROUP`",
            );
        }
        // GLSL's subgroup built-in functions, `subgroup` then a capital;
        // kernels' own names are snake case.
        let operation = identifier.strip_prefix("subgroup")?;
        if !operation.starts_with(|c: char| c.is_ascii_uppercase()) {
            return None;
        }
        match operation {
            "ShuffleXor" | "ShuffleUp" | "ShuffleDown" | "Barrier" | "MemoryBarrier"
            | "MemoryBarrierBuffer" | "MemoryBarrierShared" | "MemoryBarrierImage" => None,
            "Shuffle" => Some("its lane is a hardware lane; use `seismic_shuffle`"),
            _ if operation.starts_with("Quad") => {
                Some("quad operations are xor shuffles by a mask below 4 or `seismic_shuffle`")
            }
            _ => Some(
                "it spans the hardware subgroup, both logical subgroups on 64-lane hardware; \
                 use `seismic_subgroup_*`, `seismic_ballot`, `seismic_uniform_*`, `seismic_steer` or `seismic_hardware_first`",
            ),
        }
    }

    /// Why `identifier` at `offset` in a Vulkan asset breaks the steering form,
    /// if it does. The hardware subgroup is steered only by `seismic_steer`
    /// (`vulkan_prelude.glsl`), as the whole condition of an `if`, `while` or
    /// `for`, so the logical subgroup's predicate it binds is the literal true
    /// in a 32-lane module and a loop it steers exits at its header: steering
    /// written otherwise (the 32-lane join called directly, an exit in a
    /// loop's body) keeps the predicate and the exit in the module the 32-lane
    /// driver compiles, which it does not fold away.
    fn vulkan_steering_rule(source: &str, identifier: &str, offset: usize) -> Option<String> {
        use seismic_lang::native_source::{tokens, TokenKind};
        if identifier == "seismic_hardware_any" {
            return Some("`seismic_hardware_any` is `seismic_steer`'s; steer by `seismic_steer`".to_owned());
        }
        if identifier != "seismic_steer" {
            return None;
        }
        let whole = "`seismic_steer` is the whole condition of an `if`, `while` or `for`";
        let toks: Vec<_> = tokens(source).collect();
        let Some(pos) = toks.iter().position(|t| t.offset == offset) else {
            return Some(whole.to_owned());
        };
        // The condition's opening parenthesis: the innermost one open at `pos`.
        let mut depth = 0usize;
        let open = toks[..pos].iter().enumerate().rev().find_map(|(i, t)| {
            if t.kind == TokenKind::Punctuator {
                match t.text {
                    ")" => {
                        depth += 1;
                        None
                    }
                    "(" if depth == 0 => Some(i),
                    "(" => {
                        depth -= 1;
                        None
                    }
                    _ => None,
                }
            } else {
                None
            }
        });
        let Some(open) = open else {
            return Some(whole.to_owned());
        };
        let keyword = open
            .checked_sub(1)
            .map(|i| toks[i])
            .filter(|t| t.kind == TokenKind::Identifier)
            .map(|t| t.text)
            .unwrap_or("");
        let inside = &toks[open + 1..pos];
        let leading = match keyword {
            "if" | "while" => {
                inside.is_empty()
                    || (inside.len() == 1
                        && inside[0].kind == TokenKind::Punctuator
                        && inside[0].text == "!")
            }
            // The condition: after the header's one `;` at its own depth.
            "for" => {
                let mut semi_count = 0usize;
                let mut d = 0usize;
                for t in inside {
                    if t.kind == TokenKind::Punctuator {
                        match t.text {
                            "(" => d += 1,
                            ")" => d = d.saturating_sub(1),
                            ";" if d == 0 => semi_count += 1,
                            _ => {}
                        }
                    }
                }
                semi_count == 1
                    && inside
                        .last()
                        .is_some_and(|t| t.kind == TokenKind::Punctuator && t.text == ";")
            }
            _ => false,
        };
        // The call's closing parenthesis, and what follows it.
        if !toks
            .get(pos + 1)
            .is_some_and(|t| t.kind == TokenKind::Punctuator && t.text == "(")
        {
            return Some(whole.to_owned());
        }
        let mut depth = 0usize;
        let close = toks[pos + 1..].iter().enumerate().find_map(|(rel, t)| {
            if t.kind == TokenKind::Punctuator {
                match t.text {
                    "(" => {
                        depth += 1;
                        None
                    }
                    ")" if depth == 1 => Some(pos + 1 + rel),
                    ")" => {
                        depth -= 1;
                        None
                    }
                    _ => None,
                }
            } else {
                None
            }
        });
        let Some(close) = close else {
            return Some(whole.to_owned());
        };
        let trailing = match keyword {
            "for" => toks
                .get(close + 1)
                .is_some_and(|t| t.kind == TokenKind::Punctuator && t.text == ";"),
            _ => toks
                .get(close + 1)
                .is_some_and(|t| t.kind == TokenKind::Punctuator && t.text == ")"),
        };
        if !leading || !trailing {
            return Some(whole.to_owned());
        }
        if keyword == "if" {
            let body = toks[close + 2..]
                .iter()
                .find(|t| !(t.kind == TokenKind::Punctuator && t.text == "{"));
            if body.is_some_and(|t| {
                t.kind == TokenKind::Identifier && matches!(t.text, "break" | "continue")
            }) {
                return Some(
                    "a loop steered by `seismic_steer` exits at its header, `for (...; seismic_steer(live, p); ...)` \
                     or `while (seismic_steer(live, p))`, not by a `break` or `continue` in its body"
                        .to_owned(),
                );
            }
        }
        None
    }

    /// Directives the prefix owns.
    const VULKAN_RESERVED_DIRECTIVES: [&str; 2] = ["version", "extension"];

    pub(crate) fn location(source: &str, offset: usize) -> (usize, usize) {
        let prefix = &source[..offset];
        let line = prefix.bytes().filter(|byte| *byte == b'\n').count() + 1;
        let column = prefix
            .rsplit_once('\n')
            .map_or(prefix.len() + 1, |(_, tail)| tail.len() + 1);
        (line, column)
    }

    /// Refuses, in every dialect's authored sources, the constructs that would reach a floating
    /// result or mutated tensor other than through its owner handle, and in Vulkan the constructs
    /// the generated prefix and suffix own and the subgroup built-ins that break the logical
    /// subgroup ([`vulkan_subgroup_rule`]). Each refusal names its rule.
    fn validate_reserved(
        entry: &EntryInfo,
        dialect: SourceDialect,
        path: &std::path::Path,
        source: &str,
        violations: &mut Vec<BuildError>,
    ) {
        let reserved = |offset: usize, construct: String| {
            let (line, column) = location(source, offset);
            BuildError::NativeReserved {
                entry: entry.name.clone(),
                path: path.to_path_buf(),
                line,
                column,
                construct,
            }
        };
        if dialect == SourceDialect::Vulkan {
            use seismic_lang::native_source::{tokens, TokenKind};
            let mut iter = tokens(source).peekable();
            while let Some(tok) = iter.next() {
                if tok.first && tok.kind == TokenKind::Punctuator && tok.text == "#" {
                    if let Some(next) = iter.peek() {
                        if !next.first
                            && next.kind == TokenKind::Identifier
                            && VULKAN_RESERVED_DIRECTIVES.contains(&next.text)
                        {
                            let name = next.text;
                            violations.push(reserved(
                                tok.offset,
                                format!("prefix-owned directive: `#{name}` is generated by the Vulkan prefix"),
                            ));
                        }
                    }
                }
            }
        }
        for (identifier, offset) in identifiers(source) {
            if let Some(rule) = owner_rule(dialect, source, identifier, offset) {
                violations.push(reserved(offset, rule));
            }
            if dialect != SourceDialect::Vulkan {
                continue;
            }
            if let Some((_, why)) = VULKAN_RESERVED_IDENTIFIERS
                .iter()
                .find(|(name, _)| *name == identifier)
            {
                violations.push(reserved(
                    offset,
                    format!("reserved identifier: `{identifier}` is reserved: {why}"),
                ));
            }
            if let Some(why) = vulkan_subgroup_rule(identifier) {
                violations.push(reserved(
                    offset,
                    format!("logical subgroup: `{identifier}` is refused: {why}"),
                ));
            }
            if let Some(why) = vulkan_steering_rule(source, identifier, offset) {
                violations.push(reserved(offset, format!("hardware steering: {why}")));
            }
            if matches!(identifier, "seismic_base" | "seismic_kind" | "seismic_table") {
                violations.push(reserved(
                    offset,
                    format!("handle field: `{identifier}` is a field of the element owner's handle"),
                ));
            }
        }
    }

    /// The owner rule `identifier` at `offset` breaks, if any:
    /// - a literal buffer index (`SEISMIC_PTR(<n>`, `[[buffer(<n>`, `seismic_buffer_<n>`), which
    ///   reaches a buffer around the ABI macros that define which tensors have one;
    /// - `Cell<`: a Metal handle's typed cells come only from the generated parameter declaration
    ///   and `out_over_scratch`;
    /// - a handle construction where the dialect builds a handle from an address
    ///   ([`handle_built_from_memory`]): only the ABI macros and the library build one.
    fn owner_rule(
        dialect: SourceDialect,
        source: &str,
        identifier: &str,
        offset: usize,
    ) -> Option<String> {
        let after = &source[offset + identifier.len()..];
        let literal_argument = |after: &str| {
            after
                .trim_start()
                .strip_prefix('(')
                .is_some_and(|argument| argument.trim_start().starts_with(|c: char| c.is_ascii_digit()))
        };
        let literal = match identifier {
            "SEISMIC_PTR" => literal_argument(after),
            "buffer" => source[..offset].trim_end().ends_with("[[") && literal_argument(after),
            _ => identifier
                .strip_prefix("seismic_buffer_")
                .is_some_and(|index| index.starts_with(|c: char| c.is_ascii_digit())),
        };
        if literal {
            return Some(format!(
                "literal buffer index: `{identifier}` with a number reaches a buffer by position; \
                 a tensor is reached by its ABI macro, a floating result or mutated tensor only by \
                 its owner handle"
            ));
        }
        if identifier == "Cell" && after.trim_start().starts_with('<') {
            return Some(
                "typed cells: `Cell<` is the owner's; a handle's cells come only from the generated \
                 parameter declaration and `out_over_scratch`"
                    .to_owned(),
            );
        }
        if matches!(identifier, "abi_out" | "abi_slab_out") {
            return Some(format!(
                "ABI constructor: `{identifier}` is the ABI prefix's; a kernel reaches its handle \
                 through `SEISMIC_RESULT_<n>_OUT` or `SEISMIC_OUT_<NAME>`, and scratch through \
                 `out_over_scratch`"
            ));
        }
        if matches!(identifier, "Out" | "SlabOut" | "Stored")
            && handle_built_from_memory(dialect)
            && handle_construction(after)
        {
            return Some(format!(
                "handle construction: `{identifier}` is built over memory only by the ABI macros \
                 (`SEISMIC_RESULT_<n>_OUT`, `SEISMIC_OUT_<NAME>`) and `out_over_scratch`"
            ));
        }
        None
    }

    /// Whether the dialect builds an owner handle from an address. CUDA (`Out(u8 *)`) and Vulkan
    /// (`Out(uint64_t, kind)`) do, so a kernel may not construct one. A Metal handle is built from
    /// `device Cell<E> *` cells of its own element, which a kernel holds only from the generated
    /// parameter declaration or `out_over_scratch`, so constructing one over them is the kernel's.
    fn handle_built_from_memory(dialect: SourceDialect) -> bool {
        match dialect {
            SourceDialect::Metal => false,
            SourceDialect::Cuda | SourceDialect::Vulkan => true,
        }
    }

    /// Whether `rest`, the source after `Out` or `SlabOut`, constructs a handle: a call or brace
    /// initializer, with or without template arguments, or a declaration initialized directly
    /// (`Out<E> name(..);`, `name{..}`, `name = {..}`). A function returning a handle (`) {`), a
    /// reference or pointer to one, and the null handle (`(nullptr)`) are not.
    fn handle_construction(rest: &str) -> bool {
        let mut rest = rest.trim_start();
        if rest.starts_with('<') {
            let Some(end) = matching(rest) else {
                return false;
            };
            rest = rest[end..].trim_start();
        }
        if rest.starts_with(['(', '{']) {
            return !null_handle(rest);
        }
        let name = rest
            .bytes()
            .take_while(|byte| *byte == b'_' || byte.is_ascii_alphanumeric())
            .count();
        if name == 0 || rest.starts_with(|c: char| c.is_ascii_digit()) {
            return false;
        }
        let initializer = rest[name..].trim_start();
        if initializer.starts_with('{') {
            return !null_handle(initializer);
        }
        if let Some(value) = initializer.strip_prefix('=') {
            return value.trim_start().starts_with('{');
        }
        initializer.starts_with('(')
            && !null_handle(initializer)
            && matching(initializer)
                .is_some_and(|end| initializer[end..].trim_start().starts_with([';', ',']))
    }

    /// The end of the bracketed text `text` opens (`<`, `(` or `{`), past its closing bracket.
    /// Template arguments end at a `;`, `{` or `}` first: then `<` was not one.
    fn matching(text: &str) -> Option<usize> {
        let (open, close) = match text.as_bytes().first()? {
            b'<' => (b'<', b'>'),
            b'(' => (b'(', b')'),
            b'{' => (b'{', b'}'),
            _ => return None,
        };
        let mut depth = 0usize;
        for (index, byte) in text.bytes().enumerate() {
            if byte == open {
                depth += 1;
            } else if byte == close {
                depth -= 1;
                if depth == 0 {
                    return Some(index + 1);
                }
            } else if open == b'<' && matches!(byte, b';' | b'{' | b'}') {
                return None;
            }
        }
        None
    }

    /// Whether the bracketed `text` holds only `nullptr`: the null handle, over no memory.
    fn null_handle(text: &str) -> bool {
        matching(text).is_some_and(|end| text[1..end - 1].trim() == "nullptr")
    }

    /// Every launch's kernel is a `void <kernel>()` function of the expanded
    /// source (lexically, like the symbol checks).
    fn validate_vulkan_kernels(
        entry: &EntryInfo,
        definition: &NativeImplementation,
        source: &str,
        violations: &mut Vec<BuildError>,
    ) {
        let tokens = identifiers(source);
        for launch in &definition.launches {
            let defined = tokens.windows(2).any(|pair| {
                let [(ty, _), (name, at)] = pair else {
                    unreachable!("windows of two")
                };
                *ty == "void"
                    && *name == launch.kernel
                    && source[at + name.len()..]
                        .trim_start()
                        .strip_prefix('(')
                        .is_some_and(|rest| rest.trim_start().starts_with(')'))
            });
            if !defined {
                violations.push(BuildError::NativeKernelMissing {
                    entry: entry.name.clone(),
                    kernel: launch.kernel.clone(),
                });
            }
        }
    }

    fn native_abi_symbols(
        entry: &EntryInfo,
        definition: &NativeImplementation,
        dialect: SourceDialect,
        constants: &[(String, u64)],
        functions: &[seismic_lang::checked::ConstantFunction],
    ) -> Result<HashMap<String, Visibility>, BuildError> {
        let mut abi = AbiSymbols {
            entry: &entry.name,
            symbols: HashMap::new(),
        };
        // Every module constant and constant function, which the rendered
        // source defines when named.
        for name in constants.iter().map(|(name, _)| name.as_str()).chain(functions.iter().map(|function| function.name())) {
            abi.define(
                AbiMeaning::Constant(name.to_owned()),
                Symbols::surface([format!("SEISMIC_CONST_{}", native_macro(name))]),
            )?;
        }
        if !definition.launch_scoped() {
            for parameter in &definition.params {
                abi.define(
                    AbiMeaning::Tune(parameter.name.clone()),
                    Symbols::surface([format!("SEISMIC_TUNE_{}", native_macro(&parameter.name))]),
                )?;
            }
        } else {
            // A runtime parameter's macro is defined in the source of each
            // launch that reads it, once: the same name in several launches
            // is one meaning.
            let runtime = definition
                .params
                .iter()
                .chain(definition.launches.iter().flat_map(|launch| &launch.params))
                .filter(|parameter| !parameter.code);
            for parameter in runtime {
                abi.define(
                    AbiMeaning::Runtime(parameter.name.clone()),
                    Symbols::surface([format!("SEISMIC_RUNTIME_{}", native_macro(&parameter.name))]),
                )?;
            }
            for launch in &definition.launches {
                abi.define(
                    AbiMeaning::Forming(launch.kernel.clone()),
                    Symbols::surface([format!("SEISMIC_FORMING_{}", native_macro(&launch.kernel))]),
                )?;
            }
        }
        // A resident kernel's name, under which its grid barrier forms, and
        // the barrier itself (the runtime's `metal_resident.h`), in the
        // prefix of a source with a resident launch.
        let mut resident = false;
        for launch in definition.launches.iter().filter(|launch| launch.resident.is_some()) {
            resident = true;
            abi.define(
                AbiMeaning::Resident(launch.kernel.clone()),
                Symbols::surface([format!("SEISMIC_RESIDENT_{}", native_macro(&launch.kernel))]),
            )?;
        }
        if resident {
            abi.define(
                AbiMeaning::Abi,
                Symbols::surface(["SEISMIC_GRID_BARRIER".to_owned(), "SEISMIC_RESIDENT_GIVE_UP_WORD".to_owned()]),
            )?;
        }
        for scratch in &definition.scratch {
            abi.define(
                AbiMeaning::Scratch(scratch.name.clone()),
                Symbols::surface([format!("SEISMIC_BUFFER_SCRATCH_{}", native_macro(&scratch.name))]),
            )?;
        }
        // The ABI's own names, fixed by the dialect; and the backend's
        // fact every dialect's prefix states.
        let mut symbols = Symbols::default();
        symbols.insert("SEISMIC_BUFFER_SCALAR_RESULTS".to_owned());
        symbols.insert("SEISMIC_PARKS_IN_KERNELS".to_owned());
        // A `segmented` projection's grouping, stated in every source (no
        // segments where the entry projects none); a library pastes a
        // segment's number onto the stem.
        symbols.insert("SEISMIC_PROJECTION_SEGMENTS".to_owned());
        symbols.insert("SEISMIC_SEGMENT_WEIGHTS_".to_owned());
        symbols.insert("SEISMIC_SEGMENT_ORDERED_".to_owned());
        for segment in 0..seismic_lang::checked::PROJECTION_SEGMENTS {
            symbols.insert(format!("SEISMIC_SEGMENT_WEIGHTS_{segment}"));
            symbols.insert(format!("SEISMIC_SEGMENT_ORDERED_{segment}"));
        }
        // The device library's names, surface like the names the ABI fixes:
        // kernels spell them.
        symbols.extend(library_symbols(dialect));
        match dialect {
            SourceDialect::Cuda => {
                symbols.insert("SEISMIC_BUFFER_WORDS".to_owned());
                for symbol in [
                    "SEISMIC_KERNEL_PARAMS",
                    "SEISMIC_PTR",
                    "SEISMIC_PTR_",
                    "SEISMIC_SCALAR_RESULTS",
                    // Device facts.
                    "SEISMIC_MULTIPROCESSORS",
                ] {
                    symbols.insert(symbol.to_owned());
                }
            }
            // Vulkan words live in the argument block, read through
            // `seismic_words`; there is no words buffer.
            SourceDialect::Vulkan => {
                for symbol in [
                    "SEISMIC_PTR",
                    "SEISMIC_READONLY",
                    "SEISMIC_READONLY_",
                    "SEISMIC_SCALAR_RESULTS",
                    "SEISMIC_KERNEL",
                    "SEISMIC_SUBGROUP_LANES",
                    "SEISMIC_HAS_FLOAT16",
                    "SEISMIC_HAS_MATRIX",
                    "SEISMIC_HAS_MATRIX_HARDWARE",
                    "SEISMIC_HAS_WIDE_ACCUMULATORS",
                    "SEISMIC_HAS_MIXED_DOT",
                    "SEISMIC_HAS_F32_ATOMIC_ADD",
                    "SEISMIC_HAS_SHARED_INT64_ATOMICS",
                    "SEISMIC_HAS_MAXIMAL_RECONVERGENCE",
                ] {
                    symbols.insert(symbol.to_owned());
                }
                for view in VULKAN_SHARED_VIEWS {
                    symbols.insert(format!("SEISMIC_SHARED_{view}"));
                }
            }
            SourceDialect::Metal => {
                symbols.insert("SEISMIC_BUFFER_WORDS".to_owned());
                symbols.insert("SEISMIC_HAS_TENSOR_OPS".to_owned());
            }
        }
        abi.define(AbiMeaning::Abi, symbols)?;
        for dimension in &entry.dimensions {
            abi.define(
                AbiMeaning::Dimension(dimension.clone()),
                Symbols::surface([format!("SEISMIC_DIM_{}", native_macro(dimension))]),
            )?;
        }
        // A static extent as a plain integer, for `#if` guards.
        for dimension in &definition.statics {
            abi.define(
                AbiMeaning::Dimension(dimension.clone()),
                Symbols::surface([format!("SEISMIC_STATIC_{}", native_macro(dimension))]),
            )?;
        }
        for element in &entry.element_parameters {
            let mut symbols = Symbols::default();
            add_representation_symbols(
                &mut symbols,
                &format!("SEISMIC_ELEMENT_{}", native_macro(element)),
                registry::representations().iter(),
            );
            abi.define(AbiMeaning::Element(element.clone()), symbols)?;
        }
        // A registered conversion between two bound elements, as its word
        // form as code (the runtime ABI's `render_conversion`): its units and
        // fixed words, its layout's shares, span and store alignments, and
        // CUDA's stage (its row, tile and lanes) and fixed words' loads.
        for from in &entry.element_parameters {
            for to in &entry.element_parameters {
                let mut symbols = Symbols::default();
                let prefix = format!(
                    "SEISMIC_CONVERSION_{}_{}",
                    native_macro(from),
                    native_macro(to)
                );
                for suffix in [
                    "WORDS",
                    "UNIT",
                    "UNIT_COLUMNS",
                    "FIXED",
                    "LEADING_ALIGNMENT",
                    "TRAILING_ALIGNMENT",
                    "CHUNK_COLUMNS",
                    "KBLOCK_BYTES",
                    "KBLOCK_ALIGNMENT",
                ] {
                    symbols.conversion(&prefix, &format!("_{suffix}"));
                }
                if dialect == SourceDialect::Cuda {
                    for suffix in ["STAGE_WORDS", "TILE_ROWS", "LANES", "FIXED_LOADS", "FIXED_LOAD"] {
                        symbols.conversion(&prefix, &format!("_{suffix}"));
                    }
                }
                for role in registry::RowPlaneRole::ALL {
                    let role = native_macro(role.name());
                    for suffix in ["WORD", "BYTES", "ALIGNMENT"] {
                        symbols.conversion(&prefix, &format!("_{role}_{suffix}"));
                    }
                }
                // The code planes' segments of their span.
                for role in registry::RowPlaneRole::CODES {
                    let role = native_macro(role.name());
                    for suffix in [
                        "FIRST",
                        "UNIT_WORDS",
                        "REST_ALIGNMENT",
                        "CHUNK_BYTES",
                        "CHUNK_ALIGNMENT",
                        "FRAGMENT_ALIGNMENT",
                    ] {
                        symbols.conversion(&prefix, &format!("_{role}_{suffix}"));
                    }
                }
                // The bare prefix, a binding argument.
                symbols.conversion(&prefix, "");
                abi.define(AbiMeaning::Conversion(from.clone(), to.clone()), symbols)?;
            }
        }

        let names = native_parameter_names(
            &entry
                .parameters
                .iter()
                .map(|parameter| parameter.name.as_str())
                .collect::<Vec<_>>(),
        );
        for ((ordinal, parameter), names) in entry.parameters.iter().enumerate().zip(&names) {
            let mut symbols = Symbols::default();
            match &parameter.kind {
                // A host port defines nothing: the kernel never accesses it.
                ParameterSummaryKind::Tensor {
                    access: seismic_lang::checked::TensorAccess::Host { .. },
                    ..
                } => {}
                ParameterSummaryKind::Tensor { rank, element, access } => {
                    let mutated = matches!(access, seismic_lang::checked::TensorAccess::Mutable);
                    let record = RecordPort::Parameter(parameter.name.clone());
                    // Each launch's source is formed with that launch's record ports (an entry
                    // formed whole, at launch 0's), so the asset may name any launch's symbols.
                    for launch in 0..definition.launches.len().max(1) {
                        let record_port = definition.launch_records(launch).contains(&record);
                        match parameter_reach(element_reach(element), mutated, record_port) {
                            OwnerReach::Raw => symbols.extend(names.tensor_buffers()),
                            OwnerReach::Owner => symbols.extend(names.tensor_handles(dialect == SourceDialect::Metal)),
                            // An element parameter binds at formation: the buffer or the handle.
                            OwnerReach::Bound => {
                                symbols.extend(names.tensor_buffers());
                                symbols.extend(names.tensor_handles(dialect == SourceDialect::Metal));
                            }
                        }
                    }
                    for prefix in names.tensor_prefixes() {
                        add_shape_symbols(&mut symbols, &prefix, *rank);
                        add_element_representation_symbols(&mut symbols, &prefix, element);
                    }
                }
                ParameterSummaryKind::Scalar(_) | ParameterSummaryKind::Index => {
                    let (symbol, alias) = names.word();
                    symbols.insert(symbol);
                    symbols.extend(alias);
                }
                ParameterSummaryKind::Range => {
                    let (pair, aliases) = names.range();
                    symbols.extend(pair);
                    symbols.extend(aliases.into_iter().flatten());
                }
            }
            abi.define(
                AbiMeaning::Parameter {
                    ordinal,
                    name: parameter.name.clone(),
                },
                symbols,
            )?;
        }

        for (ordinal, result) in entry.results.iter().enumerate() {
            let mut symbols = Symbols::default();
            match &result.kind {
                ResultSummaryKind::Tensor { rank, element } => {
                    let prefix = format!("SEISMIC_RESULT_{ordinal}");
                    match element_reach(element) {
                        OwnerReach::Raw => {
                            symbols.insert(format!("{prefix}_BUFFER"));
                        }
                        // A floating result has an owner handle and no buffer index.
                        OwnerReach::Owner => {
                            symbols.extend(owner_result_symbols(dialect, ordinal));
                        }
                        OwnerReach::Bound => {
                            symbols.insert(format!("{prefix}_BUFFER"));
                            symbols.extend(owner_result_symbols(dialect, ordinal));
                        }
                    }
                    add_shape_symbols(&mut symbols, &prefix, *rank);
                    add_element_representation_symbols(&mut symbols, &prefix, element);
                }
                ResultSummaryKind::Scalar(_)
                | ResultSummaryKind::Index
                | ResultSummaryKind::Range => {
                    symbols.insert(format!("SEISMIC_RESULT_{ordinal}_WORD"));
                }
            }
            abi.define(AbiMeaning::Result(ordinal), symbols)?;
        }
        Ok(abi
            .symbols
            .into_iter()
            .map(|(symbol, (_, visibility))| (symbol, visibility))
            .collect())
    }

    /// The declaration a generated ABI symbol stands for.
    #[derive(Clone, Debug, PartialEq, Eq)]
    enum AbiMeaning {
        /// A name the ABI itself fixes: its words and scalar-result
        /// buffers, and the dialect's helpers and device facts.
        Abi,
        Constant(String),
        Tune(String),
        Runtime(String),
        Forming(String),
        Resident(String),
        Scratch(String),
        Dimension(String),
        Element(String),
        Conversion(String, String),
        Parameter { ordinal: usize, name: String },
        Result(usize),
    }

    impl fmt::Display for AbiMeaning {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            match self {
                Self::Abi => write!(f, "the ABI's own buffers and helpers"),
                Self::Constant(name) => write!(f, "module constant `{name}`"),
                Self::Tune(name) => write!(f, "tuning parameter `{name}`"),
                Self::Runtime(name) => write!(f, "runtime parameter `{name}`"),
                Self::Forming(kernel) => write!(f, "launch `{kernel}`"),
                Self::Resident(kernel) => write!(f, "resident launch `{kernel}`"),
                Self::Scratch(name) => write!(f, "scratch buffer `{name}`"),
                Self::Dimension(name) => write!(f, "dimension `{name}`"),
                Self::Element(name) => write!(f, "element `{name}`"),
                Self::Conversion(from, to) => write!(f, "conversion `{from}` to `{to}`"),
                Self::Parameter { ordinal, name } => write!(f, "parameter {ordinal} `{name}`"),
                Self::Result(ordinal) => write!(f, "result {ordinal}"),
            }
        }
    }

    /// Every symbol an entry's native ABI defines, with its meaning and
    /// visibility. Every generated symbol enters through
    /// [`AbiSymbols::define`].
    struct AbiSymbols<'a> {
        entry: &'a str,
        symbols: HashMap<String, (AbiMeaning, Visibility)>,
    }

    impl AbiSymbols<'_> {
        /// Defines `symbols` as `meaning`. A symbol already defined with
        /// another meaning is a collision: the rendered source's later
        /// definition replaces the earlier, so no asset could address both.
        fn define(&mut self, meaning: AbiMeaning, symbols: Symbols) -> Result<(), BuildError> {
            for (symbol, visibility) in symbols.0 {
                match self.symbols.entry(symbol) {
                    Entry::Vacant(vacant) => {
                        vacant.insert((meaning.clone(), visibility));
                    }
                    Entry::Occupied(occupied) if occupied.get().0 == meaning => {
                        assert_eq!(
                            occupied.get().1,
                            visibility,
                            "`{}` has one visibility for one meaning",
                            occupied.key()
                        );
                    }
                    Entry::Occupied(occupied) => {
                        return Err(BuildError::NativeAbiCollision {
                            entry: self.entry.to_owned(),
                            symbol: occupied.key().clone(),
                            first: occupied.get().0.to_string(),
                            second: meaning.to_string(),
                        });
                    }
                }
            }
            Ok(())
        }
    }

    /// Generated ABI symbols of one declaration, each with who may spell it.
    #[derive(Default)]
    struct Symbols(HashMap<String, Visibility>);

    impl Symbols {
        /// `symbols`, every one of surface visibility.
        fn surface(symbols: impl IntoIterator<Item = String>) -> Self {
            let mut out = Self::default();
            out.extend(symbols);
            out
        }

        /// A name the ABI fixes or an entry's own binding name: surface.
        fn insert(&mut self, symbol: String) {
            self.add(symbol, Visibility::Surface);
        }

        fn extend(&mut self, symbols: impl IntoIterator<Item = String>) {
            for symbol in symbols {
                self.insert(symbol);
            }
        }

        /// `<prefix><suffix>` of a tensor's, element's or result's prefix,
        /// visible as the visibility table's suffix family states.
        fn tensor(&mut self, prefix: &str, suffix: &str) {
            self.add(format!("{prefix}{suffix}"), boundary_visibility(suffix, TENSOR_FAMILIES));
        }

        /// `<prefix><suffix>` of a conversion's prefix.
        fn conversion(&mut self, prefix: &str, suffix: &str) {
            self.add(format!("{prefix}{suffix}"), boundary_visibility(suffix, CONVERSION_FAMILIES));
        }

        fn add(&mut self, symbol: String, visibility: Visibility) {
            let previous = self.0.insert(symbol, visibility);
            assert!(
                previous.is_none_or(|previous| previous == visibility),
                "a generated symbol has one visibility"
            );
        }
    }

    /// A tensor's or result's rank, extents and strides.
    fn add_shape_symbols(symbols: &mut Symbols, prefix: &str, rank: u32) {
        symbols.tensor(prefix, "_RANK");
        for axis in 0..rank {
            symbols.tensor(prefix, &format!("_EXTENT_{axis}"));
            symbols.tensor(prefix, &format!("_STRIDE_{axis}"));
        }
    }

    /// The symbol of a result's owner handle, as the dialect names it.
    fn owner_result_symbols(dialect: SourceDialect, ordinal: usize) -> Vec<String> {
        vec![seismic_lang::native_parameter::result_handle(ordinal, dialect == SourceDialect::Metal)]
    }

    fn add_element_representation_symbols(
        symbols: &mut Symbols,
        prefix: &str,
        element: &ElementSummary,
    ) {
        match element {
            ElementSummary::Fixed(name) => {
                if let Some(id) = registry::representation(name) {
                    add_representation_symbols(
                        symbols,
                        prefix,
                        std::iter::once(registry::representation_info(id)),
                    );
                }
            }
            ElementSummary::Parameter(_) => {
                add_representation_symbols(symbols, prefix, registry::representations().iter());
            }
        }
    }

    fn add_representation_symbols<'a>(
        symbols: &mut Symbols,
        prefix: &str,
        representations: impl Iterator<Item = &'a RepresentationInfo>,
    ) {
        // The bare prefix, which entries pass as a binding argument.
        symbols.tensor(prefix, "");
        symbols.tensor(prefix, "_REPRESENTATION");
        for representation in representations {
            symbols.tensor(prefix, &format!("_REPRESENTATION_{}", native_macro(representation.representation)));
            symbols.tensor(prefix, &format!("_DECODED_{}", native_macro(representation.decoded.name())));
            if representation.operands == registry::OperandRecipe::Nvfp4Rows {
                symbols.tensor(prefix, &format!("_OPERANDS_NVFP4"));
            }
            symbols.tensor(prefix, "_PACKET_SIZE");
            symbols.tensor(prefix, "_PACKET_ALIGNMENT");
            symbols.tensor(prefix, "_LOGICAL_GROUP");
            symbols.tensor(prefix, "_PLANE_COUNT");
            match &representation.kind {
                registry::RepresentationKind::Dense(_) => {
                    symbols.tensor(prefix, "_KIND_DENSE");
                }
                registry::RepresentationKind::External(_) => {
                    symbols.tensor(prefix, "_KIND_EXTERNAL");
                    symbols.tensor(prefix, "_GRID_PLANAR");
                    symbols.tensor(prefix, "_GRID_SCALE_BYTES");
                    symbols.tensor(prefix, "_GRID_TILE_ROWS");
                    symbols.tensor(prefix, "_GRID_TILE_COLUMNS");
                }
                // The row-layout ABI of the frozen weight-storage interface
                // (native-program-execution.md, "Weight storage").
                registry::RepresentationKind::PackedRows(layout) => {
                    symbols.tensor(prefix, "_KIND_PACKED");
                    symbols.tensor(prefix, &format!("_LAYOUT_{}", native_macro(layout.layout.as_str())));
                    symbols.tensor(prefix, "_ROW_STRIDE_BYTES");
                    // Row planes by role (`render_row_roles`).
                    for role in registry::RowPlaneRole::ALL {
                        let role = native_macro(role.name());
                        symbols.tensor(prefix, &format!("_ROW_{role}"));
                        symbols.tensor(prefix, &format!("_ROW_{role}_BYTES"));
                        symbols.tensor(prefix, &format!("_ROW_{role}_TAIL"));
                    }
                    for suffix in [
                        "ROW_GROUPS",
                        "GROUP_MULTIPLE",
                        "ROW_ALIGNMENT",
                        "TILE_ROWS",
                        "MMA_KBLOCK",
                        "MMA_LANES",
                        "MMA_STEP",
                        "MMA_STEP_SLOTS",
                        "MMA_FRAGMENT_ROW",
                        "MMA_FRAGMENT_COLUMN",
                        "MMA_FRAGMENT_BYTE",
                        "RECORD_ENTRY",
                        "RECORD_FACTOR",
                    ] {
                        symbols.tensor(prefix, &format!("_{suffix}"));
                    }
                    for plane in &layout.planes {
                        let plane_suffix = format!("_PLANE_{}", native_macro(plane.name));
                        symbols.tensor(prefix, &plane_suffix);
                        for suffix in [
                            "ROW_OFFSET",
                            "BYTES_PER_ROW",
                            "BYTES_PER_GROUP",
                            "CODE_BITS",
                            "CARRIED_BIT",
                            "ENTRY_BYTES",
                            "FACTORS_OFFSET",
                            "ENTRY_BITS",
                            "KBLOCK_ENTRIES",
                        ] {
                            symbols.tensor(prefix, &format!("{plane_suffix}_{suffix}"));
                        }
                    }
                }
                registry::RepresentationKind::Packed(layout) => {
                    symbols.tensor(prefix, "_KIND_PACKED");
                    symbols.tensor(prefix, "_LAYOUT_PACKET");
                    for (ordinal, plane) in layout.planes.iter().enumerate() {
                        let plane_suffix = format!("_PLANE_{ordinal}");
                        symbols.tensor(prefix, &format!("{plane_suffix}_NAME_{}", native_macro(plane.name)));
                        for suffix in [
                            "OFFSET",
                            "BYTES_PER_GROUP",
                            "ALIGNMENT",
                            "GROUP",
                            "FIELDS",
                            "ENTRY_BITS",
                        ] {
                            symbols.tensor(prefix, &format!("{plane_suffix}_{suffix}"));
                        }
                        symbols.tensor(
                            prefix,
                            &format!("{plane_suffix}_STORAGE_{}", native_macro(plane.storage_dtype.name())),
                        );
                    }
                }
            }
        }
    }

    fn seismic_identifiers(source: &str) -> Vec<(&str, usize)> {
        identifiers(source)
            .into_iter()
            .filter(|(identifier, _)| identifier.starts_with("SEISMIC_"))
            .collect()
    }

    use seismic_lang::native_source::identifiers;

    use seismic_lang::native_name::native_macro;

    fn render(
        module: &seismic_lang::checked::CheckedModule,
        name: &str,
        identity: &str,
        native_assets: &[EntryNative<'_>],
        cpu_library: Option<&CpuLibrary>,
    ) -> Result<String, BuildError> {
        let mut out = String::new();
        out.push_str("// @generated by seismic-build; do not edit.\n");
        out.push_str(&format!(
            "static __BUNDLE: &[u8] = include_bytes!({:?});\n",
            format!("{name}.seismicbundle")
        ));
        out.push_str("static __MODULE: seismic::generated::OnceLock<Result<seismic::generated::Module, seismic::CheckedBundleError>> = seismic::generated::OnceLock::new();\n");
        out.push_str(
            "/// The checked module, an opaque token for Seismic's module-wide services.\n",
        );
        out.push_str("pub fn module() -> Result<&'static seismic::generated::Module, seismic::CheckedBundleError> { seismic::generated::module_from_bundle(&__MODULE, __BUNDLE) }\n");
        out.push_str(&format!("pub const IDENTITY: &str = {:?};\n", identity));
        out.push_str("/// The module constants (`const NAME = INTEGER`) and constant functions (`const NAME(P, ..) = EXPR`), which natives read as `SEISMIC_CONST_NAME`.\npub mod constants {\n");
        for (name, value) in module.constants() {
            out.push_str(&format!("  pub const {name}: u64 = {value};\n"));
        }
        for function in module.constant_functions() {
            render_constant_function(&mut out, function);
        }
        out.push_str("}\n");
        if let Some(library) = cpu_library {
            render_cpu_library(&mut out, library);
        }
        for entry in module.entries() {
            let native = native_assets.iter().find(|native| native.entry == entry.id);
            render_entry(&mut out, module, entry, native, cpu_library)?;
        }
        Ok(out)
    }

    fn render_entry(
        out: &mut String,
        module: &seismic_lang::checked::CheckedModule,
        entry: &EntryInfo,
        native: Option<&EntryNative<'_>>,
        cpu_library: Option<&CpuLibrary>,
    ) -> Result<(), BuildError> {
        let module_name = rust_ident(&entry.name);
        out.push_str(&format!("pub mod {module_name} {{\n"));
        let borrowed = entry.parameters.iter().any(|parameter| {
            matches!(
                parameter.kind,
                ParameterSummaryKind::Tensor {
                    access: TensorAccess::Shared | TensorAccess::Mutable | TensorAccess::Host { .. },
                    ..
                }
            )
        });
        let workflow_borrowed = entry
            .parameters
            .iter()
            .any(|parameter| matches!(parameter.kind, ParameterSummaryKind::Tensor { .. }));
        if borrowed {
            out.push_str("  pub struct Args<'a> {\n");
        } else {
            out.push_str("  pub struct Args {\n");
        }
        let sources = parameter_sources(&entry.parameters);
        for parameters in &sources {
            let parameter = parameters[0];
            out.push_str(&format!(
                "    pub {}: {},\n",
                rust_ident(&parameter.name),
                source_parameter_type(parameters, 0)
            ));
        }
        out.push_str("  }\n");

        if workflow_borrowed {
            out.push_str("  pub struct WorkflowArgs<'a> {\n");
        } else {
            out.push_str("  pub struct WorkflowArgs {\n");
        }
        for parameters in &sources {
            let parameter = parameters[0];
            out.push_str(&format!(
                "    pub {}: {},\n",
                rust_ident(&parameter.name),
                source_workflow_parameter_type(parameters, 0)
            ));
        }
        out.push_str("  }\n");

        out.push_str("  pub struct Results {\n");
        for result in &entry.results {
            out.push_str(&format!(
                "    pub {}: {},\n",
                result_name(&result.path),
                result_type(&result.kind)
            ));
        }
        out.push_str("  }\n");

        let tensor_results = entry
            .results
            .iter()
            .filter(|result| matches!(&result.kind, ResultSummaryKind::Tensor { .. }))
            .collect::<Vec<_>>();
        if tensor_results.is_empty() {
            out.push_str("  pub struct OutputArgs;\n");
        } else {
            out.push_str("  pub struct OutputArgs<'a> {\n");
            for result in &tensor_results {
                out.push_str(&format!(
                    "    pub {}: &'a mut seismic::Tensor,\n",
                    result_name(&result.path)
                ));
            }
            out.push_str("  }\n");
        }

        out.push_str("  #[derive(Clone)]\n  pub struct WorkflowResults {\n");
        for result in &entry.results {
            out.push_str(&format!(
                "    pub {}: {},\n",
                result_name(&result.path),
                workflow_result_type(&result.kind)
            ));
        }
        out.push_str("  }\n");

        if !entry.element_parameters.is_empty() {
            out.push_str("  pub struct Elements {\n");
            for parameter in &entry.element_parameters {
                out.push_str(&format!(
                    "    pub {}: seismic::Element,\n",
                    rust_ident(parameter)
                ));
            }
            out.push_str("  }\n");
        }

        out.push_str("  pub struct Entry;\n");
        out.push_str("  unsafe impl seismic::Entry for Entry {\n");
        if borrowed {
            out.push_str("    type Args<'a> = Args<'a>;\n");
        } else {
            out.push_str("    type Args<'a> = Args;\n");
        }
        out.push_str("    type Results = Results;\n");
        if tensor_results.is_empty() {
            out.push_str("    type OutputArgs<'a> = OutputArgs;\n");
        } else {
            out.push_str("    type OutputArgs<'a> = OutputArgs<'a>;\n");
        }
        if workflow_borrowed {
            out.push_str("    type WorkflowArgs<'a> = WorkflowArgs<'a>;\n");
        } else {
            out.push_str("    type WorkflowArgs<'a> = WorkflowArgs;\n");
        }
        out.push_str("    type WorkflowResults = WorkflowResults;\n");
        out.push_str(&format!(
            "    const NAME: &'static str = {:?};\n",
            entry.name
        ));
        // Argument words travel by value; a prepared native implementation
        // owns only its scalar-result slots, none without scalar results.
        let scalar_words = entry
            .results
            .iter()
            .map(|result| result.kind.class().scalar_words() as u64)
            .sum::<u64>();
        out.push_str(&format!(
            "    const NATIVE_SCALAR_RESULT_BYTES: u64 = {};\n",
            scalar_words * ResultClass::WORD_BYTES
        ));
        out.push_str("    fn module() -> Result<&'static seismic::generated::Module, seismic::CheckedBundleError> { super::module() }\n");
        out.push_str(&format!(
            "    fn resolve(module: &seismic::generated::Module) -> Result<seismic::generated::EntryToken, seismic::CheckedBundleError> {{ module.entry_named({:?}).ok_or(seismic::CheckedBundleError::Corrupt) }}\n",
            entry.name
        ));
        out.push_str("    fn encode(args: Self::Args<'_>) -> seismic::generated::EncodedArgs {\n");
        out.push_str("      let mut encoder = seismic::generated::ArgsEncoder::new();\n");
        for parameters in &sources {
            let source = parameters[0];
            let field = rust_ident(&source.name);
            for parameter in parameters {
                let expression = parameter_access(&field, &parameter.path);
                match &parameter.kind {
                    ParameterSummaryKind::Tensor { access, .. } => match access {
                        TensorAccess::Owned => {
                            out.push_str(&format!("      encoder.tensor(&{expression});\n"));
                        }
                        TensorAccess::Shared | TensorAccess::Mutable | TensorAccess::Host { .. } => {
                            out.push_str(&format!("      encoder.tensor({expression});\n"));
                        }
                    },
                    ParameterSummaryKind::Scalar(dtype) => {
                        out.push_str(&format!("      encoder.{}({expression});\n", dtype.name()));
                    }
                    ParameterSummaryKind::Index => {
                        out.push_str(&format!("      encoder.index({expression});\n"))
                    }
                    ParameterSummaryKind::Range => {
                        out.push_str(&format!("      encoder.range({expression});\n"))
                    }
                }
            }
        }
        out.push_str("      encoder.finish()\n    }\n");
        out.push_str("    fn encode_outputs(outputs: Self::OutputArgs<'_>) -> seismic::generated::EncodedOutputs {\n");
        out.push_str("      let mut encoder = seismic::generated::OutputArgsEncoder::new();\n");
        for result in &tensor_results {
            out.push_str(&format!(
                "      encoder.tensor(outputs.{});\n",
                result_name(&result.path)
            ));
        }
        if tensor_results.is_empty() {
            out.push_str("      let _ = outputs;\n");
        }
        out.push_str("      encoder.finish()\n    }\n");
        // An entry without results has nothing to decode or publish.
        let (decoded, published) = if entry.results.is_empty() {
            ("_", "_")
        } else {
            ("mut results", "results")
        };
        out.push_str(&format!(
            "    fn decode({decoded}: seismic::generated::DecodedResults) -> Results {{ Results {{\n"
        ));
        for result in &entry.results {
            let getter = match &result.kind {
                ResultSummaryKind::Tensor { .. } => "take_tensor",
                ResultSummaryKind::Scalar(dtype) => match dtype {
                    DType::F32 => "take_f32",
                    DType::F16 => "take_f16",
                    DType::BF16 => "take_bf16",
                    DType::I32 => "take_i32",
                    DType::U32 => "take_u32",
                    DType::U64 => "take_u64",
                    DType::Bool => "take_bool",
                },
                ResultSummaryKind::Index => "take_index",
                ResultSummaryKind::Range => "take_range",
            };
            out.push_str(&format!(
                "      {}: seismic::generated::{getter}(&mut results),\n",
                result_name(&result.path)
            ));
        }
        out.push_str("    } }\n");

        out.push_str("    fn encode_workflow(args: Self::WorkflowArgs<'_>) -> seismic::generated::EncodedWorkflowArgs {\n");
        out.push_str("      let mut encoder = seismic::generated::WorkflowArgsEncoder::new();\n");
        for parameters in &sources {
            let source = parameters[0];
            let field = rust_ident(&source.name);
            for parameter in parameters {
                let expression = parameter_access(&field, &parameter.path);
                match &parameter.kind {
                    ParameterSummaryKind::Tensor { access, .. } => {
                        let method = match access {
                            TensorAccess::Owned => "owned_tensor",
                            // A host port binds as the device sees it.
                            TensorAccess::Shared | TensorAccess::Host { .. } => "shared_tensor",
                            TensorAccess::Mutable => "mutable_tensor",
                        };
                        out.push_str(&format!("      encoder.{method}({expression});\n"));
                    }
                    ParameterSummaryKind::Scalar(dtype) => {
                        out.push_str(&format!("      encoder.{}({expression});\n", dtype.name()));
                    }
                    ParameterSummaryKind::Index => {
                        out.push_str(&format!("      encoder.index({expression});\n"));
                    }
                    ParameterSummaryKind::Range => {
                        out.push_str(&format!("      encoder.range({expression});\n"));
                    }
                }
            }
        }
        out.push_str("      encoder.finish()\n    }\n");
        out.push_str(&format!(
            "    fn decode_workflow({decoded}: seismic::generated::PendingWorkflowResults) -> WorkflowResults {{ WorkflowResults {{\n"
        ));
        for result in &entry.results {
            let getter = match &result.kind {
                ResultSummaryKind::Tensor { .. } => "take_workflow_tensor".to_owned(),
                ResultSummaryKind::Scalar(dtype) => {
                    format!("take_workflow_scalar::<{}>", scalar_type(*dtype))
                }
                ResultSummaryKind::Index => "take_workflow_scalar::<seismic::BigUint>".to_owned(),
                ResultSummaryKind::Range => {
                    "take_workflow_scalar::<(seismic::BigUint, seismic::BigUint)>".to_owned()
                }
            };
            out.push_str(&format!(
                "      {}: seismic::generated::{getter}(&mut results),\n",
                result_name(&result.path)
            ));
        }
        out.push_str("    } }\n");
        out.push_str(&format!(
            "    fn workflow_outputs({published}: WorkflowResults) -> Vec<seismic::generated::WorkflowResultRef> {{ vec![\n"
        ));
        for result in &entry.results {
            let getter = match &result.kind {
                ResultSummaryKind::Tensor { .. } => "workflow_tensor_ref",
                ResultSummaryKind::Scalar(_)
                | ResultSummaryKind::Index
                | ResultSummaryKind::Range => "workflow_scalar_ref",
            };
            out.push_str(&format!(
                "      seismic::generated::{getter}(results.{}),\n",
                result_name(&result.path)
            ));
        }
        out.push_str("    ] }\n");
        out.push_str("  }\n");

        render_invocation_scope(out, entry);

        if entry.element_parameters.is_empty() {
            out.push_str("  pub fn for_device(charges: &seismic::Charges, options: seismic::PreparationOptions) -> Result<seismic::Kernel<Entry>, seismic::LoadError> { seismic::generated::prepare::<Entry>(charges, options, &[]) }\n");
        } else {
            out.push_str("  pub fn for_device_with(charges: &seismic::Charges, options: seismic::PreparationOptions, elements: Elements) -> Result<seismic::Kernel<Entry>, seismic::LoadError> {\n");
            out.push_str("    seismic::generated::prepare::<Entry>(charges, options, &[\n");
            for parameter in &entry.element_parameters {
                out.push_str(&format!(
                    "      ({:?}, elements.{}),\n",
                    parameter,
                    rust_ident(parameter)
                ));
            }
            out.push_str("    ])\n  }\n");
        }
        if entry.element_parameters.is_empty() {
            out.push_str("  pub fn start_feedback(charges: &seismic::Charges, precision: seismic::PrecisionPolicy, options: seismic::FeedbackOptions) -> Result<(seismic::FeedbackPreparation<'_, Entry>, seismic::Kernel<Entry>), seismic::LoadError> { seismic::generated::start_feedback::<Entry>(charges, precision, options, &[]) }\n");
        } else {
            out.push_str("  pub fn start_feedback_with(charges: &seismic::Charges, precision: seismic::PrecisionPolicy, options: seismic::FeedbackOptions, elements: Elements) -> Result<(seismic::FeedbackPreparation<'_, Entry>, seismic::Kernel<Entry>), seismic::LoadError> {\n");
            out.push_str(
                "    seismic::generated::start_feedback::<Entry>(charges, precision, options, &[\n",
            );
            for parameter in &entry.element_parameters {
                out.push_str(&format!(
                    "      ({:?}, elements.{}),\n",
                    parameter,
                    rust_ident(parameter)
                ));
            }
            out.push_str("    ])\n  }\n");
        }
        crate::render_contract::render_contract(out, module, entry)?;
        if let Some(native) = native {
            render_native(out, entry, native, cpu_library)?;
        }
        out.push_str("}\n");
        Ok(())
    }

    /// Native entry points, and for a CPU implementation its typed context
    /// and launch table.
    fn render_native(
        out: &mut String,
        entry: &EntryInfo,
        native: &EntryNative<'_>,
        cpu_library: Option<&CpuLibrary>,
    ) -> Result<(), BuildError> {
        let elements = !entry.element_parameters.is_empty();
        let element_list = |out: &mut String| {
            for parameter in &entry.element_parameters {
                out.push_str(&format!(
                    "      ({:?}, elements.{}),\n",
                    parameter,
                    rust_ident(parameter)
                ));
            }
        };
        let cpu = if native.cpu.is_some() {
            "Some(&cpu_native::KERNELS)"
        } else {
            "None"
        };
        if elements {
            out.push_str("  pub fn native_for_device_with(device: &seismic::Device, elements: Elements, specialization: &seismic::NativeSpecialization) -> Result<seismic::NativeKernel<Entry>, seismic::LoadError> {\n");
            out.push_str(
                "    seismic::generated::prepare_native::<Entry>(device, specialization, &[\n",
            );
            element_list(out);
            out.push_str(&format!("    ], {cpu})\n  }}\n"));
            out.push_str("  /// Whether preparation on a `backend` device takes `specialization` at `elements`, decided without a device.\n");
            out.push_str("  pub fn native_admission_with(backend: seismic::BackendName, elements: Elements, specialization: &seismic::NativeSpecialization) -> Result<seismic::NativeAdmission, seismic::CheckedBundleError> {\n");
            out.push_str("    seismic::generated::admit_native::<Entry>(backend, specialization, &[\n");
            element_list(out);
            out.push_str(&format!("    ], {cpu})\n  }}\n"));
            out.push_str(
                "  pub fn native_entry_with(elements: Elements) -> seismic::BoundEntry<Entry> {\n",
            );
            out.push_str("    seismic::generated::bound_entry::<Entry>(&[\n");
            element_list(out);
            out.push_str(&format!("    ], {cpu})\n  }}\n"));
            out.push_str("  pub fn native_tune_with<'a>(charges: &seismic::Charges, elements: Elements, statics: &seismic::NativeSpecialization, mut points: impl seismic::PointSource<'a, Entry>, validation: seismic::NumericalRequirement, strategy: seismic::Strategy) -> Result<seismic::TuningResult, seismic::TuneError> {\n");
            out.push_str("    seismic::generated::tune_native::<Entry>(charges, statics, &[\n");
            element_list(out);
            out.push_str(&format!(
                "    ], {cpu}, &mut points, validation, strategy)\n  }}\n"
            ));
            out.push_str("  pub fn native_search_with<'a>(charges: &seismic::Charges, elements: Elements, statics: &seismic::NativeSpecialization, validation: seismic::NumericalRequirement) -> Result<seismic::NativeSearch<'a, Entry>, seismic::TuneError> {\n");
            out.push_str("    seismic::generated::search_native::<Entry>(charges, statics, &[\n");
            element_list(out);
            out.push_str(&format!("    ], {cpu}, validation)\n  }}\n"));
            out.push_str("  pub fn native_digest_with(device: &seismic::Device, elements: Elements, statics: &seismic::NativeSpecialization) -> Result<String, seismic::TuneError> {\n");
            out.push_str("    seismic::generated::digest_native::<Entry>(device, statics, &[\n");
            element_list(out);
            out.push_str(&format!("    ], {cpu})\n  }}\n"));
        } else {
            out.push_str(&format!("  pub fn native_for_device(device: &seismic::Device, specialization: &seismic::NativeSpecialization) -> Result<seismic::NativeKernel<Entry>, seismic::LoadError> {{ seismic::generated::prepare_native::<Entry>(device, specialization, &[], {cpu}) }}\n"));
            out.push_str("  /// Whether preparation on a `backend` device takes `specialization`, decided without a device.\n");
            out.push_str(&format!("  pub fn native_admission(backend: seismic::BackendName, specialization: &seismic::NativeSpecialization) -> Result<seismic::NativeAdmission, seismic::CheckedBundleError> {{ seismic::generated::admit_native::<Entry>(backend, specialization, &[], {cpu}) }}\n"));
            out.push_str(&format!("  pub fn native_entry() -> seismic::BoundEntry<Entry> {{ seismic::generated::bound_entry::<Entry>(&[], {cpu}) }}\n"));
            out.push_str(&format!("  pub fn native_tune<'a>(charges: &seismic::Charges, statics: &seismic::NativeSpecialization, mut points: impl seismic::PointSource<'a, Entry>, validation: seismic::NumericalRequirement, strategy: seismic::Strategy) -> Result<seismic::TuningResult, seismic::TuneError> {{ seismic::generated::tune_native::<Entry>(charges, statics, &[], {cpu}, &mut points, validation, strategy) }}\n"));
            out.push_str(&format!("  pub fn native_search<'a>(charges: &seismic::Charges, statics: &seismic::NativeSpecialization, validation: seismic::NumericalRequirement) -> Result<seismic::NativeSearch<'a, Entry>, seismic::TuneError> {{ seismic::generated::search_native::<Entry>(charges, statics, &[], {cpu}, validation) }}\n"));
            out.push_str(&format!("  pub fn native_digest(device: &seismic::Device, statics: &seismic::NativeSpecialization) -> Result<String, seismic::TuneError> {{ seismic::generated::digest_native::<Entry>(device, statics, &[], {cpu}) }}\n"));
        }
        if !native.tuned {
            out.push_str("  impl seismic::Untuned for Entry {}\n");
        }
        out.push_str("  /// The checked native implementation for the device's backend.\n");
        out.push_str("  pub fn native_implementation(device: &seismic::Device) -> Result<Option<seismic::OnDevice>, seismic::CheckedBundleError> { seismic::generated::native_implementation::<Entry>(device) }\n");
        if let Some(cpu) = &native.cpu {
            let library =
                cpu_library.expect("a build with a CPU implementation scans its CPU library");
            render_cpu_native(out, entry, cpu, library)?;
        }
        Ok(())
    }

    /// The CPU library of a build: every `.rs` file of its source roots that
    /// is not a CPU native asset. Each file is one module of a tree that
    /// mirrors its directories, so a Rust path between library files, or
    /// from an asset to a library file, is a path relative to the file.
    struct CpuLibrary {
        roots: Vec<PathBuf>,
        /// Module path (path relative to its root without the extension)
        /// and canonical path of each file, sorted by module path.
        files: Vec<(Vec<String>, PathBuf)>,
        /// Covers every file's module path and contents.
        digest: [u8; 32],
    }

    fn cpu_library(
        roots: &[PathBuf],
        assets: &[&std::path::Path],
    ) -> Result<CpuLibrary, BuildError> {
        fn scan(
            root: &std::path::Path,
            directory: &std::path::Path,
            assets: &[&std::path::Path],
            files: &mut Vec<(Vec<String>, PathBuf)>,
        ) -> Result<(), BuildError> {
            // A new library file changes every CPU implementation's module
            // tree and digest.
            println!("cargo:rerun-if-changed={}", directory.display());
            let mut entries = fs::read_dir(directory)
                .map_err(BuildError::Io)?
                .map(|entry| entry.map(|entry| entry.path()))
                .collect::<Result<Vec<_>, _>>()
                .map_err(BuildError::Io)?;
            entries.sort();
            for path in entries {
                let hidden = path
                    .file_name()
                    .and_then(|name| name.to_str())
                    .is_some_and(|name| name.starts_with('.'));
                if hidden {
                    continue;
                }
                if path.is_dir() {
                    scan(root, &path, assets, files)?;
                } else if path.extension().is_some_and(|extension| extension == "rs") {
                    let path = path.canonicalize().map_err(BuildError::Io)?;
                    if assets.contains(&path.as_path()) {
                        continue;
                    }
                    println!("cargo:rerun-if-changed={}", path.display());
                    let text = fs::read_to_string(&path).map_err(BuildError::Io)?;
                    if let Some(function) = out_of_line(&text) {
                        return Err(BuildError::CpuNative { reason: out_of_line_reason(function), path });
                    }
                    let modules = path
                        .strip_prefix(root)
                        .expect("a scanned file lies under its root")
                        .with_extension("")
                        .components()
                        .map(|component| component.as_os_str().to_string_lossy().into_owned())
                        .collect::<Vec<_>>();
                    for module in &modules {
                        if rust_ident(module) != *module {
                            return Err(BuildError::CpuNative {
                                path,
                                reason: format!(
                                    "`{module}` is not a Rust module name; CPU library paths become module paths"
                                ),
                            });
                        }
                    }
                    files.push((modules, path));
                }
            }
            Ok(())
        }
        let mut files = Vec::new();
        for root in roots {
            scan(root, root, assets, &mut files)?;
        }
        files.sort();
        if let Some(pair) = files.windows(2).find(|pair| pair[0].0 == pair[1].0) {
            return Err(BuildError::CpuNative {
                path: pair[1].1.clone(),
                reason: format!(
                    "its module path `{}` is also `{}`",
                    pair[1].0.join("::"),
                    pair[0].1.display()
                ),
            });
        }
        let mut digest = Sha256::new();
        for (modules, path) in &files {
            digest.update(modules.join("/").as_bytes());
            digest.update([0u8]);
            digest.update(fs::read(path).map_err(BuildError::Io)?);
            digest.update([0u8]);
        }
        Ok(CpuLibrary {
            roots: roots.to_vec(),
            files,
            digest: digest.finalize().into(),
        })
    }

    /// The library module tree, once per bindings module.
    fn render_cpu_library(out: &mut String, library: &CpuLibrary) {
        #[derive(Default)]
        struct Node {
            file: Option<PathBuf>,
            children: std::collections::BTreeMap<String, Node>,
        }
        fn render(out: &mut String, name: &str, node: &Node, depth: usize) {
            let indent = "  ".repeat(depth);
            out.push_str(&format!("{indent}pub mod {name} {{\n"));
            if let Some(file) = &node.file {
                out.push_str(&format!(
                    "{indent}  include!({:?});\n",
                    file.to_string_lossy()
                ));
            }
            for (child, node) in &node.children {
                render(out, child, node, depth + 1);
            }
            out.push_str(&format!("{indent}}}\n"));
        }
        let mut tree = Node::default();
        for (modules, path) in &library.files {
            let node = modules.iter().fold(&mut tree, |node, module| {
                node.children.entry(module.clone()).or_default()
            });
            node.file = Some(path.clone());
        }
        out.push_str(
            "/// The CPU library files of the build's source roots, one module per file.\n",
        );
        out.push_str("#[allow(dead_code, unused_imports)]\n");
        render(out, "cpu_library", &tree, 0);
    }

    /// The CPU library element type of a dense dtype a CPU form covers.
    fn cpu_element_type(dtype: DType) -> &'static str {
        match dtype {
            DType::F32 => "F32",
            DType::BF16 => "Bf16",
            DType::F16 => "F16",
            DType::I32 => "I32",
            DType::U32 => "U32",
            DType::U64 => "U64",
            DType::Bool => unreachable!("the checker admits no `bool` CPU element"),
        }
    }

    /// How a CPU form binds each element parameter of an entry.
    struct CpuElements {
        /// Parameters bound as dense typed buffers, with the dense types the
        /// form is monomorphized over (declared, or the floating default).
        dense: Vec<(String, Vec<DType>)>,
        /// Weight operands, reaching the kernel through their resolved
        /// components.
        weights: Vec<String>,
        /// External sources the entry only converts.
        external: Vec<String>,
        /// Conversion targets: packed results written by a conversion.
        packed: Vec<String>,
    }

    /// Every element parameter's binding is the checker's
    /// ([`NativeImplementation::cpu_binding`]); a CPU native declaration is
    /// admitted only when every parameter has one.
    fn cpu_elements(entry: &EntryInfo, definition: &NativeImplementation) -> CpuElements {
        let mut elements = CpuElements {
            dense: Vec::new(),
            weights: Vec::new(),
            external: Vec::new(),
            packed: Vec::new(),
        };
        for parameter in entry.element_domain.parameters() {
            let name = parameter.name.clone();
            match definition.cpu_binding(entry, &name) {
                Some(CpuBinding::Dense) => {
                    let dtypes = definition
                        .elements
                        .iter()
                        .find(|coverage| coverage.parameter == name)
                        .map(|coverage| coverage.dtypes.clone())
                        .unwrap_or_else(|| NativeElementCoverage::DEFAULT.to_vec());
                    elements.dense.push((name, dtypes));
                }
                Some(CpuBinding::Weights) => elements.weights.push(name),
                Some(CpuBinding::External) => elements.external.push(name),
                Some(CpuBinding::Packed) => elements.packed.push(name),
                None => unreachable!(
                    "the checker admits a CPU native only where every element parameter has a binding"
                ),
            }
        }
        elements
    }

    /// The accessor of entry parameter `ordinal` on a CPU `Context`:
    /// `arg_<name>`, or `arg_<ordinal>` where another parameter shares the
    /// lowered sanitized name.
    fn cpu_argument(entry: &EntryInfo, ordinal: usize) -> String {
        let key = ident_fragment(&entry.parameters[ordinal].name).to_lowercase();
        let unique = entry
            .parameters
            .iter()
            .filter(|candidate| ident_fragment(&candidate.name).to_lowercase() == key)
            .count()
            == 1;
        if unique {
            format!("arg_{key}")
        } else {
            format!("arg_{ordinal}")
        }
    }

    /// `expression` as a Rust `u64` expression, `const` wherever `read`'s
    /// are, `read` writing each name it reads (`None`: one it cannot). Its
    /// arithmetic is the checker's (`NativeNatExpr::evaluate`): an overflow
    /// or a zero divisor, which fails evaluation there, panics here.
    fn rust_nat(
        expression: &seismic_lang::checked::NativeNatExpr,
        read: &dyn Fn(&seismic_lang::checked::NativeNatExpr) -> Option<String>,
    ) -> Option<String> {
        use seismic_lang::checked::NativeNatExpr as E;
        let pair = |left: &E, right: &E| Some((rust_nat(left, read)?, rust_nat(right, read)?));
        let checked = |left: &E, right: &E, operation: &str| {
            pair(left, right).map(|(l, r)| {
                format!("({l}).checked_{operation}({r}).expect(\"native arithmetic stays within u64\")")
            })
        };
        let select = |left: &E, right: &E, keeps: &str| {
            pair(left, right).map(|(l, r)| {
                format!("({{ let (left, right) = ({l}, {r}); if left {keeps} right {{ left }} else {{ right }} }})")
            })
        };
        match expression {
            E::Constant(value) => Some(format!("{value}u64")),
            E::Dimension(_) | E::Parameter(_) | E::Element(..) | E::Device(_) => read(expression),
            E::Add(left, right) => checked(left, right, "add"),
            E::Sub(left, right) => checked(left, right, "sub"),
            E::Mul(left, right) => checked(left, right, "mul"),
            E::Div(left, right) => checked(left, right, "div"),
            E::Rem(left, right) => checked(left, right, "rem"),
            E::CeilDiv(left, right) => pair(left, right).map(|(l, r)| format!("({l}).div_ceil({r})")),
            E::Min(left, right) => select(left, right, "<="),
            E::Max(left, right) => select(left, right, ">="),
        }
    }

    /// `expression` as a CPU kernel's `u64` ([`rust_nat`]), its dimensions
    /// and parameters read from `cx`; `None` for one that reads an element
    /// or the device, which a formation's shape never does.
    fn cpu_nat(expression: &seismic_lang::checked::NativeNatExpr) -> Option<String> {
        use seismic_lang::checked::NativeNatExpr as E;
        rust_nat(expression, &|read| match read {
            E::Dimension(name) => Some(format!("cx.dim_{}()", ident_fragment(name).to_lowercase())),
            E::Parameter(name) => Some(format!("cx.param_{}()", ident_fragment(name).to_lowercase())),
            _ => None,
        })
    }

    /// A module constant function as a `const fn` of its parameters
    /// ([`rust_nat`]), so a host derives what a declaration calls it for by
    /// the declaration's own definition.
    fn render_constant_function(out: &mut String, function: &seismic_lang::checked::ConstantFunction) {
        use seismic_lang::checked::NativeNatExpr as E;
        let parameters = function
            .parameters()
            .iter()
            .map(|parameter| format!("{}: u64", rust_ident(parameter)))
            .collect::<Vec<_>>()
            .join(", ");
        let body = rust_nat(&function.body().nat(), &|read| match read {
            E::Dimension(name) => Some(rust_ident(name)),
            _ => None,
        })
        .unwrap_or_else(|| panic!("constant function `{}` reads only its parameters", function.name()));
        out.push_str(&format!(
            "  #[allow(non_snake_case)]\n  pub const fn {}({parameters}) -> u64 {{\n    {body}\n  }}\n",
            rust_ident(function.name())
        ));
    }

    /// `condition` as a CPU kernel's `bool` ([`cpu_nat`]).
    fn cpu_condition(condition: &seismic_lang::checked::NativeCondition) -> Option<String> {
        use seismic_lang::checked::{NativeComparison as C, NativeCondition};
        Some(match condition {
            NativeCondition::Compare { comparison, left, right } => {
                let op = match comparison {
                    C::Lt => "<",
                    C::Le => "<=",
                    C::Gt => ">",
                    C::Ge => ">=",
                    C::Eq => "==",
                    C::Ne => "!=",
                };
                format!("({} {op} {})", cpu_nat(left)?, cpu_nat(right)?)
            }
            NativeCondition::And(left, right) => format!("({} && {})", cpu_condition(left)?, cpu_condition(right)?),
            NativeCondition::Or(left, right) => format!("({} || {})", cpu_condition(left)?, cpu_condition(right)?),
        })
    }

    /// The CPU operator's formation launch of `definition`
    /// (`NativeFormation`): `<entry>_form`, one work item a part of a row of
    /// every operand's rows in turn (a window's rows alone), each operand's
    /// values its prologue (its parameter's rows staged, rounded to the
    /// activation element `A`, or formed under its norm: `form_input`; or
    /// the entry's function's whole row), its forms written by
    /// `projection::form`. The entry's source writes no stage launch.
    fn render_cpu_formation(
        out: &mut String,
        entry: &EntryInfo,
        cpu: &CpuAsset<'_>,
        definition: &NativeImplementation,
        formation: &seismic_lang::checked::NativeFormation,
        elements: &CpuElements,
    ) -> Result<(), BuildError> {
        use seismic_lang::checked::{NativeFormedValues, NativeOperand};
        let refuse = |reason: String| BuildError::CpuNative { path: cpu.path.clone(), reason };
        let nat = |expression: &seismic_lang::checked::NativeNatExpr| cpu_nat(expression).ok_or_else(|| refuse(format!("a formation of `{}` reads an element or the device in its shape", entry.name)));
        let kernel = &definition.launches[formation.launch as usize].kernel;
        // Each operand's source rows and groups: it forms `groups` runs of
        // each source row, so `sources * groups` rows (group-major), which
        // the launch's grid and the operand's buffers hold.
        let mut shapes = Vec::new();
        for operand in &formation.operands {
            shapes.push((format!("({} as usize)", nat(&operand.rows)?), format!("({} as usize)", nat(&operand.groups)?)));
        }
        out.push_str(&format!(
            "    #[inline(always)]\n    fn {kernel}<L: Isa, E: Elements>(_l: L, cx: &Context<'_, E>, group: [u64; 3], shared: &mut [u8]) {{\n      use lib::projection::projection as operator;\n      let total = {};\n      let mut first = 0usize;\n",
            shapes.iter().map(|(sources, groups)| format!("{sources} * {groups}")).collect::<Vec<_>>().join(" + ")
        ));
        for (operand, (sources, groups)) in formation.operands.iter().zip(&shapes) {
            let formed = cpu_condition(&operand.formed).ok_or_else(|| refuse(format!("a formation of `{}` reads an element or the device in `formed`", entry.name)))?;
            let tiles = cpu_condition(&operand.tiles).ok_or_else(|| refuse(format!("a formation of `{}` reads an element or the device in `tiles`", entry.name)))?;
            let weight_binds = operand
                .weights
                .iter()
                .enumerate()
                .map(|(index, weight)| format!("let weight_{index} = cx.{}();", cpu_argument(entry, *weight as usize)))
                .collect::<Vec<_>>()
                .join(" ");
            let weight_refs = (0..operand.weights.len())
                .map(|index| format!("&weight_{index} as &dyn operator::Recipe"))
                .collect::<Vec<_>>()
                .join(", ");
            let scratch = |ordinal: u32| ident_fragment(&definition.scratch[ordinal as usize].name).to_lowercase();
            out.push_str(&format!(
                "      {{\n        let (sources, width, groups) = ({sources}, {} as usize, {groups});\n        let rows = sources * groups;\n        {weight_binds}\n        let weights: [&dyn operator::Recipe; {}] = [{weight_refs}];\n",
                nat(&operand.width)?,
                operand.weights.len(),
            ));
            // The rows formed (all, or the window's), whether each is formed
            // whole by its first part, the item's private bytes as the
            // values read them, and the values of columns `columns` of row
            // `row` into `out`.
            let (window, whole, shared, values) = match &operand.values {
                // The entry's function computes a whole row.
                NativeFormedValues::Function(function) => {
                    ("0..rows", true, "shared", format!("{function}(cx, row, out, shared)"))
                }
                NativeFormedValues::Parameter { at, within } => {
                    let Some(NativeOperand::Formed { parameter, norm, .. }) = &definition.scratch[operand.staged as usize].operand else {
                        return Err(refuse(format!("the formed rows of a `{}` operand are not a formed operand buffer", entry.name)));
                    };
                    if !elements.dense.iter().any(|(name, _)| name == "A") {
                        return Err(refuse(format!("`{}` forms operands but has no activation element `A`", entry.name)));
                    }
                    let ParameterSummaryKind::Tensor { rank, element, .. } = &entry.parameters[*parameter as usize].kind else {
                        return Err(refuse(format!("a `{}` operand is of no tensor parameter", entry.name)));
                    };
                    let CpuView::Dense(source_element) = cpu_view(element, elements) else {
                        return Err(refuse(format!("a `{}` operand's parameter is not dense rows", entry.name)));
                    };
                    out.push_str(&format!("        let source = cx.{}();\n", cpu_argument(entry, *parameter as usize)));
                    let row = match at {
                        Some(at) => {
                            out.push_str(&format!("        let at = cx.{}();\n", cpu_argument(entry, *at as usize)));
                            "at.get([row]) as usize".to_owned()
                        }
                        // Formed row `row` is run `row / sources` of source
                        // row `row % sources`: row `row % sources * groups +
                        // row / sources` of the source's runs. Only a formed
                        // row reads it (`row < sources * groups`), so
                        // `sources` is not zero.
                        None => "row % sources * groups + row / sources".to_owned(),
                    };
                    let window = match within {
                        Some([first, count]) => {
                            out.push_str(&format!(
                                "        let window = {{\n          let first = usize::try_from(cx.{}().get([0])).expect(\"a window's first row is not negative\");\n          first..first + usize::try_from(cx.{}().get([0])).expect(\"a window's row count is not negative\")\n        }};\n",
                                cpu_argument(entry, *first as usize),
                                cpu_argument(entry, *count as usize),
                            ));
                            "window"
                        }
                        None => "0..rows",
                    };
                    let values = match norm {
                        Some(norm) => {
                            if source_element != "seismic::cpu::F32" || *rank != 2 {
                                return Err(refuse(format!("a `{}` operand formed under a norm is not of `[ROWS, WIDTH] f32` rows", entry.name)));
                            }
                            out.push_str(&format!(
                                "        let (norm, epsilon) = (cx.{}(), cx.{}());\n",
                                cpu_argument(entry, norm.weight as usize),
                                cpu_argument(entry, norm.epsilon as usize)
                            ));
                            format!(
                                "operator::form_input_columns::<E::A>(source.row([{row}, 0]), (norm.k() > 0).then_some(&norm), epsilon, shared, width, columns, out)"
                            )
                        }
                        None if source_element == "E::A" => {
                            format!("operator::flat_values::<{source_element}, {rank}>(&source, {row}, width, columns, out)")
                        }
                        None => format!(
                            "{{ operator::flat_values::<{source_element}, {rank}>(&source, {row}, width, columns, out); operator::publish_row::<E::A>(out) }}"
                        ),
                    };
                    // Only a normed prologue decodes into the item's private bytes.
                    (window, false, if norm.is_some() { "shared" } else { "_shared" }, values)
                }
            };
            let columns = if whole { "_columns" } else { "columns" };
            out.push_str(&format!(
                "        let formed = operator::Formed {{ rows, width, groups, window: {window}, whole: {whole}, staged: cx.scratch_{}(), quantized: {formed}.then(|| cx.scratch_{}()), tiles: {tiles} && operator::tiled(sources, &weights), weights: &weights }};\n        // SAFETY: the declaration's buffers; every item of the launch forms the same operands, each column of a row once.\n        unsafe {{ operator::form(formed, first, total, group, shared, |row, {columns}, out, {shared}| {values}) }};\n        first += rows;\n      }}\n",
                scratch(operand.staged),
                scratch(operand.quantized),
            ));
        }
        out.push_str("      let _ = first;\n    }\n");
        Ok(())
    }

    /// How a CPU kernel views a tensor of one element.
    enum CpuView {
        /// A dense element type (`E::A`, `seismic::cpu::F32`).
        Dense(String),
        /// Raw integer scalars.
        Scalars(&'static str),
        /// A weight operand.
        Weights,
        /// An external source a conversion reads.
        External,
        /// A packed result a conversion writes.
        Packed,
    }

    fn cpu_view(element: &ElementSummary, elements: &CpuElements) -> CpuView {
        match element {
            ElementSummary::Parameter(name) => {
                if elements.dense.iter().any(|(dense, _)| dense == name) {
                    CpuView::Dense(format!("E::{}", rust_ident(name)))
                } else if elements.weights.contains(name) {
                    CpuView::Weights
                } else if elements.external.contains(name) {
                    CpuView::External
                } else if elements.packed.contains(name) {
                    CpuView::Packed
                } else {
                    unreachable!("every element parameter of the entry is bound")
                }
            }
            ElementSummary::Fixed(name) => match name.as_str() {
                "f32" => CpuView::Dense("seismic::cpu::F32".to_owned()),
                "bf16" => CpuView::Dense("seismic::cpu::Bf16".to_owned()),
                "f16" => CpuView::Dense("seismic::cpu::F16".to_owned()),
                "i32" => CpuView::Scalars("i32"),
                "u32" => CpuView::Scalars("u32"),
                "u64" => CpuView::Scalars("u64"),
                "bool" => CpuView::Scalars("u8"),
                // Any other fixed element is a registered packed
                // representation, read through its components.
                _ => CpuView::Weights,
            },
        }
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum RustTokenKind {
        Ident,
        Literal,
        Punct,
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    struct RustToken<'a> {
        kind: RustTokenKind,
        text: &'a str,
    }

    /// Minimal Rust token stream for [`out_of_line`]: skips whitespace and
    /// comments (including nested block comments) and keeps string/char
    /// literals opaque while distinguishing lifetimes from char literals.
    fn rust_tokens(text: &str) -> Vec<RustToken<'_>> {
        let bytes = text.as_bytes();
        let mut out = Vec::new();
        let mut i = 0;
        while i < bytes.len() {
            let start = i;
            match bytes[i] {
                b if b.is_ascii_whitespace() => {
                    i += 1;
                }
                b'/' if bytes.get(i + 1) == Some(&b'/') => {
                    i += 2;
                    while i < bytes.len() && bytes[i] != b'\n' {
                        i += 1;
                    }
                }
                b'/' if bytes.get(i + 1) == Some(&b'*') => {
                    i += 2;
                    let mut depth = 1usize;
                    while i + 1 < bytes.len() && depth > 0 {
                        if bytes[i] == b'/' && bytes[i + 1] == b'*' {
                            depth += 1;
                            i += 2;
                        } else if bytes[i] == b'*' && bytes[i + 1] == b'/' {
                            depth -= 1;
                            i += 2;
                        } else {
                            i += 1;
                        }
                    }
                    if depth > 0 {
                        i = bytes.len();
                    }
                }
                // Raw strings (`r"..."`, `r#"..."#`, `br#"..."#`, `cr#"..."#`) and raw idents (`r#ident`).
                b'r' | b'b' | b'c' => {
                    let raw_pos = if bytes[i] == b'r' {
                        Some(i + 1)
                    } else if bytes.get(i + 1) == Some(&b'r') {
                        Some(i + 2)
                    } else {
                        None
                    };
                    if let Some(p) = raw_pos {
                        let mut hashes = 0usize;
                        while bytes.get(p + hashes) == Some(&b'#') {
                            hashes += 1;
                        }
                        if bytes.get(p + hashes) == Some(&b'"') {
                            i = p + hashes + 1;
                            while i < bytes.len() {
                                if bytes[i] == b'"'
                                    && i + 1 + hashes <= bytes.len()
                                    && bytes[i + 1..i + 1 + hashes].iter().all(|&b| b == b'#')
                                {
                                    i += 1 + hashes;
                                    break;
                                }
                                i += 1;
                            }
                            out.push(RustToken {
                                kind: RustTokenKind::Literal,
                                text: &text[start..i],
                            });
                            continue;
                        } else if bytes[i] == b'r'
                            && hashes == 1
                            && bytes
                                .get(p + 1)
                                .is_some_and(|&b| b == b'_' || b.is_ascii_alphabetic())
                        {
                            i = p + 2;
                            while i < bytes.len()
                                && (bytes[i] == b'_' || bytes[i].is_ascii_alphanumeric())
                            {
                                i += 1;
                            }
                            out.push(RustToken {
                                kind: RustTokenKind::Ident,
                                text: &text[start..i],
                            });
                            continue;
                        }
                    }
                    // Byte/C string literals (`b"..."`, `c"..."`) and byte char literals (`b'...'`).
                    if (bytes[i] == b'b' || bytes[i] == b'c') && bytes.get(i + 1) == Some(&b'"') {
                        i += 2;
                        while i < bytes.len() {
                            if bytes[i] == b'\\' {
                                i = (i + 2).min(bytes.len());
                            } else if bytes[i] == b'"' {
                                i += 1;
                                break;
                            } else {
                                i += 1;
                            }
                        }
                        out.push(RustToken {
                            kind: RustTokenKind::Literal,
                            text: &text[start..i],
                        });
                    } else if bytes[i] == b'b' && bytes.get(i + 1) == Some(&b'\'') {
                        i += 2;
                        while i < bytes.len() {
                            if bytes[i] == b'\\' {
                                i = (i + 2).min(bytes.len());
                            } else if bytes[i] == b'\'' {
                                i += 1;
                                break;
                            } else {
                                i += 1;
                            }
                        }
                        out.push(RustToken {
                            kind: RustTokenKind::Literal,
                            text: &text[start..i],
                        });
                    } else {
                        i += 1;
                        while i < bytes.len()
                            && (bytes[i] == b'_' || bytes[i].is_ascii_alphanumeric())
                        {
                            i += 1;
                        }
                        out.push(RustToken {
                            kind: RustTokenKind::Ident,
                            text: &text[start..i],
                        });
                    }
                }
                b'"' => {
                    i += 1;
                    while i < bytes.len() {
                        if bytes[i] == b'\\' {
                            i = (i + 2).min(bytes.len());
                        } else if bytes[i] == b'"' {
                            i += 1;
                            break;
                        } else {
                            i += 1;
                        }
                    }
                    out.push(RustToken {
                        kind: RustTokenKind::Literal,
                        text: &text[start..i],
                    });
                }
                b'\'' => {
                    if bytes.get(i + 1) == Some(&b'\\') {
                        i += 1;
                        while i < bytes.len() {
                            if bytes[i] == b'\\' {
                                i = (i + 2).min(bytes.len());
                            } else if bytes[i] == b'\'' {
                                i += 1;
                                break;
                            } else {
                                i += 1;
                            }
                        }
                        out.push(RustToken {
                            kind: RustTokenKind::Literal,
                            text: &text[start..i],
                        });
                    } else if let Some(ch) = text[i + 1..].chars().next() {
                        let end = i + 1 + ch.len_utf8();
                        if bytes.get(end) == Some(&b'\'') {
                            i = end + 1;
                            out.push(RustToken {
                                kind: RustTokenKind::Literal,
                                text: &text[start..i],
                            });
                        } else {
                            // Lifetime tick `'a`, `'static`, `'_`.
                            i += 1;
                            out.push(RustToken {
                                kind: RustTokenKind::Punct,
                                text: &text[start..i],
                            });
                        }
                    } else {
                        i += 1;
                        out.push(RustToken {
                            kind: RustTokenKind::Punct,
                            text: &text[start..i],
                        });
                    }
                }
                b if b == b'_' || b.is_ascii_alphabetic() => {
                    i += 1;
                    while i < bytes.len()
                        && (bytes[i] == b'_' || bytes[i].is_ascii_alphanumeric())
                    {
                        i += 1;
                    }
                    out.push(RustToken {
                        kind: RustTokenKind::Ident,
                        text: &text[start..i],
                    });
                }
                b if b.is_ascii_digit() => {
                    i += 1;
                    while i < bytes.len()
                        && (bytes[i] == b'_' || bytes[i].is_ascii_alphanumeric())
                    {
                        i += 1;
                    }
                    out.push(RustToken {
                        kind: RustTokenKind::Literal,
                        text: &text[start..i],
                    });
                }
                _ => {
                    i += text[i..].chars().next().map_or(1, char::len_utf8);
                    out.push(RustToken {
                        kind: RustTokenKind::Punct,
                        text: &text[start..i],
                    });
                }
            }
        }
        out
    }

    /// The first function with a body in the CPU source `text` that is not
    /// `#[inline(always)]`. Each form of a CPU entry is a `#[target_feature]`
    /// function of its tier, into which the authored kernel and everything it
    /// calls must inline: a function compiled out of line is compiled once,
    /// without the tier's features, and so are the vector intrinsics it
    /// reaches. Tests and functions with their own target features are exempt.
    fn out_of_line(text: &str) -> Option<&str> {
        let toks = rust_tokens(text);
        for (i, tok) in toks.iter().enumerate() {
            if tok.kind != RustTokenKind::Ident || tok.text != "fn" {
                continue;
            }
            let Some(name_tok) = toks.get(i + 1) else {
                continue;
            };
            if name_tok.kind != RustTokenKind::Ident {
                continue;
            }
            let name = name_tok.text;

            // A trait's required method or extern declaration ends at `;`
            // (outside `(...)` and `[...]`) before any body `{`.
            let mut parens = 0i32;
            let mut brackets = 0i32;
            let has_body = toks[i + 2..].iter().find_map(|t| {
                if t.kind == RustTokenKind::Punct {
                    match t.text {
                        "(" => parens += 1,
                        ")" => parens -= 1,
                        "[" => brackets += 1,
                        "]" => brackets -= 1,
                        "{" if parens == 0 && brackets == 0 => return Some(true),
                        ";" if parens == 0 && brackets == 0 => return Some(false),
                        _ => {}
                    }
                }
                None
            });
            if has_body != Some(true) {
                continue;
            }

            // Walk backward past arbitrary `fn` qualifiers:
            // `pub`, `pub(...)`, `const`, `async`, `unsafe`, `default`, `extern`, `extern "ABI"`.
            let mut pos = i;
            loop {
                if pos == 0 {
                    break;
                }
                let prev = toks[pos - 1];
                if prev.kind == RustTokenKind::Ident
                    && matches!(
                        prev.text,
                        "pub" | "const" | "async" | "unsafe" | "default" | "extern"
                    )
                {
                    pos -= 1;
                } else if prev.kind == RustTokenKind::Literal
                    && pos >= 2
                    && toks[pos - 2].kind == RustTokenKind::Ident
                    && toks[pos - 2].text == "extern"
                {
                    pos -= 2;
                } else if prev.kind == RustTokenKind::Punct && prev.text == ")" {
                    let mut depth = 0usize;
                    let open = toks[..pos].iter().enumerate().rev().find_map(|(k, t)| {
                        if t.kind == RustTokenKind::Punct {
                            match t.text {
                                ")" => {
                                    depth += 1;
                                    None
                                }
                                "(" if depth == 1 => Some(k),
                                "(" => {
                                    depth -= 1;
                                    None
                                }
                                _ => None,
                            }
                        } else {
                            None
                        }
                    });
                    if let Some(open) = open {
                        if open > 0
                            && toks[open - 1].kind == RustTokenKind::Ident
                            && toks[open - 1].text == "pub"
                        {
                            pos = open - 1;
                            continue;
                        }
                    }
                    break;
                } else {
                    break;
                }
            }

            // Walk backward across consecutive outer attributes `#[...]`.
            let mut exempt = false;
            while pos > 0
                && toks[pos - 1].kind == RustTokenKind::Punct
                && toks[pos - 1].text == "]"
            {
                let mut depth = 0usize;
                let open = toks[..pos].iter().enumerate().rev().find_map(|(k, t)| {
                    if t.kind == RustTokenKind::Punct {
                        match t.text {
                            "]" => {
                                depth += 1;
                                None
                            }
                            "[" if depth == 1 => Some(k),
                            "[" => {
                                depth -= 1;
                                None
                            }
                            _ => None,
                        }
                    } else {
                        None
                    }
                });
                let Some(open) = open else {
                    break;
                };
                if open == 0
                    || toks[open - 1].kind != RustTokenKind::Punct
                    || toks[open - 1].text != "#"
                {
                    break;
                }
                let attr = &toks[open + 1..pos - 1];
                let is_inline_always = attr.len() == 4
                    && attr[0]
                        == RustToken {
                            kind: RustTokenKind::Ident,
                            text: "inline",
                        }
                    && attr[1]
                        == RustToken {
                            kind: RustTokenKind::Punct,
                            text: "(",
                        }
                    && attr[2]
                        == RustToken {
                            kind: RustTokenKind::Ident,
                            text: "always",
                        }
                    && attr[3]
                        == RustToken {
                            kind: RustTokenKind::Punct,
                            text: ")",
                        };
                let is_test = attr.len() == 1
                    && attr[0]
                        == RustToken {
                            kind: RustTokenKind::Ident,
                            text: "test",
                        };
                let is_target_feature = attr.first().is_some_and(|t| {
                    *t == RustToken {
                        kind: RustTokenKind::Ident,
                        text: "target_feature",
                    }
                });
                if is_inline_always || is_test || is_target_feature {
                    exempt = true;
                    break;
                }
                pos = open - 1;
            }

            if !exempt {
                return Some(name);
            }
        }
        None
    }

    fn out_of_line_reason(function: &str) -> String {
        format!(
            "`{function}` is not declared `#[inline(always)]`: it must inline into its tier boundary, or it is compiled without the tier's features"
        )
    }

    /// The typed CPU view of an entry's native ABI, the authored source, and
    /// per launch one function per instruction-set tier and dense element
    /// binding. Tuning parameters are runtime values.
    fn render_cpu_native(
        out: &mut String,
        entry: &EntryInfo,
        cpu: &CpuAsset<'_>,
        library: &CpuLibrary,
    ) -> Result<(), BuildError> {
        let definition = cpu.definition;
        let elements = cpu_elements(entry, definition);
        out.push_str("  pub mod cpu_native {\n");
        out.push_str(
            "    #![allow(dead_code, non_camel_case_types, unused_imports, clippy::all)]\n",
        );
        out.push_str("    use seismic::native_cpu::{CpuInvocation, CpuKernelFn, CpuLaunchVariants, CpuNativeKernels, CpuVariant};\n");
        out.push_str("    use seismic::cpu::{Dense, Element, Isa};\n");
        // The asset's own directory: its library neighbours are in scope by
        // name, so its paths are relative to its file.
        let directory = library
            .roots
            .iter()
            .filter_map(|root| {
                cpu.path
                    .parent()
                    .and_then(|parent| parent.strip_prefix(root).ok())
            })
            .min_by_key(|relative| relative.components().count())
            .map(|relative| {
                relative
                    .components()
                    .map(|component| component.as_os_str().to_string_lossy().into_owned())
                    .collect::<Vec<_>>()
            });
        if let Some(directory) = directory {
            let mut neighbours = library
                .files
                .iter()
                .filter(|(modules, _)| {
                    modules.len() > directory.len() && modules[..directory.len()] == directory[..]
                })
                .map(|(modules, _)| modules[directory.len()].clone())
                .collect::<Vec<_>>();
            neighbours.dedup();
            if !neighbours.is_empty() {
                let path = std::iter::once("cpu_library".to_owned())
                    .chain(directory)
                    .collect::<Vec<_>>()
                    .join("::");
                out.push_str(&format!(
                    "    use super::super::{path}::{{{}}};\n",
                    neighbours.join(", ")
                ));
            }
        }
        out.push_str(&format!(
            "    /// The dense element binding of `{}` a CPU form is compiled for.\n",
            entry.name
        ));
        out.push_str("    pub trait Elements: 'static {\n");
        for (element, dtypes) in &elements.dense {
            // Floating coverage converts through `f32`; wider coverage only
            // moves storage.
            let bound = if dtypes.iter().all(|dtype| dtype.is_float()) {
                "Dense"
            } else {
                "Element"
            };
            out.push_str(&format!("      type {}: {bound};\n", rust_ident(element)));
        }
        out.push_str("    }\n");
        out.push_str(&format!(
            "    /// Typed view of the CPU native ABI of `{}`.\n",
            entry.name
        ));
        out.push_str("    pub struct Context<'a, E: Elements> { invocation: &'a CpuInvocation<'a>, elements: std::marker::PhantomData<E> }\n");
        out.push_str("    impl<E: Elements> Clone for Context<'_, E> { fn clone(&self) -> Self { *self } }\n");
        out.push_str("    impl<E: Elements> Copy for Context<'_, E> {}\n");
        out.push_str("    impl<'a, E: Elements> Context<'a, E> {\n");
        out.push_str("      fn new(invocation: &'a CpuInvocation<'a>) -> Self { Self { invocation, elements: std::marker::PhantomData } }\n");
        out.push_str("      pub fn groups(&self) -> [u64; 3] { self.invocation.groups() }\n");
        out.push_str("      pub fn threads(&self) -> [u64; 3] { self.invocation.threads() }\n");
        let mut word = 0usize;
        for dimension in &entry.dimensions {
            out.push_str(&format!(
                "      pub fn dim_{}(&self) -> u64 {{ self.invocation.word({word}) }}\n",
                ident_fragment(dimension).to_lowercase()
            ));
            word += 1;
        }
        // A record port's accessor gives its records (`Records`), each the
        // port's view without its leading axis at its record's address; the
        // launches of a CPU form share one context, so they bind the same.
        let records = &definition.records;
        if (0..definition.launches.len()).any(|launch| definition.launch_records(launch) != records.as_slice()) {
            return Err(BuildError::CpuNative {
                path: cpu.path.clone(),
                reason: format!("the launches of `{}` share one context, so they list the same record ports", entry.name),
            });
        }
        let mut buffer = 0usize;
        let tensor = |out: &mut String,
                      name: &str,
                      view: CpuView,
                      buffer: usize,
                      word: usize,
                      rank: usize,
                      record: bool|
         -> Result<(), BuildError> {
            let axes = |from: usize, first: usize| {
                (first..rank)
                    .map(|axis| format!("self.invocation.word({})", from + axis))
                    .collect::<Vec<_>>()
                    .join(", ")
            };
            let first = usize::from(record);
            let (extents, strides) = (axes(word, first), axes(word + rank, first));
            let (base, constant) = match record {
                true => ("std::ptr::null_mut()".to_owned(), "std::ptr::null()".to_owned()),
                false => (format!("self.invocation.buffer({buffer})"), format!("self.invocation.buffer({buffer})")),
            };
            let view_rank = rank - first;
            let (ty, view) = match view {
                CpuView::Dense(element) => (
                    format!("seismic::cpu::Tensor<'a, {element}, {view_rank}>"),
                    format!("seismic::cpu::Tensor::from_raw({base}, [{extents}], [{strides}])"),
                ),
                CpuView::Scalars(scalar) => (
                    format!("seismic::cpu::Scalars<'a, {scalar}, {view_rank}>"),
                    format!("seismic::cpu::Scalars::from_raw({base}, [{extents}], [{strides}])"),
                ),
                CpuView::Weights => (
                    "seismic::cpu::Weights<'a>".to_owned(),
                    format!("seismic::cpu::Weights::from_tensor({constant}, &[{extents}], &[{strides}], self.invocation.weights({buffer}))"),
                ),
                CpuView::External | CpuView::Packed if record => {
                    return Err(BuildError::CpuNative {
                        path: cpu.path.clone(),
                        reason: format!("record port `{name}` of `{}`: a CPU record is a dense, scalar or weight view", entry.name),
                    });
                }
                CpuView::External => (
                    "seismic::cpu::External<'a>".to_owned(),
                    format!("seismic::cpu::External::from_tensor({constant}, &[{extents}], &[{strides}], self.invocation.representation({buffer}))"),
                ),
                CpuView::Packed => (
                    "seismic::cpu::Packed<'a>".to_owned(),
                    format!("seismic::cpu::Packed::from_tensor({base}, &[{extents}], &[{strides}], self.invocation.representation({buffer}))"),
                ),
            };
            let safety = "// SAFETY: the runtime binds every view inside its allocation for the call, and a record port as its records' table.";
            out.push_str(&match record {
                true => format!(
                    "      pub fn {name}(&self) -> seismic::cpu::slab::Records<{ty}> {{ {safety}\n        unsafe {{ seismic::cpu::slab::Records::from_table(self.invocation.buffer({buffer}), self.invocation.word({word}), {view}) }} }}\n"
                ),
                false => format!("      pub fn {name}(&self) -> {ty} {{ {safety}\n        unsafe {{ {view} }} }}\n"),
            });
            Ok(())
        };
        for (ordinal, parameter) in entry.parameters.iter().enumerate() {
            let name = cpu_argument(entry, ordinal);
            match &parameter.kind {
                // A host port is in the words but bound to no buffer
                // (`abi::bound`): the kernel never accesses it.
                ParameterSummaryKind::Tensor {
                    access: TensorAccess::Host { .. },
                    rank,
                    ..
                } => {
                    word += *rank as usize * 2;
                }
                ParameterSummaryKind::Tensor { rank, element, .. } => {
                    let rank = *rank as usize;
                    let record = records.contains(&RecordPort::Parameter(parameter.name.clone()));
                    tensor(out, &name, cpu_view(element, &elements), buffer, word, rank, record)?;
                    buffer += 1;
                    word += rank * 2;
                }
                ParameterSummaryKind::Scalar(dtype) => {
                    let (ty, decode) = match dtype {
                        DType::F32 => ("f32", "f32::from_bits(value as u32)"),
                        DType::F16 | DType::BF16 => ("u16", "value as u16"),
                        DType::I32 => ("i32", "value as u32 as i32"),
                        DType::U32 => ("u32", "value as u32"),
                        DType::U64 => ("u64", "value"),
                        DType::Bool => ("bool", "value != 0"),
                    };
                    out.push_str(&format!(
                        "      pub fn {name}(&self) -> {ty} {{ let value = self.invocation.word({word}); {decode} }}\n"
                    ));
                    word += 1;
                }
                ParameterSummaryKind::Index => {
                    out.push_str(&format!(
                        "      pub fn {name}(&self) -> u64 {{ self.invocation.word({word}) }}\n"
                    ));
                    word += 1;
                }
                ParameterSummaryKind::Range => {
                    out.push_str(&format!(
                        "      pub fn {name}(&self) -> (u64, u64) {{ (self.invocation.word({word}), self.invocation.word({})) }}\n",
                        word + 1
                    ));
                    word += 2;
                }
            }
        }
        let mut scalar = 0usize;
        for (ordinal, result) in entry.results.iter().enumerate() {
            match &result.kind {
                ResultSummaryKind::Tensor { rank, element } => {
                    let rank = *rank as usize;
                    let view = cpu_view(element, &elements);
                    if matches!(view, CpuView::Weights | CpuView::External) {
                        return Err(BuildError::CpuNative {
                            path: cpu.path.clone(),
                            reason: format!(
                                "result {ordinal} of `{}` is read-only storage; CPU kernels write dense or converted results",
                                entry.name
                            ),
                        });
                    }
                    let record = records.contains(&RecordPort::Result(ordinal as u32));
                    tensor(out, &format!("result_{ordinal}"), view, buffer, word, rank, record)?;
                    buffer += 1;
                    word += rank * 2;
                }
                ResultSummaryKind::Scalar(_) | ResultSummaryKind::Index => {
                    out.push_str(&format!(
                        "      /// Raw word of scalar result {ordinal}; one work item writes it.\n      pub fn result_{ordinal}(&self) -> *mut u64 {{ self.invocation.scalar_result({scalar}) }}\n"
                    ));
                    scalar += 1;
                }
                ResultSummaryKind::Range => {
                    out.push_str(&format!(
                        "      /// Raw words (start, end) of range result {ordinal}; one work item writes them.\n      pub fn result_{ordinal}(&self) -> (*mut u64, *mut u64) {{ (self.invocation.scalar_result({scalar}), self.invocation.scalar_result({})) }}\n",
                        scalar + 1
                    ));
                    scalar += 2;
                }
            }
        }
        // One word per emitted operand buffer, after the schema's words, in
        // buffer order (`abi::emitted_words`), set when a graph is sealed
        // from the forms of the consumer it pairs the buffer with: bit 0,
        // the call emits it; on the rows, bit 1, as NVFP4 operand rows.
        for scratch in &definition.scratch {
            if let Some(seismic_lang::checked::NativeOperand::Emitted { result, rows, .. }) = &scratch.operand {
                let name = ident_fragment(&scratch.name).to_lowercase();
                out.push_str(&format!(
                    "      /// Whether the call emits result {result}'s operand into `scratch_{name}`: a sealed graph pairs it.\n      pub fn emits_{name}(&self) -> bool {{ self.invocation.word({word}) & 1 != 0 }}\n"
                ));
                if *rows {
                    out.push_str(&format!(
                        "      /// Whether the emitted rows in `scratch_{name}` are NVFP4 operand rows: its consumer's are.\n      pub fn nvfp4_{name}(&self) -> bool {{ self.invocation.word({word}) & 2 != 0 }}\n"
                    ));
                }
                word += 1;
            }
        }
        for scratch in &definition.scratch {
            out.push_str(&format!(
                "      pub fn scratch_{}(&self) -> seismic::cpu::Scratch<'a> {{ // SAFETY: the runtime places every scratch buffer for the call, 256-byte aligned.\n        unsafe {{ seismic::cpu::Scratch::from_raw(self.invocation.buffer({buffer})) }} }}\n",
                ident_fragment(&scratch.name).to_lowercase()
            ));
            buffer += 1;
        }
        for (index, parameter) in definition.declared_params().enumerate() {
            out.push_str(&format!(
                "      /// Tuning parameter `{}`.\n      pub fn param_{}(&self) -> u64 {{ self.invocation.param({index}) }}\n",
                parameter.name,
                ident_fragment(&parameter.name).to_lowercase()
            ));
        }
        out.push_str("    }\n");
        if let Some(formation) = &definition.formation {
            render_cpu_formation(out, entry, cpu, definition, formation, &elements)?;
        }
        out.push_str(&format!(
            "    include!({:?});\n",
            cpu.path.to_string_lossy()
        ));

        // One form per dense element binding: every combination of the
        // dense elements over their covered types.
        let mut bindings: Vec<Vec<(&str, &str)>> = vec![Vec::new()];
        for (_, dtypes) in &elements.dense {
            bindings = bindings
                .into_iter()
                .flat_map(|prefix| {
                    dtypes.iter().map(move |dtype| {
                        let mut binding = prefix.clone();
                        binding.push((dtype.name(), cpu_element_type(*dtype)));
                        binding
                    })
                })
                .collect();
        }
        let binding_name = |binding: &[(&str, &str)]| {
            format!(
                "Elements_{}",
                binding
                    .iter()
                    .map(|(name, _)| *name)
                    .collect::<Vec<_>>()
                    .join("_")
            )
        };
        for binding in &bindings {
            let name = binding_name(binding);
            out.push_str(&format!(
                "    pub struct {name};\n    impl Elements for {name} {{"
            ));
            for ((element, _), (_, ty)) in elements.dense.iter().zip(binding) {
                out.push_str(&format!(" type {} = seismic::cpu::{ty};", rust_ident(element)));
            }
            out.push_str(" }\n");
        }

        // The tier boundaries: each form is a `#[target_feature]` function
        // of its tier. The asset is included from its authored path, so the
        // rule that everything inlines into them is checked here rather than
        // written in. Everything inlining makes each form a whole copy of its
        // kernel, so each is the one item of its own module: rustc partitions
        // codegen units by module, and an entry's forms sharing one would
        // optimize serially in one LLVM job.
        if let Some(function) = out_of_line(cpu.text) {
            return Err(BuildError::CpuNative { path: cpu.path.clone(), reason: out_of_line_reason(function) });
        }
        for tier in seismic_cpu_tiers::Tier::EVERY {
            for (launch_index, launch) in definition.launches.iter().enumerate() {
                for binding in &bindings {
                    let elements = binding_name(binding);
                    out.push_str(&format!(
                        "    #[cfg(target_arch = {arch:?})]\n    mod launch_{launch_index}_{tier}_{elements} {{\n      #[target_feature(enable = {features:?})]\n      pub(super) unsafe fn form(invocation: &seismic::native_cpu::CpuInvocation<'_>, group: [u64; 3], shared: &mut [u8]) {{\n        // SAFETY: the runtime calls a form only on a device whose detected tier includes the form's.\n        let l = unsafe {{ <seismic::cpu::{token} as seismic::cpu::Isa>::assume_detected() }};\n        super::{kernel}::<seismic::cpu::{token}, super::{elements}>(l, &super::Context::new(invocation), group, shared)\n      }}\n    }}\n",
                        arch = tier.arch(),
                        features = tier.features(),
                        tier = tier.name(),
                        token = tier.token(),
                        kernel = launch.kernel,
                    ));
                }
            }
        }

        let mut digest = Sha256::new();
        digest.update(cpu.text.as_bytes());
        // The generated formation is code of the form as its source is.
        digest.update(format!("{:?}", definition.formation).as_bytes());
        digest.update(library.digest);
        digest.update(seismic_cpu_tiers::VERSION.as_bytes());
        let digest = hex(&digest.finalize().into());
        let element_names = elements
            .dense
            .iter()
            .map(|(element, _)| format!("{element:?}"))
            .collect::<Vec<_>>()
            .join(", ");
        for arch in ["x86_64", "aarch64"] {
            out.push_str(&format!("    #[cfg(target_arch = {arch:?})]\n"));
            out.push_str(&format!(
                "    pub static KERNELS: CpuNativeKernels = CpuNativeKernels {{ digest: {digest:?}, elements: &[{element_names}], launches: &[\n"
            ));
            for (launch_index, launch) in definition.launches.iter().enumerate() {
                out.push_str(&format!(
                    "      CpuLaunchVariants {{ kernel: {:?}, variants: &[\n",
                    launch.kernel
                ));
                for tier in seismic_cpu_tiers::Tier::EVERY
                    .into_iter()
                    .filter(|tier| tier.arch() == arch)
                {
                    for binding in &bindings {
                        let names = binding
                            .iter()
                            .map(|(name, _)| format!("{name:?}"))
                            .collect::<Vec<_>>()
                            .join(", ");
                        out.push_str(&format!(
                            "        CpuVariant {{ tier: seismic::cpu::Tier::{token}, elements: &[{names}], function: launch_{launch_index}_{tier}_{elements}::form as CpuKernelFn }},\n",
                            token = tier.token(),
                            tier = tier.name(),
                            elements = binding_name(binding),
                        ));
                    }
                }
                out.push_str("      ] },\n");
            }
            out.push_str("    ] };\n");
        }
        out.push_str("  }\n");
        Ok(())
    }

    fn parameter_type(kind: &ParameterSummaryKind) -> String {
        match kind {
            ParameterSummaryKind::Tensor { access, .. } => match access {
                TensorAccess::Owned => "seismic::Tensor".to_owned(),
                TensorAccess::Shared | TensorAccess::Host { .. } => "&'a seismic::Tensor".to_owned(),
                TensorAccess::Mutable => "&'a mut seismic::Tensor".to_owned(),
            },
            ParameterSummaryKind::Scalar(dtype) => scalar_type(*dtype).to_owned(),
            ParameterSummaryKind::Index => "seismic::BigUint".to_owned(),
            ParameterSummaryKind::Range => "(seismic::BigUint, seismic::BigUint)".to_owned(),
        }
    }

    fn workflow_parameter_type(kind: &ParameterSummaryKind) -> String {
        match kind {
            ParameterSummaryKind::Tensor { access, .. } => match access {
                TensorAccess::Owned => "seismic::WorkflowTensorOwned<'a>".to_owned(),
                TensorAccess::Shared | TensorAccess::Host { .. } => {
                    "seismic::WorkflowTensorRef<'a>".to_owned()
                }
                TensorAccess::Mutable => "seismic::WorkflowTensorMut<'a>".to_owned(),
            },
            ParameterSummaryKind::Scalar(dtype) => format!("seismic::Argument<{}>", scalar_type(*dtype)),
            ParameterSummaryKind::Index => "seismic::BigUint".to_owned(),
            ParameterSummaryKind::Range => "(seismic::BigUint, seismic::BigUint)".to_owned(),
        }
    }

    fn parameter_sources(parameters: &[ParameterSummary]) -> Vec<Vec<&ParameterSummary>> {
        let mut sources: Vec<Vec<&ParameterSummary>> = Vec::new();
        for parameter in parameters {
            match sources.last_mut() {
                Some(source) if source[0].source == parameter.source => source.push(parameter),
                _ => sources.push(vec![parameter]),
            }
        }
        for (ordinal, source) in sources.iter().enumerate() {
            assert_eq!(
                source[0].source as usize, ordinal,
                "checked entry parameter sources are not canonical and contiguous"
            );
            assert!(
                source.iter().all(|leaf| leaf.name == source[0].name),
                "tuple parameter leaves disagree on their authored name"
            );
        }
        sources
    }

    fn source_parameter_type(parameters: &[&ParameterSummary], depth: usize) -> String {
        if let Some(parameter) = parameters
            .iter()
            .copied()
            .find(|parameter| parameter.path.len() == depth)
        {
            assert_eq!(
                parameters.len(),
                1,
                "a checked parameter path is both a leaf and a tuple prefix"
            );
            return parameter_type(&parameter.kind);
        }
        let mut children: Vec<Vec<&ParameterSummary>> = Vec::new();
        for parameter in parameters {
            let child = *parameter
                .path
                .get(depth)
                .unwrap_or_else(|| panic!("checked tuple parameter has an incomplete path"));
            let child = usize::try_from(child).expect("tuple child ordinal exceeds usize");
            while children.len() <= child {
                children.push(Vec::new());
            }
            children[child].push(*parameter);
        }
        assert!(
            children.len() >= 2 && children.iter().all(|child| !child.is_empty()),
            "checked tuple parameter children are not canonical and contiguous"
        );
        let fields = children
            .iter()
            .map(|child| source_parameter_type(child, depth + 1))
            .collect::<Vec<_>>();
        format!("({})", fields.join(", "))
    }

    fn source_workflow_parameter_type(parameters: &[&ParameterSummary], depth: usize) -> String {
        if let Some(parameter) = parameters
            .iter()
            .copied()
            .find(|parameter| parameter.path.len() == depth)
        {
            assert_eq!(
                parameters.len(),
                1,
                "a checked parameter path is both a leaf and a tuple prefix"
            );
            return workflow_parameter_type(&parameter.kind);
        }
        let mut children: Vec<Vec<&ParameterSummary>> = Vec::new();
        for parameter in parameters {
            let child = *parameter
                .path
                .get(depth)
                .unwrap_or_else(|| panic!("checked tuple parameter has an incomplete path"));
            let child = usize::try_from(child).expect("tuple child ordinal exceeds usize");
            while children.len() <= child {
                children.push(Vec::new());
            }
            children[child].push(*parameter);
        }
        assert!(
            children.len() >= 2 && children.iter().all(|child| !child.is_empty()),
            "checked tuple parameter children are not canonical and contiguous"
        );
        let fields = children
            .iter()
            .map(|child| source_workflow_parameter_type(child, depth + 1))
            .collect::<Vec<_>>();
        format!("({})", fields.join(", "))
    }

    fn parameter_access(field: &str, path: &[u32]) -> String {
        let mut value = format!("args.{field}");
        for child in path {
            value.push_str(&format!(".{child}"));
        }
        value
    }

    fn result_type(kind: &ResultSummaryKind) -> &'static str {
        match kind {
            ResultSummaryKind::Tensor { .. } => "seismic::Tensor",
            ResultSummaryKind::Scalar(dtype) => scalar_type(*dtype),
            ResultSummaryKind::Index => "seismic::BigUint",
            ResultSummaryKind::Range => "(seismic::BigUint, seismic::BigUint)",
        }
    }

    fn workflow_result_type(kind: &ResultSummaryKind) -> String {
        match kind {
            ResultSummaryKind::Tensor { .. } => "seismic::WorkflowTensor".to_owned(),
            ResultSummaryKind::Scalar(dtype) => {
                format!("seismic::WorkflowScalar<{}>", scalar_type(*dtype))
            }
            ResultSummaryKind::Index => "seismic::WorkflowScalar<seismic::BigUint>".to_owned(),
            ResultSummaryKind::Range => {
                "seismic::WorkflowScalar<(seismic::BigUint, seismic::BigUint)>".to_owned()
            }
        }
    }

    fn render_invocation_scope(out: &mut String, entry: &EntryInfo) {
        out.push_str("  pub struct OptimizeFor { scope: seismic::InvocationScope }\n");
        out.push_str("  pub fn optimize_for() -> Result<OptimizeFor, seismic::CheckedBundleError> { Ok(OptimizeFor { scope: seismic::generated::invocation_scope::<Entry>()? }) }\n");
        out.push_str("  impl OptimizeFor {\n");
        for (ordinal, name) in entry.dimensions.iter().enumerate() {
            render_range(
                out,
                &format!("dimension_{}", ident_fragment(name)),
                "seismic::BigUint",
                "Nat",
                false,
                "Dimension",
                ordinal,
            );
        }
        for (ordinal, parameter) in entry.parameters.iter().enumerate() {
            let suffix = if parameter.path.is_empty() {
                String::new()
            } else {
                format!(
                    "_{}",
                    parameter
                        .path
                        .iter()
                        .map(u32::to_string)
                        .collect::<Vec<_>>()
                        .join("_")
                )
            };
            let name = format!("{}_{}{}", "parameter", ident_fragment(&parameter.name), suffix);
            match parameter.kind {
                ParameterSummaryKind::Tensor { .. } => {}
                ParameterSummaryKind::Scalar(dtype) => {
                    let (variant, bits) = match dtype {
                        DType::F32 => ("F32", false),
                        DType::F16 => ("F16", true),
                        DType::BF16 => ("BF16", true),
                        DType::I32 => ("I32", false),
                        DType::U32 => ("U32", false),
                        DType::U64 => ("U64", false),
                        DType::Bool => ("Bool", false),
                    };
                    render_range(
                        out,
                        &name,
                        scalar_type(dtype),
                        variant,
                        bits,
                        "Scalar",
                        ordinal,
                    );
                }
                ParameterSummaryKind::Index => render_range(
                    out,
                    &name,
                    "seismic::BigUint",
                    "Nat",
                    false,
                    "Scalar",
                    ordinal,
                ),
                ParameterSummaryKind::Range => {
                    render_range(
                        out,
                        &format!("{name}_start"),
                        "seismic::BigUint",
                        "Nat",
                        false,
                        "RangeStart",
                        ordinal,
                    );
                    render_range(
                        out,
                        &format!("{name}_end"),
                        "seismic::BigUint",
                        "Nat",
                        false,
                        "RangeEnd",
                        ordinal,
                    );
                }
            }
        }
        out.push_str("    pub fn finish(self) -> seismic::InvocationScope { self.scope }\n  }\n");
    }

    fn render_range(
        out: &mut String,
        name: &str,
        ty: &str,
        variant: &str,
        bits: bool,
        parameter: &str,
        ordinal: usize,
    ) {
        let bits = if bits { ".to_bits()" } else { "" };
        out.push_str(&format!("    #[allow(non_snake_case)] pub fn {name}(mut self, values: impl Into<seismic::InvocationRange<{ty}>>) -> Self {{ let values = values.into(); self.scope.constrain(seismic::generated::InvocationParameter::{parameter}({ordinal}), seismic::generated::SymbolValue::{variant}(values.lower{bits}), seismic::generated::SymbolValue::{variant}(values.upper{bits})); self }}\n"));
    }

    fn scalar_type(dtype: DType) -> &'static str {
        match dtype {
            DType::F32 => "f32",
            DType::F16 => "seismic::F16",
            DType::BF16 => "seismic::BF16",
            DType::I32 => "i32",
            DType::U32 => "u32",
            DType::U64 => "u64",
            DType::Bool => "bool",
        }
    }

    fn result_name(path: &[u32]) -> String {
        if path.is_empty() {
            "value".to_owned()
        } else {
            format!(
                "r{}",
                path.iter()
                    .map(u32::to_string)
                    .collect::<Vec<_>>()
                    .join("_")
            )
        }
    }

    fn validate_identifier(value: &str) -> Result<(), BuildError> {
        if rust_ident(value) != value || value.is_empty() {
            return Err(BuildError::Environment(format!(
                "`{value}` is not a Rust module identifier"
            )));
        }
        Ok(())
    }

    /// Sanitized ASCII alphanumeric/underscore fragment suitable for prefixed
    /// Rust identifiers (`arg_<fragment>`, `dim_<fragment>`, `param_<fragment>`,
    /// etc.). Never emits `r#`.
    fn ident_fragment(value: &str) -> String {
        let mut result = String::with_capacity(value.len());
        for character in value.chars() {
            if character == '_' || character.is_ascii_alphanumeric() {
                result.push(character);
            } else {
                result.push('_');
            }
        }
        if result.is_empty() {
            result.push('_');
        }
        result
    }

    /// Standalone Rust identifier: escapes keywords that Rust allows as raw
    /// identifiers with `r#`, and suffixes `_` for `self`/`Self`/`super`/`crate`
    /// (which Rust forbids as raw identifiers).
    fn rust_ident(value: &str) -> String {
        let mut result = String::with_capacity(value.len());
        for (index, character) in value.chars().enumerate() {
            if character == '_'
                || character.is_ascii_alphabetic()
                || (index > 0 && character.is_ascii_digit())
            {
                result.push(character);
            } else {
                result.push('_');
            }
        }
        if result.is_empty() || result.as_bytes()[0].is_ascii_digit() {
            result.insert(0, '_');
        }
        match result.as_str() {
            "self" | "Self" | "super" | "crate" => {
                result.push('_');
            }
            "as" | "break" | "const" | "continue" | "else" | "enum" | "extern" | "false"
            | "fn" | "for" | "if" | "impl" | "in" | "let" | "loop" | "match" | "mod" | "move"
            | "mut" | "pub" | "ref" | "return" | "static" | "struct" | "trait" | "true"
            | "type" | "unsafe" | "use" | "where" | "while" | "async" | "await" | "dyn"
            | "abstract" | "become" | "box" | "do" | "final" | "macro" | "override" | "priv"
            | "try" | "typeof" | "unsized" | "virtual" | "yield" | "gen" => {
                result.insert_str(0, "r#");
            }
            _ => {}
        }
        result
    }

    fn hex(bytes: &[u8; 32]) -> String {
        const DIGITS: &[u8; 16] = b"0123456789abcdef";
        let mut output = String::with_capacity(64);
        for byte in bytes {
            output.push(DIGITS[(byte >> 4) as usize] as char);
            output.push(DIGITS[(byte & 0xf) as usize] as char);
        }
        output
    }

    /// The device library every source of `dialect` is formed with, as the
    /// runtime pastes it into the source's prefix. Metal has none: its one
    /// library, the resident grid barrier, is pasted only beside a resident
    /// launch, whose names are defined with it.
    fn device_library(dialect: SourceDialect) -> &'static str {
        match dialect {
            SourceDialect::Cuda => include_str!("../../runtime/src/native/cuda_prelude.cuh"),
            SourceDialect::Vulkan => include_str!("../../runtime/src/native/vulkan_prelude.glsl"),
            SourceDialect::Metal => "",
        }
    }

    /// The ABI names `dialect`'s device library defines (each `#define
    /// SEISMIC_*` in it): read from the library, its one source.
    fn library_symbols(dialect: SourceDialect) -> impl Iterator<Item = String> {
        device_library(dialect).lines().filter_map(|line| {
            let name = line.trim_start().strip_prefix("#define")?.trim_start();
            let name = &name[..name.find(|c: char| !(c.is_ascii_alphanumeric() || c == '_')).unwrap_or(name.len())];
            name.starts_with("SEISMIC_").then(|| name.to_owned())
        })
    }

    #[cfg(test)]
    mod visibility_tests {
        use super::*;

        /// Every symbol of a representation prefix belongs to a family of the
        /// visibility table, and only the bare prefix, the kinds, the decoded
        /// types and the shape are surface.
        #[test]
        fn representation_symbols_take_their_family_visibility() {
            let mut symbols = Symbols::default();
            add_shape_symbols(&mut symbols, "SEISMIC_WEIGHT", 2);
            add_representation_symbols(&mut symbols, "SEISMIC_WEIGHT", registry::representations().iter());
            for (symbol, visibility) in &symbols.0 {
                let suffix = &symbol["SEISMIC_WEIGHT".len()..];
                let surface = suffix.is_empty()
                    || ["_RANK", "_EXTENT_", "_STRIDE_", "_KIND_", "_DECODED_"]
                        .iter()
                        .any(|family| suffix.starts_with(family));
                let expected = if surface { Visibility::Surface } else { Visibility::Library };
                assert_eq!(*visibility, expected, "{symbol}");
            }
            for (symbol, visibility) in [
                ("SEISMIC_WEIGHT", Visibility::Surface),
                ("SEISMIC_WEIGHT_STRIDE_1", Visibility::Surface),
                ("SEISMIC_WEIGHT_KIND_PACKED", Visibility::Surface),
                ("SEISMIC_WEIGHT_DECODED_F32", Visibility::Surface),
                ("SEISMIC_WEIGHT_REPRESENTATION", Visibility::Library),
                ("SEISMIC_WEIGHT_OPERANDS_NVFP4", Visibility::Library),
                ("SEISMIC_WEIGHT_LAYOUT_ROWS16", Visibility::Library),
                ("SEISMIC_WEIGHT_PACKET_SIZE", Visibility::Library),
                ("SEISMIC_WEIGHT_ROW_STRIDE_BYTES", Visibility::Library),
                ("SEISMIC_WEIGHT_TILE_ROWS", Visibility::Library),
            ] {
                assert_eq!(symbols.0.get(symbol), Some(&visibility), "{symbol}");
            }
        }

        /// A conversion's bare prefix is a binding argument; every suffix is
        /// the library's.
        #[test]
        fn conversion_symbols_are_the_librarys_but_the_prefix() {
            let mut symbols = Symbols::default();
            symbols.conversion("SEISMIC_CONVERSION_S_D", "");
            symbols.conversion("SEISMIC_CONVERSION_S_D", "_WORDS");
            symbols.conversion("SEISMIC_CONVERSION_S_D", "_CODES_FIRST");
            assert_eq!(symbols.0["SEISMIC_CONVERSION_S_D"], Visibility::Surface);
            assert_eq!(symbols.0["SEISMIC_CONVERSION_S_D_WORDS"], Visibility::Library);
            assert_eq!(symbols.0["SEISMIC_CONVERSION_S_D_CODES_FIRST"], Visibility::Library);
        }

        #[test]
        fn rust_ident_and_ident_fragment_handle_keywords_and_prefixes() {
            assert_eq!(ident_fragment("type"), "type");
            assert_eq!(ident_fragment("self"), "self");
            assert_eq!(ident_fragment("a-b.0"), "a_b_0");
            assert_eq!(ident_fragment(""), "_");

            assert_eq!(rust_ident("type"), "r#type");
            assert_eq!(rust_ident("match"), "r#match");
            assert_eq!(rust_ident("async"), "r#async");
            assert_eq!(rust_ident("self"), "self_");
            assert_eq!(rust_ident("Self"), "Self_");
            assert_eq!(rust_ident("super"), "super_");
            assert_eq!(rust_ident("crate"), "crate_");
            assert_eq!(rust_ident("0abc"), "_abc");
            assert_eq!(rust_ident("valid_1"), "valid_1");
            assert!(validate_identifier("valid_1").is_ok());
            assert!(validate_identifier("self").is_err());
            assert!(validate_identifier("type").is_err());
            assert!(validate_identifier("").is_err());
        }

        #[test]
        fn vulkan_steering_rule_is_token_driven() {
            let check = |src: &str| {
                let offset = src.find("seismic_steer").expect("contains seismic_steer");
                vulkan_steering_rule(src, "seismic_steer", offset)
            };

            assert!(check("if (seismic_steer(live, p)) { x = 1; }").is_none());
            assert!(check("if (!seismic_steer(live, p)) { x = 1; }").is_none());
            assert!(check("while (seismic_steer(live, p)) { x = 1; }").is_none());
            assert!(check("for (uint i = 0; seismic_steer(live, i < n); ++i) { x = 1; }").is_none());

            // Exact token match: `breaker()` and `continued` are not `break`/`continue`.
            assert!(check("if (seismic_steer(live, p)) { breaker(); }").is_none());
            assert!(check("if (seismic_steer(live, p)) { continued = true; }").is_none());
            assert!(check("if (seismic_steer(live, p)) { /* comment */ break; }").is_some());
            assert!(check("if (seismic_steer(live, p)) continue;").is_some());

            // Comments containing parens or semicolons do not fool the rule.
            assert!(check("/* ( */ if (seismic_steer(live, p /* ) */)) { x = 1; }").is_none());
            assert!(check("for (uint i = 0; /* ; */ seismic_steer(live, i < n); ++i) {}").is_none());

            // Unbalanced call or non-whole condition must be rejected (never silently `None`).
            assert!(check("if (seismic_steer(live, p").is_some());
            assert!(check("if (a && seismic_steer(live, p)) {}").is_some());
            assert!(check("if (seismic_steer(live, p) && a) {}").is_some());
        }

        #[test]
        fn out_of_line_handles_comments_strings_multiline_attrs_and_qualifiers() {
            let ok = r##"
                // fn commented_out() {}
                /* nested /* fn also_commented() {} */ */
                const S: &str = "fn in_string() {}";
                const R: &str = r#"fn in_raw_string() {}"#;
                type Callback = unsafe extern "C" fn(u32) -> u32;

                trait Helper {
                    fn required(&self) -> [u8; 4];
                }

                #[inline(
                    always
                )]
                #[allow(dead_code)]
                pub(crate) unsafe const fn ok_qualifiers<'a>(x: &'a u8) -> u8 {
                    let _ = '{';
                    *x
                }

                #[target_feature(
                    enable = "avx2"
                )]
                pub unsafe extern "C" fn ok_target() {}

                #[test]
                async fn ok_test() {}
            "##;
            assert_eq!(out_of_line(ok), None);

            let bad_unsafe_const = "pub(crate) unsafe const fn missing_inline() {}";
            assert_eq!(out_of_line(bad_unsafe_const), Some("missing_inline"));

            let bad_extern = "pub extern \"C\" fn missing_inline_extern() {}";
            assert_eq!(out_of_line(bad_extern), Some("missing_inline_extern"));

            let bad_async = "pub async fn missing_inline_async() {}";
            assert_eq!(out_of_line(bad_async), Some("missing_inline_async"));
        }
    }
}

#[cfg(test)]
mod native_tests {
    use super::*;
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn native_asset_generates_explicit_direct_loader_and_affects_identity() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "seismic-native-build-{}-{unique}",
            std::process::id()
        ));
        let output = root.join("out");
        fs::create_dir_all(root.join("native")).expect("fixture directories");
        let source = root.join("ops.seismic");
        let metal = root.join("native/scale.metal");
        fs::write(
            &source,
            "fn scale[N](x: &tensor[N] f32, factor: f32, output: &mut tensor[N] f32):\n    parallel for i in 0..N:\n        output[i] = x[i] * factor\n\nnative scale for metal from \"native/scale.metal\":\n    launch scale:\n        threadgroups (ceil_div(N, 256), 1, 1)\n        threads_per_threadgroup (256, 1, 1)\n",
        )
        .expect("Seismic fixture");
        fs::write(&metal, "kernel void scale() {}\n").expect("Metal fixture");

        let first = Build::new("fixture")
            .source(&source)
            .std(false)
            .out_dir(&output)
            .run().map_err(first_violation)
            .expect("first build");
        let generated = fs::read_to_string(&first.bindings).expect("generated bindings");
        assert!(generated.contains("pub fn native_for_device"));
        assert!(generated.contains("pub fn native_implementation"));
        assert!(generated.contains("pub fn native_tune"));
        assert!(generated.contains("NativeKernel<Entry>"));

        fs::write(&metal, "kernel void scale() { /* changed */ }\n").expect("changed Metal");
        let second = Build::new("fixture")
            .source(&source)
            .std(false)
            .out_dir(&output)
            .run().map_err(first_violation)
            .expect("second build");
        assert_ne!(first.identity, second.identity);
        fs::remove_dir_all(&root).expect("remove fixture directory");
    }

    /// The first violation a failed build names: the one each test below is
    /// about, which the checks reach in the order they once stopped at.
    fn first_violation(error: BuildError) -> BuildError {
        match error {
            BuildError::Natives(mut violations) => violations.remove(0),
            other => other,
        }
    }

    /// Build a one-entry module whose Vulkan asset is `asset`.
    fn build_vulkan(asset: &str) -> Result<Artifacts, BuildError> {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "seismic-native-vulkan-build-{}-{unique}",
            std::process::id()
        ));
        fs::create_dir_all(root.join("vulkan")).expect("fixture directories");
        let source = root.join("ops.seismic");
        fs::write(
            &source,
            "fn scale[N](x: &tensor[N] f32) -> tensor[N] f32:\n    let mut output = tensor[N] f32\n    for i in 0..N:\n        output[i] = x[i]\n    return output\n\nnative scale for vulkan from \"vulkan/scale.comp\":\n    params (WIDTH in [64] default 64)\n    launch scale:\n        threadgroups (ceil_div(N, WIDTH), 1, 1)\n        threads_per_threadgroup (WIDTH, 1, 1)\n        shared_bytes (WIDTH * 4)\n",
        )
        .expect("Seismic fixture");
        fs::write(root.join("vulkan/scale.comp"), asset).expect("Vulkan fixture");
        let result = Build::new("fixture")
            .source(&source)
            .std(false)
            .out_dir(root.join("out"))
            .run().map_err(first_violation);
        fs::remove_dir_all(&root).expect("remove fixture directory");
        result
    }

    #[test]
    fn vulkan_assets_admit_the_vulkan_abi() {
        build_vulkan(
            "void scale() {\n    seismic_f32 x = seismic_f32(SEISMIC_PTR(SEISMIC_BUFFER_X));\n    const bool read_only = SEISMIC_READONLY(SEISMIC_BUFFER_X);\n#if SEISMIC_HAS_MATRIX\n#endif\n    seismic_shared_f32[0] = x[uint(SEISMIC_X_STRIDE_0 * SEISMIC_SHARED_F32)].v + float(SEISMIC_TUNE_WIDTH);\n    out_put(ELEMENT_F32, SEISMIC_RESULT_0_OUT, 0ul, seismic_shared_f32[0]);\n}\n",
        )
        .expect("a Vulkan asset over its generated ABI builds");
    }

    #[test]
    fn vulkan_assets_reject_prefix_owned_constructs() {
        for (asset, construct, expected_line) in [
            ("#version 460\nvoid scale() {}\n", "`#version`", 1),
            (
                "void scale() {}\n  #  extension GL_EXT_foo : require\n",
                "`#extension`",
                2,
            ),
            (
                "layout(local_size_x = 64) in;\nvoid scale() {}\n",
                "`local_size_x`",
                1,
            ),
            (
                "layout(push_constant) uniform p { uint x; };\nvoid scale() {}\n",
                "`push_constant`",
                1,
            ),
            (
                "shared float staging[64];\nvoid scale() {}\n",
                "`shared`",
                1,
            ),
            ("void scale() {}\nvoid main() { scale(); }\n", "`main`", 2),
            (
                "void scale() { uint64_t raw = SEISMIC_RESULT_0_OUT.seismic_base; }\n",
                "`seismic_base`",
                1,
            ),
            ("void scale() { Out o = Out(0ul, 0); }\n", "handle construction", 1),
        ] {
            match build_vulkan(asset) {
                Err(BuildError::NativeReserved {
                    entry,
                    construct: found,
                    line,
                    ..
                }) => {
                    assert_eq!(entry, "scale");
                    assert!(found.contains(construct), "{found}");
                    assert_eq!(line, expected_line, "{asset}");
                }
                other => panic!("{asset}: expected a reserved construct, got {other:?}"),
            }
        }
        // Comments and the prefix-generated `seismic_shared_*` views are not
        // declarations.
        build_vulkan("// shared main\nvoid scale() { seismic_shared_u32[0] = 1u; }\n")
            .expect("comments and generated views are admitted");
        assert!(matches!(
            build_vulkan("void scale() { uint64_t w = SEISMIC_BUFFER_WORDS; }\n"),
            Err(BuildError::NativeAbi { symbol, .. }) if symbol == "SEISMIC_BUFFER_WORDS"
        ));
    }

    /// A Vulkan asset names lanes and subgroups, and reduces, votes, ballots
    /// and shuffles by absolute lane, only through the device library's
    /// logical-subgroup helpers.
    #[test]
    fn vulkan_assets_keep_to_the_logical_subgroup() {
        for identifier in [
            "gl_SubgroupInvocationID",
            "gl_SubgroupID",
            "gl_SubgroupSize",
            "gl_NumSubgroups",
            "gl_SubgroupEqMask",
            "subgroupElect",
            "subgroupAll",
            "subgroupAny",
            "subgroupAllEqual",
            "subgroupBroadcast",
            "subgroupBroadcastFirst",
            "subgroupBallot",
            "subgroupBallotBitCount",
            "subgroupAdd",
            "subgroupMax",
            "subgroupXor",
            "subgroupInclusiveAdd",
            "subgroupExclusiveMin",
            "subgroupClusteredAdd",
            "subgroupShuffle",
            "subgroupQuadSwapHorizontal",
        ] {
            let asset = format!("void scale() {{\n    uint x = 0u;\n    x += uint({identifier});\n}}\n");
            match build_vulkan(&asset) {
                Err(BuildError::NativeReserved { construct, line, .. }) => {
                    assert!(construct.contains(&format!("`{identifier}`")), "{construct}");
                    assert_eq!(line, 3, "{identifier}");
                }
                other => panic!("{identifier}: expected a refusal, got {other:?}"),
            }
        }
        build_vulkan(
            "// gl_SubgroupID subgroupBallot\nvoid scale() {\n    const uint subgroup = SEISMIC_SUBGROUP;\n    \
             uint x = subgroupShuffleXor(subgroup, 16u);\n    x += subgroupShuffleUp(x, 1u) + subgroupShuffleDown(x, 1u);\n    \
             subgroupBarrier();\n    subgroupMemoryBarrierShared();\n    x += seismic_ballot(x != 0u);\n}\n",
        )
        .expect("xor, up and down shuffles, subgroup barriers, the library's helpers and comments are admitted");
    }

    /// A Vulkan asset steers the hardware subgroup only by `seismic_steer`, as
    /// the whole condition of an `if`, `while` or `for`, with a steered loop
    /// exiting at its header.
    #[test]
    fn vulkan_assets_steer_at_conditions() {
        for (body, refused) in [
            ("if (seismic_hardware_any(n > 0u)) x = 1u;", "`seismic_hardware_any` is `seismic_steer`'s"),
            ("const bool any = seismic_steer(live, n > 0u);", "whole condition"),
            ("if (seismic_steer(live, n > 0u) && x > 0u) x = 1u;", "whole condition"),
            ("for (uint i = 0u; i < 4u && seismic_steer(live, i < n); ++i) x += i;", "whole condition"),
            (
                "for (uint i = 0u;; ++i) { if (!seismic_steer(live, i < n)) break; x += i; }",
                "exits at its header",
            ),
            ("while (true) { if (!seismic_steer(live, x < n)) { continue; } x += 1u; }", "exits at its header"),
        ] {
            let asset = format!("void scale() {{\n    uint x = 0u;\n    bool live;\n    {body}\n}}\n");
            match build_vulkan(&asset) {
                Err(BuildError::NativeReserved { construct, line, .. }) => {
                    assert!(construct.contains(refused), "{body}: {construct}");
                    assert_eq!(line, 4, "{body}");
                }
                other => panic!("{body}: expected a refusal, got {other:?}"),
            }
        }
        build_vulkan(
            "void scale() {\n    uint x = 0u;\n    bool live;\n    if (!seismic_steer(live, n > 0u))\n        return;\n    \
             if (seismic_steer(live, x < n)) x = 1u;\n    while (seismic_steer(live, x < n)) x += 1u;\n    \
             for (uint i = min(x, 2u); seismic_steer(live, i < n); ++i) x += live ? i : 0u;\n}\n",
        )
        .expect("`seismic_steer` as an `if`, `while` or `for` condition is admitted");
    }

    #[test]
    fn vulkan_launches_name_defined_kernels() {
        match build_vulkan("void scaled() {}\nvoid other(uint x) {}\n") {
            Err(BuildError::NativeKernelMissing { entry, kernel }) => {
                assert_eq!((entry.as_str(), kernel.as_str()), ("scale", "scale"));
            }
            other => panic!("expected a missing kernel, got {other:?}"),
        }
        build_vulkan("void helper() {}\n\nvoid scale ( ) {\n    helper();\n}\n")
            .expect("the kernel is defined");
    }

    #[test]
    fn one_build_names_every_native_violation() {
        let asset = "void scaled() { uint64_t w = SEISMIC_BUFFER_WORDS; }\nshared float staging[64];\n";
        let unique = SystemTime::now().duration_since(UNIX_EPOCH).expect("clock").as_nanos();
        let root = std::env::temp_dir().join(format!("seismic-native-violations-{}-{unique}", std::process::id()));
        fs::create_dir_all(root.join("vulkan")).expect("fixture directories");
        let source = root.join("ops.seismic");
        fs::write(&source, "fn scale[N](x: &tensor[N] f32) -> tensor[N] f32:\n    let mut output = tensor[N] f32\n    for i in 0..N:\n        output[i] = x[i]\n    return output\n\nnative scale for vulkan from \"vulkan/scale.comp\":\n    launch scale:\n        threadgroups (ceil_div(N, 64), 1, 1)\n        threads_per_threadgroup (64, 1, 1)\n").expect("Seismic fixture");
        fs::write(root.join("vulkan/scale.comp"), asset).expect("Vulkan fixture");
        let result = Build::new("fixture").source(&source).std(false).out_dir(root.join("out")).run();
        fs::remove_dir_all(&root).expect("remove fixture directory");
        match result {
            Err(BuildError::Natives(violations)) => assert!(
                matches!(
                    violations.as_slice(),
                    [
                        BuildError::NativeAbi { symbol, .. },
                        BuildError::NativeReserved { line: 2, .. },
                        BuildError::NativeKernelMissing { kernel, .. },
                    ] if symbol == "SEISMIC_BUFFER_WORDS" && kernel == "scale"
                ),
                "{violations:?}"
            ),
            other => panic!("expected every violation, got {other:?}"),
        }
    }

    #[test]
    fn native_asset_rejects_dimension_macro_not_generated_for_entry() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "seismic-native-abi-build-{}-{unique}",
            std::process::id()
        ));
        let output = root.join("out");
        fs::create_dir_all(root.join("native")).expect("fixture directories");
        let source = root.join("ops.seismic");
        let metal = root.join("native/scale.metal");
        fs::write(
            &source,
            "fn scale[N](x: &tensor[N] f32) -> tensor[N] f32:\n    let mut output = tensor[N] f32\n    for i in 0..N:\n        output[i] = x[i]\n    return output\n\nnative scale for metal from \"native/scale.metal\":\n    launch scale:\n        threadgroups (ceil_div(N, 256), 1, 1)\n        threads_per_threadgroup (256, 1, 1)\n",
        )
        .expect("Seismic fixture");
        fs::write(
            &metal,
            "kernel void scale(uint index [[thread_position_in_grid]]) { if (index < SEISMIC_DIM_W) {} }\n",
        )
        .expect("Metal fixture");

        let error = Build::new("fixture")
            .source(&source)
            .std(false)
            .out_dir(&output)
            .run().map_err(first_violation)
            .expect_err("unknown entry ABI symbol must fail the consumer build");
        match error {
            BuildError::NativeAbi {
                entry,
                line,
                symbol,
                ..
            } => {
                assert_eq!(entry, "scale");
                assert_eq!(line, 1);
                assert_eq!(symbol, "SEISMIC_DIM_W");
            }
            other => panic!("unexpected error: {other}"),
        }
        fs::remove_dir_all(&root).expect("remove fixture directory");
    }

    /// Builds entry `request`, `signature` with one Metal launch, against
    /// the Metal asset `asset`.
    fn build_request(name: &str, signature: &str, asset: &str) -> Result<Artifacts, BuildError> {
        let root = fixture(name);
        let source = root.join("ops.seismic");
        fs::write(
            &source,
            format!("fn request{signature}:\n    out[0] = 1\n\nnative request for metal from \"native/request.metal\":\n    launch request:\n        threadgroups (1, 1, 1)\n        threads_per_threadgroup (32, 1, 1)\n"),
        )
        .expect("Seismic fixture");
        fs::write(root.join("native/request.metal"), asset).expect("Metal fixture");
        let result = Build::new("fixture")
            .source(&source)
            .std(false)
            .out_dir(root.join("out"))
            .run().map_err(first_violation);
        fs::remove_dir_all(&root).expect("remove fixture directory");
        result
    }

    /// A tensor parameter named `words` would be `SEISMIC_BUFFER_WORDS`,
    /// which the ABI defines for its words buffer: the build rejects it.
    #[test]
    fn a_parameter_named_as_an_abi_buffer_is_rejected() {
        match build_request(
            "abi-collision-words",
            "[N](words: &tensor[N] f32, out: &mut tensor[1] i32)",
            "kernel void request() {}\n",
        ) {
            Err(BuildError::NativeAbiCollision { symbol, second, .. }) => {
                assert_eq!(symbol, "SEISMIC_BUFFER_WORDS");
                assert_eq!(second, "parameter 0 `words`");
            }
            other => panic!("unexpected result: {:?}", other.err()),
        }
    }

    /// A tensor named `param_1` names its extents `SEISMIC_PARAM_1_EXTENT_0`,
    /// which parameter 1's ordinal macros define for another tensor.
    #[test]
    fn a_named_alias_equal_to_an_ordinal_macro_is_rejected() {
        match build_request(
            "abi-collision-ordinal",
            "[N, M](param_1: &tensor[N] f32, x: &tensor[M] f32, out: &mut tensor[1] i32)",
            "kernel void request() {}\n",
        ) {
            Err(BuildError::NativeAbiCollision {
                symbol,
                first,
                second,
                ..
            }) => {
                assert!(symbol.starts_with("SEISMIC_PARAM_1"), "{symbol}");
                assert_eq!(first, "parameter 0 `param_1`");
                assert_eq!(second, "parameter 1 `x`");
            }
            other => panic!("unexpected result: {:?}", other.err()),
        }
    }

    /// A host port defines no ABI symbol (the kernel never accesses it), so
    /// an asset naming one is refused, and its name collides with nothing.
    #[test]
    fn host_ports_define_no_abi_symbols() {
        match build_request(
            "abi-host-staged",
            "[N](staged: &host mut tensor[N] i32, out: &mut tensor[1] i32)",
            "kernel void request() { ulong b = SEISMIC_BUFFER_STAGED; }\n",
        ) {
            Err(BuildError::NativeAbi { symbol, .. }) => {
                assert_eq!(symbol, "SEISMIC_BUFFER_STAGED");
            }
            other => panic!("unexpected result: {:?}", other.err()),
        }
        build_request(
            "abi-host-words",
            "[N](words: &host tensor[N] i32, out: &mut tensor[1] i32)",
            "kernel void request(device int *out [[buffer(SEISMIC_BUFFER_OUT)]]) { out[0] = 1; }\n",
        )
        .expect("a host port named `words` collides with no ABI symbol");
    }

    #[test]
    fn native_asset_accepts_exact_generated_tensor_and_result_symbols() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "seismic-native-abi-valid-build-{}-{unique}",
            std::process::id()
        ));
        let output = root.join("out");
        fs::create_dir_all(root.join("native")).expect("fixture directories");
        let source = root.join("ops.seismic");
        let metal = root.join("native/scale.metal");
        fs::write(
            &source,
            "fn scale[N](x: &tensor[N] f32) -> tensor[N] f32:\n    let mut output = tensor[N] f32\n    for i in 0..N:\n        output[i] = x[i]\n    return output\n\nnative scale for metal from \"native/scale.metal\":\n    launch scale:\n        threadgroups (ceil_div(N, 256), 1, 1)\n        threads_per_threadgroup (256, 1, 1)\n",
        )
        .expect("Seismic fixture");
        fs::write(
            &metal,
            "// SEISMIC_DIM_NOT_AN_ABI_SYMBOL inside a comment is inert.\nkernel void scale(device const float *x [[buffer(SEISMIC_BUFFER_X)]], SEISMIC_RESULT_0_PARAM(output), constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]], uint index [[thread_position_in_grid]]) {\n#if defined(SEISMIC_X_KIND_DENSE) && defined(SEISMIC_RESULT_0_KIND_DENSE)\n    if (index < SEISMIC_DIM_N) output[index * SEISMIC_RESULT_0_STRIDE_0] = x[index * SEISMIC_X_STRIDE_0];\n#endif\n}\n",
        )
        .expect("Metal fixture");

        Build::new("fixture")
            .source(&source)
            .std(false)
            .out_dir(&output)
            .run().map_err(first_violation)
            .expect("exact entry ABI symbols must pass the consumer build");
        fs::remove_dir_all(&root).expect("remove fixture directory");
    }

    #[test]
    fn module_constants_are_generated_and_admitted_by_name() {
        let root = fixture("constants");
        let source = root.join("ops.seismic");
        fs::write(&source, "const WIDTH = 256\nconst GROUPS(STAGED) = 1 + STAGED * (WIDTH - 1)\n\nfn scale[N](x: &tensor[N] f32) -> tensor[N] f32:\n    return to_owned(x)\n\nnative scale for metal from \"native/scale.metal\":\n    launch scale:\n        threadgroups (ceil_div(N, WIDTH), 1, 1)\n        threads_per_threadgroup (WIDTH, 1, 1)\n").unwrap();
        let asset = root.join("native/scale.metal");
        let build = |text: &str, name: &str| {
            fs::write(&asset, text).unwrap();
            Build::new("fixture")
                .source(&source)
                .std(false)
                .out_dir(root.join(name))
                .run().map_err(first_violation)
        };
        let built = build(
            "kernel void scale(uint index [[thread_position_in_grid]]) { if (index < SEISMIC_CONST_WIDTH) {} }\n",
            "named",
        )
        .expect("a declared module constant is part of the ABI");
        let generated = fs::read_to_string(&built.bindings).unwrap();
        let checked = |operation: &str, left: &str, right: &str| {
            format!("({left}).checked_{operation}({right}).expect(\"native arithmetic stays within u64\")")
        };
        let groups = checked("add", "1u64", &checked("mul", "STAGED", &checked("sub", "256u64", "1u64")));
        assert!(generated.contains(&format!(
            "pub mod constants {{\n  pub const WIDTH: u64 = 256;\n  #[allow(non_snake_case)]\n  pub const fn GROUPS(STAGED: u64) -> u64 {{\n    {groups}\n  }}\n}}"
        )));
        match build(
            "kernel void scale(uint index [[thread_position_in_grid]]) { if (index < SEISMIC_CONST_HEIGHT) {} }\n",
            "unknown",
        ) {
            Err(BuildError::NativeAbi { symbol, .. }) => assert_eq!(symbol, "SEISMIC_CONST_HEIGHT"),
            other => panic!("expected an unknown constant, got {other:?}"),
        }
        fs::remove_dir_all(&root).expect("remove fixture directory");
    }

    fn fixture(name: &str) -> std::path::PathBuf {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "seismic-native-{name}-{}-{unique}",
            std::process::id()
        ));
        fs::create_dir_all(root.join("native")).expect("fixture directories");
        root
    }

    const SPECIALIZED: &str = "fn scale[N](x: &tensor[N] f32) -> tensor[N] f32:\n    let mut output = tensor[N] f32\n    for i in 0..N:\n        output[i] = x[i]\n    return output\n\nnative scale for cuda from \"native/scale.cu\":\n    static (N)\n    params (TILE in [32, 64] default 32)\n    scratch staging bytes (N * 4)\n    launch scale:\n        threadgroups (ceil_div(N, TILE), 1, 1)\n        threads_per_threadgroup (TILE, 1, 1)\n\nnative scale for cpu from \"native/scale.rs\":\n    params (TILE in [32, 64] default 32, UNROLL in [1, 2, 4] default 1)\n    launch scale:\n        threadgroups (ceil_div(N, TILE), 1, 1)\n        threads_per_threadgroup (1, 1, 1)\n";

    const CPU_SCALE: &str = "#[inline(always)]\nfn scale<L: Isa, E: Elements>(_l: L, cx: &Context<'_, E>, group: [u64; 3], _shared: &mut [u8]) {\n    let (x, output, tile) = (cx.arg_x(), cx.result_0(), cx.param_tile());\n    for i in group[0] * tile..((group[0] + 1) * tile).min(cx.dim_n()) {\n        unsafe { output.set([i as usize], x.get([i as usize])) };\n    }\n    let _ = cx.param_unroll();\n}\n";

    #[test]
    fn launch_scoped_assets_use_forming_guards_without_tuning_macros() {
        let root = fixture("scoped-abi");
        let source = root.join("ops.seismic");
        fs::write(&source, "fn scale[N](x: &tensor[N] f32) -> tensor[N] f32:\n    return to_owned(x)\n\nnative scale for cuda from \"native/scale.cu\":\n    launch scale_a when N <= 8:\n        params (TILE in [4, 8] default 4)\n        threadgroups (ceil_div(N, TILE), 1, 1)\n        threads_per_threadgroup (32, 1, 1)\n    launch scale_b when N > 8:\n        params (TILE in [16, 32] default 16)\n        threadgroups (ceil_div(N, TILE), 1, 1)\n        threads_per_threadgroup (32, 1, 1)\n").unwrap();
        let asset = root.join("native/scale.cu");
        fs::write(&asset, "#ifdef SEISMIC_FORMING_SCALE_A\ntemplate <unsigned TILE>\n__global__ void scale_a() {}\n#endif\n#ifdef SEISMIC_FORMING_SCALE_B\ntemplate <unsigned TILE>\n__global__ void scale_b() {}\n#endif\n").unwrap();
        let build = |name: &str| {
            Build::new("fixture")
                .source(&source)
                .std(false)
                .out_dir(root.join(name))
                .run().map_err(first_violation)
        };
        build("valid").expect("forming guards are part of the scoped native ABI");
        fs::write(&asset, "#ifdef SEISMIC_FORMING_SCALE_A\ntemplate <unsigned TILE>\n__global__ void scale_a() { (void)SEISMIC_TUNE_TILE; }\n#endif\n#ifdef SEISMIC_FORMING_SCALE_B\ntemplate <unsigned TILE>\n__global__ void scale_b() {}\n#endif\n").unwrap();
        match build("invalid").expect_err("a scoped asset cannot read old tuning macros") {
            BuildError::NativeAbi { symbol, .. } => assert_eq!(symbol, "SEISMIC_TUNE_TILE"),
            other => panic!("unexpected error: {other}"),
        }
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn metal_scoped_templates_match_the_launch_declaration() {
        let root = fixture("metal-scoped-template");
        let source = root.join("ops.seismic");
        fs::write(&source, "fn scale[N](x: &tensor[N] f32) -> tensor[N] f32:\n    return to_owned(x)\n\nnative scale for metal from \"native/scale.metal\":\n    params (SPLIT in [1, 2] default 1)\n    launch small when N <= 8:\n        params (ROWS in [1, 2] default 1)\n        threadgroups (ceil_div(N, ROWS), 1, 1)\n        threads_per_threadgroup (32, 1, 1)\n    launch large when N > 8:\n        params (WIDTH in [4, 8] default 4, ROWS in [1, 2] default 1)\n        threadgroups (ceil_div(N, WIDTH * ROWS), 1, 1)\n        threads_per_threadgroup (32, 1, 1)\n").unwrap();
        let asset = root.join("native/scale.metal");
        let small = "#ifdef SEISMIC_FORMING_SMALL\ntemplate <uint ROWS>\nkernel void small() { (void)SEISMIC_RUNTIME_SPLIT; }\n#endif\n";
        let large = "#ifdef SEISMIC_FORMING_LARGE\ntemplate <uint WIDTH, uint ROWS>\nkernel void large() {}\n#endif\n";
        let build = |name: &str| {
            Build::new("fixture")
                .source(&source)
                .std(false)
                .out_dir(root.join(name))
                .run().map_err(first_violation)
        };
        fs::write(&asset, format!("{small}{large}")).unwrap();
        build("valid").unwrap();
        fs::write(
            &asset,
            format!(
                "{}{large}",
                small.replace("SEISMIC_RUNTIME_SPLIT", "SEISMIC_TUNE_SPLIT")
            ),
        )
        .unwrap();
        match build("obsolete-runtime-name")
            .expect_err("scoped runtime words have their own ABI name")
        {
            BuildError::NativeAbi { symbol, .. } => assert_eq!(symbol, "SEISMIC_TUNE_SPLIT"),
            other => panic!("unexpected error: {other}"),
        }
        fs::write(
            &asset,
            format!(
                "{small}{}",
                large.replace("WIDTH, uint ROWS", "ROWS, uint WIDTH")
            ),
        )
        .unwrap();
        match build("reordered").expect_err("code parameters are positional") {
            BuildError::Source(error) => {
                assert!(error.to_string().contains("launch `large`"), "{error}")
            }
            other => panic!("unexpected error: {other}"),
        }
        fs::write(
            &asset,
            format!(
                "{small}{}",
                large.replace(
                    "#ifdef SEISMIC_FORMING_LARGE",
                    "#ifdef SEISMIC_FORMING_SMALL"
                )
            ),
        )
        .unwrap();
        match build("unguarded").expect_err("every kernel needs its own forming guard") {
            BuildError::Source(error) => {
                assert!(error.to_string().contains("launch `large`"), "{error}")
            }
            other => panic!("unexpected error: {other}"),
        }
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn cuda_scoped_templates_match_the_launch_declaration() {
        let root = fixture("cuda-scoped-template");
        let source = root.join("ops.seismic");
        fs::write(&source, "fn scale[N](x: &tensor[N] f32) -> tensor[N] f32:\n    return to_owned(x)\n\nnative scale for cuda from \"native/scale.cu\":\n    params (arithmetic INT8 in [0, 1] default 0)\n    launch small when N <= 8:\n        params (KSPLIT in [2, 4] default 2)\n        threadgroups (ceil_div(N, KSPLIT), 1, 1)\n        threads_per_threadgroup (32, 1, 1)\n    launch large when N > 8:\n        reads (INT8)\n        threadgroups (ceil_div(N, 32), 1, 1)\n        threads_per_threadgroup (32, 1, 1)\n").unwrap();
        let asset = root.join("native/scale.cu");
        let small = "#ifdef SEISMIC_FORMING_SMALL\ntemplate <int KSPLIT>\n__global__ void small() {}\n#endif\n";
        let large = "#ifdef SEISMIC_FORMING_LARGE\ntemplate <unsigned INT8>\n__global__ void large() {}\n#endif\n";
        let build = |name: &str| {
            Build::new("fixture")
                .source(&source)
                .std(false)
                .out_dir(root.join(name))
                .run().map_err(first_violation)
        };
        fs::write(&asset, format!("{small}{large}")).unwrap();
        build("valid").unwrap();
        fs::write(
            &asset,
            format!(
                "{small}{}",
                large.replace("unsigned INT8", "unsigned KSPLIT")
            ),
        )
        .unwrap();
        match build("wrong-parameter").expect_err("CUDA code parameters are positional") {
            BuildError::Source(error) => {
                assert!(error.to_string().contains("launch `large`"), "{error}")
            }
            other => panic!("unexpected error: {other}"),
        }
        fs::remove_dir_all(root).unwrap();
    }

    /// A CPU function compiled out of line would run without its tier's
    /// features, so one not declared `#[inline(always)]` is refused.
    #[test]
    fn cpu_functions_must_inline_into_their_tier_boundary() {
        let root = fixture("cpu-inline");
        let source = root.join("ops.seismic");
        fs::write(&source, "fn scale[N](x: &tensor[N] f32) -> tensor[N] f32:\n    let mut output = tensor[N] f32\n    for i in 0..N:\n        output[i] = x[i]\n    return output\n\nnative scale for cpu from \"native/scale.rs\":\n    launch scale:\n        threadgroups (N, 1, 1)\n        threads_per_threadgroup (1, 1, 1)\n").expect("Seismic fixture");
        fs::create_dir_all(root.join("native")).expect("native directory");
        fs::write(root.join("native/scale.rs"), "fn scale<L: Isa, E: Elements>(_l: L, cx: &Context<'_, E>, group: [u64; 3], _shared: &mut [u8]) {\n    let i = group[0] as usize;\n    unsafe { cx.result_0().set([i], cx.arg_x().get([i])) };\n}\n").expect("CPU fixture");
        let error = Build::new("fixture").source(&source).std(false).out_dir(root.join("out")).run().map_err(first_violation).expect_err("an out-of-line function is refused");
        assert!(error.to_string().contains("`scale` is not declared `#[inline(always)]`"), "{error}");
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn cuda_assets_admit_tuning_and_scratch_symbols_and_reject_headers() {
        let root = fixture("cuda");
        let source = root.join("ops.seismic");
        fs::write(&source, SPECIALIZED).expect("Seismic fixture");
        fs::write(root.join("native/scale.rs"), CPU_SCALE).expect("CPU fixture");
        let cuda = root.join("native/scale.cu");
        fs::write(
            &cuda,
            "extern \"C\" __global__ void scale(SEISMIC_KERNEL_PARAMS) {\n    float *x = reinterpret_cast<float *>(SEISMIC_PTR(SEISMIC_BUFFER_X));\n    float *staging = reinterpret_cast<float *>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_STAGING));\n    if (threadIdx.x < SEISMIC_TUNE_TILE && blockIdx.x * SEISMIC_TUNE_TILE < SEISMIC_DIM_N) staging[0] = x[0];\n}\n",
        )
        .expect("CUDA fixture");
        let build = |name: &str| {
            Build::new("fixture")
                .source(&source)
                .std(false)
                .out_dir(root.join(name))
                .run().map_err(first_violation)
        };
        let artifacts = build("out").expect("tuning and scratch symbols are generated ABI");
        let generated = fs::read_to_string(&artifacts.bindings).expect("generated bindings");
        assert!(generated.contains("Some(&cpu_native::KERNELS)"));
        // Tuning parameters are runtime values: one form per tier of the
        // one launch, none per value.
        assert_eq!(generated.matches("as CpuKernelFn").count(), seismic_cpu_tiers::Tier::EVERY.len());
        assert!(generated.contains("pub fn param_tile(&self) -> u64 { self.invocation.param(0) }"));
        assert!(
            generated.contains("pub fn param_unroll(&self) -> u64 { self.invocation.param(1) }")
        );
        assert!(generated.contains("#[target_feature(enable = \"sse3,ssse3,sse4.1,sse4.2,popcnt,avx,avx2,fma,f16c,bmi1,bmi2,lzcnt,movbe\")]"));
        assert!(generated.contains("super::scale::<seismic::cpu::X86V3, super::Elements_>"));

        fs::write(&cuda, "#include <cuda_fp16.h>\nextern \"C\" __global__ void scale(SEISMIC_KERNEL_PARAMS) {}\n")
            .expect("CUDA fixture");
        match build("header").expect_err("vendor headers are rejected") {
            BuildError::NativeInclude(error) => {
                assert_eq!(
                    error.reason,
                    seismic_lang::source::NativeIncludeReason::System
                );
                assert_eq!(error.line, 1);
            }
            other => panic!("unexpected error: {other}"),
        }

        fs::write(&cuda, "extern \"C\" __global__ void scale(SEISMIC_KERNEL_PARAMS) { (void)SEISMIC_TUNE_WIDTH; }\n")
            .expect("CUDA fixture");
        match build("unknown").expect_err("undeclared parameters are rejected") {
            BuildError::NativeAbi { symbol, .. } => assert_eq!(symbol, "SEISMIC_TUNE_WIDTH"),
            other => panic!("unexpected error: {other}"),
        }
        fs::remove_dir_all(&root).expect("remove fixture directory");
    }

    /// A host port occupies its extent and stride words but no buffer
    /// (`abi::bound`), so the CPU context numbers the buffers after it from
    /// the first bound one.
    #[test]
    fn cpu_host_ports_take_words_but_no_buffer() {
        let root = fixture("cpu-host-port");
        let source = root.join("ops.seismic");
        fs::write(&source, "fn scale[N](layout: &host tensor[N] i32, x: &tensor[N] f32) -> tensor[N] f32:\n    let mut output = tensor[N] f32\n    for i in 0..N:\n        output[i] = x[i]\n    return output\n\nnative scale for cpu from \"native/scale.rs\":\n    launch scale:\n        threadgroups (N, 1, 1)\n        threads_per_threadgroup (1, 1, 1)\n").expect("Seismic fixture");
        fs::write(root.join("native/scale.rs"), "#[inline(always)]\nfn scale<L: Isa, E: Elements>(_l: L, cx: &Context<'_, E>, group: [u64; 3], _shared: &mut [u8]) {\n    let i = group[0] as usize;\n    unsafe { cx.result_0().set([i], cx.arg_x().get([i])) };\n}\n").expect("CPU fixture");
        let artifacts = Build::new("fixture")
            .source(&source)
            .std(false)
            .out_dir(root.join("out"))
            .run().map_err(first_violation)
            .expect("a CPU entry with a host port builds");
        let generated = fs::read_to_string(&artifacts.bindings).expect("generated bindings");
        let accessor = |name: &str| {
            let start = generated
                .find(&format!("pub fn {name}(&self)"))
                .unwrap_or_else(|| panic!("`{name}` is generated"));
            let end = start + generated[start..].find(" }\n").expect("accessor end");
            generated[start..end].to_owned()
        };
        // Word 0 is N, words 1 and 2 are the host port's extent and stride.
        let x = accessor("arg_x");
        assert!(x.contains("self.invocation.buffer(0)"), "{x}");
        assert!(x.contains("[self.invocation.word(3)], [self.invocation.word(4)]"), "{x}");
        let result = accessor("result_0");
        assert!(result.contains("self.invocation.buffer(1)"), "{result}");
        assert!(result.contains("[self.invocation.word(5)], [self.invocation.word(6)]"), "{result}");
        assert!(!generated.contains("pub fn arg_layout("));
        fs::remove_dir_all(&root).expect("remove fixture directory");
    }

    const SCALE_METAL: &str = "fn scale[N](x: &tensor[N] f32) -> tensor[N] f32:\n    let mut output = tensor[N] f32\n    for i in 0..N:\n        output[i] = x[i]\n    return output\n\nnative scale for metal from \"native/scale.metal\":\n    launch scale:\n        threadgroups (ceil_div(N, 256), 1, 1)\n        threads_per_threadgroup (256, 1, 1)\n";

    /// A floating result has an owner handle and no buffer index, so a raw `SEISMIC_RESULT_<n>_BUFFER`
    /// of one is not an ABI symbol: the build rejects it in an asset of every dialect, while the
    /// handle passes. A mutated floating tensor's `SEISMIC_BUFFER_<NAME>` is rejected the same way.
    #[test]
    fn a_floating_result_or_mutated_tensor_has_no_buffer_symbol() {
        let metal_result = |asset: &str| build_scale("metal", "metal", asset);
        metal_result("kernel void scale(SEISMIC_RESULT_0_PARAM(output)) {}\n")
            .expect("a Metal asset reaches its floating result through the handle");
        match metal_result("kernel void scale(device float *output [[buffer(SEISMIC_RESULT_0_BUFFER)]]) {}\n") {
            Err(BuildError::NativeAbi { symbol, .. }) => assert_eq!(symbol, "SEISMIC_RESULT_0_BUFFER"),
            other => panic!("a raw Metal result buffer must be rejected, got {other:?}"),
        }

        let vulkan = "void scale() { Out out = SEISMIC_RESULT_0_OUT; }\n";
        build_vulkan(vulkan).expect("a Vulkan asset reaches its floating result through the handle");
        match build_vulkan("void scale() { uint64_t raw = SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER); }\n") {
            Err(BuildError::NativeAbi { symbol, .. }) => assert_eq!(symbol, "SEISMIC_RESULT_0_BUFFER"),
            other => panic!("a raw Vulkan result buffer must be rejected, got {other:?}"),
        }

        // A floating tensor the kernel mutates has a handle and no buffer symbol either.
        let mutated = |asset: &str| {
            let root = fixture("handle-mutated");
            let source = root.join("ops.seismic");
            fs::write(
                &source,
                "fn fill[N](grid: &mut tensor[N] f32):\n    grid[0] = 1.0\n\nnative fill for metal from \"native/fill.metal\":\n    launch fill:\n        threadgroups (1, 1, 1)\n        threads_per_threadgroup (32, 1, 1)\n",
            )
            .expect("Seismic fixture");
            fs::write(root.join("native/fill.metal"), asset).expect("Metal fixture");
            let result = Build::new("fixture").source(&source).std(false).out_dir(root.join("out")).run().map_err(first_violation);
            fs::remove_dir_all(&root).expect("remove fixture directory");
            result
        };
        mutated("kernel void fill(SEISMIC_OUT_GRID_PARAM(grid)) {}\n")
            .expect("a mutated floating tensor is reached through the handle");
        match mutated("kernel void fill(device float *grid [[buffer(SEISMIC_BUFFER_GRID)]]) {}\n") {
            Err(BuildError::NativeAbi { symbol, .. }) => assert_eq!(symbol, "SEISMIC_BUFFER_GRID"),
            other => panic!("a raw mutated-tensor buffer must be rejected, got {other:?}"),
        }

        let cuda_result = |asset: &str| build_scale("cuda", "cu", asset);
        cuda_result("void scale() { auto output = SEISMIC_RESULT_0_OUT; }\n")
            .expect("a CUDA asset reaches its floating result through the handle");
        match cuda_result("void scale() { auto output = SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER); }\n") {
            Err(BuildError::NativeAbi { symbol, .. }) => assert_eq!(symbol, "SEISMIC_RESULT_0_BUFFER"),
            other => panic!("a raw CUDA result buffer must be rejected, got {other:?}"),
        }
    }

    /// Builds `SCALE_METAL`'s entry with its native for `backend` instead, from
    /// `native/scale.<extension>` holding `asset`.
    fn build_scale(backend: &str, extension: &str, asset: &str) -> Result<Artifacts, BuildError> {
        let root = fixture(&format!("scale-{backend}"));
        let source = root.join("ops.seismic");
        let text = SCALE_METAL.replace(
            "for metal from \"native/scale.metal\"",
            &format!("for {backend} from \"native/scale.{extension}\""),
        );
        fs::write(&source, text).expect("Seismic fixture");
        fs::write(root.join(format!("native/scale.{extension}")), asset).expect("native fixture");
        let result = Build::new("fixture").source(&source).std(false).out_dir(root.join("out")).run().map_err(first_violation);
        fs::remove_dir_all(&root).expect("remove fixture directory");
        result
    }

    /// Every dialect refuses a literal buffer index and `Cell<`; CUDA and Vulkan, whose handles are
    /// built from an address, refuse every handle a kernel constructs, and Metal admits a handle
    /// over the typed cells its parameter declaration gives.
    #[test]
    fn every_dialect_refuses_raw_owner_constructs() {
        let metal = |asset: &str| build_scale("metal", "metal", asset);
        let cuda = |asset: &str| build_scale("cuda", "cu", asset);
        let refusals: [(&dyn Fn(&str) -> Result<Artifacts, BuildError>, &str, &str); 21] = [
            (&metal, "kernel void scale(device const float *x [[buffer(0)]]) {}\n", "literal buffer index"),
            (&metal, "kernel void scale(device float *seismic_buffer_1) {}\n", "literal buffer index"),
            (
                &metal,
                "kernel void scale(device uchar *s) { element::Out<element::F32>(reinterpret_cast<device element::Cell<element::F32> *>(s)); }\n",
                "typed cells",
            ),
            (&cuda, "void scale() { auto x = SEISMIC_PTR(0); }\n", "literal buffer index"),
            (&cuda, "void scale() { auto x = seismic_buffer_1; }\n", "literal buffer index"),
            (&cuda, "void scale() { element::Cell<element::F32> *c; }\n", "typed cells"),
            (
                &cuda,
                "void scale() { element::Out<element::F32>(SEISMIC_PTR(SEISMIC_BUFFER_X)).put(0, 1.0f); }\n",
                "handle construction",
            ),
            (&cuda, "void scale(u8 *p) { auto o = element::Out(p); }\n", "handle construction"),
            (&cuda, "void scale(u8 *p) { auto o = element::Out<element::F32>{p}; }\n", "handle construction"),
            (&cuda, "void scale(u8 *p) { const element::Out<element::F32> o(p); }\n", "handle construction"),
            (&cuda, "void scale(u8 *p) { element::Out<element::F32> o{p}; }\n", "handle construction"),
            (&cuda, "void scale(u8 *p) { element::Out<element::F32> o = {p}; }\n", "handle construction"),
            (&cuda, "void scale(u8 *p) { element::SlabOut<element::F32> s(p); }\n", "handle construction"),
            (&cuda, "void scale(u8 *p) { auto o = element::abi_out<element::F32>(p); }\n", "ABI constructor"),
            (&cuda, "void scale(const u8 *p) { auto s = element::abi_slab_out<element::F32>(p); }\n", "ABI constructor"),
            (&build_vulkan, "void scale() { uint64_t a = SEISMIC_PTR(0); }\n", "literal buffer index"),
            (&build_vulkan, "void scale() { uint64_t a = seismic_buffer_0; }\n", "literal buffer index"),
            (&build_vulkan, "void scale() { Out o = Out(0ul, ELEMENT_F32); }\n", "handle construction"),
            (&build_vulkan, "void scale() { SlabOut s = SlabOut(0ul, ELEMENT_F32); }\n", "handle construction"),
            (&build_vulkan, "void scale() { Stored s = Stored(0ul, ELEMENT_F32); }\n", "handle construction"),
            (&build_vulkan, "void scale() { Stored s = {0ul, ELEMENT_F32}; }\n", "handle construction"),
        ];
        for (build, asset, rule) in refusals {
            match build(asset) {
                Err(BuildError::NativeReserved { construct, line, .. }) => {
                    assert!(construct.starts_with(rule), "{asset}: {construct}");
                    assert_eq!(line, 1, "{asset}");
                }
                other => panic!("{asset}: expected `{rule}`, got {other:?}"),
            }
        }

        metal(
            "kernel void scale(device const float *x [[buffer(SEISMIC_BUFFER_X)]], SEISMIC_RESULT_0_PARAM(output_cells)) {\n    const element::Out<element::F32> result(output_cells);\n    element::Out<element::F32>(output_cells).put(0, x[0]);\n    result.put(1, x[1]);\n}\n",
        )
        .expect("a Metal handle over its parameter's typed cells builds");
        cuda(
            "__device__ element::Out<element::F32> row(const element::Out<element::F32> &out, u64 r) { return out.offset(r); }\nvoid scale() {\n    const element::Out<element::F32> out = SEISMIC_RESULT_0_OUT;\n    const element::Out<element::F32> none(nullptr);\n    row(out, 1).put(0, 1.0f);\n}\n",
        )
        .expect("a CUDA kernel holding the ABI's handle, a function returning one and the null handle build");
        build_vulkan("void scale() { Out out = SEISMIC_RESULT_0_OUT; Stored from = out_stored(out, 0u); out_move_stored(out, 1u, from, 1u); }\n")
            .expect("a Vulkan kernel binding an out_stored token builds");
    }

    #[test]
    fn includes_are_inlined_validated_and_part_of_identity() {
        let root = fixture("include");
        let source = root.join("ops.seismic");
        fs::write(&source, SCALE_METAL).expect("Seismic fixture");
        fs::create_dir_all(root.join("native/common")).expect("common directory");
        let header = root.join("native/common/copy.h");
        fs::write(
            &header,
            "#define COPY(i) output[i * SEISMIC_RESULT_0_STRIDE_0] = x[i * SEISMIC_X_STRIDE_0]\n",
        )
        .expect("header fixture");
        fs::write(
            root.join("native/scale.metal"),
            "#include \"common/copy.h\"\nkernel void scale(device const float *x [[buffer(SEISMIC_BUFFER_X)]], SEISMIC_RESULT_0_PARAM(output), constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]], uint index [[thread_position_in_grid]]) { if (index < SEISMIC_DIM_N) COPY(index); }\n",
        )
        .expect("Metal fixture");
        let build = |name: &str| {
            Build::new("fixture")
                .source(&source)
                .std(false)
                .out_dir(root.join(name))
                .run().map_err(first_violation)
        };
        let first = build("first").expect("an include builds");
        let rendered = fs::read_to_string(root.join("first/scale.metal")).expect("rendered asset");
        assert!(rendered.starts_with("#line 1 \"common/copy.h\"\n#define COPY(i)"));
        assert!(!rendered.contains("#include"));

        fs::write(&header, "#define COPY(i) output[i] = x[i]\n").expect("changed header");
        let second = build("second").expect("changed header builds");
        assert_ne!(
            first.identity, second.identity,
            "included files are part of identity"
        );

        fs::write(
            &header,
            "#define COPY(i) output[i] = x[i * SEISMIC_DIM_W]\n",
        )
        .expect("bad header");
        match build("symbol").expect_err("included files are validated against the entry ABI") {
            BuildError::NativeAbi {
                path, line, symbol, ..
            } => {
                assert_eq!(path, header.canonicalize().expect("canonical header"));
                assert_eq!(line, 1);
                assert_eq!(symbol, "SEISMIC_DIM_W");
            }
            other => panic!("unexpected error: {other}"),
        }

        // Beside the source root, outside it.
        let outside = root.with_extension("outside.h");
        fs::write(&outside, "\n").expect("outside header");
        fs::write(
            root.join("native/scale.metal"),
            format!(
                "#include \"../../{}\"\nkernel void scale() {{}}\n",
                outside.file_name().expect("file name").to_string_lossy()
            ),
        )
        .expect("Metal fixture");
        match build("outside").expect_err("includes outside the source roots are rejected") {
            BuildError::NativeInclude(error) => {
                assert_eq!(
                    error.reason,
                    seismic_lang::source::NativeIncludeReason::OutsideSourceRoots
                )
            }
            other => panic!("unexpected error: {other}"),
        }
        fs::remove_file(&outside).expect("remove outside header");
        fs::remove_dir_all(&root).expect("remove fixture directory");
    }

    #[test]
    fn metal_implementations_beyond_the_buffer_table_are_rejected() {
        let root = fixture("slots");
        let source = root.join("ops.seismic");
        // 29 tensor parameters + 1 tensor result + words + scalar slots = 32.
        let parameters = (0..29)
            .map(|index| format!("x{index}: &tensor[N] f32"))
            .collect::<Vec<_>>()
            .join(", ");
        fs::write(
            &source,
            format!("fn wide[N]({parameters}) -> tensor[N] f32:\n    let mut output = tensor[N] f32\n    for i in 0..N:\n        output[i] = x0[i]\n    return output\n\nnative wide for metal from \"native/wide.metal\":\n    launch wide:\n        threadgroups (1, 1, 1)\n        threads_per_threadgroup (1, 1, 1)\n"),
        )
        .expect("Seismic fixture");
        fs::write(root.join("native/wide.metal"), "kernel void wide() {}\n")
            .expect("Metal fixture");
        match Build::new("fixture")
            .source(&source)
            .std(false)
            .out_dir(root.join("out"))
            .run().map_err(first_violation)
        {
            Err(BuildError::MetalBufferSlots {
                entry,
                slots,
                limit,
            }) => {
                assert_eq!(entry, "wide");
                assert_eq!(slots, 32);
                assert_eq!(limit, 31);
            }
            Err(other) => panic!("unexpected error: {other}"),
            Ok(_) => panic!("32 buffers must not build for Metal"),
        }
        fs::remove_dir_all(&root).expect("remove fixture directory");
    }
}
