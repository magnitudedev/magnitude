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
//! fn module() -> Result<&'static seismic::generated::Module, seismic::CheckedBundleError>;
//! pub const IDENTITY: &str = "<hex digest of sources + compiler semantic version>";
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
//!     pub fn for_device(device: &seismic::Device, options: seismic::PreparationOptions)
//!         -> Result<seismic::Kernel<Entry>, seismic::LoadError>;                 // monomorphic
//!     pub fn for_device_with(device: &seismic::Device, options: seismic::PreparationOptions, elements: Elements)
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
    EntryInfo, NativeImplementation, ParameterSummary, ParameterSummaryKind, ResultSummaryKind,
    SourceSet, TensorAccess,
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
    /// A launch-scoped GPU kernel does not match its declared forming guard
    /// and code-parameter template.
    NativeTemplate {
        entry: String,
        path: PathBuf,
        kernel: String,
        detail: String,
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
    /// A Vulkan asset (or an included file) uses a construct the generated
    /// prefix and suffix own: `#version`, `#extension`, the workgroup size,
    /// push constants, a `shared` declaration, `main`, or an unordered float
    /// subgroup sum.
    NativeReserved {
        entry: String,
        path: PathBuf,
        line: usize,
        column: usize,
        construct: String,
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
            Self::NativeTemplate { entry, path, kernel, detail } => write!(
                f,
                "native template for `{entry}` kernel `{kernel}` in {}: {detail}",
                path.display()
            ),
            Self::NativeInclude(e) => write!(f, "{e}"),
            Self::MetalBufferSlots { entry, slots, limit } => write!(
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
                "native Vulkan implementation of `{entry}`: {}:{line}:{column}: {construct}",
                path.display()
            ),
            Self::NativeKernelMissing { entry, kernel } => write!(
                f,
                "native Vulkan implementation of `{entry}` launches `{kernel}`, but its source defines no `void {kernel}()`"
            ),
            Self::CpuNative { path, reason } => write!(f, "CPU native source {}: {reason}", path.display()),
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

mod internals {
    use super::*;
    use seismic_lang::checked::{ElementSummary, ElementTarget, NativeElementCoverage};
    use seismic_lang::registry::{self, CodeInterpretation, PlaneEncoding, RepresentationInfo};
    use std::collections::HashSet;
    use std::fs;

    /// The native implementations of one entry. Metal and CUDA sources
    /// travel in the checked bundle; a CPU source is compiled into the
    /// generated bindings.
    struct EntryNative<'a> {
        entry: seismic_lang::ids::EntryId,
        cpu: Option<CpuAsset<'a>>,
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
        for entry in module.entries() {
            let mut any = false;
            let mut cpu = None;
            for backend in BackendName::ALL {
                let Some(definition) = module.native_implementation(entry.id, backend) else {
                    continue;
                };
                any = true;
                let source = module
                    .native_asset(entry.id, backend)
                    .expect("shared loader captures native assets");
                let extension = match backend {
                    // CPU assets are Rust, compiled into the embedding crate
                    // from their authored path, so compiler diagnostics and
                    // tools point at the authored file.
                    BackendName::Cpu => {
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
                    }
                    BackendName::Metal => "metal",
                    BackendName::Cuda => "cu",
                    BackendName::Vulkan => "comp",
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
                let allowed = native_abi_symbols(entry, definition, backend);
                for file in std::iter::once(&files.asset).chain(&files.includes) {
                    validate_native_abi(entry, &allowed, &file.path, &file.text)?;
                    if backend == BackendName::Vulkan {
                        validate_vulkan_reserved(entry, &file.path, &file.text)?;
                    }
                }
                if definition.launch_scoped()
                    && matches!(backend, BackendName::Metal | BackendName::Cuda)
                {
                    validate_launch_templates(
                        entry,
                        definition,
                        backend,
                        &files.asset.path,
                        source,
                    )?;
                }
                match backend {
                    BackendName::Metal => validate_metal_buffer_slots(entry, definition)?,
                    BackendName::Vulkan => validate_vulkan_kernels(entry, definition, source)?,
                    BackendName::Cpu | BackendName::Cuda => {}
                }
            }
            if any {
                natives.push(EntryNative {
                    entry: entry.id,
                    cpu,
                });
            }
        }
        Ok(natives)
    }

    fn validate_native_abi(
        entry: &EntryInfo,
        allowed: &HashSet<String>,
        path: &std::path::Path,
        source: &str,
    ) -> Result<(), BuildError> {
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
            if !allowed.contains(symbol) {
                return Err(failure(offset, symbol));
            }
        }
        Ok(())
    }

    fn validate_launch_templates(
        entry: &EntryInfo,
        definition: &NativeImplementation,
        backend: seismic_lang::registry::BackendName,
        path: &std::path::Path,
        source: &str,
    ) -> Result<(), BuildError> {
        let tokens = identifiers(source);
        for launch in &definition.launches {
            let fail = |detail: String| BuildError::NativeTemplate {
                entry: entry.name.clone(),
                path: path.to_path_buf(),
                kernel: launch.kernel.clone(),
                detail,
            };
            let definitions = tokens
                .windows(3)
                .filter_map(|triple| {
                    let [(qualifier, offset), (void, _), (name, _)] = triple else {
                        unreachable!()
                    };
                    let matches = match backend {
                        BackendName::Metal => *qualifier == "kernel" && *void == "void",
                        BackendName::Cuda => *qualifier == "__global__" && *void == "void",
                        BackendName::Cpu | BackendName::Vulkan => false,
                    };
                    (matches && *name == launch.kernel).then_some(*offset)
                })
                .collect::<Vec<_>>();
            let [kernel_at] = definitions.as_slice() else {
                return Err(fail(format!(
                    "expected one GPU kernel definition for `{}`",
                    launch.kernel
                )));
            };
            let guard = format!("SEISMIC_FORMING_{}", native_macro(&launch.kernel));
            let mut guards = Vec::<(Option<&str>, usize)>::new();
            let mut offset = 0;
            for line in source.split_inclusive('\n') {
                if offset >= *kernel_at {
                    break;
                }
                let directive = line.trim_start();
                if let Some(name) = directive.strip_prefix("#ifdef ") {
                    guards.push((Some(name.trim()), offset));
                } else if directive.starts_with("#if ") || directive.starts_with("#ifndef ") {
                    guards.push((None, offset));
                } else if directive.starts_with("#else") || directive.starts_with("#elif") {
                    if let Some(top) = guards.last_mut() {
                        top.0 = None;
                    }
                } else if directive.starts_with("#endif") {
                    guards.pop();
                }
                offset += line.len();
            }
            let Some((_, guarded_at)) = guards
                .iter()
                .rev()
                .find(|(name, _)| *name == Some(guard.as_str()))
            else {
                return Err(fail(format!("definition must be inside `#ifdef {guard}`")));
            };
            let template = tokens
                .iter()
                .rev()
                .find(|(name, at)| *name == "template" && *at > *guarded_at && *at < *kernel_at)
                .map(|(_, at)| *at);
            let reads = launch.parameters();
            let expected = definition
                .params
                .iter()
                .filter(|param| {
                    param.code
                        && (reads.contains(&param.name)
                            || !definition
                                .launches
                                .iter()
                                .any(|other| other.parameters().contains(&param.name)))
                })
                .map(|param| param.name.as_str())
                .chain(
                    launch
                        .params
                        .iter()
                        .filter(|param| param.code)
                        .map(|param| param.name.as_str()),
                )
                .collect::<Vec<_>>();
            let actual = template.and_then(|at| {
                let header = source[at + "template".len()..*kernel_at].trim();
                let body = header.strip_prefix('<')?.strip_suffix('>')?;
                if body.trim().is_empty() {
                    return Some(Vec::new());
                }
                body.split(',')
                    .map(|field| {
                        let mut words = field.split_whitespace();
                        match (words.next(), words.next(), words.next()) {
                            (Some("int" | "uint" | "unsigned"), Some(name), None) => Some(name),
                            _ => None,
                        }
                    })
                    .collect::<Option<Vec<_>>>()
            });
            if actual.as_deref() != Some(expected.as_slice())
                && !(expected.is_empty() && actual.is_none())
            {
                return Err(fail(format!(
                    "template parameters must be `{:?}` in declaration order; found {actual:?}",
                    expected
                )));
            }
        }
        Ok(())
    }

    /// Metal's argument table holds 31 buffers. The native ABI binds every
    /// tensor parameter, every tensor result, every scratch buffer, the
    /// argument words and the scalar-result slots; all of them are known from
    /// the declaration, so the limit is checked here rather than at encode.
    fn validate_metal_buffer_slots(
        entry: &EntryInfo,
        definition: &NativeImplementation,
    ) -> Result<(), BuildError> {
        let tensors = entry
            .parameters
            .iter()
            .filter(|parameter| matches!(parameter.kind, ParameterSummaryKind::Tensor { .. }))
            .count()
            + entry
                .results
                .iter()
                .filter(|result| matches!(result.kind, ResultSummaryKind::Tensor { .. }))
                .count();
        let slots = tensors + definition.scratch.len() + 2;
        if slots > METAL_BUFFER_SLOTS {
            return Err(BuildError::MetalBufferSlots {
                entry: entry.name.clone(),
                slots,
                limit: METAL_BUFFER_SLOTS,
            });
        }
        Ok(())
    }

    /// Metal's per-stage buffer argument table size.
    const METAL_BUFFER_SLOTS: usize = 31;

    /// The typed group-memory views of the Vulkan prefix
    /// (`seismic_runtime`'s `VULKAN_SHARED_VIEWS`), as `SEISMIC_SHARED_<view>`.
    const VULKAN_SHARED_VIEWS: [&str; 8] =
        ["F32", "F16", "BF16", "U8", "U16", "U32", "I32", "UVEC4"];

    /// Identifiers a Vulkan asset may not use, with why: the prefix and
    /// suffix own the workgroup size, push constants, group memory and
    /// `main`; float subgroup sums have an implementation-defined order.
    const VULKAN_RESERVED_IDENTIFIERS: [(&str, &str); 13] = [
        ("local_size_x", "the workgroup size is generated"),
        ("local_size_y", "the workgroup size is generated"),
        ("local_size_z", "the workgroup size is generated"),
        ("local_size_x_id", "the workgroup size is generated"),
        ("local_size_y_id", "the workgroup size is generated"),
        ("local_size_z_id", "the workgroup size is generated"),
        ("push_constant", "the push constant is the generated argument block address"),
        ("shared", "group memory is the generated `seismic_shared_*` views of `shared_bytes`"),
        ("main", "`main` is generated and calls the launch's kernel"),
        ("subgroupAdd", "float subgroup sums have no fixed order; use `seismic_subgroup_sum_f32` or `seismic_redux_add_*`"),
        ("subgroupInclusiveAdd", "subgroup scans have no fixed order"),
        ("subgroupExclusiveAdd", "subgroup scans have no fixed order"),
        ("subgroupClusteredAdd", "clustered sums have no fixed order"),
    ];

    /// Directives the prefix owns.
    const VULKAN_RESERVED_DIRECTIVES: [&str; 2] = ["version", "extension"];

    fn location(source: &str, offset: usize) -> (usize, usize) {
        let prefix = &source[..offset];
        let line = prefix.bytes().filter(|byte| *byte == b'\n').count() + 1;
        let column = prefix
            .rsplit_once('\n')
            .map_or(prefix.len() + 1, |(_, tail)| tail.len() + 1);
        (line, column)
    }

    fn validate_vulkan_reserved(
        entry: &EntryInfo,
        path: &std::path::Path,
        source: &str,
    ) -> Result<(), BuildError> {
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
        let mut offset = 0;
        for line in source.split_inclusive('\n') {
            let trimmed = line.trim_start();
            if let Some(directive) = trimmed.strip_prefix('#') {
                let name = directive
                    .trim_start()
                    .split(|c: char| !c.is_ascii_alphanumeric() && c != '_')
                    .next()
                    .unwrap_or("");
                if VULKAN_RESERVED_DIRECTIVES.contains(&name) {
                    return Err(reserved(
                        offset + (line.len() - trimmed.len()),
                        format!("`#{name}` is generated by the Vulkan prefix"),
                    ));
                }
            }
            offset += line.len();
        }
        for (identifier, offset) in identifiers(source) {
            if let Some((_, why)) = VULKAN_RESERVED_IDENTIFIERS
                .iter()
                .find(|(name, _)| *name == identifier)
            {
                return Err(reserved(
                    offset,
                    format!("`{identifier}` is reserved: {why}"),
                ));
            }
        }
        Ok(())
    }

    /// Every launch's kernel is a `void <kernel>()` function of the expanded
    /// source (lexically, like the symbol checks).
    fn validate_vulkan_kernels(
        entry: &EntryInfo,
        definition: &NativeImplementation,
        source: &str,
    ) -> Result<(), BuildError> {
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
                return Err(BuildError::NativeKernelMissing {
                    entry: entry.name.clone(),
                    kernel: launch.kernel.clone(),
                });
            }
        }
        Ok(())
    }

    fn native_abi_symbols(
        entry: &EntryInfo,
        definition: &NativeImplementation,
        backend: seismic_lang::registry::BackendName,
    ) -> HashSet<String> {
        use seismic_lang::registry::BackendName;
        let mut symbols = HashSet::new();
        symbols.insert("SEISMIC_BUFFER_SCALAR_RESULTS".to_owned());
        if !definition.launch_scoped() {
            for parameter in &definition.params {
                symbols.insert(format!("SEISMIC_TUNE_{}", native_macro(&parameter.name)));
            }
        } else {
            for parameter in definition.params.iter().filter(|parameter| !parameter.code) {
                symbols.insert(format!("SEISMIC_RUNTIME_{}", native_macro(&parameter.name)));
            }
            for launch in &definition.launches {
                symbols.insert(format!("SEISMIC_FORMING_{}", native_macro(&launch.kernel)));
                for parameter in launch.params.iter().filter(|parameter| !parameter.code) {
                    symbols.insert(format!("SEISMIC_RUNTIME_{}", native_macro(&parameter.name)));
                }
            }
        }
        for scratch in &definition.scratch {
            symbols.insert(format!(
                "SEISMIC_BUFFER_SCRATCH_{}",
                native_macro(&scratch.name)
            ));
        }
        match backend {
            BackendName::Cuda => {
                symbols.insert("SEISMIC_BUFFER_WORDS".to_owned());
                for symbol in [
                    "SEISMIC_KERNEL_PARAMS",
                    "SEISMIC_PTR",
                    "SEISMIC_PTR_",
                    "SEISMIC_SCALAR_RESULTS",
                ] {
                    symbols.insert(symbol.to_owned());
                }
            }
            // Vulkan words live in the argument block, read through
            // `seismic_words`; there is no words buffer.
            BackendName::Vulkan => {
                for symbol in [
                    "SEISMIC_PTR",
                    "SEISMIC_READONLY",
                    "SEISMIC_READONLY_",
                    "SEISMIC_SCALAR_RESULTS",
                    "SEISMIC_KERNEL",
                    "SEISMIC_SUBGROUP_LANES",
                    "SEISMIC_LANE",
                    "SEISMIC_SUBGROUP",
                    "SEISMIC_SUBGROUPS",
                    "SEISMIC_HAS_FLOAT16",
                    "SEISMIC_HAS_MATRIX",
                    "SEISMIC_HAS_WIDE_ACCUMULATORS",
                    "SEISMIC_HAS_MIXED_DOT",
                    "SEISMIC_HAS_F32_ATOMIC_ADD",
                    "SEISMIC_HAS_SHARED_INT64_ATOMICS",
                ] {
                    symbols.insert(symbol.to_owned());
                }
                for view in VULKAN_SHARED_VIEWS {
                    symbols.insert(format!("SEISMIC_SHARED_{view}"));
                }
            }
            BackendName::Cpu | BackendName::Metal => {
                symbols.insert("SEISMIC_BUFFER_WORDS".to_owned());
            }
        }
        for dimension in &entry.dimensions {
            symbols.insert(format!("SEISMIC_DIM_{}", native_macro(dimension)));
        }
        for element in &entry.element_parameters {
            add_representation_symbols(
                &mut symbols,
                &format!("SEISMIC_ELEMENT_{}", native_macro(element)),
                registry::representations().iter(),
            );
        }

        for (ordinal, parameter) in entry.parameters.iter().enumerate() {
            let unique = entry
                .parameters
                .iter()
                .filter(|candidate| candidate.name == parameter.name)
                .count()
                == 1;
            let name = native_macro(&parameter.name);
            match &parameter.kind {
                ParameterSummaryKind::Tensor { rank, element, .. } => {
                    let ordinal_prefix = format!("SEISMIC_PARAM_{ordinal}");
                    symbols.insert(format!("{ordinal_prefix}_BUFFER"));
                    if unique {
                        symbols.insert(format!("SEISMIC_BUFFER_{name}"));
                    }
                    for axis in 0..*rank {
                        symbols.insert(format!("{ordinal_prefix}_EXTENT_{axis}"));
                        symbols.insert(format!("{ordinal_prefix}_STRIDE_{axis}"));
                        if unique {
                            symbols.insert(format!("SEISMIC_{name}_EXTENT_{axis}"));
                            symbols.insert(format!("SEISMIC_{name}_STRIDE_{axis}"));
                        }
                    }
                    add_element_representation_symbols(&mut symbols, &ordinal_prefix, element);
                    if unique {
                        add_element_representation_symbols(
                            &mut symbols,
                            &format!("SEISMIC_{name}"),
                            element,
                        );
                    }
                }
                ParameterSummaryKind::Scalar(_) | ParameterSummaryKind::Index => {
                    symbols.insert(format!("SEISMIC_PARAM_{ordinal}"));
                    if unique {
                        symbols.insert(format!("SEISMIC_PARAM_{name}"));
                    }
                }
                ParameterSummaryKind::Range => {
                    symbols.insert(format!("SEISMIC_PARAM_{ordinal}_START"));
                    symbols.insert(format!("SEISMIC_PARAM_{ordinal}_END"));
                    if unique {
                        symbols.insert(format!("SEISMIC_PARAM_{name}_START"));
                        symbols.insert(format!("SEISMIC_PARAM_{name}_END"));
                    }
                }
            }
        }

        for (ordinal, result) in entry.results.iter().enumerate() {
            match &result.kind {
                ResultSummaryKind::Tensor { rank, element } => {
                    let prefix = format!("SEISMIC_RESULT_{ordinal}");
                    symbols.insert(format!("{prefix}_BUFFER"));
                    for axis in 0..*rank {
                        symbols.insert(format!("{prefix}_EXTENT_{axis}"));
                        symbols.insert(format!("{prefix}_STRIDE_{axis}"));
                    }
                    add_element_representation_symbols(&mut symbols, &prefix, element);
                }
                ResultSummaryKind::Scalar(_)
                | ResultSummaryKind::Index
                | ResultSummaryKind::Range => {
                    symbols.insert(format!("SEISMIC_RESULT_{ordinal}_WORD"));
                }
            }
        }
        symbols
    }

    fn add_element_representation_symbols(
        symbols: &mut HashSet<String>,
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
        symbols: &mut HashSet<String>,
        prefix: &str,
        representations: impl Iterator<Item = &'a RepresentationInfo>,
    ) {
        // Authored helper macros commonly receive the ABI family prefix and
        // token-paste suffixes such as `_PACKET_SIZE` onto it.
        symbols.insert(prefix.to_owned());
        for representation in representations {
            symbols.insert(format!(
                "{prefix}_REPRESENTATION_{}",
                native_macro(representation.representation)
            ));
            symbols.insert(format!(
                "{prefix}_DECODED_{}",
                native_macro(representation.decoded.name())
            ));
            symbols.insert(format!("{prefix}_PACKET_SIZE"));
            symbols.insert(format!("{prefix}_PACKET_ALIGNMENT"));
            symbols.insert(format!("{prefix}_LOGICAL_GROUP"));
            symbols.insert(format!("{prefix}_PLANE_COUNT"));
            match &representation.kind {
                registry::RepresentationKind::Dense(_) => {
                    symbols.insert(format!("{prefix}_KIND_DENSE"));
                }
                registry::RepresentationKind::External(_) => {
                    symbols.insert(format!("{prefix}_KIND_EXTERNAL"));
                }
                // The row-layout ABI of the frozen weight-storage interface
                // (native-program-execution.md, "Weight storage").
                registry::RepresentationKind::PackedRows(layout) => {
                    symbols.insert(format!("{prefix}_KIND_PACKED"));
                    symbols.insert(format!(
                        "{prefix}_LAYOUT_{}",
                        native_macro(layout.layout.as_str())
                    ));
                    symbols.insert(format!("{prefix}_ROW_STRIDE_BYTES"));
                    for suffix in [
                        "ROW_GROUPS",
                        "GROUP_MULTIPLE",
                        "ROW_ALIGNMENT",
                        "TILE_ROWS",
                        "CODE_BITS",
                        "MMA_KBLOCK",
                        "MMA_LANES",
                    ] {
                        symbols.insert(format!("{prefix}_{suffix}"));
                    }
                    for plane in &layout.planes {
                        let plane_prefix = format!("{prefix}_PLANE_{}", native_macro(plane.name));
                        symbols.insert(plane_prefix.clone());
                        for suffix in [
                            "ROW_OFFSET",
                            "BYTES_PER_ROW",
                            "BYTES_PER_GROUP",
                            "CODE_SHIFT",
                            "CODE_BITS",
                        ] {
                            symbols.insert(format!("{plane_prefix}_{suffix}"));
                        }
                    }
                }
                registry::RepresentationKind::Packed(layout) => {
                    symbols.insert(format!("{prefix}_KIND_PACKED"));
                    symbols.insert(format!("{prefix}_LAYOUT_PACKET"));
                    for (ordinal, plane) in layout.planes.iter().enumerate() {
                        let plane_prefix = format!("{prefix}_PLANE_{ordinal}");
                        symbols.insert(format!("{plane_prefix}_NAME_{}", native_macro(plane.name)));
                        for suffix in [
                            "OFFSET",
                            "BYTES_PER_GROUP",
                            "ALIGNMENT",
                            "GROUP",
                            "FIELDS",
                            "ENTRY_BITS",
                        ] {
                            symbols.insert(format!("{plane_prefix}_{suffix}"));
                        }
                        symbols.insert(format!(
                            "{plane_prefix}_STORAGE_{}",
                            native_macro(plane.storage_dtype.name())
                        ));
                        match &plane.encoding {
                            PlaneEncoding::Dense(dtype) => {
                                symbols.insert(format!("{plane_prefix}_ENCODING_DENSE"));
                                symbols.insert(format!(
                                    "{plane_prefix}_ENCODING_DTYPE_{}",
                                    native_macro(dtype.name())
                                ));
                            }
                            PlaneEncoding::Packed { interpretation, .. } => {
                                symbols.insert(format!("{plane_prefix}_ENCODING_PACKED"));
                                symbols.insert(format!("{plane_prefix}_ENCODING_BITS"));
                                add_code_symbols(symbols, &plane_prefix, interpretation);
                            }
                            PlaneEncoding::FloatCode { format } => {
                                symbols.insert(format!("{plane_prefix}_ENCODING_FLOAT_CODE"));
                                let format = match format {
                                    registry::FloatCodeFormat::E2M1 => "E2M1",
                                    registry::FloatCodeFormat::E4M3 => "E4M3",
                                    registry::FloatCodeFormat::UE4M3 => "UE4M3",
                                };
                                symbols
                                    .insert(format!("{plane_prefix}_ENCODING_FLOAT_CODE_{format}"));
                                symbols.insert(format!("{plane_prefix}_ENCODING_BITS"));
                            }
                        }
                    }
                }
            }
        }
    }

    fn add_code_symbols(
        symbols: &mut HashSet<String>,
        prefix: &str,
        interpretation: &CodeInterpretation,
    ) {
        match interpretation {
            CodeInterpretation::Unsigned => {
                symbols.insert(format!("{prefix}_CODE_UNSIGNED"));
            }
            CodeInterpretation::TwosComplement => {
                symbols.insert(format!("{prefix}_CODE_TWOS_COMPLEMENT"));
            }
            CodeInterpretation::Offset(_) => {
                symbols.insert(format!("{prefix}_CODE_OFFSET"));
                symbols.insert(format!("{prefix}_CODE_OFFSET_VALUE"));
            }
            CodeInterpretation::Table(values) => {
                symbols.insert(format!("{prefix}_CODE_TABLE"));
                symbols.insert(format!("{prefix}_CODE_TABLE_COUNT"));
                for ordinal in 0..values.len() {
                    symbols.insert(format!("{prefix}_CODE_TABLE_{ordinal}"));
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

    /// Every identifier outside comments and literals, with its offset.
    fn identifiers(source: &str) -> Vec<(&str, usize)> {
        let bytes = source.as_bytes();
        let mut identifiers = Vec::new();
        let mut index = 0;
        while index < bytes.len() {
            match bytes[index] {
                b'/' if bytes.get(index + 1) == Some(&b'/') => {
                    index += 2;
                    while index < bytes.len() && bytes[index] != b'\n' {
                        index += 1;
                    }
                }
                b'/' if bytes.get(index + 1) == Some(&b'*') => {
                    index += 2;
                    while index + 1 < bytes.len()
                        && !(bytes[index] == b'*' && bytes[index + 1] == b'/')
                    {
                        index += 1;
                    }
                    index = (index + 2).min(bytes.len());
                }
                b'"' | b'\'' => {
                    let quote = bytes[index];
                    index += 1;
                    while index < bytes.len() {
                        if bytes[index] == b'\\' {
                            index = (index + 2).min(bytes.len());
                        } else if bytes[index] == quote {
                            index += 1;
                            break;
                        } else {
                            index += 1;
                        }
                    }
                }
                byte if byte == b'_' || byte.is_ascii_alphabetic() => {
                    let start = index;
                    index += 1;
                    while index < bytes.len()
                        && (bytes[index] == b'_' || bytes[index].is_ascii_alphanumeric())
                    {
                        index += 1;
                    }
                    identifiers.push((&source[start..index], start));
                }
                _ => index += 1,
            }
        }
        identifiers
    }

    fn native_macro(name: &str) -> String {
        name.chars()
            .map(|character| {
                if character.is_ascii_alphanumeric() {
                    character.to_ascii_uppercase()
                } else {
                    '_'
                }
            })
            .collect()
    }

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
        out.push_str("fn module() -> Result<&'static seismic::generated::Module, seismic::CheckedBundleError> { seismic::generated::module_from_bundle(&__MODULE, __BUNDLE) }\n");
        out.push_str(&format!("pub const IDENTITY: &str = {:?};\n", identity));
        if let Some(library) = cpu_library {
            render_cpu_library(&mut out, library);
        }
        for entry in module.entries() {
            let native = native_assets.iter().find(|native| native.entry == entry.id);
            render_entry(&mut out, entry, native, cpu_library)?;
        }
        Ok(out)
    }

    fn render_entry(
        out: &mut String,
        entry: &EntryInfo,
        native: Option<&EntryNative<'_>>,
        cpu_library: Option<&CpuLibrary>,
    ) -> Result<(), BuildError> {
        let module_name = ident(&entry.name);
        out.push_str(&format!("pub mod {module_name} {{\n"));
        let borrowed = entry.parameters.iter().any(|parameter| {
            matches!(
                parameter.kind,
                ParameterSummaryKind::Tensor {
                    access: TensorAccess::Shared | TensorAccess::Mutable,
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
                ident(&parameter.name),
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
                ident(&parameter.name),
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
                    ident(parameter)
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
        let scalar_words = entry
            .results
            .iter()
            .map(|result| match &result.kind {
                ResultSummaryKind::Range => 2,
                ResultSummaryKind::Scalar(_) | ResultSummaryKind::Index => 1,
                ResultSummaryKind::Tensor { .. } => 0,
            })
            .sum::<u64>();
        // Argument words travel by value; a prepared native implementation
        // owns only its scalar-result slots.
        let invocation_workspace_bytes = (scalar_words * 8).max(1);
        out.push_str(&format!(
            "    const NATIVE_INVOCATION_WORKSPACE_BYTES: u64 = {invocation_workspace_bytes};\n"
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
            let field = ident(&source.name);
            for parameter in parameters {
                let expression = parameter_access(&field, &parameter.path);
                match &parameter.kind {
                    ParameterSummaryKind::Tensor { access, .. } => match access {
                        TensorAccess::Owned => {
                            out.push_str(&format!("      encoder.tensor(&{expression});\n"));
                        }
                        TensorAccess::Shared | TensorAccess::Mutable => {
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
        out.push_str("    fn decode(mut results: seismic::generated::DecodedResults) -> Results { Results {\n");
        for result in &entry.results {
            let getter = match &result.kind {
                ResultSummaryKind::Tensor { .. } => "take_tensor",
                ResultSummaryKind::Scalar(dtype) => match dtype {
                    DType::F32 => "take_f32",
                    DType::F16 => "take_f16",
                    DType::BF16 => "take_bf16",
                    DType::I32 => "take_i32",
                    DType::U32 => "take_u32",
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
            let field = ident(&source.name);
            for parameter in parameters {
                let expression = parameter_access(&field, &parameter.path);
                match &parameter.kind {
                    ParameterSummaryKind::Tensor { access, .. } => {
                        let method = match access {
                            TensorAccess::Owned => "owned_tensor",
                            TensorAccess::Shared => "shared_tensor",
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
        out.push_str("    fn decode_workflow(mut results: seismic::generated::PendingWorkflowResults) -> WorkflowResults { WorkflowResults {\n");
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
        out.push_str("    fn workflow_outputs(results: WorkflowResults) -> Vec<seismic::generated::WorkflowResultRef> { vec![\n");
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
            out.push_str("  pub fn for_device(device: &seismic::Device, options: seismic::PreparationOptions) -> Result<seismic::Kernel<Entry>, seismic::LoadError> { seismic::generated::prepare::<Entry>(device, options, &[]) }\n");
        } else {
            out.push_str("  pub fn for_device_with(device: &seismic::Device, options: seismic::PreparationOptions, elements: Elements) -> Result<seismic::Kernel<Entry>, seismic::LoadError> {\n");
            out.push_str("    seismic::generated::prepare::<Entry>(device, options, &[\n");
            for parameter in &entry.element_parameters {
                out.push_str(&format!(
                    "      ({:?}, elements.{}),\n",
                    parameter,
                    ident(parameter)
                ));
            }
            out.push_str("    ])\n  }\n");
        }
        if entry.element_parameters.is_empty() {
            out.push_str("  pub fn start_feedback(device: &seismic::Device, precision: seismic::PrecisionPolicy, options: seismic::FeedbackOptions) -> Result<(seismic::FeedbackPreparation<'_, Entry>, seismic::Kernel<Entry>), seismic::LoadError> { seismic::generated::start_feedback::<Entry>(device, precision, options, &[]) }\n");
        } else {
            out.push_str("  pub fn start_feedback_with(device: &seismic::Device, precision: seismic::PrecisionPolicy, options: seismic::FeedbackOptions, elements: Elements) -> Result<(seismic::FeedbackPreparation<'_, Entry>, seismic::Kernel<Entry>), seismic::LoadError> {\n");
            out.push_str(
                "    seismic::generated::start_feedback::<Entry>(device, precision, options, &[\n",
            );
            for parameter in &entry.element_parameters {
                out.push_str(&format!(
                    "      ({:?}, elements.{}),\n",
                    parameter,
                    ident(parameter)
                ));
            }
            out.push_str("    ])\n  }\n");
        }
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
                    ident(parameter)
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
            out.push_str("  pub fn native_tune_with(device: &seismic::Device, elements: Elements, statics: &seismic::NativeSpecialization, points: Vec<seismic::TuningPoint<'_, Entry>>, validation: seismic::Validation, strategy: seismic::Strategy) -> Result<seismic::TuningResult, seismic::TuneError> {\n");
            out.push_str("    seismic::generated::tune_native::<Entry>(device, statics, &[\n");
            element_list(out);
            out.push_str(&format!(
                "    ], {cpu}, points, validation, strategy)\n  }}\n"
            ));
            out.push_str("  pub fn native_digest_with(device: &seismic::Device, elements: Elements, statics: &seismic::NativeSpecialization) -> Result<String, seismic::TuneError> {\n");
            out.push_str("    seismic::generated::digest_native::<Entry>(device, statics, &[\n");
            element_list(out);
            out.push_str(&format!("    ], {cpu})\n  }}\n"));
        } else {
            out.push_str(&format!("  pub fn native_for_device(device: &seismic::Device, specialization: &seismic::NativeSpecialization) -> Result<seismic::NativeKernel<Entry>, seismic::LoadError> {{ seismic::generated::prepare_native::<Entry>(device, specialization, &[], {cpu}) }}\n"));
            out.push_str(&format!("  pub fn native_tune(device: &seismic::Device, statics: &seismic::NativeSpecialization, points: Vec<seismic::TuningPoint<'_, Entry>>, validation: seismic::Validation, strategy: seismic::Strategy) -> Result<seismic::TuningResult, seismic::TuneError> {{ seismic::generated::tune_native::<Entry>(device, statics, &[], {cpu}, points, validation, strategy) }}\n"));
            out.push_str(&format!("  pub fn native_digest(device: &seismic::Device, statics: &seismic::NativeSpecialization) -> Result<String, seismic::TuneError> {{ seismic::generated::digest_native::<Entry>(device, statics, &[], {cpu}) }}\n"));
        }
        out.push_str("  /// The checked native implementation for the device's backend.\n");
        out.push_str("  pub fn native_implementation(device: &seismic::Device) -> Result<Option<seismic::NativeImplementation>, seismic::CheckedBundleError> { seismic::generated::native_implementation::<Entry>(device) }\n");
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
                    let modules = path
                        .strip_prefix(root)
                        .expect("a scanned file lies under its root")
                        .with_extension("")
                        .components()
                        .map(|component| component.as_os_str().to_string_lossy().into_owned())
                        .collect::<Vec<_>>();
                    for module in &modules {
                        if ident(module) != *module {
                            return Err(BuildError::CpuNative {
                                path,
                                reason: format!("`{module}` is not a Rust module name; CPU library paths become module paths"),
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
            DType::Bool => unreachable!("the checker admits no `bool` CPU element"),
        }
    }

    /// How a CPU form binds each element parameter of an entry.
    struct CpuElements {
        /// Parameters the entry stores, with the dense types the form is
        /// monomorphized over (declared, or the floating default).
        dense: Vec<(String, Vec<DType>)>,
        /// External sources the entry only converts.
        external: Vec<String>,
        /// Conversion targets: packed results written by a conversion.
        packed: Vec<String>,
    }

    /// Every other element parameter is only read: a weight operand,
    /// reaching the kernel through its resolved components.
    fn cpu_elements(entry: &EntryInfo, definition: &NativeImplementation) -> CpuElements {
        let domain = &entry.element_domain;
        let converted = |name: &str| {
            domain
                .conversions()
                .iter()
                .any(|conversion| conversion.target == ElementTarget::Parameter(name.to_owned()))
        };
        let mut elements = CpuElements {
            dense: Vec::new(),
            external: Vec::new(),
            packed: Vec::new(),
        };
        for parameter in domain.parameters() {
            let name = parameter.name.clone();
            if converted(&name) {
                elements.packed.push(name);
            } else if parameter.uses.stored || parameter.uses.partial_copy {
                let dtypes = definition
                    .elements
                    .iter()
                    .find(|coverage| coverage.parameter == name)
                    .map(|coverage| coverage.dtypes.clone())
                    .unwrap_or_else(|| NativeElementCoverage::DEFAULT.to_vec());
                elements.dense.push((name, dtypes));
            } else if parameter.uses.conversion_source && !parameter.uses.decoded_read {
                elements.external.push(name);
            }
        }
        elements
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
            ElementSummary::Parameter(name)
                if elements.dense.iter().any(|(dense, _)| dense == name) =>
            {
                CpuView::Dense(format!("E::{}", ident(name)))
            }
            ElementSummary::Parameter(name) if elements.external.contains(name) => {
                CpuView::External
            }
            ElementSummary::Parameter(name) if elements.packed.contains(name) => CpuView::Packed,
            ElementSummary::Parameter(_) => CpuView::Weights,
            ElementSummary::Fixed(name) => match name.as_str() {
                "f32" => CpuView::Dense("seismic::cpu::F32".to_owned()),
                "bf16" => CpuView::Dense("seismic::cpu::Bf16".to_owned()),
                "f16" => CpuView::Dense("seismic::cpu::F16".to_owned()),
                "i32" => CpuView::Scalars("i32"),
                "u32" => CpuView::Scalars("u32"),
                "bool" => CpuView::Scalars("u8"),
                _ => CpuView::Weights,
            },
        }
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
            out.push_str(&format!("      type {}: {bound};\n", ident(element)));
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
                ident(dimension).to_lowercase()
            ));
            word += 1;
        }
        let mut buffer = 0usize;
        let tensor = |out: &mut String,
                      name: &str,
                      view: CpuView,
                      buffer: usize,
                      word: usize,
                      rank: usize| {
            let extents = (0..rank)
                .map(|axis| format!("self.invocation.word({})", word + axis))
                .collect::<Vec<_>>()
                .join(", ");
            let strides = (0..rank)
                .map(|axis| format!("self.invocation.word({})", word + rank + axis))
                .collect::<Vec<_>>()
                .join(", ");
            let safety =
                "// SAFETY: the runtime binds every view inside its allocation for the call.";
            match view {
                CpuView::Dense(element) => out.push_str(&format!(
                    "      pub fn {name}(&self) -> seismic::cpu::Tensor<'a, {element}, {rank}> {{ {safety}\n        unsafe {{ seismic::cpu::Tensor::from_raw(self.invocation.buffer({buffer}), [{extents}], [{strides}]) }} }}\n"
                )),
                CpuView::Scalars(scalar) => out.push_str(&format!(
                    "      pub fn {name}(&self) -> seismic::cpu::Scalars<'a, {scalar}, {rank}> {{ {safety}\n        unsafe {{ seismic::cpu::Scalars::from_raw(self.invocation.buffer({buffer}), [{extents}], [{strides}]) }} }}\n"
                )),
                CpuView::Weights => out.push_str(&format!(
                    "      pub fn {name}(&self) -> seismic::cpu::Weights<'a> {{ {safety}\n        unsafe {{ seismic::cpu::Weights::from_tensor(self.invocation.buffer({buffer}), &[{extents}], &[{strides}], self.invocation.weights({buffer})) }} }}\n"
                )),
                CpuView::External => out.push_str(&format!(
                    "      pub fn {name}(&self) -> seismic::cpu::External<'a> {{ {safety}\n        unsafe {{ seismic::cpu::External::from_tensor(self.invocation.buffer({buffer}), &[{extents}], &[{strides}], self.invocation.representation({buffer})) }} }}\n"
                )),
                CpuView::Packed => out.push_str(&format!(
                    "      pub fn {name}(&self) -> seismic::cpu::Packed<'a> {{ {safety}\n        unsafe {{ seismic::cpu::Packed::from_tensor(self.invocation.buffer({buffer}), &[{extents}], &[{strides}], self.invocation.representation({buffer})) }} }}\n"
                )),
            }
        };
        for (ordinal, parameter) in entry.parameters.iter().enumerate() {
            let unique = entry
                .parameters
                .iter()
                .filter(|candidate| candidate.name == parameter.name)
                .count()
                == 1;
            let name = if unique {
                format!("arg_{}", ident(&parameter.name).to_lowercase())
            } else {
                format!("arg_{ordinal}")
            };
            match &parameter.kind {
                ParameterSummaryKind::Tensor { rank, element, .. } => {
                    let rank = *rank as usize;
                    tensor(out, &name, cpu_view(element, &elements), buffer, word, rank);
                    buffer += 1;
                    word += rank * 2;
                }
                ParameterSummaryKind::Scalar(dtype) => {
                    let (ty, decode) = match dtype {
                        DType::F32 => ("f32", "f32::from_bits(value as u32)"),
                        DType::F16 | DType::BF16 => ("u16", "value as u16"),
                        DType::I32 => ("i32", "value as u32 as i32"),
                        DType::U32 => ("u32", "value as u32"),
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
                    tensor(out, &format!("result_{ordinal}"), view, buffer, word, rank);
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
        for scratch in &definition.scratch {
            out.push_str(&format!(
                "      pub fn scratch_{}(&self) -> seismic::cpu::Scratch<'a> {{ // SAFETY: the runtime places every scratch buffer for the call, 256-byte aligned.\n        unsafe {{ seismic::cpu::Scratch::from_raw(self.invocation.buffer({buffer})) }} }}\n",
                ident(&scratch.name).to_lowercase()
            ));
            buffer += 1;
        }
        for (index, parameter) in definition.declared_params().enumerate() {
            out.push_str(&format!(
                "      /// Tuning parameter `{}`.\n      pub fn param_{}(&self) -> u64 {{ self.invocation.param({index}) }}\n",
                parameter.name,
                ident(&parameter.name).to_lowercase()
            ));
        }
        out.push_str("    }\n");
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
                out.push_str(&format!(" type {} = seismic::cpu::{ty};", ident(element)));
            }
            out.push_str(" }\n");
        }

        // The tier boundaries: each form is a `#[target_feature]` function
        // of its tier, into which the authored kernel and everything it
        // calls inline.
        for tier in seismic_native_cpu::Tier::EVERY {
            for (launch_index, launch) in definition.launches.iter().enumerate() {
                for binding in &bindings {
                    let elements = binding_name(binding);
                    out.push_str(&format!(
                        "    #[cfg(target_arch = {arch:?})]\n    #[target_feature(enable = {features:?})]\n    unsafe fn launch_{launch_index}_{tier}_{elements}(invocation: &CpuInvocation<'_>, group: [u64; 3], shared: &mut [u8]) {{\n      // SAFETY: the runtime calls a form only on a device whose detected tier includes the form's.\n      let l = unsafe {{ <seismic::cpu::{token} as Isa>::assume_detected() }};\n      {kernel}::<seismic::cpu::{token}, {elements}>(l, &Context::new(invocation), group, shared)\n    }}\n",
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
        digest.update(library.digest);
        digest.update(seismic_native_cpu::VERSION.as_bytes());
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
                for tier in seismic_native_cpu::Tier::EVERY
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
                            "        CpuVariant {{ tier: seismic::cpu::Tier::{token}, elements: &[{names}], function: launch_{launch_index}_{tier}_{elements} as CpuKernelFn }},\n",
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
                TensorAccess::Shared => "&'a seismic::Tensor".to_owned(),
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
                TensorAccess::Shared => "seismic::WorkflowTensorRef<'a>".to_owned(),
                TensorAccess::Mutable => "seismic::WorkflowTensorMut<'a>".to_owned(),
            },
            ParameterSummaryKind::Scalar(dtype) => scalar_type(*dtype).to_owned(),
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
                &format!("dimension_{}", ident(name)),
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
            let name = format!("{}_{}{}", "parameter", ident(&parameter.name), suffix);
            match parameter.kind {
                ParameterSummaryKind::Tensor { .. } => {}
                ParameterSummaryKind::Scalar(dtype) => {
                    let (variant, bits) = match dtype {
                        DType::F32 => ("F32", false),
                        DType::F16 => ("F16", true),
                        DType::BF16 => ("BF16", true),
                        DType::I32 => ("I32", false),
                        DType::U32 => ("U32", false),
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
        if ident(value) != value || value.is_empty() {
            return Err(BuildError::Environment(format!(
                "`{value}` is not a Rust module identifier"
            )));
        }
        Ok(())
    }

    fn ident(value: &str) -> String {
        let mut result = String::new();
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
        if matches!(
            result.as_str(),
            "as" | "break"
                | "const"
                | "continue"
                | "crate"
                | "else"
                | "enum"
                | "extern"
                | "false"
                | "fn"
                | "for"
                | "if"
                | "impl"
                | "in"
                | "let"
                | "loop"
                | "match"
                | "mod"
                | "move"
                | "mut"
                | "pub"
                | "ref"
                | "return"
                | "self"
                | "Self"
                | "static"
                | "struct"
                | "super"
                | "trait"
                | "true"
                | "type"
                | "unsafe"
                | "use"
                | "where"
                | "while"
                | "async"
                | "await"
                | "dyn"
        ) {
            result.insert_str(0, "r#");
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
            .run()
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
            .run()
            .expect("second build");
        assert_ne!(first.identity, second.identity);
        fs::remove_dir_all(&root).expect("remove fixture directory");
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
            "fn scale[N](x: &tensor[N] f32) -> tensor[N] f32:\n    let mut output = tensor[N] f32\n    for i in 0..N:\n        output[i] = x[i]\n    return output\n\nnative scale for vulkan from \"vulkan/scale.comp\":\n    params (WIDTH in [64])\n    launch scale:\n        threadgroups (ceil_div(N, WIDTH), 1, 1)\n        threads_per_threadgroup (WIDTH, 1, 1)\n        shared_bytes (WIDTH * 4)\n",
        )
        .expect("Seismic fixture");
        fs::write(root.join("vulkan/scale.comp"), asset).expect("Vulkan fixture");
        let result = Build::new("fixture")
            .source(&source)
            .std(false)
            .out_dir(root.join("out"))
            .run();
        fs::remove_dir_all(&root).expect("remove fixture directory");
        result
    }

    #[test]
    fn vulkan_assets_admit_the_vulkan_abi() {
        build_vulkan(
            "void scale() {\n    seismic_f32 x = seismic_f32(SEISMIC_PTR(SEISMIC_BUFFER_X));\n    const bool read_only = SEISMIC_READONLY(SEISMIC_BUFFER_X);\n#if SEISMIC_HAS_MATRIX\n#endif\n    seismic_shared_f32[0] = x[uint(SEISMIC_X_STRIDE_0 * SEISMIC_SHARED_F32)].v + float(SEISMIC_TUNE_WIDTH);\n    seismic_f32(SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER))[0].v = seismic_shared_f32[0];\n}\n",
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
                "void scale() { float x = subgroupAdd(1.0); }\n",
                "`subgroupAdd`",
                1,
            ),
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
            .run()
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
            "// SEISMIC_DIM_NOT_AN_ABI_SYMBOL inside a comment is inert.\n#define PACKET_SIZE(PREFIX) PREFIX##_PACKET_SIZE\nkernel void scale(device const float *x [[buffer(SEISMIC_BUFFER_X)]], device float *output [[buffer(SEISMIC_RESULT_0_BUFFER)]], constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]], uint index [[thread_position_in_grid]]) {\n#if defined(SEISMIC_X_REPRESENTATION_F32) && defined(SEISMIC_RESULT_0_KIND_DENSE)\n    ulong packet_size = PACKET_SIZE(SEISMIC_X);\n    if (packet_size > 0 && index < SEISMIC_DIM_N) output[index * SEISMIC_RESULT_0_STRIDE_0] = x[index * SEISMIC_X_STRIDE_0];\n#endif\n}\n",
        )
        .expect("Metal fixture");

        Build::new("fixture")
            .source(&source)
            .std(false)
            .out_dir(&output)
            .run()
            .expect("exact entry ABI symbols must pass the consumer build");
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

    const SPECIALIZED: &str = "fn scale[N](x: &tensor[N] f32) -> tensor[N] f32:\n    let mut output = tensor[N] f32\n    for i in 0..N:\n        output[i] = x[i]\n    return output\n\nnative scale for cuda from \"native/scale.cu\":\n    static (N)\n    params (TILE in [32, 64])\n    scratch staging bytes (N * 4)\n    launch scale:\n        threadgroups (ceil_div(N, TILE), 1, 1)\n        threads_per_threadgroup (TILE, 1, 1)\n\nnative scale for cpu from \"native/scale.rs\":\n    params (TILE in [32, 64], UNROLL in [1, 2, 4])\n    launch scale:\n        threadgroups (ceil_div(N, TILE), 1, 1)\n        threads_per_threadgroup (1, 1, 1)\n";

    const CPU_SCALE: &str = "fn scale<L: Isa, E: Elements>(_l: L, cx: &Context<'_, E>, group: [u64; 3], _shared: &mut [u8]) {\n    let (x, output, tile) = (cx.arg_x(), cx.result_0(), cx.param_tile());\n    for i in group[0] * tile..((group[0] + 1) * tile).min(cx.dim_n()) {\n        unsafe { output.set([i as usize], x.get([i as usize])) };\n    }\n    let _ = cx.param_unroll();\n}\n";

    #[test]
    fn launch_scoped_assets_use_forming_guards_without_tuning_macros() {
        let root = fixture("scoped-abi");
        let source = root.join("ops.seismic");
        fs::write(&source, "fn scale[N](x: &tensor[N] f32) -> tensor[N] f32:\n    return to_owned(x)\n\nnative scale for cuda from \"native/scale.cu\":\n    launch scale_a when N <= 8:\n        params (code TILE in [4, 8])\n        threadgroups (ceil_div(N, TILE), 1, 1)\n        threads_per_threadgroup (32, 1, 1)\n    launch scale_b when N > 8:\n        params (code TILE in [16, 32])\n        threadgroups (ceil_div(N, TILE), 1, 1)\n        threads_per_threadgroup (32, 1, 1)\n").unwrap();
        let asset = root.join("native/scale.cu");
        fs::write(&asset, "#ifdef SEISMIC_FORMING_SCALE_A\ntemplate <unsigned TILE>\n__global__ void scale_a() {}\n#endif\n#ifdef SEISMIC_FORMING_SCALE_B\ntemplate <unsigned TILE>\n__global__ void scale_b() {}\n#endif\n").unwrap();
        let build = |name: &str| {
            Build::new("fixture")
                .source(&source)
                .std(false)
                .out_dir(root.join(name))
                .run()
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
        fs::write(&source, "fn scale[N](x: &tensor[N] f32) -> tensor[N] f32:\n    return to_owned(x)\n\nnative scale for metal from \"native/scale.metal\":\n    params (SPLIT in [1, 2])\n    launch small when N <= 8:\n        params (code ROWS in [1, 2])\n        threadgroups (ceil_div(N, ROWS), 1, 1)\n        threads_per_threadgroup (32, 1, 1)\n    launch large when N > 8:\n        params (code WIDTH in [4, 8], code ROWS in [1, 2])\n        threadgroups (ceil_div(N, WIDTH * ROWS), 1, 1)\n        threads_per_threadgroup (32, 1, 1)\n").unwrap();
        let asset = root.join("native/scale.metal");
        let small = "#ifdef SEISMIC_FORMING_SMALL\ntemplate <uint ROWS>\nkernel void small() { (void)SEISMIC_RUNTIME_SPLIT; }\n#endif\n";
        let large = "#ifdef SEISMIC_FORMING_LARGE\ntemplate <uint WIDTH, uint ROWS>\nkernel void large() {}\n#endif\n";
        let build = |name: &str| {
            Build::new("fixture")
                .source(&source)
                .std(false)
                .out_dir(root.join(name))
                .run()
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
            BuildError::NativeTemplate { kernel, .. } => assert_eq!(kernel, "large"),
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
            BuildError::NativeTemplate { kernel, .. } => assert_eq!(kernel, "large"),
            other => panic!("unexpected error: {other}"),
        }
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn cuda_scoped_templates_match_the_launch_declaration() {
        let root = fixture("cuda-scoped-template");
        let source = root.join("ops.seismic");
        fs::write(&source, "fn scale[N](x: &tensor[N] f32) -> tensor[N] f32:\n    return to_owned(x)\n\nnative scale for cuda from \"native/scale.cu\":\n    params (code arithmetic INT8 in [0, 1])\n    launch small when N <= 8:\n        params (code KSPLIT in [2, 4])\n        threadgroups (ceil_div(N, KSPLIT), 1, 1)\n        threads_per_threadgroup (32, 1, 1)\n    launch large when N > 8:\n        reads (INT8)\n        threadgroups (ceil_div(N, 32), 1, 1)\n        threads_per_threadgroup (32, 1, 1)\n").unwrap();
        let asset = root.join("native/scale.cu");
        let small = "#ifdef SEISMIC_FORMING_SMALL\ntemplate <int KSPLIT>\n__global__ void small() {}\n#endif\n";
        let large = "#ifdef SEISMIC_FORMING_LARGE\ntemplate <unsigned INT8>\n__global__ void large() {}\n#endif\n";
        let build = |name: &str| {
            Build::new("fixture")
                .source(&source)
                .std(false)
                .out_dir(root.join(name))
                .run()
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
            BuildError::NativeTemplate { kernel, .. } => assert_eq!(kernel, "large"),
            other => panic!("unexpected error: {other}"),
        }
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
                .run()
        };
        let artifacts = build("out").expect("tuning and scratch symbols are generated ABI");
        let generated = fs::read_to_string(&artifacts.bindings).expect("generated bindings");
        assert!(generated.contains("Some(&cpu_native::KERNELS)"));
        // Tuning parameters are runtime values: one form per tier (four on
        // x86-64, one on aarch64) of the one launch, none per value.
        assert_eq!(generated.matches("as CpuKernelFn").count(), 5);
        assert!(generated.contains("pub fn param_tile(&self) -> u64 { self.invocation.param(0) }"));
        assert!(
            generated.contains("pub fn param_unroll(&self) -> u64 { self.invocation.param(1) }")
        );
        assert!(generated.contains("#[target_feature(enable = \"sse3,ssse3,sse4.1,sse4.2,popcnt,avx,avx2,fma,f16c,bmi1,bmi2,lzcnt,movbe\")]"));
        assert!(generated.contains("scale::<seismic::cpu::X86V3, Elements_>"));

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

    const SCALE_METAL: &str = "fn scale[N](x: &tensor[N] f32) -> tensor[N] f32:\n    let mut output = tensor[N] f32\n    for i in 0..N:\n        output[i] = x[i]\n    return output\n\nnative scale for metal from \"native/scale.metal\":\n    launch scale:\n        threadgroups (ceil_div(N, 256), 1, 1)\n        threads_per_threadgroup (256, 1, 1)\n";

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
            "#include \"common/copy.h\"\nkernel void scale(device const float *x [[buffer(SEISMIC_BUFFER_X)]], device float *output [[buffer(SEISMIC_RESULT_0_BUFFER)]], constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]], uint index [[thread_position_in_grid]]) { if (index < SEISMIC_DIM_N) COPY(index); }\n",
        )
        .expect("Metal fixture");
        let build = |name: &str| {
            Build::new("fixture")
                .source(&source)
                .std(false)
                .out_dir(root.join(name))
                .run()
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
            .run()
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
