//! Build-script source checking, checked-bundle emission, and typed Rust
//! binding generation (spec §14.1).
//!
//! A consumer's `build.rs` names its `.seismic` sources and a module name.
//! Generation parses and checks (with the standard library unless opted
//! out), emits `<module>.seismicbundle` and `<module>.rs` into `OUT_DIR`,
//! and records source/compiler hashes for cache identity. Generated Rust
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
    EntryInfo, NativeImplementation, ParameterSummary,
    ParameterSummaryKind, ResultSummaryKind, SourceSet, TensorAccess,
};
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
                "native Metal ABI for `{entry}`: {}:{line}:{column}: `{symbol}` is not generated for this entry",
                path.display()
            ),
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
}

mod internals {
    use super::*;
    use seismic_lang::checked::ElementSummary;
    use seismic_lang::registry::{self, CodeInterpretation, PlaneEncoding, RepresentationInfo};
    use std::collections::HashSet;
    use std::ffi::OsStr;
    use std::fs;

    /// The native implementations of one entry. Metal and CUDA sources
    /// travel in the checked bundle; a CPU source is compiled into the
    /// generated bindings.
    struct EntryNative<'a> {
        entry: seismic_lang::ids::EntryId,
        cpu: Option<(&'a NativeImplementation, PathBuf)>,
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
        let output=output.canonicalize().map_err(BuildError::Io)?;

        let prelude = if build.include_std() { seismic_std::sources() } else { SourceSet::default() };
        let (checked, dependencies) = seismic_lang::source::load(build.sources(), prelude).map_err(|e|match e {
            seismic_lang::source::LoadError::Io(e)=>BuildError::Io(e),
            seismic_lang::source::LoadError::Source(e)=>BuildError::Source(e),
            seismic_lang::source::LoadError::Invalid(e)=>BuildError::Environment(e),
        })?;
        for path in dependencies { println!("cargo:rerun-if-changed={}",path.display()); }
        for path in build.sources() { println!("cargo:rerun-if-changed={}",path.display()); }
        let encoded = seismic_lang::bundle::encode_checked_bundle(&checked);
        let native_assets = resolve_native_assets(&checked, &output)?;
        // The checked bundle contains the canonical sources, bundle format,
        // checker semantic version, registry revision, and semantic hash.
        // Addressing the emitted bundle therefore cannot accidentally reuse
        // generated bindings across a change in any of those inputs.
        let mut identity_hasher = Sha256::new();
        identity_hasher.update(&encoded);
        let bundle_digest: [u8; 32] = identity_hasher.finalize().into();
        let identity = hex(&bundle_digest);
        let bundle = output.join(format!("{}.seismicbundle", build.module()));
        let bindings = output.join(format!("{}.rs", build.module()));
        fs::write(&bundle, encoded).map_err(BuildError::Io)?;
        fs::write(
            &bindings,
            render(&checked, build.module(), &identity, &native_assets),
        )
        .map_err(BuildError::Io)?;
        Ok(Artifacts {
            bundle,
            bindings,
            identity,
        })
    }

    fn resolve_native_assets<'a>(module: &'a seismic_lang::checked::CheckedModule, output: &std::path::Path) -> Result<Vec<EntryNative<'a>>, BuildError> {
        use seismic_lang::registry::BackendName;
        let mut natives = Vec::new();
        for entry in module.entries() {
            let mut any = false;
            for (backend, extension) in [(BackendName::Metal, "metal"), (BackendName::Cuda, "cu")] {
                let Some(definition) = module.native_implementation(entry.id, backend) else { continue };
                any = true;
                let source = module.native_asset(entry.id, backend).expect("shared loader captures native assets");
                let path = output.join(format!("{}.{extension}", entry.name));
                fs::write(&path, source).map_err(BuildError::Io)?;
                validate_native_abi(entry, definition, backend, &path, source)?;
            }
            let cpu = match module.native_implementation(entry.id, BackendName::Cpu) {
                Some(definition) => {
                    any = true;
                    let source = module.native_asset(entry.id, BackendName::Cpu).expect("shared loader captures native assets");
                    let path = output.join(format!("{}.cpu.rs", entry.name));
                    fs::write(&path, source).map_err(BuildError::Io)?;
                    Some((definition, path))
                }
                None => None,
            };
            if any {
                natives.push(EntryNative { entry: entry.id, cpu });
            }
        }
        Ok(natives)
    }

    fn validate_native_abi(
        entry: &EntryInfo,
        definition: &NativeImplementation,
        backend: seismic_lang::registry::BackendName,
        path: &std::path::Path,
        source: &str,
    ) -> Result<(), BuildError> {
        let failure = |offset: usize, symbol: &str| {
            let prefix = &source[..offset];
            let line = prefix.bytes().filter(|byte| *byte == b'\n').count() + 1;
            let column = prefix
                .rsplit_once('\n')
                .map_or(prefix.len() + 1, |(_, tail)| tail.len() + 1);
            BuildError::NativeAbi {
                entry: entry.name.clone(),
                path: path.to_path_buf(),
                line,
                column,
                symbol: symbol.to_owned(),
            }
        };
        if backend == seismic_lang::registry::BackendName::Cuda {
            // CUDA sources receive their helpers from the generated prefix;
            // vendor headers are not part of the packaged runtime.
            if let Some(offset) = source.find("#include") {
                return Err(failure(offset, "#include"));
            }
        }
        let allowed = native_abi_symbols(entry, definition, backend);
        for (symbol, offset) in seismic_identifiers(source) {
            if !allowed.contains(symbol) {
                return Err(failure(offset, symbol));
            }
        }
        Ok(())
    }

    fn native_abi_symbols(
        entry: &EntryInfo,
        definition: &NativeImplementation,
        backend: seismic_lang::registry::BackendName,
    ) -> HashSet<String> {
        let mut symbols = HashSet::new();
        symbols.insert("SEISMIC_BUFFER_WORDS".to_owned());
        symbols.insert("SEISMIC_BUFFER_SCALAR_RESULTS".to_owned());
        for parameter in &definition.params {
            symbols.insert(format!("SEISMIC_TUNE_{}", native_macro(&parameter.name)));
        }
        for scratch in &definition.scratch {
            symbols.insert(format!("SEISMIC_BUFFER_SCRATCH_{}", native_macro(&scratch.name)));
        }
        if backend == seismic_lang::registry::BackendName::Cuda {
            for symbol in ["SEISMIC_KERNEL_PARAMS", "SEISMIC_PTR", "SEISMIC_PTR_", "SEISMIC_SCALAR_RESULTS"] {
                symbols.insert(symbol.to_owned());
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
                native_macro(representation.name)
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
                registry::RepresentationKind::Packed(layout) => {
                    symbols.insert(format!("{prefix}_KIND_PACKED"));
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
                    let identifier = &source[start..index];
                    if identifier.starts_with("SEISMIC_") {
                        identifiers.push((identifier, start));
                    }
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

    fn source_label(path: &std::path::Path) -> String {
        let relative = std::env::current_dir()
            .ok()
            .and_then(|directory| {
                path.strip_prefix(directory)
                    .ok()
                    .map(std::path::Path::to_path_buf)
            })
            .unwrap_or_else(|| path.to_path_buf());
        relative.to_string_lossy().replace('\\', "/")
    }

    fn collect(path: &std::path::Path, output: &mut Vec<PathBuf>) -> Result<(), BuildError> {
        let metadata = fs::metadata(path).map_err(BuildError::Io)?;
        if metadata.is_file() {
            if path.extension() != Some(OsStr::new("seismic")) {
                return Err(BuildError::Environment(format!(
                    "source `{}` is not a .seismic file",
                    path.display()
                )));
            }
            output.push(path.to_path_buf());
            return Ok(());
        }
        if !metadata.is_dir() {
            return Err(BuildError::Environment(format!(
                "source `{}` is neither a file nor a directory",
                path.display()
            )));
        }
        let mut children = fs::read_dir(path)
            .map_err(BuildError::Io)?
            .collect::<Result<Vec<_>, _>>()
            .map_err(BuildError::Io)?;
        children.sort_by_key(|entry| entry.path());
        for child in children {
            let child_path = child.path();
            if child.file_type().map_err(BuildError::Io)?.is_dir()
                || child_path.extension() == Some(OsStr::new("seismic"))
            {
                collect(&child_path, output)?;
            }
        }
        Ok(())
    }

    fn render(
        module: &seismic_lang::checked::CheckedModule,
        name: &str,
        identity: &str,
        native_assets: &[EntryNative<'_>],
    ) -> String {
        let mut out = String::new();
        out.push_str("// @generated by seismic-build; do not edit.\n");
        out.push_str(&format!(
            "static __BUNDLE: &[u8] = include_bytes!({:?});\n",
            format!("{name}.seismicbundle")
        ));
        out.push_str("static __MODULE: seismic::generated::OnceLock<Result<seismic::generated::Module, seismic::CheckedBundleError>> = seismic::generated::OnceLock::new();\n");
        out.push_str("fn module() -> Result<&'static seismic::generated::Module, seismic::CheckedBundleError> { seismic::generated::module_from_bundle(&__MODULE, __BUNDLE) }\n");
        out.push_str(&format!("pub const IDENTITY: &str = {:?};\n", identity));
        for entry in module.entries() {
            let native = native_assets
                .iter()
                .find(|native| native.entry == entry.id);
            render_entry(&mut out, entry, native);
        }
        out
    }

    fn render_entry(out: &mut String, entry: &EntryInfo, native: Option<&EntryNative<'_>>) {
        let module_name = ident(&entry.name);
        out.push_str(&format!("pub mod {module_name} {{\n"));
        out.push_str("  use super::*;\n");
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
                ResultSummaryKind::Range => "take_workflow_scalar::<(seismic::BigUint, seismic::BigUint)>".to_owned(),
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
            render_native(out, entry, native);
        }
        out.push_str("}\n");
    }

    /// Native entry points, and for a CPU implementation its typed context
    /// and monomorphized launch table.
    fn render_native(out: &mut String, entry: &EntryInfo, native: &EntryNative<'_>) {
        let elements = !entry.element_parameters.is_empty();
        let element_list = |out: &mut String| {
            for parameter in &entry.element_parameters {
                out.push_str(&format!("      ({:?}, elements.{}),\n", parameter, ident(parameter)));
            }
        };
        let cpu = if native.cpu.is_some() { "Some(&cpu_native::KERNELS)" } else { "None" };
        if elements {
            out.push_str("  pub fn native_for_device_with(device: &seismic::Device, elements: Elements, specialization: &seismic::NativeSpecialization) -> Result<seismic::NativeKernel<Entry>, seismic::LoadError> {\n");
            out.push_str("    seismic::generated::prepare_native::<Entry>(device, specialization, &[\n");
            element_list(out);
            out.push_str(&format!("    ], {cpu})\n  }}\n"));
            out.push_str("  pub fn native_tune_with(device: &seismic::Device, elements: Elements, statics: &seismic::NativeSpecialization, points: Vec<seismic::TuningPoint<'_, Entry>>, validation: seismic::Validation, measure: seismic::MeasureOptions) -> Result<seismic::TuningResult, seismic::TuneError> {\n");
            out.push_str("    seismic::generated::tune_native::<Entry>(device, statics, &[\n");
            element_list(out);
            out.push_str(&format!("    ], {cpu}, points, validation, measure)\n  }}\n"));
        } else {
            out.push_str(&format!("  pub fn native_for_device(device: &seismic::Device, specialization: &seismic::NativeSpecialization) -> Result<seismic::NativeKernel<Entry>, seismic::LoadError> {{ seismic::generated::prepare_native::<Entry>(device, specialization, &[], {cpu}) }}\n"));
            out.push_str(&format!("  pub fn native_tune(device: &seismic::Device, statics: &seismic::NativeSpecialization, points: Vec<seismic::TuningPoint<'_, Entry>>, validation: seismic::Validation, measure: seismic::MeasureOptions) -> Result<seismic::TuningResult, seismic::TuneError> {{ seismic::generated::tune_native::<Entry>(device, statics, &[], {cpu}, points, validation, measure) }}\n"));
        }
        out.push_str("  /// The checked native implementation for the device's backend.\n");
        out.push_str("  pub fn native_implementation(device: &seismic::Device) -> Result<Option<seismic::NativeImplementation>, seismic::CheckedBundleError> { seismic::generated::native_implementation::<Entry>(device) }\n");
        if let Some((definition, path)) = &native.cpu {
            render_cpu_native(out, entry, definition, path);
        }
    }

    /// The typed CPU view of an entry's native ABI, the authored source, and
    /// one function per launch and tuning configuration.
    fn render_cpu_native(
        out: &mut String,
        entry: &EntryInfo,
        definition: &NativeImplementation,
        path: &std::path::Path,
    ) {
        out.push_str("  pub mod cpu_native {\n");
        out.push_str("    #![allow(dead_code)]\n");
        out.push_str("    use seismic::native_cpu::{CpuInvocation, CpuKernelFn, CpuLaunchVariants, CpuNativeKernels, CpuTensor};\n");
        out.push_str(&format!(
            "    /// Typed view of the CPU native ABI of `{}`.\n",
            entry.name
        ));
        out.push_str("    #[derive(Clone, Copy)]\n");
        out.push_str("    pub struct Context<'a> { invocation: &'a CpuInvocation<'a> }\n");
        out.push_str("    impl<'a> Context<'a> {\n");
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
        let tensor = |out: &mut String, name: &str, buffer: usize, word: usize, rank: usize| {
            let extents = (0..rank).map(|axis| format!("self.invocation.word({})", word + axis)).collect::<Vec<_>>().join(", ");
            let strides = (0..rank).map(|axis| format!("self.invocation.word({})", word + rank + axis)).collect::<Vec<_>>().join(", ");
            out.push_str(&format!(
                "      pub fn {name}(&self) -> CpuTensor<{rank}> {{ CpuTensor {{ pointer: self.invocation.buffer({buffer}), extents: [{extents}], strides: [{strides}], representation: self.invocation.representation({buffer}) }} }}\n"
            ));
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
                ParameterSummaryKind::Tensor { rank, .. } => {
                    let rank = *rank as usize;
                    tensor(out, &name, buffer, word, rank);
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
                ResultSummaryKind::Tensor { rank, .. } => {
                    let rank = *rank as usize;
                    tensor(out, &format!("result_{ordinal}"), buffer, word, rank);
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
                "      pub fn scratch_{}(&self) -> *mut u8 {{ self.invocation.buffer({buffer}) }}\n",
                ident(&scratch.name).to_lowercase()
            ));
            buffer += 1;
        }
        out.push_str("    }\n");
        out.push_str(&format!("    include!({:?});\n", path.to_string_lossy()));
        // One function per launch and configuration of the parameter
        // domains' cartesian product; `where` filtering happens when a
        // specialization is prepared.
        let mut configurations: Vec<Vec<u64>> = vec![Vec::new()];
        for parameter in &definition.params {
            configurations = configurations
                .into_iter()
                .flat_map(|prefix| {
                    parameter.values.iter().map(move |value| {
                        let mut configuration = prefix.clone();
                        configuration.push(*value);
                        configuration
                    })
                })
                .collect();
        }
        out.push_str("    pub static KERNELS: CpuNativeKernels = CpuNativeKernels { launches: &[\n");
        let mut functions = String::new();
        for (launch_index, launch) in definition.launches.iter().enumerate() {
            out.push_str(&format!(
                "      CpuLaunchVariants {{ kernel: {:?}, variants: &[\n",
                launch.kernel
            ));
            for (variant, configuration) in configurations.iter().enumerate() {
                let function = format!("launch_{launch_index}_{variant}");
                let generics = if configuration.is_empty() {
                    String::new()
                } else {
                    format!(
                        "::<{}>",
                        configuration.iter().map(u64::to_string).collect::<Vec<_>>().join(", ")
                    )
                };
                functions.push_str(&format!(
                    "    fn {function}(invocation: &CpuInvocation<'_>, group: [u64; 3], shared: &mut [u8]) {{ {}{generics}(&Context {{ invocation }}, group, shared) }}\n",
                    launch.kernel
                ));
                out.push_str(&format!(
                    "        (&[{}], {function} as CpuKernelFn),\n",
                    configuration.iter().map(u64::to_string).collect::<Vec<_>>().join(", ")
                ));
            }
            out.push_str("      ] },\n");
        }
        out.push_str("    ] };\n");
        out.push_str(&functions);
        out.push_str("  }\n");
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
            ResultSummaryKind::Range => "seismic::WorkflowScalar<(seismic::BigUint, seismic::BigUint)>".to_owned(),
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
                ParameterSummaryKind::Index => {
                    render_range(out, &name, "seismic::BigUint", "Nat", false, "Scalar", ordinal)
                }
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

    const CPU_SCALE: &str = "fn scale<const TILE: u64, const UNROLL: u64>(context: &Context<'_>, group: [u64; 3], _shared: &mut [u8]) {\n    let x = context.arg_x();\n    let output = context.result_0();\n    for i in group[0] * TILE..((group[0] + 1) * TILE).min(context.dim_n()) {\n        unsafe { output.pointer.cast::<f32>().add(i as usize).write(x.pointer.cast::<f32>().add(i as usize).read()) };\n    }\n    let _ = UNROLL;\n}\n";

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
        // Two TILE values by three UNROLL values.
        assert_eq!(generated.matches("as CpuKernelFn").count(), 6);
        assert!(generated.contains("scale::<64, 4>"));

        fs::write(&cuda, "#include <cuda_fp16.h>\nextern \"C\" __global__ void scale(SEISMIC_KERNEL_PARAMS) {}\n")
            .expect("CUDA fixture");
        match build("header").expect_err("vendor headers are rejected") {
            BuildError::NativeAbi { symbol, line, .. } => {
                assert_eq!(symbol, "#include");
                assert_eq!(line, 1);
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
}
