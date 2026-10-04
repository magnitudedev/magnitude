//! Offline formation coverage of a checked module's native implementations.
//!
//! For one backend, every native implementation is formed with that
//! backend's real toolchain, as preparation forms it, under configurations
//! chosen so that every preprocessor group of its authored sources (the
//! asset, its includes and Seismic's native library) is compiled at least
//! once. A configuration is an element binding, a specialization (static
//! values and tuning parameters) and the device facts formation reads.
//!
//! Configurations are found without the toolchain: each candidate's
//! directive-only skeleton (its rendered `#define`s and conditionals, with a
//! marker in every authored group) is preprocessed, which says which groups
//! the candidate compiles. The search starts from one implemented
//! configuration and changes one choice at a time, keeping every
//! configuration that reaches a group no kept one reached, until no change
//! reaches more. Only kept configurations are formed. A group no implemented
//! configuration reaches, other than one that rejects with `#error`, is
//! reported: it is dead code, or reachable only through more than one change
//! from every kept configuration.
//!
//! Vulkan skeletons are preprocessed by glslang, the toolchain itself; CUDA
//! and Metal skeletons by `clang -E`. Their conditions may read no
//! toolchain-predefined macro except `__CUDA_ARCH__`, which the CUDA skeleton
//! defines for the configuration's architecture.

use super::abi::{self, render_source, Dialect};
use super::plan::{self, LaunchVariant};
use seismic_lang::checked::{CheckedModule, EntryInfo, NativeImplementation, NativeSpecialization};
use seismic_lang::entry::ElementBindings;
use seismic_lang::ids::{EntryId, RepresentationId};
use seismic_lang::registry::{self, BackendName};
use seismic_native_target::{NativeCompilationError, ProgramEntry, ProgramSource};
use std::collections::BTreeSet;
use std::fmt;

/// The device facts one formation reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Configuration {
    Vulkan(seismic_vulkan::FormationFacts),
    Cuda { architecture: u32 },
    Metal { tensor_ops: bool },
}

impl Configuration {
    fn dialect(self) -> Dialect {
        match self {
            Self::Vulkan(facts) => Dialect::Vulkan(abi::vulkan::VulkanFeatures::of(&facts)),
            Self::Cuda { .. } => Dialect::Cuda,
            Self::Metal { tensor_ops } => Dialect::Metal(abi::MetalFeatures { tensor_ops }),
        }
    }
}

/// An authored preprocessor group: the file (relative to its asset's
/// directory; Seismic library files as `seismic/<name>`) and the line of the
/// directive that opens it.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GroupSite {
    pub file: String,
    pub line: usize,
}

impl fmt::Display for GroupSite {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.file, self.line)
    }
}

/// One configuration of one entry the toolchain refused.
#[derive(Clone, Debug)]
pub struct FormationFailure {
    pub entry: String,
    pub bindings: String,
    pub specialization: String,
    pub configuration: Configuration,
    pub error: String,
}

impl fmt::Display for FormationFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "`{}` [{}] {} {:?}: {}",
            self.entry, self.bindings, self.specialization, self.configuration, self.error
        )
    }
}

/// What forming one backend's implementations established.
#[derive(Debug, Default)]
pub struct Coverage {
    /// Programs the toolchain formed.
    pub formed: usize,
    pub failures: Vec<FormationFailure>,
    /// Authored groups no implemented configuration reaches.
    pub unreached: Vec<GroupSite>,
    /// Groups reached, of all authored non-rejecting groups.
    pub reached: usize,
    pub groups: usize,
    /// Entries with no implemented configuration at all.
    pub unimplemented: Vec<String>,
    /// Configurations this host's toolchain cannot form (Metal tensor
    /// operations on a compiler without Metal 4), with the entries whose
    /// kept configurations needed them.
    pub unformable: Vec<(Configuration, String)>,
}

impl Coverage {
    /// Whether every implementation formed and every group was reached.
    pub fn complete(&self) -> bool {
        self.failures.is_empty() && self.unreached.is_empty() && self.unimplemented.is_empty()
    }
}

impl fmt::Display for Coverage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "formed {} programs; reached {} of {} groups",
            self.formed, self.reached, self.groups
        )?;
        for failure in &self.failures {
            writeln!(f, "formation failure: {failure}")?;
        }
        for entry in &self.unimplemented {
            writeln!(f, "no implemented configuration: `{entry}`")?;
        }
        for site in &self.unreached {
            writeln!(f, "unreached group: {site}")?;
        }
        for (configuration, entry) in &self.unformable {
            writeln!(f, "unformable on this host: `{entry}` {configuration:?}")?;
        }
        Ok(())
    }
}

/// Why coverage could not run.
#[derive(Debug)]
pub enum CoverageError {
    /// The backend's toolchain is not available on this host.
    ToolchainUnavailable(String),
    /// A skeleton could not be preprocessed: the coverage machinery, not a
    /// kernel, is at fault.
    Skeleton(String),
}

impl fmt::Display for CoverageError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ToolchainUnavailable(reason) => write!(f, "toolchain unavailable: {reason}"),
            Self::Skeleton(reason) => write!(f, "skeleton preprocessing failed: {reason}"),
        }
    }
}

impl std::error::Error for CoverageError {}

/// One backend's toolchain, as preparation forms with it.
enum Former {
    Vulkan,
    Cuda {
        architectures: Vec<u32>,
    },
    #[cfg(target_os = "macos")]
    Metal {
        device: seismic_metal::DeviceHandle,
        language: seismic_metal::facts::LanguageVersion,
        tensor_ops: bool,
    },
}

/// The preparation baseline of the Vulkan floor: a 32-lane fp16 device
/// without cooperative matrix, declaring every seal mode.
const VULKAN_BASE: seismic_vulkan::FormationFacts = seismic_vulkan::FormationFacts {
    subgroup_lanes: seismic_vulkan::SUBGROUP_WIDTH,
    float16: true,
    matrix: false,
    wide_accumulators: true,
    mixed_dot: true,
    f32_atomic_add: false,
    shared_int64_atomics: false,
    rounding_rte_32: true,
    denorm_preserve_32: true,
};

impl Former {
    fn open(backend: BackendName) -> Result<Self, CoverageError> {
        match backend {
            BackendName::Vulkan => Ok(Self::Vulkan),
            BackendName::Cuda => {
                let floor = u32::from(seismic_cuda::ComputeCapability::SM80.major) * 10
                    + u32::from(seismic_cuda::ComputeCapability::SM80.minor);
                let architectures = seismic_cuda::nvrtc::supported_architectures()
                    .map_err(|error| CoverageError::ToolchainUnavailable(format!("{error:?}")))?
                    .into_iter()
                    .filter(|architecture| *architecture >= floor)
                    .collect();
                Ok(Self::Cuda { architectures })
            }
            #[cfg(target_os = "macos")]
            BackendName::Metal => {
                let device = seismic_metal::DeviceHandle::system_default()
                    .map_err(|error| CoverageError::ToolchainUnavailable(format!("{error:?}")))?;
                let language = seismic_metal::profile::probe_language_version(&device)
                    .ok_or_else(|| CoverageError::ToolchainUnavailable("no supported Metal language version".into()))?;
                let tensor_ops = seismic_metal::toolchain::forms_tensor_operations(&device, language);
                Ok(Self::Metal { device, language, tensor_ops })
            }
            #[cfg(not(target_os = "macos"))]
            BackendName::Metal => Err(CoverageError::ToolchainUnavailable(
                "Metal forms on macOS only".into(),
            )),
            BackendName::Cpu => Err(CoverageError::ToolchainUnavailable(
                "CPU native implementations are Rust compiled with the program".into(),
            )),
        }
    }

    fn base(&self) -> Configuration {
        match self {
            Self::Vulkan => Configuration::Vulkan(VULKAN_BASE),
            Self::Cuda { architectures } => Configuration::Cuda {
                architecture: architectures[0],
            },
            #[cfg(target_os = "macos")]
            Self::Metal { .. } => Configuration::Metal { tensor_ops: false },
        }
    }

    /// The configurations differing from `configuration` in one fact.
    fn variations(&self, configuration: Configuration) -> Vec<Configuration> {
        match (self, configuration) {
            (Self::Vulkan, Configuration::Vulkan(facts)) => facts
                .variations()
                .into_iter()
                .map(Configuration::Vulkan)
                .collect(),
            (Self::Cuda { architectures }, Configuration::Cuda { architecture }) => architectures
                .iter()
                .filter(|candidate| **candidate != architecture)
                .map(|architecture| Configuration::Cuda {
                    architecture: *architecture,
                })
                .collect(),
            #[cfg(target_os = "macos")]
            (Self::Metal { .. }, Configuration::Metal { tensor_ops }) => {
                vec![Configuration::Metal {
                    tensor_ops: !tensor_ops,
                }]
            }
            _ => unreachable!("a former varies only its own configurations"),
        }
    }

    /// Whether this host's toolchain forms `configuration`.
    fn formable(&self, configuration: Configuration) -> bool {
        match (self, configuration) {
            #[cfg(target_os = "macos")]
            (Self::Metal { tensor_ops, .. }, Configuration::Metal { tensor_ops: needed }) => {
                *tensor_ops || !needed
            }
            _ => true,
        }
    }

    /// Form `programs` as preparation forms them, short of loading them on a
    /// device.
    fn form(
        &self,
        programs: &[ProgramSource],
        configuration: Configuration,
    ) -> Result<(), NativeCompilationError> {
        match (self, configuration) {
            (Self::Vulkan, Configuration::Vulkan(facts)) => {
                for program in programs {
                    for entry in &program.entries {
                        seismic_vulkan::formation::compile(
                            &program.text,
                            &entry.symbol,
                            facts.environment(),
                        )?;
                    }
                }
                Ok(())
            }
            (Self::Cuda { .. }, Configuration::Cuda { architecture }) => {
                for program in programs {
                    seismic_cuda::toolchain::form_cubin(program, architecture)?;
                }
                Ok(())
            }
            #[cfg(target_os = "macos")]
            (Self::Metal { device, language, .. }, Configuration::Metal { .. }) => {
                for program in programs {
                    seismic_metal::toolchain::form_functions(device, program, *language)?;
                }
                Ok(())
            }
            _ => unreachable!("a former forms only its own configurations"),
        }
    }
}

/// One candidate configuration of one entry.
#[derive(Clone, Debug)]
struct Row {
    bindings: ElementBindings,
    specialization: NativeSpecialization,
    configuration: Configuration,
}

impl Row {
    fn key(&self) -> String {
        format!(
            "{:?}|{:?}|{:?}",
            self.bindings, self.specialization, self.configuration
        )
    }
}

/// The skeleton of one rendered program: its preprocessor directives, with a
/// marker after each directive opening an authored group.
struct Skeleton {
    text: String,
    /// The group each marker opens, by marker index.
    sites: Vec<GroupSite>,
    /// Markers of groups that hold an `#error` directly.
    rejecting: BTreeSet<usize>,
}

const MARKER: &str = "seismic_group_";
/// The marker an `#error` directive becomes in a skeleton.
const REJECTED: &str = "seismic_rejected_";

/// Directives a skeleton keeps; every other line is dropped.
fn kept(directive: &str, dialect: Dialect) -> bool {
    match directive {
        "define" | "undef" | "if" | "ifdef" | "ifndef" | "elif" | "else" | "endif" | "error" => {
            true
        }
        "version" | "extension" => matches!(dialect, Dialect::Vulkan(_)),
        _ => false,
    }
}

/// Build the skeleton of `text`, whose authored part is `asset` (labelled
/// `label`) at its end, possibly followed by a generated suffix.
fn skeleton(text: &str, asset: &str, label: &str, dialect: Dialect, preamble: &str) -> Skeleton {
    let asset_start = text
        .rfind(asset)
        .expect("a rendered source contains its asset");
    let asset_end = asset_start + asset.len();
    let mut out = String::new();
    let mut sites = Vec::new();
    let mut rejecting = BTreeSet::new();
    // Open authored groups, innermost last, by marker index.
    let mut open: Vec<Option<usize>> = Vec::new();
    let mut file = label.to_owned();
    let mut line = 0usize;
    let mut offset = 0usize;
    let mut lines = text.split_inclusive('\n').peekable();
    let mut wrote_preamble = false;
    let mut rejections = 0usize;
    while let Some(raw) = lines.next() {
        let start = offset;
        offset += raw.len();
        let authored = start >= asset_start && start < asset_end;
        if authored {
            line += 1;
        }
        let trimmed = raw.trim_start();
        let Some(rest) = trimmed.strip_prefix('#') else {
            continue;
        };
        let rest = rest.trim_start();
        let directive: String = rest
            .chars()
            .take_while(|c| c.is_ascii_alphabetic())
            .collect();
        if directive == "line" && authored {
            // `#line N "file"`: the next line is line N of `file`.
            let mut words = rest["line".len()..].split_whitespace();
            if let Some(number) = words.next().and_then(|word| word.parse::<usize>().ok()) {
                line = number - 1;
                if let Some(name) = words.next() {
                    file = name.trim_matches('"').to_owned();
                }
            }
            continue;
        }
        let mut full = raw.trim_end_matches(['\n', '\r']).to_owned();
        while full.ends_with('\\') {
            full.pop();
            match lines.next() {
                Some(next) => {
                    offset += next.len();
                    if authored {
                        line += 1;
                    }
                    full.push_str(next.trim_end_matches(['\n', '\r']));
                }
                None => break,
            }
        }
        if !kept(&directive, dialect) {
            continue;
        }
        if !wrote_preamble && directive != "version" {
            out.push_str(preamble);
            wrote_preamble = true;
        }
        if directive == "error" {
            // An active `#error` shows as its marker in the output rather
            // than failing preprocessing.
            out.push_str(&format!("const int {REJECTED}{rejections} = 0;\n"));
            rejections += 1;
            if let Some(Some(marker)) = open.last() {
                rejecting.insert(*marker);
            }
            continue;
        }
        out.push_str(&full);
        out.push('\n');
        if directive == "version" {
            continue;
        }
        let opening_line = line;
        match directive.as_str() {
            "if" | "ifdef" | "ifndef" => {
                open.push(None);
            }
            "elif" | "else" => {}
            "endif" => {
                open.pop();
                continue;
            }
            _ => continue,
        }
        let marker = if authored {
            let marker = sites.len();
            sites.push(GroupSite {
                file: file.clone(),
                line: opening_line,
            });
            out.push_str(&format!("const int {MARKER}{marker} = 0;\n"));
            Some(marker)
        } else {
            None
        };
        *open
            .last_mut()
            .expect("a group directive is inside a conditional") = marker;
    }
    if !wrote_preamble {
        out.push_str(preamble);
    }
    if matches!(dialect, Dialect::Vulkan(_)) {
        out.push_str("void main() {}\n");
    }
    Skeleton {
        text: out,
        sites,
        rejecting,
    }
}

/// Identifiers in a skeleton's conditions that only a toolchain predefines.
fn predefined(skeleton: &str) -> BTreeSet<String> {
    let mut found = BTreeSet::new();
    for line in skeleton.lines() {
        let trimmed = line.trim_start();
        let Some(rest) = trimmed.strip_prefix('#') else {
            continue;
        };
        let rest = rest.trim_start();
        if !["if", "elif", "ifdef", "ifndef"]
            .iter()
            .any(|directive| rest.starts_with(directive))
        {
            continue;
        }
        let mut word = String::new();
        for c in rest.chars().chain(std::iter::once(' ')) {
            if c.is_ascii_alphanumeric() || c == '_' {
                word.push(c);
            } else {
                if word.starts_with("__") {
                    found.insert(word.clone());
                }
                word.clear();
            }
        }
    }
    found
}

/// What a skeleton's preprocessing established.
enum Outcome {
    /// The configuration is implemented and compiles these markers.
    Reached(BTreeSet<usize>),
    /// An active `#error` rejects the configuration.
    Rejected,
}

/// The outcome of a skeleton's preprocessed text.
fn outcome(preprocessed: &str) -> Outcome {
    if preprocessed.contains(REJECTED) {
        return Outcome::Rejected;
    }
    Outcome::Reached(reached_markers(preprocessed))
}

fn reached_markers(preprocessed: &str) -> BTreeSet<usize> {
    let mut reached = BTreeSet::new();
    let mut rest = preprocessed;
    while let Some(at) = rest.find(MARKER) {
        rest = &rest[at + MARKER.len()..];
        let digits: String = rest.chars().take_while(char::is_ascii_digit).collect();
        if let Ok(marker) = digits.parse() {
            reached.insert(marker);
        }
    }
    reached
}

/// Preprocess a Vulkan skeleton with glslang, once per launch kernel.
fn preprocess_vulkan(skeleton: &Skeleton, kernels: &[&str]) -> Result<Outcome, CoverageError> {
    let mut reached = BTreeSet::new();
    for kernel in kernels {
        let text = seismic_vulkan::formation::preprocess(&skeleton.text, kernel).map_err(|error| {
            CoverageError::Skeleton(format!("{error:?}\n--- skeleton ---\n{}", skeleton.text))
        })?;
        match outcome(&text) {
            Outcome::Rejected => return Ok(Outcome::Rejected),
            Outcome::Reached(markers) => reached.extend(markers),
        }
    }
    Ok(Outcome::Reached(reached))
}

/// Skeletons one `clang -E` preprocesses, within the host's argument limit.
const CLANG_BATCH: usize = 256;

/// Preprocess one batch of CUDA and Metal skeletons with `clang -E`.
fn preprocess_clang(skeletons: &[&str]) -> Result<Vec<Outcome>, CoverageError> {
    static BATCHES: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let directory = std::env::temp_dir().join(format!(
        "seismic-coverage-{}-{}",
        std::process::id(),
        BATCHES.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&directory)
        .map_err(|error| CoverageError::Skeleton(format!("temporary directory: {error}")))?;
    let mut paths = Vec::with_capacity(skeletons.len());
    for (index, skeleton) in skeletons.iter().enumerate() {
        let path = directory.join(format!("{index}.cc"));
        std::fs::write(&path, format!("{skeleton}{END}{index}\n"))
            .map_err(|error| CoverageError::Skeleton(format!("writing a skeleton: {error}")))?;
        paths.push(path);
    }
    let output = std::process::Command::new("clang")
        .args(["-x", "c++", "-E", "-P", "-w"])
        .args(&paths)
        .output()
        .map_err(|error| match error.kind() {
            std::io::ErrorKind::NotFound => CoverageError::ToolchainUnavailable(format!("clang: {error}")),
            _ => CoverageError::Skeleton(format!("clang: {error}")),
        })?;
    std::fs::remove_dir_all(&directory).ok();
    if !output.status.success() {
        return Err(CoverageError::Skeleton(
            String::from_utf8_lossy(&output.stderr).into_owned(),
        ));
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut rest: &str = &stdout;
    (0..skeletons.len())
        .map(|index| {
            let end = format!("{END}{index}");
            let at = rest.find(&end).ok_or_else(|| {
                CoverageError::Skeleton(format!("clang emitted no output for skeleton {index}"))
            })?;
            let text = &rest[..at];
            rest = &rest[at + end.len()..];
            Ok(outcome(text))
        })
        .collect()
}

/// Ends each skeleton in a batched `clang -E`.
const END: &str = "seismic_skeleton_end_";

/// Every representation of the registry, bf16 and f16 first. Which of them
/// an entry admits is what monomorphizing it accepts, as preparation decides
/// it; which the kernel implements, its `#error` and `static_assert`
/// rejections say.
fn representations() -> Vec<RepresentationId> {
    let mut found = registry::representations().iter().collect::<Vec<_>>();
    found.sort_by_key(|info| match info.name {
        "bf16" => 0,
        "f16" => 1,
        "f32" => 2,
        _ => 3,
    });
    found.into_iter().map(|info| info.id).collect()
}

/// The checked inputs of one entry's implementation on one backend.
struct Subject<'m> {
    module: &'m CheckedModule,
    info: &'m EntryInfo,
    implementation: &'m NativeImplementation,
    asset: &'m str,
    label: String,
    /// Each element parameter's candidate representations.
    candidates: Vec<(String, Vec<RepresentationId>)>,
    /// Whether the entry monomorphizes at a binding, by binding.
    admitted: std::sync::Mutex<std::collections::HashMap<String, bool>>,
    /// Searched statics, by the statics fixed.
    searched: std::sync::Mutex<std::collections::HashMap<Vec<(String, u64)>, Option<NativeSpecialization>>>,
    /// Admissible specializations, by statics.
    specializations: std::sync::Mutex<std::collections::HashMap<NativeSpecialization, std::sync::Arc<Vec<NativeSpecialization>>>>,
}

impl<'m> Subject<'m> {
    fn new(
        module: &'m CheckedModule,
        info: &'m EntryInfo,
        backend: BackendName,
        candidates: Vec<(String, Vec<RepresentationId>)>,
    ) -> Option<Self> {
        let implementation = module.native_implementation(info.id, backend)?;
        let asset = module
            .native_asset(info.id, backend)
            .expect("a checked module captures every native asset");
        let label = std::path::Path::new(&implementation.source_path)
            .file_name()
            .expect("a native asset path names a file")
            .to_string_lossy()
            .into_owned();
        Some(Self {
            module,
            info,
            implementation,
            asset,
            label,
            candidates,
            admitted: Default::default(),
            searched: Default::default(),
            specializations: Default::default(),
        })
    }

    /// [`NativeImplementation::search_statics`], once per `fixed`.
    fn statics(&self, fixed: &[(&str, u64)]) -> Option<NativeSpecialization> {
        let key = fixed
            .iter()
            .map(|(name, value)| ((*name).to_owned(), *value))
            .collect::<Vec<_>>();
        let mut searched = self.searched.lock().expect("no thread panics holding the cache");
        searched
            .entry(key)
            .or_insert_with(|| self.implementation.search_statics(fixed))
            .clone()
    }

    /// The admissible specializations at `statics`, walked once.
    fn admissible(&self, statics: &NativeSpecialization) -> std::sync::Arc<Vec<NativeSpecialization>> {
        let mut walked = self
            .specializations
            .lock()
            .expect("no thread panics holding the cache");
        walked
            .entry(statics.clone())
            .or_insert_with(|| {
                std::sync::Arc::new(
                    self.implementation
                        .admissible(statics)
                        .expect("a row's statics admit its specialization"),
                )
            })
            .clone()
    }

    fn admits(&self, bindings: &ElementBindings) -> bool {
        let key = format!("{bindings:?}");
        if let Some(admitted) = self
            .admitted
            .lock()
            .expect("no thread panics holding the cache")
            .get(&key)
        {
            return *admitted;
        }
        let admitted = self.module.entry(self.info.id, bindings).is_ok();
        self.admitted
            .lock()
            .expect("no thread panics holding the cache")
            .insert(key, admitted);
        admitted
    }

    /// The code variants of each launch at `specialization`'s statics: every
    /// template instance an admissible configuration names.
    fn variants(&self, specialization: &NativeSpecialization) -> Vec<Vec<LaunchVariant>> {
        let admissible = self.admissible(&statics_of(specialization));
        (0..self.implementation.launches.len())
            .map(|ordinal| {
                let mut variants = Vec::<LaunchVariant>::new();
                for candidate in admissible.iter() {
                    let variant = LaunchVariant {
                        code: plan::code_values(self.implementation, candidate, ordinal),
                        group_size: self.implementation.static_group_size(candidate, ordinal),
                    };
                    if !variants.contains(&variant) {
                        variants.push(variant);
                    }
                }
                variants
            })
            .collect()
    }

    /// The programs preparation forms for `row`, every code variant of a
    /// launch-scoped implementation among their entries.
    fn programs(&self, row: &Row) -> Vec<ProgramSource> {
        let logical = self
            .module
            .entry(self.info.id, &row.bindings)
            .expect("a row's bindings are admitted");
        let dialect = row.configuration.dialect();
        match dialect {
            Dialect::Vulkan(_) => vec![ProgramSource {
                text: render_source(
                    dialect,
                    &logical,
                    &row.bindings,
                    self.implementation,
                    &row.specialization,
                    self.asset,
                ),
                entries: self
                    .implementation
                    .launches
                    .iter()
                    .map(|launch| ProgramEntry::named(launch.kernel.as_str()))
                    .collect(),
            }],
            Dialect::Cuda | Dialect::Metal(_) => super::implementation_programs(
                dialect,
                &logical,
                &row.bindings,
                self.implementation,
                &row.specialization,
                self.asset,
                &self.variants(&row.specialization),
            ),
        }
    }

    fn skeletons(&self, row: &Row) -> Vec<Skeleton> {
        let preamble = match row.configuration {
            Configuration::Cuda { architecture } => {
                format!("#define __CUDA_ARCH__ {}\n", architecture * 10)
            }
            Configuration::Vulkan(_) | Configuration::Metal { .. } => String::new(),
        };
        let dialect = row.configuration.dialect();
        self.programs(row)
            .iter()
            .map(|program| skeleton(&program.text, self.asset, &self.label, dialect, &preamble))
            .collect()
    }

    fn kernels(&self) -> Vec<&str> {
        self.implementation
            .launches
            .iter()
            .map(|launch| launch.kernel.as_str())
            .collect()
    }

    fn describe(&self, row: &Row, error: String) -> FormationFailure {
        FormationFailure {
            entry: self.info.name.clone(),
            bindings: row
                .bindings
                .iter()
                .map(|(name, representation)| {
                    format!("{name}={}", registry::representation_info(representation).name)
                })
                .collect::<Vec<_>>()
                .join(", "),
            specialization: format!("{:?}", row.specialization),
            configuration: row.configuration,
            error,
        }
    }
}

fn statics_of(specialization: &NativeSpecialization) -> NativeSpecialization {
    specialization
        .statics()
        .iter()
        .fold(NativeSpecialization::new(), |statics, (name, value)| {
            statics.with_static(name.clone(), *value)
        })
}

/// The changes of one choice from `row` the search considers.
fn variations(subject: &Subject<'_>, former: &Former, row: &Row) -> Vec<Row> {
    let mut rows = Vec::new();
    for (parameter, candidates) in &subject.candidates {
        for representation in candidates {
            if row.bindings.get(parameter) == Some(*representation) {
                continue;
            }
            let bindings = row.bindings.clone().bind(parameter, *representation);
            if subject.admits(&bindings) {
                rows.push(Row {
                    bindings,
                    ..row.clone()
                });
            }
        }
    }
    let implementation = subject.implementation;
    let statics = statics_of(&row.specialization);
    for name in &implementation.statics {
        for value in [0, 1] {
            if statics.static_value(name) == Some(value) {
                continue;
            }
            if let Some(found) = subject.statics(&[(name.as_str(), value)]) {
                rows.push(Row {
                    specialization: subject.admissible(&found)[0].clone(),
                    ..row.clone()
                });
            }
        }
    }
    let admissible = subject.admissible(&statics);
    let distance = |candidate: &NativeSpecialization| {
        candidate
            .params()
            .iter()
            .filter(|(name, value)| row.specialization.param(name) != Some(**value))
            .count()
            + candidate
                .launch_params()
                .iter()
                .filter(|((launch, name), value)| {
                    row.specialization.launch_param(*launch, name) != Some(**value)
                })
                .count()
    };
    let mut nearest = |matches: &dyn Fn(&NativeSpecialization) -> bool| {
        if let Some(candidate) = admissible
            .iter()
            .filter(|candidate| matches(candidate))
            .min_by_key(|candidate| distance(candidate))
        {
            rows.push(Row {
                specialization: candidate.clone(),
                ..row.clone()
            });
        }
    };
    for parameter in &implementation.params {
        for value in &parameter.values {
            if row.specialization.param(&parameter.name) != Some(*value) {
                nearest(&|candidate| candidate.param(&parameter.name) == Some(*value));
            }
        }
    }
    for (launch, declaration) in implementation.launches.iter().enumerate() {
        for parameter in &declaration.params {
            for value in &parameter.values {
                if row.specialization.launch_param(launch, &parameter.name) != Some(*value) {
                    nearest(&|candidate| {
                        candidate.launch_param(launch, &parameter.name) == Some(*value)
                    });
                }
            }
        }
    }
    for configuration in former.variations(row.configuration) {
        rows.push(Row {
            configuration,
            ..row.clone()
        });
    }
    rows
}

/// What preprocessing one row's skeletons established: the authored groups
/// its programs contain (`universe`, apart from `rejecting` ones that hold an
/// `#error`) and, when no `#error` rejects it, the groups it compiles.
struct Evaluated {
    reached: Option<BTreeSet<GroupSite>>,
    universe: BTreeSet<GroupSite>,
    rejecting: BTreeSet<GroupSite>,
}

/// Preprocess every row's skeletons, in parallel.
fn evaluate(subject: &Subject<'_>, rows: &[Row]) -> Result<Vec<Evaluated>, CoverageError> {
    let skeletons = in_parallel(rows, |row| gated(|| subject.skeletons(row)));
    let vulkan = rows
        .first()
        .is_some_and(|row| matches!(row.configuration, Configuration::Vulkan(_)));
    let outcomes: Vec<Vec<Outcome>> = if vulkan {
        let kernels = subject.kernels();
        in_parallel(&skeletons, |programs| {
            gated(|| {
                programs
                    .iter()
                    .map(|skeleton| preprocess_vulkan(skeleton, &kernels))
                    .collect::<Result<Vec<_>, _>>()
            })
        })
        .into_iter()
        .collect::<Result<_, _>>()?
    } else {
        // CUDA and Metal: clang over every program of every row, in
        // parallel batches.
        let texts = skeletons
            .iter()
            .flat_map(|programs| programs.iter().map(|skeleton| skeleton.text.as_str()))
            .collect::<Vec<_>>();
        for text in &texts {
            let unknown = predefined(text)
                .into_iter()
                .filter(|name| name != "__CUDA_ARCH__")
                .collect::<Vec<_>>();
            if !unknown.is_empty() {
                return Err(CoverageError::Skeleton(format!(
                    "conditions read toolchain-predefined macros {unknown:?}, which skeleton preprocessing does not define"
                )));
            }
        }
        let batches = texts.chunks(CLANG_BATCH).collect::<Vec<_>>();
        let mut flat = in_parallel(&batches, |batch| gated(|| preprocess_clang(batch)))
            .into_iter()
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .flatten();
        skeletons
            .iter()
            .map(|programs| programs.iter().map(|_| flat.next().expect("one outcome per skeleton")).collect())
            .collect()
    };
    Ok(skeletons
        .iter()
        .zip(outcomes)
        .map(|(programs, outcomes)| {
            let mut evaluated = Evaluated {
                reached: Some(BTreeSet::new()),
                universe: BTreeSet::new(),
                rejecting: BTreeSet::new(),
            };
            for (skeleton, outcome) in programs.iter().zip(outcomes) {
                let markers = match outcome {
                    Outcome::Rejected => {
                        evaluated.reached = None;
                        BTreeSet::new()
                    }
                    Outcome::Reached(markers) => markers,
                };
                for (marker, site) in skeleton.sites.iter().enumerate() {
                    if skeleton.rejecting.contains(&marker) {
                        evaluated.rejecting.insert(site.clone());
                        continue;
                    }
                    evaluated.universe.insert(site.clone());
                    if markers.contains(&marker) {
                        if let Some(reached) = &mut evaluated.reached {
                            reached.insert(site.clone());
                        }
                    }
                }
            }
            evaluated
        })
        .collect())
}

/// What forming one row established.
enum Formation {
    /// The toolchain formed these programs.
    Formed(usize),
    /// The kernel rejects the configuration: a `static_assert` fired, which
    /// is how C++ kernels decline bindings the preprocessor cannot see.
    Rejected,
    Failed(String),
    /// This host's toolchain cannot form the configuration.
    Unformable,
}

fn form_row(subject: &Subject<'_>, former: &Former, row: &Row) -> Formation {
    if !former.formable(row.configuration) {
        return Formation::Unformable;
    }
    let programs = subject.programs(row);
    match gated(|| former.form(&programs, row.configuration)) {
        Ok(()) => Formation::Formed(programs.len()),
        Err(error) if asserted(&error) => Formation::Rejected,
        Err(error) => Formation::Failed(compilation_message(&error)),
    }
}

/// Whether a compilation failed because a `static_assert` fired.
fn asserted(error: &NativeCompilationError) -> bool {
    let text = format!("{error:?}");
    text.contains("static assertion failed") || text.contains("static_assert failed")
}

/// One entry's search: its kept rows, what forming them established, and
/// the groups they reach.
#[derive(Default)]
struct Search {
    formed: usize,
    failures: Vec<FormationFailure>,
    unformable: Vec<Configuration>,
    reached: BTreeSet<GroupSite>,
    universe: BTreeSet<GroupSite>,
    rejecting: BTreeSet<GroupSite>,
}

impl Search {
    /// Record a kept row's formation; `false` when the kernel rejects it.
    fn record(&mut self, subject: &Subject<'_>, row: &Row, formation: Formation) -> bool {
        match formation {
            Formation::Formed(programs) => self.formed += programs,
            Formation::Rejected => return false,
            Formation::Failed(error) => self.failures.push(subject.describe(row, error)),
            Formation::Unformable => self.unformable.push(row.configuration),
        }
        true
    }

    fn complete(&self) -> bool {
        self.universe
            .iter()
            .all(|site| self.reached.contains(site) || self.rejecting.contains(site))
    }
}

fn search(subject: &Subject<'_>, former: &Former) -> Result<Option<Search>, CoverageError> {
    let implementation = subject.implementation;
    let Some(statics) = subject.statics(&[]) else {
        return Ok(None);
    };
    let specialization = implementation
        .default_specialization(&statics)
        .expect("searched statics admit a configuration");
    let configuration = former.base();
    let mut search = Search::default();
    // The base: the first binding in candidate order that is admitted,
    // implemented and not rejected by the toolchain.
    let mut base = None;
    let mut choice = vec![0usize; subject.candidates.len()];
    if subject.candidates.iter().all(|(_, candidates)| !candidates.is_empty()) {
        loop {
            let bindings = subject.candidates.iter().zip(&choice).fold(
                ElementBindings::new(),
                |bindings, ((parameter, candidates), index)| {
                    bindings.bind(parameter, candidates[*index])
                },
            );
            if subject.admits(&bindings) {
                let row = Row {
                    bindings,
                    specialization: specialization.clone(),
                    configuration,
                };
                let evaluated = evaluate(subject, std::slice::from_ref(&row))?
                    .pop()
                    .expect("one row evaluated");
                search.universe.extend(evaluated.universe);
                search.rejecting.extend(evaluated.rejecting);
                if let Some(reached) = evaluated.reached {
                    if search.record(subject, &row, form_row(subject, former, &row)) {
                        search.reached.extend(reached);
                        base = Some(row);
                        break;
                    }
                }
            }
            let mut position = choice.len();
            let advanced = loop {
                if position == 0 {
                    break false;
                }
                position -= 1;
                choice[position] += 1;
                if choice[position] < subject.candidates[position].1.len() {
                    break true;
                }
                choice[position] = 0;
            };
            if !advanced {
                break;
            }
        }
    }
    let Some(base) = base else {
        return Ok(None);
    };
    let mut seen = BTreeSet::from([base.key()]);
    let mut frontier = vec![base.clone()];
    let mut paired = false;
    loop {
        // Change one choice at a time from every newly kept configuration.
        while !frontier.is_empty() {
            let mut candidates = Vec::new();
            for row in &frontier {
                for candidate in variations(subject, former, row) {
                    if seen.insert(candidate.key()) {
                        candidates.push(candidate);
                    }
                }
            }
            frontier = keep(subject, former, candidates, &mut search)?;
        }
        if search.complete() || paired {
            break;
        }
        // Groups guarded by two element bindings together (a source and the
        // resident it converts into) need both changed at once: every pair
        // from the base configuration, once. Single changes then continue
        // from whatever the pairs reach.
        paired = true;
        let candidates = binding_pairs(subject, &base, &mut seen);
        frontier = keep(subject, former, candidates, &mut search)?;
    }
    let rejecting = std::mem::take(&mut search.rejecting);
    search.universe.retain(|site| !rejecting.contains(site));
    Ok(Some(search))
}

/// Evaluate `candidates` and keep each that reaches a group no kept row
/// reached and that the toolchain does not reject: candidates are chosen in
/// order and formed in parallel, and the groups of a rejected one are sought
/// again among the rest. The kept candidates are returned.
fn keep(
    subject: &Subject<'_>,
    former: &Former,
    candidates: Vec<Row>,
    search: &mut Search,
) -> Result<Vec<Row>, CoverageError> {
    let evaluated = evaluate(subject, &candidates)?;
    let mut open = Vec::new();
    for (candidate, evaluated) in candidates.into_iter().zip(evaluated) {
        search.universe.extend(evaluated.universe);
        search.rejecting.extend(evaluated.rejecting);
        if let Some(reached) = evaluated.reached {
            open.push((candidate, reached));
        }
    }
    let mut kept = Vec::new();
    loop {
        let mut claimed = search.reached.clone();
        let mut chosen = Vec::new();
        let mut rest = Vec::new();
        for (candidate, reached) in open {
            if reached.is_subset(&claimed) {
                rest.push((candidate, reached));
            } else {
                claimed.extend(reached.iter().cloned());
                chosen.push((candidate, reached));
            }
        }
        if chosen.is_empty() {
            return Ok(kept);
        }
        let formations = in_parallel(&chosen, |(candidate, _)| form_row(subject, former, candidate));
        let mut rejected = false;
        for ((candidate, reached), formation) in chosen.into_iter().zip(formations) {
            if search.record(subject, &candidate, formation) {
                search.reached.extend(reached);
                kept.push(candidate);
            } else {
                rejected = true;
            }
        }
        if !rejected {
            return Ok(kept);
        }
        open = rest;
    }
}

/// The unseen admitted configurations differing from `row` in exactly two
/// element bindings, marked seen.
fn binding_pairs(subject: &Subject<'_>, row: &Row, seen: &mut BTreeSet<String>) -> Vec<Row> {
    let mut rows = Vec::new();
    for (first, (parameter, candidates)) in subject.candidates.iter().enumerate() {
        for (other, other_candidates) in &subject.candidates[first + 1..] {
            for representation in candidates {
                if row.bindings.get(parameter) == Some(*representation) {
                    continue;
                }
                for other_representation in other_candidates {
                    if row.bindings.get(other) == Some(*other_representation) {
                        continue;
                    }
                    let candidate = Row {
                        bindings: row
                            .bindings
                            .clone()
                            .bind(parameter, *representation)
                            .bind(other, *other_representation),
                        ..row.clone()
                    };
                    if seen.insert(candidate.key()) && subject.admits(&candidate.bindings) {
                        rows.push(candidate);
                    }
                }
            }
        }
    }
    rows
}

/// Form every native implementation of `module` for `backend`. Entries are
/// searched in parallel, each search preprocessing and forming its
/// candidates in parallel; the toolchain work itself runs at most one job
/// per core.
pub fn cover(module: &CheckedModule, backend: BackendName) -> Result<Coverage, CoverageError> {
    let former = Former::open(backend)?;
    let subjects = module
        .entries()
        .iter()
        .filter_map(|info| {
            let candidates = info
                .element_parameters
                .iter()
                .map(|parameter| (parameter.clone(), representations()))
                .collect();
            Subject::new(module, info, backend, candidates)
        })
        .collect::<Vec<_>>();
    let searches = in_parallel(&subjects, |subject| search(subject, &former));
    let mut coverage = Coverage::default();
    let mut reached = BTreeSet::new();
    let mut universe = BTreeSet::new();
    for (subject, searched) in subjects.iter().zip(searches) {
        let info = subject.info;
        let Some(searched) = searched? else {
            coverage.unimplemented.push(info.name.clone());
            continue;
        };
        coverage.formed += searched.formed;
        coverage.failures.extend(searched.failures);
        coverage.unformable.extend(
            searched
                .unformable
                .into_iter()
                .map(|configuration| (configuration, info.name.clone())),
        );
        reached.extend(searched.reached);
        universe.extend(searched.universe);
    }
    coverage.groups = universe.len();
    coverage.reached = reached.len();
    coverage.unreached = universe.difference(&reached).cloned().collect();
    Ok(coverage)
}

/// `work` over every item on every core, results in item order.
fn in_parallel<T: Sync, R: Send>(items: &[T], work: impl Fn(&T) -> R + Sync) -> Vec<R> {
    let next = std::sync::atomic::AtomicUsize::new(0);
    let results = std::sync::Mutex::new(Vec::with_capacity(items.len()));
    let workers = std::thread::available_parallelism()
        .map_or(1, usize::from)
        .min(items.len());
    std::thread::scope(|scope| {
        for _ in 0..workers {
            scope.spawn(|| loop {
                let index = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                let Some(item) = items.get(index) else {
                    return;
                };
                let result = work(item);
                results
                    .lock()
                    .expect("no worker panics holding the lock")
                    .push((index, result));
            });
        }
    });
    let mut results = results.into_inner().expect("the lock is not poisoned");
    results.sort_by_key(|(index, _)| *index);
    results.into_iter().map(|(_, result)| result).collect()
}

/// Run `work` holding one of a core's worth of permits: the toolchain work
/// of every concurrent search shares the cores without oversubscribing them.
fn gated<R>(work: impl FnOnce() -> R) -> R {
    static PERMITS: std::sync::OnceLock<(std::sync::Mutex<usize>, std::sync::Condvar)> =
        std::sync::OnceLock::new();
    let (free, released) = PERMITS.get_or_init(|| {
        (
            std::sync::Mutex::new(std::thread::available_parallelism().map_or(1, usize::from)),
            std::sync::Condvar::new(),
        )
    });
    {
        let mut free = released
            .wait_while(free.lock().expect("no thread panics holding the permits"), |free| *free == 0)
            .expect("no thread panics holding the permits");
        *free -= 1;
    }
    /// Returns the permit however `work` exits.
    struct Permit<'p>(&'p std::sync::Mutex<usize>, &'p std::sync::Condvar);
    impl Drop for Permit<'_> {
        fn drop(&mut self) {
            *self.0.lock().expect("no thread panics holding the permits") += 1;
            self.1.notify_one();
        }
    }
    let _permit = Permit(free, released);
    work()
}

fn compilation_message(error: &NativeCompilationError) -> String {
    let text = format!("{error:?}");
    text.lines()
        .filter(|line| line.contains("ERROR") || line.contains("error"))
        .take(4)
        .collect::<Vec<_>>()
        .join(" | ")
}

/// A kernel request a program makes: an entry at element bindings and
/// static values.
#[derive(Clone, Debug)]
pub struct Request {
    pub entry: EntryId,
    pub bindings: ElementBindings,
    pub statics: NativeSpecialization,
}

/// Why a request cannot be prepared on a backend.
#[derive(Debug)]
pub enum RequestFailure {
    /// The statics lie outside the implementation's domain.
    Inadmissible(String),
    /// The kernel rejects the binding (`#error` or `static_assert`).
    Unimplemented,
    /// The toolchain refuses the request's default configuration.
    Formation(String),
}

impl fmt::Display for RequestFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Inadmissible(reason) => write!(f, "statics outside the kernel's domain: {reason}"),
            Self::Unimplemented => f.write_str("the kernel rejects the binding"),
            Self::Formation(error) => write!(f, "formation failed: {error}"),
        }
    }
}

/// Forms requests as preparation forms them on one backend's base
/// configuration, at each request's default specialization.
pub struct RequestFormer {
    former: Former,
    backend: BackendName,
}

impl RequestFormer {
    pub fn open(backend: BackendName) -> Result<Self, CoverageError> {
        Ok(Self {
            former: Former::open(backend)?,
            backend,
        })
    }

    pub fn backend(&self) -> BackendName {
        self.backend
    }

    pub fn form(&self, module: &CheckedModule, request: &Request) -> Result<(), RequestFailure> {
        let info = module
            .entries()
            .iter()
            .find(|info| info.id == request.entry)
            .expect("a request names an entry of the module");
        let subject = Subject::new(module, info, self.backend, Vec::new())
            .expect("a request names an entry with a native implementation");
        let specialization = subject
            .implementation
            .default_specialization(&request.statics)
            .map_err(|error| RequestFailure::Inadmissible(error.to_string()))?;
        let row = Row {
            bindings: request.bindings.clone(),
            specialization,
            configuration: self.former.base(),
        };
        let evaluated = evaluate(&subject, std::slice::from_ref(&row))
            .map_err(|error| RequestFailure::Formation(error.to_string()))?
            .pop()
            .expect("one row evaluated");
        if evaluated.reached.is_none() {
            return Err(RequestFailure::Unimplemented);
        }
        match form_row(&subject, &self.former, &row) {
            Formation::Formed(_) | Formation::Unformable => Ok(()),
            Formation::Rejected => Err(RequestFailure::Unimplemented),
            Formation::Failed(error) => Err(RequestFailure::Formation(error)),
        }
    }

    /// [`Self::form`] over every request, in parallel, results in order.
    pub fn form_all(
        &self,
        module: &CheckedModule,
        requests: &[Request],
    ) -> Vec<Result<(), RequestFailure>> {
        in_parallel(requests, |request| self.form(module, request))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn skeletons_mark_authored_groups_and_follow_line_markers() {
        let asset = "#if A\nx\n#line 1 \"lib/x.glsl\"\n#ifdef B\n#error \"no\"\n#endif\n#line 3 \"k.comp\"\n#else\ny\n#endif\n";
        let text = format!("#version 460\n#define A 1\n#if Q\n#endif\n{asset}void main() {{}}\n");
        let skeleton = skeleton(
            &text,
            asset,
            "k.comp",
            Dialect::Vulkan(abi::vulkan::VulkanFeatures::of(&VULKAN_BASE)),
            "",
        );
        assert_eq!(
            skeleton.sites,
            vec![
                GroupSite { file: "k.comp".into(), line: 1 },
                GroupSite { file: "lib/x.glsl".into(), line: 1 },
                GroupSite { file: "k.comp".into(), line: 3 },
            ]
        );
        assert_eq!(skeleton.rejecting, BTreeSet::from([1]));
        assert!(skeleton.text.starts_with("#version 460\n#define A 1\n#if Q\n#endif\n#if A\n"));
        assert!(skeleton.text.ends_with("void main() {}\n"));
    }

    #[test]
    fn markers_are_read_back() {
        assert_eq!(
            reached_markers("const int seismic_group_3 = 0;\nconst int seismic_group_12 = 0;\n"),
            BTreeSet::from([3, 12])
        );
    }
}
