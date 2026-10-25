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
//! Planning gives a weight any representation of its backend's resident row
//! layout, and gives each weight its own. So a weight slot (an element
//! parameter the entry reads, other than a conversion target, which its
//! source chooses) that takes one of them must take every one: the base
//! configuration with that slot changed to it, and the slots the entry's
//! element predicates tie to its operand recipe, must be implemented. A
//! representation it is not implemented at is reported as untaken. Every
//! change of a binding the search makes carries those tied slots with it.
//!
//! A `form` tuning parameter selects a structurally different algorithm,
//! which a kernel may take without a preprocessor group (a template chosen
//! by a constant, say) and which its declaration may admit only at some
//! statics (a head width). Every value of every form parameter is therefore
//! formed as well, once under each device configuration: by a kept
//! configuration that holds it, or else by the first configuration the
//! declaration admits with it, at statics searched for it. A form value no
//! configuration forms is reported like an unreached group.
//!
//! Vulkan skeletons are preprocessed by glslang, the toolchain itself; CUDA
//! and Metal skeletons by `clang -E`. Their conditions may read no
//! toolchain-predefined macro except the architecture macros, which the CUDA
//! skeleton defines as the configuration's formation target does
//! (`seismic_cuda::nvrtc::architecture_macros`).
//!
//! The search decides what to form; the caller's [`Forming`] forms it, each
//! [`Formation`] identified by everything it reads.

mod forming;

pub use forming::{form, Formation, Forming, InProcess, Progress, Target, Verdict};
pub use seismic_native_target::{ProgramEntry, ProgramSource};

use super::abi::{self, Dialect, render_source};
use super::plan::{self, LaunchVariant};
use seismic_lang::checked::{
    CheckedModule, ElementDomain, ElementTarget, ElementTerm, EntryInfo, NativeImplementation,
    NativeParameter, NativeSpecialization,
};
use seismic_lang::entry::ElementBindings;
use seismic_lang::ids::{EntryId, RepresentationId};
use seismic_lang::registry::{self, BackendName};
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
            Self::Cuda { .. } => Dialect::Cuda(abi::CudaFeatures::UNOPENED),
            Self::Metal { tensor_ops } => Dialect::Metal(abi::MetalFeatures { tensor_ops }),
        }
    }
}

/// An authored preprocessor group: the file (relative to its asset's
/// directory; Seismic library files as `seismic/<name>`) and the line of the
/// directive that opens it. Searches hold sites by the million, so the file
/// name is shared.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GroupSite {
    pub file: std::sync::Arc<str>,
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

/// A resident representation a weight slot of an entry does not take,
/// though the slot takes other representations of the backend's resident
/// layout.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Untaken {
    pub entry: String,
    pub parameter: String,
    pub representation: &'static str,
}

impl fmt::Display for Untaken {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "`{}` {}={}",
            self.entry, self.parameter, self.representation
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
    /// Values of `form` tuning parameters formed, of all the implemented
    /// entries declare.
    pub forms_formed: usize,
    pub forms: usize,
    /// Form values no configuration forms: none is admissible over the
    /// searched statics, or the kernel rejects each one.
    pub unformed: Vec<String>,
    /// Entries with no implemented configuration at all.
    pub unimplemented: Vec<String>,
    /// Resident representations a weight slot does not take, though it
    /// takes others ([`untaken`]).
    pub untaken: Vec<Untaken>,
    /// Configurations this host's toolchain cannot form (Metal tensor
    /// operations on a compiler without Metal 4), with the entries whose
    /// kept configurations needed them.
    pub unformable: Vec<(Configuration, String)>,
}

impl Coverage {
    /// Whether every implementation formed, every group was reached and
    /// every form was formed.
    pub fn complete(&self) -> bool {
        self.failures.is_empty()
            && self.unreached.is_empty()
            && self.unformed.is_empty()
            && self.unimplemented.is_empty()
            && self.untaken.is_empty()
    }
}

impl fmt::Display for Coverage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "formed {} programs; reached {} of {} groups; formed {} of {} forms",
            self.formed, self.reached, self.groups, self.forms_formed, self.forms
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
        for untaken in &self.untaken {
            writeln!(f, "untaken representation: {untaken}")?;
        }
        for form in &self.unformed {
            writeln!(f, "unformed form: {form}")?;
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

/// One backend's formation toolchain on this host.
pub struct Toolchain {
    former: Former,
}

impl Toolchain {
    /// The toolchain for `backend`, or why this host cannot form for it.
    pub fn open(backend: BackendName) -> Result<Self, CoverageError> {
        Ok(Self {
            former: Former::open(backend)?,
        })
    }

    pub fn backend(&self) -> BackendName {
        self.former.backend()
    }

    /// What identifies this toolchain across every configuration it forms,
    /// with the preprocessor that chooses the configurations.
    pub fn identity(&self) -> String {
        self.former.identity()
    }
}

/// One backend's toolchain as preparation forms with it: the configurations
/// it forms, and what identifies and forms each formation.
enum Former {
    Vulkan,
    Cuda {
        architectures: Vec<u32>,
        release: (u32, u32),
        /// `clang --version`, which preprocesses the skeletons.
        preprocessor: String,
    },
    #[cfg(target_os = "macos")]
    Metal {
        language: seismic_metal::facts::LanguageVersion,
        tensor_ops: bool,
        operating_system: String,
        preprocessor: String,
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
                let unavailable = |error| CoverageError::ToolchainUnavailable(format!("{error:?}"));
                let architectures = seismic_cuda::nvrtc::supported_architectures()
                    .map_err(unavailable)?
                    .into_iter()
                    .filter(|architecture| *architecture >= floor)
                    .collect();
                Ok(Self::Cuda {
                    architectures,
                    release: seismic_cuda::nvrtc::release().map_err(unavailable)?,
                    preprocessor: clang_identity()?,
                })
            }
            #[cfg(target_os = "macos")]
            BackendName::Metal => {
                let device = seismic_metal::DeviceHandle::system_default()
                    .map_err(|error| CoverageError::ToolchainUnavailable(format!("{error:?}")))?;
                let language =
                    seismic_metal::profile::probe_language_version(&device).ok_or_else(|| {
                        CoverageError::ToolchainUnavailable(
                            "no supported Metal language version".into(),
                        )
                    })?;
                let tensor_ops =
                    seismic_metal::toolchain::forms_tensor_operations(&device, language);
                Ok(Self::Metal {
                    language,
                    tensor_ops,
                    operating_system: seismic_metal::profile::operating_system(),
                    preprocessor: clang_identity()?,
                })
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
            Self::Cuda { architectures, .. } => Configuration::Cuda {
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
            (Self::Cuda { architectures, .. }, Configuration::Cuda { architecture }) => {
                architectures
                    .iter()
                    .filter(|candidate| **candidate != architecture)
                    .map(|architecture| Configuration::Cuda {
                        architecture: *architecture,
                    })
                    .collect()
            }
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

    /// The row layout weights are resident in on this backend.
    fn resident_layout(&self) -> registry::Layout {
        self.backend().resident_layout()
    }

    fn backend(&self) -> BackendName {
        match self {
            Self::Vulkan => BackendName::Vulkan,
            Self::Cuda { .. } => BackendName::Cuda,
            #[cfg(target_os = "macos")]
            Self::Metal { .. } => BackendName::Metal,
        }
    }

    /// What identifies this toolchain across every configuration it forms,
    /// with the preprocessor that chooses the configurations.
    fn identity(&self) -> String {
        match self {
            Self::Vulkan => seismic_vulkan::formation::compiler(),
            Self::Cuda {
                architectures,
                release,
                preprocessor,
            } => format!(
                "cuda;nvrtc {}.{};{};sm {architectures:?};{preprocessor}",
                release.0,
                release.1,
                seismic_cuda::nvrtc::COMPILE_OPTIONS.join(" ")
            ),
            #[cfg(target_os = "macos")]
            Self::Metal {
                language,
                tensor_ops,
                operating_system,
                preprocessor,
            } => format!(
                "{};tensor_ops {tensor_ops};{preprocessor}",
                seismic_metal::toolchain::material(operating_system, *language)
            ),
        }
    }

    /// The toolchain namespace and the identity material of a formation
    /// under `configuration`: with the program, everything it reads.
    fn material(&self, configuration: Configuration) -> (&'static str, String) {
        match (self, configuration) {
            (Self::Vulkan, Configuration::Vulkan(facts)) => (
                "vulkan",
                format!("{};{facts:?}", seismic_vulkan::formation::compiler()),
            ),
            (Self::Cuda { release, .. }, Configuration::Cuda { architecture }) => (
                "cuda",
                seismic_cuda::toolchain::material(&seismic_cuda::nvrtc::Formation {
                    release: *release,
                    architecture,
                    options: seismic_cuda::nvrtc::COMPILE_OPTIONS,
                }),
            ),
            #[cfg(target_os = "macos")]
            (
                Self::Metal {
                    language,
                    operating_system,
                    ..
                },
                Configuration::Metal { tensor_ops },
            ) => (
                "metal",
                format!(
                    "{};tensor_ops {tensor_ops}",
                    seismic_metal::toolchain::material(operating_system, *language)
                ),
            ),
            _ => unreachable!("a former identifies only its own configurations"),
        }
    }

    /// Forming `program` under `configuration`, with what identifies it.
    fn formation<'a>(
        &self,
        configuration: Configuration,
        program: &'a ProgramSource,
    ) -> Formation<'a> {
        let (namespace, material) = self.material(configuration);
        let target = match (self, configuration) {
            (Self::Vulkan, Configuration::Vulkan(facts)) => Target::Vulkan(facts),
            (Self::Cuda { .. }, Configuration::Cuda { architecture }) => {
                Target::Cuda { architecture }
            }
            #[cfg(target_os = "macos")]
            (Self::Metal { language, .. }, Configuration::Metal { .. }) => Target::Metal {
                language: *language,
            },
            _ => unreachable!("a former targets only its own configurations"),
        };
        Formation {
            namespace,
            material,
            target,
            program,
        }
    }
}

/// A search's toolchain and how its formations run.
#[derive(Clone, Copy)]
struct Forms<'a> {
    former: &'a Former,
    forming: &'a dyn Forming,
}

impl Forms<'_> {
    fn form(&self, configuration: Configuration, program: &ProgramSource) -> Verdict {
        self.forming
            .form(self.former.formation(configuration, program))
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
    // The last occurrence, sought from the end: at most a generated suffix
    // follows the asset, so few starts are compared and no searcher is built
    // over the (long) asset.
    let asset_start = (0..=text.len().saturating_sub(asset.len()))
        .rev()
        .find(|start| text.as_bytes()[*start..].starts_with(asset.as_bytes()))
        .expect("a rendered source contains its asset");
    let asset_end = asset_start + asset.len();
    let mut out = String::new();
    let mut sites = Vec::new();
    let mut rejecting = BTreeSet::new();
    // Open authored groups, innermost last, by marker index.
    let mut open: Vec<Option<usize>> = Vec::new();
    let mut file: std::sync::Arc<str> = label.into();
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
                    file = name.trim_matches('"').into();
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
        let text =
            seismic_vulkan::formation::preprocess(&skeleton.text, kernel).map_err(|error| {
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
            std::io::ErrorKind::NotFound => {
                CoverageError::ToolchainUnavailable(format!("clang: {error}"))
            }
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

/// `clang --version`: the preprocessor that decides which CUDA and Metal
/// configurations a search keeps.
fn clang_identity() -> Result<String, CoverageError> {
    let output = std::process::Command::new("clang")
        .arg("--version")
        .output()
        .map_err(|error| CoverageError::ToolchainUnavailable(format!("clang: {error}")))?;
    if !output.status.success() {
        return Err(CoverageError::ToolchainUnavailable(format!(
            "clang --version: {}",
            String::from_utf8_lossy(&output.stderr)
        )));
    }
    Ok(String::from_utf8_lossy(&output.stdout).trim().to_owned())
}

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
    searched: std::sync::Mutex<
        std::collections::HashMap<Vec<(String, u64)>, Option<NativeSpecialization>>,
    >,
    /// Admissible specializations, by statics.
    specializations: std::sync::Mutex<
        std::collections::HashMap<NativeSpecialization, std::sync::Arc<Admissible<'m>>>,
    >,
    /// Whether programs are formed per code variant (CUDA and Metal).
    variant_programs: bool,
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
            variant_programs: backend != BackendName::Vulkan,
        })
    }

    /// [`NativeImplementation::search_statics`], once per `fixed`.
    fn statics(&self, fixed: &[(&str, u64)]) -> Option<NativeSpecialization> {
        let key = fixed
            .iter()
            .map(|(name, value)| ((*name).to_owned(), *value))
            .collect::<Vec<_>>();
        let mut searched = self
            .searched
            .lock()
            .expect("no thread panics holding the cache");
        searched
            .entry(key)
            .or_insert_with(|| self.implementation.search_statics(fixed))
            .clone()
    }

    /// The admissible specializations at `statics`, walked once.
    fn admissible(&self, statics: &NativeSpecialization) -> std::sync::Arc<Admissible<'m>> {
        let mut walked = self
            .specializations
            .lock()
            .expect("no thread panics holding the cache");
        walked
            .entry(statics.clone())
            .or_insert_with(|| {
                std::sync::Arc::new(Admissible::walk(
                    self.implementation,
                    statics,
                    self.variant_programs,
                ))
            })
            .clone()
    }

    /// `bindings` with `parameter` bound to `representation`, and every
    /// parameter the entry's element predicates declare shares its operand
    /// recipe (`operands(P) == operands(Q)`) and holds another bound to it
    /// too: one choice of the search, the smallest change that keeps those
    /// predicates.
    fn rebind(
        &self,
        bindings: &ElementBindings,
        parameter: &str,
        representation: RepresentationId,
    ) -> ElementBindings {
        let recipe = |representation| registry::representation_info(representation).operands;
        let mut tied = vec![parameter.to_owned()];
        let mut index = 0;
        while let Some(current) = tied.get(index).cloned() {
            for predicate in self.info.element_domain.predicates() {
                let (ElementTerm::Operands(left), ElementTerm::Operands(right)) =
                    (&predicate.left, &predicate.right)
                else {
                    continue;
                };
                if !predicate.equal {
                    continue;
                }
                for (from, to) in [(left, right), (right, left)] {
                    if *from == current && !tied.contains(to) {
                        tied.push(to.clone());
                    }
                }
            }
            index += 1;
        }
        tied.iter().fold(bindings.clone(), |bindings, name| {
            let held = bindings.get(name);
            if name == parameter || held.is_some_and(|held| recipe(held) != recipe(representation)) {
                bindings.bind(name, representation)
            } else {
                bindings
            }
        })
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
        self.admissible(&statics_of(specialization)).variants.clone()
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
            Dialect::Cuda(_) | Dialect::Metal(_) => super::implementation_programs(
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

    /// How many programs [`Self::programs`] renders for a row: one per
    /// launch for a launch-scoped CUDA or Metal implementation, else one.
    fn programs_per_row(&self) -> usize {
        if self.variant_programs && self.implementation.launch_scoped() {
            self.implementation.launches.len().max(1)
        } else {
            1
        }
    }

    fn skeletons(&self, row: &Row) -> Vec<Skeleton> {
        let preamble = match row.configuration {
            Configuration::Cuda { architecture } => {
                seismic_cuda::nvrtc::architecture_macros(architecture)
                    .into_iter()
                    .map(|(name, value)| format!("#define {name} {value}\n"))
                    .collect()
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
                    format!(
                        "{name}={}",
                        registry::representation_info(representation).name
                    )
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

/// The admissible specializations at one set of statics. Where the `where`
/// condition cannot fail to evaluate at them, a walk visits exactly the
/// configurations of its parameter product the condition admits, in order,
/// so each question is answered by walks over a part of the product, and the
/// first configuration such a walk visits is the first of the whole walk's
/// within that part. Otherwise each question is answered by one walk over
/// every admissible configuration, holding only its answers: a declaration
/// can admit hundreds of millions, so they are never held.
struct Admissible<'m> {
    implementation: &'m NativeImplementation,
    statics: NativeSpecialization,
    axes: Vec<(Option<usize>, &'m NativeParameter)>,
    first: NativeSpecialization,
    /// Whether the condition can fail to evaluate at `statics`, so questions
    /// are answered by whole walks rather than walks over parts.
    fallible: bool,
    variants: Vec<Vec<LaunchVariant>>,
}

/// A parameter a configuration leaves unset.
const UNSET: u16 = u16::MAX;

impl<'m> Admissible<'m> {
    fn walk(implementation: &'m NativeImplementation, statics: &NativeSpecialization, variant_programs: bool) -> Self {
        if implementation.infallible(statics) {
            Self::declared(implementation, statics, variant_programs)
        } else {
            Self::walked(implementation, statics, variant_programs)
        }
    }

    fn new(implementation: &'m NativeImplementation, statics: &NativeSpecialization) -> Self {
        Self {
            implementation,
            statics: statics.clone(),
            axes: implementation
                .params
                .iter()
                .map(|parameter| (None, parameter))
                .chain(implementation.launches.iter().enumerate().flat_map(
                    |(launch, declaration)| {
                        declaration
                            .params
                            .iter()
                            .map(move |parameter| (Some(launch), parameter))
                    },
                ))
                .collect(),
            first: NativeSpecialization::new(),
            fallible: false,
            variants: Vec::new(),
        }
    }

    /// Answered by walks over parts of the product: only where the condition
    /// cannot fail to evaluate at `statics`.
    fn declared(
        implementation: &'m NativeImplementation,
        statics: &NativeSpecialization,
        variant_programs: bool,
    ) -> Self {
        let mut admissible = Self::new(implementation, statics);
        admissible.first = implementation
            .default_specialization(statics)
            .expect("a row's statics admit its specialization");
        if variant_programs {
            admissible.variants = (0..implementation.launches.len())
                .map(|launch| admissible.launch_variants(launch))
                .collect();
        }
        admissible
    }

    fn walked(
        implementation: &'m NativeImplementation,
        statics: &NativeSpecialization,
        variant_programs: bool,
    ) -> Self {
        let mut admissible = Self::new(implementation, statics);
        admissible.fallible = true;
        admissible.variants = vec![
            Vec::new();
            if variant_programs {
                implementation.launches.len()
            } else {
                0
            }
        ];
        let mut first = None;
        implementation
            .walk_admissible(statics, |candidate| {
                if first.is_none() {
                    first = Some(candidate.clone());
                }
                for (ordinal, variants) in admissible.variants.iter_mut().enumerate() {
                    let variant = variant_of(implementation, candidate, ordinal);
                    if !variants.contains(&variant) {
                        variants.push(variant);
                    }
                }
                true
            })
            .expect("a row's statics admit its specialization");
        admissible.first = first.expect("a row's statics admit its specialization");
        admissible
    }

    /// `specialization`'s code on every axis.
    fn codes_of(&self, specialization: &NativeSpecialization) -> Vec<u16> {
        self.axes
            .iter()
            .map(|(launch, parameter)| {
                let value = match launch {
                    None => specialization.param(&parameter.name),
                    Some(launch) => specialization.launch_param(*launch, &parameter.name),
                };
                value
                    .and_then(|value| parameter.values.iter().position(|declared| *declared == value))
                    .map_or(UNSET, |code| code as u16)
            })
            .collect()
    }

    /// The first admissible configuration in walk order whose every
    /// parameter takes one of `values` (one list per axis, in declaration
    /// order). Only for statics at which the condition cannot fail.
    fn first_within(&self, values: &[Vec<u64>]) -> Option<NativeSpecialization> {
        let mut within = self.implementation.clone();
        let mut lists = values.iter();
        for parameter in within.params.iter_mut().chain(
            within
                .launches
                .iter_mut()
                .flat_map(|launch| launch.params.iter_mut()),
        ) {
            parameter.values = lists.next().expect("one list per axis").clone();
        }
        let within = within.narrowed(&self.statics)?;
        let mut first = None;
        within
            .walk_admissible(&self.statics, |configuration| {
                first = Some(configuration.clone());
                false
            })
            .expect("a walk at statics where the condition cannot fail succeeds");
        first
    }

    /// The code variants of launch `launch` in the order of their first
    /// configurations: each from the first configuration holding one
    /// combination of the parameters a variant reads.
    fn launch_variants(&self, launch: usize) -> Vec<LaunchVariant> {
        let declaration = &self.implementation.launches[launch];
        let mut extent = Vec::new();
        for expression in &declaration.group_extent {
            expression.parameters(&mut extent);
        }
        let reads = self
            .axes
            .iter()
            .enumerate()
            .filter(|(_, (scope, parameter))| {
                scope.is_none_or(|scope| scope == launch)
                    && (parameter.code || extent.contains(&parameter.name))
            })
            .map(|(axis, _)| axis)
            .collect::<Vec<_>>();
        let mut firsts = Vec::new();
        let mut choice = vec![0usize; reads.len()];
        loop {
            let values = self
                .axes
                .iter()
                .enumerate()
                .map(
                    |(axis, (_, parameter))| match reads.iter().position(|read| *read == axis) {
                        Some(at) => vec![parameter.values[choice[at]]],
                        None => parameter.values.clone(),
                    },
                )
                .collect::<Vec<_>>();
            if let Some(first) = self.first_within(&values) {
                firsts.push((
                    self.codes_of(&first),
                    variant_of(self.implementation, &first, launch),
                ));
            }
            let Some(at) = (0..reads.len())
                .rev()
                .find(|at| choice[*at] + 1 < self.axes[reads[*at]].1.values.len())
            else {
                break;
            };
            choice[at] += 1;
            for later in &mut choice[at + 1..] {
                *later = 0;
            }
        }
        firsts.sort();
        let mut variants = Vec::new();
        for (_, variant) in firsts {
            if !variants.contains(&variant) {
                variants.push(variant);
            }
        }
        variants
    }

    /// For each of `helds`: for every axis and every code of it the held
    /// codes do not hold, in axis then code order, the first configuration in
    /// walk order holding that code with the fewest set parameters differing
    /// from them; codes no configuration holds are skipped.
    fn nearest_each(&self, helds: &[Vec<u16>]) -> Vec<Vec<NativeSpecialization>> {
        if !self.fallible {
            return helds
                .iter()
                .map(|held| {
                    self.axes
                        .iter()
                        .enumerate()
                        .flat_map(|(axis, (_, parameter))| {
                            (0..parameter.values.len() as u16)
                                .filter(move |code| held[axis] != *code)
                                .filter_map(move |code| self.nearest(held, axis, code))
                        })
                        .collect()
                })
                .collect();
        }
        // One walk for every held: each configuration is a candidate for the
        // code it holds on each axis, at its distance from each held. A
        // strictly nearer candidate replaces an answer, so ties keep the first
        // in walk order.
        let empty = self
            .axes
            .iter()
            .map(|(_, parameter)| vec![None; parameter.values.len()])
            .collect::<Vec<Vec<Option<(usize, NativeSpecialization)>>>>();
        let mut nearest = vec![empty; helds.len()];
        self.implementation
            .walk_admissible(&self.statics, |candidate| {
                let codes = self.codes_of(candidate);
                debug_assert_eq!(
                    codes.iter().filter(|code| **code != UNSET).count(),
                    candidate.params().len() + candidate.launch_params().len(),
                    "every parameter a configuration sets is declared"
                );
                for (held, nearest) in helds.iter().zip(&mut nearest) {
                    let distance = codes
                        .iter()
                        .zip(held)
                        .filter(|(code, held)| **code != UNSET && code != held)
                        .count();
                    for (axis, code) in codes.iter().enumerate() {
                        if *code == UNSET || *code == held[axis] {
                            continue;
                        }
                        let answer = &mut nearest[axis][*code as usize];
                        if answer.as_ref().is_none_or(|(least, _)| distance < *least) {
                            *answer = Some((distance, candidate.clone()));
                        }
                    }
                }
                true
            })
            .expect("a row's statics admit its specialization");
        nearest
            .into_iter()
            .map(|answers| {
                answers
                    .into_iter()
                    .flatten()
                    .flatten()
                    .map(|(_, specialization)| specialization)
                    .collect()
            })
            .collect()
    }

    /// The first configuration in walk order holding `code` on `axis` with
    /// the fewest set parameters differing from `held`, answered by walks
    /// over parts of the product. Only for statics at which the condition
    /// cannot fail.
    fn nearest(&self, held: &[u16], axis: usize, code: u16) -> Option<NativeSpecialization> {
        // Every configuration differs from `held` on the axes it leaves
        // unset; on each other axis a configuration holds its value or one
        // of the rest. So the configurations differing in exactly the axes
        // of a set are a product, and the nearest is the first in walk order
        // of the first products, by the size of the set, that admit one.
        let open = (0..self.axes.len())
            .filter(|other| {
                *other != axis && held[*other] != UNSET && self.axes[*other].1.values.len() > 1
            })
            .collect::<Vec<_>>();
        // The first configuration holding `code` bounds the sets to try.
        let holding = self
            .axes
            .iter()
            .enumerate()
            .map(|(other, (_, parameter))| {
                if other == axis {
                    vec![parameter.values[code as usize]]
                } else {
                    parameter.values.clone()
                }
            })
            .collect::<Vec<_>>();
        let first = self.first_within(&holding)?;
        let codes = self.codes_of(&first);
        let bound = open
            .iter()
            .filter(|other| codes[**other] != held[**other])
            .count();
        for size in 0..bound {
            let mut nearest: Option<(Vec<u16>, NativeSpecialization)> = None;
            let mut set = (0..size).collect::<Vec<_>>();
            loop {
                let values = self
                    .axes
                    .iter()
                    .enumerate()
                    .map(|(other, (_, parameter))| {
                        if other == axis {
                            vec![parameter.values[code as usize]]
                        } else if held[other] == UNSET {
                            parameter.values.clone()
                        } else if set.iter().any(|at| open[*at] == other) {
                            parameter
                                .values
                                .iter()
                                .enumerate()
                                .filter(|(index, _)| *index != held[other] as usize)
                                .map(|(_, value)| *value)
                                .collect()
                        } else {
                            vec![parameter.values[held[other] as usize]]
                        }
                    })
                    .collect::<Vec<_>>();
                if let Some(first) = self.first_within(&values) {
                    let codes = self.codes_of(&first);
                    if nearest.as_ref().is_none_or(|(least, _)| codes < *least) {
                        nearest = Some((codes, first));
                    }
                }
                // The next set of `size` open axes, in lexicographic order.
                let Some(at) = (0..size).rev().find(|at| set[*at] < open.len() - size + at) else {
                    break;
                };
                set[at] += 1;
                for later in at + 1..size {
                    set[later] = set[later - 1] + 1;
                }
            }
            if let Some((_, nearest)) = nearest {
                return Some(nearest);
            }
        }
        // No nearer one: the nearest differ in `bound` open axes, and the
        // first in walk order of all holding `code` is one of them.
        Some(first)
    }
}

/// The code variant of launch `launch` a configuration names.
fn variant_of(
    implementation: &NativeImplementation,
    configuration: &NativeSpecialization,
    launch: usize,
) -> LaunchVariant {
    LaunchVariant {
        code: plan::code_values(implementation, configuration, launch),
        group_size: implementation.static_group_size(configuration, launch),
    }
}

/// The changes of one choice from each of `rows` the search considers, in
/// row order. The nearest configurations of rows sharing statics come from
/// one set of walks.
fn variations(subject: &Subject<'_>, former: &Former, rows: &[Row]) -> Vec<Row> {
    let mut groups: Vec<(NativeSpecialization, Vec<usize>)> = Vec::new();
    for (index, row) in rows.iter().enumerate() {
        let statics = statics_of(&row.specialization);
        match groups.iter_mut().find(|(held, _)| *held == statics) {
            Some((_, indices)) => indices.push(index),
            None => groups.push((statics, vec![index])),
        }
    }
    let mut nearest = vec![Vec::new(); rows.len()];
    for (statics, indices) in groups {
        let admissible = subject.admissible(&statics);
        let helds = indices
            .iter()
            .map(|index| admissible.codes_of(&rows[*index].specialization))
            .collect::<Vec<_>>();
        for (index, answers) in indices.into_iter().zip(admissible.nearest_each(&helds)) {
            nearest[index] = answers;
        }
    }
    rows.iter()
        .zip(nearest)
        .flat_map(|(row, nearest)| row_variations(subject, former, row, nearest))
        .collect()
}

/// The changes of one choice from `row`: its bindings, its statics, the
/// `nearest` configuration holding each parameter value it does not hold
/// (ties to the first in walk order), and its device configuration.
fn row_variations(
    subject: &Subject<'_>,
    former: &Former,
    row: &Row,
    nearest: Vec<NativeSpecialization>,
) -> Vec<Row> {
    let mut rows = Vec::new();
    for (parameter, candidates) in &subject.candidates {
        for representation in candidates {
            if row.bindings.get(parameter) == Some(*representation) {
                continue;
            }
            let bindings = subject.rebind(&row.bindings, parameter, *representation);
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
                    specialization: subject.admissible(&found).first.clone(),
                    ..row.clone()
                });
            }
        }
    }
    for specialization in nearest {
        rows.push(Row {
            specialization,
            ..row.clone()
        });
    }
    for configuration in former.variations(row.configuration) {
        rows.push(Row {
            configuration,
            ..row.clone()
        });
    }
    rows
}

/// What preprocessing rows' skeletons established: for each row, when no
/// `#error` rejects it, the groups it compiles; and over all the rows, the
/// authored groups their programs contain (`universe`, apart from
/// `rejecting` ones that hold an `#error`).
struct Evaluation {
    reached: Vec<Option<BTreeSet<GroupSite>>>,
    universe: BTreeSet<GroupSite>,
    rejecting: BTreeSet<GroupSite>,
}

/// Preprocess every row's skeletons. A round can hold tens of thousands of
/// rows, each rendering several programs, so rows are evaluated a chunk at a
/// time under one permit (a chunk is one clang batch of programs): a chunk's
/// skeletons live only while it is preprocessed, and at most one chunk per
/// core is held, whatever the round's size.
fn evaluate(subject: &Subject<'_>, rows: &[Row]) -> Result<Evaluation, CoverageError> {
    let rows_per_chunk = (CLANG_BATCH / subject.programs_per_row()).max(1);
    let chunks = rows.chunks(rows_per_chunk).collect::<Vec<_>>();
    let mut evaluation = Evaluation {
        reached: Vec::with_capacity(rows.len()),
        universe: BTreeSet::new(),
        rejecting: BTreeSet::new(),
    };
    for chunk in in_parallel(&chunks, |chunk| gated(|| evaluate_chunk(subject, chunk))) {
        let chunk = chunk?;
        evaluation.reached.extend(chunk.reached);
        evaluation.universe.extend(chunk.universe);
        evaluation.rejecting.extend(chunk.rejecting);
    }
    Ok(evaluation)
}

fn evaluate_chunk(subject: &Subject<'_>, rows: &[Row]) -> Result<Evaluation, CoverageError> {
    let skeletons = rows
        .iter()
        .map(|row| subject.skeletons(row))
        .collect::<Vec<_>>();
    let vulkan = rows
        .first()
        .is_some_and(|row| matches!(row.configuration, Configuration::Vulkan(_)));
    let outcomes: Vec<Vec<Outcome>> = if vulkan {
        let kernels = subject.kernels();
        skeletons
            .iter()
            .map(|programs| {
                programs
                    .iter()
                    .map(|skeleton| preprocess_vulkan(skeleton, &kernels))
                    .collect::<Result<Vec<_>, _>>()
            })
            .collect::<Result<_, _>>()?
    } else {
        // CUDA and Metal: clang over every program of every row, in batches.
        let texts = skeletons
            .iter()
            .flat_map(|programs| programs.iter().map(|skeleton| skeleton.text.as_str()))
            .collect::<Vec<_>>();
        for text in &texts {
            let unknown = predefined(text)
                .into_iter()
                .filter(|name| !seismic_cuda::nvrtc::ARCHITECTURE_MACROS.contains(&name.as_str()))
                .collect::<Vec<_>>();
            if !unknown.is_empty() {
                return Err(CoverageError::Skeleton(format!(
                    "conditions read toolchain-predefined macros {unknown:?}, which skeleton preprocessing does not define"
                )));
            }
        }
        let mut flat = texts
            .chunks(CLANG_BATCH)
            .map(preprocess_clang)
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .flatten();
        skeletons
            .iter()
            .map(|programs| {
                programs
                    .iter()
                    .map(|_| flat.next().expect("one outcome per skeleton"))
                    .collect()
            })
            .collect()
    };
    let mut evaluation = Evaluation {
        reached: Vec::with_capacity(rows.len()),
        universe: BTreeSet::new(),
        rejecting: BTreeSet::new(),
    };
    for (programs, outcomes) in skeletons.iter().zip(outcomes) {
        let mut reached = Some(BTreeSet::new());
        for (skeleton, outcome) in programs.iter().zip(outcomes) {
            let markers = match outcome {
                Outcome::Rejected => {
                    reached = None;
                    BTreeSet::new()
                }
                Outcome::Reached(markers) => markers,
            };
            for (marker, site) in skeleton.sites.iter().enumerate() {
                if skeleton.rejecting.contains(&marker) {
                    evaluation.rejecting.insert(site.clone());
                    continue;
                }
                evaluation.universe.insert(site.clone());
                if markers.contains(&marker) {
                    if let Some(reached) = &mut reached {
                        reached.insert(site.clone());
                    }
                }
            }
        }
        evaluation.reached.push(reached);
    }
    Ok(evaluation)
}

/// What forming one row established.
enum RowOutcome {
    /// The toolchain formed these programs.
    Formed(usize),
    /// The kernel rejects the configuration: a `static_assert` fired, which
    /// is how C++ kernels decline bindings the preprocessor cannot see.
    Rejected,
    Failed(String),
    /// This host's toolchain cannot form the configuration.
    Unformable,
}

/// Form `row`'s programs in order: the first that does not form decides.
fn form_row(subject: &Subject<'_>, forms: Forms<'_>, row: &Row) -> RowOutcome {
    if !forms.former.formable(row.configuration) {
        return RowOutcome::Unformable;
    }
    let programs = subject.programs(row);
    gated(|| {
        for program in &programs {
            match forms.form(row.configuration, program) {
                Verdict::Formed => {}
                Verdict::Rejected => return RowOutcome::Rejected,
                Verdict::Failed(error) => return RowOutcome::Failed(error),
            }
        }
        RowOutcome::Formed(programs.len())
    })
}

/// One entry's search: its kept rows, what forming them established, and
/// the groups they reach.
#[derive(Default)]
struct Search {
    formed: usize,
    /// The rows the toolchain formed.
    kept: Vec<Row>,
    /// Form values formed, of the entry's, and the ones that were not.
    forms_formed: usize,
    forms: usize,
    unformed: Vec<String>,
    failures: Vec<FormationFailure>,
    unformable: Vec<Configuration>,
    reached: BTreeSet<GroupSite>,
    universe: BTreeSet<GroupSite>,
    rejecting: BTreeSet<GroupSite>,
    untaken: Vec<Untaken>,
}

impl Search {
    /// Record a kept row's formation; `false` when the kernel rejects it.
    fn record(&mut self, subject: &Subject<'_>, row: &Row, formation: RowOutcome) -> bool {
        match formation {
            RowOutcome::Formed(programs) => {
                self.formed += programs;
                self.kept.push(row.clone());
            }
            RowOutcome::Rejected => return false,
            RowOutcome::Failed(error) => self.failures.push(subject.describe(row, error)),
            RowOutcome::Unformable => self.unformable.push(row.configuration),
        }
        true
    }

    fn complete(&self) -> bool {
        self.universe
            .iter()
            .all(|site| self.reached.contains(site) || self.rejecting.contains(site))
    }
}

fn search(subject: &Subject<'_>, forms: Forms<'_>) -> Result<Option<Search>, CoverageError> {
    let implementation = subject.implementation;
    let Some(statics) = subject.statics(&[]) else {
        return Ok(None);
    };
    let specialization = implementation
        .default_specialization(&statics)
        .expect("searched statics admit a configuration");
    let mut search = Search::default();
    // The base: the first binding in candidate order that is admitted,
    // implemented and not rejected by the toolchain.
    let base = base_row(
        subject,
        Row {
            bindings: ElementBindings::new(),
            specialization,
            configuration: forms.former.base(),
        },
        |row, mut evaluated| {
            search
                .universe
                .extend(std::mem::take(&mut evaluated.universe));
            search
                .rejecting
                .extend(std::mem::take(&mut evaluated.rejecting));
            match evaluated.reached.pop().expect("one row evaluated") {
                Some(reached) if search.record(subject, row, form_row(subject, forms, row)) => {
                    search.reached.extend(reached);
                    true
                }
                _ => false,
            }
        },
    )?;
    let Some(base) = base else {
        return Ok(None);
    };
    search.untaken = untaken_at(subject, forms.former.resident_layout(), &base)?;
    let mut seen = BTreeSet::from([base.key()]);
    let mut frontier = vec![base.clone()];
    let mut paired = false;
    loop {
        // Change one choice at a time from every newly kept configuration.
        while !frontier.is_empty() {
            let candidates = variations(subject, forms.former, &frontier)
                .into_iter()
                .filter(|candidate| seen.insert(candidate.key()))
                .collect();
            frontier = keep(subject, forms, candidates, &mut search)?;
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
        frontier = keep(subject, forms, candidates, &mut search)?;
    }
    form_values(subject, forms, &base, &mut search)?;
    let rejecting = std::mem::take(&mut search.rejecting);
    search.universe.retain(|site| !rejecting.contains(site));
    Ok(Some(search))
}

/// The first binding in candidate order that is admitted and that `accept`
/// takes, given its evaluation, as a row like `template`.
fn base_row(
    subject: &Subject<'_>,
    template: Row,
    mut accept: impl FnMut(&Row, Evaluation) -> bool,
) -> Result<Option<Row>, CoverageError> {
    if subject
        .candidates
        .iter()
        .any(|(_, candidates)| candidates.is_empty())
    {
        return Ok(None);
    }
    let mut choice = vec![0usize; subject.candidates.len()];
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
                ..template.clone()
            };
            let evaluated = evaluate(subject, std::slice::from_ref(&row))?;
            if accept(&row, evaluated) {
                return Ok(Some(row));
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
            return Ok(None);
        }
    }
}

/// The representations of `layout` that a weight slot of `base`'s entry is
/// not implemented at while it is at another of them, each with only that
/// slot of `base` changed. Only preprocessing decides; nothing is formed.
fn untaken_at(
    subject: &Subject<'_>,
    layout: registry::Layout,
    base: &Row,
) -> Result<Vec<Untaken>, CoverageError> {
    let domain = &subject.info.element_domain;
    let mut untaken = Vec::new();
    for (parameter, candidates) in &subject.candidates {
        if !weight_slot(domain, parameter) {
            continue;
        }
        let rows = candidates
            .iter()
            .filter(|representation| {
                registry::representation_info(**representation).layout == layout
            })
            .map(|representation| Row {
                bindings: subject.rebind(&base.bindings, parameter, *representation),
                ..base.clone()
            })
            .filter(|row| subject.admits(&row.bindings))
            .collect::<Vec<_>>();
        let reached = evaluate(subject, &rows)?.reached;
        if reached.iter().all(Option::is_none) {
            continue;
        }
        for (row, _) in rows
            .iter()
            .zip(reached)
            .filter(|(_, reached)| reached.is_none())
        {
            let representation = row.bindings.get(parameter).expect("the slot is bound");
            untaken.push(Untaken {
                entry: subject.info.name.clone(),
                parameter: parameter.clone(),
                representation: registry::representation_info(representation).name,
            });
        }
    }
    Ok(untaken)
}

/// Whether `parameter` is a weight slot: the entry reads its elements, and no
/// conversion's source chooses it (`repack[U = U]`).
fn weight_slot(domain: &ElementDomain, parameter: &str) -> bool {
    let read = domain.parameters().iter().any(|element| {
        element.name == parameter && (element.uses.decoded_read || element.uses.component_read)
    });
    let converted = domain.conversions().iter().any(|conversion| {
        matches!(&conversion.target, ElementTarget::Parameter(target) if target == parameter)
    });
    read && !converted
}

/// Every resident representation a weight slot of a native implementation
/// of `module` does not take while it takes another, on `toolchain`'s
/// backend: the check [`cover`] makes, by preprocessing alone. Each entry's
/// base is its first binding in candidate order that is admitted and
/// implemented.
pub fn untaken(
    module: &CheckedModule,
    toolchain: &Toolchain,
) -> Result<Vec<Untaken>, CoverageError> {
    let former = &toolchain.former;
    let subjects = subjects(module, former.backend());
    let found = in_parallel(&subjects, |subject| {
        let Some(statics) = subject.statics(&[]) else {
            return Ok(Vec::new());
        };
        let specialization = subject
            .implementation
            .default_specialization(&statics)
            .expect("searched statics admit a configuration");
        let base = base_row(
            subject,
            Row {
                bindings: ElementBindings::new(),
                specialization,
                configuration: former.base(),
            },
            |_, mut evaluated| {
                evaluated
                    .reached
                    .pop()
                    .expect("one row evaluated")
                    .is_some()
            },
        )?;
        match base {
            Some(base) => untaken_at(subject, former.resident_layout(), &base),
            None => Ok(Vec::new()),
        }
    });
    let mut untaken = Vec::new();
    for found in found {
        untaken.extend(found?);
    }
    Ok(untaken)
}

/// One value of a `form` tuning parameter: the launch that declares it (none
/// for the entry's own), its name and the value.
type Form<'a> = (Option<usize>, &'a str, u64);

fn holds(specialization: &NativeSpecialization, (launch, name, value): Form<'_>) -> bool {
    match launch {
        None => specialization.param(name) == Some(value),
        Some(launch) => specialization.launch_param(launch, name) == Some(value),
    }
}

/// Form every value of the entry's `form` parameters under each device
/// configuration, at `base`'s bindings: nothing more where a kept row holds
/// the value under that configuration; otherwise a row of a formed
/// specialization that holds it, or of the first one the declaration admits
/// with it ([`NativeImplementation::search_form`]). One row per value and
/// device configuration, never the product with the other parameters.
fn form_values(
    subject: &Subject<'_>,
    forms: Forms<'_>,
    base: &Row,
    search: &mut Search,
) -> Result<(), CoverageError> {
    let implementation = subject.implementation;
    let values: Vec<Form<'_>> = implementation
        .params
        .iter()
        .map(|parameter| (None, parameter))
        .chain(
            implementation
                .launches
                .iter()
                .enumerate()
                .flat_map(|(launch, declaration)| {
                    declaration
                        .params
                        .iter()
                        .map(move |parameter| (Some(launch), parameter))
                }),
        )
        .filter(|(_, parameter)| parameter.form)
        .flat_map(|(launch, parameter)| {
            parameter
                .values
                .iter()
                .map(move |value| (launch, parameter.name.as_str(), *value))
        })
        .collect();
    let mut configurations = vec![base.configuration];
    configurations.extend(forms.former.variations(base.configuration));
    // Whether each form is settled (formed, or beyond this host's toolchain)
    // and whether a row of it failed, which is reported as that failure.
    let mut settled = vec![false; values.len()];
    let mut failed = vec![false; values.len()];
    let mut rows: Vec<(usize, Row)> = Vec::new();
    for (index, form) in values.iter().enumerate() {
        let specialization = search
            .kept
            .iter()
            .chain(rows.iter().map(|(_, row)| row))
            .map(|row| &row.specialization)
            .find(|specialization| holds(specialization, *form))
            .cloned()
            .or_else(|| implementation.search_form(form.0, form.1, form.2));
        let Some(specialization) = specialization else {
            continue;
        };
        for configuration in &configurations {
            let formed = search.kept.iter().any(|row| {
                row.configuration == *configuration && holds(&row.specialization, *form)
            });
            if formed {
                settled[index] = true;
            } else {
                rows.push((
                    index,
                    Row {
                        bindings: base.bindings.clone(),
                        specialization: specialization.clone(),
                        configuration: *configuration,
                    },
                ));
            }
        }
    }
    let candidates = rows.iter().map(|(_, row)| row.clone()).collect::<Vec<_>>();
    let mut implemented = Vec::new();
    let evaluated = evaluate(subject, &candidates)?;
    search.universe.extend(evaluated.universe);
    search.rejecting.extend(evaluated.rejecting);
    for ((index, row), reached) in rows.into_iter().zip(evaluated.reached) {
        if let Some(reached) = reached {
            implemented.push((index, row, reached));
        }
    }
    let formations = in_parallel(&implemented, |(_, row, _)| form_row(subject, forms, row));
    for ((index, row, reached), formation) in implemented.into_iter().zip(formations) {
        match formation {
            RowOutcome::Failed(_) => failed[index] = true,
            RowOutcome::Formed(_) | RowOutcome::Unformable => settled[index] = true,
            RowOutcome::Rejected => {}
        }
        if search.record(subject, &row, formation) {
            search.reached.extend(reached);
        }
    }
    search.forms = values.len();
    search.forms_formed = settled.iter().filter(|settled| **settled).count();
    search.unformed = values
        .iter()
        .zip(settled.iter().zip(&failed))
        .filter(|(_, (settled, failed))| !**settled && !**failed)
        .map(|((launch, name, value), _)| match launch {
            None => format!("`{}` {name} = {value}", subject.info.name),
            Some(launch) => format!(
                "`{}` launch `{}` {name} = {value}",
                subject.info.name, implementation.launches[*launch].kernel
            ),
        })
        .collect();
    Ok(())
}

/// Evaluate `candidates` and keep each that reaches a group no kept row
/// reached and that the toolchain does not reject: candidates are chosen in
/// order and formed in parallel, and the groups of a rejected one are sought
/// again among the rest. The kept candidates are returned.
fn keep(
    subject: &Subject<'_>,
    forms: Forms<'_>,
    candidates: Vec<Row>,
    search: &mut Search,
) -> Result<Vec<Row>, CoverageError> {
    let evaluated = evaluate(subject, &candidates)?;
    search.universe.extend(evaluated.universe);
    search.rejecting.extend(evaluated.rejecting);
    let mut open = Vec::new();
    for (candidate, reached) in candidates.into_iter().zip(evaluated.reached) {
        if let Some(reached) = reached {
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
        let formations = in_parallel(&chosen, |(candidate, _)| {
            form_row(subject, forms, candidate)
        });
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
                        bindings: subject.rebind(
                            &subject.rebind(&row.bindings, parameter, *representation),
                            other,
                            *other_representation,
                        ),
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

/// Form every native implementation of `module` with `toolchain`, through
/// `forming`. Entries are searched in parallel, each search preprocessing
/// and forming its candidates in parallel; the toolchain work itself runs at
/// most one job per core. `progress` counts the entries searched.
pub fn cover(
    module: &CheckedModule,
    toolchain: &Toolchain,
    forming: &dyn Forming,
    progress: &Progress,
) -> Result<Coverage, CoverageError> {
    let forms = Forms {
        former: &toolchain.former,
        forming,
    };
    let subjects = subjects(module, toolchain.backend());
    progress.begin(subjects.len());
    let searches = in_parallel(&subjects, |subject| {
        let searched = search(subject, forms);
        progress.advance();
        searched
    });
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
        coverage.forms_formed += searched.forms_formed;
        coverage.forms += searched.forms;
        coverage.unformed.extend(searched.unformed);
        coverage.untaken.extend(searched.untaken);
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

/// Every entry of `module` with a native implementation on `backend`, each
/// element parameter's candidates every registry representation.
fn subjects(module: &CheckedModule, backend: BackendName) -> Vec<Subject<'_>> {
    module
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
        .collect()
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
            scope.spawn(|| {
                loop {
                    let index = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    let Some(item) = items.get(index) else {
                        return;
                    };
                    let result = work(item);
                    results
                        .lock()
                        .expect("no worker panics holding the lock")
                        .push((index, result));
                }
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
            .wait_while(
                free.lock().expect("no thread panics holding the permits"),
                |free| *free == 0,
            )
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
            Self::Inadmissible(reason) => {
                write!(f, "statics outside the kernel's domain: {reason}")
            }
            Self::Unimplemented => f.write_str("the kernel rejects the binding"),
            Self::Formation(error) => write!(f, "formation failed: {error}"),
        }
    }
}

/// Form every request as preparation forms it on `toolchain`'s base
/// configuration, at its default specialization, in parallel through
/// `forming`; results in request order. `progress` counts the requests
/// formed.
pub fn form_requests(
    module: &CheckedModule,
    toolchain: &Toolchain,
    forming: &dyn Forming,
    requests: &[Request],
    progress: &Progress,
) -> Vec<Result<(), RequestFailure>> {
    let forms = Forms {
        former: &toolchain.former,
        forming,
    };
    progress.begin(requests.len());
    in_parallel(requests, |request| {
        let formed = form_request(module, forms, request);
        progress.advance();
        formed
    })
}

fn form_request(
    module: &CheckedModule,
    forms: Forms<'_>,
    request: &Request,
) -> Result<(), RequestFailure> {
    let info = module
        .entries()
        .iter()
        .find(|info| info.id == request.entry)
        .expect("a request names an entry of the module");
    let subject = Subject::new(module, info, forms.former.backend(), Vec::new())
        .expect("a request names an entry with a native implementation");
    let specialization = subject
        .implementation
        .default_specialization(&request.statics)
        .map_err(|error| RequestFailure::Inadmissible(error.to_string()))?;
    let row = Row {
        bindings: request.bindings.clone(),
        specialization,
        configuration: forms.former.base(),
    };
    let reached = evaluate(&subject, std::slice::from_ref(&row))
        .map_err(|error| RequestFailure::Formation(error.to_string()))?
        .reached
        .pop()
        .expect("one row evaluated");
    if reached.is_none() {
        return Err(RequestFailure::Unimplemented);
    }
    match form_row(&subject, forms, &row) {
        RowOutcome::Formed(_) | RowOutcome::Unformable => Ok(()),
        RowOutcome::Rejected => Err(RequestFailure::Unimplemented),
        RowOutcome::Failed(error) => Err(RequestFailure::Formation(error)),
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
                GroupSite {
                    file: "k.comp".into(),
                    line: 1
                },
                GroupSite {
                    file: "lib/x.glsl".into(),
                    line: 1
                },
                GroupSite {
                    file: "k.comp".into(),
                    line: 3
                },
            ]
        );
        assert_eq!(skeleton.rejecting, BTreeSet::from([1]));
        assert!(
            skeleton
                .text
                .starts_with("#version 460\n#define A 1\n#if Q\n#endif\n#if A\n")
        );
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
