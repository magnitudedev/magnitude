//! The opaque checked module (spec §3.2).
//!
//! Exactly two constructors exist, one per kind of input:
//!
//! - [`check_source`] parses and checks source text: files, the dynamic
//!   API's inline source, development overrides, and the source section of a
//!   bundle file that arrives from outside the binary
//!   ([`crate::bundle::check_bundle_sources`]).
//! - [`crate::bundle::decode_checked_bundle`] decodes the checked module a
//!   bundle embedded by `seismic-build` carries: the output of the build's
//!   checker, validated structurally and never re-checked.
//!
//! There is no struct literal, `Default`, deserializer outside the bundle
//! decoder's context, arena mutation, or constructor that accepts
//! already-typed nodes. Consumers read through accessors and obtain a
//! [`LogicalEntry`] through [`CheckedModule::entry`], the only builder of
//! entry semantics.
//!
//! W1 owns the internals behind `internals::Module`.

use crate::entry::{ElementBindings, LogicalEntry};
use crate::ids::{EntryId, ModuleHash, RepresentationId, StableEntryId};
use crate::registry::{self, BackendName, RepresentationAccess, RepresentationKind};
use crate::span::Span;
use crate::types::DType;
use serde::{Deserialize, Serialize};

/// One source file. Paths are diagnostic labels; they grant nothing.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct SourceFile {
    pub path: String,
    pub text: String,
}

/// The closed set of sources checked as one module.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct SourceSet {
    files: Vec<SourceFile>,
}

impl SourceSet {
    pub fn new(files: Vec<SourceFile>) -> Self {
        Self { files }
    }

    pub fn push(&mut self, file: SourceFile) {
        self.files.push(file);
    }

    pub fn extend(&mut self, files: impl IntoIterator<Item = SourceFile>) {
        self.files.extend(files);
    }

    pub fn files(&self) -> &[SourceFile] {
        &self.files
    }

    pub(crate) fn canonicalized(mut self) -> Result<Self, Diagnostics> {
        for file in &mut self.files {
            file.path = canonical_path(&file.path);
        }
        self.files.sort_by(|a, b| a.path.cmp(&b.path));
        let diagnostics = self
            .files
            .windows(2)
            .filter(|pair| pair[0].path == pair[1].path)
            .map(|pair| {
                SourceDiagnostic::new(
                    &pair[1],
                    Span::default(),
                    DiagnosticRule::Resolution,
                    "duplicate source path in one module",
                )
            })
            .collect();
        match Diagnostics::new(diagnostics) {
            Some(diagnostics) => Err(diagnostics),
            None => Ok(self),
        }
    }
}

fn canonical_path(path: &str) -> String {
    let normalized = path.replace('\\', "/");
    let absolute = normalized.starts_with('/');
    let mut components: Vec<&str> = Vec::new();
    for component in normalized.split('/') {
        match component {
            "" | "." => {}
            ".." => {
                if components.last().is_some_and(|last| *last != "..") {
                    components.pop();
                } else if !absolute {
                    components.push("..");
                }
            }
            value => components.push(value),
        }
    }
    let body = components.join("/");
    if absolute {
        format!("/{body}")
    } else {
        body
    }
}

/// The language rule a diagnostic reports.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum DiagnosticRule {
    Syntax,
    Resolution,
    Type,
    Ownership,
    Initialization,
    Independence,
    CallContract,
    Dimension,
    Capability,
    Placement,
    Atomic,
    Recursion,
    NativeDeclaration,
}

impl DiagnosticRule {
    pub fn name(self) -> &'static str {
        match self {
            Self::Syntax => "Syntax",
            Self::Resolution => "Resolution",
            Self::Type => "Type",
            Self::Ownership => "Ownership",
            Self::Initialization => "Initialization",
            Self::Independence => "Independence",
            Self::CallContract => "CallContract",
            Self::Dimension => "Dimension",
            Self::Capability => "Capability",
            Self::Placement => "Placement",
            Self::Atomic => "Atomic",
            Self::Recursion => "Recursion",
            Self::NativeDeclaration => "NativeDeclaration",
        }
    }
}

impl std::fmt::Display for DiagnosticRule {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

/// One-based line and column (in characters) of a span's start in `text`.
pub(crate) fn line_column(text: &str, span: Span) -> (u32, u32) {
    let offset = (span.start as usize).min(text.len());
    let before = &text[..offset];
    let line_start = before.rfind('\n').map_or(0, |newline| newline + 1);
    let line = before.matches('\n').count() + 1;
    let column = before[line_start..].chars().count() + 1;
    (
        u32::try_from(line).expect("source has more than u32::MAX lines"),
        u32::try_from(column).expect("source line has more than u32::MAX characters"),
    )
}

/// Where a diagnostic is anchored: a span of one source file, with its
/// one-based line and column and the text of that line.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct SourceLocation {
    pub path: String,
    pub span: Span,
    pub line: u32,
    pub column: u32,
    source_line: String,
}

impl SourceLocation {
    fn new(file: &SourceFile, span: Span) -> Self {
        let (line, column) = line_column(&file.text, span);
        let source_line = file
            .text
            .lines()
            .nth(line as usize - 1)
            .unwrap_or("")
            .to_owned();
        Self {
            path: file.path.clone(),
            span,
            line,
            column,
            source_line,
        }
    }

    /// The source line containing the span's start.
    pub fn source_line(&self) -> &str {
        &self.source_line
    }
}

/// One diagnostic anchored in source.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct SourceDiagnostic {
    pub location: SourceLocation,
    pub rule: DiagnosticRule,
    pub message: String,
}

impl SourceDiagnostic {
    pub(crate) fn new(
        file: &SourceFile,
        span: Span,
        rule: DiagnosticRule,
        message: impl Into<String>,
    ) -> Self {
        Self {
            location: SourceLocation::new(file, span),
            rule,
            message: message.into(),
        }
    }

    pub(crate) fn located(file: &SourceFile, diagnostic: crate::span::Diagnostic) -> Self {
        Self::new(file, diagnostic.span, diagnostic.rule, diagnostic.message)
    }
}

impl std::fmt::Display for SourceDiagnostic {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let location = &self.location;
        write!(
            f,
            "{}:{}:{}: {}: {}",
            location.path, location.line, location.column, self.rule, self.message
        )?;
        let line = location.source_line();
        let before = location.column as usize - 1;
        let span_bytes = location.span.end.saturating_sub(location.span.start) as usize;
        let start = line
            .char_indices()
            .nth(before)
            .map_or(line.len(), |(offset, _)| offset);
        let width = line[start..]
            .char_indices()
            .take_while(|(offset, _)| *offset < span_bytes)
            .count()
            .max(1);
        write!(
            f,
            "\n  {line}\n  {}{}",
            " ".repeat(before),
            "^".repeat(width)
        )
    }
}

/// A non-empty, ordered, deduplicated set of diagnostics.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Diagnostics {
    items: Vec<SourceDiagnostic>,
}

impl Diagnostics {
    /// Orders the items by path, span, rule and message, and drops repeated
    /// items. Sorting on the whole key makes equal items adjacent.
    pub(crate) fn new(mut items: Vec<SourceDiagnostic>) -> Option<Self> {
        items.sort_by(|left, right| {
            (
                &left.location.path,
                left.location.span.start,
                left.location.span.end,
                left.rule,
                &left.message,
            )
                .cmp(&(
                    &right.location.path,
                    right.location.span.start,
                    right.location.span.end,
                    right.rule,
                    &right.message,
                ))
        });
        items.dedup();
        (!items.is_empty()).then_some(Self { items })
    }

    pub fn items(&self) -> &[SourceDiagnostic] {
        &self.items
    }

    pub(crate) fn single(item: SourceDiagnostic) -> Self {
        Self { items: vec![item] }
    }
}

/// A rejected source set: every diagnostic the checker produced.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SourceError {
    diagnostics: Diagnostics,
}

impl SourceError {
    pub(crate) fn new(diagnostics: Diagnostics) -> Self {
        Self { diagnostics }
    }

    pub fn diagnostics(&self) -> &Diagnostics {
        &self.diagnostics
    }
}

impl std::fmt::Display for SourceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for (ordinal, item) in self.diagnostics.items().iter().enumerate() {
            if ordinal > 0 {
                f.write_str("\n")?;
            }
            write!(f, "{item}")?;
        }
        Ok(())
    }
}

impl std::error::Error for SourceError {}

/// Parses and checks a source set as one closed module. The only source-side
/// constructor of a [`CheckedModule`].
pub fn check_source(sources: SourceSet) -> Result<CheckedModule, SourceError> {
    let sources = sources.canonicalized().map_err(SourceError::new)?;
    internals::check(sources).map(|inner| CheckedModule {
        inner,
        assets: Default::default(),
    })
}

/// An opaque checked semantic object: source semantics, types, effects,
/// canonical bodies, lowering declarations, capability requirements, and
/// stable identities. It decides nothing about targets, schedules,
/// allocations, or native code (§2.1).
#[derive(Debug)]
pub struct CheckedModule {
    inner: internals::Module,
    pub(crate) assets: std::collections::BTreeMap<(EntryId, BackendName), String>,
}

impl CheckedModule {
    /// Content-derived semantic hash; the cache identity of this module.
    pub fn semantic_hash(&self) -> ModuleHash {
        self.inner.semantic_hash()
    }

    /// Every exported entry, in declaration order.
    pub fn entries(&self) -> &[EntryInfo] {
        self.inner.entries()
    }

    pub fn entry_named(&self, name: &str) -> Option<EntryId> {
        self.inner
            .entries()
            .iter()
            .find(|e| e.name == name)
            .map(|e| e.id)
    }

    /// The explicitly authored top-level native implementation for an entry
    /// and backend, when one exists.
    pub fn native_implementation(
        &self,
        entry: EntryId,
        backend: BackendName,
    ) -> Option<&NativeImplementation> {
        self.inner
            .native_implementations
            .iter()
            .find(|native| native.entry == entry && native.backend == backend)
    }

    /// Builds the monomorphized semantics of one entry under one set of
    /// compile-time element bindings. This is the only constructor of
    /// `LogicalEntry` (§3.4). An entry with no element parameters takes
    /// `ElementBindings::default()`.
    pub fn entry(
        &self,
        entry: EntryId,
        bindings: &ElementBindings,
    ) -> Result<LogicalEntry, SourceError> {
        self.inner.entry(entry, bindings)
    }

    /// Snapshot the native asset of an entry's native implementation for one
    /// backend. The checked declaration remains authoritative.
    pub fn capture_native_asset(
        &mut self,
        entry: EntryId,
        backend: BackendName,
        source: String,
    ) -> Result<(), String> {
        if self.native_implementation(entry, backend).is_none() {
            return Err(format!(
                "entry `{}` has no native implementation for `{}`",
                self.entries()[entry.index()].name,
                backend.as_str()
            ));
        }
        self.assets.insert((entry, backend), source);
        Ok(())
    }

    pub fn native_asset(&self, entry: EntryId, backend: BackendName) -> Option<&str> {
        self.assets.get(&(entry, backend)).map(String::as_str)
    }

    /// The source set this module was checked from, for diagnostics and
    /// build-script fingerprinting only.
    pub fn sources(&self) -> &SourceSet {
        self.inner.sources()
    }

    pub(crate) fn internal(&self) -> &internals::Module {
        &self.inner
    }

    /// The bundle decoder's constructor: a module the build's checker
    /// produced, decoded and structurally validated by
    /// [`crate::bundle::decode_checked_bundle`].
    pub(crate) fn decoded(inner: internals::Module) -> Self {
        Self {
            inner,
            assets: Default::default(),
        }
    }
}

/// Summary of one entry sufficient for binding generation before
/// monomorphization.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct EntryInfo {
    pub id: EntryId,
    pub stable: StableEntryId,
    pub name: String,
    /// Runtime-inferred shape dimensions, in contract order.
    pub dimensions: Vec<String>,
    /// Compile-time element parameters (`T`, `U`) an entry is polymorphic in.
    pub element_parameters: Vec<String>,
    /// Complete source structure, including unit values, for dynamic callers.
    pub parameter_types: Vec<(String, SignatureType)>,
    pub result_type: SignatureType,
    pub parameters: Vec<ParameterSummary>,
    pub results: Vec<ResultSummary>,
    /// The admissible bindings of the element parameters.
    pub element_domain: ElementDomain,
}

/// The admissible bindings of an entry's element parameters, computed by the
/// checker. The only owner of binding legality.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ElementDomain {
    parameters: Vec<ElementParameter>,
    conversions: Vec<ElementConversion>,
}

/// One element parameter and every use the entry makes of it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ElementParameter {
    pub name: String,
    pub uses: ElementUses,
}

/// How an entry uses the elements of one element parameter.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ElementUses {
    pub decoded_read: bool,
    pub stored: bool,
    pub conversion_source: bool,
    /// `to_owned` of a view selecting part of the packing axis.
    pub partial_copy: bool,
}

/// `repack[U = target](t)` with `t: tensor[..] source`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ElementConversion {
    pub source: String,
    pub target: ElementTarget,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ElementTarget {
    Parameter(String),
    Concrete(RepresentationId),
}

/// Element bindings outside an entry's [`ElementDomain`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ElementBindingError {
    Missing {
        parameter: String,
    },
    Unexpected {
        parameter: String,
    },
    Inadmissible {
        parameter: String,
        representation: RepresentationId,
        uses: ElementUses,
    },
    NoConversion {
        source: RepresentationId,
        target: RepresentationId,
    },
}

impl std::fmt::Display for ElementBindingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name =
            |representation: &RepresentationId| registry::representation_info(*representation).name;
        match self {
            Self::Missing { parameter } => {
                write!(f, "element parameter `{parameter}` is not bound")
            }
            Self::Unexpected { parameter } => {
                write!(f, "`{parameter}` is not an element parameter of this entry")
            }
            Self::Inadmissible {
                parameter,
                representation,
                uses,
            } => write!(
                f,
                "representation `{}` is not admissible for element parameter `{parameter}` ({uses:?})",
                name(representation)
            ),
            Self::NoConversion { source, target } => write!(
                f,
                "no exact representation conversion is registered from `{}` to `{}`",
                name(source),
                name(target)
            ),
        }
    }
}

impl std::error::Error for ElementBindingError {}

impl ElementUses {
    /// Whether an element parameter with these uses may be bound to `representation`.
    pub fn admits(self, representation: RepresentationId) -> bool {
        let info = registry::representation_info(representation);
        let dense_float = matches!(
            info.kind,
            RepresentationKind::Dense(DType::F32 | DType::F16 | DType::BF16)
        );
        let decodable = dense_float
            || (matches!(
                info.kind,
                RepresentationKind::Packed(_) | RepresentationKind::PackedRows(_)
            ) && registry::decode_recipe(representation, DType::F32).is_some());
        info.decoded.is_float()
            && (!self.stored || (info.access == RepresentationAccess::ReadWrite && dense_float))
            && (!self.decoded_read
                || (matches!(
                    info.access,
                    RepresentationAccess::ReadWrite | RepresentationAccess::ReadOnly
                ) && decodable))
            && (!self.partial_copy || matches!(info.kind, RepresentationKind::Dense(_)))
    }
}

impl ElementDomain {
    pub(crate) fn new(
        parameters: Vec<ElementParameter>,
        conversions: Vec<ElementConversion>,
    ) -> Self {
        Self {
            parameters,
            conversions,
        }
    }

    pub fn parameters(&self) -> &[ElementParameter] {
        &self.parameters
    }

    pub fn conversions(&self) -> &[ElementConversion] {
        &self.conversions
    }

    /// Decides whether `bindings` bind exactly this domain's parameters to
    /// admissible representations with every required conversion registered.
    pub fn admit(&self, bindings: &ElementBindings) -> Result<(), ElementBindingError> {
        for parameter in &self.parameters {
            if bindings.get(&parameter.name).is_none() {
                return Err(ElementBindingError::Missing {
                    parameter: parameter.name.clone(),
                });
            }
        }
        for (name, _) in bindings.iter() {
            if !self
                .parameters
                .iter()
                .any(|parameter| parameter.name == name)
            {
                return Err(ElementBindingError::Unexpected {
                    parameter: name.to_owned(),
                });
            }
        }
        let bound = |name: &str| {
            bindings
                .get(name)
                .expect("element domain names a parameter outside its own parameter list")
        };
        for parameter in &self.parameters {
            let representation = bound(&parameter.name);
            if !parameter.uses.admits(representation) {
                return Err(ElementBindingError::Inadmissible {
                    parameter: parameter.name.clone(),
                    representation,
                    uses: parameter.uses,
                });
            }
        }
        for conversion in &self.conversions {
            let source = bound(&conversion.source);
            let target = match &conversion.target {
                ElementTarget::Parameter(name) => bound(name),
                ElementTarget::Concrete(representation) => *representation,
            };
            if registry::representation_conversion(source, target).is_none() {
                return Err(ElementBindingError::NoConversion { source, target });
            }
        }
        Ok(())
    }
}

/// Read-only signature structure projected from checked source types.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SignatureType {
    Unit,
    Tuple(Vec<SignatureType>),
    Tensor {
        access: TensorAccess,
        rank: u32,
        element: ElementSummary,
    },
    Scalar(DType),
    Index,
    Range,
}

/// One direct top-level native implementation attached to a checked entry.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct NativeImplementation {
    pub entry: EntryId,
    pub backend: BackendName,
    /// Canonical module source label containing the declaration.
    pub declared_in: String,
    /// The native source path, relative to the declaring source file.
    pub source_path: String,
    /// Entry dimensions whose values are fixed when the implementation is
    /// prepared, in declaration order.
    pub statics: Vec<String>,
    /// Tuning parameters, in declaration order.
    pub params: Vec<NativeParameter>,
    /// On CPU, the dense element types the form is compiled for, for element
    /// parameters that do not take the default (`f32`, `bf16`, `f16`).
    pub elements: Vec<NativeElementCoverage>,
    /// The `where` condition restricting admissible parameter
    /// configurations. It reads only static dimensions and parameters.
    pub constraint: Option<NativeCondition>,
    /// Call-private scratch buffers, in ABI order.
    pub scratch: Vec<NativeScratch>,
    /// Ordered dispatches of one call. Never empty.
    pub launches: Vec<NativeLaunch>,
}

/// The dense element types a build-time compiled native form binds one
/// element parameter to.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct NativeElementCoverage {
    pub parameter: String,
    pub dtypes: Vec<DType>,
}

impl NativeElementCoverage {
    /// The coverage of an element parameter a CPU form stores when its
    /// declaration lists none: the floating element types.
    pub const DEFAULT: [DType; 3] = [DType::F32, DType::BF16, DType::F16];
}

/// A tuning parameter with its finite domain. `values[0]` is the default.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct NativeParameter {
    pub name: String,
    /// Its value changes generated kernel code and must be fixed when that
    /// launch's function is formed.
    pub code: bool,
    /// The parameter changes the arithmetic order of a row's result. Other
    /// parameters must produce bit-identical results across their values.
    pub arithmetic: bool,
    pub values: Vec<u64>,
    pub role: NativeParameterRole,
}

/// What a native tuning parameter chooses.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum NativeParameterRole {
    /// Declared in the `params` clause; the implementation's source and
    /// launch geometry read it.
    Declared,
    /// Seismic-owned, on CPU devices: how many pool participants claim the
    /// work items of launch `launch`. `0` is every participant.
    Workers { launch: u32 },
    /// Seismic-owned, on CPU devices: the instruction-set tier the
    /// implementation runs at. `0` is the device's detected tier; `t` is the
    /// tier of [`NativeParameterRole::TIERS`] ordinal `t - 1`.
    Tier,
}

impl NativeParameterRole {
    /// The tiers a `Tier` parameter value names, by ordinal from 1.
    pub const TIERS: [&'static str; 5] = ["x86v2", "x86v3", "x86v4", "x86v4vnni", "neon"];
}

impl NativeParameter {
    /// The Seismic-owned parameters a CPU device adds to an implementation
    /// with `launches` launches: one participant count per launch (every
    /// participant, then each power of two below `participants`), then the
    /// tier (the detected one, then each tier in `lower`, by
    /// [`NativeParameterRole::TIERS`] name). Their names contain `.`, which no
    /// declared parameter name can.
    pub fn cpu_parameters(
        launches: usize,
        participants: usize,
        lower: &[&str],
    ) -> Vec<NativeParameter> {
        let counts = std::iter::once(0)
            .chain(
                (0..)
                    .map(|power| 1u64 << power)
                    .take_while(|count| *count < participants as u64),
            )
            .collect::<Vec<_>>();
        let tiers = std::iter::once(0)
            .chain(lower.iter().map(|tier| {
                NativeParameterRole::TIERS
                    .iter()
                    .position(|known| known == tier)
                    .expect("a CPU tier is one of NativeParameterRole::TIERS")
                    as u64
                    + 1
            }))
            .collect();
        (0..launches)
            .map(|launch| NativeParameter {
                name: format!("cpu.workers.{launch}"),
                code: false,
                arithmetic: false,
                values: counts.clone(),
                role: NativeParameterRole::Workers {
                    launch: launch as u32,
                },
            })
            .chain(std::iter::once(NativeParameter {
                name: "cpu.tier".to_owned(),
                code: false,
                arithmetic: false,
                values: tiers,
                role: NativeParameterRole::Tier,
            }))
            .collect()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum NativeComparison {
    Lt,
    Le,
    Gt,
    Ge,
    Eq,
    Ne,
}

/// A boolean formula over native natural-number expressions: a native
/// `where` or `when` condition.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum NativeCondition {
    Compare {
        comparison: NativeComparison,
        left: NativeNatExpr,
        right: NativeNatExpr,
    },
    And(Box<Self>, Box<Self>),
    Or(Box<Self>, Box<Self>),
}

/// Call-private device memory of one native call.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct NativeScratch {
    pub name: String,
    pub bytes: NativeNatExpr,
    /// The buffer is sized by `bytes` only when this holds; otherwise it
    /// keeps its ABI slot at the minimum charge and `bytes` is not
    /// evaluated. `None` is always active.
    pub when: Option<NativeCondition>,
}

/// One ordered dispatch of a native call.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct NativeLaunch {
    /// Kernel function name in the native source.
    pub kernel: String,
    /// Parameters scoped to this launch. Another launch may reuse a name.
    pub params: Vec<NativeParameter>,
    /// Entry parameters used by this kernel beyond its launch expressions.
    pub reads: Vec<String>,
    /// The launch runs only when this holds; an inactive launch is not
    /// encoded, not limit-checked, and its geometry is not evaluated. It
    /// keeps its ordinal. `None` is always active.
    pub when: Option<NativeCondition>,
    /// Number of groups on each axis.
    pub groups: [NativeNatExpr; 3],
    /// Participants of one group on each axis.
    pub group_extent: [NativeNatExpr; 3],
    /// Dynamic group-shared memory in bytes.
    pub shared_bytes: NativeNatExpr,
}

/// Closed integer language used by native declarations.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum NativeNatExpr {
    Constant(u64),
    Dimension(String),
    Parameter(String),
    Add(Box<Self>, Box<Self>),
    Sub(Box<Self>, Box<Self>),
    Mul(Box<Self>, Box<Self>),
    Div(Box<Self>, Box<Self>),
    Rem(Box<Self>, Box<Self>),
    CeilDiv(Box<Self>, Box<Self>),
    Min(Box<Self>, Box<Self>),
    Max(Box<Self>, Box<Self>),
}

/// A native expression could not be evaluated.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum NativeEvalError {
    /// A name has no value in the evaluation environment.
    Unbound(String),
    /// Overflow, underflow, or division by zero.
    Arithmetic,
}

impl std::fmt::Display for NativeEvalError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unbound(name) => write!(f, "native expression name `{name}` has no value"),
            Self::Arithmetic => {
                f.write_str("native expression overflowed, underflowed, or divided by zero")
            }
        }
    }
}

impl NativeNatExpr {
    /// Evaluate with `dimension` and `parameter` supplying named values.
    pub fn evaluate(
        &self,
        dimension: &impl Fn(&str) -> Option<u64>,
        parameter: &impl Fn(&str) -> Option<u64>,
    ) -> Result<u64, NativeEvalError> {
        let binary = |left: &Self, right: &Self, operation: fn(u64, u64) -> Option<u64>| {
            let left = left.evaluate(dimension, parameter)?;
            let right = right.evaluate(dimension, parameter)?;
            operation(left, right).ok_or(NativeEvalError::Arithmetic)
        };
        match self {
            Self::Constant(value) => Ok(*value),
            Self::Dimension(name) => {
                dimension(name).ok_or_else(|| NativeEvalError::Unbound(name.clone()))
            }
            Self::Parameter(name) => {
                parameter(name).ok_or_else(|| NativeEvalError::Unbound(name.clone()))
            }
            Self::Add(left, right) => binary(left, right, u64::checked_add),
            Self::Sub(left, right) => binary(left, right, u64::checked_sub),
            Self::Mul(left, right) => binary(left, right, u64::checked_mul),
            Self::Div(left, right) => binary(left, right, u64::checked_div),
            Self::Rem(left, right) => binary(left, right, u64::checked_rem),
            Self::CeilDiv(left, right) => binary(left, right, |left, right| {
                if right == 0 {
                    None
                } else {
                    Some(left.div_ceil(right))
                }
            }),
            Self::Min(left, right) => binary(left, right, |left, right| Some(left.min(right))),
            Self::Max(left, right) => binary(left, right, |left, right| Some(left.max(right))),
        }
    }

    /// Every dimension name the expression reads.
    pub fn dimensions(&self, out: &mut Vec<String>) {
        match self {
            Self::Constant(_) | Self::Parameter(_) => {}
            Self::Dimension(name) => {
                if !out.contains(name) {
                    out.push(name.clone());
                }
            }
            Self::Add(left, right)
            | Self::Sub(left, right)
            | Self::Mul(left, right)
            | Self::Div(left, right)
            | Self::Rem(left, right)
            | Self::CeilDiv(left, right)
            | Self::Min(left, right)
            | Self::Max(left, right) => {
                left.dimensions(out);
                right.dimensions(out);
            }
        }
    }

    /// Every tuning parameter name the expression reads.
    pub fn parameters(&self, out: &mut Vec<String>) {
        match self {
            Self::Constant(_) | Self::Dimension(_) => {}
            Self::Parameter(name) => {
                if !out.contains(name) {
                    out.push(name.clone());
                }
            }
            Self::Add(left, right)
            | Self::Sub(left, right)
            | Self::Mul(left, right)
            | Self::Div(left, right)
            | Self::Rem(left, right)
            | Self::CeilDiv(left, right)
            | Self::Min(left, right)
            | Self::Max(left, right) => {
                left.parameters(out);
                right.parameters(out);
            }
        }
    }
}

impl NativeCondition {
    /// Evaluate with `dimension` and `parameter` supplying named values.
    /// `and` and `or` evaluate their right side only when the left side
    /// does not decide the result.
    pub fn holds(
        &self,
        dimension: &impl Fn(&str) -> Option<u64>,
        parameter: &impl Fn(&str) -> Option<u64>,
    ) -> Result<bool, NativeEvalError> {
        match self {
            Self::Compare {
                comparison,
                left,
                right,
            } => {
                let left = left.evaluate(dimension, parameter)?;
                let right = right.evaluate(dimension, parameter)?;
                Ok(match comparison {
                    NativeComparison::Lt => left < right,
                    NativeComparison::Le => left <= right,
                    NativeComparison::Gt => left > right,
                    NativeComparison::Ge => left >= right,
                    NativeComparison::Eq => left == right,
                    NativeComparison::Ne => left != right,
                })
            }
            Self::And(left, right) => {
                Ok(left.holds(dimension, parameter)? && right.holds(dimension, parameter)?)
            }
            Self::Or(left, right) => {
                Ok(left.holds(dimension, parameter)? || right.holds(dimension, parameter)?)
            }
        }
    }

    /// Every dimension name the condition reads.
    pub fn dimensions(&self, out: &mut Vec<String>) {
        match self {
            Self::Compare { left, right, .. } => {
                left.dimensions(out);
                right.dimensions(out);
            }
            Self::And(left, right) | Self::Or(left, right) => {
                left.dimensions(out);
                right.dimensions(out);
            }
        }
    }

    /// Every tuning parameter name the condition reads.
    pub fn parameters(&self, out: &mut Vec<String>) {
        match self {
            Self::Compare { left, right, .. } => {
                left.parameters(out);
                right.parameters(out);
            }
            Self::And(left, right) | Self::Or(left, right) => {
                left.parameters(out);
                right.parameters(out);
            }
        }
    }
}

impl NativeImplementation {
    /// Whether preparation must choose any declared tuning parameter, either
    /// for the entry as a whole or for an individual launch.
    pub fn has_tuning_parameters(&self) -> bool {
        !self.params.is_empty() || self.launches.iter().any(|launch| !launch.params.is_empty())
    }

    /// `specialization` with every Seismic-owned parameter it leaves unset at
    /// its default. Callers choose declared parameters; Seismic-owned ones
    /// are chosen by tuning or default.
    pub fn with_owned_defaults(
        &self,
        specialization: NativeSpecialization,
    ) -> NativeSpecialization {
        self.params
            .iter()
            .filter(|parameter| parameter.role != NativeParameterRole::Declared)
            .fold(specialization, |specialization, parameter| {
                if specialization.param(&parameter.name).is_some() {
                    specialization
                } else {
                    specialization.with_param(parameter.name.clone(), parameter.values[0])
                }
            })
    }

    /// The parameters of the `params` clause, in declaration order.
    pub fn declared_params(&self) -> impl Iterator<Item = &NativeParameter> {
        self.params
            .iter()
            .filter(|parameter| parameter.role == NativeParameterRole::Declared)
    }

    /// Every tuning parameter launch `launch` reads: those its declaration
    /// reads, and its Seismic-owned participant count.
    pub fn launch_parameters(&self, launch: usize) -> Vec<String> {
        let mut out = self.launches[launch].parameters();
        for parameter in &self.launches[launch].params {
            if !out.contains(&parameter.name) {
                out.push(parameter.name.clone());
            }
        }
        out.extend(
            self.params
                .iter()
                .filter(|parameter| {
                    parameter.role
                        == NativeParameterRole::Workers {
                            launch: launch as u32,
                        }
                })
                .map(|parameter| parameter.name.clone()),
        );
        out
    }

    /// A launch-local choice or an explicit kernel read needs per-launch
    /// source formation rather than entry-wide tuning macros.
    pub fn launch_scoped(&self) -> bool {
        self.launches
            .iter()
            .any(|launch| !launch.params.is_empty() || !launch.reads.is_empty())
    }
}

impl NativeLaunch {
    /// Every tuning parameter the launch's declaration reads: its `when`
    /// condition, groups, group extent and shared bytes.
    pub fn parameters(&self) -> Vec<String> {
        let mut out = self.reads.clone();
        if let Some(when) = &self.when {
            when.parameters(&mut out);
        }
        for expression in self.groups.iter().chain(&self.group_extent) {
            expression.parameters(&mut out);
        }
        self.shared_bytes.parameters(&mut out);
        out
    }
}

/// One value for every static dimension and tuning parameter of a native
/// implementation. It is the complete compile-time input of native formation
/// beyond element bindings.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct NativeSpecialization {
    statics: std::collections::BTreeMap<String, u64>,
    params: std::collections::BTreeMap<String, u64>,
    launch_params: std::collections::BTreeMap<(usize, String), u64>,
}

impl NativeSpecialization {
    pub fn new() -> Self {
        Self::default()
    }
    /// Fix a static dimension.
    pub fn with_static(mut self, name: impl Into<String>, value: u64) -> Self {
        self.statics.insert(name.into(), value);
        self
    }
    /// Choose a tuning parameter value.
    pub fn with_param(mut self, name: impl Into<String>, value: u64) -> Self {
        self.params.insert(name.into(), value);
        self
    }
    /// Choose a parameter in one launch's lexical scope.
    pub fn with_launch_param(mut self, launch: usize, name: impl Into<String>, value: u64) -> Self {
        self.launch_params.insert((launch, name.into()), value);
        self
    }
    pub fn statics(&self) -> &std::collections::BTreeMap<String, u64> {
        &self.statics
    }
    pub fn params(&self) -> &std::collections::BTreeMap<String, u64> {
        &self.params
    }
    pub fn launch_params(&self) -> &std::collections::BTreeMap<(usize, String), u64> {
        &self.launch_params
    }
    pub fn static_value(&self, name: &str) -> Option<u64> {
        self.statics.get(name).copied()
    }
    pub fn param(&self, name: &str) -> Option<u64> {
        self.params.get(name).copied()
    }
    pub fn launch_param(&self, launch: usize, name: &str) -> Option<u64> {
        self.launch_params.get(&(launch, name.to_owned())).copied()
    }
}

/// A specialization does not match its native implementation's declaration.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum NativeSpecializationError {
    MissingStatic(String),
    UnknownStatic(String),
    MissingParameter(String),
    UnknownParameter(String),
    OutsideDomain {
        parameter: String,
        value: u64,
    },
    MissingLaunchParameter {
        launch: usize,
        name: String,
    },
    UnknownLaunchParameter {
        launch: usize,
        name: String,
    },
    OutsideLaunchDomain {
        launch: usize,
        name: String,
        value: u64,
    },
    /// The configuration violates the `where` condition.
    Inadmissible,
    Evaluation(NativeEvalError),
}

impl std::fmt::Display for NativeSpecializationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingStatic(name) => write!(f, "static dimension `{name}` has no value"),
            Self::UnknownStatic(name) => write!(f, "`{name}` is not a static dimension"),
            Self::MissingParameter(name) => write!(f, "native parameter `{name}` has no value"),
            Self::UnknownParameter(name) => write!(f, "`{name}` is not a native parameter"),
            Self::OutsideDomain { parameter, value } => {
                write!(f, "native parameter `{parameter}` does not admit {value}")
            }
            Self::MissingLaunchParameter { launch, name } => {
                write!(f, "launch {launch} parameter `{name}` has no value")
            }
            Self::UnknownLaunchParameter { launch, name } => {
                write!(f, "`{name}` is not a parameter of launch {launch}")
            }
            Self::OutsideLaunchDomain {
                launch,
                name,
                value,
            } => {
                write!(
                    f,
                    "launch {launch} parameter `{name}` does not admit {value}"
                )
            }
            Self::Inadmissible => {
                f.write_str("configuration violates the native `where` condition")
            }
            Self::Evaluation(error) => write!(f, "{error}"),
        }
    }
}

impl std::error::Error for NativeSpecializationError {}

impl NativeImplementation {
    /// Check a complete specialization against the declaration.
    pub fn validate(
        &self,
        specialization: &NativeSpecialization,
    ) -> Result<(), NativeSpecializationError> {
        for name in &self.statics {
            if specialization.static_value(name).is_none() {
                return Err(NativeSpecializationError::MissingStatic(name.clone()));
            }
        }
        if let Some(name) = specialization
            .statics()
            .keys()
            .find(|name| !self.statics.contains(name))
        {
            return Err(NativeSpecializationError::UnknownStatic(name.clone()));
        }
        for parameter in &self.params {
            let value = specialization.param(&parameter.name).ok_or_else(|| {
                NativeSpecializationError::MissingParameter(parameter.name.clone())
            })?;
            if !parameter.values.contains(&value) {
                return Err(NativeSpecializationError::OutsideDomain {
                    parameter: parameter.name.clone(),
                    value,
                });
            }
        }
        if let Some(name) = specialization
            .params()
            .keys()
            .find(|name| !self.params.iter().any(|parameter| &parameter.name == *name))
        {
            return Err(NativeSpecializationError::UnknownParameter(name.clone()));
        }
        for (launch, declaration) in self.launches.iter().enumerate() {
            for parameter in &declaration.params {
                let value = specialization
                    .launch_param(launch, &parameter.name)
                    .ok_or_else(|| NativeSpecializationError::MissingLaunchParameter {
                        launch,
                        name: parameter.name.clone(),
                    })?;
                if !parameter.values.contains(&value) {
                    return Err(NativeSpecializationError::OutsideLaunchDomain {
                        launch,
                        name: parameter.name.clone(),
                        value,
                    });
                }
            }
        }
        if let Some(((launch, name), _)) =
            specialization
                .launch_params()
                .iter()
                .find(|((launch, name), _)| {
                    !self.launches.get(*launch).is_some_and(|declaration| {
                        declaration
                            .params
                            .iter()
                            .any(|parameter| &parameter.name == name)
                    })
                })
        {
            return Err(NativeSpecializationError::UnknownLaunchParameter {
                launch: *launch,
                name: name.clone(),
            });
        }
        let dimension = |name: &str| specialization.static_value(name);
        let parameter = |name: &str| {
            specialization.param(name).or_else(|| {
                let mut owners =
                    self.launches
                        .iter()
                        .enumerate()
                        .filter_map(|(launch, declaration)| {
                            declaration
                                .params
                                .iter()
                                .any(|parameter| parameter.name == name)
                                .then_some(launch)
                        });
                let launch = owners.next()?;
                owners
                    .next()
                    .is_none()
                    .then(|| specialization.launch_param(launch, name))
                    .flatten()
            })
        };
        if let Some(constraint) = &self.constraint {
            if !constraint
                .holds(&dimension, &parameter)
                .map_err(NativeSpecializationError::Evaluation)?
            {
                return Err(NativeSpecializationError::Inadmissible);
            }
        }
        Ok(())
    }

    /// Every admissible specialization for the given static values: the
    /// cartesian product of the parameter domains in declaration order,
    /// filtered by the `where` condition.
    pub fn admissible(
        &self,
        statics: &NativeSpecialization,
    ) -> Result<Vec<NativeSpecialization>, NativeSpecializationError> {
        let mut base = NativeSpecialization::new();
        for name in &self.statics {
            let value = statics
                .static_value(name)
                .ok_or_else(|| NativeSpecializationError::MissingStatic(name.clone()))?;
            base = base.with_static(name.clone(), value);
        }
        if let Some(name) = statics
            .statics()
            .keys()
            .find(|name| !self.statics.contains(name))
        {
            return Err(NativeSpecializationError::UnknownStatic(name.clone()));
        }
        let domains = self
            .params
            .iter()
            .map(|parameter| (None, parameter))
            .chain(
                self.launches
                    .iter()
                    .enumerate()
                    .flat_map(|(launch, declaration)| {
                        declaration
                            .params
                            .iter()
                            .map(move |parameter| (Some(launch), parameter))
                    }),
            )
            .collect::<Vec<_>>();
        // Walk the product of the parameter domains in place (the last
        // parameter varies fastest), keeping each admissible configuration.
        let mut configuration = base;
        for (launch, parameter) in &domains {
            configuration = match launch {
                None => configuration.with_param(parameter.name.clone(), parameter.values[0]),
                Some(launch) => configuration.with_launch_param(
                    *launch,
                    parameter.name.clone(),
                    parameter.values[0],
                ),
            };
        }
        let mut steps = vec![0usize; domains.len()];
        let mut admissible = Vec::new();
        loop {
            match self.validate(&configuration) {
                Ok(()) => admissible.push(configuration.clone()),
                Err(NativeSpecializationError::Inadmissible) => {}
                Err(error) => return Err(error),
            }
            let Some(position) = (0..domains.len())
                .rev()
                .find(|&position| steps[position] + 1 < domains[position].1.values.len())
            else {
                return Ok(admissible);
            };
            let next = steps[position] + 1;
            let mut set = |position: usize, step: usize| {
                steps[position] = step;
                let (launch, parameter) = domains[position];
                match launch {
                    None => {
                        *configuration
                            .params
                            .get_mut(&parameter.name)
                            .expect("every parameter was valued") = parameter.values[step];
                    }
                    Some(launch) => {
                        *configuration
                            .launch_params
                            .get_mut(&(launch, parameter.name.clone()))
                            .expect("every launch parameter was valued") = parameter.values[step];
                    }
                }
            };
            for reset in position + 1..domains.len() {
                set(reset, 0);
            }
            set(position, next);
        }
    }

    /// The configuration of declared defaults (`values[0]`) at the given
    /// static values.
    pub fn default_specialization(
        &self,
        statics: &NativeSpecialization,
    ) -> Result<NativeSpecialization, NativeSpecializationError> {
        let mut specialization = statics.clone();
        for parameter in &self.params {
            specialization = specialization.with_param(parameter.name.clone(), parameter.values[0]);
        }
        for (launch, declaration) in self.launches.iter().enumerate() {
            for parameter in &declaration.params {
                specialization = specialization.with_launch_param(
                    launch,
                    parameter.name.clone(),
                    parameter.values[0],
                );
            }
        }
        self.validate(&specialization)?;
        Ok(specialization)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ParameterSummary {
    /// Authored parameter ordinal and tuple path. The summary is leaf-flat;
    /// generated bindings group leaves by these canonical coordinates.
    pub source: u32,
    pub path: Vec<u32>,
    pub name: String,
    pub kind: ParameterSummaryKind,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ParameterSummaryKind {
    Tensor {
        access: TensorAccess,
        rank: u32,
        element: ElementSummary,
    },
    Scalar(DType),
    Index,
    Range,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TensorAccess {
    Owned,
    Shared,
    Mutable,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ElementSummary {
    /// A fixed dtype or packed representation, by registry name.
    Fixed(String),
    /// Bound at `for_device` time by an element parameter name.
    Parameter(String),
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResultSummary {
    /// Ordinal tuple path; empty for a non-tuple result.
    pub path: Vec<u32>,
    pub kind: ResultSummaryKind,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ResultSummaryKind {
    Tensor { rank: u32, element: ElementSummary },
    Scalar(DType),
    Index,
    Range,
}

pub(crate) mod internals {
    //! W1-owned. Must satisfy: private arenas, region-qualified node ids,
    //! typed registry ids, content-derived stable identities, and no
    //! constructor reachable from outside `seismic-lang` other than
    //! `check` and the bundle decoder. Every field other than the owners,
    //! the semantic hash and the sources is the bundle's checked section.

    use super::{Diagnostics, EntryInfo, NativeImplementation, SourceError, SourceSet};
    use crate::entry::{ElementBindings, LogicalEntry};
    use crate::ids::{EntryId, ModuleHash, ModuleId, ProgramId};

    #[derive(Debug)]
    pub(crate) struct Module {
        pub(crate) id: ModuleId,
        /// The owner of every `FunctionId` and `FamilyId` below.
        pub(crate) program: ProgramId,
        pub(crate) semantic_hash: ModuleHash,
        pub(crate) sources: SourceSet,
        pub(crate) entries: Vec<EntryInfo>,
        pub(crate) native_implementations: Vec<NativeImplementation>,
        pub(crate) entry_families: Vec<usize>,
        pub(crate) definitions: Vec<crate::check::ir::Definition>,
        pub(crate) families: Vec<crate::check::ir::Family>,
    }

    pub(crate) fn check(sources: SourceSet) -> Result<Module, SourceError> {
        let id = ModuleId::fresh();
        crate::check::check_closed(sources, id, ProgramId::fresh())
    }

    impl Module {
        pub(crate) fn semantic_hash(&self) -> ModuleHash {
            self.semantic_hash
        }
        pub(crate) fn entries(&self) -> &[EntryInfo] {
            &self.entries
        }
        pub(crate) fn entry(
            &self,
            entry: EntryId,
            bindings: &ElementBindings,
        ) -> Result<LogicalEntry, SourceError> {
            assert_eq!(
                entry.module(),
                self.id,
                "CheckedModule received an EntryId owned by another module (§13.3.2)"
            );
            assert!(
                entry.index() < self.entries.len(),
                "CheckedModule received an EntryId outside its entry arena (§13.3.2)"
            );
            crate::check::build_entry(self, entry, bindings)
                .map_err(|diagnostic| SourceError::new(Diagnostics::single(diagnostic)))
        }
        pub(crate) fn sources(&self) -> &SourceSet {
            &self.sources
        }
    }
}

#[cfg(test)]
mod native_tests {
    use super::*;

    fn source(native: &str) -> SourceSet {
        let mut sources = SourceSet::default();
        sources.push(SourceFile {
            path: "ops.seismic".to_owned(),
            text: format!(
                "fn scale[N](x: &tensor[N] f32, factor: f32, output: &mut tensor[N] f32):\n    parallel for i in 0..N:\n        output[i] = x[i] * factor\n\n{native}"
            ),
        });
        sources
    }

    #[test]
    fn observed_composed_products_survive_entry_monomorphization() {
        let mut sources = SourceSet::default();
        sources.push(SourceFile {
            path: "shapes.seismic".into(),
            text: "fn dimensions[NK, GV, W](q: &tensor[(2 * NK + NK * GV) * W] f32, p: &tensor[NK * GV] f32, w: &tensor[W] f32):\n    let value = w[0]\n".into(),
        });
        let module = check_source(sources).expect("composed observations determine dimensions");
        let entry = module
            .entry(
                module.entry_named("dimensions").unwrap(),
                &ElementBindings::default(),
            )
            .expect("monomorphization retains the same dimension equations");
        let plan = entry
            .schema()
            .compile_dimension_inference(entry.arena(), &crate::expr::PartialAssignment::new());
        let mut values = crate::expr::compiled::InvocationValues::new();
        plan.infer(&[40, 6, 4], &mut values)
            .expect("all dimensions solve before checking original equations");
        assert!(plan
            .infer(
                &[41, 6, 4],
                &mut crate::expr::compiled::InvocationValues::new()
            )
            .is_err());
    }

    #[test]
    fn private_parallel_storage_does_not_require_shared_write_authority() {
        let mut sources = SourceSet::default();
        sources.push(SourceFile {
            path: "private.seismic".into(),
            text: "fn update[N, W](x: &tensor[N, W] f32, output: &mut tensor[N, W] f32):\n    parallel for i in 0..N:\n        let mut row = to_owned(x[i])\n        for j in 0..W:\n            row[j] = row[j] + 1.0\n        output[i] = row\n".into(),
        });
        let module = check_source(sources).expect("private writes and disjoint publication check");
        module
            .entry(
                module.entry_named("update").unwrap(),
                &ElementBindings::default(),
            )
            .expect("lowering preserves checked private and shared ownership");
    }

    #[test]
    fn shared_parallel_storage_still_requires_disjoint_writes() {
        let mut sources = SourceSet::default();
        sources.push(SourceFile {
            path: "shared.seismic".into(),
            text: "fn update[N](output: &mut tensor[N] f32):\n    parallel for i in 0..N:\n        output[0] = 1.0\n".into(),
        });
        assert!(check_source(sources).is_err());
    }

    #[test]
    fn native_implementation_attaches_to_the_portable_entry() {
        let module = check_source(source(
            "native scale for metal from \"scale.metal\":\n    launch scale:\n        threadgroups (ceil_div(N, 256), 1, 1)\n        threads_per_threadgroup (256, 1, 1)\n",
        ))
        .expect("native declaration should check");
        let entry = &module.entries()[0];
        let native = module
            .native_implementation(entry.id, BackendName::Metal)
            .expect("native implementation");
        assert_eq!(entry.dimensions, ["N"]);
        assert_eq!(native.source_path, "scale.metal");
        assert_eq!(native.launches.len(), 1);
        assert_eq!(native.launches[0].kernel, "scale");
        assert!(matches!(
            native.launches[0].groups[0],
            NativeNatExpr::CeilDiv(_, _)
        ));
        assert_eq!(native.launches[0].shared_bytes, NativeNatExpr::Constant(0));
    }

    const SPECIALIZED: &str = "native scale for cuda from \"scale.cu\":\n    static (N)\n    params (arithmetic PARTS in [1, 2, 4], WIDTH in [64, 128])\n    where PARTS * WIDTH <= N and WIDTH >= 64\n    scratch partials bytes (PARTS * N * 4)\n    launch scale_partial:\n        threadgroups (ceil_div(N, WIDTH), PARTS, 1)\n        threads_per_threadgroup (WIDTH, 1, 1)\n        shared_bytes (max(WIDTH * 4, 256))\n    launch scale_merge:\n        threadgroups (ceil_div(N, 256), 1, 1)\n        threads_per_threadgroup (min(N, 256), 1, 1)\n";

    #[test]
    fn specialized_multi_launch_declaration_checks() {
        let module = check_source(source(SPECIALIZED)).expect("specialized declaration checks");
        let entry = &module.entries()[0];
        let native = module
            .native_implementation(entry.id, BackendName::Cuda)
            .expect("cuda native implementation");
        assert_eq!(native.statics, ["N"]);
        assert_eq!(native.params.len(), 2);
        assert!(native.params[0].arithmetic);
        assert!(!native.params[1].arithmetic);
        assert_eq!(native.params[1].values, [64, 128]);
        assert!(matches!(native.constraint, Some(NativeCondition::And(..))));
        assert_eq!(native.scratch[0].name, "partials");
        assert_eq!(native.launches.len(), 2);
        assert_eq!(native.launches[1].kernel, "scale_merge");
    }

    #[test]
    fn launch_parameters_have_independent_scopes_and_domains() {
        let module = check_source(source(
            "native scale for metal from \"scale.metal\":\n    params (BOUND in [8, 16])\n    launch small when N < BOUND:\n        params (code ROWS in [1, 2])\n        threadgroups (ceil_div(N, ROWS), 1, 1)\n        threads_per_threadgroup (32, 1, 1)\n    launch large when N >= BOUND:\n        params (ROWS in [4, 8])\n        threadgroups (ceil_div(N, ROWS), 1, 1)\n        threads_per_threadgroup (32, 1, 1)\n",
        ))
        .expect("launch-local parameters check");
        let native = module
            .native_implementation(module.entries()[0].id, BackendName::Metal)
            .unwrap();
        assert!(native.launches[0].params[0].code);
        assert!(!native.launches[1].params[0].code);
        let configurations = native.admissible(&NativeSpecialization::new()).unwrap();
        assert_eq!(configurations.len(), 8);
        let default = native
            .default_specialization(&NativeSpecialization::new())
            .unwrap();
        assert_eq!(default.launch_param(0, "ROWS"), Some(1));
        assert_eq!(default.launch_param(1, "ROWS"), Some(4));
        assert!(matches!(
            native.validate(&default.clone().with_launch_param(1, "ROWS", 2)),
            Err(NativeSpecializationError::OutsideLaunchDomain { launch: 1, .. })
        ));
    }

    #[test]
    fn launch_parameters_cannot_escape_their_launch() {
        let sibling = "native scale for metal from \"scale.metal\":\n    launch small:\n        params (ROWS in [1, 2])\n        threadgroups (ceil_div(N, ROWS), 1, 1)\n        threads_per_threadgroup (32, 1, 1)\n    launch large:\n        threadgroups (ceil_div(N, ROWS), 1, 1)\n        threads_per_threadgroup (32, 1, 1)\n";
        let scratch = "native scale for metal from \"scale.metal\":\n    scratch temporary bytes (ROWS * 4)\n    launch small:\n        params (ROWS in [1, 2])\n        threadgroups (ceil_div(N, ROWS), 1, 1)\n        threads_per_threadgroup (32, 1, 1)\n";
        for declaration in [sibling, scratch] {
            let error = check_source(source(declaration)).expect_err("launch scope is local");
            assert!(error.to_string().contains("references `ROWS`"), "{error}");
        }
    }

    #[test]
    fn where_conjuncts_keep_launch_ownership() {
        let source_text = |condition: &str| {
            format!(
            "native scale for metal from \"scale.metal\":\n    where {condition}\n    launch small:\n        params (ROWS in [1, 2])\n        threadgroups (ceil_div(N, ROWS), 1, 1)\n        threads_per_threadgroup (32, 1, 1)\n    launch large:\n        params (TILE in [4, 8])\n        threadgroups (ceil_div(N, TILE), 1, 1)\n        threads_per_threadgroup (32, 1, 1)\n"
        )
        };
        let module = check_source(source(&source_text("ROWS == 1 and TILE >= 4"))).unwrap();
        let native = module
            .native_implementation(module.entries()[0].id, BackendName::Metal)
            .unwrap();
        let configurations = native.admissible(&NativeSpecialization::new()).unwrap();
        assert_eq!(configurations.len(), 2);
        let error =
            check_source(source(&source_text("ROWS < TILE"))).expect_err("cross-launch conjunct");
        assert!(error.to_string().contains("only one launch"), "{error}");
    }

    #[test]
    fn native_domain_enumerates_admissible_configurations() {
        let module = check_source(source(SPECIALIZED)).expect("specialized declaration checks");
        let native = module
            .native_implementation(module.entries()[0].id, BackendName::Cuda)
            .unwrap();
        let statics = NativeSpecialization::new().with_static("N", 256);
        let admissible = native.admissible(&statics).expect("statics are complete");
        // PARTS * WIDTH <= 256 admits (1,64) (1,128) (2,64) (2,128) (4,64).
        assert_eq!(admissible.len(), 5);
        assert!(admissible
            .iter()
            .all(|configuration| configuration.static_value("N") == Some(256)));
        let default = native.default_specialization(&statics).unwrap();
        assert_eq!(default.param("PARTS"), Some(1));
        assert_eq!(default.param("WIDTH"), Some(64));
        let small = NativeSpecialization::new().with_static("N", 32);
        assert!(matches!(
            native.default_specialization(&small),
            Err(NativeSpecializationError::Inadmissible)
        ));
        assert!(matches!(
            native.validate(&default.clone().with_param("WIDTH", 96)),
            Err(NativeSpecializationError::OutsideDomain { .. })
        ));
        assert!(matches!(
            native.admissible(&NativeSpecialization::new()),
            Err(NativeSpecializationError::MissingStatic(_))
        ));
    }

    #[test]
    fn conditions_gate_launches_and_scratch_and_restrict_configurations() {
        let module = check_source(source(
            "native scale for cuda from \"scale.cu\":\n    params (PARTS in [1, 2, 4])\n    where PARTS == 1 or (PARTS >= 2 and PARTS < 4)\n    scratch partials bytes (PARTS * N * 4) when PARTS > 1 and N > 256\n    launch scale_small when N <= 256:\n        threadgroups (1, 1, 1)\n        threads_per_threadgroup (N, 1, 1)\n    launch scale_large when N > 256 or PARTS > 1:\n        threadgroups (ceil_div(N, 256), PARTS, 1)\n        threads_per_threadgroup (256, 1, 1)\n    launch scale_merge:\n        threadgroups (1, 1, 1)\n        threads_per_threadgroup (1, 1, 1)\n",
        ))
        .expect("`when` may read a dimension that is not static");
        let native = module
            .native_implementation(module.entries()[0].id, BackendName::Cuda)
            .unwrap();
        // `where` gained `or`: PARTS 4 is outside it.
        let admissible = native.admissible(&NativeSpecialization::new()).unwrap();
        assert_eq!(
            admissible
                .iter()
                .map(|configuration| configuration.param("PARTS").unwrap())
                .collect::<Vec<_>>(),
            [1, 2]
        );
        let at = |n: u64, parts: u64| {
            move |condition: &NativeCondition| {
                condition
                    .holds(&|name| (name == "N").then_some(n), &|name| {
                        (name == "PARTS").then_some(parts)
                    })
                    .unwrap()
            }
        };
        let small = native.launches[0].when.as_ref().unwrap();
        let large = native.launches[1].when.as_ref().unwrap();
        let partials = native.scratch[0].when.as_ref().unwrap();
        assert!(native.launches[2].when.is_none());
        assert!(at(64, 1)(small) && !at(64, 1)(large) && !at(64, 1)(partials));
        assert!(!at(1024, 1)(small) && at(1024, 1)(large) && !at(1024, 1)(partials));
        assert!(at(64, 2)(small) && at(64, 2)(large) && !at(64, 2)(partials));
        assert!(at(1024, 2)(partials));
        let mut read = Vec::new();
        large.dimensions(&mut read);
        assert_eq!(read, ["N"]);
        // `and` and `or` evaluate their right side only when needed.
        let unbound = NativeCondition::Compare {
            comparison: NativeComparison::Eq,
            left: NativeNatExpr::Dimension("Z".into()),
            right: NativeNatExpr::Constant(0),
        };
        let never = NativeCondition::Compare {
            comparison: NativeComparison::Lt,
            left: NativeNatExpr::Constant(1),
            right: NativeNatExpr::Constant(0),
        };
        let none = |_: &str| None;
        assert_eq!(
            NativeCondition::And(Box::new(never.clone()), Box::new(unbound.clone()))
                .holds(&none, &none),
            Ok(false)
        );
        assert_eq!(
            NativeCondition::Or(Box::new(never), Box::new(unbound)).holds(&none, &none),
            Err(NativeEvalError::Unbound("Z".into()))
        );
    }

    #[test]
    fn vulkan_launch_geometry_reads_only_static_dimensions_and_parameters() {
        let module = check_source(SourceSet::new(vec![SourceFile {
            path: "rows.seismic".to_owned(),
            text: "fn rows[M, N](x: &tensor[M, N] f32) -> tensor[M] f32:\n    let mut output = tensor[M] f32\n    for i in 0..M:\n        output[i] = x[i, 0]\n    return output\n\nnative rows for vulkan from \"vulkan/rows.comp\":\n    static (N)\n    params (WIDTH in [64, 128])\n    scratch partials bytes (M * 4) when M > 1\n    launch rows when M > 0:\n        threadgroups (ceil_div(M, WIDTH), 1, 1)\n        threads_per_threadgroup (WIDTH, 1, 1)\n        shared_bytes (min(N, 8) * WIDTH * 4)\n".to_owned(),
        }]))
        .expect("group counts, `when` and scratch may read per-call dimensions");
        let native = module
            .native_implementation(module.entries()[0].id, BackendName::Vulkan)
            .expect("vulkan implementation");
        assert_eq!(native.backend.as_str(), "vulkan");
        for (property, expected) in [
            (
                "threads_per_threadgroup (M, 1, 1)",
                "`M` is only known per call",
            ),
            (
                "threads_per_threadgroup (64, 1, 1)\n        shared_bytes (min(M, 8) * 72)",
                "`M` is only known per call",
            ),
        ] {
            let error = check_source(SourceSet::new(vec![SourceFile {
                path: "rows.seismic".to_owned(),
                text: format!("fn rows[M, N](x: &tensor[M, N] f32) -> tensor[M] f32:\n    let mut output = tensor[M] f32\n    for i in 0..M:\n        output[i] = x[i, 0]\n    return output\n\nnative rows for vulkan from \"vulkan/rows.comp\":\n    static (N)\n    launch rows:\n        threadgroups (1, 1, 1)\n        {property}\n"),
            }]))
            .expect_err("per-call launch geometry is rejected on Vulkan");
            let text = error.to_string();
            assert!(
                text.contains(expected) && text.contains("Vulkan fixes the group size"),
                "{text}"
            );
        }
        // The same declaration is admitted for CUDA, where geometry is per launch.
        check_source(SourceSet::new(vec![SourceFile {
            path: "rows.seismic".to_owned(),
            text: "fn rows[M, N](x: &tensor[M, N] f32) -> tensor[M] f32:\n    let mut output = tensor[M] f32\n    for i in 0..M:\n        output[i] = x[i, 0]\n    return output\n\nnative rows for cuda from \"cuda/rows.cu\":\n    launch rows:\n        threadgroups (1, 1, 1)\n        threads_per_threadgroup (M, 1, 1)\n        shared_bytes (min(M, 8) * 72)\n".to_owned(),
        }]))
        .expect("CUDA launch geometry may read per-call dimensions");
    }

    #[test]
    fn lowerings_are_rejected_for_the_native_only_vulkan_backend() {
        let source = |target: &str| {
            SourceSet::new(vec![SourceFile {
                path: "copy.seismic".to_owned(),
                text: format!("fn copy[N](x: &tensor[N] f32) -> tensor[N] f32:\n    let mut output = tensor[N] f32\n    for i in 0..N:\n        output[i] = x[i]\n    return output\n\nlower copy[N](x: &tensor[N] f32) -> tensor[N] f32\n    for {target}:\n    let mut output = tensor[N] f32\n    for i in 0..N:\n        output[i] = x[i]\n    return output\n"),
            }])
        };
        check_source(source("cpu")).expect("a lowering for a compiler target is admitted");
        let text = check_source(source("vulkan"))
            .expect_err("vulkan has no compiler target")
            .to_string();
        assert!(
            text.contains("`vulkan` runs only native implementations"),
            "{text}"
        );
    }

    #[test]
    fn duplicate_native_implementations_are_rejected() {
        let declaration = "native scale for metal from \"scale.metal\":\n    launch scale:\n        threadgroups (1, 1, 1)\n        threads_per_threadgroup (1, 1, 1)\n";
        let error = check_source(source(&format!("{declaration}\n{declaration}")))
            .expect_err("duplicate implementation must fail");
        assert!(error
            .to_string()
            .contains("already has a native implementation"));
    }

    /// `elements` lists the dense types a CPU form compiles a stored element
    /// parameter for, and is rejected anywhere else.
    #[test]
    fn cpu_element_coverage_names_stored_element_parameters() {
        let check = |native: &str| {
            let mut sources = SourceSet::default();
            sources.push(SourceFile {
                path: "ops.seismic".to_owned(),
                text: format!(
                    "fn copy[N](src: &tensor[N] A, dst: &mut tensor[N] A, w: &tensor[N] W):\n    dst[0:N] = src[0:N]\n\nfn widen[N](w: &tensor[N] W) -> tensor[N] U:\n    return repack[U = U](w)\n\n{native}"
                ),
            });
            check_source(sources)
        };
        let launch = "    launch copy:\n        threadgroups (1, 1, 1)\n        threads_per_threadgroup (1, 1, 1)\n";
        let module = check(&format!(
            "native copy for cpu from \"copy.rs\":\n    elements (A in [f32, u32, i32])\n{launch}"
        ))
        .expect("CPU element coverage checks");
        let native = module
            .native_implementation(module.entry_named("copy").unwrap(), BackendName::Cpu)
            .expect("cpu native implementation");
        assert_eq!(
            native.elements,
            [NativeElementCoverage {
                parameter: "A".to_owned(),
                dtypes: vec![DType::F32, DType::U32, DType::I32]
            }]
        );
        let widen = "    launch widen:\n        threadgroups (1, 1, 1)\n        threads_per_threadgroup (1, 1, 1)\n";
        for (native, message) in [
            (format!("native copy for metal from \"copy.metal\":\n    elements (A in [f32])\n{launch}"), "other backends compile each binding"),
            (format!("native copy for cpu from \"copy.rs\":\n    elements (B in [f32])\n{launch}"), "`B` is not an element parameter"),
            (format!("native copy for cpu from \"copy.rs\":\n    elements (W in [f32])\n{launch}"), "`W` is not stored"),
            (format!("native widen for cpu from \"widen.rs\":\n    elements (U in [f32])\n{widen}"), "`U` is not stored"),
            (format!("native copy for cpu from \"copy.rs\":\n    elements (A in [f32, f32])\n{launch}"), "`f32` is listed twice"),
            (format!("native copy for cpu from \"copy.rs\":\n    elements (A in [bool])\n{launch}"), "`bool` is not a CPU element type"),
            (format!("native copy for cpu from \"copy.rs\":\n    elements (A in [f32], A in [u32])\n{launch}"), "declared twice"),
        ] {
            let error = check(&native).expect_err(message).to_string();
            assert!(error.contains(message), "{message}: {error}");
        }
    }

    #[test]
    fn native_implementation_rejects_unknown_contract_facts() {
        let launch = "    launch scale:\n        threadgroups (1, 1, 1)\n        threads_per_threadgroup (1, 1, 1)\n";
        let cases = [
            (
                format!("native missing for metal from \"scale.metal\":\n{launch}"),
                "unknown portable function `missing`",
            ),
            (
                "native scale for metal from \"scale.metal\":\n    launch scale:\n        threadgroups (ceil_div(M, 256), 1, 1)\n        threads_per_threadgroup (256, 1, 1)\n".to_owned(),
                "references `M`, which is neither a dimension nor a native parameter",
            ),
            (
                format!("native scale for tpu from \"scale.c\":\n{launch}"),
                "unknown native backend `tpu`",
            ),
            (
                format!("native scale for metal from \"scale.metal\":\n    static (K)\n{launch}"),
                "`K` is not a shape dimension of `scale`",
            ),
            (
                format!("native scale for metal from \"scale.metal\":\n    params (N in [1])\n{launch}"),
                "native parameter `N` shadows a dimension",
            ),
            (
                format!("native scale for metal from \"scale.metal\":\n    params (P in [1, 1])\n{launch}"),
                "lists 1 twice",
            ),
            (
                format!("native scale for metal from \"scale.metal\":\n    params (P in [1])\n    where P <= N\n{launch}"),
                "reads dimension `N`, which is not static",
            ),
            (
                format!("native scale for metal from \"scale.metal\":\n    params (P in [1])\n    where P + 1\n{launch}"),
                "a native `where` condition is a comparison",
            ),
            (
                format!("native scale for metal from \"scale.metal\":\n    params (P in [1])\n    where P == 1 or P < N\n{launch}"),
                "reads dimension `N`, which is not static",
            ),
            (
                "native scale for metal from \"scale.metal\":\n    launch scale when N:\n        threadgroups (1, 1, 1)\n        threads_per_threadgroup (1, 1, 1)\n".to_owned(),
                "a native `when` condition is a comparison",
            ),
            (
                "native scale for metal from \"scale.metal\":\n    launch scale when N + 1:\n        threadgroups (1, 1, 1)\n        threads_per_threadgroup (1, 1, 1)\n".to_owned(),
                "a native `when` condition is a comparison",
            ),
            (
                "native scale for metal from \"scale.metal\":\n    launch scale when N > 1 and (N < 8 or N):\n        threadgroups (1, 1, 1)\n        threads_per_threadgroup (1, 1, 1)\n".to_owned(),
                "a native `when` condition is a comparison",
            ),
            (
                "native scale for metal from \"scale.metal\":\n    launch scale when M > 1:\n        threadgroups (1, 1, 1)\n        threads_per_threadgroup (1, 1, 1)\n".to_owned(),
                "references `M`, which is neither a dimension nor a native parameter",
            ),
            (
                format!("native scale for metal from \"scale.metal\":\n    scratch partials bytes (N * 4) when N\n{launch}"),
                "a native `when` condition is a comparison",
            ),
            (
                format!("native scale for metal from \"scale.metal\":\n    scratch partials bytes (N * 4) when Q >= 2\n{launch}"),
                "references `Q`, which is neither a dimension nor a native parameter",
            ),
            (
                "native scale for metal from \"scale.metal\":\n    static (N)\n".to_owned(),
                "expected `launch <kernel>:`",
            ),
        ];

        for (declaration, expected) in cases {
            let error = check_source(source(&declaration))
                .expect_err("invalid native declaration must fail checking");
            assert!(
                error.to_string().contains(expected),
                "expected diagnostic containing {expected:?}, got {error}"
            );
        }
    }

    #[test]
    fn equivalent_family_bodies_propagate_renamed_element_bindings() {
        let module = check_source(SourceSet::new(vec![SourceFile {
            path: "repack.seismic".to_owned(),
            text: "fn repack_weight[N](source: &tensor[N] E) -> tensor[N] U:\n    return repack[U = U](source)\n\nfn repack_weight[N](source: &tensor[N] T) -> tensor[N] V:\n    return repack[U = V](source)\n"
                .to_owned(),
        }]))
        .expect("equivalent generic spellings form one checked family");
        let entry = module.entry_named("repack_weight").unwrap();
        let bindings = ElementBindings::new()
            .bind("E", registry::representation("gguf_q8_0").unwrap())
            .bind("U", registry::representation("q8g32s").unwrap());
        let logical = module
            .entry(entry, &bindings)
            .expect("every family body inherits the contract element binding");
        assert_eq!(
            logical
                .program()
                .family(logical.program().root())
                .candidates()
                .len(),
            2
        );
    }

    #[test]
    fn equivalent_family_bodies_propagate_renamed_tuple_result_bindings() {
        let module = check_source(SourceSet::new(vec![SourceFile {
            path: "repack_tuple.seismic".to_owned(),
            text: "fn repack_pair[N](source: &tensor[N] E) -> (tensor[N] U, f32):\n    return (repack[U = U](source), f32(0.0))\n\nfn repack_pair[N](source: &tensor[N] T) -> (tensor[N] V, f32):\n    return (repack[U = V](source), f32(0.0))\n"
                .to_owned(),
        }]))
        .expect("equivalent tuple result generics form one checked family");
        let entry = module.entry_named("repack_pair").unwrap();
        let bindings = ElementBindings::new()
            .bind("E", registry::representation("gguf_q8_0").unwrap())
            .bind("U", registry::representation("q8g32s").unwrap());
        let logical = module
            .entry(entry, &bindings)
            .expect("tuple result generic binding reaches every family body");
        assert_eq!(
            logical
                .program()
                .family(logical.program().root())
                .candidates()
                .len(),
            2
        );
    }
}

#[cfg(test)]
mod diagnostic_tests {
    use super::*;

    fn rejected(text: &str) -> SourceError {
        check_source(SourceSet::new(vec![SourceFile {
            path: "probe.seismic".into(),
            text: text.into(),
        }]))
        .expect_err("the source must be rejected")
    }

    fn single(error: &SourceError) -> &SourceDiagnostic {
        let [item] = error.diagnostics().items() else {
            panic!("expected one diagnostic, got {error}")
        };
        item
    }

    #[test]
    fn display_renders_location_rule_message_and_caret() {
        let error = rejected("fn h(x: f32) -> f32 for cpu:\n    return x\n");
        let item = single(&error);
        assert_eq!(item.rule, DiagnosticRule::Syntax);
        assert_eq!((item.location.line, item.location.column), (1, 21));
        assert_eq!(
            error.to_string(),
            "probe.seismic:1:21: Syntax: backend code is written as `lower NAME … for BACKEND`; a `fn` is portable\n  fn h(x: f32) -> f32 for cpu:\n                      ^^^"
        );
    }

    #[test]
    fn diagnostics_drop_every_repeated_item() {
        let file = SourceFile {
            path: "probe.seismic".into(),
            text: "fn f() -> i32:\n    return 0\n".into(),
        };
        let item = |end: usize, message: &str| {
            SourceDiagnostic::new(&file, Span::new(3, end), DiagnosticRule::Type, message)
        };
        let diagnostics =
            Diagnostics::new(vec![item(4, "first"), item(5, "other"), item(4, "first")])
                .expect("the items are not empty");
        assert_eq!(diagnostics.items(), &[item(4, "first"), item(5, "other")]);
    }

    #[test]
    fn line_column_counts_characters_from_one() {
        let text = "ab\n→cd\n";
        assert_eq!(line_column(text, Span::new(0, 1)), (1, 1));
        assert_eq!(line_column(text, Span::new(3, 4)), (2, 1));
        assert_eq!(line_column(text, Span::new(6, 7)), (2, 2));
    }

    #[test]
    fn resolution_rules_are_typed() {
        let recursion = rejected(
            "fn a[N](x: &tensor[N] i32) -> i32:\n    return b(x)\n\nfn b[N](x: &tensor[N] i32) -> i32:\n    return a(x)\n",
        );
        let item = single(&recursion);
        assert_eq!(item.rule, DiagnosticRule::Recursion);
        assert!(item.message.contains("`a` -> `b` -> `a`"), "{recursion}");

        let names = rejected(
            "fn diff(a: i32, b: i32) -> i32:\n    return a - b\n\nfn diff(b: i32, a: i32) -> i32:\n    return a - b\n",
        );
        let item = single(&names);
        assert_eq!(item.rule, DiagnosticRule::CallContract);
        assert_eq!(item.location.line, 4);
        assert!(item.message.contains("parameter names differ"), "{names}");

        for builtin in ["max", "index", "range", "f32", "to_owned"] {
            let error = rejected(&format!("fn {builtin}(a: f32) -> f32:\n    return a\n"));
            let item = single(&error);
            assert_eq!(item.rule, DiagnosticRule::Resolution, "{error}");
            assert!(
                item.message.contains("names a builtin operation"),
                "{error}"
            );
        }
        // L20: each recursion is one diagnostic and hides no other definition.
        let recursions = rejected(
            "fn a(x: i32) -> i32:\n    return b(x)\n\nfn b(x: i32) -> i32:\n    return a(x)\n\nfn c(x: i32) -> i32:\n    return c(x)\n\nfn d(x: i32) -> i32:\n    return a(x)\n\nfn e(x: i32) -> i32:\n    return x + true\n",
        );
        let items = recursions.diagnostics().items();
        let rules = items.iter().map(|item| item.rule).collect::<Vec<_>>();
        assert_eq!(
            rules,
            [
                DiagnosticRule::Recursion,
                DiagnosticRule::Recursion,
                DiagnosticRule::Type
            ],
            "{recursions}"
        );
        assert!(
            items[0].message.contains("`a` -> `b` -> `a`"),
            "{recursions}"
        );
        assert!(items[1].message.contains("`c` -> `c`"), "{recursions}");

        // L8: retired spellings are ordinary names.
        for retired in [
            "exp_fast", "load", "clone", "decode", "valid", "capacity", "coord",
        ] {
            check_source(SourceSet::new(vec![SourceFile {
                path: "probe.seismic".into(),
                text: format!("fn {retired}(a: f32) -> f32:\n    return a\n"),
            }]))
            .unwrap_or_else(|error| panic!("`{retired}` is an ordinary name: {error}"));
        }
    }

    #[test]
    fn retired_keywords_are_ordinary_names() {
        check_source(SourceSet::new(vec![SourceFile {
            path: "names.seismic".into(),
            text:
                "fn f(x: f32) -> f32:\n    let stage = x\n    let tile = stage\n    return tile\n"
                    .into(),
        }]))
        .expect("retired keywords are ordinary names");
    }

    #[test]
    fn element_uses_admit_by_representation() {
        let dense = registry::dense;
        let q4g64 = registry::representation("q4g64").unwrap();
        let external = registry::representation("gguf_q4_k").unwrap();
        let stored = ElementUses {
            stored: true,
            ..ElementUses::default()
        };
        let read = ElementUses {
            decoded_read: true,
            ..ElementUses::default()
        };
        let copied = ElementUses {
            decoded_read: true,
            partial_copy: true,
            ..ElementUses::default()
        };
        assert!(stored.admits(dense(DType::BF16)));
        assert!(!stored.admits(q4g64));
        assert!(!stored.admits(dense(DType::I32)));
        assert!(read.admits(q4g64));
        assert!(!read.admits(external));
        assert!(!copied.admits(q4g64));
        assert!(ElementUses::default().admits(external));
        assert!(!ElementUses::default().admits(dense(DType::Bool)));
    }

    #[test]
    fn element_domain_admits_exactly_its_bindings() {
        let external = registry::representation("gguf_q4_k").unwrap();
        let q4k = registry::representation("q4k").unwrap();
        let domain = ElementDomain::new(
            vec![
                ElementParameter {
                    name: "T".into(),
                    uses: ElementUses {
                        conversion_source: true,
                        ..ElementUses::default()
                    },
                },
                ElementParameter {
                    name: "U".into(),
                    uses: ElementUses {
                        stored: true,
                        ..ElementUses::default()
                    },
                },
            ],
            vec![ElementConversion {
                source: "T".into(),
                target: ElementTarget::Concrete(q4k),
            }],
        );
        let f32 = registry::dense(DType::F32);
        assert_eq!(
            domain.admit(&ElementBindings::new().bind("T", external).bind("U", f32)),
            Ok(())
        );
        assert_eq!(
            domain.admit(&ElementBindings::new().bind("T", external)),
            Err(ElementBindingError::Missing {
                parameter: "U".into()
            })
        );
        assert_eq!(
            domain.admit(
                &ElementBindings::new()
                    .bind("T", external)
                    .bind("U", f32)
                    .bind("V", f32)
            ),
            Err(ElementBindingError::Unexpected {
                parameter: "V".into()
            })
        );
        assert_eq!(
            domain.admit(&ElementBindings::new().bind("T", external).bind("U", q4k)),
            Err(ElementBindingError::Inadmissible {
                parameter: "U".into(),
                representation: q4k,
                uses: ElementUses {
                    stored: true,
                    ..ElementUses::default()
                },
            })
        );
        assert_eq!(
            domain.admit(&ElementBindings::new().bind("T", f32).bind("U", f32)),
            Err(ElementBindingError::NoConversion {
                source: f32,
                target: q4k
            })
        );
    }
}
