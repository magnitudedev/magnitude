//! The opaque checked module (spec §3.2).
//!
//! Exactly two constructors exist: [`check_source`] and
//! [`crate::bundle::decode_checked_bundle`]. There is no struct literal,
//! `Default`, unchecked deserializer, arena mutation, or constructor that
//! accepts already-typed nodes. Consumers read through accessors and obtain a
//! [`LogicalEntry`] through [`CheckedModule::entry`], the only builder of
//! entry semantics.
//!
//! W1 owns the internals behind `internals::Module`.

use crate::entry::{ElementBindings, LogicalEntry};
use crate::ids::{EntryId, ModuleHash, RepresentationId, StableEntryId};
use crate::registry::{self, BackendName, RepresentationAccess, RepresentationKind};
use crate::span::Span;
use crate::types::DType;

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
        let start = line.char_indices().nth(before).map_or(line.len(), |(offset, _)| offset);
        let width = line[start..]
            .char_indices()
            .take_while(|(offset, _)| *offset < span_bytes)
            .count()
            .max(1);
        write!(f, "\n  {line}\n  {}{}", " ".repeat(before), "^".repeat(width))
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
    internals::check(sources).map(|inner| CheckedModule { inner, assets: Default::default() })
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
}

/// Summary of one entry sufficient for binding generation before
/// monomorphization.
#[derive(Clone, Debug, PartialEq, Eq)]
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
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ElementDomain {
    parameters: Vec<ElementParameter>,
    conversions: Vec<ElementConversion>,
}

/// One element parameter and every use the entry makes of it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ElementParameter {
    pub name: String,
    pub uses: ElementUses,
}

/// How an entry uses the elements of one element parameter.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct ElementUses {
    pub decoded_read: bool,
    pub stored: bool,
    pub conversion_source: bool,
    /// `to_owned` of a view selecting part of the packing axis.
    pub partial_copy: bool,
}

/// `repack[U = target](t)` with `t: tensor[..] source`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ElementConversion {
    pub source: String,
    pub target: ElementTarget,
}

#[derive(Clone, Debug, PartialEq, Eq)]
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
        let name = |representation: &RepresentationId| registry::representation_info(*representation).name;
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
            || (matches!(info.kind, RepresentationKind::Packed(_))
                && registry::decode_recipe(representation, DType::F32).is_some());
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

    pub(crate) fn conversions(&self) -> &[ElementConversion] {
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
            if !self.parameters.iter().any(|parameter| parameter.name == name) {
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
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SignatureType {
    Unit,
    Tuple(Vec<SignatureType>),
    Tensor { access: TensorAccess, rank: u32, element: ElementSummary },
    Scalar(DType),
    Index,
    Range,
}

/// One direct top-level native implementation attached to a checked entry.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NativeImplementation {
    pub entry: EntryId,
    pub backend: BackendName,
    /// Canonical module source label containing the declaration.
    pub declared_in: String,
    /// The native source path, relative to the declaring source file.
    pub source_path: String,
    pub launch: NativeLaunch,
}

/// Launch geometry of a native implementation over the entry's dimensions.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NativeLaunch {
    /// Number of groups on each axis.
    pub groups: [NativeNatExpr; 3],
    /// Participants of one group on each axis.
    pub group_extent: [NativeNatExpr; 3],
}

/// Closed integer language used by native launch geometry.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum NativeNatExpr {
    Constant(u64),
    Dimension(String),
    Add(Box<Self>, Box<Self>),
    Sub(Box<Self>, Box<Self>),
    Mul(Box<Self>, Box<Self>),
    Div(Box<Self>, Box<Self>),
    Rem(Box<Self>, Box<Self>),
    CeilDiv(Box<Self>, Box<Self>),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParameterSummary {
    /// Authored parameter ordinal and tuple path. The summary is leaf-flat;
    /// generated bindings group leaves by these canonical coordinates.
    pub source: u32,
    pub path: Vec<u32>,
    pub name: String,
    pub kind: ParameterSummaryKind,
}

#[derive(Clone, Debug, PartialEq, Eq)]
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TensorAccess {
    Owned,
    Shared,
    Mutable,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ElementSummary {
    /// A fixed dtype or packed representation, by registry name.
    Fixed(String),
    /// Bound at `for_device` time by an element parameter name.
    Parameter(String),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResultSummary {
    /// Ordinal tuple path; empty for a non-tuple result.
    pub path: Vec<u32>,
    pub kind: ResultSummaryKind,
}

#[derive(Clone, Debug, PartialEq, Eq)]
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
    //! `check` and the bundle decoder.

    use super::{Diagnostics, EntryInfo, NativeImplementation, SourceError, SourceSet};
    use crate::entry::{ElementBindings, LogicalEntry};
    use crate::ids::{EntryId, ModuleHash, ModuleId, ProgramId};

    #[derive(Debug)]
    pub(crate) struct Module {
        pub(crate) id: ModuleId,
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
            "native scale for metal from \"scale.metal\":\n    threadgroups (ceil_div(N, 256), 1, 1)\n    threads_per_threadgroup (256, 1, 1)\n",
        ))
        .expect("native declaration should check");
        let entry = &module.entries()[0];
        let native = module
            .native_implementation(entry.id, BackendName::Metal)
            .expect("native implementation");
        assert_eq!(entry.dimensions, ["N"]);
        assert_eq!(native.source_path, "scale.metal");
        assert!(matches!(
            native.launch.groups[0],
            NativeNatExpr::CeilDiv(_, _)
        ));
    }

    #[test]
    fn duplicate_native_implementations_are_rejected() {
        let declaration = "native scale for metal from \"scale.metal\":\n    threadgroups (1, 1, 1)\n    threads_per_threadgroup (1, 1, 1)\n";
        let error = check_source(source(&format!("{declaration}\n{declaration}")))
            .expect_err("duplicate implementation must fail");
        assert!(error
            .to_string()
            .contains("already has a native implementation"));
    }

    #[test]
    fn native_implementation_rejects_unknown_contract_facts() {
        let cases = [
            (
                "native missing for metal from \"scale.metal\":\n    threadgroups (1, 1, 1)\n    threads_per_threadgroup (1, 1, 1)\n",
                "unknown portable function `missing`",
            ),
            (
                "native scale for metal from \"scale.metal\":\n    threadgroups (ceil_div(M, 256), 1, 1)\n    threads_per_threadgroup (256, 1, 1)\n",
                "unknown dimension `M`",
            ),
            (
                "native scale for cpu from \"scale.c\":\n    threadgroups (1, 1, 1)\n    threads_per_threadgroup (1, 1, 1)\n",
                "currently support only `metal`",
            ),
        ];

        for (declaration, expected) in cases {
            let error = check_source(source(declaration))
                .expect_err("invalid native declaration must fail checking");
            assert!(
                error.to_string().contains(expected),
                "expected diagnostic containing {expected:?}, got {error}"
            );
        }
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

        for builtin in [
            "max", "exp_fast", "index", "range", "f32", "to_owned", "load", "clone", "decode",
            "valid", "capacity", "coord",
        ] {
            let error = rejected(&format!("fn {builtin}(a: f32) -> f32:\n    return a\n"));
            let item = single(&error);
            assert_eq!(item.rule, DiagnosticRule::Resolution, "{error}");
            assert!(item.message.contains("names a builtin operation"), "{error}");
        }
    }

    #[test]
    fn retired_keywords_are_ordinary_names() {
        check_source(SourceSet::new(vec![SourceFile {
            path: "names.seismic".into(),
            text: "fn f(x: f32) -> f32:\n    let stage = x\n    let tile = stage\n    return tile\n".into(),
        }]))
        .expect("retired keywords are ordinary names");
    }

    #[test]
    fn element_uses_admit_by_representation() {
        let dense = registry::dense;
        let q4g64 = registry::representation("q4g64").unwrap();
        let external = registry::representation("gguf_q4_k").unwrap();
        let stored = ElementUses { stored: true, ..ElementUses::default() };
        let read = ElementUses { decoded_read: true, ..ElementUses::default() };
        let copied = ElementUses { decoded_read: true, partial_copy: true, ..ElementUses::default() };
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
                    uses: ElementUses { conversion_source: true, ..ElementUses::default() },
                },
                ElementParameter {
                    name: "U".into(),
                    uses: ElementUses { stored: true, ..ElementUses::default() },
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
            Err(ElementBindingError::Missing { parameter: "U".into() })
        );
        assert_eq!(
            domain.admit(
                &ElementBindings::new().bind("T", external).bind("U", f32).bind("V", f32)
            ),
            Err(ElementBindingError::Unexpected { parameter: "V".into() })
        );
        assert_eq!(
            domain.admit(&ElementBindings::new().bind("T", external).bind("U", q4k)),
            Err(ElementBindingError::Inadmissible {
                parameter: "U".into(),
                representation: q4k,
                uses: ElementUses { stored: true, ..ElementUses::default() },
            })
        );
        assert_eq!(
            domain.admit(&ElementBindings::new().bind("T", f32).bind("U", f32)),
            Err(ElementBindingError::NoConversion { source: f32, target: q4k })
        );
    }
}
