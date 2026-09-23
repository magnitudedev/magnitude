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
use crate::ids::{EntryId, ModuleHash, StableEntryId};
use crate::registry::BackendName;
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
        let mut diagnostics = Vec::new();
        for pair in self.files.windows(2) {
            if pair[0].path == pair[1].path {
                diagnostics.push(SourceDiagnostic {
                    path: pair[1].path.clone(),
                    span: Span::default(),
                    message: "duplicate source path in one module".to_owned(),
                });
            }
        }
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

/// One rendered diagnostic anchored in source.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SourceDiagnostic {
    pub path: String,
    pub span: Span,
    pub message: String,
}

impl SourceDiagnostic {
    pub fn render(&self, text: &str) -> String {
        crate::span::Diagnostic::new(self.span, self.message.clone()).render(&self.path, text)
    }
}

/// A non-empty, ordered, deduplicated set of diagnostics.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Diagnostics {
    items: Vec<SourceDiagnostic>,
}

impl Diagnostics {
    pub(crate) fn new(items: Vec<SourceDiagnostic>) -> Option<Self> {
        (!items.is_empty()).then_some(Self { items })
    }

    pub fn items(&self) -> &[SourceDiagnostic] {
        &self.items
    }

    pub(crate) fn single(item: SourceDiagnostic) -> Self {
        Self { items: vec![item] }
    }
}

/// Source failure taxonomy (§13.2).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SourceError {
    Parse(Diagnostics),
    Type(Diagnostics),
    Effect(Diagnostics),
    Capability(Diagnostics),
}

impl SourceError {
    pub fn diagnostics(&self) -> &Diagnostics {
        match self {
            Self::Parse(d) | Self::Type(d) | Self::Effect(d) | Self::Capability(d) => d,
        }
    }
}

impl std::fmt::Display for SourceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let kind = match self {
            Self::Parse(_) => "parse",
            Self::Type(_) => "type",
            Self::Effect(_) => "effect",
            Self::Capability(_) => "capability",
        };
        write!(f, "{kind} error")?;
        for item in self.diagnostics().items() {
            write!(f, "\n{}: {}", item.path, item.message)?;
        }
        Ok(())
    }
}

impl std::error::Error for SourceError {}

/// Parses and checks a source set as one closed module. The only source-side
/// constructor of a [`CheckedModule`].
pub fn check_source(sources: SourceSet) -> Result<CheckedModule, SourceError> {
    let sources = sources.canonicalized().map_err(SourceError::Parse)?;
    internals::check(sources).map(|inner| CheckedModule { inner, assets: Default::default() })
}

/// An opaque checked semantic object: source semantics, types, effects,
/// canonical bodies, lowering declarations, capability requirements, and
/// stable identities. It decides nothing about targets, schedules,
/// allocations, or native code (§2.1).
#[derive(Debug)]
pub struct CheckedModule {
    inner: internals::Module,
    pub(crate) assets: std::collections::BTreeMap<String, String>,
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

    /// Snapshot a native asset for an entry. The checked declaration remains authoritative.
    pub fn capture_native_asset(&mut self, entry: &str, source: String) -> Result<(), String> {
        let id = self.entry_named(entry).ok_or_else(|| format!("unknown entry {entry}"))?;
        if self.native_implementation(id, BackendName::Metal).is_none() {
            return Err(format!("entry {entry} has no native declaration"));
        }
        self.assets.insert(entry.to_owned(), source);
        Ok(())
    }
    pub fn native_asset(&self, entry: &str) -> Option<&str> {
        self.assets.get(entry).map(String::as_str)
    }

    /// The source set this module was checked from, for diagnostics and
    /// build-script fingerprinting only.
    pub fn sources(&self) -> &SourceSet {
        self.inner.sources()
    }

    pub(crate) fn from_internal(inner: internals::Module) -> Self {
        Self { inner, assets: Default::default() }
    }

    pub(crate) fn internal(&self) -> &internals::Module {
        &self.inner
    }

    pub(crate) fn into_internal(self) -> internals::Module {
        self.inner
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
    pub source: String,
    pub threadgroups: [NativeNatExpr; 3],
    pub threads_per_threadgroup: [NativeNatExpr; 3],
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
        pub(crate) template_program: ProgramId,
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
        let template_program = ProgramId::fresh();
        crate::check::check_closed(sources, id, template_program)
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
                .map_err(|diagnostic| SourceError::Type(Diagnostics::single(diagnostic)))
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
        assert_eq!(native.source, "scale.metal");
        assert!(matches!(
            native.threadgroups[0],
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
