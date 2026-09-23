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
//! Scalar results are `f32|i32|u32|bool|u64|(u64,u64)` by kind. Nothing
//! else is generated; consumers never see schema internals.

use seismic_lang::checked::SourceError;
use seismic_lang::checked::{
    EntryInfo, NativeImplementation, NativeNatExpr, ParameterSummary,
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
}

impl std::fmt::Display for BuildError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Source(e) => write!(f, "{e}"),
            Self::Io(e) => write!(f, "io: {e}"),
            Self::Environment(s) => write!(f, "build environment: {s}"),
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
    use std::fs;

    struct NativeAsset<'a> {
        definition: &'a NativeImplementation,
        path: PathBuf,
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

    fn resolve_native_assets<'a>(module: &'a seismic_lang::checked::CheckedModule, output: &std::path::Path) -> Result<Vec<NativeAsset<'a>>, BuildError> {
        let mut assets=Vec::new();
        for entry in module.entries() {
            let Some(definition)=module.native_implementation(entry.id,seismic_lang::registry::BackendName::Metal) else {continue};
            let source=module.native_asset(entry.id,seismic_lang::registry::BackendName::Metal).expect("shared loader captures native assets");
            let path=output.join(format!("{}.metal",entry.name));
            fs::write(&path,source).map_err(BuildError::Io)?;
            assets.push(NativeAsset{definition,path});
        }
        Ok(assets)
    }

    fn render(
        module: &seismic_lang::checked::CheckedModule,
        name: &str,
        identity: &str,
        native_assets: &[NativeAsset<'_>],
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
                .find(|asset| asset.definition.entry == entry.id);
            render_entry(&mut out, entry, native);
        }
        out
    }

    fn render_entry(out: &mut String, entry: &EntryInfo, native: Option<&NativeAsset<'_>>) {
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
            let path = native.path.to_string_lossy();
            out.push_str("  fn native_definition() -> seismic::generated::NativeDefinition {\n");
            out.push_str(&format!(
                "    seismic::generated::NativeDefinition {{ source: include_str!({path:?}).into(), entry: {:?}.into(), threadgroups: [\n",
                entry.name
            ));
            for expression in &native.definition.launch.groups {
                out.push_str("      ");
                render_native_expr(out, expression);
                out.push_str(",\n");
            }
            out.push_str("    ], threads_per_threadgroup: [\n");
            for expression in &native.definition.launch.group_extent {
                out.push_str("      ");
                render_native_expr(out, expression);
                out.push_str(",\n");
            }
            out.push_str("    ] }\n  }\n");
            if entry.element_parameters.is_empty() {
                out.push_str("  pub fn native_for_device(device: &seismic::Device) -> Result<seismic::NativeKernel<Entry>, seismic::LoadError> { seismic::generated::prepare_native::<Entry>(device, native_definition(), &[]) }\n");
            } else {
                out.push_str("  pub fn native_for_device_with(device: &seismic::Device, elements: Elements) -> Result<seismic::NativeKernel<Entry>, seismic::LoadError> {\n");
                out.push_str("    seismic::generated::prepare_native::<Entry>(device, native_definition(), &[\n");
                for parameter in &entry.element_parameters {
                    out.push_str(&format!(
                        "      ({:?}, elements.{}),\n",
                        parameter,
                        ident(parameter)
                    ));
                }
                out.push_str("    ])\n  }\n");
            }
        }
        out.push_str("}\n");
    }

    fn render_native_expr(out: &mut String, expression: &NativeNatExpr) {
        let (name, left, right) = match expression {
            NativeNatExpr::Constant(value) => {
                out.push_str(&format!(
                    "seismic::generated::NativeExpr::constant({value})"
                ));
                return;
            }
            NativeNatExpr::Dimension(name) => {
                out.push_str(&format!(
                    "seismic::generated::NativeExpr::dimension({name:?})"
                ));
                return;
            }
            NativeNatExpr::Add(left, right) => ("add", left, right),
            NativeNatExpr::Sub(left, right) => ("sub", left, right),
            NativeNatExpr::Mul(left, right) => ("mul", left, right),
            NativeNatExpr::Div(left, right) => ("div", left, right),
            NativeNatExpr::Rem(left, right) => ("rem", left, right),
            NativeNatExpr::CeilDiv(left, right) => ("ceil_div", left, right),
        };
        out.push_str(&format!("seismic::generated::NativeExpr::{name}("));
        render_native_expr(out, left);
        out.push_str(", ");
        render_native_expr(out, right);
        out.push(')');
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
            "fn scale[N](x: &tensor[N] f32, factor: f32, output: &mut tensor[N] f32):\n    parallel for i in 0..N:\n        output[i] = x[i] * factor\n\nnative scale for metal from \"native/scale.metal\":\n    threadgroups (ceil_div(N, 256), 1, 1)\n    threads_per_threadgroup (256, 1, 1)\n",
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
        assert!(generated.contains("NativeExpr::ceil_div"));
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
}
