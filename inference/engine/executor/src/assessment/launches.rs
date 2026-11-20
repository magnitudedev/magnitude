//! The native launches of a call: what a backend's declaration of the
//! call's entry dispatches at the call's rows, under the default
//! specialization at the statics a load prepares the entry with.
//!
//! The statics are the load's own: its kernel inventory, listed by the walk
//! preparation makes ([`crate::native::kernel_requests`]). A call names its
//! entry by type ([`Kernel`]); its launches at `rows` rows are the
//! declaration's launches in dispatch order whose `when` clause holds with
//! every row dimension (`M`, `O`) at `rows`, and whose launch conditions
//! (`if (w[i])`, `unless (w[i])`) hold under the host-written words the
//! call's form states ([`Words`]). Every other dimension a clause reads is a
//! static of the load's request, or one the call states
//! ([`Declarations::launches`]); a clause or condition reading a name
//! neither supplies is an error, never a default.

use super::AssessmentError;
use crate::{ModelLoadPlan, ProgramPlan, TuningLimits};
use magnitude_family_contracts::ModelDefinition;
use seismic::{
    BackendName, CheckedBundleError, Element, Entry, KernelRequest, NativeImplementation,
    NativeSpecialization, OperandRecipe,
};
use std::collections::BTreeMap;
use std::fmt;

/// The row dimensions of the engine's entries: the rows a call reads
/// (`M`) and the rows it writes (`O`). A decode step's calls read and write
/// every row of the step.
const ROW_DIMENSIONS: [&str; 2] = ["M", "O"];

/// A native entry a call runs, named by its generated type.
#[derive(Clone, Copy)]
pub struct Kernel {
    name: &'static str,
    /// The entry's declaration on a backend at a request's elements.
    implementation: Option<
        fn(
            BackendName,
            &[(&str, Element)],
        ) -> Result<Option<&'static NativeImplementation>, CheckedBundleError>,
    >,
}

impl Kernel {
    /// The entry `E`.
    pub fn of<E: Entry>() -> Self {
        Self {
            name: E::NAME,
            implementation: Some(seismic::generated::native_implementation_at::<E>),
        }
    }

    /// A kernel identified by its entry name (from `SizingDraft`'s census).
    pub(crate) fn named(name: &'static str) -> Self {
        Self {
            name,
            implementation: None,
        }
    }

    /// The entry's declared name.
    pub fn name(&self) -> &'static str {
        self.name
    }
}

impl PartialEq for Kernel {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name
    }
}

impl Eq for Kernel {}

impl fmt::Debug for Kernel {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.name)
    }
}

/// The host-written words a form encodes its calls' launches under
/// (`seismic::LaunchCondition`): each by its port's name and word index,
/// with whether the form holds it set. A streamed layer's words are what
/// decides which of its declared launches a run encodes; a form states
/// every word its active launches are conditioned on.
pub type Words = [(&'static str, u64, bool)];

fn launch_error(message: impl Into<String>) -> AssessmentError {
    AssessmentError::Demand(message.into())
}

/// One prepared specialization of an entry: its declaration, bound as
/// every device offers it, and the default specialization at the statics
/// the load prepares it with.
#[derive(Clone, Debug)]
struct Prepared {
    implementation: &'static NativeImplementation,
    specialization: NativeSpecialization,
}

/// The declarations of the entries a load prepares on one backend, at the
/// statics it prepares each with.
#[derive(Clone, Debug)]
pub struct Declarations {
    backend: BackendName,
    requests: Vec<KernelRequest>,
}

impl Declarations {
    /// The declarations of the entries a load prepares on `backend`: its
    /// kernel inventory ([`crate::native::kernel_requests`]) for `load` of
    /// `definition` under `programs`, at the statics `limits` tunes.
    pub fn of_load(
        backend: BackendName,
        programs: &ProgramPlan,
        definition: &ModelDefinition,
        load: &ModelLoadPlan,
        limits: TuningLimits,
    ) -> Result<Self, AssessmentError> {
        let requests = crate::native::kernel_requests(backend, programs, definition, load, limits)
            .map_err(|failure| AssessmentError::Catalog(crate::CatalogError::native(backend, failure)))?;
        Self::new(backend, requests)
    }

    /// The declarations of `requests`, the load's kernel inventory on
    /// `backend`.
    fn new(backend: BackendName, requests: Vec<KernelRequest>) -> Result<Self, AssessmentError> {
        if let Some(request) = requests.iter().find(|request| request.backend != backend) {
            return Err(launch_error(format!(
                "kernel request {request} is not of {}",
                backend.as_str()
            )));
        }
        Ok(Self { backend, requests })
    }

    /// The backend the declarations are of.
    pub fn backend(&self) -> BackendName {
        self.backend
    }

    /// Every specialization of `kernel` the load prepares at operand recipe
    /// `recipe` of the call's form `statics` (one layer's, where an entry
    /// serves several), each at its request's elements. A request holds
    /// exactly the statics its backend's implementation declares
    /// (`domain_statics`), so the form selects on those: an axis the backend
    /// leaves dynamic (the CPU's `KV` and `G`) names no specialization.
    fn prepared(
        &self,
        kernel: Kernel,
        recipe: OperandRecipe,
        statics: &[(&'static str, u64)],
    ) -> Result<Vec<Prepared>, AssessmentError> {
        let prepared = self
            .requests
            .iter()
            .filter(|request| {
                request.entry == kernel.name
                    && operand_recipe(request.elements.values().copied()) == recipe
                    && statics.iter().all(|(name, value)| {
                        request.statics.get(*name).is_none_or(|held| held == value)
                    })
            })
            .map(|request| {
                let elements = request
                    .elements
                    .iter()
                    .map(|(name, element)| (name.as_str(), *element))
                    .collect::<Vec<_>>();
                let implementation_fn = kernel.implementation.ok_or_else(|| {
                    launch_error(format!(
                        "`{}` was named without a bundle declaration",
                        kernel.name
                    ))
                })?;
                let implementation = (implementation_fn)(self.backend, &elements)
                    .map_err(|error| launch_error(format!("`{}`: {error}", kernel.name)))?
                    .ok_or(AssessmentError::Unimplemented {
                        entry: kernel.name,
                        backend: self.backend,
                    })?;
                let statics = request
                    .statics
                    .iter()
                    .fold(NativeSpecialization::new(), |statics, (name, value)| {
                        statics.with_static(name.clone(), *value)
                    });
                // An assessment names a backend, not a device: when a
                // parameter's default reads the device, fall back to the first
                // admissible specialization if neither the admission constraint
                // nor any launch-activation expression (`repeat`, `when`) reads
                // a device property or tuning parameter.
                let specialization = match implementation
                    .default_specialization_without_device(&statics)
                {
                    Some(result) => {
                        result.map_err(|error| launch_error(format!("{request}: {error}")))?
                    }
                    None => {
                        let constraint_reads_device =
                            implementation.constraint.as_ref().is_some_and(|constraint| {
                                let mut reads = Vec::new();
                                constraint.device_reads(&mut reads);
                                !reads.is_empty()
                            });
                        let mut launch_params = Vec::new();
                        if let Some(repeat) = &implementation.repeat {
                            repeat.count.parameters(&mut launch_params);
                        }
                        for launch in &implementation.launches {
                            if let Some(when) = &launch.when {
                                when.parameters(&mut launch_params);
                            }
                        }
                        if constraint_reads_device || !launch_params.is_empty() {
                            return Err(launch_error(format!(
                                "`{}`'s defaults read the device's properties, and an assessment has no device",
                                kernel.name
                            )));
                        }
                        let mut first = None;
                        implementation
                            .walk_admissible(&statics, |configuration| {
                                first = Some(configuration.clone());
                                false
                            })
                            .map_err(|error| launch_error(format!("{request}: {error}")))?;
                        first.ok_or_else(|| {
                            launch_error(format!(
                                "{request}: {}",
                                seismic::NativeSpecializationError::Inadmissible
                            ))
                        })?
                    }
                };
                Ok(Prepared {
                    implementation,
                    specialization,
                })
            })
            .collect::<Result<Vec<_>, AssessmentError>>()?;
        if prepared.is_empty() {
            return Err(launch_error(format!(
                "the load prepares no `{}` at {recipe:?} operands and statics {statics:?} on {}",
                kernel.name,
                self.backend.as_str()
            )));
        }
        Ok(prepared)
    }

    /// The launches of a call of `kernel` at operand recipe `recipe` (its
    /// weights', `operand_recipe`) of the specializations with `statics`,
    /// stating `dimensions`, encoded under `words`, at any row count.
    pub fn launches(
        &self,
        kernel: Kernel,
        recipe: OperandRecipe,
        statics: &[(&'static str, u64)],
        dimensions: &[(&'static str, u64)],
        words: &'static Words,
    ) -> Result<Launches, AssessmentError> {
        Ok(Launches {
            kernel,
            source: LaunchSource::Declared {
                prepared: self.prepared(kernel, recipe, statics)?,
                recipe,
                dimensions: dimensions.to_vec(),
                words,
            },
        })
    }
}

/// The operand recipe of products over `elements`: NVFP4 rows where any of
/// them takes them (weights that share rows share their recipe), else the
/// rounded rows. Element bindings change a declaration only through its
/// `operands(W)`, so the recipe is what makes an entry's prepared
/// specializations launch differently.
pub fn operand_recipe(elements: impl IntoIterator<Item = Element>) -> OperandRecipe {
    if elements
        .into_iter()
        .any(|element| element.operand_recipe() == OperandRecipe::Nvfp4Rows)
    {
        OperandRecipe::Nvfp4Rows
    } else {
        OperandRecipe::Rounded
    }
}

#[derive(Clone)]
enum LaunchSource {
    Declared {
        prepared: Vec<Prepared>,
        recipe: OperandRecipe,
        dimensions: Vec<(&'static str, u64)>,
        words: &'static Words,
    },
    Contract {
        backend: BackendName,
        elements: Vec<(String, Element)>,
        dimensions: Vec<(String, u64)>,
        words: &'static Words,
        eval_launches: crate::programs::graph::sizing::LaunchesEvalFn,
    },
}

impl fmt::Debug for LaunchSource {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Declared { recipe, dimensions, words, .. } => formatter
                .debug_struct("Declared")
                .field("recipe", recipe)
                .field("dimensions", dimensions)
                .field("words", words)
                .finish(),
            Self::Contract { backend, elements, dimensions, words, .. } => formatter
                .debug_struct("Contract")
                .field("backend", backend)
                .field("elements", elements)
                .field("dimensions", dimensions)
                .field("words", words)
                .finish(),
        }
    }
}

/// What a call dispatches as a function of its rows.
#[derive(Clone, Debug)]
pub struct Launches {
    kernel: Kernel,
    source: LaunchSource,
}

impl PartialEq for Launches {
    fn eq(&self, other: &Self) -> bool {
        if self.kernel != other.kernel {
            return false;
        }
        match (&self.source, &other.source) {
            (
                LaunchSource::Declared { recipe: r1, dimensions: d1, words: w1, .. },
                LaunchSource::Declared { recipe: r2, dimensions: d2, words: w2, .. },
            ) => r1 == r2 && d1 == d2 && w1 == w2,
            (
                LaunchSource::Contract { backend: b1, elements: e1, dimensions: d1, words: w1, .. },
                LaunchSource::Contract { backend: b2, elements: e2, dimensions: d2, words: w2, .. },
            ) => b1 == b2 && e1 == e2 && d1 == d2 && w1 == w2,
            _ => false,
        }
    }
}

impl Launches {
    /// Construct launch evaluator directly from a `SizingDraft` recorded call,
    /// evaluating active launches via compiled `EntryContract::contract_active_launches`
    /// without decoding `.seismicbundle` metadata.
    pub(crate) fn from_sized_call(
        call: &crate::programs::graph::sizing::SizedCall,
        words: &'static Words,
    ) -> Self {
        Self {
            kernel: Kernel::named(call.entry),
            source: LaunchSource::Contract {
                backend: call.backend,
                elements: call.elements.clone(),
                dimensions: call.dimensions.clone(),
                words,
                eval_launches: call.eval_launches,
            },
        }
    }

    /// The entry the call runs.
    pub fn kernel(&self) -> Kernel {
        self.kernel
    }

    /// The launches one call dispatches at `rows` rows.
    pub fn at(&self, rows: u64) -> Result<u64, AssessmentError> {
        match &self.source {
            LaunchSource::Contract {
                backend,
                elements,
                dimensions,
                words,
                eval_launches,
            } => {
                let els: Vec<(&str, Element)> =
                    elements.iter().map(|(n, e)| (n.as_str(), *e)).collect();
                let dims: Vec<(&str, u64)> = dimensions
                    .iter()
                    .map(|(n, v)| {
                        let value = if ROW_DIMENSIONS.contains(&n.as_str()) {
                            rows
                        } else {
                            *v
                        };
                        (n.as_str(), value)
                    })
                    .collect();
                (eval_launches)(*backend, &els, &dims, words).map_err(|error| {
                    launch_error(format!("`{}` at {rows} rows: {error}", self.kernel.name))
                })
            }
            LaunchSource::Declared { prepared, dimensions, words, .. } => {
                let mut counts = prepared
                    .iter()
                    .map(|prepared| self.count(prepared, dimensions, words, rows))
                    .collect::<Result<Vec<_>, _>>()?;
                counts.sort_unstable();
                counts.dedup();
                match counts[..] {
                    [count] => Ok(count),
                    _ => Err(launch_error(format!(
                        "the prepared specializations of `{}` dispatch {counts:?} launches at {rows} rows",
                        self.kernel.name
                    ))),
                }
            }
        }
    }

    fn count(
        &self,
        prepared: &Prepared,
        dimensions: &[(&'static str, u64)],
        words: &'static Words,
        rows: u64,
    ) -> Result<u64, AssessmentError> {
        let statics = prepared.specialization.statics();
        let mut values = BTreeMap::<&str, u64>::new();
        for (name, value) in statics {
            values.insert(name.as_str(), *value);
        }
        for (name, value) in dimensions {
            values.insert(name, *value);
        }
        for name in ROW_DIMENSIONS {
            values.insert(name, rows);
        }
        let dimension = |name: &str| values.get(name).copied();
        let specialization = &prepared.specialization;
        let name = self.kernel.name;
        let failed = move |error| launch_error(format!("`{name}` at {rows} rows: {error}"));
        let implementation = &prepared.implementation;
        let rounds = match &implementation.repeat {
            Some(repeat) => repeat
                .count
                .evaluate(&dimension, &|name| specialization.param(name))
                .map_err(failed)?,
            None => 0,
        };
        let mut launches = 0u64;
        for ordinal in implementation.dispatch_order(rounds) {
            let launch = &implementation.launches[ordinal];
            let parameter = |name: &str| {
                specialization
                    .launch_param(ordinal, name)
                    .or(specialization.param(name))
            };
            let active = match &launch.when {
                Some(condition) => condition.holds(&dimension, &parameter).map_err(failed)?,
                None => true,
            };
            if !active {
                continue;
            }
            let mut encoded = true;
            for condition in &launch.conditions {
                let (_, _, set) = words
                    .iter()
                    .find(|(word, index, _)| *word == condition.word && *index == condition.index)
                    .ok_or_else(|| {
                        launch_error(format!(
                            "`{name}`'s launch `{}` is conditioned on {}[{}], which its form does not state",
                            launch.kernel, condition.word, condition.index
                        ))
                    })?;
                encoded &= match condition.encoded {
                    seismic::LaunchEncoding::WhileSet => *set,
                    seismic::LaunchEncoding::WhileClear => !*set,
                };
            }
            launches += u64::from(encoded);
        }
        Ok(launches)
    }
}
