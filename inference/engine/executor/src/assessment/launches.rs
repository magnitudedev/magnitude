//! The native launches of a call: what a backend's declaration of the
//! call's entry dispatches at the call's rows, under the default
//! specialization at the statics a load prepares the entry with.
//!
//! The statics are the load's own: its kernel inventory, listed by the walk
//! preparation makes ([`crate::native::kernel_requests`]). A call names its
//! entry by type ([`Kernel`]); its launches at `rows` rows are the
//! declaration's launches in dispatch order whose `when` clause holds with
//! every row dimension (`M`, `O`) at `rows`. Every other dimension a clause
//! reads is a static of the load's request, or one the call states
//! ([`Declarations::launches`]); a clause reading a name neither supplies
//! is an error, never a default.

use super::AssessmentError;
use crate::{ModelLoadPlan, ProgramPlan, TuningLimits};
use magnitude_family_contracts::ModelDefinition;
use seismic::{
    BackendName, CheckedBundleError, Entry, KernelRequest, NativeImplementation,
    NativeSpecialization,
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
    implementation: fn(BackendName) -> Result<Option<NativeImplementation>, CheckedBundleError>,
}

impl Kernel {
    /// The entry `E`.
    pub fn of<E: Entry>() -> Self {
        Self {
            name: E::NAME,
            implementation: seismic::generated::native_implementation_for_backend::<E>,
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

fn launch_error(message: impl Into<String>) -> AssessmentError {
    AssessmentError::Demand(message.into())
}

/// One prepared specialization of an entry: its declaration and the
/// default specialization at the statics the load prepares it with.
#[derive(Clone, Debug)]
struct Prepared {
    implementation: NativeImplementation,
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
            .map_err(|failure| {
                launch_error(crate::CatalogError::native(backend, failure).to_string())
            })?;
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

    /// Every specialization of `kernel` the load prepares.
    fn prepared(&self, kernel: Kernel) -> Result<Vec<Prepared>, AssessmentError> {
        let implementation = (kernel.implementation)(self.backend)
            .map_err(|error| launch_error(format!("`{}`: {error}", kernel.name)))?
            .ok_or_else(|| {
                launch_error(format!(
                    "`{}` has no native implementation for {}",
                    kernel.name,
                    self.backend.as_str()
                ))
            })?;
        let prepared = self
            .requests
            .iter()
            .filter(|request| request.entry == kernel.name)
            .map(|request| {
                let statics = request
                    .statics
                    .iter()
                    .fold(NativeSpecialization::new(), |statics, (name, value)| {
                        statics.with_static(name.clone(), *value)
                    });
                let specialization = implementation
                    .default_specialization(&statics)
                    .map_err(|error| launch_error(format!("{request}: {error}")))?;
                Ok(Prepared {
                    implementation: implementation.clone(),
                    specialization,
                })
            })
            .collect::<Result<Vec<_>, AssessmentError>>()?;
        if prepared.is_empty() {
            return Err(launch_error(format!(
                "the load prepares no `{}` on {}",
                kernel.name,
                self.backend.as_str()
            )));
        }
        Ok(prepared)
    }

    /// The launches of a call of `kernel` stating `dimensions`, at any row
    /// count.
    pub fn launches(
        &self,
        kernel: Kernel,
        dimensions: &[(&'static str, u64)],
    ) -> Result<Launches, AssessmentError> {
        Ok(Launches {
            kernel,
            prepared: self.prepared(kernel)?,
            dimensions: dimensions.to_vec(),
        })
    }
}

/// What a call dispatches as a function of its rows.
#[derive(Clone, Debug)]
pub struct Launches {
    kernel: Kernel,
    prepared: Vec<Prepared>,
    dimensions: Vec<(&'static str, u64)>,
}

impl PartialEq for Launches {
    fn eq(&self, other: &Self) -> bool {
        self.kernel == other.kernel && self.dimensions == other.dimensions
    }
}

impl Launches {
    /// The entry the call runs.
    pub fn kernel(&self) -> Kernel {
        self.kernel
    }

    /// The launches one call dispatches at `rows` rows. Every specialization
    /// of the entry the load prepares must dispatch as many: the walk names
    /// the entry, not which of its specializations a call takes.
    pub fn at(&self, rows: u64) -> Result<u64, AssessmentError> {
        let mut counts = self
            .prepared
            .iter()
            .map(|prepared| self.count(prepared, rows))
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

    fn count(&self, prepared: &Prepared, rows: u64) -> Result<u64, AssessmentError> {
        let statics = prepared.specialization.statics();
        let mut values = BTreeMap::<&str, u64>::new();
        for (name, value) in statics {
            values.insert(name.as_str(), *value);
        }
        for (name, value) in &self.dimensions {
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
            launches += u64::from(active);
        }
        Ok(launches)
    }
}
