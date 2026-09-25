//! The specialization each native entry is prepared with.
//!
//! A specialization fixes the entry's declared static dimensions to this
//! model's values and chooses its tuning parameters. An entry with tuning
//! parameters is tuned on the device when the program is prepared (see
//! `tuning`). Every entry has an implementation for every backend a device
//! can have: the kernels crate's build proves it, so preparation never meets
//! a missing one.

use super::tuning::{EntryTuning, Tuner};
use super::CatalogFailure;
use seismic::{Device, Entry, LoadError, NativeImplementation, NativeKernel, NativeSpecialization};

pub(super) struct Specializer<'a> {
    device: &'a Device,
    /// A tuning census walks the program to count tuning units; it forms
    /// nothing and every entry comes back `None`.
    census: bool,
}

fn failure(entry: &'static str, bindings: &str, outcome: String) -> CatalogFailure {
    CatalogFailure::Preparation {
        entry,
        bindings: bindings.to_owned(),
        outcome,
    }
}

/// The static values of `implementation`. `values` supplies this model's
/// value of every dimension the call site can fix; each dimension the
/// implementation declares static must be among them.
fn statics(
    implementation: &NativeImplementation,
    entry: &'static str,
    bindings: &str,
    values: &[(&str, u64)],
) -> Result<NativeSpecialization, CatalogFailure> {
    let mut specialization = NativeSpecialization::new();
    for name in &implementation.statics {
        let value = values
            .iter()
            .find(|(candidate, _)| candidate == name)
            .map(|(_, value)| *value)
            .ok_or_else(|| {
                failure(
                    entry,
                    bindings,
                    format!(
                        "the implementation declares `{name}` static, but the engine supplies no value for it"
                    ),
                )
            })?;
        specialization = specialization.with_static(name.clone(), value);
    }
    Ok(specialization)
}

impl<'a> Specializer<'a> {
    pub fn new(device: &'a Device) -> Self {
        Self { device, census: false }
    }

    /// A specializer for a tuning census (with [`Tuner::census`]).
    pub fn census(device: &'a Device) -> Self {
        Self { device, census: true }
    }

    fn implementation<E: Entry>(&self, bindings: &str) -> Result<NativeImplementation, CatalogFailure> {
        let implementation = seismic::generated::native_implementation::<E>(self.device)
            .map_err(|error| failure(E::NAME, bindings, error.to_string()))?;
        Ok(implementation.unwrap_or_else(|| {
            panic!(
                "`{}` has no {} implementation, which the kernels crate's build rules out",
                E::NAME,
                self.device.backend().as_str()
            )
        }))
    }

    /// Prepare an entry without tuning parameters; `None` during a census.
    pub fn fixed<E: Entry>(
        &mut self,
        bindings: &str,
        values: &[(&str, u64)],
        prepare: impl FnOnce(&NativeSpecialization) -> Result<NativeKernel<E>, LoadError>,
    ) -> Result<Option<NativeKernel<E>>, CatalogFailure> {
        if self.census {
            return Ok(None);
        }
        let implementation = self.implementation::<E>(bindings)?;
        if !implementation.params.is_empty() {
            return Err(failure(
                E::NAME,
                bindings,
                "the implementation declares tuning parameters, but its call site registers no tuning case".into(),
            ));
        }
        let specialization = statics(&implementation, E::NAME, bindings, values)?;
        prepare(&specialization)
            .map(Some)
            .map_err(|error| failure(E::NAME, bindings, error.to_string()))
    }

    /// Prepare an entry through its tuning case: static values from the
    /// case, parameters tuned on the device when the implementation declares
    /// any. `None` during a tuning census, which forms nothing.
    pub fn tuned<T: EntryTuning>(
        &mut self,
        tuner: &mut Tuner<'_>,
        case: &T,
    ) -> Result<Option<NativeKernel<T::Entry>>, CatalogFailure> {
        let entry = <T::Entry as Entry>::NAME;
        let bindings = case.bindings();
        let implementation = self.implementation::<T::Entry>(&bindings)?;
        let values = tuner.statics(case)?;
        let fixed = statics(&implementation, entry, &bindings, &values)?;
        let specialization = if implementation.params.is_empty() {
            fixed
        } else {
            tuner.tune(case, &implementation, &fixed)?
        };
        if self.census {
            return Ok(None);
        }
        case.prepare(self.device, &specialization)
            .map(Some)
            .map_err(|error| failure(entry, &bindings, error.to_string()))
    }
}
