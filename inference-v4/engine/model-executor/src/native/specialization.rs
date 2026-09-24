//! The specialization each native entry is prepared with.
//!
//! A specialization fixes the entry's declared static dimensions to this
//! model's values and chooses its tuning parameters. An entry with tuning
//! parameters is tuned on the device when the program is prepared: its call
//! site supplies the entry's workload points to the generated `native_tune`
//! and prepares the result's overall configuration.

use super::CatalogError;
use crate::ExecutionPath;
use seismic::{Device, Entry, NativeSpecialization};

fn failure(entry: &'static str, bindings: &str, outcome: String) -> CatalogError {
    CatalogError::Preparation {
        path: ExecutionPath::NativeMetal,
        entry,
        bindings: bindings.to_owned(),
        outcome,
    }
}

/// The static values of entry `E`'s native implementation for this device.
/// `statics` supplies this model's value of every dimension the call site
/// can fix; each dimension the implementation declares static must be among
/// them.
pub(crate) fn statics<E: Entry>(
    device: &Device,
    bindings: &str,
    statics: &[(&str, u64)],
) -> Result<NativeSpecialization, CatalogError> {
    let implementation = seismic::generated::native_implementation::<E>(device)
        .map_err(|error| failure(E::NAME, bindings, error.to_string()))?
        .ok_or_else(|| {
            failure(
                E::NAME,
                bindings,
                format!("no native implementation for `{}`", device.backend().as_str()),
            )
        })?;
    let mut specialization = NativeSpecialization::new();
    for name in &implementation.statics {
        let value = statics
            .iter()
            .find(|(candidate, _)| candidate == name)
            .map(|(_, value)| *value)
            .ok_or_else(|| {
                failure(
                    E::NAME,
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

/// The specialization of an entry without tuning parameters.
pub(crate) fn fixed<E: Entry>(
    device: &Device,
    bindings: &str,
    values: &[(&str, u64)],
) -> Result<NativeSpecialization, CatalogError> {
    let implementation = seismic::generated::native_implementation::<E>(device)
        .map_err(|error| failure(E::NAME, bindings, error.to_string()))?;
    if implementation.is_some_and(|implementation| !implementation.params.is_empty()) {
        return Err(failure(
            E::NAME,
            bindings,
            "the implementation declares tuning parameters; its call site must tune it".into(),
        ));
    }
    statics::<E>(device, bindings, values)
}
