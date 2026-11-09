//! Kernels prepared at their declarations' defaults.

/// `entry`'s configuration as `device` offers it at the entry's bound
/// elements: each static it declares taking `statics(name)` (a static the
/// backend leaves dynamic is never asked for), each parameter `pinned` names
/// held at its value, and every other parameter, and every form its `where`
/// reads, decided by the declaration's defaults. A test so names only the
/// statics and the parameters it pins, and a parameter a declaration gains
/// needs no test to name it. A pinned parameter the declaration lacks fails
/// by name; a configuration the declaration refuses is its error.
#[allow(dead_code)]
pub fn declared<E: seismic::Entry>(
    device: &seismic::Device,
    entry: &seismic::BoundEntry<E>,
    statics: impl Fn(&str) -> u64,
    pinned: &[(&str, u64)],
) -> Result<seismic::NativeSpecialization, seismic::NativeSpecializationError> {
    let implementation = entry
        .offered(device)
        .unwrap()
        .expect("the entry has a native implementation at its elements");
    let fixed = implementation
        .statics
        .iter()
        .fold(seismic::NativeSpecialization::new(), |fixed, name| {
            fixed.with_static(name.clone(), statics(name))
        });
    let fixed = pinned.iter().fold(fixed, |fixed, (name, value)| {
        assert!(
            implementation.params.iter().any(|parameter| parameter.name == *name),
            "{} declares no parameter `{name}` on {:?}",
            E::NAME,
            device.backend()
        );
        fixed.with_param(*name, *value)
    });
    implementation.default_specialization(&fixed)
}

/// [`declared`] at the statics and parameters of `given`, a test's
/// specialization: a static the backend declares that `given` has no value
/// for fails by name.
#[allow(dead_code)]
pub fn declared_at<E: seismic::Entry>(
    device: &seismic::Device,
    entry: &seismic::BoundEntry<E>,
    given: &seismic::NativeSpecialization,
) -> Result<seismic::NativeSpecialization, seismic::NativeSpecializationError> {
    let pinned = given
        .params()
        .iter()
        .map(|(name, value)| (name.as_str(), *value))
        .collect::<Vec<_>>();
    declared(
        device,
        entry,
        |name| {
            given
                .static_value(name)
                .unwrap_or_else(|| panic!("{} declares `{name}` static, which the case does not give", E::NAME))
        },
        &pinned,
    )
}
