//! Kernels prepared at their declarations' defaults.

/// `entry`'s default configuration as `device` offers it at the entry's bound
/// elements, each static taking `statics(name)`: the declaration decides
/// every parameter and every form its `where` reads, so a test names only the
/// statics and the parameters it pins on top.
pub fn declared_default<E: seismic::Entry>(
    device: &seismic::Device,
    entry: &seismic::BoundEntry<E>,
    statics: impl Fn(&str) -> u64,
) -> seismic::NativeSpecialization {
    let implementation = entry
        .offered(device)
        .unwrap()
        .expect("the entry has a native implementation at its elements");
    let statics = implementation
        .statics
        .iter()
        .fold(seismic::NativeSpecialization::new(), |fixed, name| {
            fixed.with_static(name.clone(), statics(name))
        });
    implementation.default_specialization(&statics).unwrap()
}
