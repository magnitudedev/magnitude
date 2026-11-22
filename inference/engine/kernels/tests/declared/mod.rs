//! Kernels prepared at their declarations' defaults.

/// `entry`'s configuration as `device` offers it at the entry's bound
/// elements: each static it declares taking `statics(name)` (a static the
/// backend leaves dynamic is never asked for), each parameter `pinned` names
/// held at its value, and every other parameter, and every form its `where`
/// reads, decided by the declaration's defaults ([`declared_at`]).
#[allow(dead_code)]
pub fn declared<E: seismic::Entry>(
    device: &seismic::Device,
    entry: &seismic::BoundEntry<E>,
    statics: impl Fn(&str) -> u64,
    pinned: &[(&str, u64)],
) -> Result<seismic::NativeSpecialization, seismic::NativeSpecializationError> {
    let implementation = offered(device, entry);
    let given = implementation
        .statics
        .iter()
        .fold(seismic::NativeSpecialization::new(), |given, name| {
            given.with_static(name.clone(), statics(name))
        });
    let given = pinned.iter().fold(given, |given, (name, value)| given.with_param(*name, *value));
    declared_at(device, entry, &given)
}

/// `entry`'s configuration at `given`, a test's specialization, as
/// preparation completes one: the declaration's default walk (each
/// parameter trying its declared default first, as a launch of the product
/// takes it) at the statics the backend declares, read from `given`, with
/// every parameter `given` values, entry-wide or a launch's, held at that
/// value. A test so names only the statics and the parameters it exercises,
/// and a parameter a declaration gains needs no test to name it. A static
/// the backend declares that `given` lacks, or a parameter `given` values
/// that the declaration lacks, fails by name; a value outside its
/// parameter's domain, or a configuration no admissible one holds, is the
/// declaration's error.
#[allow(dead_code)]
pub fn declared_at<E: seismic::Entry>(
    device: &seismic::Device,
    entry: &seismic::BoundEntry<E>,
    given: &seismic::NativeSpecialization,
) -> Result<seismic::NativeSpecialization, seismic::NativeSpecializationError> {
    use seismic_lang::checked::{NativeParameter, NativeSpecializationError};
    let implementation = offered(device, entry);
    let statics = implementation.statics.iter().fold(seismic::NativeSpecialization::new(), |statics, name| {
        let value = given
            .static_value(name)
            .unwrap_or_else(|| panic!("{} declares `{name}` static, which the case does not give", E::NAME));
        statics.with_static(name.clone(), value)
    });
    for (name, _) in given.params() {
        assert!(
            implementation.params.iter().any(|parameter| parameter.name == *name),
            "{} declares no parameter `{name}` on {:?}",
            E::NAME,
            device.backend()
        );
    }
    for ((launch, name), _) in given.launch_params() {
        assert!(
            implementation
                .launches
                .get(*launch)
                .is_some_and(|declaration| declaration.params.iter().any(|parameter| parameter.name == *name)),
            "{} declares no parameter `{name}` on launch {launch} on {:?}",
            E::NAME,
            device.backend()
        );
    }
    let hold = |parameter: &mut NativeParameter, value: Option<u64>| -> Result<(), NativeSpecializationError> {
        let Some(value) = value else { return Ok(()) };
        if !parameter.values.contains(&value) {
            return Err(NativeSpecializationError::OutsideDomain { parameter: parameter.name.clone(), value });
        }
        parameter.fix(value);
        Ok(())
    };
    implementation
        .adjusted(|implementation| {
            for parameter in &mut implementation.params {
                hold(parameter, given.param(&parameter.name))?;
            }
            for (launch, declaration) in implementation.launches.iter_mut().enumerate() {
                for parameter in &mut declaration.params {
                    hold(parameter, given.launch_param(launch, &parameter.name))?;
                }
            }
            Ok(())
        })?
        .default_specialization(&statics)
}

/// `entry` prepared on `device` at `given` as [`declared_at`] completes it:
/// what a test that names only the parameters it exercises launches. A
/// configuration the declaration refuses is prepared with every parameter
/// `given` leaves unset at its declared default, which preparation refuses
/// with its own error, so a test of a refusal sees what a launch would.
#[allow(dead_code)]
pub fn prepared<E: seismic::Entry>(
    device: &seismic::Device,
    entry: &seismic::BoundEntry<E>,
    given: &seismic::NativeSpecialization,
) -> Result<seismic::NativeKernel<E>, seismic::LoadError> {
    let specialization = declared_at(device, entry, given).unwrap_or_else(|_| {
        let implementation = offered(device, entry);
        let defaults = implementation
            .parameter_defaults(&|name| given.static_value(name))
            .unwrap_or_else(|error| panic!("{} on {:?}: {error}", E::NAME, device.backend()));
        let parameters = implementation
            .params
            .iter()
            .map(|parameter| (None, &parameter.name))
            .chain(implementation.launches.iter().enumerate().flat_map(|(launch, declaration)| {
                declaration.params.iter().map(move |parameter| (Some(launch), &parameter.name))
            }));
        parameters.zip(defaults).fold(given.clone(), |given, ((launch, name), default)| match launch {
            None if given.param(name).is_none() => given.with_param(name.clone(), default),
            Some(launch) if given.launch_param(launch, name).is_none() => {
                given.with_launch_param(launch, name.clone(), default)
            }
            _ => given,
        })
    });
    entry.prepare(device, &specialization)
}

/// `entry`'s implementation as `device` offers it at its bound elements.
fn offered<E: seismic::Entry>(device: &seismic::Device, entry: &seismic::BoundEntry<E>) -> seismic::OnDevice {
    entry
        .offered(device)
        .unwrap()
        .expect("the entry has a native implementation at its elements")
}

/// The decode row classes of `native`: its `alike` launches, each with the
/// launch it computes each row as alone (`alike (LAUNCH)`), whose reduction
/// orders the checker holds both to read alike (`derive_codes`), so a row's
/// bits do not depend on how many rows its call carries. Pairs are
/// `(launch, reference launch)` ordinals, in launch order.
#[allow(dead_code)]
pub fn row_classes(native: &seismic::NativeImplementation) -> Vec<(usize, usize)> {
    native
        .launches
        .iter()
        .enumerate()
        .filter_map(|(ordinal, launch)| launch.alike.map(|alone| (ordinal, alone)))
        .collect()
}

/// Every launch of the decode row classes of `native`: the `alike` launches
/// ([`row_classes`]) and the launches they name, in launch order.
#[allow(dead_code)]
pub fn row_launches(native: &seismic::NativeImplementation) -> Vec<usize> {
    let classes = row_classes(native);
    (0..native.launches.len())
        .filter(|ordinal| classes.iter().any(|(alike, alone)| alike == ordinal || alone == ordinal))
        .collect()
}
