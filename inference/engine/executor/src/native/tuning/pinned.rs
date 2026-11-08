//! Tuning pin: a development-only measurement tool, which only the harness
//! installs; a served engine installs none, and with none installed tuning
//! runs as it would without it (one lock a tuned unit, at load).
//!
//! Tuning runs at every load and may choose different configurations between
//! runs, so two runs are bit-comparable only when every entry runs the same
//! configuration. A process that installs [`TuningPin::Record`] tunes each
//! entry (keyed by entry, element bindings and static values) once and reuses
//! that choice for every later load in the process; [`recorded`] returns the
//! choices. A process that installs [`TuningPin::Replay`] with those choices
//! runs exactly them and tunes nothing. A unit without a recorded choice (a
//! shape the recording run never prepared) runs its declared default
//! configuration, untuned, and is reported as not replayed, and so does one
//! whose recorded configuration names parameters this build does not declare
//! ([`TuningEvent::Pinned`](super::TuningEvent::Pinned) with
//! [`PinnedChoice::NotReplayed`]), so its timing is never mistaken for the
//! recorded choice's.
//!
//! A pin that replays or runs defaults tunes nothing: a load under it asks
//! for no search ([`tunes_nothing`]).
//!
//! Only `magnitude-harness` installs a pin (`--tuning-record` /
//! `--tuning-replay` / `--tuning-defaults`). The pin is process-global so
//! that no production API carries it; nothing acts on it unless one was
//! installed.

use super::{PinnedChoice, TuningKey};
use seismic::{NativeSpecialization, OnDevice};
use std::collections::BTreeMap;
use std::sync::Mutex;

/// The configuration chosen for one entry at one set of bindings and static
/// values.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PinnedConfiguration {
    pub entry: String,
    pub bindings: String,
    pub statics: BTreeMap<String, u64>,
    pub params: BTreeMap<String, u64>,
    /// Launch-scoped parameters, by launch ordinal and name.
    pub launch_params: BTreeMap<(usize, String), u64>,
}

pub enum TuningPin {
    /// Tune each entry once per process and record the choice, on top of
    /// these configurations, which an earlier load of another shape
    /// recorded.
    Record(Vec<PinnedConfiguration>),
    /// Run exactly these configurations; tune nothing.
    Replay(Vec<PinnedConfiguration>),
    /// Run every entry's declared default configuration; tune nothing.
    Defaults,
}

struct Installed {
    replay: bool,
    defaults: bool,
    configurations: Vec<PinnedConfiguration>,
}

static PIN: Mutex<Option<Installed>> = Mutex::new(None);

impl Installed {
    fn of(pin: TuningPin) -> Self {
        match pin {
            TuningPin::Record(configurations) => Self {
                replay: false,
                defaults: false,
                configurations,
            },
            TuningPin::Replay(configurations) => Self {
                replay: true,
                defaults: false,
                configurations,
            },
            TuningPin::Defaults => Self {
                replay: true,
                defaults: true,
                configurations: Vec::new(),
            },
        }
    }
}

/// Install the pin for every later program preparation in this process.
pub fn install(pin: TuningPin) {
    *PIN.lock().expect("tuning pin lock poisoned") = Some(Installed::of(pin));
}

/// The configurations recorded (or replayed) so far.
pub fn recorded() -> Vec<PinnedConfiguration> {
    PIN.lock()
        .expect("tuning pin lock poisoned")
        .as_ref()
        .map_or_else(Vec::new, |installed| installed.configurations.clone())
}

/// Whether the installed pin runs what it holds and tunes nothing: a load
/// under it has no use for a search.
pub fn tunes_nothing() -> bool {
    PIN.lock()
        .expect("tuning pin lock poisoned")
        .as_ref()
        .is_some_and(|installed| installed.replay)
}

/// What the pin decides for one tuning key.
pub(super) enum Pinned {
    /// No pin is installed, or recording has not seen the key: tune.
    Tune,
    /// Run this configuration, untuned, chosen as the pin says.
    Chosen(NativeSpecialization, PinnedChoice),
}

pub(super) fn lookup(
    key: &TuningKey,
    implementation: &OnDevice,
) -> Result<Pinned, String> {
    let guard = PIN.lock().expect("tuning pin lock poisoned");
    let Some(installed) = guard.as_ref() else {
        return Ok(Pinned::Tune);
    };
    decide(installed, key, implementation)
}

/// What `installed` decides for one tuning key.
fn decide(
    installed: &Installed,
    key: &TuningKey,
    implementation: &OnDevice,
) -> Result<Pinned, String> {
    let (entry, bindings, fixed) = key;
    // The unit's statics and the parameters its call site fixes, which
    // every choice keeps (a dependant's arithmetic, `Tuner::following`).
    let at = fixed.specialization();
    if installed.defaults {
        return implementation
            .default_specialization(&at)
            .map(|chosen| Pinned::Chosen(chosen, PinnedChoice::Defaults))
            .map_err(|error| format!("default configuration of {entry}: {error:?}"));
    }
    let Some(pinned) = installed.configurations.iter().find(|pinned| {
        pinned.entry == *entry && pinned.bindings == *bindings && pinned.statics == fixed.statics
    }) else {
        if !installed.replay {
            return Ok(Pinned::Tune);
        }
        // A unit the recording run never prepared (a shape first seen in this
        // run) runs its declared default, untuned, and is reported as not
        // replayed: a replay never tunes.
        return implementation
            .default_specialization(&at)
            .map(|chosen| Pinned::Chosen(chosen, PinnedChoice::NotReplayed))
            .map_err(|error| format!("default configuration of {entry}: {error:?}"));
    };
    // A parameter the pin does not name (one the implementation gained after
    // the pin was recorded) takes its value in the default configuration at
    // these values, so another build's configurations replay unchanged; one
    // the call site fixes keeps its fixed value. A value applies wherever
    // this build declares its parameter: one recorded on the native, on
    // every launch that declares it, and one recorded on launches, on the
    // native, so a parameter moved between the native and its launches
    // replays its recorded value.
    let defaults = implementation
        .default_specialization(&at)
        .map_err(|error| format!("default configuration of {entry}: {error}"))?;
    let declared = |name: &String| {
        implementation
            .params
            .iter()
            .any(|parameter| &parameter.name == name)
    };
    let declared_in = |launch: usize, name: &String| {
        implementation
            .launches
            .get(launch)
            .is_some_and(|declaration| {
                declaration
                    .params
                    .iter()
                    .any(|parameter| &parameter.name == name)
            })
    };
    let declared_on_launches = |name: &String| {
        (0..implementation.launches.len()).any(|launch| declared_in(launch, name))
    };
    // The one value the pin's launches recorded for `name`, if they agree.
    let on_launches = |name: &str| {
        let mut values = pinned
            .launch_params
            .iter()
            .filter(|((_, recorded), _)| recorded == name)
            .map(|(_, &value)| value);
        let first = values.next()?;
        values.all(|value| value == first).then_some(first)
    };
    // A configuration of another build's parameters (one the pin names that
    // this implementation declares nowhere, a launch's this one declares on
    // other launches only, or launches' values that disagree where this one
    // takes one) is a unit the recording never prepared: it runs its
    // default, reported as not replayed.
    let foreign = pinned
        .params
        .keys()
        .any(|name| !declared(name) && !declared_on_launches(name))
        || pinned.launch_params.keys().any(|(launch, name)| {
            !declared_in(*launch, name) && !(declared(name) && on_launches(name).is_some())
        });
    if foreign {
        return Ok(Pinned::Chosen(defaults, PinnedChoice::NotReplayed));
    }
    let mut specialization = fixed.statics.iter().fold(NativeSpecialization::new(), |spec, (name, value)| {
        spec.with_static(name.clone(), *value)
    });
    for parameter in &implementation.params {
        let value = fixed
            .params
            .get(&parameter.name)
            .or_else(|| pinned.params.get(&parameter.name))
            .copied()
            .or_else(|| on_launches(&parameter.name))
            .or_else(|| defaults.param(&parameter.name))
            .expect("the default configuration values every parameter");
        specialization = specialization.with_param(parameter.name.clone(), value);
    }
    for (launch, declaration) in implementation.launches.iter().enumerate() {
        for parameter in &declaration.params {
            let value = pinned
                .launch_params
                .get(&(launch, parameter.name.clone()))
                .or_else(|| pinned.params.get(&parameter.name))
                .copied()
                .or_else(|| defaults.launch_param(launch, &parameter.name))
                .expect("the default configuration values every launch parameter");
            specialization =
                specialization.with_launch_param(launch, parameter.name.clone(), value);
        }
    }
    implementation
        .validate(&specialization)
        .map_err(|error| format!("pinned configuration {:?}: {error}", pinned.params))?;
    Ok(Pinned::Chosen(specialization, PinnedChoice::Replayed))
}

/// Record a tuned choice when recording.
pub(super) fn record(key: &TuningKey, chosen: &NativeSpecialization) {
    let mut guard = PIN.lock().expect("tuning pin lock poisoned");
    let Some(installed) = guard.as_mut().filter(|installed| !installed.replay) else {
        return;
    };
    let (entry, bindings, fixed) = key;
    installed.configurations.push(PinnedConfiguration {
        entry: (*entry).to_owned(),
        bindings: bindings.clone(),
        statics: fixed.statics.clone(),
        params: chosen.params().clone(),
        launch_params: chosen.launch_params().clone(),
    });
}

#[cfg(test)]
mod tests {
    use super::super::FixedValues;
    use super::*;
    use seismic::{BackendName, NativeImplementation};

    /// A unit's key of `statics` and the parameters its call site fixes.
    fn key(entry: &'static str, statics: &[(&str, u64)], params: &[(&str, u64)]) -> TuningKey {
        let named = |values: &[(&str, u64)]| values.iter().map(|(name, value)| ((*name).to_owned(), *value)).collect();
        (entry, "test".to_owned(), FixedValues { statics: named(statics), params: named(params) })
    }

    /// A unit whose call site fixes a parameter resolves under every pin at
    /// that value: a resident `attention_project` at `RESIDUE` 1, and a
    /// streamed route at its resident's `LANES` 16 (away from its default),
    /// under the declared defaults, a replay of a configuration recorded at
    /// other values, and a replay that recorded none of it. The pinned
    /// configuration keeps its statics alone. Each implementation is the
    /// one tuning resolves, at its element bindings (a declaration whose
    /// `where` reads its weights' forms, `form(QW, PACK)`, admits nothing
    /// unbound), keyed at statics its own search admits, so the test
    /// follows its declaration's statics.
    #[test]
    fn a_unit_fixed_at_a_parameter_resolves_under_every_pin() {
        // What a decision was, where it was not the one expected.
        let what = |decided: &Result<Pinned, String>| match decided {
            Err(error) => error.clone(),
            Ok(Pinned::Tune) => "tune".to_owned(),
            Ok(Pinned::Chosen(_, choice)) => format!("{choice:?}"),
        };
        let check = |implementation: NativeImplementation, key: TuningKey, (name, value): (&str, u64), other: u64| {
            let implementation = implementation
                .device_independent()
                .expect("the declaration reads no device property")
                .owned();
            let defaults = Installed::of(TuningPin::Defaults);
            let decided = decide(&defaults, &key, &implementation);
            let Ok(Pinned::Chosen(chosen, PinnedChoice::Defaults)) = decided else {
                panic!("{} resolves under the defaults: {}", key.0, what(&decided));
            };
            assert_eq!(chosen.param(name), Some(value));
            assert_eq!(chosen.statics(), &key.2.statics);
            let recorded = PinnedConfiguration {
                entry: key.0.to_owned(),
                bindings: key.1.clone(),
                statics: key.2.statics.clone(),
                params: chosen.params().iter().map(|(param, &v)| (param.clone(), if param == name { other } else { v })).collect(),
                launch_params: chosen.launch_params().clone(),
            };
            let replay = Installed::of(TuningPin::Replay(vec![recorded]));
            let decided = decide(&replay, &key, &implementation);
            let Ok(Pinned::Chosen(replayed, PinnedChoice::Replayed)) = decided else {
                panic!("{} replays its recorded configuration: {}", key.0, what(&decided));
            };
            assert_eq!(replayed.param(name), Some(value), "the fixed value is kept");
            let absent = Installed::of(TuningPin::Replay(Vec::new()));
            let decided = decide(&absent, &key, &implementation);
            let Ok(Pinned::Chosen(unrecorded, PinnedChoice::NotReplayed)) = decided else {
                panic!("{} runs its default where nothing was recorded: {}", key.0, what(&decided));
            };
            assert_eq!(unrecorded.param(name), Some(value));
            // Another build's configuration, of a parameter this one does not
            // declare, runs the default, as unrecorded.
            let foreign = PinnedConfiguration {
                entry: key.0.to_owned(),
                bindings: key.1.clone(),
                statics: key.2.statics.clone(),
                params: [("ANOTHER_BUILDS".to_owned(), 1)].into(),
                launch_params: BTreeMap::new(),
            };
            let decided = decide(&Installed::of(TuningPin::Replay(vec![foreign])), &key, &implementation);
            let Ok(Pinned::Chosen(_, PinnedChoice::NotReplayed)) = decided else {
                panic!("{} runs its default where another build's configuration was recorded: {}", key.0, what(&decided));
            };
        };
        let metal = |implementation: Result<Option<&NativeImplementation>, _>| implementation.unwrap().unwrap().clone();
        // A unit's key at the first statics its implementation admits (its
        // own search, so the key follows the declaration's statics) and
        // the parameter its call site fixes.
        let at = |entry: &'static str, implementation: &NativeImplementation, fixed: (&str, u64)| {
            let statics = implementation.search_statics(&[]).unwrap().expect("the implementation admits some statics");
            let statics = statics.statics().iter().map(|(name, &value)| (name.as_str(), value)).collect::<Vec<_>>();
            key(entry, &statics, &[fixed])
        };
        let (weights, activation) = (seismic::Element::bf16(), seismic::Element::f32());
        let projection = metal(seismic::generated::native_implementation_at::<magnitude_kernels::attention_project::Entry>(
            BackendName::Metal,
            &[("NW", activation), ("QW", weights), ("GW", weights), ("KW", weights), ("VW", weights), ("A", activation)],
        ));
        let key_of = at("attention_project", &projection, ("RESIDUE", 1));
        check(projection, key_of, ("RESIDUE", 1), 2);
        let route = metal(seismic::generated::native_implementation_at::<magnitude_kernels::routed_route_streamed::Entry>(
            BackendName::Metal,
            &[("NW", activation), ("RW", weights), ("SGW", weights), ("SUW", weights), ("A", activation)],
        ));
        let key_of = at("routed_route_streamed", &route, ("LANES", 16));
        check(route, key_of, ("LANES", 16), 32);
    }

    /// A configuration recorded before a parameter moved between a native
    /// and its launches replays its values where this build declares them:
    /// Metal `routed_gate_up`'s parameters, on its native, from a recording
    /// on its first launch, and `routed_expand`'s, on its launch, from one
    /// on its native. Launches that recorded different values for what is
    /// now one native parameter are another build's configuration.
    #[test]
    fn a_parameter_moved_between_native_and_launches_replays_its_value() {
        let bf16 = seismic::Element::bf16();
        let device_independent = |implementation: Result<Option<&NativeImplementation>, _>| {
            implementation
                .unwrap()
                .unwrap()
                .clone()
                .device_independent()
                .expect("the declaration reads no device property")
                .owned()
        };
        let unit = |entry: &'static str, implementation: &OnDevice| {
            let statics = implementation.search_statics(&[]).unwrap().expect("the implementation admits some statics");
            let statics = statics.statics().iter().map(|(name, &value)| (name.as_str(), value)).collect::<Vec<_>>();
            key(entry, &statics, &[])
        };
        let recorded = |key: &TuningKey, params: &[(&str, u64)], launch_params: &[(usize, &str, u64)]| PinnedConfiguration {
            entry: key.0.to_owned(),
            bindings: key.1.clone(),
            statics: key.2.statics.clone(),
            params: params.iter().map(|(name, value)| ((*name).to_owned(), *value)).collect(),
            launch_params: launch_params.iter().map(|(launch, name, value)| ((*launch, (*name).to_owned()), *value)).collect(),
        };
        let replayed = |key: &TuningKey, implementation: &OnDevice, configuration: PinnedConfiguration| {
            match decide(&Installed::of(TuningPin::Replay(vec![configuration])), key, implementation) {
                Ok(Pinned::Chosen(chosen, choice)) => (chosen, choice),
                Ok(Pinned::Tune) => panic!("{} replays, never tunes", key.0),
                Err(error) => panic!("{} resolves: {error}", key.0),
            }
        };

        let gate_up = device_independent(seismic::generated::native_implementation_at::<magnitude_kernels::routed_gate_up::Entry>(
            BackendName::Metal,
            &[("A", bf16), ("EGW", bf16), ("EUW", bf16)],
        ));
        assert!(gate_up.launches.iter().all(|launch| launch.params.is_empty()), "routed_gate_up's parameters are its native's");
        let key_of = unit("routed_gate_up", &gate_up);
        let (chosen, choice) = replayed(
            &key_of,
            &gate_up,
            recorded(&key_of, &[], &[(0, "SIMDGROUPS", 4), (0, "ROWS", 2), (0, "LANES", 32)]),
        );
        assert_eq!(choice, PinnedChoice::Replayed);
        assert_eq!(
            [chosen.param("SIMDGROUPS"), chosen.param("ROWS"), chosen.param("LANES")],
            [Some(4), Some(2), Some(32)]
        );
        let (_, choice) = replayed(
            &key_of,
            &gate_up,
            recorded(&key_of, &[], &[(0, "SIMDGROUPS", 4), (1, "SIMDGROUPS", 8)]),
        );
        assert_eq!(choice, PinnedChoice::NotReplayed, "launches' values that disagree name no one native value");

        let expand = device_independent(seismic::generated::native_implementation_at::<magnitude_kernels::routed_expand::Entry>(
            BackendName::Metal,
            &[("A", bf16), ("EGW", bf16), ("EUW", bf16), ("SGW", bf16), ("SUW", bf16)],
        ));
        assert!(expand.params.is_empty(), "routed_expand's parameters are its launch's");
        let key_of = unit("routed_expand", &expand);
        let (chosen, choice) = replayed(
            &key_of,
            &expand,
            recorded(&key_of, &[("SIMDGROUPS", 4), ("ROWS", 2), ("LANES", 16)], &[]),
        );
        assert_eq!(choice, PinnedChoice::Replayed);
        assert_eq!(
            [chosen.launch_param(0, "SIMDGROUPS"), chosen.launch_param(0, "ROWS"), chosen.launch_param(0, "LANES")],
            [Some(4), Some(2), Some(16)]
        );
    }
}
