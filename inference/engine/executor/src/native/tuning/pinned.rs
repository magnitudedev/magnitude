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
//! there ([`TuningEvent::Pinned`](super::TuningEvent::Pinned) with
//! [`PinnedChoice::NotReplayed`]), so its timing is never mistaken for the
//! recorded choice's.
//!
//! A pin that replays or runs defaults tunes nothing: a load under it asks
//! for no search ([`tunes_nothing`]). A composed recipe is one more pinned
//! unit: [`recipe`] decides it, and whatever a search stored for it is not
//! read, so what a pinned load runs depends on the pin alone.
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
    #[cfg(test)]
    if THREAD_DEFAULTS.get() {
        return decide(&Installed::of(TuningPin::Defaults), key, implementation);
    }
    let guard = PIN.lock().expect("tuning pin lock poisoned");
    let Some(installed) = guard.as_ref() else {
        return Ok(Pinned::Tune);
    };
    decide(installed, key, implementation)
}

#[cfg(test)]
thread_local! {
    /// Tests share one process: a test pins the preparations of its own
    /// thread, never the process.
    static THREAD_DEFAULTS: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// [`TuningPin::Defaults`] for the tuners of the calling thread, while the
/// returned guard lives.
#[cfg(test)]
pub(super) fn defaults_on_this_thread() -> impl Drop {
    struct Unpin;
    impl Drop for Unpin {
        fn drop(&mut self) {
            THREAD_DEFAULTS.set(false);
        }
    }
    THREAD_DEFAULTS.set(true);
    Unpin
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
    // the call site fixes keeps its fixed value.
    let defaults = implementation
        .default_specialization(&at)
        .map_err(|error| format!("default configuration of {entry}: {error}"))?;
    let mut specialization = fixed.statics.iter().fold(NativeSpecialization::new(), |spec, (name, value)| {
        spec.with_static(name.clone(), *value)
    });
    for parameter in &implementation.params {
        let value = fixed
            .params
            .get(&parameter.name)
            .or_else(|| pinned.params.get(&parameter.name))
            .copied()
            .or_else(|| defaults.param(&parameter.name))
            .expect("the default configuration values every parameter");
        specialization = specialization.with_param(parameter.name.clone(), value);
    }
    for (launch, declaration) in implementation.launches.iter().enumerate() {
        for parameter in &declaration.params {
            let value = pinned
                .launch_params
                .get(&(launch, parameter.name.clone()))
                .copied()
                .or_else(|| defaults.launch_param(launch, &parameter.name))
                .expect("the default configuration values every launch parameter");
            specialization =
                specialization.with_launch_param(launch, parameter.name.clone(), value);
        }
    }
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
    // A configuration of another build's parameters (one the pin names that
    // this implementation does not declare where it was recorded) is a unit
    // the recording never prepared: it runs its default, reported as not
    // replayed.
    let foreign = pinned.params.keys().any(|name| !declared(name))
        || pinned.launch_params.keys().any(|(launch, name)| !declared_in(*launch, name));
    if foreign {
        return Ok(Pinned::Chosen(defaults, PinnedChoice::NotReplayed));
    }
    implementation
        .validate(&specialization)
        .map_err(|error| format!("pinned configuration {:?}: {error}", pinned.params))?;
    Ok(Pinned::Chosen(specialization, PinnedChoice::Replayed))
}

/// The parameter a composed recipe's pinned configuration holds: whether
/// the recipe was selected over its canonical one.
const SELECTED: &str = "selected";

/// A composed recipe's choice under the pin, and where it came from; none
/// when no pin decides it (recording, or no pin). `identity` names the
/// recipe across builds and devices, which its stored key does not.
pub(super) fn recipe(label: &str, identity: &str) -> Option<(bool, PinnedChoice)> {
    #[cfg(test)]
    if THREAD_DEFAULTS.get() {
        return Some((false, PinnedChoice::Defaults));
    }
    let guard = PIN.lock().expect("tuning pin lock poisoned");
    decide_recipe(guard.as_ref()?, label, identity)
}

/// What `installed` decides for one composed recipe.
fn decide_recipe(installed: &Installed, label: &str, identity: &str) -> Option<(bool, PinnedChoice)> {
    if installed.defaults {
        return Some((false, PinnedChoice::Defaults));
    }
    let pinned = installed
        .configurations
        .iter()
        .find(|pinned| pinned.entry == label && pinned.bindings == identity);
    match pinned {
        Some(pinned) => Some((
            pinned.params.get(SELECTED).is_some_and(|selected| *selected != 0),
            PinnedChoice::Replayed,
        )),
        None => installed.replay.then_some((false, PinnedChoice::NotReplayed)),
    }
}

/// `selected` as the configuration a pin states a composed recipe's choice
/// by.
pub(super) fn recipe_configuration(selected: bool) -> NativeSpecialization {
    NativeSpecialization::new().with_param(SELECTED, u64::from(selected))
}

/// Record a composed recipe's stored choice when recording.
pub(super) fn record_recipe(label: &str, identity: &str, selected: bool) {
    let mut guard = PIN.lock().expect("tuning pin lock poisoned");
    let Some(installed) = guard.as_mut().filter(|installed| !installed.replay) else {
        return;
    };
    let recorded = |pinned: &PinnedConfiguration| pinned.entry == label && pinned.bindings == identity;
    if !installed.configurations.iter().any(recorded) {
        installed.configurations.push(PinnedConfiguration {
            entry: label.to_owned(),
            bindings: identity.to_owned(),
            statics: BTreeMap::new(),
            params: BTreeMap::from([(SELECTED.to_owned(), u64::from(selected))]),
            launch_params: BTreeMap::new(),
        });
    }
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

    /// A composed recipe's choice is the pin's: recorded, it replays
    /// selected; the defaults run the canonical recipe whatever is recorded;
    /// a replay that lacks it runs the canonical recipe and says so; a
    /// recording decides nothing until it has seen it.
    #[test]
    fn a_composed_recipe_is_decided_by_the_pin() {
        let recorded = vec![PinnedConfiguration {
            entry: "recipe".into(),
            bindings: "rows=1".into(),
            statics: BTreeMap::new(),
            params: BTreeMap::from([(SELECTED.to_owned(), 1)]),
            launch_params: BTreeMap::new(),
        }];
        let replay = Installed::of(TuningPin::Replay(recorded.clone()));
        assert_eq!(decide_recipe(&replay, "recipe", "rows=1"), Some((true, PinnedChoice::Replayed)));
        assert_eq!(decide_recipe(&replay, "recipe", "rows=2"), Some((false, PinnedChoice::NotReplayed)));
        let defaults = Installed::of(TuningPin::Defaults);
        assert_eq!(decide_recipe(&defaults, "recipe", "rows=1"), Some((false, PinnedChoice::Defaults)));
        let recording = Installed::of(TuningPin::Record(recorded));
        assert_eq!(decide_recipe(&recording, "recipe", "rows=1"), Some((true, PinnedChoice::Replayed)));
        assert_eq!(decide_recipe(&recording, "recipe", "rows=2"), None);
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
}
