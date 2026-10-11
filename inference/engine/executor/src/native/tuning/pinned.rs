//! Tuning pin: a development-only measurement tool (cargo feature
//! `pinned-tuning`), never part of a served engine.
//!
//! Tuning runs at every load and may choose different configurations between
//! runs, so two runs are bit-comparable only when every entry runs the same
//! configuration. A process that installs [`TuningPin::Record`] tunes each
//! entry (keyed by entry, element bindings and static values) once and reuses
//! that choice for every later load in the process; [`recorded`] returns the
//! choices. A process that installs [`TuningPin::Replay`] with those choices
//! runs exactly them and tunes nothing. A unit without a recorded choice (a
//! shape the recording run never prepared) runs its declared default
//! configuration, untuned, and is reported as not replayed
//! ([`TuningEvent::Pinned`](super::TuningEvent::Pinned) with
//! [`PinnedChoice::NotReplayed`]), so its timing is never mistaken for the
//! recorded choice's.
//!
//! Only `magnitude-harness` enables the feature (`--tuning-record` /
//! `--tuning-replay` / `--tuning-defaults`). The pin is process-global so that no production API
//! carries it; nothing reads it unless a pin was installed.

use super::{PinnedChoice, TuningKey};
use seismic::{NativeImplementation, NativeSpecialization};
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
    /// Tune each entry once per process and record the choice.
    Record,
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
            TuningPin::Record => Self {
                replay: false,
                defaults: false,
                configurations: Vec::new(),
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

/// What the pin decides for one tuning key.
pub(super) enum Pinned {
    /// No pin is installed, or recording has not seen the key: tune.
    Tune,
    /// Run this configuration, untuned, chosen as the pin says.
    Chosen(NativeSpecialization, PinnedChoice),
}

pub(super) fn lookup(
    key: &TuningKey,
    implementation: &NativeImplementation,
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
    implementation: &NativeImplementation,
) -> Result<Pinned, String> {
    let (entry, bindings, statics) = key;
    if installed.defaults {
        let mut fixed = NativeSpecialization::new();
        for (name, value) in statics {
            fixed = fixed.with_static(name.clone(), *value);
        }
        return implementation
            .default_specialization(&fixed)
            .map(|chosen| Pinned::Chosen(chosen, PinnedChoice::Defaults))
            .map_err(|error| format!("default configuration of {entry}: {error:?}"));
    }
    let Some(pinned) = installed.configurations.iter().find(|pinned| {
        pinned.entry == *entry && pinned.bindings == *bindings && pinned.statics == *statics
    }) else {
        if !installed.replay {
            return Ok(Pinned::Tune);
        }
        // A unit the recording run never prepared (a shape first seen in this
        // run) runs its declared default, untuned, and is reported as not
        // replayed: a replay never tunes.
        let mut fixed = NativeSpecialization::new();
        for (name, value) in statics {
            fixed = fixed.with_static(name.clone(), *value);
        }
        return implementation
            .default_specialization(&fixed)
            .map(|chosen| Pinned::Chosen(chosen, PinnedChoice::NotReplayed))
            .map_err(|error| format!("default configuration of {entry}: {error:?}"));
    };
    // A parameter the pin does not name (one the implementation gained after
    // the pin was recorded) takes its value in the default configuration at
    // these statics, so another build's configurations replay unchanged.
    let mut statics_only = NativeSpecialization::new();
    for (name, value) in &pinned.statics {
        statics_only = statics_only.with_static(name.clone(), *value);
    }
    let defaults = implementation
        .default_specialization(&statics_only)
        .map_err(|error| format!("default configuration of {entry}: {error}"))?;
    let mut specialization = statics_only;
    for parameter in &implementation.params {
        let value = pinned
            .params
            .get(&parameter.name)
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
    if let Some(name) = pinned.params.keys().find(|name| !declared(name)) {
        return Err(format!(
            "pinned parameter `{name}` is not a native parameter of {entry}"
        ));
    }
    if let Some((launch, name)) = pinned
        .launch_params
        .keys()
        .find(|(launch, name)| !declared_in(*launch, name))
    {
        return Err(format!(
            "pinned parameter `{name}` is not a parameter of launch {launch} of {entry}"
        ));
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
    let (entry, bindings, statics) = key;
    installed.configurations.push(PinnedConfiguration {
        entry: (*entry).to_owned(),
        bindings: bindings.clone(),
        statics: statics.clone(),
        params: chosen.params().clone(),
        launch_params: chosen.launch_params().clone(),
    });
}
