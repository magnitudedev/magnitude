//! Tuning pin: a development-only measurement tool (cargo feature
//! `pinned-tuning`), never part of a served engine.
//!
//! Tuning runs at every load and may choose different configurations between
//! runs, so two runs are bit-comparable only when every entry runs the same
//! configuration. A process that installs [`TuningPin::Record`] tunes each
//! entry (keyed by entry, element bindings and static values) once and reuses
//! that choice for every later load in the process; [`recorded`] returns the
//! choices. A process that installs [`TuningPin::Replay`] with those choices
//! runs exactly them and tunes nothing; an entry without a pinned choice fails
//! preparation.
//!
//! Only `forward_bench` enables the feature (`--tuning-record` /
//! `--tuning-replay`). The pin is process-global so that no production API
//! carries it; nothing reads it unless a pin was installed.

use super::TuningKey;
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
}

pub enum TuningPin {
    /// Tune each entry once per process and record the choice.
    Record,
    /// Run exactly these configurations; tune nothing.
    Replay(Vec<PinnedConfiguration>),
}

struct Installed {
    replay: bool,
    configurations: Vec<PinnedConfiguration>,
}

static PIN: Mutex<Option<Installed>> = Mutex::new(None);

/// Install the pin for every later program preparation in this process.
pub fn install(pin: TuningPin) {
    let installed = match pin {
        TuningPin::Record => Installed {
            replay: false,
            configurations: Vec::new(),
        },
        TuningPin::Replay(configurations) => Installed {
            replay: true,
            configurations,
        },
    };
    *PIN.lock().expect("tuning pin lock poisoned") = Some(installed);
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
    Chosen(NativeSpecialization),
}

pub(super) fn lookup(
    key: &TuningKey,
    implementation: &NativeImplementation,
) -> Result<Pinned, String> {
    let guard = PIN.lock().expect("tuning pin lock poisoned");
    let Some(installed) = guard.as_ref() else {
        return Ok(Pinned::Tune);
    };
    let (entry, bindings, statics) = key;
    let Some(pinned) = installed.configurations.iter().find(|pinned| {
        pinned.entry == *entry && pinned.bindings == *bindings && pinned.statics == *statics
    }) else {
        return if installed.replay {
            Err(format!(
                "the replayed tuning pin has no configuration for statics {statics:?}"
            ))
        } else {
            Ok(Pinned::Tune)
        };
    };
    let mut specialization = NativeSpecialization::new();
    for (name, value) in &pinned.statics {
        specialization = specialization.with_static(name.clone(), *value);
    }
    for (name, value) in &pinned.params {
        specialization = specialization.with_param(name.clone(), *value);
    }
    implementation
        .validate(&specialization)
        .map_err(|error| format!("pinned configuration {:?}: {error}", pinned.params))?;
    Ok(Pinned::Chosen(specialization))
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
    });
}
