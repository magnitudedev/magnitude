//! An entry bound to its elements under the specialization each mapping of
//! a test gives it on a device: what the test prepares per mapping, and
//! whether the entry's declaration admits that mapping there. A mapping
//! names the parameters it sets; the rest are at the declaration's defaults.

use seismic::{BoundEntry, Device, Entry, NativeKernel, NativeSpecialization};

pub struct Mapped<E, S> {
    entry: BoundEntry<E>,
    specialization: S,
}

impl<E: Entry, S> Mapped<E, S> {
    pub fn new(entry: BoundEntry<E>, specialization: S) -> Self {
        Self {
            entry,
            specialization,
        }
    }

    /// Whether the entry admits `mapping` on `device` as preparation takes
    /// it (`BoundEntry::specialization_valid`): its `where` at the bound
    /// elements.
    pub fn admits<M>(&self, device: &Device, mapping: M) -> bool
    where
        S: Fn(&Device, M) -> NativeSpecialization,
    {
        self.entry
            .specialization_valid(device, &self.complete(device, (self.specialization)(device, mapping)))
            .unwrap()
    }

    pub fn kernel<M>(&self, device: &Device, mapping: M) -> NativeKernel<E>
    where
        S: Fn(&Device, M) -> NativeSpecialization,
    {
        self.entry
            .prepare(device, &self.complete(device, (self.specialization)(device, mapping)))
            .unwrap()
    }

    /// `set` over the entry's defaults at its statics: the parameters and
    /// launch parameters the mapping does not name take their declared default.
    fn complete(&self, device: &Device, set: NativeSpecialization) -> NativeSpecialization {
        let statics = set
            .statics()
            .iter()
            .fold(NativeSpecialization::new(), |spec, (name, value)| spec.with_static(name.clone(), *value));
        let Ok(Some(defaults)) = self.entry.default_specialization(device, &statics) else {
            return set;
        };
        let defaults = set
            .params()
            .iter()
            .fold(defaults, |spec, (name, value)| spec.with_param(name.clone(), *value));
        set.launch_params().iter().fold(defaults, |spec, ((launch, name), value)| {
            spec.with_launch_param(*launch, name.clone(), *value)
        })
    }
}

/// The candidate mappings `admits` holds for, of the entry (or entries) named
/// `entry`. A test whose entry admits none has no case to run and would pass
/// having proved nothing, so it fails here, naming the entry.
pub fn admitted<M>(entry: &str, candidates: impl IntoIterator<Item = M>, admits: impl Fn(&M) -> bool) -> Vec<M> {
    let candidates: Vec<M> = candidates.into_iter().collect();
    let total = candidates.len();
    let admitted: Vec<M> = candidates.into_iter().filter(|mapping| admits(mapping)).collect();
    assert!(
        !admitted.is_empty(),
        "`{entry}` admits none of the {total} mappings its test offers: the test would run no case"
    );
    admitted
}
