//! The specialization each native entry is prepared with.
//!
//! A specialization fixes the entry's declared static dimensions to this
//! model's values and chooses its tuning parameters. An entry with tuning
//! parameters is tuned on the device when the program is prepared (see
//! `tuning`). Every entry has an implementation for every backend a device
//! can have: the kernels crate's build proves it, so preparation never meets
//! a missing one.

use super::CatalogFailure;
use super::tuning::{ComposedContender, EntryTuning, ModelInputs, Tuner};
use seismic::{
    BackendName, BoundEntry, Device, Entry, KernelRequest, NativeImplementation, NativeKernel,
    NativeSpecialization, NativeSpecializationError,
};

pub(super) struct Specializer<'a> {
    backend: BackendName,
    /// Whether the device forms Metal tensor operations
    /// (`Device::forms_tensor_operations`).
    tensor_operations: bool,
    mode: Mode<'a>,
}

enum Mode<'a> {
    /// Prepare each entry on the device.
    Prepare(&'a Device),
    /// A tuning count or census walks the program through its tuner; it
    /// forms nothing and every entry comes back `None`.
    Count,
    /// A search's count: as a count, and it collects every entry without
    /// tuning parameters for the search to form ([`UntunedEntries`]).
    CountUntuned(&'a Device, UntunedEntries<'a>),
    /// A listing names the request of every entry the walk would prepare,
    /// without a device; every entry comes back `None`.
    List(Vec<KernelRequest>),
}

/// The formation of one entry without tuning parameters, on any thread:
/// its programs are formed as a load forms them and released.
pub(super) type UntunedFormation<'a> = Box<dyn FnOnce() -> Result<(), CatalogFailure> + Send + 'a>;

/// The entries without tuning parameters a search's count met, each request
/// once. The search forms them with its units' canonical programs, so the
/// load that follows it finds every program it prepares already formed
/// (kept by the toolchain's store and the driver's caches) instead of
/// forming them one at a time itself.
#[derive(Default)]
pub(super) struct UntunedEntries<'a> {
    requests: std::collections::HashSet<KernelRequest>,
    formations: Vec<UntunedFormation<'a>>,
}

impl<'a> UntunedEntries<'a> {
    fn add<E: Entry>(
        &mut self,
        device: &'a Device,
        entry: BoundEntry<E>,
        bindings: &str,
        specialization: NativeSpecialization,
    ) {
        if !self
            .requests
            .insert(entry.request(device.backend(), &specialization))
        {
            return;
        }
        let bindings = bindings.to_owned();
        self.formations.push(Box::new(move || {
            entry
                .prepare(device, &specialization)
                .map(drop)
                .map_err(|error| failure(E::NAME, &bindings, error.to_string()))
        }));
    }

    pub fn len(&self) -> usize {
        self.formations.len()
    }

    pub fn into_formations(self) -> Vec<UntunedFormation<'a>> {
        self.formations
    }
}

/// What a walk consults for its tuned entries.
pub(super) type InvocationDemand = std::collections::HashMap<
    super::tuning::TuningKey,
    crate::programs::invocation_demand::InvocationCounts,
>;

pub(super) enum Tuning<'a> {
    /// A preparation's tuner, in its count, census or search phase.
    Tuner(Tuner<'a>),
    /// A listing's model: its entries' static values, and nothing to tune.
    Listing(ModelInputs<'a>),
    /// Metadata-only call-site collection, before any search is opened. It
    /// walks the same preparation as the tuner, under the same settings, so
    /// every key it records is the key the tuner later looks up.
    Demand {
        model: ModelInputs<'a>,
        device: &'a Device,
        settings: super::tuning::TuningSettings,
        invocations: InvocationDemand,
        current: crate::programs::invocation_demand::InvocationForecast,
        repeat_case: bool,
    },
}

impl<'a> Tuning<'a> {
    pub fn demand(
        model: ModelInputs<'a>,
        device: &'a Device,
        settings: super::tuning::TuningSettings,
    ) -> Self {
        Self::Demand {
            model,
            device,
            settings,
            invocations: InvocationDemand::new(),
            current: Default::default(),
            repeat_case: true,
        }
    }
    pub fn set_invocations(
        &mut self,
        current: crate::programs::invocation_demand::InvocationForecast,
        repeat_case: bool,
    ) {
        if let Self::Demand {
            current: active,
            repeat_case: repeat,
            ..
        } = self
        {
            *active = current;
            *repeat = repeat_case;
        }
    }
    pub fn into_invocations(self) -> InvocationDemand {
        match self {
            Self::Demand { invocations, .. } => invocations,
            _ => panic!("only a demand walk collects invocations"),
        }
    }

    pub fn dependency<T: EntryTuning>(
        &self,
        case: &T,
    ) -> Result<Option<super::tuning::TuningKey>, CatalogFailure> {
        let Some(device) = self.device() else {
            return Ok(None);
        };
        let entry = <T::Entry as Entry>::NAME;
        let bindings = case.bindings();
        let implementation =
            seismic::generated::native_implementation_for_backend::<T::Entry>(device.backend())
                .map_err(|error| failure(entry, &bindings, error.to_string()))?
                .ok_or_else(|| failure(entry, &bindings, "native implementation absent".into()))?;
        let values = case
            .statics(&self.model())
            .map_err(|error| failure(entry, &bindings, error))?;
        let fixed = domain_statics(&implementation, entry, &bindings, &values)?;
        Ok(Some((entry, bindings, fixed.statics().clone())))
    }

    pub fn choice(&self, key: &super::tuning::TuningKey) -> Option<&NativeSpecialization> {
        match self {
            Self::Tuner(tuner) => tuner.choice(key),
            Self::Listing(_) | Self::Demand { .. } => None,
        }
    }
    pub fn cache(&self) -> Option<&'a crate::kernel_cache::KernelCache> {
        match self {
            Self::Tuner(tuner) => tuner.cache(),
            Self::Listing(_) | Self::Demand { .. } => None,
        }
    }
    /// The cache whose stored tuning results this preparation reads: none for
    /// a fresh preparation, which only replaces them.
    pub fn stored_results(&self) -> Option<&'a crate::kernel_cache::KernelCache> {
        match self {
            Self::Tuner(tuner) => tuner.cache().filter(|_| !tuner.settings().fresh),
            Self::Listing(_) | Self::Demand { .. } => None,
        }
    }
    pub fn settings(&self) -> Option<super::tuning::TuningSettings> {
        match self {
            Self::Tuner(tuner) => Some(tuner.settings()),
            Self::Demand { settings, .. } => Some(*settings),
            Self::Listing(_) => None,
        }
    }
    pub fn device(&self) -> Option<&'a Device> {
        match self {
            Self::Tuner(tuner) => Some(tuner.device()),
            Self::Demand { device, .. } => Some(device),
            Self::Listing(_) => None,
        }
    }

    pub fn register_composed(
        &mut self,
        contender: Box<dyn ComposedContender<'a> + 'a>,
        rows: u64,
    ) -> Result<Option<bool>, CatalogFailure> {
        match self {
            Self::Tuner(tuner) => tuner.register_composed(contender),
            Self::Demand {
                invocations,
                current,
                ..
            } => {
                invocations
                    .entry((
                        contender.label(),
                        contender.key(),
                        std::collections::BTreeMap::new(),
                    ))
                    .or_default()
                    .add(&current.represented(|logical| {
                        (logical > rows / 2 && logical <= rows).then_some(rows)
                    }));
                Ok(None)
            }
            Self::Listing(_) => Ok(None),
        }
    }

    pub fn model(&self) -> ModelInputs<'a> {
        match self {
            Self::Tuner(tuner) => tuner.model(),
            Self::Listing(model) | Self::Demand { model, .. } => *model,
        }
    }

    pub fn into_tuner(self) -> Tuner<'a> {
        match self {
            Self::Tuner(tuner) => tuner,
            Self::Listing(_) | Self::Demand { .. } => panic!("a metadata walk has no tuner"),
        }
    }
}

fn failure(entry: &'static str, bindings: &str, outcome: String) -> CatalogFailure {
    CatalogFailure::Preparation {
        entry,
        bindings: bindings.to_owned(),
        outcome,
    }
}

/// The static values of `implementation`, which must lie in the kernel's
/// domain (some configuration admits them) or the model cannot run on this
/// backend. `values` supplies this model's value of every dimension the call
/// site can fix; each dimension the implementation declares static must be
/// among them.
fn domain_statics(
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
    match implementation.default_specialization(&specialization) {
        Ok(_) => Ok(specialization),
        Err(NativeSpecializationError::Inadmissible) => Err(CatalogFailure::KernelDomain {
            entry,
            statics: specialization
                .statics()
                .iter()
                .map(|(name, value)| (name.clone(), *value))
                .collect(),
        }),
        Err(error) => Err(failure(entry, bindings, error.to_string())),
    }
}

impl<'a> Specializer<'a> {
    pub fn new(device: &'a Device) -> Self {
        Self {
            backend: device.backend(),
            tensor_operations: device.forms_tensor_operations(),
            mode: Mode::Prepare(device),
        }
    }

    /// A specializer for a tuning count or census of `device`'s programs,
    /// which forms nothing.
    pub fn count(device: &Device) -> Self {
        Self {
            backend: device.backend(),
            tensor_operations: device.forms_tensor_operations(),
            mode: Mode::Count,
        }
    }

    /// A specializer for a search's count of `device`'s programs: it forms
    /// nothing, and collects every entry without tuning parameters
    /// ([`Specializer::into_untuned`]).
    pub fn count_untuned(device: &'a Device) -> Self {
        Self {
            backend: device.backend(),
            tensor_operations: device.forms_tensor_operations(),
            mode: Mode::CountUntuned(device, UntunedEntries::default()),
        }
    }

    /// The entries without tuning parameters a search's count collected.
    pub fn into_untuned(self) -> UntunedEntries<'a> {
        match self.mode {
            Mode::CountUntuned(_, untuned) => untuned,
            Mode::Prepare(_) | Mode::Count | Mode::List(_) => {
                panic!("only a search's count collects untuned entries")
            }
        }
    }

    /// A specializer that lists the request of every entry, without a
    /// device: those of a device with every optional form, so the listing
    /// covers any device of the backend.
    pub fn list(backend: BackendName) -> Self {
        Self {
            backend,
            tensor_operations: backend == BackendName::Metal,
            mode: Mode::List(Vec::new()),
        }
    }

    pub fn backend(&self) -> BackendName {
        self.backend
    }

    pub fn forms_tensor_operations(&self) -> bool {
        self.tensor_operations
    }

    /// The requests a listing named, in walk order.
    pub fn into_requests(self) -> Vec<KernelRequest> {
        match self.mode {
            Mode::List(requests) => requests,
            Mode::Prepare(_) | Mode::Count | Mode::CountUntuned(..) => {
                panic!("only a listing names requests")
            }
        }
    }

    fn implementation<E: Entry>(
        &self,
        bindings: &str,
    ) -> Result<NativeImplementation, CatalogFailure> {
        let implementation =
            seismic::generated::native_implementation_for_backend::<E>(self.backend)
                .map_err(|error| failure(E::NAME, bindings, error.to_string()))?;
        Ok(implementation.unwrap_or_else(|| {
            panic!(
                "`{}` has no {} implementation, which the kernels crate's build rules out",
                E::NAME,
                self.backend.as_str()
            )
        }))
    }

    /// Prepare `entry` at `specialization`, or list it.
    fn form<E: Entry>(
        &mut self,
        entry: BoundEntry<E>,
        bindings: &str,
        specialization: &NativeSpecialization,
    ) -> Result<Option<NativeKernel<E>>, CatalogFailure> {
        match &mut self.mode {
            Mode::Prepare(device) => entry
                .prepare(device, specialization)
                .map(Some)
                .map_err(|error| failure(E::NAME, bindings, error.to_string())),
            Mode::Count | Mode::CountUntuned(..) => Ok(None),
            Mode::List(requests) => {
                requests.push(entry.request(self.backend, specialization));
                Ok(None)
            }
        }
    }

    /// Prepare an entry without tuning parameters; `None` during a count or
    /// a listing.
    pub fn fixed<E: Entry>(
        &mut self,
        bindings: &str,
        values: &[(&str, u64)],
        entry: BoundEntry<E>,
    ) -> Result<Option<NativeKernel<E>>, CatalogFailure> {
        let implementation = self.implementation::<E>(bindings)?;
        if implementation.has_tuning_parameters() {
            return Err(failure(
                E::NAME,
                bindings,
                "the implementation declares tuning parameters, but its call site registers no tuning case".into(),
            ));
        }
        let specialization = domain_statics(&implementation, E::NAME, bindings, values)?;
        if let Mode::CountUntuned(device, untuned) = &mut self.mode {
            untuned.add(*device, entry, bindings, specialization);
            return Ok(None);
        }
        self.form(entry, bindings, &specialization)
    }

    /// Prepare an entry through its tuning case: static values from the
    /// case, parameters tuned on the device when the implementation declares
    /// any. `None` during a tuning count, which forms nothing, or a listing,
    /// which tunes nothing.
    pub fn tuned<T: EntryTuning>(
        &mut self,
        tuning: &mut Tuning<'_>,
        case: &T,
    ) -> Result<Option<NativeKernel<T::Entry>>, CatalogFailure> {
        let entry = <T::Entry as Entry>::NAME;
        let bindings = case.bindings();
        let implementation = self.implementation::<T::Entry>(&bindings)?;
        let values = case
            .statics(&tuning.model())
            .map_err(|outcome| failure(entry, &bindings, outcome))?;
        let fixed = domain_statics(&implementation, entry, &bindings, &values)?;
        let tuning_model = tuning.model();
        let specialization = match tuning {
            Tuning::Tuner(tuner) if implementation.has_tuning_parameters() => {
                tuner.tune(case, &implementation, &fixed)?
            }
            Tuning::Demand {
                invocations,
                current,
                repeat_case,
                ..
            } => {
                let mut representatives = case
                    .points(tuning_model.limits)
                    .into_iter()
                    .map(|point| point.rows)
                    .collect::<Vec<_>>();
                representatives.sort_unstable();
                representatives.dedup();
                let calls = current
                    .represented(|rows| case.invocation_rows(rows, &representatives))
                    .scaled(if *repeat_case { case.launches() } else { 1 });
                invocations
                    .entry((entry, bindings.clone(), fixed.statics().clone()))
                    .or_default()
                    .add(&calls);
                fixed
            }
            Tuning::Tuner(_) | Tuning::Listing(_) => fixed,
        };
        self.form(case.entry(), &bindings, &specialization)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use magnitude_kernels::routed_output;
    use seismic::{BackendName, DeviceCatalog};

    /// The load's twin of graph construction's domain check: a down
    /// projection wider than the Metal kernel's register budget admits is
    /// refused before any tuning, naming the call's statics.
    #[test]
    fn statics_outside_the_kernel_domain_are_refused() {
        let Ok(device) = DeviceCatalog::discover()
            .unwrap()
            .open_backend(BackendName::Metal)
        else {
            return;
        };
        let implementation =
            seismic::generated::native_implementation::<routed_output::Entry>(&device)
                .unwrap()
                .unwrap();
        let values = [
            ("H", 256),
            ("SN", 1),
            ("IN", 0),
            ("K", 2),
            ("F", 16_384),
            ("S", 256),
        ];
        let Err(CatalogFailure::KernelDomain { entry, statics }) =
            domain_statics(&implementation, "routed_output", "A=bf16", &values)
        else {
            panic!("out-of-domain statics were admitted");
        };
        assert_eq!(entry, "routed_output");
        assert_eq!(
            statics,
            [
                ("F", 16_384),
                ("H", 256),
                ("IN", 0),
                ("K", 2),
                ("S", 256),
                ("SN", 1),
            ]
                .map(|(name, value)| (name.to_owned(), value))
                .to_vec()
        );
        assert!(
            domain_statics(
                &implementation,
                "routed_output",
                "A=bf16",
                &[
                    ("H", 256),
                    ("SN", 1),
                    ("IN", 0),
                    ("K", 2),
                    ("F", 512),
                    ("S", 256),
                ],
            )
            .is_ok()
        );
    }
}
