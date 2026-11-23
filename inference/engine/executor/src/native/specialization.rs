//! The specialization each native entry is prepared with.
//!
//! A specialization fixes the entry's declared static dimensions to this
//! model's values and chooses its tuning parameters. An entry with tuning
//! parameters is tuned on the device when the program is prepared (see
//! `tuning`). Every entry has an implementation for every backend a device
//! can have: the kernels crate's build proves it, so preparation never meets
//! a missing one.

use super::tuning::ranged::{Ranged, row_cuts, row_ranges};
use super::tuning::{EntryTuning, ModelInputs, Tuner};
use super::{CatalogFailure, RangedKernels};
use seismic::{
    BackendName, BoundEntry, Device, Entry, KernelRequest, NativeImplementation, NativeKernel,
    NativeSpecialization, NativeSpecializationError,
};
use std::ops::Range;

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
    Count(seismic::device::DeviceProperties),
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

    /// Every request formed by these entries' formations or by those of
    /// the rounds before them ([`Specializer::count_untuned`]).
    pub fn formed(&self) -> std::collections::HashSet<KernelRequest> {
        self.requests.clone()
    }

    pub fn into_formations(self) -> Vec<UntunedFormation<'a>> {
        self.formations
    }
}

/// What a walk consults for its tuned entries: each unit's launches for the
/// load's objective, and the objective with each term's time at the units'
/// defaults once measured ([`super::tuning::Objective`]).
#[derive(Clone, Debug, Default)]
pub(crate) struct InvocationDemand {
    pub units: std::collections::HashMap<
        super::tuning::DemandKey,
        crate::programs::invocation_demand::InvocationCounts,
    >,
    pub objective: super::tuning::Objective,
}

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
        invocations: InvocationDemand,
        current: crate::programs::invocation_demand::InvocationCounts,
        repeat_case: bool,
    },
}

impl<'a> Tuning<'a> {
    pub fn demand(
        model: ModelInputs<'a>,
        device: &'a Device,
        objective: super::tuning::Objective,
    ) -> Self {
        Self::Demand {
            model,
            device,
            invocations: InvocationDemand {
                units: Default::default(),
                objective,
            },
            current: Default::default(),
            repeat_case: true,
        }
    }
    pub fn set_invocations(
        &mut self,
        current: crate::programs::invocation_demand::InvocationCounts,
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
    pub fn swap_invocations(
        &mut self,
        current: crate::programs::invocation_demand::InvocationCounts,
        repeat_case: bool,
    ) -> Option<(crate::programs::invocation_demand::InvocationCounts, bool)> {
        match self {
            Self::Demand {
                current: active,
                repeat_case: repeat,
                ..
            } => Some((
                std::mem::replace(active, current),
                std::mem::replace(repeat, repeat_case),
            )),
            _ => None,
        }
    }
    pub fn into_invocations(self) -> InvocationDemand {
        match self {
            Self::Demand { invocations, .. } => invocations,
            _ => panic!("only a demand walk collects invocations"),
        }
    }

    /// `fixed`, the static values of the entry `entry` at `bindings`, with
    /// `implementation`'s arithmetic parameters at the choice of the unit
    /// of `prerequisite`'s case ([`Tuner::following`]). A walk without
    /// choices leaves them free: a demand walk, which collects a unit's
    /// launches by its static values alone, and a listing.
    fn following<P: EntryTuning>(
        &mut self,
        prerequisite: &P,
        (entry, bindings): (&'static str, &str),
        implementation: &NativeImplementation,
        fixed: NativeSpecialization,
    ) -> Result<NativeSpecialization, CatalogFailure> {
        match self {
            Self::Tuner(tuner) => {
                let key = unit_of(&tuner.device(), &tuner.model(), prerequisite)?;
                tuner
                    .following(&key, implementation, fixed)
                    .map_err(|outcome| failure(entry, bindings, outcome))
            }
            Self::Demand { .. } | Self::Listing(_) => Ok(fixed),
        }
    }

    pub fn device(&self) -> Option<Device> {
        match self {
            Self::Tuner(tuner) => Some(tuner.device()),
            Self::Demand { device, .. } => Some((*device).clone()),
            Self::Listing(_) => None,
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

/// Prepare program entry `entry` at `specialization` on `device`: every
/// kernel the prepared programs hold is formed here. A program entry
/// publishes its results as tensors, so a prepared one holds no device
/// storage of its own (scalar-result slots) and the programs hold only their
/// owner (`AttestedPrograms::DEVICE_BYTES`).
pub(super) fn prepare_program<E: Entry>(
    device: &Device,
    entry: BoundEntry<E>,
    specialization: &NativeSpecialization,
) -> Result<NativeKernel<E>, seismic::LoadError> {
    const {
        assert!(
            E::NATIVE_SCALAR_RESULT_BYTES == 0,
            "a program entry has no scalar results"
        )
    };
    entry.prepare(device, specialization)
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
/// among them, and a declared parameter among them is fixed at its value.
fn domain_statics(
    implementation: &seismic::OnDevice,
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
    // A value the engine fixes for a declared parameter (one only a single
    // value of applies at these bindings) holds it at that value.
    for (name, value) in values {
        if implementation.params.iter().any(|parameter| parameter.name == *name) {
            specialization = specialization.with_param(*name, *value);
        }
    }
    match implementation.admits(&specialization) {
        Ok(()) => Ok(specialization),
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

/// [`domain_statics`] of `entry` on `backend`, at the element bindings it
/// prepares (its `where` may read them): a call site that drives several
/// entries from one set of `values` gives each only what it declares.
pub(crate) fn entry_statics<E: Entry>(
    backend: BackendName,
    entry: &BoundEntry<E>,
    bindings: &str,
    values: &[(&str, u64)],
) -> Result<NativeSpecialization, CatalogFailure> {
    let implementation =
        seismic::generated::native_implementation_at::<E>(backend, entry.elements())
            .map_err(|error| failure(E::NAME, bindings, error.to_string()))?
            .ok_or_else(|| failure(E::NAME, bindings, "native implementation absent".into()))?
            .at_device(&seismic::device::BackendFloor(backend));
    domain_statics(&implementation, E::NAME, bindings, values)
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
            mode: Mode::Count(*device.properties()),
        }
    }

    /// A specializer for a search's count of `device`'s programs: it forms
    /// nothing, and collects every entry without tuning parameters
    /// ([`Specializer::into_untuned`]) but those of `formed`, which an
    /// earlier round of the search formed ([`UntunedEntries::formed`]).
    pub fn count_untuned(
        device: &'a Device,
        formed: std::collections::HashSet<KernelRequest>,
    ) -> Self {
        Self {
            backend: device.backend(),
            tensor_operations: device.forms_tensor_operations(),
            mode: Mode::CountUntuned(
                device,
                UntunedEntries {
                    requests: formed,
                    formations: Vec::new(),
                },
            ),
        }
    }

    /// The entries without tuning parameters a search's count collected.
    pub fn into_untuned(self) -> UntunedEntries<'a> {
        match self.mode {
            Mode::CountUntuned(_, untuned) => untuned,
            Mode::Prepare(_) | Mode::Count(_) | Mode::List(_) => {
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
            Mode::Prepare(_) | Mode::Count(_) | Mode::CountUntuned(..) => {
                panic!("only a listing names requests")
            }
        }
    }

    /// `entry`'s implementation at its element bindings, bound to the target
    /// device's properties (or the backend floor during device-less listing),
    /// whose configurations its preparation forms.
    fn implementation<E: Entry>(
        &self,
        entry: &BoundEntry<E>,
        bindings: &str,
    ) -> Result<seismic::OnDevice, CatalogFailure> {
        let implementation =
            seismic::generated::native_implementation_at::<E>(self.backend, entry.elements())
                .map_err(|error| failure(E::NAME, bindings, error.to_string()))?
                .unwrap_or_else(|| {
                    panic!(
                        "`{}` has no {} implementation, which the kernels crate's build rules out",
                        E::NAME,
                        self.backend.as_str()
                    )
                });
        Ok(match &self.mode {
            Mode::Prepare(device) | Mode::CountUntuned(device, _) => {
                implementation.at_device(device.properties())
            }
            Mode::Count(properties) => implementation.at_device(properties),
            Mode::List(_) => implementation
                .at_device(&seismic::device::BackendFloor(self.backend)),
        })
    }

    /// Prepare `entry` at `specialization`, or list it.
    fn form<E: Entry>(
        &mut self,
        entry: BoundEntry<E>,
        bindings: &str,
        specialization: &NativeSpecialization,
    ) -> Result<Option<NativeKernel<E>>, CatalogFailure> {
        match &mut self.mode {
            Mode::Prepare(device) => prepare_program(device, entry, specialization)
                .map(Some)
                .map_err(|error| failure(E::NAME, bindings, error.to_string())),
            Mode::Count(_) | Mode::CountUntuned(..) => Ok(None),
            Mode::List(requests) => {
                requests.push(entry.request(self.backend, specialization));
                Ok(None)
            }
        }
    }

    /// Prepare an entry without tuning parameters on any backend
    /// ([`seismic::Untuned`]); `None` during a count or a listing.
    pub fn fixed<E: seismic::Untuned>(
        &mut self,
        bindings: &str,
        values: &[(&str, u64)],
        entry: BoundEntry<E>,
    ) -> Result<Option<NativeKernel<E>>, CatalogFailure> {
        let implementation = self.implementation(&entry, bindings)?;
        let specialization =
            domain_statics(&implementation, E::NAME, bindings, values)?;
        if let Mode::CountUntuned(device, untuned) = &mut self.mode {
            untuned.add(*device, entry, bindings, specialization);
            return Ok(None);
        }
        self.form(entry, bindings, &specialization)
    }

    /// Prepare an entry at its declared defaults, resolving any device-derived
    /// parameter defaults on the opened device; `None` during a count or a
    /// listing.
    pub fn at_defaults<E: Entry>(
        &mut self,
        bindings: &str,
        values: &[(&str, u64)],
        entry: BoundEntry<E>,
    ) -> Result<Option<NativeKernel<E>>, CatalogFailure> {
        let implementation = self.implementation(&entry, bindings)?;
        let statics = domain_statics(&implementation, E::NAME, bindings, values)?;
        let specialization = match &self.mode {
            Mode::Prepare(_) | Mode::CountUntuned(..) => implementation
                .default_specialization(&statics)
                .map_err(|error| failure(E::NAME, bindings, error.to_string()))?,
            Mode::Count(_) | Mode::List(_) => statics,
        };
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
        let specialization = self.specialization(tuning, case)?;
        self.form(case.entry(), &case.bindings(), &specialization)
    }

    /// The specialization [`Self::tuned`] prepares `case`'s entry at.
    fn specialization<T: EntryTuning>(
        &mut self,
        tuning: &mut Tuning<'_>,
        case: &T,
    ) -> Result<NativeSpecialization, CatalogFailure> {
        let implementation = self.implementation(&case.entry(), &case.bindings())?;
        let fixed = case_statics(&tuning.model(), case, &implementation)?;
        self.chosen(tuning, case, implementation.implementation(), fixed)
    }

    /// [`Self::tuned`] of `case` at each range of the rows `served` that it
    /// is tuned by ([`Self::partition`]), each range a unit of its own
    /// ([`Ranged`]), and the kernels bound by rows, one for each run of
    /// neighbouring ranges tuned to one choice ([`Self::by_range`]). `None`
    /// as `tuned` is.
    pub fn tuned_ranges<T: EntryTuning>(
        &mut self,
        tuning: &mut Tuning<'_>,
        served: Range<u64>,
        case: T,
    ) -> Result<Option<RangedKernels<T::Entry>>, CatalogFailure> {
        let ranges = self.partition(tuning, served, &case)?;
        self.by_range(&case, ranges, |this, unit| this.specialization(&mut *tuning, unit))
    }

    /// [`Self::tuned_following`] of `case` at each range of
    /// `prerequisite`'s partition of `served` ([`Self::tuned_ranges`]): each
    /// range follows the unit of `prerequisite`'s same range, so its kernel
    /// keeps the bits of the prerequisite's kernel at those rows. An
    /// implementation without tuning parameters has nothing to follow and
    /// one specialization at every row.
    pub fn tuned_ranges_following<T: EntryTuning, P: EntryTuning>(
        &mut self,
        tuning: &mut Tuning<'_>,
        served: Range<u64>,
        case: T,
        prerequisite: P,
    ) -> Result<Option<RangedKernels<T::Entry>>, CatalogFailure> {
        if !self
            .implementation(&case.entry(), &case.bindings())?
            .implementation()
            .has_tuning_parameters()
        {
            return self.tuned_ranges(tuning, served, case);
        }
        let ranges = self.partition(tuning, served, &prerequisite)?;
        self.by_range(&case, ranges, |this, unit| {
            this.specialization_following(
                &mut *tuning,
                unit,
                &Ranged::new(prerequisite.clone(), unit.rows().clone()),
            )
        })
    }

    /// The ranges `case` is tuned at over `served`: `served` cut where its
    /// declaration says its choice can differ by its rows between the
    /// case's points, and between every two of them on a backend where a
    /// unit forms no program ([`row_cuts`], at the case's statics), each
    /// range holding points of the case's own ([`row_ranges`]). An implementation
    /// without tuning parameters has one specialization at every row: its
    /// partition is the one range `served`.
    fn partition<T: EntryTuning>(
        &self,
        tuning: &Tuning<'_>,
        served: Range<u64>,
        case: &T,
    ) -> Result<Vec<Range<u64>>, CatalogFailure> {
        let implementation = self.implementation(&case.entry(), &case.bindings())?;
        if !implementation.implementation().has_tuning_parameters() {
            return Ok(vec![served]);
        }
        let model = tuning.model();
        let statics = case_statics(&model, case, &implementation)?;
        let points = case
            .points(model.limits)
            .into_iter()
            .map(|point| point.rows)
            .collect::<Vec<_>>();
        let mut cuts = row_cuts(
            &implementation,
            &statics,
            case.row_dimension(),
            &served,
            &points,
        );
        cuts.extend_from_slice(case.row_regime_cuts());
        Ok(row_ranges(served, &cuts, &points))
    }

    /// `case`'s kernels at the specialization `choose` gives its unit of
    /// each of `ranges` (a partition, ascending), bound by rows. Each range
    /// is a unit with a choice of its own; neighbouring ranges of one
    /// choice are one kernel's, prepared once and serving the rows of them
    /// all, so the kernels are the entry's distinct choices by its rows
    /// and nothing is prepared twice.
    fn by_range<T: EntryTuning>(
        &mut self,
        case: &T,
        ranges: Vec<Range<u64>>,
        mut choose: impl FnMut(&mut Self, &Ranged<T>) -> Result<NativeSpecialization, CatalogFailure>,
    ) -> Result<Option<RangedKernels<T::Entry>>, CatalogFailure> {
        let mut choices: Vec<(Range<u64>, String, NativeSpecialization)> = Vec::new();
        for rows in ranges {
            let unit = Ranged::new(case.clone(), rows.clone());
            let specialization = choose(&mut *self, &unit)?;
            match choices.last_mut() {
                Some((served, _, last)) if *last == specialization => served.end = rows.end,
                _ => choices.push((rows, unit.bindings(), specialization)),
            }
        }
        let mut parts = Vec::with_capacity(choices.len());
        for (rows, bindings, specialization) in choices {
            parts.push(
                self.form(case.entry(), &bindings, &specialization)?
                    .map(|kernel| (rows, kernel)),
            );
        }
        Ok(parts
            .into_iter()
            .collect::<Option<Vec<_>>>()
            .map(RangedKernels::new))
    }

    /// Prepare an entry through its tuning case as [`Self::tuned`] does, a
    /// dependant of the unit of `prerequisite`'s case: its arithmetic
    /// parameters are at that unit's choice ([`Tuner::following`]), so its
    /// results carry the prerequisite's bits, and it is tuned in the others.
    pub fn tuned_following<T: EntryTuning, P: EntryTuning>(
        &mut self,
        tuning: &mut Tuning<'_>,
        case: &T,
        prerequisite: &P,
    ) -> Result<Option<NativeKernel<T::Entry>>, CatalogFailure> {
        let specialization = self.specialization_following(tuning, case, prerequisite)?;
        self.form(case.entry(), &case.bindings(), &specialization)
    }

    /// The specialization [`Self::tuned_following`] prepares `case`'s entry
    /// at.
    fn specialization_following<T: EntryTuning, P: EntryTuning>(
        &mut self,
        tuning: &mut Tuning<'_>,
        case: &T,
        prerequisite: &P,
    ) -> Result<NativeSpecialization, CatalogFailure> {
        let bindings = case.bindings();
        let implementation = self.implementation(&case.entry(), &bindings)?;
        if !implementation.implementation().has_tuning_parameters() {
            return self.specialization(tuning, case);
        }
        let fixed = case_statics(&tuning.model(), case, &implementation)?;
        let fixed = tuning.following(
            prerequisite,
            (<T::Entry as Entry>::NAME, &bindings),
            implementation.implementation(),
            fixed,
        )?;
        self.chosen(tuning, case, implementation.implementation(), fixed)
    }

    /// The specialization `case`'s entry is prepared at from `fixed`: tuned
    /// in its free parameters by a tuner, `fixed` in any other walk.
    fn chosen<T: EntryTuning>(
        &mut self,
        tuning: &mut Tuning<'_>,
        case: &T,
        implementation: &NativeImplementation,
        fixed: NativeSpecialization,
    ) -> Result<NativeSpecialization, CatalogFailure> {
        let entry = <T::Entry as Entry>::NAME;
        let bindings = case.bindings();
        let tuning_model = tuning.model();
        Ok(match tuning {
            Tuning::Tuner(tuner) if implementation.has_tuning_parameters() => {
                tuner.tune(case, implementation, &fixed)?
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
                    .represented(|launch| case.invocation_rows(launch, &representatives))
                    .scaled(if *repeat_case { case.launches() } else { 1 });
                invocations
                    .units
                    .entry((entry, bindings, fixed.statics().clone()))
                    .or_default()
                    .add(&calls);
                fixed
            }
            Tuning::Tuner(_) | Tuning::Listing(_) => fixed,
        })
    }
}

/// The static values `case` fixes `implementation` at for `model`.
fn case_statics<T: EntryTuning>(
    model: &ModelInputs<'_>,
    case: &T,
    implementation: &seismic::OnDevice,
) -> Result<NativeSpecialization, CatalogFailure> {
    let entry = <T::Entry as Entry>::NAME;
    let bindings = case.bindings();
    let values = case
        .statics(model)
        .map_err(|outcome| failure(entry, &bindings, outcome))?;
    domain_statics(implementation, entry, &bindings, &values)
}

/// The unit of `case` on `device` for `model`.
fn unit_of<T: EntryTuning>(
    device: &Device,
    model: &ModelInputs<'_>,
    case: &T,
) -> Result<super::tuning::TuningKey, CatalogFailure> {
    let entry = <T::Entry as Entry>::NAME;
    let bindings = case.bindings();
    let implementation = seismic::generated::native_implementation_at::<T::Entry>(
        device.backend(),
        case.entry().elements(),
    )
    .map_err(|error| failure(entry, &bindings, error.to_string()))?
    .ok_or_else(|| failure(entry, &bindings, "native implementation absent".into()))?
    .at_device(device.properties());
    let fixed = case_statics(model, case, &implementation)?;
    Ok(super::tuning::unit_key(entry, bindings, &fixed))
}

#[cfg(test)]
mod tests {
    use super::*;
    use magnitude_kernels::{
        routed_down, routed_down_project, routed_gate_up, routed_gate_up_part, routed_output,
        routed_output_project, routed_output_streamed, routed_route_shared, routed_route_streamed, routed_select,
        routed_select_streamed,
    };
    use seismic::BackendName;

    /// A streamed layer's route, selection, gate/up passes and sited
    /// projections follow their resident entry: on every backend that declares both, a dependant is fixed at
    /// exactly its arithmetic parameters, each at the resident choice's
    /// value, which the resident declares arithmetic over the same values;
    /// each other parameter, its launch shape, is left to its search with
    /// every value it declares.
    #[test]
    fn a_streamed_route_follows_its_residents_arithmetic_and_searches_its_launch() {
        fn follows<R: Entry, S: Entry>(backend: BackendName) {
            let Some(resident) =
                seismic::generated::native_implementation_for_backend::<R>(backend).unwrap()
            else {
                return;
            };
            let streamed = seismic::generated::native_implementation_for_backend::<S>(backend)
                .unwrap()
                .unwrap();
            if !streamed.has_tuning_parameters() {
                assert!(
                    !resident.has_tuning_parameters(),
                    "{} on {backend:?} has parameters that {} does not follow",
                    R::NAME,
                    S::NAME
                );
                return;
            }
            // A resident choice away from every default.
            let choice = resident
                .params
                .iter()
                .fold(NativeSpecialization::new(), |choice, parameter| {
                    choice.with_param(parameter.name.clone(), *parameter.values.last().unwrap())
                });
            let fixed = super::super::tuning::arithmetic_of(&streamed, &choice).unwrap();
            let searched = streamed
                .params
                .iter()
                .filter(|parameter| !fixed.iter().any(|(name, _)| *name == parameter.name))
                .collect::<Vec<_>>();
            assert!(
                !searched.is_empty(),
                "{} on {backend:?} has no launch shape to search",
                S::NAME
            );
            for parameter in searched {
                assert!(parameter.values.len() > 1, "{parameter:?}");
            }
            for (name, value) in fixed {
                let own = resident.params.iter().find(|parameter| parameter.name == name).unwrap();
                let followed = streamed.params.iter().find(|parameter| parameter.name == name).unwrap();
                assert!(own.arithmetic && own.values == followed.values, "{name} on {backend:?}");
                assert_eq!(choice.param(&name), Some(value));
            }
        }
        for backend in [BackendName::Metal, BackendName::Cuda, BackendName::Vulkan] {
            follows::<routed_route_shared::Entry, routed_route_streamed::Entry>(backend);
            follows::<routed_select::Entry, routed_select_streamed::Entry>(backend);
            follows::<routed_gate_up::Entry, routed_gate_up_part::Entry>(backend);
            follows::<routed_output::Entry, routed_output_project::Entry>(backend);
            if matches!(backend, BackendName::Cuda) {
                follows::<routed_output::Entry, routed_output_streamed::Entry>(backend);
            }
            follows::<routed_down::Entry, routed_down_project::Entry>(backend);
        }
    }

    /// The load's twin of graph construction's domain check: a down
    /// projection wider than the Metal kernel's register budget admits is
    /// refused before any tuning, naming the call's statics.
    #[test]
    fn statics_outside_the_kernel_domain_are_refused() {
        let catalog = seismic::Devices::discover(&[]).unwrap();
        // Only a host with Metal runs it.
        if !crate::test_support::available(&catalog, BackendName::Metal) {
            return;
        }
        let device = crate::test_support::open_backend(&catalog, BackendName::Metal).expect("an available Metal executor opens");
        let implementation =
            seismic::generated::native_implementation::<routed_output::Entry>(&device)
                .unwrap()
                .unwrap()
                .at_device(device.properties());
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

    /// Preparation's `domain_statics`, `default_specialization`, and `row_cuts`
    /// must bind device properties before evaluating any `where` clause or
    /// default expression: `attention_prefill` (including the W=512 BF16
    /// M=1..9 shape), `dense_output`, `routed_expand`, `stream_mix_sinkhorn`,
    /// and `sample_rows` must succeed with zero `Unbound("device.*")` errors
    /// on both CUDA and Metal.
    #[test]
    fn device_dependent_entries_pass_domain_statics_and_row_cuts_with_zero_unbound_errors() {
        use magnitude_kernels::{
            attention_prefill, dense_output, routed_expand, sample_rows, stream_mix_sinkhorn,
        };
        use seismic::Element;

        let bf16 = Element::bf16();
        let prefill_values = [
            ("KV", 8),
            ("G", 4),
            ("P", 256),
            ("S", 0),
            ("I", 0),
            ("U", 0),
            ("F", 1),
            ("NQ", 1),
            ("N", 1),
            ("NV", 0),
            ("B", 0),
            ("BR", 1),
            ("SK", 0),
            ("RA", 0),
            ("RT", 0),
            ("RO", 0),
            ("VK", 0),
            ("KR", 0),
            ("AB", 0),
            ("KS", 0),
        ];

        for backend in [BackendName::Cuda, BackendName::Metal] {
            let floor = seismic::device::BackendFloor(backend);

            // 1. attention_prefill with W=2*P+S=512, A=bf16, M=1..9
            let prefill_impl = seismic::generated::native_implementation_at::<
                attention_prefill::Entry,
            >(backend, &[("A", bf16)])
            .unwrap()
            .unwrap()
            .at_device(&floor);
            let prefill_statics = domain_statics(
                &prefill_impl,
                "attention_prefill",
                "A=bf16,M=1..9",
                &prefill_values,
            )
            .unwrap_or_else(|e| panic!("attention_prefill domain_statics on {backend:?}: {e:?}"));
            prefill_impl
                .default_specialization(&prefill_statics)
                .unwrap_or_else(|e| {
                    panic!("attention_prefill default_specialization on {backend:?}: {e:?}")
                });
            let _ = row_cuts(
                &prefill_impl,
                &prefill_statics,
                "M",
                &(1..10),
                &[1, 2, 4, 8],
            );

            // 2. dense_output
            let dense_impl = seismic::generated::native_implementation_at::<dense_output::Entry>(
                backend,
                &[("A", bf16), ("DW", bf16)],
            )
            .unwrap()
            .unwrap()
            .at_device(&floor);
            let dense_statics = domain_statics(
                &dense_impl,
                "dense_output",
                "A=bf16,DW=bf16",
                &[("H", 2048), ("F", 5632), ("DS", 0), ("DB", 0)],
            )
            .unwrap_or_else(|e| panic!("dense_output domain_statics on {backend:?}: {e:?}"));
            dense_impl
                .default_specialization(&dense_statics)
                .unwrap_or_else(|e| {
                    panic!("dense_output default_specialization on {backend:?}: {e:?}")
                });

            // 3. routed_expand
            let routed_impl = seismic::generated::native_implementation_at::<routed_expand::Entry>(
                backend,
                &[("A", bf16), ("EGW", bf16), ("EUW", bf16), ("SGW", bf16), ("SUW", bf16)],
            )
            .unwrap()
            .unwrap()
            .at_device(&floor);
            let routed_statics = domain_statics(
                &routed_impl,
                "routed_expand",
                "A=bf16",
                &[("H", 2048), ("E", 64), ("K", 8), ("F", 1536), ("S", 0), ("GS", 0), ("US", 0)],
            )
            .unwrap_or_else(|e| panic!("routed_expand domain_statics on {backend:?}: {e:?}"));
            routed_impl
                .default_specialization(&routed_statics)
                .unwrap_or_else(|e| {
                    panic!("routed_expand default_specialization on {backend:?}: {e:?}")
                });

            // 4. stream_mix_sinkhorn
            let stream_impl =
                seismic::generated::native_implementation_at::<stream_mix_sinkhorn::Entry>(
                    backend,
                    &[],
                )
                .unwrap()
                .unwrap()
                .at_device(&floor);
            let stream_statics = domain_statics(
                &stream_impl,
                "stream_mix_sinkhorn",
                "",
                &[("H", 2048), ("S", 4), ("P", 4)],
            )
            .unwrap_or_else(|e| panic!("stream_mix_sinkhorn domain_statics on {backend:?}: {e:?}"));
            stream_impl
                .default_specialization(&stream_statics)
                .unwrap_or_else(|e| {
                    panic!("stream_mix_sinkhorn default_specialization on {backend:?}: {e:?}")
                });

            // 5. sample_rows
            let sample_impl =
                seismic::generated::native_implementation_at::<sample_rows::Entry>(backend, &[])
                    .unwrap()
                    .unwrap()
                    .at_device(&floor);
            let sample_statics =
                domain_statics(&sample_impl, "sample_rows", "", &[("V", 151_936)])
                    .unwrap_or_else(|e| panic!("sample_rows domain_statics on {backend:?}: {e:?}"));
            sample_impl
                .default_specialization(&sample_statics)
                .unwrap_or_else(|e| {
                    panic!("sample_rows default_specialization on {backend:?}: {e:?}")
                });
        }
    }
}
