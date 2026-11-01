//! An entry bound to its elements under the specialization each mapping of
//! a test gives it on a device: what the test prepares per mapping, and
//! whether the entry's declaration admits that mapping there.

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
            .specialization_valid(device, &(self.specialization)(device, mapping))
            .unwrap()
    }

    pub fn kernel<M>(&self, device: &Device, mapping: M) -> NativeKernel<E>
    where
        S: Fn(&Device, M) -> NativeSpecialization,
    {
        self.entry
            .prepare(device, &(self.specialization)(device, mapping))
            .unwrap()
    }
}
