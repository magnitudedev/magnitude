//! Shared timing of the kernel timing tests: a kernel's calls, or a formed
//! point, measured through `seismic-measure` as each test's own
//! [`Measure`] says.
#![allow(dead_code)]

use seismic::measure::{Measure, Measurement};

/// A kernel's device time over a rotation of its calls, as `how` measures
/// it.
pub trait Timed<E: seismic::Entry> {
    fn time(
        &self,
        rotation: Vec<E::Args<'_>>,
        how: &Measure,
    ) -> Result<Measurement, seismic::CallError>;
}

impl<E: seismic::Entry> Timed<E> for seismic::NativeKernel<E> {
    fn time(
        &self,
        rotation: Vec<E::Args<'_>>,
        how: &Measure,
    ) -> Result<Measurement, seismic::CallError> {
        seismic::measure::measure(self.point(rotation)?, how)
    }
}

/// `point` measured as `how` says; a failure fails the test.
pub fn measured(point: seismic::KernelPoint, how: &Measure) -> Measurement {
    seismic::measure::measure(point, how).unwrap()
}
