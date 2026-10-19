//! Measuring a point: [`measure`], the one way to time a [`Work`] on its
//! own.

use crate::deadline::{Deadline, WorkError};
use crate::session::{Counted, Measurement, Slot, SlotFailure, sample};
use crate::statistics::Shuffle;
use crate::work::Work;
use std::num::NonZeroUsize;

/// How a point is measured: how many samples it takes, and the device time
/// a sample covers at least (its repetitions follow from it). Nothing is
/// defaulted.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Measure {
    /// The samples taken.
    pub samples: NonZeroUsize,
    /// The device time a sample covers at least.
    pub min_sample_seconds: f64,
}

/// Measure `work` as `how` says, at the device's clock as it is. The one
/// way to time a point.
pub fn measure<W: Work>(work: W, how: &Measure) -> Result<Measurement, W::Error> {
    let mut slots = [Slot::new(work)];
    sample(
        &mut slots,
        how.min_sample_seconds,
        &mut Counted::new(1, how.samples.get()),
        &mut Shuffle::default(),
        None,
        Deadline::default(),
    )
    .map_err(|failure: SlotFailure<W::Error>| match failure.error {
        WorkError::Failed(error) => error,
        WorkError::Expired => unreachable!("sampling without a deadline does not expire"),
    })?;
    Ok(slots[0].measurement())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::work::{Gauge, Pending};

    /// Work whose passes take the seconds it holds each.
    struct Fixed(f64);

    struct Done(f64);

    impl Pending<&'static str> for Done {
        fn seconds(&self) -> Result<f64, &'static str> {
            Ok(self.0)
        }
        fn wait(&self) -> Result<(), &'static str> {
            Ok(())
        }
    }

    impl Work for Fixed {
        type Error = &'static str;
        type Pending = Done;
        fn submit(&self, passes: usize) -> Result<Done, &'static str> {
            Ok(Done(self.0 * passes as f64))
        }
        fn calls_per_pass(&self) -> usize {
            1
        }
        fn gauge(&self) -> Result<Box<dyn Gauge<&'static str>>, &'static str> {
            unreachable!("a measurement holds no clock, so never gauges")
        }
    }

    /// A measurement takes the samples it was asked for, each covering the
    /// device time asked of it.
    #[test]
    fn a_measurement_takes_the_samples_asked_for() {
        let how = Measure {
            samples: NonZeroUsize::new(3).unwrap(),
            min_sample_seconds: 0.25,
        };
        let measured = measure(Fixed(0.0625), &how).unwrap();
        assert_eq!(measured.samples, [0.0625; 3]);
        assert_eq!(measured.repetitions, 4);
    }
}
