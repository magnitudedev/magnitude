//! A tuning point's slot in `seismic-measure`'s timing session
//! ([`PointTiming`]): a [`KernelPoint`] placed for one configuration at one
//! workload, with what tuning keeps beside it (the point's initializer, the
//! prices of its operation classes) and its numerical observation.
//!
//! The session, the clock and the statistics are `seismic-measure`'s; this
//! is only what tuning observes of them and runs besides them.

use super::budget::PointPrices;
use super::point::KernelPoint;
use super::{CallError, NativePrepared, StandaloneCalls};
use crate::api::kernel::EncodedArgs;
use crate::api::tensor::TensorInner;
use seismic_measure::{Deadline, Measurement, Slot, Slotted, WorkError};
use std::cell::Cell;
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// One point's calls, placed once and submitted for every sample, with
/// what tuning observes of them.
pub(crate) struct PointTiming<'a> {
    slot: Slot<KernelPoint>,
    pub(super) initialize: Option<super::validation::Initializer<'a>>,
    placement_seconds: f64,
    prices: Rc<Cell<PointPrices>>,
}

/// Result storage of each point's argument sets, shared by every placement at
/// that point in one tuning run. Samples complete one at a time, so the
/// configurations placed at a point can write the same results. Numerical
/// observation clears that storage before each candidate's first invocation.
/// Points are named by label, so any subset of a run's points finds its own
/// storage.
#[derive(Default)]
pub(crate) struct OutputPool(HashMap<String, Vec<Vec<Arc<TensorInner>>>>);

impl OutputPool {
    /// The storage of the argument sets of the point labeled `point`.
    pub(crate) fn at(&mut self, point: &str) -> &mut Vec<Vec<Arc<TensorInner>>> {
        self.0.entry(point.to_owned()).or_default()
    }
}

impl Slotted for PointTiming<'_> {
    type Work = KernelPoint;

    fn slot(&self) -> &Slot<KernelPoint> {
        &self.slot
    }

    fn slot_mut(&mut self) -> &mut Slot<KernelPoint> {
        &mut self.slot
    }

    /// One timed submission is an observation of the sample class.
    fn sampled(&self, seconds: f64) {
        self.observe(|prices| prices.sample.observe(seconds));
    }

    /// Raising the device for a session this point opens is one
    /// observation of the warm class.
    fn warmed(&self, seconds: f64) {
        self.observe(|prices| prices.warm.observe(seconds));
    }
}

impl<'a> PointTiming<'a> {
    /// Validate and place every call of `rotation`, then of its cold
    /// `copies` ([`super::point::cold_copies`]). Results of each argument
    /// set are allocated once and reused by every sample; the calls share
    /// one scratch arena, held while the timing is.
    pub(crate) fn new(
        kernel: &Arc<NativePrepared>,
        rotation: Vec<EncodedArgs>,
        copies: Vec<EncodedArgs>,
    ) -> Result<Self, CallError> {
        Self::place(kernel, rotation, copies, None)
    }

    /// Place a configuration against a point's shared result storage
    /// ([`OutputPool`]).
    pub(crate) fn reusing_outputs(
        kernel: &Arc<NativePrepared>,
        rotation: Vec<EncodedArgs>,
        copies: Vec<EncodedArgs>,
        outputs: &mut Vec<Vec<Arc<TensorInner>>>,
    ) -> Result<Self, CallError> {
        Self::place(kernel, rotation, copies, Some(outputs))
    }

    fn place(
        kernel: &Arc<NativePrepared>,
        rotation: Vec<EncodedArgs>,
        copies: Vec<EncodedArgs>,
        outputs: Option<&mut Vec<Vec<Arc<TensorInner>>>>,
    ) -> Result<Self, CallError> {
        let began = Instant::now();
        let point = KernelPoint::place(kernel, rotation, copies, outputs)?;
        Ok(Self {
            slot: Slot::new(point),
            initialize: None,
            placement_seconds: began.elapsed().as_secs_f64(),
            prices: Rc::new(Cell::new(PointPrices::default())),
        })
    }

    /// Observe this placement's operations in its point's classes.
    pub(super) fn track(&mut self, prices: Rc<Cell<PointPrices>>) {
        self.prices = prices;
    }

    /// The placement is complete: placing its calls and `restoring` its
    /// point's state are one observation of the placement class.
    pub(super) fn placed(&self, restoring: Duration) {
        self.observe(|prices| {
            prices
                .placement
                .observe(self.placement_seconds + restoring.as_secs_f64())
        });
    }

    fn observe(&self, update: impl FnOnce(&mut PointPrices)) {
        let mut prices = self.prices.get();
        update(&mut prices);
        self.prices.set(prices);
    }

    /// Declaration ordinals of the launches that do work at this point: active
    /// (`when`) with a nonempty grid in some call of the rotation.
    pub(crate) fn working_launches(&self) -> Vec<usize> {
        let mut working = self
            .slot
            .work()
            .calls()
            .iter()
            .flat_map(|call| {
                call.launches
                    .geometry
                    .iter()
                    .enumerate()
                    .filter(|(_, geometry)| {
                        geometry.is_some_and(|geometry| {
                            geometry.groups.iter().all(|groups| *groups > 0)
                        })
                    })
                    .map(|(ordinal, _)| ordinal)
            })
            .collect::<Vec<_>>();
        working.sort_unstable();
        working.dedup();
        working
    }

    pub(super) fn prepared(&self) -> &Arc<NativePrepared> {
        self.slot.work().kernel()
    }

    /// Distinct bytes the point's rotation reads and writes.
    pub(crate) fn rotation_bytes(&self) -> u64 {
        self.slot.work().rotation_bytes()
    }

    /// The samples so far as a measurement. Tuning counts or matches its
    /// samples and samples to no stopping rule, so it has no interval.
    pub(crate) fn measurement(&self) -> Measurement {
        self.slot.measurement(None)
    }

    /// Run every call of the rotation once from pristine state and hand
    /// what it published to `observe`: the placement's numerical
    /// observations, before any timed pass or subsequent candidate can
    /// overwrite the result and state buffers. Nothing here is timed: each
    /// call runs alone with the host before and after it, which is not how
    /// a sample runs (`seismic_measure::session`).
    pub(super) fn observe_first(
        &mut self,
        deadline: Deadline,
        mut observe: impl FnMut(
            usize,
            Vec<crate::api::kernel::DecodedValue>,
            &EncodedArgs,
        ) -> Result<(), super::tune::Exclusion>,
    ) -> Result<(), WorkError<super::tune::Exclusion>> {
        let point = self.slot.work();
        let kernel = point.kernel();
        for (rotation, call) in point.calls().iter().enumerate() {
            deadline.check(Duration::ZERO)?;
            let witness = Instant::now();
            use super::tune::Exclusion;
            if let Some(initialize) = &self.initialize {
                initialize.borrow_mut()().map_err(Exclusion::execution)?;
            }
            // Match fresh-call zeroed result storage. A pooled output must not
            // let a missing write inherit the preceding candidate's correct bytes.
            // Initialization remains outside the submitted device interval.
            for result in &call.results {
                let bytes = usize::try_from(result.byte_len())
                    .map_err(|_| Exclusion::Execution("result bytes exceed usize".into()))?;
                result
                    .write_from_host(&vec![0; bytes])
                    .map_err(Exclusion::execution)?;
            }
            kernel.reset_scalar_results().map_err(Exclusion::execution)?;
            deadline.check(Duration::ZERO)?;
            StandaloneCalls {
                kernel,
                calls: std::slice::from_ref(call),
            }
            .submit(1)
            .and_then(|submission| submission.wait())
            .map_err(Exclusion::execution)?;
            let actual = kernel.read_call_results(call).map_err(Exclusion::execution)?;
            let observed = observe(rotation, actual.into_values(), &point.arguments()[rotation]);
            // Restoring state, the call, its readback and its observation
            // are one witness.
            self.observe(|prices| prices.witness.observe(witness.elapsed().as_secs_f64()));
            observed?;
        }
        Ok(())
    }
}
