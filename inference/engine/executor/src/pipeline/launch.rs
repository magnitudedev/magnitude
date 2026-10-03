//! Join already-checked device-local launches without equating local addresses.
use crate::{InvariantError, TargetLaunchCore, TargetTokens, ValidatedTargetLaunch};
use magnitude_state::TentativeAdvance;
use seismic::BackendName;

pub type PipelineLaunchRefusal = Box<([ValidatedTargetLaunch; 2], InvariantError)>;

pub struct ValidatedPipelineLaunch {
    pub(crate) stages: [ValidatedTargetLaunch; 2],
}
impl ValidatedPipelineLaunch {
    /// Refusal retains both unchanged checked launches. Neither reservation
    /// release nor this logical comparison rolls back device writes.
    pub fn new(
        stages: [ValidatedTargetLaunch; 2],
        devices: [&seismic::Device; 2],
    ) -> Result<Self, PipelineLaunchRefusal> {
        if let Err(error) = validate(&stages, devices) {
            return Err(Box::new((stages, error)));
        }
        Ok(Self { stages })
    }
}
fn invalid(detail: &str) -> InvariantError {
    InvariantError {
        context: "pipeline launch",
        detail: detail.into(),
    }
}
fn validate(
    stages: &[ValidatedTargetLaunch; 2],
    devices: [&seismic::Device; 2],
) -> Result<(), InvariantError> {
    let [a, b] = stages.each_ref().map(ValidatedTargetLaunch::core);
    if stages[0].domain() == stages[1].domain()
        || devices[0].info().selector == devices[1].info().selector
        || devices.iter().any(|d| d.backend() != BackendName::Cuda)
        || !a.store().belongs_to_device(devices[0])
        || !b.store().belongs_to_device(devices[1])
    {
        return Err(invalid(
            "requires exact distinct CUDA owners and local resource domains",
        ));
    }
    for core in [a, b] {
        if core.batch().actual_slots() != 1
            || !matches!(core.batch().class().rows(), 1 | 2)
            || core.batch().class().segments() != 1
            || !matches!(core.tokens(), TargetTokens::Host)
            || !matches!(core.advances(), [TentativeAdvance::Accepted(_)])
            || core.conditioning().iter().any(Option::is_some)
            || core.conditioning_slices().iter().any(|s| !s.is_empty())
            || core.advances()[0].bindings().stop != core.batch().actual_rows()
        {
            return Err(invalid(
                "requires one ordinary unconditioned whole-step request",
            ));
        }
    }
    if identity(a) != identity(b) {
        return Err(invalid(
            "logical tokens, coordinates, row extent or position disagree",
        ));
    }
    if a.batch().demand_bits().iter().any(|bits| *bits != 0) {
        return Err(invalid("only final stage may select or publish readout"));
    }
    Ok(())
}

/// Only common logical metadata from a checked launch. KV plane tables, banks
/// and history addresses intentionally do not appear in this identity.
#[derive(Debug, PartialEq, Eq)]
struct LogicalRows<'a> {
    rows: usize,
    class_rows: usize,
    position: usize,
    tokens: &'a [i32],
    coordinates: &'a [[i32; 4]],
    row_slots: &'a [i32],
    segments: &'a [[i32; 2]],
}
fn identity(core: &TargetLaunchCore) -> LogicalRows<'_> {
    let batch = core.batch().upload();
    let rows = batch.actual_rows;
    LogicalRows {
        rows,
        class_rows: batch.class.rows(),
        position: core.advances()[0].position(),
        tokens: &batch.tokens[..rows],
        coordinates: &batch.coordinates[..rows],
        row_slots: &batch.row_slots[..rows],
        segments: &batch.segments[..1],
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn common_logical_identity_detects_feedback_and_position_mismatch() {
        let make = || LogicalRows {
            rows: 2,
            class_rows: 2,
            position: 17,
            tokens: &[11, 12],
            coordinates: &[[17, 17, 17, 17], [18, 18, 18, 18]],
            row_slots: &[0, 0],
            segments: &[[0, 2]],
        };
        assert_eq!(make(), make());
        let mut other = make();
        other.position += 1;
        assert_ne!(make(), other);
        let mut other = make();
        other.tokens = &[11, 13];
        assert_ne!(make(), other);
        let mut other = make();
        other.coordinates = &[[16; 4], [18; 4]];
        assert_ne!(make(), other);
        let mut other = make();
        other.segments = &[[0, 1]];
        assert_ne!(make(), other);
        let mut other = make();
        other.class_rows = 1;
        assert_ne!(make(), other);
        let mut other = make();
        other.row_slots = &[0, 1];
        assert_ne!(make(), other);
    }
}
