//! Owned numerical submissions and completed selection decoding.

use super::*;

pub struct VisionFlight<S: ProgramSubmission<CompletedWork = crate::CompletedVisionWork> = <NativeFamily as ProgramFamily>::VisionSubmission> {
    pub(super) request: RequestId,
    pub(super) image: ImageRef,
    pub(super) submission: S,
    pub(super) started: Instant,
}

impl<S: ProgramSubmission<CompletedWork = crate::CompletedVisionWork>> VisionFlight<S> {
    pub fn completion(&mut self) -> &mut dyn Completion {
        self.submission.completion()
    }
}

pub struct HeadFlight<S: ProgramSubmission<CompletedWork = crate::CompletedHeadWork> = <NativeFamily as ProgramFamily>::HeadSubmission> {
    /// Per slot: request, entry rows, proposals.
    pub(super) requests: Vec<(RequestId, usize, usize)>,
    /// Selections per slot in the submitted graph.
    pub(super) steps: usize,
    pub(super) submission: S,
    pub(super) started: Instant,
}

impl<S: ProgramSubmission<CompletedWork = crate::CompletedHeadWork>> HeadFlight<S> {
    pub fn completion(&mut self) -> &mut dyn Completion {
        self.submission.completion()
    }
}

pub struct StateFlight<S: ProgramSubmission<CompletedWork = crate::CompletedStateWork> = <NativeFamily as ProgramFamily>::StateSubmission> {
    pub(super) request: RequestId,
    pub(super) submission: S,
    pub(super) started: Instant,
}

impl<S: ProgramSubmission<CompletedWork = crate::CompletedStateWork>> StateFlight<S> {
    pub fn completion(&mut self) -> &mut dyn Completion {
        self.submission.completion()
    }
}

pub struct TargetFlight<S: ProgramSubmission<CompletedWork = crate::CompletedTargetWork> = <NativeFamily as ProgramFamily>::TargetSubmission> {
    pub(super) requests: Vec<(
        RequestId,
        usize,
        Option<crate::ConditioningRef>,
        WorkKind,
        usize,
    )>,
    pub(super) submission: S,
    pub(super) started: Instant,
    /// When the domain last read a selection before this step was submitted.
    pub(super) previous_selection: Option<Instant>,
    pub(super) slots: Vec<Slot>,
    pub(super) conditioning_slices: Vec<Vec<crate::ConditioningSlice>>,
    /// Identifies the flight a lookahead continues.
    pub(super) id: u64,
    /// For a claimed lookahead, per slot: the accepted state its successor
    /// advance attaches to at finish, or `None` for a slot nobody claimed
    /// (its rows are discarded). `None` for an ordinary flight.
    pub(super) continuation: Option<Vec<Option<SequenceState>>>,
}

impl<S: ProgramSubmission<CompletedWork = crate::CompletedTargetWork>> TargetFlight<S> {
    pub fn completion(&mut self) -> &mut dyn Completion {
        self.submission.completion()
    }
}

pub(super) fn decode_selected(bytes: &[u8]) -> Result<Vec<Selected>, String> {
    if !bytes.len().is_multiple_of(8) {
        return Err("selection byte count is not a row multiple".into());
    }
    bytes
        .chunks_exact(8)
        .map(|row| {
            let token = i32::from_le_bytes(row[0..4].try_into().expect("four token bytes"));
            let status = i32::from_le_bytes(row[4..8].try_into().expect("four status bytes"));
            let status = u8::try_from(status)
                .ok()
                .filter(|value| *value <= 2)
                .ok_or("selection status is invalid")?;
            let token = if status == 0 {
                crate::TokenId(u32::try_from(token).map_err(|_| "selected token is negative")?)
            } else {
                crate::TokenId(0)
            };
            Ok(Selected { token, status })
        })
        .collect::<Result<Vec<_>, &str>>()
        .map_err(str::to_owned)
}
