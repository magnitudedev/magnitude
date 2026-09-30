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
    /// Optional one-flight launch attribution for diagnosing a proposing head.
    pub(super) launch_trace: Option<seismic::SubmissionTrace>,
}

impl<S: ProgramSubmission<CompletedWork = crate::CompletedHeadWork>> HeadFlight<S> {
    pub fn completion(&mut self) -> &mut dyn Completion {
        self.submission.completion()
    }
}

/// A prompt chunk's drafter entry, drafted on the device behind its target
/// flight, conditioned by the flight's feature output. It commits with the
/// chunk.
pub(super) struct PrimingFlight<H> {
    pub(super) request: RequestId,
    pub(super) submission: H,
    /// For a claimed lookahead: the accepted head state its successor
    /// advance attaches to at finish, or `None` when nobody claimed it (its
    /// rows are discarded). `None` for an ordinary flight.
    pub(super) continuation: Option<Option<InFlightState>>,
}

pub struct TargetFlight<
    S: ProgramSubmission<CompletedWork = crate::CompletedTargetWork> = <NativeFamily as ProgramFamily>::TargetSubmission,
    H: ProgramSubmission<CompletedWork = crate::CompletedHeadWork> = <NativeFamily as ProgramFamily>::HeadSubmission,
> {
    pub(super) requests: Vec<(
        RequestId,
        usize,
        Option<crate::ConditioningRef>,
        WorkKind,
        usize,
    )>,
    pub(super) submission: S,
    /// The drafter entry of the flight's prompt chunk, when it primes one.
    pub(super) priming: Option<PrimingFlight<H>>,
    /// Optional attribution for one prefill flight selected by diagnostics.
    pub(super) launch_trace: Option<seismic::SubmissionTrace>,
    pub(super) started: Instant,
    /// When the device could begin this step: its submission, or for a step
    /// queued behind its predecessor (lookahead), the predecessor's
    /// completion. The step's physical duration is measured from here, so
    /// pipelined steps never charge their predecessor's time again.
    pub(super) runnable: Instant,
    /// When the domain last read a selection before this step was submitted.
    pub(super) previous_selection: Option<Instant>,
    /// Identifies the flight a lookahead continues.
    pub(super) id: u64,
    /// For a claimed lookahead, per slot: the accepted state its successor
    /// advance attaches to at finish, or `None` for a slot nobody claimed
    /// (its rows are discarded). `None` for an ordinary flight.
    pub(super) continuation: Option<Vec<Option<InFlightState>>>,
}

impl<
        S: ProgramSubmission<CompletedWork = crate::CompletedTargetWork>,
        H: ProgramSubmission<CompletedWork = crate::CompletedHeadWork>,
    > TargetFlight<S, H>
{
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
            // 3: a draft declined its proposal (`draft_confidence`).
            let status = u8::try_from(status)
                .ok()
                .filter(|value| *value <= 3)
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
