//! Owned numerical submissions and completed selection decoding. Every flight
//! holds the domain's [`StateBindings`] until its finish returns them.

use super::*;
use crate::programs::SubmittedTarget;

pub struct VisionFlight<F: ProgramFamily = NativeFamily> {
    pub(super) request: RequestId,
    pub(super) image: ImageRef,
    pub(super) submission: F::VisionSubmission,
    pub(super) started: Instant,
    pub(super) bindings: StateBindings<F>,
}

impl<F: ProgramFamily> VisionFlight<F> {
    pub fn completion(&mut self) -> &mut dyn Completion {
        self.submission.completion()
    }
}

pub struct HeadFlight<F: ProgramFamily = NativeFamily> {
    /// Per slot: request, entry rows, proposals.
    pub(super) requests: Vec<(RequestId, usize, usize)>,
    /// Selections per slot in the submitted graph.
    pub(super) steps: usize,
    pub(super) submission: F::HeadSubmission,
    pub(super) started: Instant,
    /// Optional one-flight launch attribution for diagnosing a proposing head.
    pub(super) launch_trace: Option<seismic::SubmissionTrace>,
    pub(super) bindings: StateBindings<F>,
}

impl<F: ProgramFamily> HeadFlight<F> {
    pub fn completion(&mut self) -> &mut dyn Completion {
        self.submission.completion()
    }
}

/// A prompt chunk's drafter entry, drafted on the device behind its target
/// flight, conditioned by the flight's feature output. Only completion may
/// resolve its head advance; a final target token may publish earlier.
pub(super) struct PrimingFlight<H> {
    pub(super) request: RequestId,
    pub(super) submission: H,
    /// For a claimed lookahead: the accepted head state its successor
    /// advance attaches to at finish, or `None` when nobody claimed it (its
    /// rows are discarded). `None` for an ordinary flight.
    pub(super) continuation: Option<Option<InFlightState>>,
}

/// Per request of a target step: its rows, conditioning, work kind and
/// committed rows.
pub(super) type TargetRequests = Vec<(
    RequestId,
    usize,
    Option<crate::ConditioningRef>,
    WorkKind,
    usize,
)>;

/// One submitted target step: a group's flight, or a step queued behind one.
pub(super) struct TargetWork<F: ProgramFamily> {
    pub(super) requests: TargetRequests,
    pub(super) submission: F::TargetSubmission,
    /// The drafter entry of the step's prompt chunk, when it primes one.
    pub(super) priming: Option<PrimingFlight<F::HeadSubmission>>,
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
    /// Identifies the step a lookahead continues.
    pub(super) id: u64,
    /// For a claimed lookahead, per slot: the accepted state its successor
    /// advance attaches to at finish, or `None` for a slot nobody claimed
    /// (its rows are discarded). `None` for an ordinary step.
    pub(super) continuation: Option<Vec<Option<InFlightState>>>,
}

impl<F: ProgramFamily> TargetWork<F> {
    /// A step just submitted at `started`: runnable from then, with no
    /// drafter entry yet and no claimed continuation.
    pub(super) fn submitted(
        id: u64,
        requests: TargetRequests,
        submission: F::TargetSubmission,
        launch_trace: Option<seismic::SubmissionTrace>,
        started: Instant,
        previous_selection: Option<Instant>,
    ) -> Self {
        Self {
            requests,
            submission,
            priming: None,
            launch_trace,
            started,
            runnable: started,
            previous_selection,
            id,
            continuation: None,
        }
    }
}

pub struct TargetFlight<F: ProgramFamily = NativeFamily> {
    pub(super) work: TargetWork<F>,
    /// Carries the lookahead queued behind `work`.
    pub(super) bindings: StateBindings<F>,
}

impl<F: ProgramFamily> TargetFlight<F> {
    pub fn completion(&mut self) -> &mut dyn Completion {
        self.work.submission.completion()
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

/// The binding right returns only after every submitted use is complete.
pub enum TargetContinuation<F: ProgramFamily> {
    Ready(StateBindings<F>),
    Priming(PrimingCompletionFlight<F>),
}

/// A final prompt entry finishing after the first token was published.
/// The binding right stays here, so neither lane can be checkpointed,
/// evicted, or reused before the head advance resolves.
pub struct PrimingCompletionFlight<F: ProgramFamily> {
    pub(super) work: PrimingFlight<F::HeadSubmission>,
    pub(super) target_end: usize,
    pub(super) bindings: StateBindings<F>,
    pub(super) started: Instant,
}

impl<F: ProgramFamily> PrimingCompletionFlight<F> {
    pub fn completion(&mut self) -> &mut dyn Completion {
        self.work.submission.completion()
    }
}

impl<F: ProgramFamily> TargetFlight<F> {
    /// Separate entry completion when the caller publishes this final
    /// chunk's target result before waiting for drafter history.
    pub(super) fn detach_priming(&mut self) -> Option<(PrimingFlight<F::HeadSubmission>, usize)> {
        let work = self.work.priming.take()?;
        let advances = self.work.submission.launch().advances();
        let target = &advances[0];
        Some((work, target.position() + target.rows()))
    }
}
