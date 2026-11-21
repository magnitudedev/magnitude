//! Owned numerical submissions and completed selection decoding. Every flight
//! holds the domain's [`StateBindings`] until its finish returns them.

use super::*;
use crate::programs::SubmittedTarget;

/// The host's turn after a flight finished: until the next launch, the host
/// reconciles the flight and prepares that launch for the flight's requests.
/// The owner's own work between flights, which serves no request (memory
/// observation and what it releases or grows), runs inside the turn but is
/// no part of it: `owner` is the time it took.
pub(super) struct HostTurn {
    since: Instant,
    owner: Duration,
    requests: Vec<RequestId>,
}

#[derive(Debug)]
struct OpenVerifyTurn {
    since: Instant,
    owner: Duration,
    begun_epoch: u64,
    requests: Vec<RequestId>,
}

/// The host's turns after verify passes (§7.1): a verify pass's verdict
/// decides the next pass's tokens, so nothing is queued behind it, and from
/// its completion to the next target launch of one of its requests (the
/// host reconciling it, drafting the next guesses in a head transaction and
/// preparing that launch) the device idles on the host. The mean of the
/// turns over the bank-wide turnover's epoch window
/// (`experts::Window<Turns>` rolling on `bank_epoch`) is the regime's `O`
/// (`magnitude_cost_model::Regime::overhead`). A turn the owner idled in,
/// or another launch ended first, is no verify pass's; a turn of a load
/// with no expert bank prices no regime and is not timed. The owner's own
/// work in the turn (`owner_worked`) is excluded, and a turn spanned by a
/// bank-epoch transition (such as a bank resize) is dropped without rolling
/// the window so clean pre-resize turns survive.
#[derive(Debug)]
pub(super) struct VerifyTurns {
    open: Option<OpenVerifyTurn>,
    latest: crate::experts::Window<Turns>,
}

impl Default for VerifyTurns {
    fn default() -> Self {
        Self { open: None, latest: crate::experts::Window::new(0) }
    }
}

/// Timed turns' seconds and how many, counted per bank-wide epoch in a
/// [`Window`](crate::experts::Window).
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct Turns {
    seconds: f64,
    turns: u64,
}

impl crate::experts::Tally for Turns {
    fn is_empty(&self) -> bool {
        self.turns == 0
    }
}

impl VerifyTurns {
    /// A verify pass of `requests` completed at `since` at bank-wide `epoch`
    /// (none without a bank).
    pub(super) fn begin(&mut self, since: Instant, epoch: Option<u64>, requests: Vec<RequestId>) {
        self.open = epoch.map(|begun_epoch| OpenVerifyTurn {
            since,
            owner: Duration::ZERO,
            begun_epoch,
            requests,
        });
    }

    /// The owner spent `spent` on its own work during the open verify turn:
    /// excluded from the turn's duration, mirroring [`HostTurn`].
    pub(super) fn owner_worked(&mut self, spent: Duration) {
        if let Some(turn) = &mut self.open {
            turn.owner += spent;
        }
    }

    /// A target launch of `requests` submitted at `started` ends the open
    /// turn: timed where it carries one of the turn's requests and the
    /// bank-wide `epoch` did not change during the turn, counted into
    /// `epoch` (none without a bank).
    pub(super) fn launched(&mut self, requests: &[RequestId], started: Instant, epoch: Option<u64>) {
        let Some(turn) = self.open.take() else {
            return;
        };
        let Some(epoch) = epoch else {
            return;
        };
        if epoch != turn.begun_epoch {
            return;
        }
        if requests.iter().any(|request| turn.requests.contains(request)) {
            let seconds = started
                .saturating_duration_since(turn.since)
                .saturating_sub(turn.owner)
                .as_secs_f64();
            let tally = self.latest.count(epoch);
            tally.seconds += seconds;
            tally.turns += 1;
        }
    }

    /// The open turn ends untimed: the owner idled, or the device ran
    /// other work.
    pub(super) fn interrupt(&mut self) {
        self.open = None;
    }

    /// `O`: a turn's mean over the bank-wide epoch window at `epoch`; none
    /// before one is timed there.
    pub(super) fn mean(&self, epoch: u64) -> Option<f64> {
        let (first, second) = self.latest.snapshot(epoch)?;
        let seconds = first.seconds + second.map_or(0.0, |t| t.seconds);
        let turns = first.turns + second.map_or(0, |t| t.turns);
        (turns > 0).then(|| (seconds / turns as f64).max(0.))
    }
}

impl<F: ProgramFamily> ExecutorDomain<F> {
    /// A finished flight of `requests` hands the host its turn.
    pub(super) fn begin_host_turn(&mut self, requests: impl IntoIterator<Item = RequestId>) {
        self.host_turn = Some(HostTurn {
            since: Instant::now(),
            owner: Duration::ZERO,
            requests: requests.into_iter().collect(),
        });
    }

    /// When the service of a launch of `requests` submitted at `submitted`
    /// starts: where the flight before it finished, when that flight
    /// carried one of its requests and the owner did not idle since, so a
    /// launch is charged the host work that prepared it (a round's host
    /// time is part of its cost), less the owner's own work in that turn;
    /// at its submission otherwise, so neither idle time nor another
    /// request's work is charged to it.
    pub(super) fn service_start(&mut self, requests: &[RequestId], submitted: Instant) -> Instant {
        match self.host_turn.take() {
            Some(turn) if requests.iter().any(|request| turn.requests.contains(request)) => {
                turn.since + turn.owner
            }
            _ => submitted,
        }
    }

    /// The owner spent `spent` on its own work, which serves no request: the
    /// running host and verify turns, if any, are charged none of it.
    pub fn owner_worked(&mut self, spent: Duration) {
        if let Some(turn) = &mut self.host_turn {
            turn.owner += spent;
        }
        self.verify_turns.owner_worked(spent);
    }

    /// The owner has no work: time until its next launch is idle, not
    /// service, and no verify pass's turn.
    pub fn host_idle(&mut self) {
        self.host_turn = None;
        self.verify_turns.interrupt();
    }
}

pub struct VisionFlight<F: ProgramFamily = NativeFamily> {
    pub(super) request: RequestId,
    pub(super) image: ImageRef,
    pub(super) submission: F::VisionSubmission,
    pub(super) started: Instant,
    /// The epoch the flight's launches are timed across ([`LaunchEpoch`]).
    pub(super) epoch: Option<LaunchEpoch>,
    pub(super) bindings: StateBindings,
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
    /// The epoch the flight's launches are timed across ([`LaunchEpoch`]).
    pub(super) epoch: Option<LaunchEpoch>,
    /// Optional one-flight launch attribution for diagnosing a proposing head.
    pub(super) launch_trace: Option<seismic::SubmissionTrace>,
    pub(super) bindings: StateBindings,
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
    /// How the drafter entry's head advance resolves at finish.
    pub(super) continuation: PrimingContinuation,
}

/// How a [`PrimingFlight`]'s head advance resolves at finish.
pub(super) enum PrimingContinuation {
    /// An ordinary flight: its advance already owns the accepted head state.
    Ordinary,
    /// A claimed queued step whose head state nobody claimed: its successor
    /// rows are discarded.
    Unclaimed,
    /// A claimed queued step whose successor advance attaches to this
    /// accepted head state.
    Accepted(SequenceState),
}

/// A front head launch's parts secured before it launches, behind a target
/// step whose forwards chain (`crate::two_row`): per slot of its step, in
/// slot order, its request, its head advance `A` (one committed entry row,
/// then a scratch row per front row of the slot) and its entry rows'
/// selections; and its launch leases. A step launched first secures
/// [`HeldFront`]; a step queued behind another, successors of its
/// predecessor's head advances or of its drafter entry, which drop with it
/// where nobody claims it, its launch never run.
pub(super) struct FrontHead<A> {
    pub(super) slots: Vec<SecuredFrontSlot<A>>,
    pub(super) graph_workspace: NativeGraphWorkspaceLease,
    pub(super) graph_output: NativeGraphOutputLease,
}

pub(super) struct SecuredFrontSlot<A> {
    pub(super) request: RequestId,
    pub(super) advance: A,
    pub(super) selects: Vec<crate::SelectSpec>,
}

impl<A> FrontHead<A> {
    /// The same launch, each slot's advance `advance`'s of it.
    pub(super) fn map<B>(self, mut advance: impl FnMut(A) -> B) -> FrontHead<B> {
        let Self { slots, graph_workspace, graph_output } = self;
        FrontHead {
            slots: slots
                .into_iter()
                .map(|slot| SecuredFrontSlot { request: slot.request, advance: advance(slot.advance), selects: slot.selects })
                .collect(),
            graph_workspace,
            graph_output,
        }
    }
}

/// A front head launch's advances that join nothing as it finishes, each
/// holding what its outcome commits: a launched step's, as its reservation
/// secures them (`reserve_front`), or a claimed closing chunk's.
pub(super) enum HeldFront {
    /// A decode launch's: each slot's advance over its request's accepted
    /// head state, taken out of the domain, which its outcome commits
    /// (`FrontAdvance::Step`).
    Accepted(FrontHead<OwnedStateAdvance>),
    /// The prompt's closing chunk's: its advance following the chunk's
    /// drafter entry, which its outcome joins once that commits
    /// (`FrontAdvance::Entry`).
    Entry(FrontHead<OwnedSuccessorAdvance>),
}

impl HeldFront {
    /// Return what it secured, unlaunched, to `head`: an advance over a
    /// request's accepted head state returns that state; one following a
    /// drafter entry drops, its rows and bank with it.
    pub(super) fn release(self, requests: &mut BTreeMap<RequestId, RequestEntry>) {
        match self {
            Self::Accepted(front) => {
                for slot in front.slots {
                    requests
                        .get_mut(&slot.request)
                        .and_then(RequestEntry::resident_mut)
                        .expect("invariant violated: a released front head's request is not resident")
                        .put_head(slot.advance.abort());
                }
            }
            Self::Entry(front) => drop(front),
        }
    }
}

/// The front head launch behind a step just launched or claimed: it
/// launches right behind the step, before the step queued behind it, once
/// that step's rows are known (`ExecutorDomain::queue_successor`). Its
/// advances' kind is paired with what they join as it finishes.
pub(super) enum DueFront {
    /// Its advances join nothing ([`HeldFront`]).
    Held(HeldFront),
    /// A claimed queued decode pass's: each slot's advance following its
    /// predecessor's, and per slot the accepted head state that advance
    /// attaches to as it finishes, none of a slot nobody claimed (its rows
    /// are discarded).
    Claimed(FrontHead<OwnedSuccessorAdvance>, Vec<Option<SequenceState>>),
}

impl DueFront {
    /// Its slots, their advances as the head lane takes them, and its
    /// leases; and what those advances join as it finishes.
    pub(super) fn into_launch(self) -> (FrontHead<TentativeAdvance>, FrontJoins) {
        match self {
            Self::Held(HeldFront::Accepted(head)) => (head.map(TentativeAdvance::Accepted), FrontJoins::Accepted),
            Self::Held(HeldFront::Entry(head)) => (head.map(TentativeAdvance::Successor), FrontJoins::Entry),
            Self::Claimed(head, states) => (head.map(TentativeAdvance::Successor), FrontJoins::Claimed(states)),
        }
    }
}

/// What the head advances of a front head launch on the device join as it
/// finishes (`finish_front`), by the kind of advance it launched
/// ([`DueFront::into_launch`]).
pub(super) enum FrontJoins {
    /// Advances over accepted head states ([`HeldFront::Accepted`]).
    Accepted,
    /// The closing chunk's advance following its drafter entry
    /// ([`HeldFront::Entry`]).
    Entry,
    /// A claimed decode pass's successor advances, and the states they
    /// attach to ([`DueFront::Claimed`]).
    Claimed(Vec<Option<SequenceState>>),
}

/// A front head launch on the device.
pub(super) struct FrontFlight<H> {
    pub(super) submission: H,
    pub(super) joins: FrontJoins,
}

/// One submitted target step: a group's flight, or a step queued behind one
/// (`queue`).
pub(super) struct TargetWork<F: ProgramFamily> {
    pub(super) requests: Vec<(
        RequestId,
        usize,
        Option<crate::ConditioningRef>,
        WorkKind,
        usize,
    )>,
    pub(super) submission: F::TargetSubmission,
    /// The drafter entry of the step's prompt chunk, when it primes one.
    pub(super) priming: Option<PrimingFlight<F::HeadSubmission>>,
    /// The front head launch on the device behind the step, when its
    /// forwards chain and it launched: a queued step's launches as it is
    /// claimed (`QueuedStep`).
    pub(super) front: Option<FrontFlight<F::HeadSubmission>>,
    /// Optional attribution for one prefill flight selected by diagnostics.
    pub(super) launch_trace: Option<seismic::SubmissionTrace>,
    pub(super) started: Instant,
    /// The epoch the flight's launches are timed across ([`LaunchEpoch`]).
    pub(super) epoch: Option<LaunchEpoch>,
    /// When the device could begin this step: its submission, or for a step
    /// queued behind its predecessor, the predecessor's completion. The step's physical duration is measured from here, so
    /// pipelined steps never charge their predecessor's time again.
    pub(super) runnable: Instant,
    /// When the domain last read a selection before this step was submitted.
    pub(super) previous_selection: Option<Instant>,
    /// Identifies the step a queued step follows.
    pub(super) id: u64,
    /// For a claimed queued step, per slot, what its advance continues at
    /// finish. `None` for an ordinary step.
    pub(super) continuation: Option<Vec<Source>>,
    /// Whether it runs a verify pass (`ExecutorDomain::verify_pass`): the
    /// host's turn after it is a verify turn (`VerifyTurns`). A queued step
    /// never is.
    pub(super) verifies: bool,
}

/// What a claimed queued step's slot continues at finish.
pub(super) enum Source {
    /// Nothing: nobody claimed the slot, and its rows are discarded.
    Unclaimed,
    /// The accepted state its successor advance attaches to.
    Accepted(SequenceState),
    /// The chain its chained chunk joins, which the chunk carries until its
    /// outcome reconciles or aborts.
    Chain(OwnedChainAdvance),
}

pub struct TargetFlight<F: ProgramFamily = NativeFamily> {
    pub(super) work: TargetWork<F>,
    /// The binding right, held while `work` is on the device.
    pub(super) bindings: StateBindings,
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
            match status {
                0 => Ok(Selected::Token(crate::TokenId(
                    u32::try_from(token).map_err(|_| "selected token is negative")?,
                ))),
                1 => Ok(Selected::Empty),
                2 => Ok(Selected::NonFinite),
                // 3: a draft declined its proposal (`draft_confidence`).
                3 => Ok(Selected::Declined),
                _ => Err("selection status is invalid"),
            }
        })
        .collect::<Result<Vec<_>, &str>>()
        .map_err(str::to_owned)
}

/// The binding right returns only after every submitted use is complete.
pub enum TargetContinuation<F: ProgramFamily> {
    Ready(StateBindings),
    Priming(PrimingCompletionFlight<F>),
}

/// A final prompt entry finishing after the first token was published.
/// The binding right stays here, so neither lane can be checkpointed,
/// evicted, or reused before the head advance resolves.
pub struct PrimingCompletionFlight<F: ProgramFamily> {
    pub(super) work: PrimingFlight<F::HeadSubmission>,
    pub(super) target_end: usize,
    pub(super) bindings: StateBindings,
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
        let target = &self
            .work
            .submission
            .launch()
            .slot(0)
            .expect("invariant violated: target launch has no slot")
            .advance;
        Some((work, target.position() + target.rows()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Owner work during an open verify turn is excluded from `Regime::overhead`,
    /// and a turn spanned by a bank-epoch change (e.g. a bank resize) is dropped
    /// without rolling the window or evicting clean pre-resize turns.
    #[test]
    fn verify_turns_excludes_owner_work_and_drops_epoch_spanning_turns() {
        let mut turns = VerifyTurns::default();
        let req = RequestId(1);
        let t0 = Instant::now();

        // Clean turn at epoch 0: 10 ms wall time with 7 ms of owner work -> 3 ms.
        turns.begin(t0, Some(0), vec![req]);
        turns.owner_worked(Duration::from_millis(7));
        turns.launched(&[req], t0 + Duration::from_millis(10), Some(0));
        let clean = turns.mean(0).expect("clean turn recorded");
        assert!((clean - 0.003).abs() < 1e-9, "expected 3 ms after subtracting owner work, got {clean}");

        // Resize-spanning turn (epoch 0 -> epoch 1) with a 25 ms pause must be
        // skipped AND must not roll the window or evict the clean 3 ms turn.
        turns.begin(t0 + Duration::from_millis(20), Some(0), vec![req]);
        turns.launched(&[req], t0 + Duration::from_millis(45), Some(1));
        let after_resize = turns.mean(1).expect("clean pre-resize turn survives at new epoch");
        assert!(
            (after_resize - 0.003).abs() < 1e-9,
            "resize-spanning turn must not contaminate or evict clean mean: {after_resize}"
        );

        // Owner work exceeding wall time saturates at 0 without panicking.
        turns.begin(t0 + Duration::from_millis(50), Some(1), vec![req]);
        turns.owner_worked(Duration::from_millis(10));
        turns.launched(&[req], t0 + Duration::from_millis(52), Some(1));
        let with_zero = turns.mean(1).expect("clean turn at epoch 1 recorded");
        assert!((with_zero - 0.0015).abs() < 1e-9, "mean of 3 ms (epoch 0) and 0 ms (epoch 1): {with_zero}");
    }
}
