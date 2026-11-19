//! Time for streamed decode's rates: the token clock, each tier's turnover,
//! the window every rate is measured over, and the holds of choices paid once
//! a change (`designs/streamed-decode-ranking.md` §3.1, §3.2, §3.6).
//!
//! Every lead, window, τ, stay and hold is a difference on one clock, and
//! every rate rolls on one turnover's epochs, so no reader keeps time of its
//! own.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex, MutexGuard};

use super::watcher::Step;

/// A position on the token axis: tokens emitted before a step.
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub(crate) struct Tokens(f64);

impl Tokens {
    pub(crate) const ZERO: Self = Self(0.0);

    /// Panics where `tokens` is not finite and at least zero: no position
    /// on the clock is either.
    pub(crate) fn new(tokens: f64) -> Self {
        assert!(tokens.is_finite() && tokens >= 0.0, "{tokens} tokens is no position on the clock");
        Self(tokens)
    }

    pub(crate) fn get(self) -> f64 {
        self.0
    }

    /// The tokens from `earlier` to `self`. Panics where `earlier` is
    /// later: the clock never runs back.
    pub(crate) fn since(self, earlier: Self) -> Self {
        assert!(earlier.0 <= self.0, "{earlier:?} is after {self:?}");
        Self(self.0 - earlier.0)
    }
}

/// What a decode step's requests emitted: one token for a front or plain
/// pass, one plus its accepted rows for a verify pass, the mean over
/// requests.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Advance {
    /// Known at registration: a front or plain pass's exact rows.
    Known(Tokens),
    /// A verify step of `requests` passes before every verdict: at least
    /// its exact row, 1. Final once each request is decided
    /// ([`Times::decided`]).
    Pending { requests: u32 },
}

/// A registered step's advance as the clock holds it.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Emitted {
    Known(Tokens),
    /// `decided` of `requests` verdicts in, `accepted` rows among them.
    Pending { requests: u32, decided: u32, accepted: u64 },
}

/// The token clock (§3.1). Written by the step's registration before its
/// launch is queued and by a verify verdict, both through [`Times`] so
/// that every turnover counts the step it registers; a verify pass is not
/// pipelined behind its own verdict, so every step's position is final
/// before its launch.
pub(crate) struct Clock {
    /// The latest `behind` registered steps, oldest first: the steps a
    /// reader may still ask for. Each is (step, position, advance).
    ring: VecDeque<(Step, Tokens, Emitted)>,
    behind: usize,
    /// The position after the newest step whose advance is known.
    total: Tokens,
}

impl Clock {
    /// A clock keeping the latest `behind` registered steps: its owner sizes
    /// it to the steps in flight plus the farthest a reader looks back.
    /// Panics where `behind` is 0: the latest step is always readable.
    pub(crate) fn new(behind: usize) -> Self {
        assert!(behind > 0, "a clock keeps at least the latest step");
        Self { ring: VecDeque::with_capacity(behind), behind, total: Tokens::ZERO }
    }

    /// Registers decode step `step` with what it emits; a prompt chunk is
    /// not registered and advances nothing. Panics where `step` is not after
    /// the latest registered, or its predecessor's advance is still pending,
    /// or the advance is below a step's minimum, one token from one request.
    fn register(&mut self, step: Step, advance: Advance) {
        if let Some(&(latest, _, emitted)) = self.ring.back() {
            assert!(step > latest, "step {step} registered after step {latest}");
            assert!(
                matches!(emitted, Emitted::Known(_)),
                "step {step} registered before step {latest}'s verdicts"
            );
        }
        let emitted = match advance {
            Advance::Known(tokens) => {
                assert!(tokens.get() >= 1.0, "step {step} emits {tokens:?}, below one token");
                Emitted::Known(tokens)
            }
            Advance::Pending { requests } => {
                assert!(requests > 0, "verify step {step} has no requests");
                Emitted::Pending { requests, decided: 0, accepted: 0 }
            }
        };
        if self.ring.len() == self.behind {
            self.ring.pop_front();
        }
        self.ring.push_back((step, self.total, emitted));
        if let Emitted::Known(tokens) = emitted {
            self.total = Tokens(self.total.0 + tokens.0);
        }
    }

    /// One request's verdict of verify step `step`: `accepted` of its rows
    /// stood. A discarded or closed request's pass is decided with 0 (its
    /// exact row), so the step's advance stays at least 1. The step's
    /// advance is the mean over its requests once all are decided. Panics
    /// where `step` is not pending, or has more verdicts than requests.
    fn decided(&mut self, step: Step, accepted: u32) {
        // Only the latest step can be pending: its successor registers
        // after its verdicts.
        let Some((latest, _, Emitted::Pending { requests, decided, accepted: rows })) = self.ring.back_mut() else {
            panic!("step {step} decided while no step is pending");
        };
        assert_eq!(*latest, step, "step {step} decided while step {latest} is pending");
        *decided += 1;
        *rows += u64::from(accepted);
        if *decided == *requests {
            let advance = Tokens::new(1.0 + *rows as f64 / f64::from(*requests));
            self.ring.back_mut().expect("the pending step").2 = Emitted::Known(advance);
            self.total = Tokens(self.total.0 + advance.0);
        }
    }

    /// `step`'s position. Panics where `step` has left the ring or was never
    /// registered: a reader past the ring is an invariant violation.
    pub(crate) fn at(&self, step: Step) -> Tokens {
        let found = self.ring.binary_search_by_key(&step, |&(registered, ..)| registered);
        let Ok(index) = found else {
            panic!("step {step} is not on the clock's ring of {:?}", self.ring.iter().map(|entry| entry.0).collect::<Vec<_>>());
        };
        self.ring[index].1
    }

    /// Tokens through the latest begun step, counted at its advance if known,
    /// else at its minimum, 1. Zero before any decode step. Every τ and ā
    /// reads this T.
    pub(crate) fn through_latest(&self) -> Tokens {
        match self.ring.back() {
            Some(&(_, position, Emitted::Pending { .. })) => Tokens(position.0 + 1.0),
            _ => self.total,
        }
    }

    /// The latest begun step's sequence; `None` before any decode step.
    pub(crate) fn latest(&self) -> Option<Step> {
        self.ring.back().map(|&(step, _, _)| step)
    }

    /// The latest begun step's position; `None` before any decode step.
    fn latest_position(&self) -> Option<Tokens> {
        self.ring.back().map(|&(_, position, _)| position)
    }
}

/// A tier of a group's records (§3.2): tier 0, what the GPU computes from
/// (the bank beside a host tier, the one store on unified memory), and
/// tier 1, the host tier beside a bank. Unified memory has tier 0 alone.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) enum Tier {
    Compute,
    Host,
}

/// Which consumer of τ is asking (§3.2).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Reader {
    /// Prices a move it may make: admission, promotion, a seed's
    /// displacement, a write-back, a shadow's simulated landing. The move is
    /// counted in.
    Move,
    /// Makes no move: a shrink's eviction order, telemetry.
    Observe,
}

/// A turnover as one reader sees it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Horizon {
    /// τ tokens, finite and positive.
    Tokens(Tokens),
    /// An observer before the first move: rank by the τ → ∞ limit order.
    Unbounded,
}

/// The counts of one complete epoch of a turnover.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct Span {
    pub(crate) tokens: f64,
    pub(crate) steps: u64,
    pub(crate) moves: u64,
}

/// The running epoch's counts; its tokens are the clock's since its start.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct Counts {
    steps: u64,
    moves: u64,
}

/// One tier's turnover (§3.2): per (group, tier), plus one bank-wide over
/// every tier-0 record for global rates. A move is a landing that fills or
/// replaces a record; seeds, shrink evictions and lends are not moves.
///
/// Its counts start at the latest begun step, that step counted in (at the
/// load, before any step, nothing is): so once any decode step has begun,
/// every window holds a step and T ≥ 1, a resize's included.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Turnover {
    /// Fewer than K moves since the load or the last resize: the window is
    /// all of it.
    Unturned { epoch: u64, since: Tokens, running: Counts },
    /// The last complete epoch and the running one.
    Turned { epoch: u64, previous: Span, since: Tokens, running: Counts },
}

impl Turnover {
    /// A turnover at `clock`'s latest begun step, nothing else counted.
    pub(crate) fn new(clock: &Clock) -> Self {
        Self::started(0, clock)
    }

    /// Epoch `epoch` restarted `Unturned` at `clock`'s latest begun step.
    fn started(epoch: u64, clock: &Clock) -> Self {
        match clock.latest_position() {
            Some(since) => Self::Unturned { epoch, since, running: Counts { steps: 1, moves: 0 } },
            None => Self::Unturned { epoch, since: Tokens::ZERO, running: Counts::default() },
        }
    }

    /// A move at `clock`'s T; closes the epoch where it brings the running
    /// moves to `records`. Returns the epoch now running. Panics where
    /// `records` is 0: a move fills a record.
    pub(crate) fn moved(&mut self, clock: &Clock, records: u64) -> u64 {
        assert!(records > 0, "a move into a tier of no records");
        let now = clock.through_latest();
        let (epoch, since, running) = match self {
            Self::Unturned { epoch, since, running } | Self::Turned { epoch, since, running, .. } => (*epoch, *since, running),
        };
        running.moves += 1;
        if running.moves >= records {
            let previous = Span { tokens: now.since(since).get(), steps: running.steps, moves: running.moves };
            *self = Self::Turned { epoch: epoch + 1, previous, since: now, running: Counts::default() };
        }
        self.epoch()
    }

    /// A decode step begun, counted in S.
    pub(crate) fn stepped(&mut self) {
        match self {
            Self::Unturned { running, .. } | Self::Turned { running, .. } => running.steps += 1,
        }
    }

    /// A resize: closes the running epoch and restarts the counts
    /// `Unturned` at `clock`'s latest begun step, so no epoch measured at
    /// the old capacity is read as a turnover at the new one. Returns the
    /// epoch now running.
    pub(crate) fn resized(&mut self, clock: &Clock) -> u64 {
        *self = Self::started(self.epoch() + 1, clock);
        self.epoch()
    }

    /// τ for `reader` with `records` records, T from `clock`. `None` only
    /// before the first decode step, when no reader asks; a caller there is
    /// an invariant violation and unwraps. Panics where `records` is 0: no
    /// reader ranks in a tier of no records.
    pub(crate) fn horizon(&self, reader: Reader, records: u64, clock: &Clock) -> Option<Horizon> {
        assert!(records > 0, "a horizon asked of a tier of no records");
        let window = self.window(clock)?;
        let capacity = records as f64 * window.tokens;
        Some(match (reader, window.moves) {
            (Reader::Observe, 0) => Horizon::Unbounded,
            (Reader::Observe, moves) => Horizon::Tokens(Tokens::new(capacity / moves as f64)),
            (Reader::Move, moves) => Horizon::Tokens(Tokens::new(capacity / (moves + 1) as f64)),
        })
    }

    /// ā: tokens per decode step over the window, T/S. Exactly 1 at the
    /// first step; `None` before it.
    pub(crate) fn tokens_per_step(&self, clock: &Clock) -> Option<f64> {
        self.window(clock).map(|window| window.tokens / window.steps as f64)
    }

    /// The epoch now running; it only grows.
    pub(crate) fn epoch(&self) -> u64 {
        match self {
            Self::Unturned { epoch, .. } | Self::Turned { epoch, .. } => *epoch,
        }
    }

    /// The window: the last complete epoch, if any, and the running one,
    /// through `clock`'s latest begun step; `None` while it holds no step.
    fn window(&self, clock: &Clock) -> Option<Span> {
        let (previous, since, running) = match self {
            Self::Unturned { since, running, .. } => (Span::default(), since, running),
            Self::Turned { previous, since, running, .. } => (*previous, since, running),
        };
        let window = Span {
            tokens: previous.tokens + clock.through_latest().since(*since).get(),
            steps: previous.steps + running.steps,
            moves: previous.moves + running.moves,
        };
        if window.steps == 0 {
            return None;
        }
        assert!(window.tokens >= window.steps as f64, "{window:?}: a step emits at least one token");
        Some(window)
    }
}

/// What a window counts: a tally that adds and is empty by default.
pub(crate) trait Tally: Default + Clone {
    fn is_empty(&self) -> bool;
}

/// A rate measured over the last epoch with an outcome and the running one
/// (§3.2): two tallies, rolled lazily on the epoch its owner names.
#[derive(Clone, Debug)]
pub(crate) struct Window<T: Tally> {
    epoch: u64,
    previous: T,
    current: T,
}

impl<T: Tally> Window<T> {
    pub(crate) fn new(epoch: u64) -> Self {
        Self { epoch, previous: T::default(), current: T::default() }
    }

    /// The tally to count into at `epoch`, after rolling. Panics where
    /// `epoch` regresses.
    pub(crate) fn count(&mut self, epoch: u64) -> &mut T {
        self.roll(epoch);
        &mut self.current
    }

    /// The window's two tallies, (previous, current), at `epoch`, after
    /// rolling; `None` until the first count with an outcome, never again
    /// after it. Panics where `epoch` regresses.
    pub(crate) fn read(&mut self, epoch: u64) -> Option<(&T, &T)> {
        self.roll(epoch);
        (!self.previous.is_empty() || !self.current.is_empty()).then_some((&self.previous, &self.current))
    }

    /// The window's non-empty tallies as of `epoch` without rolling in place:
    /// `(previous, current)` at `self.epoch`, or the surviving latest epoch
    /// once `epoch` has advanced past it. Panics where `epoch` regresses.
    pub(crate) fn snapshot(&self, epoch: u64) -> Option<(&T, Option<&T>)> {
        assert!(epoch >= self.epoch, "a window at epoch {} read at epoch {epoch}", self.epoch);
        if epoch > self.epoch {
            if !self.current.is_empty() {
                Some((&self.current, None))
            } else if !self.previous.is_empty() {
                Some((&self.previous, None))
            } else {
                None
            }
        } else if !self.previous.is_empty() || !self.current.is_empty() {
            Some((&self.previous, Some(&self.current)))
        } else {
            None
        }
    }

    /// Rolls to `epoch`: the current tally becomes the previous where it
    /// holds an outcome, else the previous stays the latest measurement.
    fn roll(&mut self, epoch: u64) {
        assert!(epoch >= self.epoch, "a window at epoch {} read at epoch {epoch}", self.epoch);
        if epoch > self.epoch {
            let current = std::mem::take(&mut self.current);
            if !current.is_empty() {
                self.previous = current;
            }
            self.epoch = epoch;
        }
    }
}

/// A window whose source may stop running (Δ_j, a hint lead's exclusive
/// catch, §3.5, §3.7): its rate holds its last measurement while the source
/// does not run, with its age, the tokens since it last counted.
#[derive(Clone, Debug)]
pub(crate) struct Aged<T: Tally> {
    window: Window<T>,
    counted: Option<Tokens>,
}

impl<T: Tally> Aged<T> {
    pub(crate) fn new(epoch: u64) -> Self {
        Self { window: Window::new(epoch), counted: None }
    }

    /// [`Window::count`], at clock `now`. Panics where `now` is before the
    /// last count.
    pub(crate) fn count(&mut self, epoch: u64, now: Tokens) -> &mut T {
        assert!(self.counted.is_none_or(|counted| counted <= now), "counted at {now:?} after {:?}", self.counted);
        self.counted = Some(now);
        self.window.count(epoch)
    }

    /// [`Window::read`], with the window's age at `now`: none until the
    /// first count, never again after it. Panics where `now` is before the
    /// last count.
    pub(crate) fn read(&mut self, epoch: u64, now: Tokens) -> Option<((&T, &T), Tokens)> {
        let counted = self.counted?;
        self.window.read(epoch).map(|tallies| (tallies, now.since(counted)))
    }
}

/// How long a choice is expected to stay (§3.6): time on test over finished
/// holds, the open one in the time.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Hold {
    finished_tokens: f64,
    finished: u64,
    latest: Tokens,
}

/// A hold's expectation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Held {
    /// No change yet: any positive gain pays.
    Unbounded,
    Tokens(Tokens),
}

impl Hold {
    /// A choice first made at `now`: its first hold opens.
    pub(crate) fn started(now: Tokens) -> Self {
        Self { finished_tokens: 0.0, finished: 0, latest: now }
    }

    /// The choice changed at `now`: the open hold finishes and another
    /// opens. Panics where `now` is before the latest change.
    pub(crate) fn changed(&mut self, now: Tokens) {
        self.finished_tokens += now.since(self.latest).get();
        self.finished += 1;
        self.latest = now;
    }

    /// H = (Σ finished holds + the open one to `now`) / finished holds:
    /// `Unbounded` before the first change. Panics where `now` is before
    /// the latest change.
    pub(crate) fn expected(&self, now: Tokens) -> Held {
        let open = now.since(self.latest).get();
        match self.finished {
            0 => Held::Unbounded,
            finished => Held::Tokens(Tokens::new((self.finished_tokens + open) / finished as f64)),
        }
    }

    /// Whether a change gaining `gain` a token repays `cost` tokens over the
    /// hold: any positive gain while `Unbounded`.
    pub(crate) fn pays(&self, now: Tokens, gain: f64, cost: f64) -> bool {
        gain > 0.0
            && match self.expected(now) {
                Held::Unbounded => true,
                Held::Tokens(hold) => gain * hold.get() >= cost,
            }
    }
}

/// The load's time, one lock shared by its writers and readers on every
/// thread (§3.1, §3.2): the clock, each (group, tier)'s turnover with its
/// records, and the bank-wide turnover over every tier-0 record. Writers:
/// the step's registration and verdicts ([`Times::register`],
/// [`Times::decided`]), the transitions that move (`HostRecords::landed`,
/// a promotion's completion, `Directory::written_back`), resizes. Readers:
/// the ranking and its shadows, the ledger, the gate, the directory, the
/// holds. Every operation is O(1) but a step's registration, O(groups).
#[derive(Clone)]
pub(crate) struct Time(Arc<Mutex<Times>>);

/// [`Time`]'s state.
pub(crate) struct Times {
    /// Read here; written only through [`Times::register`] and
    /// [`Times::decided`], so every turnover counts each step it registers.
    pub(crate) clock: Clock,
    /// Per group, per tier present (one on unified memory, two beside a
    /// host tier): the turnover and the tier's records now (its K).
    tiers: Vec<Vec<(Turnover, u64)>>,
    /// Every tier-0 record of the load, every tier-0 move: global rates
    /// (σ, Δ_j, the regime's and gate's terms) roll on its epochs. Its K
    /// is the sum of the groups' tier-0 records.
    bank: (Turnover, u64),
}

impl Time {
    /// Nothing counted: groups of `records[group]` records per tier present
    /// (`[tier 0]` or `[tier 0, tier 1]`), the clock keeping the latest
    /// `behind` steps ([`Clock::new`]).
    pub(crate) fn new(records: &[Vec<u64>], behind: usize) -> Self {
        let clock = Clock::new(behind);
        let tiers: Vec<Vec<(Turnover, u64)>> = records
            .iter()
            .map(|tiers| {
                assert!(matches!(tiers.len(), 1 | 2), "a group has tier 0 and at most a host tier: {tiers:?}");
                tiers.iter().map(|&records| (Turnover::new(&clock), records)).collect()
            })
            .collect();
        let bank = (Turnover::new(&clock), records.iter().map(|tiers| tiers[0]).sum());
        Self(Arc::new(Mutex::new(Times { clock, tiers, bank })))
    }

    /// The state, held for one read or write.
    pub(crate) fn lock(&self) -> MutexGuard<'_, Times> {
        self.0.lock().expect("the load's time")
    }
}

impl Times {
    /// Decode step `step` begun ([`Clock::register`]'s contract), before its
    /// launch is queued: placed on the clock and counted in every turnover's
    /// S.
    pub(crate) fn register(&mut self, step: Step, advance: Advance) {
        self.clock.register(step, advance);
        for (turnover, _) in self.tiers.iter_mut().flatten() {
            turnover.stepped();
        }
        self.bank.0.stepped();
    }

    /// One request's verdict of verify step `step` ([`Clock::decided`]'s
    /// contract).
    pub(crate) fn decided(&mut self, step: Step, accepted: u32) {
        self.clock.decided(step, accepted);
    }

    /// A move into `group`'s `tier` (§3.2: a landing that fills or replaces
    /// a record; never a seed, a shrink's eviction or a lend), at the clock's
    /// T. A tier-0 move is also a bank-wide move.
    pub(crate) fn moved(&mut self, group: usize, tier: Tier) {
        let Self { clock, tiers, bank } = self;
        let (turnover, records) = Self::tier(tiers, group, tier);
        turnover.moved(clock, *records);
        if tier == Tier::Compute {
            bank.0.moved(clock, bank.1);
        }
    }

    /// `group`'s `tier` resized to `records`: its turnover closes its epoch
    /// and restarts `Unturned`; a tier-0 resize does so for the bank-wide
    /// turnover too, at the bank's new sum.
    pub(crate) fn resized(&mut self, group: usize, tier: Tier, records: u64) {
        let Self { clock, tiers, bank } = self;
        let (turnover, held) = Self::tier(tiers, group, tier);
        let before = std::mem::replace(held, records);
        turnover.resized(clock);
        if tier == Tier::Compute {
            bank.1 = bank.1 - before + records;
            bank.0.resized(clock);
        }
    }

    /// τ of `group`'s `tier` for `reader` ([`Turnover::horizon`]).
    pub(crate) fn horizon(&self, group: usize, tier: Tier, reader: Reader) -> Option<Horizon> {
        let (turnover, records) = &self.tiers[group][Self::index(&self.tiers[group], group, tier)];
        turnover.horizon(reader, *records, &self.clock)
    }

    /// ā of `group`'s tier-0 turnover ([`Turnover::tokens_per_step`]): the
    /// spacing of a group's future decode steps.
    pub(crate) fn tokens_per_step(&self, group: usize) -> Option<f64> {
        self.tiers[group][0].0.tokens_per_step(&self.clock)
    }

    /// The epoch a group's rates roll on: its tier-0 turnover's (A4).
    pub(crate) fn group_epoch(&self, group: usize) -> u64 {
        self.tiers[group][0].0.epoch()
    }

    /// The epoch global rates roll on: the bank-wide turnover's.
    pub(crate) fn bank_epoch(&self) -> u64 {
        self.bank.0.epoch()
    }

    fn tier(tiers: &mut [Vec<(Turnover, u64)>], group: usize, tier: Tier) -> &mut (Turnover, u64) {
        let index = Self::index(&tiers[group], group, tier);
        &mut tiers[group][index]
    }

    /// `tier`'s index in `group`'s tiers. Panics where the group has no
    /// such tier: unified memory has no host tier.
    fn index(tiers: &[(Turnover, u64)], group: usize, tier: Tier) -> usize {
        let index = match tier {
            Tier::Compute => 0,
            Tier::Host => 1,
        };
        assert!(index < tiers.len(), "group {group} has no {tier:?} tier");
        index
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A count of outcomes.
    #[derive(Clone, Debug, Default, PartialEq)]
    struct Count(u64);

    impl Tally for Count {
        fn is_empty(&self) -> bool {
            self.0 == 0
        }
    }

    fn tokens(tokens: f64) -> Tokens {
        Tokens::new(tokens)
    }

    fn front() -> Advance {
        Advance::Known(tokens(1.0))
    }

    fn tau(times: &Times, group: usize, tier: Tier, reader: Reader) -> Horizon {
        times.horizon(group, tier, reader).expect("a decode step has begun")
    }

    #[test]
    fn a_front_step_advances_one_token_and_a_verify_step_its_accepted() {
        let time = Time::new(&[vec![4]], 8);
        let mut times = time.lock();
        assert_eq!(times.clock.through_latest(), Tokens::ZERO, "no decode step yet");
        times.register(1, front());
        assert_eq!((times.clock.at(1), times.clock.through_latest()), (tokens(0.0), tokens(1.0)));
        // A verify step counts at its minimum, 1, until decided.
        times.register(2, Advance::Pending { requests: 1 });
        assert_eq!((times.clock.at(2), times.clock.through_latest()), (tokens(1.0), tokens(2.0)));
        times.decided(2, 0);
        assert_eq!(times.clock.through_latest(), tokens(2.0), "nothing accepted: its exact row");
        times.register(3, Advance::Pending { requests: 1 });
        times.decided(3, 3);
        assert_eq!((times.clock.at(3), times.clock.through_latest()), (tokens(2.0), tokens(6.0)));
        // Step 4 is a prompt chunk: never registered, it advances nothing.
        times.register(5, Advance::Pending { requests: 2 });
        assert_eq!(times.clock.at(5), tokens(6.0));
        times.decided(5, 1);
        assert_eq!(times.clock.through_latest(), tokens(7.0), "pending until every request is decided");
        times.decided(5, 2);
        assert_eq!(times.clock.through_latest(), tokens(8.5), "the mean over requests: 1 + 3/2");
        times.register(6, Advance::Known(tokens(1.0)));
        assert_eq!(times.clock.at(6), tokens(8.5));
    }

    #[test]
    #[should_panic(expected = "not on the clock's ring")]
    fn a_step_past_the_ring_is_unreadable() {
        let mut clock = Clock::new(2);
        for step in 1..=3 {
            clock.register(step, front());
        }
        clock.at(1);
    }

    #[test]
    #[should_panic(expected = "before step 1's verdicts")]
    fn a_step_registers_only_after_its_predecessors_verdicts() {
        let mut clock = Clock::new(4);
        clock.register(1, Advance::Pending { requests: 1 });
        clock.register(2, front());
    }

    #[test]
    #[should_panic(expected = "no position on the clock")]
    fn a_non_finite_position_panics() {
        Tokens::new(f64::INFINITY);
    }

    #[test]
    fn tau_is_capacity_over_moves_with_the_move_priced() {
        const K: u64 = 3;
        let time = Time::new(&[vec![K]], 8);
        let mut times = time.lock();
        assert_eq!(times.horizon(0, Tier::Compute, Reader::Move), None, "before any decode step, no τ");
        assert_eq!(times.tokens_per_step(0), None);
        times.register(1, front());
        assert_eq!(times.tokens_per_step(0), Some(1.0), "ā is exactly 1 at the first step");
        assert_eq!(tau(&times, 0, Tier::Compute, Reader::Move), Horizon::Tokens(tokens(3.0)), "K·1/(0 + 1)");
        assert_eq!(tau(&times, 0, Tier::Compute, Reader::Observe), Horizon::Unbounded, "an observer before a move");
        times.register(2, Advance::Pending { requests: 1 });
        assert_eq!(tau(&times, 0, Tier::Compute, Reader::Move), Horizon::Tokens(tokens(6.0)), "T counts step 2 at 1");
        times.decided(2, 2);
        times.moved(0, Tier::Compute);
        // T = 4 over S = 2, M = 1.
        assert_eq!(times.tokens_per_step(0), Some(2.0));
        assert_eq!(tau(&times, 0, Tier::Compute, Reader::Move), Horizon::Tokens(tokens(6.0)), "3·4/(1 + 1)");
        assert_eq!(tau(&times, 0, Tier::Compute, Reader::Observe), Horizon::Tokens(tokens(12.0)), "3·4/1");
        times.moved(0, Tier::Compute);
        assert_eq!(times.group_epoch(0), 0, "unturned below K moves");
        times.moved(0, Tier::Compute);
        assert_eq!(times.group_epoch(0), 1, "the K-th move turns it");
        assert!(matches!(times.tiers[0][0].0, Turnover::Turned { .. }));
        // The window is the closed epoch: T = 4, M = 3.
        assert_eq!(tau(&times, 0, Tier::Compute, Reader::Move), Horizon::Tokens(tokens(3.0)), "3·4/(3 + 1)");
        times.register(3, front());
        times.moved(0, Tier::Compute);
        // Previous (4 tokens, 2 steps, 3 moves) plus running (1, 1, 1).
        assert_eq!(tau(&times, 0, Tier::Compute, Reader::Observe), Horizon::Tokens(tokens(3.0 * 5.0 / 4.0)));
        assert_eq!(times.tokens_per_step(0), Some(5.0 / 3.0));
    }

    #[test]
    fn a_window_keeps_the_last_two_epochs_with_outcomes() {
        let mut window = Window::<Count>::new(0);
        assert!(window.read(0).is_none(), "never counted");
        window.count(0).0 += 2;
        assert_eq!(window.read(0), Some((&Count(0), &Count(2))));
        window.count(1).0 += 3;
        assert_eq!(window.read(1), Some((&Count(2), &Count(3))), "a roll moves current to previous");
        assert_eq!(window.read(2), Some((&Count(3), &Count(0))));
        assert_eq!(window.read(5), Some((&Count(3), &Count(0))), "an empty epoch keeps previous");
        window.count(6).0 += 1;
        window.count(7);
        assert_eq!(window.read(9), Some((&Count(1), &Count(0))), "a counted rate never reads None again");
        assert_eq!(std::mem::size_of::<Window<Count>>(), 3 * std::mem::size_of::<u64>(), "two tallies and an epoch");
    }

    #[test]
    #[should_panic(expected = "read at epoch 1")]
    fn a_window_never_rolls_back() {
        let mut window = Window::<Count>::new(2);
        window.count(1);
    }

    #[test]
    fn an_aged_window_holds_its_measurement_with_its_age() {
        let mut aged = Aged::<Count>::new(0);
        assert!(aged.read(0, tokens(4.0)).is_none());
        aged.count(0, tokens(5.0)).0 += 1;
        let ((previous, current), age) = aged.read(3, tokens(12.0)).expect("counted");
        assert_eq!((previous, current, age), (&Count(1), &Count(0), tokens(7.0)));
    }

    #[test]
    fn a_resize_closes_the_epoch_and_restarts_the_turnover() {
        let time = Time::new(&[vec![2, 4]], 8);
        let mut times = time.lock();
        let mut rate = Window::<Count>::new(times.group_epoch(0));
        times.register(1, front());
        times.register(2, front());
        times.moved(0, Tier::Compute);
        times.moved(0, Tier::Compute);
        times.register(3, front());
        times.moved(0, Tier::Compute);
        assert_eq!(times.group_epoch(0), 1);
        rate.count(times.group_epoch(0)).0 += 1;
        // A grow: the epoch closes, and the counts restart at step 3.
        times.resized(0, Tier::Compute, 6);
        assert_eq!(times.group_epoch(0), 2);
        assert!(matches!(times.tiers[0][0].0, Turnover::Unturned { .. }), "restarted unturned");
        assert_eq!(rate.read(times.group_epoch(0)), Some((&Count(1), &Count(0))), "the window rolls once, keeping its tally");
        assert_eq!(times.tokens_per_step(0), Some(1.0), "S and T restart at the latest begun step");
        assert_eq!(tau(&times, 0, Tier::Compute, Reader::Move), Horizon::Tokens(tokens(6.0)), "K_new·1/(0 + 1)");
        assert_eq!(tau(&times, 0, Tier::Compute, Reader::Observe), Horizon::Unbounded, "no old move is read at the new K");
        times.register(4, front());
        times.moved(0, Tier::Compute);
        assert_eq!(tau(&times, 0, Tier::Compute, Reader::Move), Horizon::Tokens(tokens(6.0)), "6·2/(1 + 1)");
        // A shrink of the host tier closes its own epoch, and not tier 0's.
        let host = times.tiers[0][1].0.epoch();
        times.resized(0, Tier::Host, 1);
        assert_eq!((times.tiers[0][1].0.epoch(), times.group_epoch(0)), (host + 1, 2));
        assert_eq!(tau(&times, 0, Tier::Host, Reader::Move), Horizon::Tokens(tokens(1.0)));
    }

    #[test]
    fn global_rates_roll_on_the_bank_wide_turnover() {
        // The bank's K is every tier-0 record: 2 + 3.
        let time = Time::new(&[vec![2], vec![3, 8]], 8);
        let mut times = time.lock();
        for step in 1..=5 {
            times.register(step, front());
            times.moved(0, Tier::Compute);
        }
        times.moved(1, Tier::Host);
        assert_eq!(times.bank_epoch(), 1, "group 1 never moving in tier 0 does not hold the bank back");
        assert_eq!(times.group_epoch(0), 2);
        assert_eq!(times.group_epoch(1), 0, "no move, no aging");
        assert_eq!(tau(&times, 1, Tier::Compute, Reader::Observe), Horizon::Unbounded);
        assert_eq!(tau(&times, 1, Tier::Compute, Reader::Move), Horizon::Tokens(tokens(15.0)), "3·5/(0 + 1): every step since the load");
        assert_eq!(times.tokens_per_step(1), Some(1.0));
        // A tier-0 resize resizes the bank: 2 + 1 records, restarted at step 5.
        times.resized(1, Tier::Compute, 1);
        assert_eq!(times.bank_epoch(), 2);
        assert_eq!(times.bank.1, 3);
        assert_eq!(times.bank.0.horizon(Reader::Move, times.bank.1, &times.clock), Some(Horizon::Tokens(tokens(3.0))));
    }

    #[test]
    fn a_hold_is_time_on_test_over_changes() {
        let mut hold = Hold::started(tokens(5.0));
        assert_eq!(hold.expected(tokens(100.0)), Held::Unbounded, "no change yet");
        assert!(hold.pays(tokens(100.0), 1e-9, 1e9), "any positive gain pays while unbounded");
        assert!(!hold.pays(tokens(100.0), 0.0, 0.0), "no gain never pays");
        hold.changed(tokens(15.0));
        assert_eq!(hold.expected(tokens(15.0)), Held::Tokens(tokens(10.0)), "one finished hold of 10");
        hold.changed(tokens(45.0));
        assert_eq!(hold.expected(tokens(45.0)), Held::Tokens(tokens(20.0)), "(10 + 30)/2");
        assert_eq!(hold.expected(tokens(105.0)), Held::Tokens(tokens(50.0)), "the open hold lengthens it: (40 + 60)/2");
        assert!(hold.pays(tokens(105.0), 0.5, 25.0), "0.5 a token over 50 repays 25");
        assert!(!hold.pays(tokens(105.0), 0.5, 26.0));
    }
}
