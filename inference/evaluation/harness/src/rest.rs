//! Whether a load's expert bank is at rest across a timed span: the one
//! owner of what a steady prefill sample or decode step is, which every
//! cell's timing takes ([`BankRest`]).

use crate::engine::Engine;
use magnitude_executor::ExpertTargets;
use serde_json::{json, Value};

/// The most spans a cell runs while the bank sits unmoved and not at rest
/// ([`Progress`]): at them the bank is taken never to rest, and the cell is
/// refused with what moved.
pub const STEADY_ATTEMPTS: usize = 4;

/// What moved across a span that did not run steady: the bank before and
/// after it ([`BankRest::moved`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Moved {
    before: BankRest,
    after: BankRest,
}

impl Moved {
    /// What moved, as the cell reports it.
    pub fn record(&self) -> Value {
        let Self { before, after } = self;
        json!({
            "resizes_begun": after.resizes.begun - before.resizes.begun,
            "resizing": [before.resizes.running, after.resizes.running],
            "resize_advances": after.resizes.advances - before.resizes.advances,
            "slots": [before.slots, after.slots],
            "host_slots": [before.host, after.host],
            "cover_begun": after.cover.begun - before.cover.begun,
            "covering": [before.cover.reading, after.cover.reading],
        })
    }
}

/// Whether a bank that is not at rest is on its way there: the one bound on
/// how long a cell waits for it. A bank's slots and its host tier's grow
/// (or shrink) toward their targets (`ExpertTargets`) until they rest, as
/// a resize advances at flight boundaries and changes them as it installs
/// or joins them, so the wait has no span count; it ends where the moves
/// stop making progress, which is a defect of the bank, not a slow
/// measurement: a slot count that turns back on an earlier move's
/// direction while the targets stand where they stood at that move (a
/// resize reversing the one before it toward the same targets), or a bank
/// that sits unmoved and not at rest for `STEADY_ATTEMPTS` spans since its
/// slots last moved. Targets that move (memory freed or taken, or the
/// decode priced, which shrinks an unpriced bank grown to its ceiling
/// back to where its misses stop falling) forget the directions taken
/// toward the old ones. A direction is taken only from a move toward the
/// targets: a resize installs at a flight boundary several spans after it
/// begins, so one begun toward the old targets may land away from the new
/// ones, and the move back toward them is no reversal. A span moves the
/// bank where its slots or its host tier's change across it; one a running
/// resize is advanced across (`ExpertResizes::advanced_to`) is on its way,
/// neither moved nor unmoved; any other is unmoved, whatever resizes or
/// cover reads began and ended within it, so a bank that begins resizes or
/// reads that change no slots is refused.
#[derive(Default)]
pub struct Progress {
    unmoved: usize,
    /// The targets at the last span taken, which the directions were taken
    /// toward.
    targets: Option<ExpertTargets>,
    slots: Option<std::cmp::Ordering>,
    host: Option<std::cmp::Ordering>,
}

impl Progress {
    /// Take a span's `moved` and the bank's targets at its end
    /// ([`BankRest::targets`]); why the bank is making no progress, if it
    /// is not.
    pub fn note(&mut self, moved: &Moved, targets: Option<ExpertTargets>) -> Result<(), String> {
        use std::cmp::Ordering::{Equal, Greater};
        let Moved { before, after } = moved;
        if self.targets != targets {
            (self.targets, self.slots, self.host) = (targets, None, None);
        }
        let mut resized = false;
        let slots = targets.map(|targets| (targets.slots.cmp(&before.slots), targets.slots.cmp(&after.slots)));
        let host = targets.map(|targets| (targets.host.cmp(&before.host), targets.host.cmp(&after.host)));
        for (name, direction, now, toward) in [
            ("slots", &mut self.slots, after.slots.cmp(&before.slots), slots),
            ("host tier slots", &mut self.host, after.host.cmp(&before.host), host),
        ] {
            if now == Equal {
                continue;
            }
            resized = true;
            let word = |direction| if direction == Greater { "grew" } else { "shrank" };
            if let Some(earlier) = direction.filter(|earlier| *earlier != now) {
                return Err(format!(
                    "the bank's {name} {} after they {}, its targets unchanged ({targets:?}): \
                     its resizes do not converge",
                    word(now),
                    word(earlier)
                ));
            }
            // A move that stops at the target or short of it is toward it; one
            // that overshoots (a stale install landing past targets that have
            // since dropped) sets nothing.
            if toward.is_some_and(|(from_before, from_after)| from_before == now && from_after != now.reverse()) {
                *direction = Some(now);
            }
        }
        match (resized, before.resizes.advanced_to(after.resizes)) {
            (true, _) => self.unmoved = 0,
            (false, true) => {}
            (false, false) => self.unmoved += 1,
        }
        match self.unmoved >= STEADY_ATTEMPTS {
            true => {
                Err(format!("the bank sat unmoved and not at rest for {STEADY_ATTEMPTS} spans since its slots moved"))
            }
            false => Ok(()),
        }
    }
}

/// What a timed span's regime follows: the expert bank's resizes begun and
/// whether one runs, its slots, its host tier's slots, and its cover's
/// reads begun and whether they run (`magnitude_executor::ExpertCover`: a
/// tier with a record for every expert read in behind the steps). A span
/// runs steady when none of these moved across it and no resize or cover
/// read ran at either end: its misses are served by a store of the
/// capacity it will keep, and no cover's reads share the drive with its
/// own. Which experts it holds churns at that capacity, each miss landing
/// by use and displacing another once the store is full, which is the
/// steady state itself. A load without a bank is always at rest.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BankRest {
    resizes: magnitude_executor::ExpertResizes,
    slots: u64,
    host: Option<u64>,
    cover: magnitude_executor::ExpertCover,
}

impl BankRest {
    /// Its host tier's slots; none without a tier.
    pub fn host_slots(&self) -> Option<u64> {
        self.host
    }

    /// The bank as `engine` holds it now; none without a bank.
    pub fn of(engine: &Engine) -> Result<Option<Self>, String> {
        let residency = engine.domain.expert_residency().map_err(|error| format!("the expert residency: {error}"))?;
        Ok(residency.map(|residency| Self {
            resizes: engine.domain.expert_resizes(),
            slots: residency.slots,
            host: residency.host.map(|host| host.slots),
            cover: residency.cover,
        }))
    }

    /// The targets `engine`'s bank follows now ([`Progress`]); none without
    /// a bank. Read apart from the bank, only where a span did not run
    /// steady: they are derived afresh from the decode's price and the
    /// ledgers, which nothing between timed steps reads.
    pub fn targets(engine: &Engine) -> Result<Option<ExpertTargets>, String> {
        engine.domain.expert_targets().map_err(|error| format!("the expert bank's targets: {error}"))
    }

    /// The bank as `engine` holds it once its cover reads nothing, waited
    /// for: a span begins only there. The cover reads on the bank's own
    /// threads, for minutes where it reads much, so no span need run to
    /// drive it; the wait is woken by its publication
    /// (`ExecutorDomain::wait_expert_cover`) and bounded only by the run's
    /// own.
    pub fn covered(engine: &Engine) -> Result<Option<Self>, String> {
        engine.domain.wait_expert_cover().map_err(|error| format!("the expert store's cover: {error}"))?;
        Self::of(engine)
    }

    /// Whether a span may begin or end at this bank: no resize runs and no
    /// cover read.
    fn settled(self) -> bool {
        !self.resizes.running && !self.cover.reading
    }

    /// What moved between `before`, read before a span, and `after`; none
    /// where the span ran steady.
    pub fn moved(before: Option<Self>, after: Option<Self>) -> Option<Moved> {
        let (before, after) = before.zip(after)?;
        (before != after || !before.settled()).then_some(Moved { before, after })
    }
}

/// The rows past its history a timed decode of `warm` warm steps, windows
/// of `steps` and `attribution` attributed steps after its window advances
/// at most, its discarded attempts too ([`steady_window`]): what its load's
/// context holds beyond the history.
pub fn decode_context_rows(warm: usize, steps: usize, attribution: usize) -> usize {
    warm + steps + attribution + 1
}

/// A decode's one window the bank rests across, the decode never deeper
/// than its histories by more than `warm` steps and its window: each
/// attempt warms the decode to rest, then runs its window (`window`: what
/// it gives, or what the bank moved across it, discarding it), and a
/// discarded attempt's decode begins again from its histories (`rewind`),
/// so the next runs at the depth the first did, as the bank stands.
///
/// The warm-up is `warm` steps (`step` one step, saying what moved across
/// it: [`BankRest::moved`], and `targets` the bank's targets after a step
/// it moved across: [`BankRest::targets`]), then more until one runs at
/// rest, so no window begins inside a resize; each `warm` steps it begins
/// again from the histories. A warm-up past `warm` steps is run again from
/// them, `warm` steps whose last ran at rest, so the window follows a
/// warm-up as the first ran. A warm-up whose bank stops making progress
/// ([`Progress`]), or `STEADY_ATTEMPTS` discarded windows, take the bank
/// never to rest.
/// Gives the window and the discarded attempts' records, in order.
pub fn steady_window<S, W>(
    decode: &mut S,
    warm: usize,
    mut step: impl FnMut(&mut S) -> Result<Option<Moved>, String>,
    mut targets: impl FnMut(&mut S) -> Result<Option<ExpertTargets>, String>,
    mut window: impl FnMut(&mut S) -> Result<Result<W, Value>, String>,
    mut rewind: impl FnMut(&mut S) -> Result<(), String>,
) -> Result<(W, Vec<Value>), String> {
    let mut discarded = Vec::new();
    loop {
        let mut moved = None;
        let mut warmed = 0;
        let mut progress = Progress::default();
        while warmed < warm || moved.is_some() {
            if let Some(moved) = moved.as_ref() {
                if let Err(why) = progress.note(moved, targets(decode)?) {
                    let record = moved.record();
                    let steps = format!("{warmed} warm decode steps");
                    return Err(format!("the expert bank was not at rest after {steps}: {why}: {record}"));
                }
            }
            if warmed > 0 && warmed % warm == 0 {
                rewind(decode)?;
            }
            moved = step(decode)?;
            warmed += 1;
        }
        if warmed > warm {
            rewind(decode)?;
            for _ in 0..warm {
                moved = step(decode)?;
            }
        }
        let outcome = match moved {
            Some(moved) => Err(json!({ "warm": moved.record() })),
            None => window(decode)?,
        };
        match outcome {
            Ok(window) => return Ok((window, discarded)),
            Err(moved) => discarded.push(moved),
        }
        if discarded.len() == STEADY_ATTEMPTS {
            return Err(format!(
                "no decode window ran with the expert bank at rest in {STEADY_ATTEMPTS} attempts: {}",
                Value::from(discarded)
            ));
        }
        rewind(decode)?;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Targets a bank follows unchanged across every span of a test.
    const AIMED: Option<ExpertTargets> = Some(ExpertTargets { slots: 48, host: Some(256) });

    /// A bank on its way to rest is waited for however many spans that
    /// takes; one whose moves turn back on an earlier one's direction
    /// toward unchanged targets, or that sits unmoved without resting, is
    /// refused.
    #[test]
    fn a_bank_is_waited_for_while_its_moves_make_progress() {
        let span = |before, after| {
            BankRest::moved(rest(1, true, 48, Some(before)), rest(1, true, 48, Some(after))).unwrap()
        };
        let mut progress = Progress::default();
        let growing = (0..100).all(|slots| progress.note(&span(slots, slots + 1), AIMED).is_ok());
        assert!(growing, "a growing tier makes progress");
        let reversed = progress.note(&span(100, 99), AIMED).unwrap_err();
        assert!(reversed.contains("host tier slots shrank after they grew"), "{reversed}");
        let mut progress = Progress::default();
        let spans = (0..).take_while(|_| progress.note(&span(9, 9), AIMED).is_ok()).count();
        assert_eq!(spans, STEADY_ATTEMPTS - 1, "a resize no boundary advanced moves nothing");
    }

    /// A bank grows toward its ceiling until its decode is priced, then
    /// shrinks to where the decode's misses stop falling, as the served
    /// bank does: its targets moved, so the shrink is progress, not a
    /// reversal. Turning back again toward the targets it shrank to is
    /// refused.
    #[test]
    fn a_bank_grown_unpriced_shrinks_once_priced() {
        let unpriced = Some(ExpertTargets { slots: 4, host: None });
        let priced = Some(ExpertTargets { slots: 3, host: None });
        let span = |from, to| BankRest::moved(resizing(1, false, 0, from), resizing(2, false, 0, to)).unwrap();
        let mut progress = Progress::default();
        assert_eq!(progress.note(&span(2, 3), unpriced), Ok(()), "a growth toward the ceiling");
        assert_eq!(progress.note(&span(3, 4), unpriced), Ok(()), "a growth to the ceiling");
        assert_eq!(progress.note(&span(4, 3), priced), Ok(()), "a shrink toward the priced target");
        let reversed = progress.note(&span(3, 4), priced).unwrap_err();
        assert!(reversed.contains("the bank's slots grew after they shrank"), "{reversed}");
    }

    /// A growth begun toward the unpriced ceiling installs at a flight
    /// boundary spans later, after the decode is priced and the targets
    /// drop: it lands away from them and sets no direction, so the shrink
    /// back toward them is progress. Growing again, the targets unchanged,
    /// is refused.
    #[test]
    fn a_growth_installed_after_the_targets_drop_is_followed_by_a_shrink() {
        let unpriced = Some(ExpertTargets { slots: 4, host: None });
        let priced = Some(ExpertTargets { slots: 3, host: None });
        let readings = [
            (resizing(1, false, 0, 2), None),
            (resizing(2, true, 0, 2), unpriced),
            (resizing(2, true, 1, 2), unpriced),
            (resizing(2, true, 2, 2), priced),
            (resizing(2, false, 3, 4), priced),
            (resizing(3, true, 0, 4), priced),
            (resizing(3, false, 1, 3), priced),
        ];
        let mut progress = Progress::default();
        for (span, pair) in readings.windows(2).enumerate() {
            let ((before, _), (after, targets)) = (pair[0], pair[1]);
            let verdict = progress.note(&BankRest::moved(before, after).unwrap(), targets);
            assert_eq!(verdict, Ok(()), "span {span}");
        }
        let regrown = BankRest::moved(resizing(3, false, 1, 3), resizing(4, false, 1, 4)).unwrap();
        let reversed = progress.note(&regrown, priced).unwrap_err();
        assert!(reversed.contains("the bank's slots grew after they shrank"), "{reversed}");
    }

    /// One resize running across many more spans than `STEADY_ATTEMPTS`
    /// (a shrink whose slots change only as it installs) is waited for
    /// while the flight boundaries advance it, and its install moves the
    /// slots.
    #[test]
    fn a_long_resize_advanced_at_each_span_is_waited_for() {
        let mut progress = Progress::default();
        let running = |advances| resizing(3, true, advances, 48);
        let spans = 40 * STEADY_ATTEMPTS as u64;
        let span = |at| BankRest::moved(running(at), running(at + 1)).unwrap();
        let waited = (0..spans).all(|at| progress.note(&span(at), AIMED).is_ok());
        assert!(waited, "a resize advanced at every span is on its way");
        let installed = BankRest::moved(running(spans), resizing(3, false, spans + 1, 40)).unwrap();
        assert_eq!(progress.note(&installed, AIMED), Ok(()), "its install moves the slots");
    }

    /// A bank that begins and ends resizes or cover reads with no slot
    /// moved sits unmoved, however many it begins: refused at
    /// `STEADY_ATTEMPTS` such spans since its slots last moved, a span a
    /// running resize is advanced across between them not resetting it.
    #[test]
    fn a_bank_that_begins_moves_changing_no_slots_is_refused() {
        let at = |begun, running, advances, covers| {
            let resizes = magnitude_executor::ExpertResizes { begun, running, advances };
            let cover = magnitude_executor::ExpertCover { begun: covers, reading: false };
            Some(BankRest { resizes, slots: 48, host: None, cover })
        };
        // A resize begun (unmoved), advanced across a span, ended at the
        // same slots (unmoved); a cover read begun and ended (unmoved);
        // another resize begun (unmoved).
        assert_eq!(STEADY_ATTEMPTS, 4, "the spans below are four unmoved");
        let readings = [
            at(1, false, 0, 1),
            at(2, true, 1, 1),
            at(2, true, 2, 1),
            at(2, false, 3, 1),
            at(2, false, 3, 2),
            at(3, true, 4, 2),
        ];
        let mut progress = Progress::default();
        for (span, pair) in readings.windows(2).enumerate() {
            let verdict = progress.note(&BankRest::moved(pair[0], pair[1]).unwrap(), AIMED);
            assert_eq!(verdict.is_err(), span == readings.len() - 2, "span {span}: {verdict:?}");
        }
    }

    const COVERED: magnitude_executor::ExpertCover = magnitude_executor::ExpertCover { begun: 1, reading: false };

    fn rest(begun: u64, running: bool, slots: u64, host: Option<u64>) -> Option<BankRest> {
        covered(begun, running, slots, host, COVERED)
    }

    fn covered(
        begun: u64,
        running: bool,
        slots: u64,
        host: Option<u64>,
        cover: magnitude_executor::ExpertCover,
    ) -> Option<BankRest> {
        let resizes = magnitude_executor::ExpertResizes { begun, running, advances: 0 };
        Some(BankRest { resizes, slots, host, cover })
    }

    /// A bank without a tier, its cover read, at `slots`, its resizes
    /// `begun` and whether one runs, advanced `advances` times.
    fn resizing(begun: u64, running: bool, advances: u64, slots: u64) -> Option<BankRest> {
        let resizes = magnitude_executor::ExpertResizes { begun, running, advances };
        Some(BankRest { resizes, slots, host: None, cover: COVERED })
    }

    /// The cover reading at either end, or begun again across a span, makes
    /// it unsteady: its reads share the drive with the span's, and the
    /// span's misses read what it has not yet landed.
    #[test]
    fn a_cover_read_is_unsteady() {
        let cover = |begun, reading| magnitude_executor::ExpertCover { begun, reading };
        let reading = |begun| covered(3, false, 48, Some(256), cover(begun, true));
        let read = |begun| covered(3, false, 48, Some(256), cover(begun, false));
        assert!(BankRest::moved(reading(1), reading(1)).is_some());
        assert!(BankRest::moved(reading(1), read(1)).is_some());
        assert!(BankRest::moved(read(1), reading(2)).is_some());
        assert!(BankRest::moved(read(1), read(2)).is_some(), "a cover begun and ended across the span");
        assert_eq!(BankRest::moved(read(1), read(1)), None);
    }

    /// Experts churning through a bank and tier at fixed slots (the bytes
    /// only the file holds moving with each stored size swapped) leave
    /// nothing of the rest to move: steady, beside a tier or none.
    #[test]
    fn churn_at_fixed_slots_is_steady() {
        for at_rest in [rest(3, false, 49, None), rest(3, false, 48, Some(256))] {
            assert_eq!(BankRest::moved(at_rest, at_rest), None, "{at_rest:?}");
        }
    }

    /// A resize begun across a span or running at either end, or slots
    /// moved, make it unsteady.
    #[test]
    fn a_resize_is_unsteady() {
        let at_rest = rest(3, false, 48, Some(256));
        for after in [
            rest(4, false, 48, Some(256)),
            rest(3, true, 48, Some(256)),
            rest(3, false, 49, Some(256)),
            rest(3, false, 48, Some(240)),
        ] {
            assert!(BankRest::moved(at_rest, after).is_some(), "{after:?}");
        }
        let running = rest(3, true, 48, Some(225));
        assert!(BankRest::moved(running, running).is_some(), "a resize running at both ends");
    }

    /// A load without a bank is always steady.
    #[test]
    fn no_bank_is_steady() {
        assert_eq!(BankRest::moved(None, None), None);
    }

    /// How a fake's bank moves across a step it does not rest across.
    #[derive(Clone, Copy, Debug)]
    enum Motion {
        /// A tier growing a slot each step.
        Grows,
        /// One resize running, advanced at each step, its slots unchanged.
        Resizes,
        /// A resize running that nothing advances.
        Stands,
    }

    /// A decode of a history of `history` tokens, `warm` warm steps and
    /// windows of `steps`, whose load holds `context` rows; its bank moves
    /// across its first `unrested` steps and every window (`never`), or
    /// none after them, as `motion` says.
    struct Fake {
        history: usize,
        steps: usize,
        context: usize,
        position: usize,
        deepest: usize,
        ran: usize,
        unrested: usize,
        never: bool,
        motion: Motion,
    }

    impl Fake {
        fn advance(&mut self, rows: usize) -> Result<(), String> {
            self.position += rows;
            self.deepest = self.deepest.max(self.position);
            match self.position <= self.context {
                true => Ok(()),
                false => Err(format!("advance exceeds context capacity: {} of {}", self.position, self.context)),
            }
        }
    }

    fn steady(fake: &mut Fake, warm: usize) -> Result<(usize, Vec<Value>), String> {
        let moved = json!({ "step": 0 });
        steady_window(
            fake,
            warm,
            |fake| {
                fake.advance(1)?;
                fake.ran += 1;
                let ran = fake.ran as u64;
                let (before, after) = match fake.motion {
                    Motion::Grows => (rest(1, true, 180 + ran, None), rest(1, true, 181 + ran, None)),
                    Motion::Resizes => (resizing(1, true, ran, 180), resizing(1, true, ran + 1, 180)),
                    Motion::Stands => (rest(1, true, 180, None), rest(1, true, 180, None)),
                };
                let moved = BankRest::moved(before, after).unwrap();
                Ok((fake.ran <= fake.unrested).then_some(moved))
            },
            |_| Ok(AIMED),
            |fake| {
                fake.advance(fake.steps)?;
                fake.ran += fake.steps;
                Ok(match fake.never {
                    true => Err(json!({ "step": 0, "moved": moved.clone() })),
                    false => Ok(fake.position - fake.history),
                })
            },
            |fake| {
                fake.position = fake.history;
                Ok(())
            },
        )
    }

    /// A decode whose bank moves across every window, or rests only after
    /// a long warm-up, runs no deeper than its histories, its warm steps
    /// and its window, the context `time` sizes its load at: every
    /// discarded attempt, and every warm-up past its warm steps, begins
    /// again from the histories.
    #[test]
    fn a_decode_whose_bank_does_not_rest_stays_within_its_context() {
        for (warm, steps) in [(8, 48), (0, 2), (120, 200)] {
            let context = 4096 + decode_context_rows(warm, steps, 0);
            let fake = |unrested, never, motion| Fake {
                history: 4096,
                steps,
                context,
                position: 4096,
                deepest: 4096,
                ran: 0,
                unrested,
                never,
                motion,
            };
            // Every window moved across: refused once its attempts are spent.
            let mut never = fake(0, true, Motion::Grows);
            let refused = steady(&mut never, warm);
            assert!(
                refused.as_ref().is_err_and(|error| error.starts_with("no decode window ran")),
                "{warm} warm + {steps} steps: {refused:?}"
            );
            assert!(never.deepest <= context, "{warm} warm + {steps} steps reached {} of {context}", never.deepest);
            // A bank still moving is waited for however long it takes, its
            // tier growing or one resize advanced at every step: a warm-up
            // three times the cell's, or forty, still gives its window
            // `warm` steps from the histories, at rest.
            for unrested in [3 * warm, 40 * warm] {
                for motion in [Motion::Grows, Motion::Resizes] {
                    let mut slow = fake(unrested, false, motion);
                    let (depth, discarded) = steady(&mut slow, warm).unwrap();
                    let cell = format!("{warm} warm + {steps} steps, {unrested} unrested, {motion:?}");
                    assert_eq!((depth, discarded.len()), (warm + steps, 0), "{cell}");
                    assert!(slow.deepest <= context);
                }
            }
        }
        // A bank that sits unmoved without resting in a warm-up is refused
        // within it.
        let mut stuck = Fake {
            history: 4096,
            steps: 48,
            context: 4096 + decode_context_rows(8, 48, 0),
            position: 4096,
            deepest: 4096,
            ran: 0,
            unrested: usize::MAX,
            never: false,
            motion: Motion::Stands,
        };
        let refused = steady(&mut stuck, 8);
        let after = format!("the expert bank was not at rest after {STEADY_ATTEMPTS} ");
        assert!(refused.is_err_and(|error| error.starts_with(&after)));
        assert!(stuck.deepest <= 4096 + 8);
    }
}
