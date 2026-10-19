//! When a timed comparison of two configurations is resolved.
//!
//! # What is compared
//!
//! A challenger is compared with the choice its unit serves (the
//! incumbent) in rounds. A round samples both, once each at every point
//! where they run different work, in one session and in random order
//! ([`seismic_measure::Shuffle`]). From a round `r` comes one number,
//!
//! ```text
//! d_r = Σ_p w_p · I_p · (c_{p,r} / i_{p,r} − 1)
//! ```
//!
//! over those points `p`: `w_p` the point's demand, `I_p` the incumbent's
//! time there in the unit's canonical units, and `c` and `i` the two
//! samples of the round. `d_r < 0` says that in this round the challenger's
//! service cost was below the incumbent's: the round is *ahead*.
//! `d_r > 0`: it is *behind*.
//!
//! The ratio is taken inside the round because the samples recorded from
//! real loads say that is where the measurement is good (Qwen3.5-4B on
//! Metal, 2026-10-10, 215 confirmed point measurements and 250 first
//! measurements re-measured later): within one session a point's samples
//! spread by 0.26% of their median at the median point and 2% at the 90th
//! percentile, while the same configuration measured in two sessions
//! differed by 9% at the median and by 79% at the 90th percentile, in
//! either direction (48% of first measurements were the slower of the
//! two), in steps of about 1.6× and 2.2×: the device's clock state, which
//! follows how busy the session keeps it.
//!
//! The ratio does not make a round indifferent to that state. Two kernels
//! do not slow by the same factor at a lower state, and a session's
//! samples do not move the device out of the state it began in, so rounds
//! taken there agree with one another and with no other session
//! ([`seismic_measure::clock`], "The state of the device"). A session's
//! rounds are therefore taken with the device raised to its sustained
//! clock, each sample behind a lead of its own work, and both sides of a
//! pair with the same repetitions, one after the other. What the ratio
//! then cancels is what is left: a disturbance that outlasts a pair.
//!
//! # What is decided, and by which statistic
//!
//! The median of `d` over rounds, by the paired sign test of
//! [`seismic_measure::statistics`]: a lead (the median is below zero,
//! faster than the served choice) or a trail (above it, slower) is resolved
//! when the mixture of likelihood ratios of the rounds ahead, or behind,
//! reaches `1 / α` ([`Tally::verdict`]). No margin is asked of a lead
//! beyond that, and nothing is assumed of the distribution of `d`.
//!
//! # The error rate
//!
//! `α` is [`super::search::SearchSettings::error_rate`], the one place it
//! is expressed. Because the bound is on *ever* reaching `1 / α`, it holds
//! over every look at a comparison together, however many rounds it takes,
//! whoever decides after each round whether it gets another, and across
//! the sessions it is continued in. So, of the comparisons of a
//! configuration that is not faster than the served choice, at most `α`
//! ever resolve as a lead (and it is then promoted, once covered); and of
//! the comparisons of one that is not slower, at most `α` ever resolve as
//! a trail (and it is dropped for good). A
//! comparison that has resolved neither way decides nothing: its
//! challenger stays measured, and whether it is given more rounds is a
//! question of the budget, answered like every other ([`Rates`]).
//!
//! # How many rounds
//!
//! The fewest rounds in which a comparison can resolve as a lead follow
//! from `α` and what it has seen ([`Tally::fewest`]): every one of them
//! ahead. A session admits a comparison exactly those, and its part ends
//! with its first round that is not ahead: within what was admitted, no
//! further round can then change whether it leads. With `α = 0.05` a new
//! comparison leads after seven rounds all ahead
//! (`M_7(7) = 255 / 8 ≥ 20 > 127 / 7 = M_6(6)`), and most challengers, which
//! are slower, take one.

use super::budget::Observed;
use seismic_measure::Tally;

/// The rounds this unit's sessions have taken of a comparison, for pricing
/// a session before it runs.
///
/// A session admits a comparison the fewest rounds in which it can lead and
/// its part ends with its first round that is not ahead, so the rounds it
/// takes follow from how the unit's rounds spread about the served choice's
/// cost: one for a challenger that is slower, all of them for one that is
/// clearly faster, and something between for one close to it or under an
/// outlier. That
/// number is a class of operation like any other of the load
/// ([`Observed`]): it is priced by the mean of its own observations, and
/// has no price before its first. The sessions that open a comparison and
/// those that continue one are two classes: most comparisons opened are of
/// slower challengers, and every one continued had a measured lead.
///
/// The mean is the price, not a model of it: a unit's challengers differ in
/// how often their rounds are ahead, and a single chance per round, fitted
/// to all of them, priced the sessions of a synthetic unit 7% high.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Rates {
    opening: Observed,
    continuing: Observed,
}

impl Rates {
    /// A session took `rounds` rounds of a comparison that stood at
    /// `before` when it began.
    pub fn observe(&mut self, before: &Tally, rounds: u32) {
        if before.rounds() == 0 {
            &mut self.opening
        } else {
            &mut self.continuing
        }
        .observe(f64::from(rounds));
    }

    /// The rounds a session is expected to take of a comparison that
    /// stands at `evidence`; none before a session of its class ran.
    pub fn expected(&self, evidence: &Tally) -> Option<f64> {
        if evidence.rounds() == 0 {
            self.opening
        } else {
            self.continuing
        }
        .price()
    }
}

/// Which of a challenger's two tests a unit takes first.
///
/// A challenger is served only when it passes both: its rounds lead the
/// served choice, and its numerical witnesses meet the unit's requirement.
/// Either failing ends it, so the test taken first is the one that ends
/// more challengers for what it costs: by the sequencing rule for tests in
/// series, the one with the smaller cost divided by its probability of
/// failing. Both costs and both frequencies are this unit's own
/// observations in this load: the host seconds one challenger's witnesses
/// take and those its part of a session's rounds takes, and how often each
/// test was failed.
///
/// A frequency observed `k` times in `n` is taken as `(k + 1) / (n + 2)`
/// (the rule of succession): a test never yet failed is not taken to never
/// fail, and the first observations do not decide the order for good. A
/// unit whose arithmetic variants are refused admission is thereby
/// witnessed first again as its refusals accumulate.
///
/// Both frequencies are kept for the unit as a whole, not for each
/// arithmetic assignment of its challengers: a refusal counts against
/// every challenger of the unit alike, whichever arithmetic the refused one
/// ran, so a unit with one arithmetic that is always refused and another
/// that never is takes all its challengers in the one order their mixture
/// gives.
///
/// Before a cost of each test has been observed there is nothing to order
/// them by, and the witness is taken first: a session in that order
/// observes both costs, so the unit's first session of challengers settles
/// the order of the rest.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Sequence {
    witness: Observed,
    rounds: Observed,
    witnessed: u32,
    refused: u32,
    compared: u32,
    behind: u32,
}

impl Sequence {
    /// One challenger's witnesses took `seconds`, and the requirement
    /// `admitted` it or refused it.
    pub fn witnessed(&mut self, seconds: f64, admitted: bool) {
        self.witness.observe(seconds);
        self.witnessed += 1;
        self.refused += u32::from(!admitted);
    }

    /// One challenger's part of its first session's rounds took `seconds`
    /// and left it `ahead` of the served choice or not.
    pub fn compared(&mut self, seconds: f64, ahead: bool) {
        self.rounds.observe(seconds);
        self.compared += 1;
        self.behind += u32::from(!ahead);
    }

    /// Whether a challenger is timed before it is witnessed.
    pub fn timed_first(&self) -> bool {
        let (Some(witness), Some(rounds)) = (self.witness.price(), self.rounds.price()) else {
            return false;
        };
        rounds / succession(self.behind, self.compared)
            < witness / succession(self.refused, self.witnessed)
    }

    /// The chance a challenger's rounds leave it ahead of the served
    /// choice, as the unit's challengers so far were left: what a
    /// challenger timed first is witnessed with.
    pub fn ahead(&self) -> f64 {
        succession(self.compared - self.behind, self.compared)
    }
}

/// A frequency observed `times` in `taken`, by the rule of succession.
fn succession(times: u32, taken: u32) -> f64 {
    f64::from(times + 1) / f64::from(taken + 2)
}

#[cfg(test)]
mod tests {
    use super::*;
    use seismic_measure::{Shuffle, Verdict};

    const RATE: f64 = 0.05;

    /// A synthetic device clock like the recorded one: a sample is its
    /// kernel's time, lengthened by a right-skewed jitter (exponential,
    /// 1% of the time at its mean) and, one time in twenty, by an outlier of
    /// 30% to 80%; and the whole round runs at one of three clock states
    /// (1×, 1.6×, 2.2×) that changes from round to round.
    struct Clock(Shuffle);

    impl Clock {
        fn uniform(&mut self) -> f64 {
            (self.0.below(1 << 30) as f64 + 0.5) / f64::from(1u32 << 30)
        }

        fn sample(&mut self, seconds: f64) -> f64 {
            let jitter = -0.01 * self.uniform().ln();
            let outlier = if self.uniform() < 0.05 {
                0.3 + 0.5 * self.uniform()
            } else {
                0.
            };
            seconds * (1. + jitter + outlier)
        }

        /// One round's `d` of a challenger taking `challenger` of the
        /// incumbent's time.
        fn round(&mut self, challenger: f64) -> f64 {
            let state = [1., 1.6, 2.2][self.0.below(3)];
            let incumbent = self.sample(100e-6 * state);
            let sampled = self.sample(100e-6 * challenger * state);
            sampled / incumbent - 1.
        }
    }

    /// A comparison run as a unit runs it: sessions, each admitting the
    /// fewest rounds in which it can lead and ending with its first round
    /// that is not ahead, continued until it resolves or `sessions` ran.
    /// The verdict, and the rounds each session took.
    fn compare(
        clock: &mut Clock,
        challenger: f64,
        sessions: usize,
        rates: &mut Rates,
    ) -> (Option<Verdict>, Vec<(Option<f64>, u32)>) {
        let mut evidence = Tally::default();
        let mut taken = Vec::new();
        for _ in 0..sessions {
            if evidence.verdict(RATE).is_some() {
                break;
            }
            let expected = rates.expected(&evidence);
            let before = evidence;
            let mut rounds = 0;
            for _ in 0..evidence.fewest(RATE) {
                let difference = clock.round(challenger);
                evidence.observe(difference);
                rounds += 1;
                if difference >= 0. {
                    break;
                }
            }
            rates.observe(&before, rounds);
            taken.push((expected, rounds));
        }
        (evidence.verdict(RATE), taken)
    }

    /// A clear difference is resolved in the fewest rounds there are, and a
    /// small one takes more.
    #[test]
    fn a_clear_difference_stops_early_and_a_small_one_continues() {
        let mut clock = Clock(Shuffle::default());
        let mut rates = Rates::default();
        // 20% faster: far past the jitter.
        let rounds =
            |taken: &[(Option<f64>, u32)]| taken.iter().map(|(_, rounds)| rounds).sum::<u32>();
        let mut clear = Vec::new();
        for _ in 0..200 {
            let (verdict, taken) = compare(&mut clock, 0.8, usize::MAX, &mut rates);
            assert_eq!(verdict, Some(Verdict::Leads));
            clear.push(rounds(&taken));
        }
        clear.sort_unstable();
        assert_eq!(clear[clear.len() / 2], 7, "{clear:?}");
        // Twice as slow, which no outlier of the incumbent's hides: every
        // session ends with its first round, and the trail is resolved in
        // seven.
        let (verdict, taken) = compare(&mut clock, 2., usize::MAX, &mut rates);
        assert_eq!(verdict, Some(Verdict::Trails));
        assert!(taken.iter().all(|(_, rounds)| *rounds == 1));
        assert_eq!(rounds(&taken), 7);
        // Faster by the jitter's mean: its rounds are not all ahead, so it
        // takes more of them than a clear one, and it leads.
        let mut small = Vec::new();
        let mut led = 0;
        for _ in 0..200 {
            let (verdict, taken) = compare(&mut clock, 0.99, usize::MAX, &mut rates);
            led += usize::from(verdict == Some(Verdict::Leads));
            small.push(rounds(&taken));
        }
        small.sort_unstable();
        assert!(small[small.len() / 2] > 7, "{small:?}");
        assert!(led >= 190, "{led} of 200 led");
    }

    /// The realised error rates on the skewed clock, over every look at
    /// each comparison: a challenger exactly as fast as the served choice
    /// (not faster) resolves as a lead, or as a trail, in no more than the
    /// named rate of its comparisons, however long each is continued.
    #[test]
    fn the_realised_error_rate_stays_within_the_named_rate() {
        let mut clock = Clock(Shuffle::default());
        let mut rates = Rates::default();
        let trials = 4000;
        let (mut leads, mut trails) = (0, 0);
        for _ in 0..trials {
            // As fast as the served choice: a round is ahead as often as
            // behind.
            match compare(&mut clock, 1., 60, &mut rates).0 {
                Some(Verdict::Leads) => leads += 1,
                Some(Verdict::Trails) => trails += 1,
                None => {}
            }
        }
        let realised = |errors: i32| f64::from(errors) / f64::from(trials);
        println!(
            "realised error rates at equal speed over {trials} comparisons: \
             lead {:.4}, trail {:.4} (named {RATE})",
            realised(leads),
            realised(trails)
        );
        assert!(realised(leads) <= RATE, "{leads} of {trials} led");
        assert!(realised(trails) <= RATE, "{trails} of {trials} trailed");
        // The bound is not vacuous: the test does resolve at equal speed.
        assert!(leads + trails > 0);
        // A slower challenger is resolved as a lead no more often.
        let slower = (0..trials)
            .filter(|_| compare(&mut clock, 1.02, 60, &mut rates).0 == Some(Verdict::Leads))
            .count();
        assert!(slower as f64 <= RATE * f64::from(trials), "{slower}");
    }

    /// The rounds sessions were priced before they ran match the rounds
    /// they took, in aggregate, once the unit has seen its rounds; before
    /// it has, they have no price.
    #[test]
    fn expected_rounds_price_the_rounds_sessions_take() {
        let mut clock = Clock(Shuffle::default());
        let mut rates = Rates::default();
        assert_eq!(rates.expected(&Tally::default()), None);
        // A unit's mix: most challengers slower, some close to the served
        // choice, a few clearly faster.
        let challengers = [1.32, 1.12, 1.07, 1.04, 1.01, 0.995, 0.99, 0.92, 1.22, 1.42];
        let mut run = |rates: &mut Rates| {
            let (mut expected, mut realised) = (0., 0.);
            for trial in 0..2000 {
                let (_, taken) =
                    compare(&mut clock, challengers[trial % challengers.len()], 3, rates);
                for (priced, rounds) in taken {
                    expected += priced.unwrap_or(0.);
                    realised += f64::from(rounds);
                }
            }
            (expected, realised)
        };
        // The first pass is what the rates are learned from.
        run(&mut rates);
        let (expected, realised) = run(&mut rates);
        println!("rounds priced {expected:.0}, taken {realised:.0}");
        assert!(
            (expected / realised - 1.).abs() < 0.02,
            "priced {expected} rounds, took {realised}"
        );
    }

    /// The order of a challenger's tests follows what the unit observed:
    /// the witness first until both costs are known, then the test that
    /// ends more challengers for its cost, and the witness first again in
    /// a unit whose challengers are refused.
    #[test]
    fn the_test_that_ends_more_challengers_for_its_cost_is_taken_first() {
        let mut sequence = Sequence::default();
        assert!(!sequence.timed_first());
        // Witnesses and rounds that cost alike (2.6 ms and 2.7 ms a
        // challenger, as recorded on Metal): nine challengers in ten trail,
        // none is refused.
        for challenger in 0..10 {
            sequence.witnessed(2.6e-3, true);
            assert!(!sequence.timed_first() || challenger > 0);
            sequence.compared(2.7e-3, challenger == 0);
        }
        assert!(sequence.timed_first());
        // Witnesses seven times the rounds (CUDA prefill): the more so.
        let mut costly = Sequence::default();
        costly.witnessed(0.09, true);
        costly.compared(0.012, false);
        assert!(costly.timed_first());
        // A unit whose challengers are refused as often as they trail, at
        // the same costs: the cheaper test first, here the witness.
        let mut refused = Sequence::default();
        for _ in 0..10 {
            refused.witnessed(2.6e-3, false);
            refused.compared(2.7e-3, false);
        }
        assert!(!refused.timed_first());
        // Refused always, trailing one time in two: the witness first even
        // at twice the rounds' cost.
        let mut arithmetic = Sequence::default();
        for challenger in 0..20 {
            arithmetic.witnessed(5e-3, false);
            arithmetic.compared(2.7e-3, challenger % 2 == 0);
        }
        assert!(!arithmetic.timed_first());
    }
}
