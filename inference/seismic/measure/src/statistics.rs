//! The sampling statistics: the median and its anytime-valid interval, the
//! paired sign test a timed comparison is resolved by, the one stopping rule
//! of a measured time, and the order of a round's samples.
//!
//! # What a comparison decides, and by which statistic
//!
//! A challenger is compared with the choice it would replace in rounds. A
//! round samples both and yields one number `d_r`, below zero when in that
//! round the challenger's cost was below the other's (the round is *ahead*)
//! and above zero when it was *behind*. The quantity estimated is the median
//! of `d` over rounds. The two claims a comparison can resolve are
//!
//! - *lead*: the median is below zero (faster);
//! - *trail*: the median is above zero (slower).
//!
//! No margin is asked of a lead beyond that: how sure a lead must be is the
//! error rate's question alone.
//!
//! Nothing is assumed of the distribution of `d`. Recorded samples are not
//! normal: the slowest sample of a point lay 31% above its median at the
//! 90th percentile and 63% at the 99th, the fastest 10% and 21% below, so a
//! mean, a standard deviation and any interval built from them are governed
//! by the outliers. The sign of `d_r` is not: whatever the distribution, a
//! round is ahead with probability at most one half when the median is not
//! below zero, and that is all the test uses.
//!
//! Let `s` of `n` rounds be ahead. For any `p > ½` the likelihood ratio of
//! "a round is ahead with probability `p`" against "with probability ½" is
//! `(2p)^s (2(1 − p))^(n − s)`. When rounds are ahead with a probability
//! `q ≤ ½`, whatever happened before, one more round multiplies it by `2p`
//! with probability `q` and by `2(1 − p)` otherwise, in expectation
//! `2(1 − p) + 2q(2p − 1) ≤ 1`: it is a nonnegative supermartingale that
//! starts at one. So is its average over `p`, and no `p` is known in
//! advance, so the average is taken uniformly over `(½, 1)`:
//!
//! ```text
//! M_n(s) = ∫_{½}^{1} 2 · (2p)^s (2(1 − p))^(n − s) dp
//!        = Σ_{j ≤ s} C(n + 1, j) / ((n + 1) · C(n, s))
//! ```
//!
//! (the beta integral, and the identity between a beta and a binomial
//! tail). By Ville's inequality a nonnegative supermartingale that starts
//! at one ever reaches `1 / α` with probability at most `α`. A lead is
//! resolved when `M_n(ahead) ≥ 1 / α`, a trail when `M_n(behind) ≥ 1 / α`
//! ([`Tally::verdict`]). Because the bound is on *ever* reaching `1 / α`,
//! it holds over every look at a comparison together, however many rounds
//! it takes and whoever decides after each round whether it gets another.
//!
//! The fewest rounds in which a comparison can resolve as a lead follow
//! from `α` and what it has seen ([`Tally::fewest`]): every one of them
//! ahead. With `α = 0.05` a new comparison leads after seven rounds all
//! ahead (`M_7(7) = 255 / 8 ≥ 20 > 127 / 7 = M_6(6)`).

use serde::{Deserialize, Serialize};

/// The median of `samples`; the mean of the middle two for an even count.
pub fn median(samples: &[f64]) -> f64 {
    median_of(samples.to_vec())
}

/// The median of `values`; the mean of the middle two for an even count.
pub fn median_of(mut values: Vec<f64>) -> f64 {
    values.sort_by(f64::total_cmp);
    let middle = values.len() / 2;
    if values.len() % 2 == 0 {
        (values[middle - 1] + values[middle]) / 2.0
    } else {
        values[middle]
    }
}

/// What the rounds of one comparison have shown: how many there were, and
/// how many were ahead and behind (a round in which the two cost exactly
/// the same is neither).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Tally {
    rounds: u32,
    ahead: u32,
    behind: u32,
}

/// What a comparison has resolved.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Verdict {
    /// The challenger is faster than the choice it is compared with.
    Leads,
    /// It is slower.
    Trails,
}

/// `M_n(s)`: what the uniform mixture of likelihood ratios stands at after
/// `rounds` rounds of which `favourable` favoured its claim (module
/// documentation). Each term `C(n + 1, j) / C(n, s)` follows from the one
/// above it, so nothing overflows however many rounds there were.
fn wealth(rounds: u32, favourable: u32) -> f64 {
    let (n, s) = (f64::from(rounds), f64::from(favourable));
    // C(n + 1, s) / C(n, s).
    let mut term = (n + 1.) / (n + 1. - s);
    let mut sum = 0.;
    for j in (0..=favourable).rev() {
        sum += term;
        // C(n + 1, j − 1) = C(n + 1, j) · j / (n + 2 − j).
        term *= f64::from(j) / (n + 2. - f64::from(j));
    }
    sum / (n + 1.)
}

/// An anytime-valid confidence interval for the median of the distribution
/// `samples` are drawn from, at error rate `rate`; none while no interval
/// is resolved (too few samples).
///
/// The same test as a comparison's, of a claim about one value `m`: "the
/// median is above `m`" counts as favourable the samples above `m`, and
/// while the median is not above `m` a sample is above it with probability
/// at most one half, so `M_n(above)` is a nonnegative supermartingale that
/// starts at one; likewise "below". Each claim is given `rate / 2`, so the
/// median lies outside the interval of every `m` neither claim resolves with
/// probability at most `rate`, however many samples are taken and whenever
/// the caller stops. With `s` the fewest of `n` favourable samples that
/// resolve a claim, "above `m`" is resolved for every `m` below the `s`-th
/// largest sample and "below `m`" for every `m` above the `s`-th smallest:
/// the interval is those two order statistics.
pub fn median_interval(samples: &[f64], rate: f64) -> Option<(f64, f64)> {
    let n = u32::try_from(samples.len()).expect("fewer than 2^32 samples");
    let resolving = resolving(n, rate)?;
    let mut sorted = samples.to_vec();
    sorted.sort_by(f64::total_cmp);
    // A claim with at most half its samples favourable is never resolved
    // (`M_n(s) ≤ 1` for `s ≤ n / 2`), so `resolving > n / 2` and the lower
    // order statistic is at most the upper.
    Some((
        sorted[(n - resolving) as usize],
        sorted[resolving as usize - 1],
    ))
}

/// The fewest of `samples` favourable samples that resolve a claim about
/// the median at error rate `rate` ([`median_interval`]); none while no
/// count of them does.
fn resolving(samples: u32, rate: f64) -> Option<u32> {
    (0..=samples).find(|&favourable| wealth(samples, favourable) * rate / 2. >= 1.)
}

/// The fewest samples [`median_interval`] bounds the median of at error
/// rate `rate`: below them no sampling is resolved, whatever it read.
pub(crate) fn fewest_bounding(rate: f64) -> usize {
    // All of `n` samples favourable multiply the mixture by a factor that
    // tends to two each, so some count resolves.
    (1..)
        .find(|&samples| resolving(samples, rate).is_some())
        .expect("samples all favourable resolve a claim") as usize
}

impl Tally {
    /// One more round came out at `difference` (`d_r`).
    pub fn observe(&mut self, difference: f64) {
        self.rounds += 1;
        if difference < 0. {
            self.ahead += 1;
        } else if difference > 0. {
            self.behind += 1;
        }
    }

    /// The rounds observed.
    pub fn rounds(&self) -> u32 {
        self.rounds
    }

    /// What the comparison has resolved at the error rate `rate`; none
    /// while it has resolved nothing.
    pub fn verdict(&self, rate: f64) -> Option<Verdict> {
        if wealth(self.rounds, self.ahead) * rate >= 1. {
            Some(Verdict::Leads)
        } else if wealth(self.rounds, self.behind) * rate >= 1. {
            Some(Verdict::Trails)
        } else {
            None
        }
    }

    /// The fewest further rounds in which the comparison can resolve as a
    /// lead: all of them ahead. Zero once it has.
    pub fn fewest(&self, rate: f64) -> u32 {
        // Each round ahead multiplies the mixture by a factor that tends to
        // two, so the threshold is reached.
        (0..)
            .find(|more| wealth(self.rounds + more, self.ahead + more) * rate >= 1.)
            .expect("rounds all ahead resolve a lead")
    }
}

/// How close to its median a measured time is resolved, as a fraction of
/// the median on either side: 10%. Measured times price the engine's
/// estimate, whose own error is about ±15% typical
/// (specs/26-10-18/device-topology.md §2.14), so precision finer than 10%
/// is discarded (§2.6). The one place it is said; [`Timing::resolved`]
/// reads it.
pub const MEDIAN_TOLERANCE: f64 = 0.10;

/// One measured time: the median of its samples, the anytime-valid
/// confidence interval for it at the error rate it was sampled at (none
/// while too few samples bound one), and how many samples it is of.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Timing {
    /// Median of the samples, in seconds per call.
    pub median: f64,
    /// The anytime-valid interval for the median; none while too few
    /// samples bound one, or where no stopping rule decided their count.
    pub interval: Option<(f64, f64)>,
    /// How many samples it is of.
    pub samples: u32,
}

impl Timing {
    /// The timing of `samples` at error rate `error_rate`: the one stopping
    /// rule every measured time is read by.
    pub fn of(samples: &[f64], error_rate: f64) -> Self {
        Self {
            median: median(samples),
            interval: median_interval(samples, error_rate),
            samples: u32::try_from(samples.len()).expect("fewer than 2^32 samples"),
        }
    }

    /// Whether the interval lies within [`MEDIAN_TOLERANCE`] of the median
    /// on either side, so sampling may stop.
    pub fn resolved(&self) -> bool {
        self.interval.is_some_and(|(lower, upper)| {
            lower >= self.median * (1. - MEDIAN_TOLERANCE)
                && upper <= self.median * (1. + MEDIAN_TOLERANCE)
        })
    }
}

/// When the stopping rule is next read: at counts growing by a sixteenth,
/// so reading it over `n` samples costs O(n log² n) rather than
/// O(n² log n), and sampling ends at most a sixteenth past where it first
/// holds; an anytime-valid interval may be read at any count.
pub(crate) fn next_reading(samples: usize) -> usize {
    (samples + samples / 16).max(samples + 1)
}

/// The order of a round's samples: a new random one each round
/// (xorshift64*). A device whose speed is still changing gives the later
/// samples of a round a different clock than the earlier ones; in a fixed
/// order that would favour the same side every round, and in a random one
/// it favours each side as often as the other, so a round is still ahead
/// with probability at most one half when the challenger is not.
pub struct Shuffle(u64);

impl Default for Shuffle {
    fn default() -> Self {
        Self(0x9e37_79b9_7f4a_7c15)
    }
}

impl Shuffle {
    /// A uniformly random index below `bound`.
    pub fn below(&mut self, bound: usize) -> usize {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        (self.0.wrapping_mul(0x2545_f491_4f6c_dd1d) >> 33) as usize % bound
    }

    /// Put `order` in a uniformly random order.
    pub fn shuffle(&mut self, order: &mut [usize]) {
        for last in (1..order.len()).rev() {
            order.swap(last, self.below(last + 1));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const RATE: f64 = 0.05;

    /// The interval needs the samples a one-sided claim at half the rate
    /// needs all favourable (8 at one in twenty), and then lies between
    /// order statistics that close on the median as samples accumulate.
    #[test]
    fn a_median_interval_is_resolved_by_order_statistics() {
        let fewest = fewest_bounding(RATE);
        assert_eq!(fewest, 8);
        let samples = (0..fewest)
            .map(|index| 1. + index as f64)
            .collect::<Vec<_>>();
        assert_eq!(median_interval(&samples[..fewest - 1], RATE), None);
        assert_eq!(median_interval(&samples, RATE), Some((1., fewest as f64)));
        let many = (0..1000)
            .map(|index| f64::from(index % 101))
            .collect::<Vec<_>>();
        let (lower, upper) = median_interval(&many, RATE).unwrap();
        assert!(lower < 50. && 50. < upper && upper - lower < 20.);
    }

    #[test]
    fn the_mixture_is_the_closed_form_of_its_integral() {
        // Midpoint quadrature of ∫ 2 (2p)^s (2(1 − p))^(n − s) dp over (½, 1).
        for (rounds, favourable) in [(1u32, 0u32), (1, 1), (7, 7), (9, 4), (30, 22), (200, 120)] {
            let steps = 200_000;
            let integral = (0..steps)
                .map(|step| {
                    let p = 0.5 + (f64::from(step) + 0.5) / f64::from(2 * steps);
                    2. * (2. * p).powi(favourable as i32)
                        * (2. * (1. - p)).powi((rounds - favourable) as i32)
                })
                .sum::<f64>()
                / f64::from(2 * steps);
            let closed = wealth(rounds, favourable);
            assert!(
                (closed - integral).abs() <= 1e-6 * integral.max(1.),
                "{rounds} {favourable}: {closed} {integral}"
            );
        }
        // Seven rounds all ahead resolve a lead at one in twenty; six do not.
        assert_eq!(Tally::default().fewest(RATE), 7);
        assert!(wealth(6, 6) < 20. && wealth(7, 7) >= 20.);
    }

    /// No margin is asked of a lead: a challenger faster than the other by
    /// less than 2% in every one of its rounds leads as soon as the test
    /// resolves, in the fewest rounds there are.
    #[test]
    fn a_challenger_faster_by_less_than_two_percent_in_every_round_leads() {
        for gain in [0.019, 0.01, 0.001] {
            let mut evidence = Tally::default();
            let fewest = evidence.fewest(RATE);
            for round in 0..fewest {
                assert_eq!(evidence.verdict(RATE), None, "{gain}: after {round} rounds");
                evidence.observe((1. - gain) - 1.);
            }
            assert_eq!(evidence.verdict(RATE), Some(Verdict::Leads), "{gain}");
            assert_eq!(evidence.rounds(), 7);
        }
    }

    #[test]
    fn a_shuffle_puts_every_slot_first_as_often_as_another() {
        let mut shuffle = Shuffle::default();
        let mut first = [0usize; 4];
        for _ in 0..40_000 {
            let mut order = [0, 1, 2, 3];
            shuffle.shuffle(&mut order);
            first[order[0]] += 1;
        }
        assert!(
            first.iter().all(|count| (9_500..10_500).contains(count)),
            "{first:?}"
        );
    }

    /// The even-count median is the mean of the middle two.
    #[test]
    fn an_even_count_median_is_the_mean_of_the_middle_two() {
        assert_eq!(median(&[4., 1., 3., 2.]), 2.5);
        assert_eq!(median(&[3., 1., 2.]), 2.);
    }

    /// The stopping rule is read at every count to 32 and then at counts
    /// growing by a sixteenth.
    #[test]
    fn the_stopping_rule_is_read_at_counts_growing_by_a_sixteenth() {
        let mut readings = Vec::new();
        let mut next = 1;
        for _ in 0..200 {
            readings.push(next);
            next = next_reading(next);
        }
        assert!(readings.windows(2).all(|pair| pair[1] > pair[0]));
        assert!(readings
            .windows(2)
            .all(|pair| pair[0] < 16 || pair[1] <= pair[0] * 17 / 16));
    }
}
