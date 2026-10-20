//! The sampling statistics: the median, the paired sign test a timed
//! comparison is resolved by, and the order of a round's samples.
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
}
