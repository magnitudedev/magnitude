//! A comparison freezes its arms, invocation and content seed. Scheduling owns
//! when to run the next stage; pausing never restarts its error allowance.
use super::evolution::Random;
use super::statistics::{alpha, median_interval};
use std::time::Duration;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Arm {
    Incumbent,
    Challenger,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Verdict {
    Pending,
    Improved,
    Inferior,
    EnvironmentChanged,
    Inconclusive,
}

pub(super) struct Comparison {
    event: u64,
    seed: u64,
    stage: u32,
    environment: [u8; 32],
    incumbent: Vec<f64>,
    challenger: Vec<f64>,
    pending: Option<Arm>,
    pair_first: Option<Arm>,
    verdict: Verdict,
}

impl Comparison {
    pub fn new(event: u64, seed: u64, environment: [u8; 32]) -> Self {
        assert!(event > 0, "comparison event numbering starts at one");
        Self {
            event,
            seed,
            environment,
            stage: 0,
            incumbent: Vec::new(),
            challenger: Vec::new(),
            pending: None,
            pair_first: None,
            verdict: Verdict::Pending,
        }
    }

    pub fn seed(&self) -> u64 {
        self.seed
    }
    pub fn stage(&self) -> u32 {
        self.stage
    }
    pub fn verdict(&self) -> Verdict {
        self.verdict
    }
    pub fn stage_complete(&self) -> bool {
        let target = 8usize.checked_shl(self.stage).unwrap_or(usize::MAX);
        self.incumbent.len() == target && self.challenger.len() == target
    }

    /// One pair has randomized order, preventing either arm from owning the
    /// systematically colder or later position. An interrupted request repeats
    /// the same arm until its completed observation is recorded.
    pub fn next(&mut self, random: &mut Random) -> Option<Arm> {
        if self.verdict != Verdict::Pending || self.stage_complete() {
            return None;
        }
        if let Some(arm) = self.pending {
            return Some(arm);
        }
        let arm = match self.pair_first.take() {
            Some(Arm::Incumbent) => Arm::Challenger,
            Some(Arm::Challenger) => Arm::Incumbent,
            None => {
                let arm = if random.next() & 1 == 0 {
                    Arm::Incumbent
                } else {
                    Arm::Challenger
                };
                self.pair_first = Some(arm);
                arm
            }
        };
        self.pending = Some(arm);
        Some(arm)
    }

    pub fn record(&mut self, sample: Duration, environment: [u8; 32]) {
        let arm = self
            .pending
            .take()
            .expect("observation requires a pending trial");
        if environment != self.environment {
            self.verdict = Verdict::EnvironmentChanged;
            return;
        }
        match arm {
            Arm::Incumbent => &mut self.incumbent,
            Arm::Challenger => &mut self.challenger,
        }
        .push(sample.as_secs_f64());
        if self.stage_complete() {
            let allowance = alpha(0.05, self.event, self.stage);
            let incumbent = median_interval(&self.incumbent, allowance);
            let challenger = median_interval(&self.challenger, allowance);
            if challenger.upper < incumbent.lower {
                self.verdict = Verdict::Improved;
            } else if challenger.lower > incumbent.upper {
                self.verdict = Verdict::Inferior;
            }
        }
    }

    /// A scheduler must explicitly buy another stage. Merely asking for the
    /// verdict cannot inspect a new interval or spend fresh error probability.
    pub fn advance(&mut self) -> bool {
        if self.verdict != Verdict::Pending || !self.stage_complete() {
            return false;
        }
        // Bounded per-comparison storage; equal arms must not consume the
        // entire campaign. This stops without claiming either arm is better.
        if self.incumbent.len() >= 4096 {
            self.verdict = Verdict::Inconclusive;
            return false;
        }
        self.stage += 1;
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stage(comparison: &mut Comparison, random: &mut Random, incumbent: u64, challenger: u64) {
        while let Some(arm) = comparison.next(random) {
            assert_eq!(comparison.next(random), Some(arm));
            comparison.record(
                Duration::from_nanos(match arm {
                    Arm::Incumbent => incumbent,
                    Arm::Challenger => challenger,
                }),
                [1; 32],
            );
        }
    }

    #[test]
    fn promotion_uses_fresh_stages_and_preserves_event_allowance() {
        let mut comparison = Comparison::new(1, 19, [1; 32]);
        let mut random = Random(12);
        stage(&mut comparison, &mut random, 100, 50);
        assert_eq!(comparison.seed(), 19);
        assert_eq!(comparison.verdict(), Verdict::Pending);
        assert_eq!(comparison.stage(), 0);
        assert!(comparison.advance());
        stage(&mut comparison, &mut random, 100, 50);
        assert_eq!(comparison.verdict(), Verdict::Improved);
        assert!(!comparison.advance());
        // Later events retain the smaller allowance, even with a large gain.
        let mut late = Comparison::new(1_000_000, 19, [1; 32]);
        stage(&mut late, &mut random, 100, 1);
        late.advance();
        stage(&mut late, &mut random, 100, 1);
        assert_eq!(late.verdict(), Verdict::Pending);
    }

    #[test]
    fn ties_remain_inconclusive_and_environment_changes_invalidate() {
        let mut random = Random(99);
        let mut tie = Comparison::new(1, 20, [1; 32]);
        for _ in 0..4 {
            stage(&mut tie, &mut random, 100, 100);
            assert_eq!(tie.verdict(), Verdict::Pending);
            assert!(tie.advance());
        }
        let arm = tie.next(&mut random).unwrap();
        assert_eq!(tie.next(&mut random), Some(arm));
        tie.record(Duration::from_nanos(1), [2; 32]);
        assert_eq!(tie.verdict(), Verdict::EnvironmentChanged);
        assert!(tie.next(&mut random).is_none());
    }

    #[test]
    fn tied_comparison_stops_at_its_sample_bound() {
        let mut comparison = Comparison::new(1, 0, [1; 32]);
        let mut random = Random(0);
        loop {
            stage(&mut comparison, &mut random, 10, 10);
            if !comparison.advance() {
                break;
            }
        }
        assert_eq!(comparison.verdict(), Verdict::Inconclusive);
        assert_eq!(comparison.incumbent.len(), 4096);
        assert_eq!(comparison.challenger.len(), 4096);
        assert!(comparison.next(&mut random).is_none());
    }

    #[test]
    fn each_pair_contains_both_arms_in_seeded_order() {
        let mut a = Comparison::new(1, 0, [1; 32]);
        let mut b = Comparison::new(1, 0, [1; 32]);
        let mut ra = Random(91);
        let mut rb = Random(91);
        for _ in 0..8 {
            let first = a.next(&mut ra).unwrap();
            assert_eq!(b.next(&mut rb), Some(first));
            a.record(Duration::from_nanos(1), [1; 32]);
            b.record(Duration::from_nanos(1), [1; 32]);
            let second = a.next(&mut ra).unwrap();
            assert_ne!(first, second);
            assert_eq!(b.next(&mut rb), Some(second));
            a.record(Duration::from_nanos(1), [1; 32]);
            b.record(Duration::from_nanos(1), [1; 32]);
        }
        assert!(a.stage_complete());
    }
    #[test]
    fn separate_case_events_have_fresh_samples_and_independent_allowances() {
        let mut first = Comparison::new(1, 11, [1; 32]);
        let mut random = Random(12);
        stage(&mut first, &mut random, 100, 50);
        first.advance();
        stage(&mut first, &mut random, 100, 50);
        assert_eq!(first.verdict(), Verdict::Improved);
        let mut second = Comparison::new(2, 12, [1; 32]);
        assert!(second.incumbent.is_empty() && second.challenger.is_empty());
        stage(&mut second, &mut random, 100, 200);
        assert_eq!(second.verdict(), Verdict::Pending);
        second.advance();
        stage(&mut second, &mut random, 100, 200);
        assert_eq!(second.verdict(), Verdict::Inferior);
        assert_ne!(first.seed(), second.seed());
        assert!(
            alpha(0.05, second.event, second.stage()) < alpha(0.05, first.event, first.stage())
        );
        assert_eq!(first.verdict(), Verdict::Improved);
    }

    #[derive(Clone, Copy, Debug)]
    enum Distribution {
        EqualUniform,
        EqualSkewed,
        Faster,
        Slower,
        Correlated,
        Drift,
    }
    fn simulate(distribution: Distribution, replicate: u64, case: u64) -> (Verdict, usize) {
        let mut comparison = Comparison::new(case + 1, 0xabc0 + case, [1; 32]);
        // Distinct streams for arm observations and randomized pair ordering.
        let mut incumbent = Random(replicate.wrapping_mul(7919) ^ 0x7323 ^ case);
        let mut challenger = Random(replicate.wrapping_mul(7919) ^ 0xf871 ^ case);
        let mut order = Random(replicate.wrapping_mul(7919) ^ 0x56ac ^ case);
        let mut trial = 0usize;
        loop {
            while let Some(arm) = comparison.next(&mut order) {
                let random = match arm {
                    Arm::Incumbent => &mut incumbent,
                    Arm::Challenger => &mut challenger,
                };
                let uniform = 500 + random.index(1001) as u64;
                let latency = match distribution {
                    Distribution::EqualUniform => uniform,
                    Distribution::EqualSkewed => 500 + (uniform - 500).pow(3) / 1_000_000,
                    Distribution::Faster if arm == Arm::Challenger => uniform * 4 / 5,
                    Distribution::Slower if arm == Arm::Challenger => uniform * 6 / 5,
                    Distribution::Faster | Distribution::Slower => uniform,
                    // Identical marginal medians, but samples inside an event
                    // share a latent variable: the IID premise does not hold.
                    Distribution::Correlated => {
                        if arm == Arm::Incumbent {
                            1000
                        } else {
                            500 + Random(replicate.wrapping_mul(7919) ^ 0x3914).index(1001) as u64
                        }
                    }
                    // Common abrupt clock/load drift inside both arms. Pair
                    // randomization cannot establish stationarity.
                    Distribution::Drift => uniform + if trial >= 20 { 1000 } else { 0 },
                };
                comparison.record(Duration::from_nanos(latency), [1; 32]);
                trial += 1;
            }
            if !comparison.advance() {
                break;
            }
        }
        (
            comparison.verdict(),
            comparison.incumbent.len() + comparison.challenger.len(),
        )
    }

    #[test]
    #[ignore = "deterministic quantitative statistical qualification; run explicitly with --nocapture"]
    fn seeded_iid_and_adversarial_confirmation_qualification() {
        const REPLICATES: u64 = 256;
        println!("confirmation_qualification_v1,distribution,replicates,two_case_promotions,any_case_improvements,inconclusive_cases,median_samples,max_samples,iid_guarantee");
        for distribution in [
            Distribution::EqualUniform,
            Distribution::EqualSkewed,
            Distribution::Faster,
            Distribution::Slower,
            Distribution::Correlated,
            Distribution::Drift,
        ] {
            let mut promoted = 0;
            let mut any_improved = 0;
            let mut inconclusive = 0;
            let mut samples = Vec::new();
            for replicate in 0..REPLICATES {
                let first = simulate(distribution, replicate, 0);
                let second = simulate(distribution, replicate, 1);
                promoted +=
                    u64::from(first.0 == Verdict::Improved && second.0 == Verdict::Improved);
                any_improved +=
                    u64::from(first.0 == Verdict::Improved || second.0 == Verdict::Improved);
                inconclusive += u64::from(first.0 == Verdict::Inconclusive)
                    + u64::from(second.0 == Verdict::Inconclusive);
                samples.push(first.1 + second.1);
            }
            samples.sort_unstable();
            let iid = matches!(
                distribution,
                Distribution::EqualUniform
                    | Distribution::EqualSkewed
                    | Distribution::Faster
                    | Distribution::Slower
            );
            println!("confirmation_qualification_v1,{distribution:?},{REPLICATES},{promoted},{any_improved},{inconclusive},{},{},{iid}",samples[samples.len()/2],samples.last().unwrap());
            match distribution {
                Distribution::EqualUniform | Distribution::EqualSkewed | Distribution::Slower => {
                    assert!(
                        any_improved <= REPLICATES / 20,
                        "fixed seeded IID qualification exceeded campaign delta"
                    )
                }
                Distribution::Faster => assert!(
                    promoted >= REPLICATES * 9 / 10,
                    "substantial IID gains should be resolved within bounded samples"
                ),
                Distribution::Correlated => assert!(
                    promoted > REPLICATES / 4,
                    "adversarial correlation must expose the premise, not claim a guarantee"
                ),
                Distribution::Drift => {}
            }
        }
    }
}
