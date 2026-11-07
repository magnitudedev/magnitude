//! The load's own serving rates: what every memory holding's value is
//! multiplied by (a bank slot's step saving by decode steps a second, a
//! prefix's saved tokens by requests a second and the prefill's seconds a
//! token). Measured from this load's serving alone, never from recorded
//! data of a model.
//!
//! Each count decays exponentially with a time constant the owner chooses
//! (the span the memory decisions are earned over), so the rates follow
//! the workload as it changes. Before the load serves, the counts start at
//! a prior's rates over one time constant: the measurements replace the
//! prior as they accumulate, at the same decay, with no weight of their own
//! to choose.
use std::time::{Duration, Instant};

/// The load's serving rates.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WorkloadRates {
    /// Decode steps a second of serving.
    pub decode_steps_per_s: f64,
    /// The prefill's speed while it runs: tokens a second of prefill.
    pub prefill_tokens_per_s: f64,
    /// Requests a second of serving: each prefills its prompt once, in
    /// whole or from a resumed prefix, so the prompts prefilled a second
    /// too.
    pub requests_per_s: f64,
    /// Requests a second whose input the vision projector encodes.
    pub vision_uses_per_s: f64,
}

impl WorkloadRates {
    /// The rates before the load serves, with no request history to read:
    /// one stream that prefills `prefill_tokens` (the planned context) in
    /// `prefill_seconds`, then decodes `decode_steps` steps of
    /// `step_seconds`, then repeats.
    ///
    /// # Panics
    /// Where a figure is not positive and finite: a load that cannot
    /// prefill or step has no rates.
    pub fn prior(step_seconds: f64, decode_steps: f64, prefill_tokens: f64, prefill_seconds: f64) -> Self {
        for figure in [step_seconds, decode_steps, prefill_tokens, prefill_seconds] {
            assert!(figure.is_finite() && figure > 0.0, "a workload prior needs positive figures, not {figure}");
        }
        let span = prefill_seconds + decode_steps * step_seconds;
        Self {
            decode_steps_per_s: decode_steps / span,
            prefill_tokens_per_s: prefill_tokens / prefill_seconds,
            requests_per_s: 1.0 / span,
            vision_uses_per_s: 0.0,
        }
    }
}

/// The decaying counts the rates are read from.
#[derive(Clone, Debug)]
pub struct WorkloadMeter {
    /// The time constant every count decays with.
    constant: Duration,
    /// When the counts were last decayed.
    at: Instant,
    decode_steps: f64,
    requests: f64,
    vision_uses: f64,
    /// Tokens prefilled and the prefill seconds they took, decayed alike;
    /// the speed is the prior's until a prefill is measured.
    prefill_tokens: f64,
    prefill_seconds: f64,
    prior_prefill_speed: f64,
}

impl WorkloadMeter {
    /// A meter at `now` whose counts decay with `constant`, starting from
    /// `prior` over one constant.
    ///
    /// # Panics
    /// Where `constant` is zero: a rate needs a span.
    pub fn new(prior: WorkloadRates, constant: Duration, now: Instant) -> Self {
        assert!(!constant.is_zero(), "a workload meter decays over a span, not zero");
        let span = constant.as_secs_f64();
        Self {
            constant,
            at: now,
            decode_steps: prior.decode_steps_per_s * span,
            requests: prior.requests_per_s * span,
            vision_uses: prior.vision_uses_per_s * span,
            prefill_tokens: 0.0,
            prefill_seconds: 0.0,
            prior_prefill_speed: prior.prefill_tokens_per_s,
        }
    }

    /// Decay every count to `now`; a reading earlier than the last is the
    /// same instant.
    fn decay(&mut self, now: Instant) {
        let elapsed = now.saturating_duration_since(self.at).as_secs_f64();
        let kept = (-elapsed / self.constant.as_secs_f64()).exp();
        for count in [
            &mut self.decode_steps,
            &mut self.requests,
            &mut self.vision_uses,
            &mut self.prefill_tokens,
            &mut self.prefill_seconds,
        ] {
            *count *= kept;
        }
        self.at = self.at.max(now);
    }

    /// `steps` decode steps ended at `now`.
    pub fn decoded(&mut self, now: Instant, steps: u64) {
        self.decay(now);
        self.decode_steps += steps as f64;
    }

    /// A prefill round of `tokens` tokens ended at `now`, after `seconds`:
    /// what the prefill's speed is read from.
    pub fn prefilled(&mut self, now: Instant, tokens: u64, seconds: f64) {
        self.decay(now);
        self.prefill_tokens += tokens as f64;
        self.prefill_seconds += seconds.max(0.0);
    }

    /// A request arrived at `now`, its input encoded by the vision
    /// projector where `vision`.
    pub fn requested(&mut self, now: Instant, vision: bool) {
        self.decay(now);
        self.requests += 1.0;
        self.vision_uses += f64::from(u8::from(vision));
    }

    /// The rates at `now`: each decayed count over the time constant.
    pub fn rates(&self, now: Instant) -> WorkloadRates {
        let mut meter = self.clone();
        meter.decay(now);
        let span = self.constant.as_secs_f64();
        WorkloadRates {
            decode_steps_per_s: meter.decode_steps / span,
            prefill_tokens_per_s: if meter.prefill_seconds > 0.0 {
                meter.prefill_tokens / meter.prefill_seconds
            } else {
                meter.prior_prefill_speed
            },
            requests_per_s: meter.requests / span,
            vision_uses_per_s: meter.vision_uses / span,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() <= 1e-9 * a.abs().max(b.abs()).max(1.0)
    }

    /// One stream: prefill 2,000 tokens in 2 s, decode 100 steps of 0.08 s.
    #[test]
    fn the_prior_is_one_stream_prefilling_then_decoding() {
        let prior = WorkloadRates::prior(0.08, 100.0, 2000.0, 2.0);
        assert!(close(prior.requests_per_s, 0.1));
        assert!(close(prior.decode_steps_per_s, 10.0));
        assert!(close(prior.prefill_tokens_per_s, 1000.0));
    }

    /// The prior holds until serving is measured; it fades with the time
    /// constant as the load's own counts replace it.
    #[test]
    fn rates_start_at_the_prior_and_decay_to_what_the_load_serves() {
        let start = Instant::now();
        let constant = Duration::from_secs(10);
        let prior = WorkloadRates::prior(0.08, 100.0, 2000.0, 2.0);
        let mut meter = WorkloadMeter::new(prior, constant, start);
        assert!(close(meter.rates(start).decode_steps_per_s, 10.0));
        // Idle for one constant: the prior's count falls by e.
        let idle = start + constant;
        assert!(close(meter.rates(idle).decode_steps_per_s, 10.0 / std::f64::consts::E));
        // Then 20 steps a second for many constants: the rate reaches 20.
        let mut now = idle;
        for _ in 0..2000 {
            now += Duration::from_millis(50);
            meter.decoded(now, 1);
        }
        assert!((meter.rates(now).decode_steps_per_s - 20.0).abs() < 0.2);
        // The prefill speed is the prior's until a prefill is measured.
        assert!(close(meter.rates(now).prefill_tokens_per_s, 1000.0));
        meter.prefilled(now, 600, 0.3);
        assert!(close(meter.rates(now).prefill_tokens_per_s, 2000.0));
    }
}
