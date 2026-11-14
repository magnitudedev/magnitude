//! How many units one host-timed observation measures: sized from the
//! resolution its qualification band needs, by one rule on every backend.

use crate::DurationInterval;
use std::time::Duration;

/// The timer uncertainty one observation carries, in timer resolutions. An
/// observation is the difference of two timer readings, each quantized to the
/// resolution, which is the widening every backend applies to an observed
/// interval on each side.
const READING_UNCERTAINTY_RESOLUTIONS: u64 = 2;

/// How many times the timer's uncertainty an observation's band must hold.
/// The timer then takes at most a quarter of the band, leaving three quarters
/// for the operation's own spread; doubling calibration stops below twice the
/// required duration, so the timer's share lands between an eighth and a
/// quarter.
const BAND_PER_TIMER_UNCERTAINTY: u64 = 4;

/// The most units calibration or escalation measures in one observation. It
/// bounds calibration of an operation the timer never resolves, whose
/// observations come out empty however many units they hold.
pub const MAX_OBSERVATION_UNITS: u64 = 1 << 30;

/// The longest an acquisition round may take. Escalation sizes the next
/// round by how far the last missed its band, which grows a round's
/// duration with it; a round that would take longer is not run and the
/// probe does not qualify ([`escalate_observation_units`]), so a probe on a
/// host too noisy for its band fails within its rounds' seconds rather than
/// growing toward [`MAX_OBSERVATION_UNITS`] (an hour and more for an
/// operation of microseconds). A second is about twenty times the round a
/// band needs at a microsecond timer (64 observations of the 800 us that a
/// 1% band requires), room for the escalation jitter calls for.
pub const MAX_ROUND: Duration = Duration::from_secs(1);

/// The bytes of the instance `core.allocation-instance` acquires, on every
/// backend. The model charges one latency per instance whatever its size,
/// so it is measured at one small instance.
pub const ALLOCATION_INSTANCE_BYTES: u64 = 64 * 1024;

/// The shortest observation, in nanoseconds, whose relative half-width band
/// of `percent` holds [`BAND_PER_TIMER_UNCERTAINTY`] times the timer's
/// uncertainty: a half-width `h` qualifies when `h * 100 <= midpoint *
/// percent`.
pub fn required_observation_ns(timer_resolution_ns: u64, percent: u64) -> u64 {
    required_beyond_ns(timer_resolution_ns.saturating_mul(READING_UNCERTAINTY_RESOLUTIONS), percent)
}

/// The shortest observation, in nanoseconds, whose band of `percent` holds
/// [`BAND_PER_TIMER_UNCERTAINTY`] times `uncertainty_ns`: everything an
/// observation carries that its own length does not shrink, the timer's
/// readings ([`required_observation_ns`]) and a separately measured cost
/// subtracted from it (the width of that cost's interval) alike.
pub fn required_beyond_ns(uncertainty_ns: u64, percent: u64) -> u64 {
    assert!(percent != 0, "a qualification band has a nonzero width");
    uncertainty_ns.max(1).saturating_mul(BAND_PER_TIMER_UNCERTAINTY * 100) / percent
}

/// The unit count one observation measures: doubled from one until
/// `observe_ns(units)`, the duration of one observation of that many units,
/// reaches `required_ns`, or [`MAX_OBSERVATION_UNITS`].
pub fn calibrate_observation_units<E>(
    required_ns: u64,
    mut observe_ns: impl FnMut(u64) -> Result<u64, E>,
) -> Result<u64, E> {
    let mut units = 1;
    while units < MAX_OBSERVATION_UNITS && observe_ns(units)? < required_ns {
        units *= 2;
    }
    Ok(units)
}

/// The unit count of the round after one of `units` that took `round`
/// and whose interval `failed` did not qualify to `percent`: as many times
/// more as its relative half-width exceeded the band, at least twice, within
/// [`MAX_OBSERVATION_UNITS`]; none where that round would take longer than
/// [`MAX_ROUND`], as a round's duration grows with its units, and the probe
/// then does not qualify. What widens an observation's band beyond the
/// timer is mostly jitter added to
/// each observation whole (the host's scheduling, a subtracted setup's
/// spread), which shrinks against the observation in proportion to its
/// length, so one round sized by the miss reaches the band where doubling
/// would take as many rounds as the miss has factors of two. A round with
/// no interval (a difference that came out empty) doubles.
pub fn escalate_observation_units(
    units: u64,
    round: Duration,
    failed: Option<DurationInterval>,
    percent: u64,
) -> Option<u64> {
    let factor = failed
        .map(|interval| {
            // Half-width over midpoint, against percent / 100.
            let width = u128::from(interval.upper_numerator.saturating_sub(interval.lower_numerator));
            let sum = u128::from(interval.upper_numerator) + u128::from(interval.lower_numerator);
            (width * 100).div_ceil((sum * u128::from(percent)).max(1))
        })
        .unwrap_or(2)
        .max(2);
    (round.as_nanos().max(1).saturating_mul(factor) <= MAX_ROUND.as_nanos()).then(|| {
        u64::try_from(u128::from(units).saturating_mul(factor))
            .unwrap_or(u64::MAX)
            .min(MAX_OBSERVATION_UNITS)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn required_duration_gives_the_timer_a_quarter_of_the_band() {
        // 2 resolutions of uncertainty, 4 of them per band, at 2%: 400x.
        assert_eq!(required_observation_ns(42, 2), 16_800);
        assert_eq!(required_observation_ns(1_000, 1), 800_000);
    }

    #[test]
    fn calibration_stops_at_the_first_count_reaching_the_duration() {
        let units = calibrate_observation_units::<()>(1_000, |units| Ok(units * 3)).unwrap();
        assert_eq!(units, 512);
    }

    #[test]
    fn calibration_is_bounded_for_an_unresolved_operation() {
        let units = calibrate_observation_units::<()>(1, |_| Ok(0)).unwrap();
        assert_eq!(units, MAX_OBSERVATION_UNITS);
        assert_eq!(escalate_observation_units(units, Duration::from_nanos(1), None, 2), Some(MAX_OBSERVATION_UNITS));
    }

    #[test]
    fn escalation_scales_by_the_miss() {
        // A 10% half-width against a 2% band: five times the units.
        assert_eq!(escalate_observation_units(100, Duration::from_micros(1), Some(DurationInterval::new(90, 110, 1)), 2), Some(500));
        // A near miss still doubles; no interval doubles.
        assert_eq!(escalate_observation_units(100, Duration::from_micros(1), Some(DurationInterval::new(97, 103, 1)), 2), Some(200));
        assert_eq!(escalate_observation_units(100, Duration::from_micros(1), None, 2), Some(200));
        assert_eq!(required_beyond_ns(1_000, 2), 200_000);
    }

    #[test]
    fn escalation_stops_at_the_round_budget() {
        // Five times a 0.2 s round is a second: run. Five times 0.25 s is not.
        let missed = Some(DurationInterval::new(90, 110, 1));
        assert_eq!(escalate_observation_units(100, MAX_ROUND / 5, missed, 2), Some(500));
        assert_eq!(escalate_observation_units(100, MAX_ROUND / 4, missed, 2), None);
        assert_eq!(escalate_observation_units(100, MAX_ROUND, None, 2), None);
    }
}
