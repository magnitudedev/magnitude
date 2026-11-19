//! Metal device timestamps, mapped onto the system mach clock, are evidence
//! of device time only when they lie inside the work's submission envelope.

/// Seconds on the system mach clock, which GPU timestamp samples are
/// mapped onto. CLOCK_UPTIME_RAW is mach_absolute_time in nanoseconds.
pub fn host_seconds() -> f64 {
    unsafe extern "C" {
        fn clock_gettime_nsec_np(clock_id: std::ffi::c_int) -> u64;
    }
    const CLOCK_UPTIME_RAW: std::ffi::c_int = 8;
    // SAFETY: a pure libc clock read.
    unsafe { clock_gettime_nsec_np(CLOCK_UPTIME_RAW) as f64 / 1e9 }
}

/// `[start, end]`, device timestamps on the [`host_seconds`] clock, when it
/// lies inside the submission envelope `[before, after]`: from the host
/// clock immediately before the work's first commit to a reading after it
/// completed, widened on each side by `uncertainty`, how far the mapping of
/// device timestamps onto the host clock may err. Missing or stale
/// timestamps are an error, not zero time or host time.
pub(crate) fn checked_interval(
    before: f64,
    start: f64,
    end: f64,
    after: f64,
    uncertainty: f64,
) -> Result<(f64, f64), String> {
    if start.is_finite() && end.is_finite() && start > 0.0
        && before - uncertainty <= start && start <= end && end <= after + uncertainty {
        Ok((start, end))
    } else {
        Err(format!("Metal device timestamp interval [{start}, {end}] is invalid \
            within submission envelope [{before}, {after}] ± {uncertainty} s on the system mach clock"))
    }
}

#[cfg(test)]
mod tests {
    use super::checked_interval;

    #[test]
    fn command_timestamps_must_belong_to_the_submission() {
        for (start, end) in [
            (0.0, 455_114.0), // A missing start cannot become machine uptime.
            (0.0, 0.0),
            (f64::NAN, 12.0),
            (11.0, f64::INFINITY),
            (12.0, 11.0),
            (1.0, 2.0), // Positive, ordered timestamps from an earlier command.
            (9.0, 12.0),
            (11.0, 14.0),
        ] {
            let error = checked_interval(10.0, start, end, 13.0, 0.0).unwrap_err();
            assert!(error.contains(&format!("[{start}, {end}]")));
            assert!(error.contains("[10, 13]"));
        }
        assert_eq!(checked_interval(10.0, 10.0, 13.0, 13.0, 0.0), Ok((10.0, 13.0)));
        assert_eq!(checked_interval(10.0, 11.0, 11.0, 13.0, 0.0), Ok((11.0, 11.0)));
        assert_eq!(checked_interval(10.0, 11.0, 12.0, 13.0, 0.0), Ok((11.0, 12.0)));
        // The mapping's error widens the envelope, and only by itself.
        assert_eq!(checked_interval(10.0, 9.5, 13.5, 13.0, 0.5), Ok((9.5, 13.5)));
        assert!(checked_interval(10.0, 9.4, 13.0, 13.0, 0.5).is_err());
        assert!(checked_interval(10.0, 10.0, 13.6, 13.0, 0.5).is_err());
    }
}
