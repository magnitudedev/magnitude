//! Metal command-buffer timestamps share the system mach clock. A duration
//! is evidence only when its endpoints lie inside the command's lifetime.

/// Seconds on the system mach clock used by MTLCommandBuffer GPUStartTime
/// and GPUEndTime. CLOCK_UPTIME_RAW is mach_absolute_time in nanoseconds.
pub fn host_seconds() -> f64 {
    unsafe extern "C" {
        fn clock_gettime_nsec_np(clock_id: std::ffi::c_int) -> u64;
    }
    const CLOCK_UPTIME_RAW: std::ffi::c_int = 8;
    // SAFETY: a pure libc clock read.
    unsafe { clock_gettime_nsec_np(CLOCK_UPTIME_RAW) as f64 / 1e9 }
}

/// The lower endpoint of one command's submission envelope. Observation
/// stores one f64 per committed command in its retained timing vector,
/// including commands before a host gate or an abandoned batch.
pub(crate) struct SubmissionTime {
    before: f64,
}

impl SubmissionTime {
    /// Call immediately before committing the command, never after commit.
    pub(crate) fn before_commit() -> Self {
        Self { before: host_seconds() }
    }

    /// Read after completion and after retrieving both device endpoints.
    /// Missing or stale timestamps are an error, not zero time or host time.
    pub(crate) fn interval(&self, start: f64, end: f64) -> Result<(f64, f64), String> {
        checked_interval(self.before, start, end, host_seconds())
    }
}

fn checked_interval(before: f64, start: f64, end: f64, after: f64) -> Result<(f64, f64), String> {
    if start.is_finite() && end.is_finite() && start > 0.0
        && before <= start && start <= end && end <= after {
        Ok((start, end))
    } else {
        Err(format!("Metal command-buffer timestamp interval [{start}, {end}] is invalid \
            within submission envelope [{before}, {after}] on the system mach clock"))
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
            let error = checked_interval(10.0, start, end, 13.0).unwrap_err();
            assert!(error.contains(&format!("[{start}, {end}]")));
            assert!(error.contains("[10, 13]"));
        }
        assert_eq!(checked_interval(10.0, 10.0, 13.0, 13.0), Ok((10.0, 13.0)));
        assert_eq!(checked_interval(10.0, 11.0, 11.0, 13.0), Ok((11.0, 11.0)));
        assert_eq!(checked_interval(10.0, 11.0, 12.0, 13.0), Ok((11.0, 12.0)));
    }
}
