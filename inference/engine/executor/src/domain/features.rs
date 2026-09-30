//! Completed feature rows copied to host memory for generation methods.

use super::*;
use std::time::Instant;

impl<F: ProgramFamily> FeatureReader for ExecutorDomain<F> {
    fn read(&mut self, span: &FeatureSpan) -> Result<FeatureRows, String> {
        self.healthy().map_err(|error| error.to_string())?;
        let started = Instant::now();
        let source = self
            .domain
            .feature_span(span)
            .map_err(|error| error.to_string())?;
        let bytes = source.read_to_host().map_err(|error| error.to_string())?;
        let result = FeatureRows::new(bytes.into(), span.count).map_err(|error| error.to_string());
        if std::env::var_os("MAGNITUDE_TRACE_FEATURE_READS").is_some() {
            eprintln!(
                "feature_read rows={} elapsed_ms={:.3}",
                span.count,
                started.elapsed().as_secs_f64() * 1000.0
            );
        }
        result
    }
}
