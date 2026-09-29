//! Completed feature rows copied to host memory for generation methods.

use super::*;

impl<F: ProgramFamily> FeatureReader for ExecutorDomain<F> {
    fn read(&mut self, span: &FeatureSpan) -> Result<FeatureRows, String> {
        self.healthy().map_err(|error| error.to_string())?;
        let source = self
            .domain
            .feature_span(span)
            .map_err(|error| error.to_string())?;
        let bytes = source.read_to_host().map_err(|error| error.to_string())?;
        FeatureRows::new(bytes.into(), span.count).map_err(|error| error.to_string())
    }
}
