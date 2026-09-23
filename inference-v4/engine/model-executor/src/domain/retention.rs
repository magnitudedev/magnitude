//! Owned feature retention charged against the admitted budget.

use super::*;
use crate::ResourceAllocator;

impl<F: ProgramFamily> FeatureRetainer for ExecutorDomain<F> {
    fn retain(&mut self, span: FeatureSpan) -> Result<RetainedFeatureSpan, String> {
        self.healthy().map_err(|error| error.to_string())?;
        let end = span
            .start
            .checked_add(span.count)
            .ok_or("retained feature range overflow")?;
        let source = self
            .domain
            .feature_span(&span)
            .map_err(|error| error.to_string())?;
        if end > span.features.allocation().rows() {
            return Err("retained feature range exceeds source".into());
        }
        let bytes = source.byte_len();
        let (mut destination, claim) = ResourceAllocator::retained_feature(
            &self.execution,
            self.domain.device(),
            &source,
            &self.retained_used,
        )
        .map_err(|error| error.to_string())?;
        let host = source.read_to_host().map_err(|error| error.to_string())?;
        destination
            .write_from_host(&host)
            .map_err(|error| error.to_string())?;
        let features = self
            .domain
            .publish_retained_features(destination, claim)
            .map_err(|error| error.to_string())?;
        let retained =
            FeatureSpan::new(features, 0, span.count).map_err(|error| error.to_string())?;
        RetainedFeatureSpan::new(retained, bytes).map_err(|error| error.to_string())
    }
}
