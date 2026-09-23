use crate::method::{
    Method, MethodCheckpoint, MethodCheckpointError, MethodEffects, MethodRequirements,
    MethodState, Propose, Verification,
};
use magnitude_model_executor::{Demand, FeatureRef, Operation, Outcome, RequestId, TokenId};

#[derive(Clone, Copy, Debug, Default)]
pub struct Plain;

impl Method for Plain {
    fn identity(&self) -> &str {
        "plain"
    }

    fn requires(&self) -> MethodRequirements {
        MethodRequirements {
            prefill_demand: Demand::NONE,
            verify_demand: Demand::NONE,
            head: false,
        }
    }

    fn create(&self, checkpoint: Option<&MethodCheckpoint>) -> Box<dyn MethodState> {
        debug_assert!(checkpoint.is_none_or(|checkpoint| *checkpoint == MethodCheckpoint::Plain));
        Box::new(PlainState)
    }
}

#[derive(Clone)]
struct PlainState;

impl MethodState for PlainState {
    fn fork_transition(&self) -> Result<Box<dyn MethodState>, String> {
        Ok(Box::new(self.clone()))
    }
    fn prime(
        &mut self,
        _: RequestId,
        _tokens: &[TokenId],
        _features: FeatureRef,
    ) -> Result<MethodEffects, String> {
        Ok(MethodEffects::default())
    }

    fn propose(
        &mut self,
        _: RequestId,
        _context: &[TokenId],
        _limit: usize,
        _: magnitude_model_executor::SelectSpec,
    ) -> Propose {
        Propose::Tokens(Vec::new())
    }

    fn observe(&mut self, _verification: Verification<'_>) -> Result<MethodEffects, String> {
        Ok(MethodEffects::default())
    }

    fn reconcile(
        &mut self,
        _operation: &Operation,
        _outcome: Outcome,
        _: Option<magnitude_model_executor::SelectSpec>,
    ) -> Result<MethodEffects, String> {
        Err("plain generation cannot receive method operations".into())
    }

    fn checkpoint(
        &self,
        _retainer: &mut dyn magnitude_model_executor::FeatureRetainer,
    ) -> Result<MethodCheckpoint, MethodCheckpointError> {
        Ok(MethodCheckpoint::Plain)
    }

    fn evict(&mut self) {}

    fn restore(&mut self) {}

    fn reclaimable(&self) -> u64 {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::method::Method;
    use magnitude_model_executor::{FeatureRetainer, FeatureSpan, RetainedFeatureSpan};

    struct NoFeatures;

    impl FeatureRetainer for NoFeatures {
        fn retain(&mut self, _span: FeatureSpan) -> Result<RetainedFeatureSpan, String> {
            Err("plain method does not retain features".into())
        }
    }

    #[test]
    fn plain_has_no_head_work_or_state() {
        let plain = Plain;
        assert_eq!(plain.identity(), "plain");
        assert_eq!(
            plain.requires(),
            MethodRequirements {
                prefill_demand: Demand::NONE,
                verify_demand: Demand::NONE,
                head: false,
            }
        );
        let mut state = plain.create(None);
        let select = magnitude_model_executor::SelectSpec {
            sampling: magnitude_model_executor::Sampling::Greedy,
            seed: 0,
            position: 0,
            domain: 0,
            mask: None,
            shaping: magnitude_model_executor::Shaping::default(),
            history: None,
        };
        assert_eq!(
            state.propose(RequestId(1), &[TokenId(1)], 4, select),
            Propose::Tokens(vec![])
        );
        assert_eq!(
            state.checkpoint(&mut NoFeatures).unwrap(),
            MethodCheckpoint::Plain
        );
        assert_eq!(state.reclaimable(), 0);
    }
}
