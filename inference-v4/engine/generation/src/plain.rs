use crate::method::{
    Method, MethodCheckpoint, MethodCheckpointError, MethodEffects, MethodRequirements,
    MethodState, Propose, Verification,
};
use magnitude_model_executor::{
    Demand, FeatureReader, FeatureRef, Operation, Outcome, RequestId, SelectSpec, TokenId,
};

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

    fn proposals(&self) -> usize {
        0
    }

    fn create(
        &self,
        checkpoint: Option<&MethodCheckpoint>,
    ) -> Result<Box<dyn MethodState>, String> {
        match checkpoint {
            None | Some(MethodCheckpoint::Plain) => Ok(Box::new(PlainState)),
            Some(MethodCheckpoint::Mtp(_)) => {
                Err("an MTP checkpoint cannot restore plain state".into())
            }
        }
    }
}

#[derive(Clone)]
struct PlainState;

impl MethodState for PlainState {
    fn fork_transition(&self) -> Box<dyn MethodState> {
        Box::new(self.clone())
    }

    fn prime(
        &mut self,
        _: RequestId,
        _: &[TokenId],
        _: Option<TokenId>,
        _: FeatureRef,
        _: &mut dyn FeatureReader,
    ) -> Result<MethodEffects, String> {
        Ok(MethodEffects::default())
    }

    fn propose(&mut self, _: RequestId, _: &[SelectSpec]) -> Propose {
        Propose::Tokens(Vec::new())
    }

    fn observe(
        &mut self,
        _: RequestId,
        _: Verification<'_>,
        _: &mut dyn FeatureReader,
    ) -> Result<MethodEffects, String> {
        Ok(MethodEffects::default())
    }

    fn reconcile(&mut self, _: &Operation, _: Outcome) -> Result<(), String> {
        Err("plain generation cannot receive method operations".into())
    }

    fn checkpoint(&self) -> Result<MethodCheckpoint, MethodCheckpointError> {
        Ok(MethodCheckpoint::Plain)
    }

    fn evict(&mut self) {}

    fn reclaimable(&self) -> u64 {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plain_has_no_head_work_or_state() {
        let plain = Plain;
        assert_eq!(plain.identity(), "plain");
        assert_eq!(plain.proposals(), 0);
        assert_eq!(
            plain.requires(),
            MethodRequirements {
                prefill_demand: Demand::NONE,
                verify_demand: Demand::NONE,
                head: false,
            }
        );
        let mut state = plain.create(None).unwrap();
        let select = SelectSpec {
            sampling: magnitude_model_executor::Sampling::Greedy,
            seed: 0,
            position: 0,
            domain: 0,
            mask: None,
            shaping: magnitude_model_executor::Shaping::default(),
            history: None,
        };
        assert_eq!(
            state.propose(RequestId(1), &[select]),
            Propose::Tokens(vec![])
        );
        assert_eq!(state.checkpoint().unwrap(), MethodCheckpoint::Plain);
        assert_eq!(state.reclaimable(), 0);
    }
}
