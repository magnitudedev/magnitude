use crate::{
    accept_prefix, verification_selects, Constraint, Demand, MethodRequirements, Sampling,
    SelectSpec, Shaping, TokenId, WorkKind,
};
use magnitude_model_executor::{ConditioningRef, Operation, RequestId};
use std::collections::BTreeSet;

#[derive(Clone, Debug, PartialEq)]
pub struct RoundForward {
    pub kind: WorkKind,
    pub tokens: Vec<TokenId>,
    pub demand: Demand,
    /// One selection row per input row for Decode/Verify.
    pub selects: Vec<SelectSpec>,
    pub committed: usize,
}

impl RoundForward {
    pub fn into_operation(
        self,
        request: RequestId,
        position: usize,
        conditioning: Option<ConditioningRef>,
    ) -> Result<Operation, String> {
        let operation = Operation::Forward {
            request,
            kind: self.kind,
            tokens: self.tokens,
            position,
            conditioning,
            demand: self.demand,
            select: self.selects,
            committed: self.committed,
        };
        operation.validate().map_err(|error| error.to_string())?;
        Ok(operation)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RoundAcceptance {
    pub inputs: Vec<TokenId>,
    pub emitted: Vec<TokenId>,
    pub committed_rows: usize,
    pub proposed: usize,
    pub accepted_proposals: usize,
    pub method_update: MethodUpdate,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MethodUpdate {
    None,
    Prime,
    Observe,
}

#[derive(Clone, Debug, PartialEq)]
enum RoundKind {
    Progress {
        emitted: Vec<TokenId>,
        method_update: MethodUpdate,
    },
    Forced {
        emitted: Vec<TokenId>,
    },
    Verification {
        proposal: Vec<TokenId>,
    },
}

/// A host round suspended across target execution. It owns every immutable
/// control row required to validate the returned per-row samples.
#[derive(Clone, Debug, PartialEq)]
pub struct RoundState {
    forward: RoundForward,
    kind: RoundKind,
}

impl RoundState {
    pub fn progress(
        kind: WorkKind,
        tokens: Vec<TokenId>,
        emitted: Vec<TokenId>,
        select: Option<SelectSpec>,
        requirements: MethodRequirements,
    ) -> Result<Self, String> {
        if tokens.is_empty()
            || !matches!(kind, WorkKind::Prefill | WorkKind::Replay)
            || (kind == WorkKind::Replay && (!emitted.is_empty() || select.is_some()))
            || (!emitted.is_empty() && select.is_some())
            || emitted.len() > 1
        {
            return Err("invalid prefill or replay round".into());
        }
        let demand = if matches!(kind, WorkKind::Prefill | WorkKind::Replay) {
            requirements.prefill_demand
        } else {
            Demand::NONE
        } | if select.is_some() {
            Demand::SELECT
        } else {
            Demand::NONE
        };
        let round = Self {
            forward: RoundForward {
                kind,
                committed: tokens.len(),
                tokens,
                demand,
                selects: select.into_iter().collect(),
            },
            kind: RoundKind::Progress {
                emitted,
                method_update: if matches!(kind, WorkKind::Prefill | WorkKind::Replay) {
                    MethodUpdate::Prime
                } else {
                    MethodUpdate::None
                },
            },
        };
        round
            .forward
            .clone()
            .into_operation(RequestId(0), 0, None)?;
        Ok(round)
    }

    pub fn proposal_limit(
        allowance: usize,
        remaining: usize,
        output_credit: usize,
        method_is_plain: bool,
    ) -> usize {
        if method_is_plain {
            return 0;
        }
        allowance
            .min(remaining)
            .min(output_credit)
            .saturating_sub(1)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn verification(
        anchor: TokenId,
        proposal: Vec<TokenId>,
        proposal_limit: usize,
        generated_position: usize,
        accepted_history: &[TokenId],
        constraint: Option<&dyn Constraint>,
        sampling: Sampling,
        shaping: Shaping,
        seed: u64,
        requirements: MethodRequirements,
    ) -> Result<Self, String> {
        if proposal.len() > proposal_limit {
            return Err("method proposal exceeds the round width limit".into());
        }
        let selects = verification_selects(
            generated_position,
            accepted_history,
            &proposal,
            constraint,
            sampling,
            shaping,
            seed,
        )?;
        let mut tokens = Vec::with_capacity(proposal.len() + 1);
        tokens.push(anchor);
        tokens.extend_from_slice(&proposal);
        Ok(Self {
            forward: RoundForward {
                kind: if proposal.is_empty() {
                    WorkKind::Decode
                } else {
                    WorkKind::Verify
                },
                tokens,
                demand: Demand::SELECT | requirements.verify_demand,
                selects,
                committed: 1,
            },
            kind: RoundKind::Verification { proposal },
        })
    }

    pub fn forced(
        anchor: TokenId,
        forced: Vec<TokenId>,
        requirements: MethodRequirements,
    ) -> Result<Self, String> {
        if forced.is_empty() {
            return Err("forced round requires at least one accepted token".into());
        }
        let mut tokens = Vec::with_capacity(forced.len());
        tokens.push(anchor);
        tokens.extend_from_slice(&forced[..forced.len() - 1]);
        let committed = tokens.len();
        Ok(Self {
            forward: RoundForward {
                // M6.2 defines forced work as a committed Decode forward with
                // no selection.
                kind: WorkKind::Decode,
                tokens,
                demand: requirements.verify_demand & Demand::FEATURES,
                selects: Vec::new(),
                committed,
            },
            kind: RoundKind::Forced { emitted: forced },
        })
    }

    pub fn forward(&self) -> &RoundForward {
        &self.forward
    }

    pub fn reconcile(
        self,
        samples: &[TokenId],
        stops: &BTreeSet<TokenId>,
    ) -> Result<RoundAcceptance, String> {
        match self.kind {
            RoundKind::Progress {
                mut emitted,
                method_update,
            } => {
                let selecting = !self.forward.selects.is_empty();
                if samples.len() != usize::from(selecting) {
                    return Err("progress round returned the wrong number of samples".into());
                }
                if selecting {
                    emitted.push(samples[0]);
                }
                Ok(RoundAcceptance {
                    inputs: self.forward.tokens.clone(),
                    committed_rows: self.forward.tokens.len(),
                    proposed: 0,
                    accepted_proposals: 0,
                    emitted,
                    method_update,
                })
            }
            RoundKind::Forced { emitted } => {
                if !samples.is_empty() {
                    return Err("forced round returned unexpected selections".into());
                }
                Ok(RoundAcceptance {
                    inputs: self.forward.tokens,
                    committed_rows: emitted.len(),
                    proposed: 0,
                    accepted_proposals: 0,
                    emitted,
                    method_update: MethodUpdate::Observe,
                })
            }
            RoundKind::Verification { proposal } => {
                if samples.len() != proposal.len() + 1 {
                    return Err("verification returned the wrong number of samples".into());
                }
                let accepted_proposals = accept_prefix(&proposal, samples, stops)?;
                let bonus = samples[accepted_proposals];
                let mut emitted = proposal[..accepted_proposals].to_vec();
                emitted.push(bonus);
                Ok(RoundAcceptance {
                    inputs: self.forward.tokens,
                    committed_rows: accepted_proposals + 1,
                    proposed: proposal.len(),
                    accepted_proposals,
                    emitted,
                    method_update: MethodUpdate::Observe,
                })
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn requirements() -> MethodRequirements {
        MethodRequirements {
            prefill_demand: Demand::NONE,
            verify_demand: Demand::NONE,
            head: false,
        }
    }

    fn round(proposal: &[u32]) -> RoundState {
        RoundState::verification(
            TokenId(7),
            proposal.iter().copied().map(TokenId).collect(),
            proposal.len(),
            10,
            &[TokenId(1), TokenId(2)],
            None,
            Sampling::Greedy,
            Shaping {
                temperature: 0.0,
                ..Default::default()
            },
            4,
            requirements(),
        )
        .unwrap()
    }

    #[test]
    fn plain_and_each_draft_acceptance_count_have_stable_transcripts() {
        let stops = BTreeSet::new();
        let plain = round(&[]);
        assert_eq!(plain.forward().kind, WorkKind::Decode);
        assert_eq!(plain.forward().tokens, [TokenId(7)]);
        assert_eq!(plain.forward().selects.len(), 1);
        assert_eq!(
            plain.reconcile(&[TokenId(8)], &stops).unwrap().emitted,
            [TokenId(8)]
        );

        for (samples, accepted, emitted) in [
            (vec![8, 9, 10, 11], 0, vec![8]),
            (vec![3, 9, 10, 11], 1, vec![3, 9]),
            (vec![3, 4, 10, 11], 2, vec![3, 4, 10]),
            (vec![3, 4, 5, 11], 3, vec![3, 4, 5, 11]),
        ] {
            let result = round(&[3, 4, 5])
                .reconcile(
                    &samples.into_iter().map(TokenId).collect::<Vec<_>>(),
                    &stops,
                )
                .unwrap();
            assert_eq!(result.accepted_proposals, accepted);
            assert_eq!(
                result.emitted,
                emitted.into_iter().map(TokenId).collect::<Vec<_>>()
            );
            assert_eq!(result.committed_rows, accepted + 1);
        }
    }

    #[test]
    fn stop_bonus_forced_and_credit_limit_follow_round_rules() {
        let stops = BTreeSet::from([TokenId(99)]);
        let accepted = round(&[3, 4])
            .reconcile(&[TokenId(3), TokenId(4), TokenId(99)], &stops)
            .unwrap();
        assert_eq!(accepted.emitted, [TokenId(3), TokenId(4), TokenId(99)]);
        assert_eq!(RoundState::proposal_limit(8, 8, 2, false), 1);
        assert_eq!(RoundState::proposal_limit(8, 8, 2, true), 0);

        let forced =
            RoundState::forced(TokenId(7), vec![TokenId(8), TokenId(9)], requirements()).unwrap();
        assert_eq!(forced.forward().kind, WorkKind::Decode);
        assert_eq!(forced.forward().tokens, [TokenId(7), TokenId(8)]);
        assert!(forced.forward().selects.is_empty());
        assert_eq!(
            forced.reconcile(&[], &stops).unwrap().emitted,
            [TokenId(8), TokenId(9)]
        );
    }

    #[test]
    fn forward_lowers_without_losing_per_row_selection_controls() {
        let conditioning = ConditioningRef::logical(
            magnitude_model_executor::ResourceDomainId::new("test").unwrap(),
            1,
            1,
            vec![magnitude_model_executor::ConditioningRange { destination: 0..1 }],
        )
        .unwrap();
        let operation = round(&[3, 4])
            .forward
            .into_operation(RequestId(9), 17, Some(conditioning.clone()))
            .unwrap();
        let Operation::Forward {
            request,
            kind,
            tokens,
            position,
            demand,
            select,
            committed,
            conditioning,
            ..
        } = operation
        else {
            unreachable!()
        };
        assert_eq!(request, RequestId(9));
        assert_eq!(kind, WorkKind::Verify);
        assert_eq!(tokens, [TokenId(7), TokenId(3), TokenId(4)]);
        assert_eq!(position, 17);
        assert_eq!(conditioning, conditioning);
        assert!(demand.contains(Demand::SELECT));
        assert_eq!(select.len(), 3);
        assert_eq!(
            select.iter().map(|row| row.position).collect::<Vec<_>>(),
            [10, 11, 12]
        );
        assert_eq!(committed, 1);
    }

    #[test]
    fn replay_reprimes_feature_dependent_methods_without_sampling() {
        let requirements = MethodRequirements {
            prefill_demand: Demand::FEATURES,
            verify_demand: Demand::FEATURES,
            head: true,
        };
        let replay = RoundState::progress(
            WorkKind::Replay,
            vec![TokenId(1), TokenId(2)],
            vec![],
            None,
            requirements,
        )
        .unwrap();
        assert_eq!(replay.forward().demand, Demand::FEATURES);
        assert!(replay.forward().selects.is_empty());
        assert_eq!(
            replay
                .reconcile(&[], &BTreeSet::new())
                .unwrap()
                .method_update,
            MethodUpdate::Prime
        );
    }
}
