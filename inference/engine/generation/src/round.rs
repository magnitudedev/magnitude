use crate::{
    accept_prefix, verification_selects, Constraint, Demand, MethodRequirements, Sampling,
    SelectSpec, Shaping, TokenId, WorkKind,
};
use magnitude_executor::{AheadVerdict, ChainEntry, ConditioningRef, FrontRow, Guess, Operation, PhysicalDecision, Priming, RequestId, RowResult, Rows, Selected, TwoRowDecode};
use std::collections::BTreeSet;

#[derive(Clone, Debug, PartialEq)]
pub struct RoundForward {
    pub kind: WorkKind,
    pub tokens: Vec<TokenId>,
    pub demand: Demand,
    /// One selection row per input row a decision may accept, for
    /// Decode/Verify.
    pub selects: Vec<SelectSpec>,
    pub committed: usize,
    /// A prompt chunk's drafter entry, drafted behind the chunk.
    pub prime: Option<Priming>,
    /// Its front rows, its last token rows by depth (`Operation::Forward`'s
    /// `front`).
    pub front: Vec<FrontRow>,
    /// Its entry into the MTP head behind it, of a sequence decoding in
    /// front passes (`Operation::Forward`'s `chain`).
    pub chain: Option<ChainEntry>,
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
            prime: self.prime,
            front: self.front,
            chain: self.chain,
        };
        operation.validate().map_err(|error| error.to_string())?;
        Ok(operation)
    }

    /// The samples its round is reconciled by ([`RoundState::reconcile`]),
    /// from its outcome's `rows` (the rows a decision may accept, never a
    /// front row's): each selected row's token, in row order. The one
    /// reading of a forward's selections, the scheduler's and the
    /// evaluation harness's alike.
    pub fn samples(&self, rows: &[RowResult]) -> Result<Vec<TokenId>, String> {
        if rows.len() != self.tokens.len() - self.front.len() {
            return Err("target round returned the wrong number of rows".into());
        }
        let selected = rows.iter().filter_map(|row| row.selected).collect::<Vec<_>>();
        if selected.len() != self.selects.len() {
            return Err(match self.selects.is_empty() {
                true => "non-selecting target round returned an unexpected sample".into(),
                false => "target round selection count differs from its rows".into(),
            });
        }
        selected.into_iter().map(selected_token).collect()
    }
}

/// A row's selected token, or why its sampling selected none.
fn selected_token(selected: Selected) -> Result<TokenId, String> {
    match selected {
        Selected::Token(token) => Ok(token),
        Selected::Empty => Err("sampling distribution is empty".into()),
        Selected::NonFinite => Err("sampling distribution is nonfinite".into()),
        Selected::Declined => Err("target sampling cannot decline".into()),
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
    /// How the pass's guessed rows fared, for the executor's reads ahead.
    /// Empty of any round that is not a pass.
    pub verdict: AheadVerdict,
}

impl RoundAcceptance {
    /// The decision the executor reconciles the round's outcome by: its
    /// committed rows and its verdict, moved. The one constructor of a
    /// round's decision, generation's and the evaluation harness's alike.
    pub fn decision(self) -> PhysicalDecision {
        PhysicalDecision { accepted_rows: self.committed_rows, ahead: self.verdict }
    }

    /// Leave the chain its outcome guessed (`guesses`, `Outcome::Forward`'s:
    /// empty of a round that made none) to the sequence's front-row decode
    /// (`two_row`), the next pass's front tokens: the one advance of a
    /// front-row decode, generation's and the evaluation harness's alike. A
    /// sequence not decoding in front passes (none) runs no front row,
    /// whatever its rounds guess.
    pub fn advance(&self, two_row: Option<&mut TwoRowDecode>, guesses: Vec<Guess>) {
        if let Some(decode) = two_row {
            decode.guessed(guesses);
        }
    }
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
    /// A decode pass: its exact row, then its rows ahead, each checked
    /// against the selection of the row before it.
    Pass {
        rows: Rows,
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
    /// A prompt or replay chunk of `tokens` at `position`; the chunk closing
    /// the prompt selects or forces its successor, and of a sequence decoding
    /// in front passes enters the head behind its drafter entry (`chain`).
    #[allow(clippy::too_many_arguments)]
    pub fn progress(
        kind: WorkKind,
        tokens: Vec<TokenId>,
        position: usize,
        emitted: Vec<TokenId>,
        select: Option<SelectSpec>,
        prime: Option<Priming>,
        chain: Option<ChainEntry>,
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
        // Only the chunk closing the prompt selects or forces a token.
        let closing = select.is_some() || !emitted.is_empty();
        let demand = requirements.prefill_demand
            | if closing {
                requirements.closing_demand
            } else {
                Demand::NONE
            }
            | if select.is_some() {
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
                prime,
                front: Vec::new(),
                chain,
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
            .into_operation(RequestId(0), position, None)?;
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

    /// A decode pass of `rows` behind the accepted history, whose pending
    /// token is `pending` ([`Rows::tokens`]): each row selects the token
    /// after it, with the rows ahead before it as history, publishing its
    /// features where `demand` demands them. A plain decode, a verification
    /// of proposals or a front pass; of a sequence decoding in front passes,
    /// entering the head behind it (`chain`).
    #[allow(clippy::too_many_arguments)]
    pub fn pass(
        pending: TokenId,
        rows: Rows,
        generated_position: usize,
        accepted_history: &[TokenId],
        constraint: Option<&dyn Constraint>,
        sampling: Sampling,
        shaping: Shaping,
        seed: u64,
        demand: Demand,
        chain: Option<ChainEntry>,
    ) -> Result<Self, String> {
        let tokens = rows.tokens(pending);
        let mut selects = verification_selects(
            generated_position,
            accepted_history,
            &tokens[1..],
            constraint,
            sampling,
            shaping,
            seed,
        )?;
        let front = selects
            .split_off(1 + rows.acceptable().len())
            .into_iter()
            .zip(rows.front_kinds())
            .map(|(select, kind)| FrontRow { select, kind })
            .collect::<Vec<_>>();
        assert_eq!(front.len(), rows.front_rows(), "a select for each front row");
        Ok(Self {
            forward: RoundForward {
                kind: match rows.acceptable().is_empty() {
                    true => WorkKind::Decode,
                    false => WorkKind::Verify,
                },
                tokens,
                demand: Demand::SELECT | demand,
                selects,
                committed: 1,
                prime: None,
                front,
                chain,
            },
            kind: RoundKind::Pass { rows },
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
                prime: None,
                front: Vec::new(),
                chain: None,
            },
            kind: RoundKind::Forced { emitted: forced },
        })
    }

    pub fn forward(&self) -> &RoundForward {
        &self.forward
    }

    /// Maximum tokens this round can accept after its target work completes.
    /// Selection rows alone do not account for constrained forced tokens or
    /// tokens emitted while a prefill round advances.
    pub fn publication_bound(&self) -> usize {
        match &self.kind {
            RoundKind::Progress { emitted, .. } | RoundKind::Forced { emitted } => {
                emitted.len() + self.forward.selects.len()
            }
            RoundKind::Pass { .. } => self.forward.selects.len(),
        }
    }

    /// The round's acceptance of `samples`, one per row it selects in row
    /// order: the rows a decision may accept, then its front rows. A pass's
    /// rows ahead are guesses checked against the selection of the row
    /// before each: its acceptable rows commit as far as acceptance takes
    /// them (`accept_prefix`), the selection of the last row committed is
    /// emitted after them, and a front row's guess stood or not.
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
                    verdict: AheadVerdict::default(),
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
                    verdict: AheadVerdict::default(),
                })
            }
            RoundKind::Pass { rows } => {
                // Every row selects but a front row, whose selection is spent
                // on the next guess (`guess_chain`).
                if samples.len() != rows.row_count() - rows.front_rows() {
                    return Err("a pass returned the wrong number of samples".into());
                }
                let acceptable = rows.acceptable();
                // Each guess, a row ahead's token, is checked against the
                // selection of the row before it.
                let verdict = AheadVerdict::new(&rows, samples);
                let accepted = accept_prefix(acceptable, samples, stops)?;
                let mut emitted = acceptable[..accepted].to_vec();
                emitted.push(samples[accepted]);
                Ok(RoundAcceptance {
                    inputs: self.forward.tokens[..1 + acceptable.len()].to_vec(),
                    committed_rows: 1 + accepted,
                    proposed: acceptable.len(),
                    accepted_proposals: accepted,
                    emitted,
                    method_update: MethodUpdate::Observe,
                    verdict,
                })
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use magnitude_executor::{FrontGrant, FrontKind, GuessKind};

    fn requirements() -> MethodRequirements {
        MethodRequirements {
            prefill_demand: Demand::NONE,
            closing_demand: Demand::NONE,
            verify_demand: Demand::NONE,
            head: false,
        }
    }

    fn tokens(tokens: &[u32]) -> Vec<TokenId> {
        tokens.iter().copied().map(TokenId).collect()
    }

    /// A pass of `rows` behind the history [1, 2], its pending token 7.
    fn pass(rows: Rows) -> RoundState {
        RoundState::pass(
            TokenId(7),
            rows,
            10,
            &[TokenId(1), TokenId(2)],
            None,
            Sampling::Greedy,
            Shaping {
                temperature: 0.0,
                ..Default::default()
            },
            4,
            Demand::NONE,
            None,
        )
        .unwrap()
    }

    /// A decode, or verification of `proposal`, behind its anchor 7.
    fn round(proposal: &[u32]) -> RoundState {
        pass(Rows::Exact { acceptable: tokens(proposal) })
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
        assert_eq!(forced.publication_bound(), 2);
        assert_eq!(
            forced.reconcile(&[], &stops).unwrap().emitted,
            [TokenId(8), TokenId(9)]
        );
    }

    /// A two-row pass is a verification of one proposal that is never
    /// accepted: its exact row's token is emitted, its front row's
    /// selection is no sample (it is spent on the next guess), and the
    /// guess its outcome made is the next pass's, whether its own stood or
    /// not. Its front row carries its guess's kind. Each verdict names its
    /// guesses and their kind.
    #[test]
    fn a_two_row_pass_checks_its_guess_as_one_proposal() {
        let stops = BTreeSet::from([TokenId(99)]);
        let guess = Guess { token: TokenId(8), kind: GuessKind::Head };
        let paired = pass(Rows::front(vec![guess], 0));
        assert_eq!(paired.forward().kind, WorkKind::Decode);
        assert_eq!(paired.forward().tokens, tokens(&[7, 8]));
        assert_eq!((paired.forward().committed, paired.forward().selects.len()), (1, 1));
        assert_eq!(paired.forward().front.iter().map(|front| front.kind).collect::<Vec<_>>(), [FrontKind::Host(GuessKind::Head)]);
        assert_eq!(paired.publication_bound(), 1);
        // Its samples are its exact row's selection alone.
        let selected = |token| Selected::Token(TokenId(token));
        let exact = [RowResult { selected: Some(selected(8)), ..Default::default() }];
        assert_eq!(paired.forward().samples(&exact).unwrap(), tokens(&[8]));
        assert!(paired.forward().samples(&[]).is_err(), "its exact row's outcome");
        let held = paired.clone().reconcile(&tokens(&[8]), &stops).unwrap();
        assert_eq!((held.emitted.clone(), held.committed_rows, held.inputs.clone()), (tokens(&[8]), 1, tokens(&[7])));
        assert_eq!((held.proposed, held.accepted_proposals, held.method_update), (0, 0, MethodUpdate::Observe));
        assert_eq!((held.verdict.stood(), held.verdict.acceptable()), (1, 0));
        // The guess its outcome made is the next pass's, of a sequence
        // decoding in two-row passes alone; none leaves the next pass one
        // row.
        let mut decode = TwoRowDecode::default();
        let next = Guess { token: TokenId(9), kind: GuessKind::Chained };
        held.advance(Some(&mut decode), vec![next]);
        assert_eq!(decode.pass(FrontGrant::UpTo(1)), Rows::front(vec![next], 0));
        held.advance(Some(&mut decode), Vec::new());
        assert_eq!(decode.pass(FrontGrant::UpTo(1)), Rows::Exact { acceptable: Vec::new() });
        assert_eq!(held.verdict.checked(), [(TokenId(8), TokenId(8))]);
        assert_eq!((held.verdict.front(), held.verdict.truths()), (&tokens(&[8])[..], &tokens(&[8])[..]));
        assert_eq!(held.decision(), PhysicalDecision { accepted_rows: 1, ahead: AheadVerdict::new(&Rows::front(vec![guess], 0), &tokens(&[8])) });
        // A wrong guess emits the exact token all the same.
        let wrong = paired.clone().reconcile(&tokens(&[5]), &stops).unwrap();
        assert_eq!((wrong.emitted.clone(), wrong.committed_rows, wrong.verdict.stood()), (tokens(&[5]), 1, 0));
        // A stop token the front row guessed stood, though acceptance
        // accepts none.
        let stopped = pass(Rows::front(vec![Guess { token: TokenId(99), ..guess }], 0)).reconcile(&tokens(&[99]), &stops).unwrap();
        assert_eq!((stopped.emitted, stopped.verdict.stood()), (tokens(&[99]), 1));
        assert!(paired.reconcile(&tokens(&[8, 9]), &stops).is_err(), "no front row's sample");
    }

    /// A front pass of three rows runs each on its guess, by depth, and
    /// emits its exact row's token alone: its verdict holds every front
    /// token and the one truth its round learned, which judges only its
    /// first (a later round learns the rest).
    #[test]
    fn a_front_pass_runs_each_guess_by_depth_and_learns_one_truth() {
        let stops = BTreeSet::from([TokenId(99)]);
        let chain = [8, 9, 10].map(|token| Guess { token: TokenId(token), kind: GuessKind::Chained });
        let deep = pass(Rows::front(chain.to_vec(), 0));
        assert_eq!(deep.forward().tokens, tokens(&[7, 8, 9, 10]));
        assert_eq!((deep.forward().committed, deep.forward().selects.len(), deep.forward().front.len()), (1, 1, 3));
        let held = deep.reconcile(&tokens(&[8]), &stops).unwrap();
        assert_eq!((held.emitted.clone(), held.committed_rows), (tokens(&[8]), 1));
        assert_eq!(held.verdict.front(), &tokens(&[8, 9, 10])[..]);
        assert_eq!((held.verdict.truths(), held.verdict.checked()), (&tokens(&[8])[..], vec![(TokenId(8), TokenId(8))]));
    }

    #[test]
    fn publication_bound_includes_prefill_emission_without_selection() {
        let progress = RoundState::progress(
            WorkKind::Prefill,
            vec![TokenId(1)],
            0,
            vec![TokenId(2)],
            None,
            None,
            None,
            requirements(),
        )
        .unwrap();
        assert!(progress.forward().selects.is_empty());
        assert_eq!(progress.publication_bound(), 1);
        assert_eq!(round(&[3, 4]).publication_bound(), 3);
    }

    #[test]
    fn forward_lowers_without_losing_per_row_selection_controls() {
        let conditioning = ConditioningRef::logical(
            magnitude_executor::ResourceDomainId::new("test").unwrap(),
            1,
            1,
            vec![magnitude_executor::ConditioningRange { destination: 0..1 }],
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
            closing_demand: Demand::NONE,
            verify_demand: Demand::FEATURES,
            head: true,
        };
        let replay = RoundState::progress(
            WorkKind::Replay,
            vec![TokenId(1), TokenId(2)],
            0,
            vec![],
            None,
            None,
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

    #[test]
    fn closing_demand_applies_only_to_the_chunk_closing_the_prompt() {
        let requirements = MethodRequirements {
            prefill_demand: Demand::NONE,
            closing_demand: Demand::FEATURES,
            verify_demand: Demand::FEATURES,
            head: true,
        };
        let chunk = |emitted: Vec<TokenId>, select: Option<SelectSpec>, kind| {
            RoundState::progress(kind, vec![TokenId(1), TokenId(2)], 0, emitted, select, None, None, requirements)
                .unwrap()
                .forward()
                .demand
        };
        let select = SelectSpec {
            sampling: Sampling::Greedy,
            seed: 0,
            position: 0,
            domain: 0,
            mask: None,
            shaping: Shaping::default(),
            history: None,
        };
        assert_eq!(chunk(vec![], None, WorkKind::Prefill), Demand::NONE);
        assert_eq!(chunk(vec![], None, WorkKind::Replay), Demand::NONE);
        assert_eq!(
            chunk(vec![], Some(select), WorkKind::Prefill),
            Demand::FEATURES | Demand::SELECT
        );
        assert_eq!(chunk(vec![TokenId(3)], None, WorkKind::Prefill), Demand::FEATURES);
    }
}
