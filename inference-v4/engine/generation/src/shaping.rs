use crate::{Constraint, Sampling, SelectSpec, Shaping, TokenId};
use std::sync::Arc;

pub const HISTORY_WIDTH: usize = 64;

pub fn selection_history(
    accepted: &[TokenId],
    proposal_prefix: &[TokenId],
) -> Result<Arc<[i32]>, String> {
    let mut packed = [-1; HISTORY_WIDTH];
    let keep_proposal = proposal_prefix.len().min(HISTORY_WIDTH);
    let keep_accepted = accepted
        .len()
        .min(HISTORY_WIDTH.saturating_sub(keep_proposal));
    let source = accepted[accepted.len() - keep_accepted..]
        .iter()
        .chain(&proposal_prefix[proposal_prefix.len() - keep_proposal..]);
    let start = HISTORY_WIDTH - keep_accepted - keep_proposal;
    for (destination, token) in packed[start..].iter_mut().zip(source) {
        *destination = i32::try_from(token.0)
            .map_err(|_| "selection history token exceeds the packed i32 domain")?;
    }
    Ok(Arc::from(packed))
}

pub fn verification_selects(
    generated_position: usize,
    accepted: &[TokenId],
    proposal: &[TokenId],
    constraint: Option<&dyn Constraint>,
    sampling: Sampling,
    shaping: Shaping,
    seed: u64,
) -> Result<Vec<SelectSpec>, String> {
    let mut preview = constraint.map(Constraint::fork);
    let mut poisoned = false;
    let mut selects = Vec::with_capacity(proposal.len() + 1);
    for row in 0..=proposal.len() {
        let mask = preview
            .as_ref()
            .map(|constraint| constraint.mask())
            .transpose()?;
        selects.push(SelectSpec {
            sampling,
            seed,
            position: generated_position
                .checked_add(row)
                .ok_or("selection position exhausted")?,
            domain: 0,
            mask,
            shaping,
            history: shaping
                .uses_history()
                .then(|| selection_history(accepted, &proposal[..row]))
                .transpose()?,
        });
        if row < proposal.len() && !poisoned {
            if let Some(current) = preview.as_ref() {
                match current.stage(&proposal[row..row + 1]) {
                    Ok(next) if next.position() == current.position() + 1 => preview = Some(next),
                    Ok(_) => {
                        return Err("constraint preview advanced to an invalid position".into());
                    }
                    Err(_) => poisoned = true,
                }
            }
        }
    }
    Ok(selects)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone)]
    struct RejectThree {
        position: usize,
    }

    impl Constraint for RejectThree {
        fn position(&self) -> usize {
            self.position
        }
        fn fork(&self) -> Box<dyn Constraint> {
            Box::new(self.clone())
        }
        fn stage(&self, tokens: &[TokenId]) -> Result<Box<dyn Constraint>, String> {
            if tokens.contains(&TokenId(3)) {
                Err("rejected".into())
            } else {
                Ok(Box::new(Self {
                    position: self.position + tokens.len(),
                }))
            }
        }
        fn forced(&self, _limit: usize) -> Result<Vec<TokenId>, String> {
            Ok(vec![])
        }
        fn mask(&self) -> Result<Arc<[u32]>, String> {
            Ok(Arc::from([self.position as u32]))
        }
    }

    #[test]
    fn histories_are_width_64_and_include_each_proposal_prefix() {
        let history = selection_history(
            &(0..70).map(TokenId).collect::<Vec<_>>(),
            &[TokenId(70), TokenId(71)],
        )
        .unwrap();
        assert_eq!(history.len(), 64);
        assert_eq!(history[0], 8);
        assert_eq!(history[62..], [70, 71]);
    }

    #[test]
    fn verification_previews_positions_and_poisoned_masks() {
        let shaping = Shaping {
            temperature: 0.0,
            repetition_penalty: 1.1,
            ..Default::default()
        };
        let selects = verification_selects(
            5,
            &[TokenId(1)],
            &[TokenId(2), TokenId(3), TokenId(4)],
            Some(&RejectThree { position: 1 }),
            Sampling::Greedy,
            shaping,
            7,
        )
        .unwrap();
        assert_eq!(selects.len(), 4);
        assert_eq!(
            selects
                .iter()
                .map(|select| select.position)
                .collect::<Vec<_>>(),
            [5, 6, 7, 8]
        );
        assert_eq!(selects[0].mask.as_deref(), Some([1].as_slice()));
        assert_eq!(selects[1].mask.as_deref(), Some([2].as_slice()));
        assert_eq!(selects[2].mask.as_deref(), Some([2].as_slice()));
        assert_eq!(selects[3].mask.as_deref(), Some([2].as_slice()));
        assert_eq!(selects[3].history.as_ref().unwrap()[61..], [2, 3, 4]);
    }
}
