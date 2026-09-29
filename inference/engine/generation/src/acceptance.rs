use crate::TokenId;
use std::collections::BTreeSet;

/// Number of leading draft tokens confirmed by the target samples. A stop
/// token is never accepted as a draft match; it may only be emitted as the
/// target-sampled bonus token.
pub fn accept_prefix(
    proposal: &[TokenId],
    samples: &[TokenId],
    stops: &BTreeSet<TokenId>,
) -> Result<usize, String> {
    if samples.len() < proposal.len() {
        return Err("verification samples do not cover every proposal token".into());
    }
    Ok(proposal
        .iter()
        .zip(samples)
        .take_while(|(proposed, sampled)| proposed == sampled && !stops.contains(proposed))
        .count())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn target_sampled_prefix_table() {
        let stops = BTreeSet::from([TokenId(9)]);
        for (proposal, samples, expected) in [
            (vec![1, 2, 3], vec![1, 2, 3, 4], 3),
            (vec![1, 2, 3], vec![1, 8, 3, 4], 1),
            (vec![1, 2, 3], vec![8, 2, 3, 4], 0),
            (vec![1, 9, 3], vec![1, 9, 3, 4], 1),
        ] {
            assert_eq!(
                accept_prefix(
                    &proposal.into_iter().map(TokenId).collect::<Vec<_>>(),
                    &samples.into_iter().map(TokenId).collect::<Vec<_>>(),
                    &stops,
                )
                .unwrap(),
                expected
            );
        }
        assert!(accept_prefix(&[TokenId(1)], &[], &stops).is_err());
    }
}
