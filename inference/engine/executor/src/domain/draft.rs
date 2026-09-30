//! The rows of a separate draft's (DFlash, DSpark) transactions: entry rows
//! inject the target's conditioning into every history domain of the draft
//! store, then one block `[anchor, mask, …]` drafts the proposals.

use super::*;
use crate::batching::BlockSlot;
use crate::DraftForm;

impl<F: ProgramFamily> ExecutorDomain<F> {
    /// The draft rows of one transaction. Entry row `i` sits at head
    /// position `p = position + i` and injects target row `p`'s feature at
    /// its destination in each domain, attending nothing else. The block's
    /// rows sit at `n = position + entry rows` onward: the anchor (the entry's
    /// last token, the one after the last committed row), then mask tokens.
    /// Every block row reads each domain's accepted rows and injected entry
    /// rows from its window start, and the block through itself (the program
    /// widens the fresh span to the whole block for a bidirectional layer).
    /// Proposal `k < steps` selects on its layout's row;
    /// a slot's surplus proposals repeat its last selection and are
    /// discarded.
    pub(super) fn draft_slot(
        &self,
        operation: &Operation,
        advance: &TentativeAdvance,
        steps: usize,
    ) -> Result<BlockSlot, String> {
        let Operation::Head {
            tokens,
            position,
            proposals,
            form: DraftForm::Block,
            ..
        } = operation
        else {
            return Err("a draft slot needs a block-form head operation".into());
        };
        self.block_slot(tokens, *position, proposals, advance, steps)
    }

    /// A block drafter's slot: entry rows `tokens` at `position`, then with
    /// `steps` one drafting block selecting `proposals`.
    pub(super) fn block_slot(
        &self,
        tokens: &[TokenId],
        position: usize,
        proposals: &[SelectSpec],
        advance: &TentativeAdvance,
        steps: usize,
    ) -> Result<BlockSlot, String> {
        let draft = self
            .definition
            .draft
            .as_ref()
            .ok_or("a block-form head operation without a draft")?;
        let store = self
            .head_store
            .as_ref()
            .ok_or("a draft operation without a draft store")?;
        let binding = advance.bindings();
        let i32_of = |value: usize, what: &str| {
            i32::try_from(value).map_err(|_| format!("draft {what} exceeds i32"))
        };
        // The draft rotates by plain sequence positions.
        let coordinates = |position: usize| -> Result<[i32; 4], String> {
            let position = i32_of(position, "position")?;
            Ok([position, position, position, 0])
        };
        let span = |start: usize, count: usize| -> Result<[i32; 2], String> {
            Ok([
                i32_of(start, "history start")?,
                i32_of(
                    start
                        .checked_add(count)
                        .ok_or("draft history end overflow")?,
                    "history end",
                )?,
            ])
        };
        let domains = store.history_domains().collect::<Vec<_>>();
        let mut entry = Vec::with_capacity(tokens.len());
        for (index, token) in tokens.iter().enumerate() {
            let histories = domains
                .iter()
                .map(|domain| {
                    Ok(RowHistory {
                        visible: Vec::new(),
                        fresh_start: i32_of(index, "entry row")?,
                        bidirectional_end: None,
                        destination: binding.destinations[domain.0]
                            .get(index)
                            .map_or(Ok(-1), |row| i32_of(*row, "destination"))?,
                    })
                })
                .collect::<Result<Vec<_>, String>>()?;
            entry.push(Row {
                token: i32::try_from(token.0).map_err(|_| "draft token exceeds i32")?,
                coordinates: coordinates(position + index)?,
                histories,
                demand: crate::batching::Demand::NONE,
                select: None,
            });
        }
        let mut block = Vec::new();
        if steps > 0 {
            let anchor = tokens
                .last()
                .ok_or("a drafting transaction has no entry rows")?;
            let start = position
                .checked_add(tokens.len())
                .ok_or("draft block position overflows")?;
            let rows = usize::try_from(draft.block_rows(steps as u64))
                .map_err(|_| "draft block exceeds host")?;
            let proposing = (0..steps)
                .map(|proposal| {
                    usize::try_from(draft.proposal_row(proposal as u64))
                        .map_err(|_| "draft proposal row exceeds host".to_owned())
                })
                .collect::<Result<Vec<_>, String>>()?;
            let last = proposals.last();
            for row in 0..rows {
                let row_position = start + row;
                let histories = domains
                    .iter()
                    .map(|&domain| {
                        let from = store.history_domain_kind(domain).visible_from(row_position);
                        let mut visible = advance
                            .visible_ranges(domain, from)
                            .into_iter()
                            .map(|(start, count)| span(start, count))
                            .collect::<Result<Vec<_>, String>>()?;
                        for (index, destination) in
                            binding.destinations[domain.0].iter().enumerate()
                        {
                            if position + index >= from {
                                visible.push(span(*destination, 1)?);
                            }
                        }
                        let slab_rows = binding
                            .history
                            .iter()
                            .find(|plane| plane.domain == domain)
                            .ok_or("a draft history domain has no plane")?
                            .slab_rows;
                        Ok(RowHistory {
                            visible: coalesce_within_slabs(&visible, slab_rows),
                            fresh_start: 0,
                            bidirectional_end: None,
                            destination: -1,
                        })
                    })
                    .collect::<Result<Vec<_>, String>>()?;
                let select = match proposing.iter().position(|proposing| *proposing == row) {
                    Some(proposal) => Some(super::target::select_row(
                        proposals
                            .get(proposal)
                            .or(last)
                            .ok_or("a causal draft transaction in a drafting batch")?,
                    )),
                    None => None,
                };
                block.push(Row {
                    token: if row == 0 {
                        i32::try_from(anchor.0).map_err(|_| "draft token exceeds i32")?
                    } else {
                        i32::try_from(draft.mask_token.0).map_err(|_| "mask token exceeds i32")?
                    },
                    coordinates: coordinates(row_position)?,
                    histories,
                    demand: crate::batching::Demand::NONE,
                    select,
                });
            }
        }
        Ok(BlockSlot {
            entry: Slot {
                rows: entry,
                bank: i32_of(binding.previous_bank, "bank")?,
                previous_tape: i32_of(binding.previous_tape, "tape rows")?,
                following_bank: i32_of(binding.following_bank, "successor bank")?,
                stop: i32_of(binding.stop, "committed rows")?,
            },
            block,
        })
    }
}

/// `spans` with each span merged into its predecessor when they are
/// adjacent and the merged span stays within one `slab_rows`-row slab.
pub(super) fn coalesce_within_slabs(spans: &[[i32; 2]], slab_rows: u32) -> Vec<[i32; 2]> {
    let slab = |row: i32| row / slab_rows as i32;
    let mut merged: Vec<[i32; 2]> = Vec::with_capacity(spans.len());
    for &span in spans {
        match merged.last_mut() {
            Some(previous) if previous[1] == span[0] && slab(previous[0]) == slab(span[1] - 1) => {
                previous[1] = span[1]
            }
            _ => merged.push(span),
        }
    }
    merged
}

#[cfg(test)]
mod tests {
    use super::coalesce_within_slabs;

    #[test]
    fn adjacent_spans_merge_only_within_a_slab() {
        let spans = [[0, 3], [3, 4], [4, 5], [5, 6], [9, 10]];
        assert_eq!(coalesce_within_slabs(&spans, 4), [[0, 4], [4, 6], [9, 10]]);
    }
}
