//! Pack row-independent model stages while stateful mixers retain per-request
//! views. Packing shares the decoder's prepared embedding composition; the
//! only per-total state is buffer extent, and a packed batch selects a
//! prepared capacity class without compiling.
use super::*;
use crate::inputs::TokenId;

pub(super) struct PackedBuffers {
    pub(super) hidden: Buffer,
    pub(super) tokens: Buffer,
}
struct Member {
    geometry: (usize, usize),
    position: usize,
    ranges: Vec<(usize, usize)>,
    hidden: Buffer,
}
impl Decoder {
    fn packed_for(&mut self, total: usize) -> Result<&mut PackedBuffers, Error> {
        if total == 0
            || total > i32::MAX as usize
            || total as u64 > self.packed_rows
        {
            return Err("packed row count exceeds the prepared envelope".into());
        }
        if !self.packed.contains_key(&total) {
            let hidden = self
                .device
                .buffer(
                    total
                        .checked_mul(row_hidden_bytes(&self.geometry)?)
                        .ok_or("packed hidden extent overflow")?,
                )?;
            let tokens = self
                .device
                .buffer(total.checked_mul(4).ok_or("packed token extent overflow")?)?;
            self.packed.insert(total, PackedBuffers { hidden, tokens });
        }
        Ok(self.packed.get_mut(&total).expect("packed buffers prepared"))
    }
    pub(super) fn execute_generation_states(
        &mut self,
        states: &mut [&mut SequenceState],
        work: &[GenerationWork<'_>],
    ) -> Result<Vec<Result<Option<TokenId>, String>>, Error> {
        if states.len() != work.len() || work.is_empty() {
            return Err("packed state and request membership differ".into());
        }
        for (state, row) in states.iter().zip(work) {
            if !state.belongs_to(&self.store)
                || row.proposal.tokens().is_empty()
                || row.proposal.tokens().len() > self.context_capacity - state.position()
                || row.proposal.tokens().iter().any(|token| {
                    u64::from(token.0) >= self.geometry.vocabulary || token.0 > i32::MAX as u32
                })
                || state.history_ranges().len() > self.max_ranges
                || row.mask.is_some_and(|mask| {
                    match usize::try_from(self.geometry.vocabulary) {
                        Ok(vocabulary) => mask.len() != vocabulary.div_ceil(32),
                        // A vocabulary beyond the host index domain admits no mask.
                        Err(_) => true,
                    }
                })
            {
                return Err("invalid packed model, context, vocabulary, or mask".into());
            }
        }
        let total = work.iter().try_fold(0usize, |sum, row| {
            sum.checked_add(row.proposal.tokens().len())
                .ok_or("packed row count overflow")
        })?;
        self.packed_for(total)?;
        for (state, row) in states.iter().zip(work) {
            self.rows_for(
                row.proposal.tokens().len(),
                state.history_ranges().len().max(1),
            )?;
        }
        let Decoder {
            geometry,
            device,
            context_capacity,
            embedding,
            blocks,
            readout: readout_stage,
            sampler,
            rows,
            packed,
            ..
        } = self;
        let packed = packed.get_mut(&total).expect("packed buffers prepared");
        packed.tokens.write(
            &work
                .iter()
                .flat_map(|row| {
                    row.proposal
                        .tokens()
                        .iter()
                        .flat_map(|token| (token.0 as i32).to_le_bytes())
                })
                .collect::<Vec<_>>(),
        )?;
        let width = row_hidden_bytes(geometry)?;
        let mut offset = 0;
        let mut members = Vec::with_capacity(work.len());
        for (state, row) in states.iter().zip(work) {
            let count = row.proposal.tokens().len();
            let mut ranges = state.history_ranges();
            if ranges.is_empty() {
                ranges.push((0, 0));
            }
            let bytes = count
                .checked_mul(width)
                .ok_or("member hidden extent overflow")?;
            members.push(Member {
                geometry: (count, ranges.len()),
                position: state.position(),
                ranges,
                hidden: packed.hidden.view(offset..offset + bytes)?,
            });
            offset += bytes;
        }
        let mut advances = states
            .iter_mut()
            .zip(work)
            .map(|(state, row)| state.begin(row.proposal.tokens().len()))
            .collect::<Result<Vec<_>, _>>()?;
        let mut selected = Vec::with_capacity(work.len());
        let mut state_results = vec![Vec::new(); work.len()];
        StateAdvance::execute_batch(&mut advances, |transitions| {
            let embedded = execution::execute(
                embedding,
                &forward_shapes(total),
                &HashMap::from([("tokens".into(), packed.tokens.clone())]),
                &HashMap::new(),
            )?;
            packed.hidden = result_buffer(&embedded, &[1])?;
            let mut packed_offset = 0;
            for member in &mut members {
                let bytes = member
                    .geometry
                    .0
                    .checked_mul(width)
                    .ok_or("member hidden extent overflow")?;
                member.hidden = packed.hidden.view(packed_offset..packed_offset + bytes)?;
                packed_offset += bytes;
            }
            for index in 0..blocks.len() {
                for (member_index, (member, transition)) in
                    members.iter_mut().zip(transitions).enumerate()
                {
                    let rows = rows_entry(
                        geometry,
                        device,
                        rows,
                        member.geometry.0,
                        member.geometry.1,
                        *context_capacity,
                    )?;
                    let block = &blocks[index];
                    let mut tensors = HashMap::from([("hidden".into(), member.hidden.clone())]);
                    let i = block.state_index;
                    if block.attention {
                        rows.coordinates.write(
                            &(0..member.geometry.0)
                                .flat_map(|row| [((member.position + row) as i32); 4])
                                .flat_map(i32::to_le_bytes)
                                .collect::<Vec<_>>(),
                        )?;
                        rows.visible.write(
                            &(0..member.geometry.0)
                                .flat_map(|_| {
                                    member.ranges.iter().flat_map(|&(start, count)| {
                                        [start as i32, (start + count) as i32]
                                    })
                                })
                                .flat_map(i32::to_le_bytes)
                                .collect::<Vec<_>>(),
                        )?;
                        rows.destinations.write(
                            &transition
                                .destinations
                                .iter()
                                .flat_map(|&destination| (destination as i32).to_le_bytes())
                                .collect::<Vec<_>>(),
                        )?;
                        tensors.extend([
                            ("coordinates".into(), rows.coordinates.clone()),
                            ("visible".into(), rows.visible.clone()),
                            ("destinations".into(), rows.destinations.clone()),
                            ("history_key".into(), transition.history[i].clone()),
                            ("history_value".into(), transition.history[i + 1].clone()),
                        ]);
                    } else {
                        tensors.extend([
                            ("window".into(), transition.previous[i].clone()),
                            ("delta".into(), transition.previous[i + 1].clone()),
                        ]);
                    }
                    // Complete this member before reusing geometry-local scratch
                    // or control buffers for another request of the same shape.
                    let shapes = if block.attention {
                        attention_shapes(member.geometry.0, member.geometry.1)
                    } else {
                        forward_shapes(member.geometry.0)
                    };
                    let mixed = execution::execute(&block.mixer, &shapes, &tensors, &HashMap::new())?;
                    member.hidden = if block.attention {
                        result_buffer(&mixed, &[])?
                    } else {
                        state_results[member_index].push((i, result_buffer(&mixed, &[0])?));
                        state_results[member_index].push((i + 1, result_buffer(&mixed, &[1])?));
                        result_buffer(&mixed, &[2])?
                    };
                    let fed = execution::execute(
                        &block.feedforward,
                        &forward_shapes(member.geometry.0),
                        &HashMap::from([("residual".into(), member.hidden.clone())]),
                        &HashMap::new(),
                    )?;
                    member.hidden =
                        result_buffer(&fed, &[6]).or_else(|_| result_buffer(&fed, &[14]))?;
                }
            }
            for (row, member) in work.iter().zip(&members) {
                if !row.proposal.needs_sample() {
                    selected.push(Ok(None));
                    continue;
                }
                let rows = rows_entry(
                    geometry,
                    device,
                    rows,
                    member.geometry.0,
                    member.geometry.1,
                    *context_capacity,
                )?;
                let logits = execution::execute(
                    readout_stage,
                    &forward_shapes(member.geometry.0),
                    &HashMap::from([("hidden".into(), member.hidden.clone())]),
                    &HashMap::new(),
                )?;
                rows.logits = result_buffer(&logits, &[1])?;
                let selection = sampler.sample(
                    &rows.logits,
                    row.mask,
                    row.proposal.sampling(),
                    row.proposal.seed(),
                    row.proposal.sample_position(),
                )?;
                selected.push(match selection {
                    Selection::Token(token) => Ok(Some(token)),
                    Selection::Empty => Err("empty sampling distribution".into()),
                    Selection::Nonfinite => Err("nonfinite sampling distribution".into()),
                });
            }
            Ok(())
        })?;
        for (advance, replacements) in advances.iter_mut().zip(state_results) {
            for (index, buffer) in replacements {
                advance.replace_following(index, buffer)?;
            }
        }
        for advance in advances {
            advance.commit()?;
        }
        Ok(selected)
    }
}
