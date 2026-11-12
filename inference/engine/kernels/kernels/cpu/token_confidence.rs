// token_confidence on CPU (contract and portable body in dflash.seismic), in
// two launches. `token_confidence_partition` gives each work item one of
// PARTS contiguous slices of a selection's logits row and keeps its (maximum,
// sum of exp(logit - maximum)) in scratch; `token_confidence_merge` combines
// a selection's slices and publishes the probability of the selected token (0
// when its selection failed).

const PARTS: usize = 64;

#[inline(always)]
fn token_confidence_partition<L: Isa, E: Elements>(_l: L, cx: &Context<'_, E>, group: [u64; 3], _shared: &mut [u8]) {
    let (part, row) = (group[0] as usize, group[1] as usize);
    let vocabulary = cx.dim_v() as usize;
    let span = vocabulary.div_ceil(PARTS);
    let begin = (part * span).min(vocabulary);
    let end = (begin + span).min(vocabulary);
    let line = &cx.arg_logits().row([cx.arg_rows().get([row]) as usize, 0])[begin..end];
    let maximum = line.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let total = line.iter().map(|logit| (logit - maximum).exp()).sum::<f32>();
    // SAFETY: each work item writes its own (row, part) pair.
    let out = unsafe { cx.scratch_partials().slice_mut::<f32>(8 * (row * PARTS + part), 2) };
    out.copy_from_slice(&[maximum, total]);
}

#[inline(always)]
fn token_confidence_merge<L: Isa, E: Elements>(_l: L, cx: &Context<'_, E>, group: [u64; 3], _shared: &mut [u8]) {
    let row = group[0] as usize;
    // SAFETY: the partition launch wrote every pair before this launch.
    let partials = unsafe { cx.scratch_partials().slice::<f32>(8 * row * PARTS, 2 * PARTS) };
    let maximum = partials.chunks_exact(2).map(|pair| pair[0]).fold(f32::NEG_INFINITY, f32::max);
    let total = partials
        .chunks_exact(2)
        .filter(|pair| pair[1] > 0.0)
        .map(|pair| pair[1] * (pair[0] - maximum).exp())
        .sum::<f32>();
    let line = cx.arg_logits().row([cx.arg_rows().get([row]) as usize, 0]);
    let selection = cx.arg_selection();
    let probability = if selection.get([row, 1]) == 0 {
        (line[selection.get([row, 0]) as usize] - maximum).exp() / total
    } else {
        0.0
    };
    // SAFETY: each work item writes its own row.
    unsafe { cx.arg_confidence().set([row], probability) };
}
