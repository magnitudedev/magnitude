// indexer_score on CPU (contract and portable body in indexer.seismic): one
// work item per (256 scores, row) prepares the row's query heads (normed
// when NQ = 1, rotated at the row's coordinates, rounded), then scores the
// rows of its share of the row's list: scale * sum_h relu(q[h] . K) * w[h],
// each dot an ascending FMA chain. Scores past the list are -inf.

use lib::attention::attention::{self, Layout};
use lib::core::rounding::Rounding;
use seismic::cpu::slab::SlabTensor;

fn indexer_score<L: Isa, E: Elements>(_l: L, cx: &Context<'_, E>, group: [u64; 3], shared: &mut [u8]) {
    let (share, row) = (group[0] as usize, group[1] as usize);
    let (h, p, w) = (cx.dim_h() as usize, cx.dim_p() as usize, (2 * cx.dim_p() + cx.dim_s()) as usize);
    let (n, r) = (cx.dim_n() as usize, cx.dim_r() as usize);
    let (first, last) = (share * 256, (share * 256 + 256).min(n));
    let scores = cx.result_0();
    // The row's list: the plane rows of its spans in order.
    let spans = cx.arg_spans();
    let list = (0..r)
        .flat_map(|span| {
            let (lo, hi) = (spans.get([row, span, 0]), spans.get([row, span, 1]));
            if lo >= 0 && hi >= lo { lo..hi } else { 0..0 }
        })
        .skip(first)
        .take(last - first)
        .collect::<Vec<_>>();
    for ordinal in first + list.len()..last {
        // SAFETY: each work item writes its own share of its row.
        unsafe { scores.set([row, ordinal], f32::NEG_INFINITY) };
    }
    if list.is_empty() {
        return;
    }
    let storage = seismic::cpu::tensor::floats(shared, 2 * p + (h + 1) * w);
    let (angles, vectors) = storage.split_at_mut(2 * p);
    let (cosines, sines) = angles.split_at_mut(p);
    let (queries, key) = vectors.split_at_mut(h * w);
    attention::scaled_angles(
        cx.arg_coordinates().row([row, 0]),
        cx.arg_rotary_components().row([0]),
        cx.arg_rotary_frequencies().row([0]),
        cx.arg_rotary_amplitudes().row([0]),
        cosines,
        sines,
    );
    let rounding = Rounding::of(cx.arg_block(), cx.arg_element(), cx.arg_rounding_scale(), cx.arg_amax_floor());
    let norm = (cx.dim_nq() == 1).then(|| cx.arg_query_norm().row([0, 0]));
    let layout = Layout::new(cx.dim_ra(), cx.dim_rt());
    let query = cx.arg_query();
    for head in 0..h {
        query.decode_row(row * h + head, key);
        let prepared = &mut queries[head * w..][..w];
        attention::prepare_head::<seismic::cpu::F32>(key, norm, layout, cosines, sines, cx.arg_epsilon(), prepared);
        if let Some(rounding) = rounding {
            rounding.apply(prepared);
        }
    }
    let mut weights = vec![1.0f32; h];
    if cx.dim_hw() > 0 {
        cx.arg_head_weights().decode_row(row, &mut weights);
    }
    let keys = SlabTensor::from_bound(cx.arg_keys(), cx.arg_slab_rows() as usize);
    let scale = cx.arg_scale();
    for (ordinal, &at) in (first..).zip(&list) {
        key.copy_from_slice(keys.row([at as usize, 0]));
        let mut score = 0.0f32;
        for (head, weight) in weights.iter().enumerate() {
            let mut dot = 0.0f32;
            for (q, x) in queries[head * w..][..w].iter().zip(key.iter()) {
                dot = q.mul_add(*x, dot);
            }
            score += dot.max(0.0) * weight;
        }
        // SAFETY: each work item writes its own share of its row.
        unsafe { scores.set([row, ordinal], scale * score) };
    }
}
