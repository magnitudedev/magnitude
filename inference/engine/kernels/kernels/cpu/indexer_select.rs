// indexer_select on CPU (contract and portable body in indexer.seismic).
// `append`: one work item per row stores its raw key (widened to F32) and
// coordinates in the window planes. `complete`: one work item per row that
// completes a block stores the block's F32 key (mean of its raw keys: the
// last SB - 1 rows of its window spans, then its own; normed, rotated at the
// coordinates stored with the first) in the block plane. `select` (classes that score): one work item per
// row prepares its query heads, scores its complete blocks, selects the K
// best (score descending, block ascending) past the dense limit, and emits
// its block bitmap and, with C > 0, its segments.

use lib::attention::attention;
use lib::core::activation;
use seismic::cpu::slab::SlabTensor;

fn indexer_select_append<L: Isa, E: Elements>(_l: L, cx: &Context<'_, E>, group: [u64; 3], _shared: &mut [u8]) {
    let row = group[0] as usize;
    let destination = cx.arg_window_destinations().get([row]);
    if destination < 0 {
        return;
    }
    let window = SlabTensor::from_bound(cx.arg_window_keys(), cx.arg_window_slab_rows() as usize);
    // SAFETY: window destinations are distinct freshly reserved rows no
    // work item of this launch reads.
    let target = unsafe { window.row_mut([destination as usize, 0]) };
    cx.arg_key().decode_row(row, target);
    let slab_rows = cx.arg_window_slab_rows() as usize;
    // SAFETY: as above.
    let target = unsafe { coordinate_row(cx.arg_window_coordinates(), slab_rows, destination as usize) };
    target.copy_from_slice(cx.arg_coordinates().row([row, 0]));
}

/// Row `at` of the window coordinates plane (`[TW, 4]` i32 slabs of
/// `slab_rows` rows).
///
/// # Safety
/// The caller gives each work item distinct writes and no concurrent readers.
#[allow(clippy::mut_from_ref)]
unsafe fn coordinate_row<'a>(plane: seismic::cpu::Scalars<'a, i32, 2>, slab_rows: usize, at: usize) -> &'a mut [i32] {
    // SAFETY: native launch validation owns every named slab and row; a row
    // is four contiguous words.
    unsafe {
        let region = seismic::cpu::slab::region::<i32>(plane.pointer().cast(), at / slab_rows);
        std::slice::from_raw_parts_mut(region.add((at % slab_rows) * 4), 4)
    }
}

/// The physical rows of row `row`'s `count` spans `spans`, in order.
fn span_rows(spans: &impl Fn(usize, usize) -> i32, count: usize) -> impl Iterator<Item = i32> + '_ {
    (0..count).flat_map(move |span| spans(span, 0)..spans(span, 1).max(spans(span, 0)))
}

fn indexer_select_complete<L: Isa, E: Elements>(_l: L, cx: &Context<'_, E>, group: [u64; 3], shared: &mut [u8]) {
    let row = group[0] as usize;
    let destination = cx.arg_block_destinations().get([row]);
    if destination < 0 {
        return;
    }
    let (p, w, sb) = (cx.dim_p() as usize, (2 * cx.dim_p() + cx.dim_s()) as usize, cx.dim_sb() as usize);
    let storage = seismic::cpu::tensor::floats(shared, 2 * p + 3 * w);
    let (angles, vectors) = storage.split_at_mut(2 * p);
    let (cosines, sines) = angles.split_at_mut(p);
    let (sum, vectors) = vectors.split_at_mut(w);
    let (mean, finished) = vectors.split_at_mut(w);
    // The block's raw keys: the last SB - 1 rows of the window spans, then
    // the row's own.
    let window_visible = cx.arg_window_visible();
    let earlier = span_rows(&|span, bound| window_visible.get([row, span, bound]), cx.dim_r() as usize)
        .collect::<Vec<_>>();
    let own = cx.arg_window_destinations().get([row]);
    let members = earlier[earlier.len().saturating_sub(sb - 1)..]
        .iter()
        .copied()
        .chain(std::iter::repeat(own))
        .take(sb)
        .collect::<Vec<_>>();
    // SAFETY: only read.
    let first = unsafe {
        coordinate_row(cx.arg_window_coordinates(), cx.arg_window_slab_rows() as usize, members[0] as usize)
    };
    attention::scaled_angles(
        first,
        cx.arg_rotary_components().row([0]),
        cx.arg_rotary_frequencies().row([0]),
        cx.arg_rotary_amplitudes().row([0]),
        cosines,
        sines,
    );
    let window = SlabTensor::from_bound(cx.arg_window_keys(), cx.arg_window_slab_rows() as usize);
    sum.fill(0.0);
    for &at in &members {
        for (total, value) in sum.iter_mut().zip(window.row([at as usize, 0]).iter()) {
            *total += *value;
        }
    }
    // The F32 mean, as `head_rotary` takes its row.
    for (column, total) in sum.iter().enumerate() {
        mean[column] = *total / sb as f32;
    }
    attention::prepare_head::<seismic::cpu::F32>(
        mean,
        Some(cx.arg_key_norm().row([0, 0])),
        cosines,
        sines,
        cx.arg_epsilon(),
        finished,
    );
    let blocks = SlabTensor::from_bound(cx.arg_block_keys(), cx.arg_block_slab_rows() as usize);
    // SAFETY: block destinations are distinct freshly reserved rows no work
    // item of this launch reads.
    unsafe { blocks.row_mut([destination as usize, 0]) }.copy_from_slice(finished);
}

fn indexer_select_select<L: Isa, E: Elements>(_l: L, cx: &Context<'_, E>, group: [u64; 3], shared: &mut [u8]) {
    let row = group[0] as usize;
    let (h, p, w) = (cx.dim_h() as usize, cx.dim_p() as usize, (2 * cx.dim_p() + cx.dim_s()) as usize);
    let (sb, k, nb) = (cx.dim_sb() as usize, cx.arg_selected().max(0) as usize, 32 * cx.dim_b() as usize);
    let (r, c, b) = (cx.dim_r() as usize, cx.dim_c() as usize, cx.dim_b() as usize);
    let storage = seismic::cpu::tensor::floats(shared, 2 * p + h * w + w);
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
    let query = cx.arg_query();
    let norm = cx.arg_query_norm().row([0, 0]);
    for head in 0..h {
        query.decode_row(row * h + head, key);
        attention::prepare_head::<seismic::cpu::F32>(
            key,
            Some(norm),
            cosines,
            sines,
            cx.arg_epsilon(),
            &mut queries[head * w..][..w],
        );
    }
    // The row's complete blocks in block order, scored.
    let spans = cx.arg_block_visible();
    let own = cx.arg_block_destinations().get([row]);
    let physical = span_rows(&|span, bound| spans.get([row, span, bound]), r)
        .chain((own >= 0).then_some(own))
        .take(nb)
        .collect::<Vec<_>>();
    let complete = physical.len();
    let block_keys = SlabTensor::from_bound(cx.arg_block_keys(), cx.arg_block_slab_rows() as usize);
    let root = (w as f32).sqrt();
    // SAFETY: each work item owns its row of the scores.
    let scores = unsafe { cx.scratch_scores().slice_mut::<f32>(4 * row * nb, nb) };
    for (block, &at) in physical.iter().enumerate() {
        key.copy_from_slice(block_keys.row([at as usize, 0]));
        let mut score = 0.0f32;
        for head in 0..h {
            let mut dot = 0.0f32;
            for (q, x) in queries[head * w..][..w].iter().zip(key.iter()) {
                dot = q.mul_add(*x, dot);
            }
            score += dot.max(0.0);
        }
        scores[block] = score / root;
    }
    // Read blocks: every block while dense; else the K best and every block
    // from the tail's on.
    let mut reads = vec![true; nb + 1];
    if complete > k {
        let mut order = (0..complete).collect::<Vec<_>>();
        let better = |left: &usize, right: &usize| {
            scores[*right]
                .partial_cmp(&scores[*left])
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(left.cmp(right))
        };
        order.select_nth_unstable_by(k - 1, better);
        reads[..complete].fill(false);
        for &block in &order[..k] {
            reads[block] = true;
        }
    }
    let read = |block: usize| reads[block.min(nb)];
    {
        let selection = cx.result_1();
        for word in 0..b {
            let bits = (0..32).filter(|bit| read(word * 32 + bit)).fold(0u32, |bits, bit| bits | 1 << bit);
            // SAFETY: each work item writes its own row.
            unsafe { selection.set([row, word], bits) };
        }
    }
    if c > 0 {
        // The row's keys in its visible spans (its fresh rows, its tail,
        // attention reads itself).
        let visible = cx.arg_visible();
        let slab_rows = cx.arg_slab_rows() as i32;
        let bounds = |span, bound| visible.get([row, span, bound]);
        let keys = span_rows(&bounds, r);
        let mut runs: Vec<[i32; 2]> = Vec::new();
        let mut open = false;
        for (position, stored) in keys.enumerate() {
            if !read(position / sb) {
                open = false;
                continue;
            }
            match runs.last_mut() {
                Some(run) if open && run[1] == stored && stored % slab_rows != 0 => run[1] += 1,
                _ => runs.push([stored, stored + 1]),
            }
            open = true;
        }
        let segments = cx.result_0();
        for entry in 0..c {
            let run = runs.get(entry).copied().unwrap_or([0, 0]);
            // SAFETY: each work item writes its own row.
            unsafe {
                segments.set([row, entry, 0], run[0]);
                segments.set([row, entry, 1], run[1]);
            }
        }
    }
}
