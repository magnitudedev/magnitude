// ngram_rows on CPU (contract and portable body in ngram.seismic): the n-gram
// table rows of each launch row from its token and its predecessors, the ones
// before the slot read from the slot's token window. Work item r < M forms
// row r's C * P rows ((0, 0) past the slots); work item M + b publishes slot
// b's successor window. The hash is the body's u64 words, so every backend
// names the same rows. The successor window is no slot's source, so the
// publishing work items run beside the row work items.

use lib::recurrent::versions::Slots;
use seismic::cpu::slab::SlabTensor;

fn ngram_rows<L: Isa, E: Elements>(_l: L, cx: &Context<'_, E>, group: [u64; 3], _shared: &mut [u8]) {
    let slots = Slots {
        segments: cx.arg_segments(),
        stop: cx.arg_stop(),
        previous_bank: cx.arg_previous_bank(),
        previous_tape: cx.arg_previous_tape(),
        following_bank: cx.arg_following_bank(),
        count: cx.dim_b() as usize,
    };
    let tokens = cx.arg_tokens();
    let multipliers = cx.arg_multipliers();
    let sizes = cx.arg_sizes();
    let offsets = cx.arg_offsets();
    let window = SlabTensor::from_scalars(cx.arg_window(), cx.arg_slab_banks() as usize);
    let rows = cx.result_0();
    let context = cx.dim_c() as usize;
    let heads_per_order = cx.dim_p() as usize;
    let launch_rows = cx.dim_m() as usize;
    let item = group[0] as usize;
    // The token of launch row `row`, stored as the window stores it.
    let stored = |row: usize| tokens.get([row, 0]).wrapping_add(1) as u32;
    if item >= launch_rows {
        let slot = slots.slot(item - launch_rows);
        for tap in 0..context + slot.recorded(cx.dim_t() as usize) {
            let value = match (slot.stop + tap).checked_sub(context) {
                None => window.row([slot.source, slot.taped + slot.stop + tap, 0])[0],
                Some(position) => stored(slot.lo + position),
            };
            // SAFETY: each slot's work item writes its own successor bank,
            // which no slot reads.
            unsafe { window.row_mut([slot.target, tap, 0])[0] = value };
        }
        return;
    }
    let heads = context * heads_per_order;
    // SAFETY: each work item writes its own rows.
    let put = |head: usize, row: i32, status: i32| unsafe {
        rows.set([item * heads + head, 0], row);
        rows.set([item * heads + head, 1], status);
    };
    let Some(slot) = slots.slot_of_row(item) else {
        for head in 0..heads {
            put(head, 0, 0);
        }
        return;
    };
    let local = item - slot.lo;
    let reset = cx.arg_reset();
    let reset_stored = reset.wrapping_add(1) as u32;
    let status = tokens.get([item, 1]);
    let mut mixed = (i64::from(tokens.get([item, 0])) as u64).wrapping_mul(multipliers.get([0]));
    let mut cut = false;
    for distance in 1..=context {
        let value = match local.checked_sub(distance) {
            Some(_) => stored(item - distance),
            None => window.row([slot.source, slot.taped + context + local - distance, 0])[0],
        };
        cut |= value == 0 || value == reset_stored;
        let token = if cut {
            i64::from(reset) as u64
        } else {
            u64::from(value - 1)
        };
        mixed ^= token.wrapping_mul(multipliers.get([distance]));
        for p in 0..heads_per_order {
            let head = (distance - 1) * heads_per_order + p;
            let row = offsets.get([head]).wrapping_add(mixed % sizes.get([head]));
            put(head, row as i32, status);
        }
    }
}
