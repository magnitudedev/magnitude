// Read directory-selected slots, rows16 rows of the table's resident
// representation, through the row decoder, and round once to the activation
// element.

use lib::core::{activation, gathered};

fn stored_rows<L: Isa, E: Elements>(_l: L, cx: &Context<'_, E>, group: [u64; 3], shared: &mut [u8]) {
    let row = group[0] as usize;
    let slot = gathered::row(cx.arg_tokens().get([row, 0]), cx.dim_r(), cx.dim_f(), |entry, field| {
        cx.arg_directory().get([entry, field])
    });
    let values = seismic::cpu::tensor::floats(shared, cx.dim_w() as usize);
    cx.arg_slots().decode_row(slot, values);
    for value in values.iter_mut() {
        *value = activation::publish::<E::A>(*value);
    }
    // SAFETY: each work item writes its own row.
    activation::store::<E::A>(values, unsafe { cx.result_0().row_mut([row, 0]) });
}
