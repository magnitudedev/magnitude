// stored_rows on CPU (contract and portable body in target.seismic). One work
// item per row finds its slot through the directory and decodes the slot's
// stored gguf_iq4_nl_block blocks (an f16 scale, then sixteen bytes whose low
// nibbles are values 0..16 and high nibbles 16..32, indexing the iq4 table)
// into A.

use lib::core::{activation, gathered};

const IQ4: [f32; 16] = [
    -127.0, -104.0, -83.0, -65.0, -49.0, -35.0, -22.0, -10.0, 1.0, 13.0, 25.0, 38.0, 53.0, 69.0, 89.0,
    113.0,
];
const BLOCK_VALUES: usize = 32;
const BLOCK_BYTES: usize = 18;

fn stored_rows<L: Isa, E: Elements>(_l: L, cx: &Context<'_, E>, group: [u64; 3], shared: &mut [u8]) {
    let row = group[0] as usize;
    let slot = gathered::row(cx.arg_tokens().get([row, 0]), cx.dim_r(), cx.dim_f(), |entry, field| {
        cx.arg_directory().get([entry, field])
    });
    let width = cx.dim_w() as usize;
    let stored = cx.arg_slots().row(slot);
    let values = seismic::cpu::tensor::floats(shared, width);
    for (column, value) in values.iter_mut().enumerate() {
        let block = &stored[column / BLOCK_VALUES * BLOCK_BYTES..][..BLOCK_BYTES];
        let scale = seismic::f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let within = column % BLOCK_VALUES;
        let code = (block[2 + within % 16] >> (4 * (within / 16))) & 15;
        *value = activation::publish::<E::A>(IQ4[code as usize] * scale);
    }
    // SAFETY: each work item writes its own row.
    activation::store::<E::A>(values, unsafe { cx.result_0().row_mut([row, 0]) });
}
