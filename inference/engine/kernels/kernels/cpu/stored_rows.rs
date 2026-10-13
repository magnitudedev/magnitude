// Read directory-selected slots in their stored IQ4_NL or Q8_0 encoding.
// Both have an F16 scale per 32 values; codes are nonlinear nibbles or
// signed bytes. Decode directly and round once to the activation element.

use lib::core::{activation, gathered};

const IQ4: [f32; 16] = [
    -127.0, -104.0, -83.0, -65.0, -49.0, -35.0, -22.0, -10.0, 1.0, 13.0, 25.0, 38.0, 53.0, 69.0, 89.0,
    113.0,
];
const BLOCK_VALUES: usize = 32;


fn stored_rows<L: Isa, E: Elements>(_l: L, cx: &Context<'_, E>, group: [u64; 3], shared: &mut [u8]) {
    let row = group[0] as usize;
    let slot = gathered::row(cx.arg_tokens().get([row, 0]), cx.dim_r(), cx.dim_f(), |entry, field| {
        cx.arg_directory().get([entry, field])
    });
    let width = cx.dim_w() as usize;
    let source = cx.arg_slots();
    let q8 = match source.representation() {
        "gguf_q8_0" => true,
        "gguf_iq4_nl_block" => false,
        other => panic!("unsupported stored rows: {other}"),
    };
    let block_bytes = if q8 { 34 } else { 18 };
    let stored = source.row(slot);
    let values = seismic::cpu::tensor::floats(shared, width);
    for (column, value) in values.iter_mut().enumerate() {
        let block = &stored[column / BLOCK_VALUES * block_bytes..][..block_bytes];
        let scale = seismic::f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let within = column % BLOCK_VALUES;
        let code = if q8 {
            block[2 + within] as i8 as f32
        } else {
            IQ4[((block[2 + within % 16] >> (4 * (within / 16))) & 15) as usize]
        };
        *value = activation::publish::<E::A>(code * scale);
    }
    // SAFETY: each work item writes its own row.
    activation::store::<E::A>(values, unsafe { cx.result_0().row_mut([row, 0]) });
}
