// copy_rows on CPU (contract and portable body in state.seismic). One work
// item per copied item moves its KV rows of W stored elements bit for bit
// (storage words, never decoded). Items whose source or destination row lies
// outside its tensor are skipped, as on Metal and CUDA.

use seismic::cpu::slab::SlabTensor;

fn copy_rows<L: Isa, E: Elements>(_l: L, cx: &Context<'_, E>, group: [u64; 3], _shared: &mut [u8]) {
    let item = group[0] as usize;
    let slab_rows = cx.arg_slab_rows() as usize;
    let rows = SlabTensor::from_bound(cx.arg_rows(), slab_rows);
    let (from, to) = (cx.arg_from().get([item]), cx.arg_to().get([item]));
    if from < 0 || to < 0 || from as u64 >= cx.dim_t() || to as u64 >= cx.dim_t() {
        return;
    }
    let (from, to) = (from as usize, to as usize);
    if from == to {
        return;
    }
    for head in 0..cx.dim_kv() as usize {
        // SAFETY: compaction names distinct destination rows, so each item
        // writes its own rows.
        let (source, length) = {
            let source = rows.row([from, head, 0]);
            (source.as_ptr(), source.len())
        };
        let target = unsafe { rows.row_mut([to, head, 0]) };
        // SAFETY: a compaction source is occupied and its destination is free.
        // They cannot overlap, and the row bounds were checked above.
        unsafe { std::ptr::copy_nonoverlapping(source, target.as_mut_ptr(), length) };
    }
}
