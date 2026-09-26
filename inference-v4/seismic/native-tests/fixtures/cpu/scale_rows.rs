// One work item per `ROWS` rows.
fn scale_rows<L: Isa, E: Elements>(_l: L, cx: &Context<'_, E>, group: [u64; 3], _shared: &mut [u8]) {
    let x = cx.arg_x();
    let result = cx.result_0();
    let factor = cx.arg_factor();
    let rows = cx.param_rows();
    for local in 0..rows {
        let row = group[0] * rows + local;
        if row >= cx.dim_m() {
            return;
        }
        for column in 0..cx.dim_n() as usize {
            // SAFETY: each work item writes its own rows.
            unsafe { result.set([row as usize, column], x.get([row as usize, column]) * factor) };
        }
    }
}
