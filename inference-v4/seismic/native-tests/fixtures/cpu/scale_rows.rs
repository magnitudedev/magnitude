fn scale_rows<const ROWS: u64>(context: &Context<'_>, group: [u64; 3], _shared: &mut [u8]) {
    let x = context.arg_x();
    let result = context.result_0();
    let factor = context.arg_factor();
    for local in 0..ROWS {
        let row = group[0] * ROWS + local;
        if row >= context.dim_m() {
            return;
        }
        for column in 0..context.dim_n() {
            unsafe {
                let value = x
                    .pointer
                    .cast::<f32>()
                    .add((row * x.strides[0] + column * x.strides[1]) as usize)
                    .read();
                result
                    .pointer
                    .cast::<f32>()
                    .add((row * result.strides[0] + column * result.strides[1]) as usize)
                    .write(value * factor);
            }
        }
    }
}
