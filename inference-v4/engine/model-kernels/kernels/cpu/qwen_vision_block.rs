// qwen_vision_block on CPU (contract and portable body in vision.seismic).
// Rows are D = H * 4P wide; every intermediate is held as F32 in scratch,
// the published ones rounded to A where the portable body stores them.
//
// - `norm1` / `norm2`: the layer norm of each residual row (the input rows,
//   then the attention's residual), one work item per row, into `normalized`.
// - `qkv`, `output`, `up`, `down`: projections, each work item ROWS weight
//   rows against every normalized (or attended, activated) row, with the bias
//   added in F32: `qkv` publishes A(x . w + b) to `projected` [M][3][H][4P];
//   `output` adds the projection to the input row into `residual`; `up`
//   publishes A(gelu_tanh(A(x . w + b))) to `activated`; `down` adds the
//   projection to `residual` into the result.
// - `rotate`: the 2D rotary embedding of each row's query and key heads, in
//   place in `projected`, one work item per row.
// - `attend`: the full attention of one query row of one head over every
//   patch row, into `attended`.

use lib::core::activation;
use lib::projection::projection;
use lib::vision::vision;

/// The model width D = H * 4P.
fn width<E: Elements>(cx: &Context<'_, E>) -> usize {
    (cx.dim_h() * 4 * cx.dim_p()) as usize
}

/// The layer norm of `source` with the decoded norm vectors into row `row` of
/// `normalized`.
fn normalize<E: Elements>(
    cx: &Context<'_, E>,
    row: usize,
    source: &[f32],
    weight: &seismic::cpu::Weights<'_>,
    bias: &seismic::cpu::Weights<'_>,
    shared: &mut [u8],
) {
    let d = width(cx);
    let shared = seismic::cpu::tensor::floats(shared, 2 * d);
    let (decoded_weight, decoded_bias) = shared.split_at_mut(d);
    let decoded_weight = vision::decode_vector(weight, decoded_weight);
    let decoded_bias = vision::decode_vector(bias, decoded_bias);
    // SAFETY: each work item writes its own row of the scratch.
    let normalized = unsafe { cx.scratch_normalized().slice_mut::<f32>(4 * row * d, d) };
    vision::layer_norm::<E::A>(source, decoded_weight, decoded_bias, cx.arg_epsilon(), normalized);
}

fn qwen_vision_block_norm1<L: Isa, E: Elements>(_l: L, cx: &Context<'_, E>, group: [u64; 3], shared: &mut [u8]) {
    let row = group[0] as usize;
    let (p, d) = (cx.dim_p() as usize, width(cx));
    let hidden = cx.arg_hidden();
    // The input row, staged in `residual` for the output launch to add to.
    // SAFETY: each work item writes its own row of the scratch.
    let source = unsafe { cx.scratch_residual().slice_mut::<f32>(4 * row * d, d) };
    for (index, part) in source.chunks_exact_mut(p).enumerate() {
        part.copy_from_slice(hidden.row([row, index / 4, index % 4, 0]));
    }
    normalize(cx, row, source, &cx.arg_norm1_weight(), &cx.arg_norm1_bias(), shared);
}

fn qwen_vision_block_qkv<L: Isa, E: Elements>(_l: L, cx: &Context<'_, E>, group: [u64; 3], shared: &mut [u8]) {
    let (m, d) = (cx.dim_m() as usize, width(cx));
    let rows = projection::item_rows(group[0], cx.param_rows(), 3 * d);
    let bias = vision::decode_vector(&cx.arg_qkv_bias(), seismic::cpu::tensor::floats(shared, 3 * d));
    // SAFETY: the norm1 launch wrote every row before this launch.
    let normalized = unsafe { cx.scratch_normalized().slice::<f32>(0, m * d) };
    let projected = cx.scratch_projected();
    vision::project_bias(&cx.arg_qkv_weight(), bias, rows, normalized, |row, first, values| {
        // SAFETY: each work item writes its own columns of every row.
        let out = unsafe { projected.slice_mut::<f32>(4 * (row * 3 * d + first), values.len()) };
        for (target, value) in out.iter_mut().zip(values) {
            *target = activation::publish::<E::A>(*value);
        }
    });
}

fn qwen_vision_block_rotate<L: Isa, E: Elements>(_l: L, cx: &Context<'_, E>, group: [u64; 3], _shared: &mut [u8]) {
    let row = group[0] as usize;
    let (p, d) = (cx.dim_p() as usize, width(cx));
    let coordinates = cx.arg_coordinates();
    let coordinates = [coordinates.get([row, 0]), coordinates.get([row, 1])];
    // SAFETY: each work item rewrites the query and key heads of its own row.
    let queries_and_keys = unsafe { cx.scratch_projected().slice_mut::<f32>(4 * row * 3 * d, 2 * d) };
    for head in queries_and_keys.chunks_exact_mut(4 * p) {
        vision::rotate::<E::A>(head, coordinates, p);
    }
}

fn qwen_vision_block_attend<L: Isa, E: Elements>(_l: L, cx: &Context<'_, E>, group: [u64; 3], shared: &mut [u8]) {
    let (row, head) = (group[0] as usize, group[1] as usize);
    let (m, p, d) = (cx.dim_m() as usize, cx.dim_p() as usize, width(cx));
    let w = 4 * p;
    // SAFETY: the rotate launch wrote every row before this launch.
    let projected = unsafe { cx.scratch_projected().slice::<f32>(0, m * 3 * d) };
    let head_row = |row: usize, part: usize| &projected[row * 3 * d + part * d + head * w..][..w];
    let scores = seismic::cpu::tensor::floats(shared, m);
    // SAFETY: each work item writes its own head of its own row.
    let out = unsafe { cx.scratch_attended().slice_mut::<f32>(4 * (row * d + head * w), w) };
    vision::attend::<E::A>(head_row(row, 0), m, |key| head_row(key, 1), |key| head_row(key, 2), scores, out);
}

fn qwen_vision_block_output<L: Isa, E: Elements>(_l: L, cx: &Context<'_, E>, group: [u64; 3], shared: &mut [u8]) {
    let (m, d) = (cx.dim_m() as usize, width(cx));
    let rows = projection::item_rows(group[0], cx.param_rows(), d);
    let bias = vision::decode_vector(&cx.arg_projection_bias(), seismic::cpu::tensor::floats(shared, d));
    // SAFETY: the attend launch wrote every row before this launch.
    let attended = unsafe { cx.scratch_attended().slice::<f32>(0, m * d) };
    let residual = cx.scratch_residual();
    vision::project_bias(&cx.arg_projection_weight(), bias, rows, attended, |row, first, values| {
        // SAFETY: each work item reads and rewrites its own columns of every
        // row (the norm1 launch staged the input rows there).
        let out = unsafe { residual.slice_mut::<f32>(4 * (row * d + first), values.len()) };
        for (target, value) in out.iter_mut().zip(values) {
            *target += value;
        }
    });
}

fn qwen_vision_block_norm2<L: Isa, E: Elements>(_l: L, cx: &Context<'_, E>, group: [u64; 3], shared: &mut [u8]) {
    let row = group[0] as usize;
    let d = width(cx);
    // SAFETY: the output launch wrote every row before this launch.
    let source = unsafe { cx.scratch_residual().slice::<f32>(4 * row * d, d) };
    normalize(cx, row, source, &cx.arg_norm2_weight(), &cx.arg_norm2_bias(), shared);
}

fn qwen_vision_block_up<L: Isa, E: Elements>(_l: L, cx: &Context<'_, E>, group: [u64; 3], shared: &mut [u8]) {
    let (m, d, f) = (cx.dim_m() as usize, width(cx), cx.dim_f() as usize);
    let rows = projection::item_rows(group[0], cx.param_rows(), f);
    let bias = vision::decode_vector(&cx.arg_up_bias(), seismic::cpu::tensor::floats(shared, f));
    // SAFETY: the norm2 launch wrote every row before this launch.
    let normalized = unsafe { cx.scratch_normalized().slice::<f32>(0, m * d) };
    let activated = cx.scratch_activated();
    vision::project_bias(&cx.arg_up_weight(), bias, rows, normalized, |row, first, values| {
        // SAFETY: each work item writes its own columns of every row.
        let out = unsafe { activated.slice_mut::<f32>(4 * (row * f + first), values.len()) };
        for (target, value) in out.iter_mut().zip(values) {
            let up = activation::publish::<E::A>(*value);
            *target = activation::publish::<E::A>(vision::gelu_tanh(up));
        }
    });
}

fn qwen_vision_block_down<L: Isa, E: Elements>(_l: L, cx: &Context<'_, E>, group: [u64; 3], shared: &mut [u8]) {
    let (m, d, f) = (cx.dim_m() as usize, width(cx), cx.dim_f() as usize);
    let rows = projection::item_rows(group[0], cx.param_rows(), d);
    let bias = vision::decode_vector(&cx.arg_down_bias(), seismic::cpu::tensor::floats(shared, d));
    // SAFETY: the up launch wrote every row before this launch.
    let activated = unsafe { cx.scratch_activated().slice::<f32>(0, m * f) };
    // SAFETY: the output launch wrote every row before this launch.
    let residual = unsafe { cx.scratch_residual().slice::<f32>(0, m * d) };
    let result = cx.result_0();
    vision::project_bias(&cx.arg_down_weight(), bias, rows, activated, |row, first, values| {
        let source = &residual[row * d + first..][..values.len()];
        // SAFETY: each work item writes its own columns of every row.
        let out = unsafe { result.span_mut([row, first], values.len()) };
        for ((target, residual), value) in out.iter_mut().zip(source).zip(values) {
            *target = residual + value;
        }
    });
}
