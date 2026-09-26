// qwen_vision_merger on CPU (contract and portable body in vision.seismic).
// `qwen_vision_merger_norm` publishes the layer norm of each F32 patch row
// (rounded to A) as F32 into `normalized`, one work item per patch row; G
// consecutive rows are one merged row of G * H. `qwen_vision_merger_up` gives
// each work item ROWS up weight rows, publishing A(gelu(A(x . w + b))) to
// `activated`; `qwen_vision_merger_down` gives each ROWS down weight rows,
// storing the F32 x . w + b as the image features.

use lib::core::activation;
use lib::projection::projection;
use lib::vision::vision;

fn qwen_vision_merger_norm<L: Isa, E: Elements>(_l: L, cx: &Context<'_, E>, group: [u64; 3], shared: &mut [u8]) {
    let row = group[0] as usize;
    let h = cx.dim_h() as usize;
    let shared = seismic::cpu::tensor::floats(shared, 2 * h);
    let (weight, bias) = shared.split_at_mut(h);
    let weight = vision::decode_vector(&cx.arg_norm_weight(), weight);
    let bias = vision::decode_vector(&cx.arg_norm_bias(), bias);
    // SAFETY: each work item writes its own row of the scratch.
    let normalized = unsafe { cx.scratch_normalized().slice_mut::<f32>(4 * row * h, h) };
    vision::layer_norm::<E::A>(cx.arg_hidden().row([row, 0]), weight, bias, cx.arg_epsilon(), normalized);
}

fn qwen_vision_merger_up<L: Isa, E: Elements>(_l: L, cx: &Context<'_, E>, group: [u64; 3], shared: &mut [u8]) {
    let (m, width) = (cx.dim_m() as usize, (cx.dim_g() * cx.dim_h()) as usize);
    let rows = projection::item_rows(group[0], cx.param_rows(), width);
    let bias = vision::decode_vector(&cx.arg_up_bias(), seismic::cpu::tensor::floats(shared, width));
    // SAFETY: the norm launch wrote every row before this launch.
    let normalized = unsafe { cx.scratch_normalized().slice::<f32>(0, m * width) };
    let activated = cx.scratch_activated();
    vision::project_bias(&cx.arg_up_weight(), bias, rows, normalized, |row, first, values| {
        // SAFETY: each work item writes its own columns of every row.
        let out = unsafe { activated.slice_mut::<f32>(4 * (row * width + first), values.len()) };
        for (target, value) in out.iter_mut().zip(values) {
            let up = activation::publish::<E::A>(*value);
            *target = activation::publish::<E::A>(vision::gelu_erf(up));
        }
    });
}

fn qwen_vision_merger_down<L: Isa, E: Elements>(_l: L, cx: &Context<'_, E>, group: [u64; 3], shared: &mut [u8]) {
    let (m, width, d) = (cx.dim_m() as usize, (cx.dim_g() * cx.dim_h()) as usize, cx.dim_d() as usize);
    let rows = projection::item_rows(group[0], cx.param_rows(), d);
    let bias = vision::decode_vector(&cx.arg_down_bias(), seismic::cpu::tensor::floats(shared, d));
    // SAFETY: the up launch wrote every row before this launch.
    let activated = unsafe { cx.scratch_activated().slice::<f32>(0, m * width) };
    let result = cx.result_0();
    vision::project_bias(&cx.arg_down_weight(), bias, rows, activated, |row, first, values| {
        // SAFETY: each work item writes its own columns of every row.
        let out = unsafe { result.span_mut([row, first], values.len()) };
        out.copy_from_slice(values);
    });
}
