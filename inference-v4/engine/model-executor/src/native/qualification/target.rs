use super::super::*;
use crate::programs::graph::routed::{grouped_blocks, DECODE_ROWS, TILE_ROWS};
use super::*;

impl<'a> QualificationView<'a> {
    pub(super) fn qualify_target(&self, device: &Device) -> Result<(), CatalogFailure> {
        let out_rows = semantic_i32(device, &[1], &[0], "target", "out_rows")?;
        let hidden = self.geometry.hidden;
        let hidden_values = vec![1.0_f32; hidden as usize];
        let hidden_residual =
            semantic_f32(device, &[1, hidden], &hidden_values, "target", "hidden residual")?;

        for binding in [self.plan.target().embedding()] {
            let label = format!("{binding:?}");
            let table = semantic_pattern(
                device,
                binding.table,
                &[1, hidden],
                "embedding_rows",
                &label,
            )?;
            let tokens = semantic_i32(device, &[1, 2], &[0, 0], "embedding_rows", &label)?;
            let result = self
                .programs
                .target
                .embedding
                .call(embedding_rows::Args {
                    table: &table,
                    tokens: &tokens,
                })
                .map_err(|error| qualification_dynamic("embedding_rows", &label, error))?;
            require_finite_nonzero_f32(&result.r1, "embedding_rows", &label)?;
        }

        for (slot, attested) in self
            .plan
            .target()
            .blocks()
            .iter()
            .zip(&self.programs.target.blocks)
        {
            let (MixerProgramSlot::Attention(binding), AttestedMixer::Attention(kernel)) =
                (slot.mixer(), &attested.mixer)
            else {
                continue;
            };
            qualify_attention(
                device,
                kernel,
                binding.shape,
                &AttentionElements {
                    input_norm: binding.norm,
                    query_gate: binding.query_gate,
                    key: binding.key,
                    value: binding.value,
                    output: binding.output,
                    activation: binding.activation,
                },
                &format!("{binding:?}"),
            )?;
        }

        let mut qualified = std::collections::HashSet::new();
        for (slot, attested) in self
            .plan
            .target()
            .blocks()
            .iter()
            .zip(&self.programs.target.blocks)
        {
            let (MixerProgramSlot::Recurrent(binding), AttestedMixer::Recurrent(kernels)) =
                (slot.mixer(), &attested.mixer)
            else {
                continue;
            };
            if !qualified.insert(binding) {
                continue;
            }
            let label = format!("{binding:?}");
            // The model's recurrent geometry (the entries fix it statically)
            // over a two-bank arena: bank 0 is the zero seed the advance
            // reads, bank 1 its successor. Zero weights leave the residual
            // unchanged.
            let w = binding.width;
            let (key_heads, value_heads) = (binding.key_heads, binding.value_heads);
            let taps = binding.convolution_width;
            let channels = (2 * key_heads + value_heads) * w;
            let norm = semantic_zeros(device, binding.norm, &[hidden], "target_recurrent", &label)?;
            let qkv = semantic_zeros(
                device,
                binding.qkv,
                &[channels, hidden],
                "target_recurrent",
                &label,
            )?;
            let gate = semantic_zeros(
                device,
                binding.gate,
                &[value_heads * w, hidden],
                "target_recurrent",
                &label,
            )?;
            let alpha = semantic_zeros(
                device,
                binding.alpha,
                &[value_heads, hidden],
                "target_recurrent",
                &label,
            )?;
            let beta = semantic_zeros(
                device,
                binding.beta,
                &[value_heads, hidden],
                "target_recurrent",
                &label,
            )?;
            let convolution = semantic_zeros(
                device,
                Element::f32(),
                &[channels, taps],
                "target_recurrent",
                &label,
            )?;
            let rate =
                semantic_zeros(device, Element::f32(), &[value_heads], "target_recurrent", &label)?;
            let time_bias =
                semantic_zeros(device, Element::f32(), &[value_heads], "target_recurrent", &label)?;
            let recurrent_norm = semantic_zeros(
                device,
                binding.recurrent_norm,
                &[w],
                "target_recurrent",
                &label,
            )?;
            let output = semantic_zeros(
                device,
                binding.output,
                &[hidden, value_heads * w],
                "target_recurrent",
                &label,
            )?;
            let segments =
                semantic_i32(device, &[2, 2], &[0, 1, 1, 1], "target_recurrent", &label)?;
            let stop = semantic_i32(device, &[1], &[1], "target_recurrent", &label)?;
            let previous_bank = semantic_i32(device, &[1], &[0], "target_recurrent", &label)?;
            let previous_tape = semantic_i32(device, &[1], &[0], "target_recurrent", &label)?;
            let following_bank = semantic_i32(device, &[1], &[1], "target_recurrent", &label)?;
            // One tape row, the least a bank holds.
            let mut window = semantic_zeros(
                device,
                binding.activation,
                &[2, taps, channels],
                "target_recurrent",
                &label,
            )?;
            let mut delta = semantic_zeros(
                device,
                Element::f32(),
                &[2, value_heads, w, w],
                "target_recurrent",
                &label,
            )?;
            let mut tape = semantic_zeros(
                device,
                Element::f32(),
                &[2, 1, (value_heads + key_heads) * w + value_heads],
                "target_recurrent",
                &label,
            )?;
            let projection = kernels
                .project
                .call(qwen_recurrent_project::Args {
                    hidden: &hidden_residual,
                    input_norm: &norm,
                    qkv_weight: &qkv,
                    gate_weight: &gate,
                    alpha_weight: &alpha,
                    beta_weight: &beta,
                    epsilon: 1.0e-5,
                })
                .map_err(|error| qualification_dynamic("qwen_recurrent_project", &label, error))?
                .value;
            let mixed = kernels
                .step
                .call(qwen_recurrent_step::Args {
                    projection: &projection,
                    convolution: &convolution,
                    rate: &rate,
                    time_bias: &time_bias,
                    segments: &segments,
                    stop: &stop,
                    previous_bank: &previous_bank,
                    previous_tape: &previous_tape,
                    following_bank: &following_bank,
                    window: &mut window,
                    delta: &mut delta,
                    tape: &mut tape,
                    norm_epsilon: 1.0e-5,
                    grouped: false,
                })
                .map_err(|error| qualification_dynamic("qwen_recurrent_step", &label, error))?
                .value;
            kernels
                .chunk
                .call(qwen_recurrent_chunk::Args {
                    projection: &projection,
                    convolution: &convolution,
                    rate: &rate,
                    time_bias: &time_bias,
                    segments: &segments,
                    stop: &stop,
                    previous_bank: &previous_bank,
                    previous_tape: &previous_tape,
                    following_bank: &following_bank,
                    window: &mut window,
                    delta: &mut delta,
                    tape: &mut tape,
                    norm_epsilon: 1.0e-5,
                    grouped: false,
                })
                .map_err(|error| qualification_dynamic("qwen_recurrent_chunk", &label, error))?;
            let result = kernels
                .output
                .call(qwen_recurrent_output::Args {
                    hidden: &hidden_residual,
                    mixed: &mixed,
                    projection: &projection,
                    recurrent_norm: &recurrent_norm,
                    output_weight: &output,
                    epsilon: 1.0e-5,
                })
                .map_err(|error| qualification_dynamic("qwen_recurrent_output", &label, error))?
                .value;
            require_f32_values(&result, &hidden_values, "qwen_recurrent_output", &label)?;
        }

        let mut qualified = std::collections::HashSet::new();
        for ((slot, attested), block) in self
            .plan
            .target()
            .blocks()
            .iter()
            .zip(&self.programs.target.blocks)
            .zip(&self.geometry.blocks)
        {
            let (
                FeedForwardProgramSlot::Dense(binding),
                AttestedFeedForward::Dense(kernels),
                magnitude_model_contracts::FeedForwardGeometry::Dense { intermediate },
            ) = (slot.feed_forward(), &attested.feed_forward, &block.feedforward)
            else {
                continue;
            };
            if !qualified.insert((binding, *intermediate)) {
                continue;
            }
            let label = format!("{binding:?}");
            let f = *intermediate;
            let norm = semantic_zeros(device, binding.norm, &[hidden], "target_dense", &label)?;
            let gate = semantic_zeros(device, binding.gate, &[f, hidden], "target_dense", &label)?;
            let up = semantic_zeros(device, binding.up, &[f, hidden], "target_dense", &label)?;
            let down = semantic_zeros(device, binding.down, &[hidden, f], "target_dense", &label)?;
            let product = kernels
                .expand
                .call(qwen_dense_expand::Args {
                    residual: &hidden_residual,
                    norm: &norm,
                    gate_weight: &gate,
                    up_weight: &up,
                    out_rows: &out_rows,
                    eps: 1.0e-5,
                })
                .map_err(|error| qualification_dynamic("qwen_dense_expand", &label, error))?
                .value;
            let result = kernels
                .output
                .call(qwen_dense_output::Args {
                    residual: &hidden_residual,
                    product: &product,
                    down_weight: &down,
                    out_rows: &out_rows,
                })
                .map_err(|error| qualification_dynamic("qwen_dense_output", &label, error))?
                .value;
            require_f32_values(&result, &hidden_values, "qwen_dense_output", &label)?;
        }

        let mut qualified = std::collections::HashSet::new();
        for (slot, attested) in self
            .plan
            .target()
            .blocks()
            .iter()
            .zip(&self.programs.target.blocks)
        {
            let (FeedForwardProgramSlot::Routed(binding), AttestedFeedForward::Routed(kernels)) =
                (slot.feed_forward(), &attested.feed_forward)
            else {
                continue;
            };
            if qualified.insert(binding) {
                qualify_routed(device, binding, kernels)?;
            }
        }

        for binding in [self.plan.target().readout()] {
            let label = format!("{binding:?}");
            let norm = semantic_ones(device, binding.norm, &[hidden], "target_readout", &label)?;
            let weight = semantic_zeros(
                device,
                binding.weight,
                &[self.geometry.vocabulary, hidden],
                "target_readout",
                &label,
            )?;
            let features = self
                .programs
                .target
                .readout
                .features
                .call(readout_features_rows::Args {
                    hidden: &hidden_residual,
                    norm: &norm,
                    out_rows: &out_rows,
                    epsilon: 1.0e-5,
                })
                .map_err(|error| qualification_dynamic("readout_features_rows", &label, error))?
                .value;
            require_finite_nonzero(&features, "readout_features_rows", &label)?;
            let logits = self
                .programs
                .target
                .readout
                .head
                .call(readout_head_rows::Args {
                    hidden: &hidden_residual,
                    norm: &norm,
                    weight: &weight,
                    out_rows: &out_rows,
                    epsilon: 1.0e-5,
                })
                .map_err(|error| qualification_dynamic("readout_head_rows", &label, error))?
                .value;
            require_zero_result(&logits, "readout_head_rows", &label)?;
            let selected = semantic_i32(device, &[1], &[0], "readout_selected_rows", &label)?;
            let selected_result = self
                .programs
                .target
                .selected
                .call(readout_selected_rows::Args {
                    hidden: &hidden_residual,
                    norm: &norm,
                    weight: &weight,
                    out_rows: &out_rows,
                    selected: &selected,
                    epsilon: 1.0e-5,
                })
                .map_err(|error| qualification_dynamic("readout_selected_rows", &label, error))?
                .value;
            require_zero_result(&selected_result, "readout_selected_rows", &label)?;
        }

        for (binding, kernel) in self
            .plan
            .target()
            .features()
            .zip(self.programs.target.features.as_ref())
        {
            let label = format!("{binding:?}");
            let norm = semantic_ones(device, binding.norm, &[hidden], "target_features", &label)?;
            let result = kernel
                .call(readout_features_rows::Args {
                    hidden: &hidden_residual,
                    norm: &norm,
                    out_rows: &out_rows,
                    epsilon: 1.0e-5,
                })
                .map_err(|error| qualification_dynamic("readout_features_rows", &label, error))?
                .value;
            require_finite_nonzero(&result, "readout_features_rows", &label)?;
        }
        Ok(())
    }
}

/// The routed entries at the model's routed geometry (they fix it
/// statically) with zero weights, so both forms leave the residual unchanged.
/// The route runs over the full expert range; the projections read expert 0
/// of one-expert tensors through all-zero routes.
fn qualify_routed(
    device: &Device,
    binding: RoutedBinding,
    kernels: &RoutedKernels,
) -> Result<(), CatalogFailure> {
    let label = format!("{binding:?}");
    let (h, e, k, f, s) = (
        binding.hidden,
        binding.experts,
        binding.selected,
        binding.features,
        binding.shared,
    );
    let zeros = |element: Element, extents: &[u64]| {
        semantic_zeros(device, element, extents, "target_routed", &label)
    };
    let norm = zeros(binding.norm, &[h])?;
    let router = zeros(binding.router, &[e, h])?;
    let shared_router = zeros(Element::f32(), &[h])?;
    let expert_gate = zeros(binding.expert_gate, &[1, f, h])?;
    let expert_up = zeros(binding.expert_up, &[1, f, h])?;
    let expert_down = zeros(binding.expert_down, &[1, h, f])?;
    let shared_gate = zeros(binding.shared_gate, &[s, h])?;
    let shared_up = zeros(binding.shared_up, &[s, h])?;
    let shared_down = zeros(binding.shared_down, &[h, s])?;
    let grouped_rows = DECODE_ROWS * 2;
    for rows in [1, grouped_rows] {
        let values = vec![1.0_f32; (rows * h) as usize];
        let residual = semantic_f32(device, &[rows, h], &values, "target_routed", &label)?;
        let mut routes = zeros(Element::i32(), &[rows, k])?;
        let mut scores = zeros(Element::f32(), &[rows, k])?;
        let routed = kernels
            .route
            .call(qwen_routed_route::Args {
                residual: &residual,
                norm: &norm,
                router: &router,
                shared_router: &shared_router,
                routes: &mut routes,
                scores: &mut scores,
                eps: 1.0e-5,
                normalize: 1,
            })
            .map_err(|error| qualification_dynamic("qwen_routed_route", &label, error))?;
        let first = zeros(Element::i32(), &[rows, k])?;
        let result = if rows <= DECODE_ROWS {
            let expanded = kernels
                .expand
                .call(qwen_routed_expand::Args {
                    normalized: &routed.r0,
                    routes: &first,
                    expert_gate: &expert_gate,
                    expert_up: &expert_up,
                    shared_gate: &shared_gate,
                    shared_up: &shared_up,
                })
                .map_err(|error| qualification_dynamic("qwen_routed_expand", &label, error))?;
            kernels
                .output
                .call(qwen_routed_output::Args {
                    residual: &residual,
                    expert_product: &expanded.r0,
                    shared_product: &expanded.r1,
                    routes: &first,
                    scores: &scores,
                    coefficient: &routed.r1,
                    expert_down: &expert_down,
                    shared_down: &shared_down,
                })
                .map_err(|error| qualification_dynamic("qwen_routed_output", &label, error))?
                .value
        } else {
            let blocks = grouped_blocks(rows, e, k)
                .map_err(|error| qualification_dynamic("qwen_routed_group", &label, error))?;
            let mut counts = zeros(Element::i32(), &[e])?;
            let mut order = zeros(Element::i32(), &[blocks, TILE_ROWS])?;
            let mut inverse = zeros(Element::i32(), &[rows, k])?;
            let mut block_experts = zeros(Element::i32(), &[blocks])?;
            kernels
                .group
                .call(qwen_routed_group::Args {
                    routes: &first,
                    counts: &mut counts,
                    order: &mut order,
                    inverse: &mut inverse,
                    blocks: &mut block_experts,
                })
                .map_err(|error| qualification_dynamic("qwen_routed_group", &label, error))?;
            let experts = kernels
                .experts
                .call(qwen_routed_experts::Args {
                    normalized: &routed.r0,
                    order: &order,
                    blocks: &block_experts,
                    expert_gate: &expert_gate,
                    expert_up: &expert_up,
                    expert_down: &expert_down,
                })
                .map_err(|error| qualification_dynamic("qwen_routed_experts", &label, error))?
                .value;
            kernels
                .combine
                .call(qwen_routed_combine::Args {
                    residual: &residual,
                    expert_output: &experts,
                    inverse: &inverse,
                    scores: &scores,
                    normalized: &routed.r0,
                    coefficient: &routed.r1,
                    shared_gate: &shared_gate,
                    shared_up: &shared_up,
                    shared_down: &shared_down,
                })
                .map_err(|error| qualification_dynamic("qwen_routed_combine", &label, error))?
                .value
        };
        require_f32_values(&result, &values, "qwen_routed", &label)?;
    }
    Ok(())
}
