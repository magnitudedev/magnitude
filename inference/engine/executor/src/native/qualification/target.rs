use super::super::*;
use super::*;
use crate::operators::routed::fused_graph::{grouped_blocks, DECODE_ROWS, TILE_ROWS};

impl<'a> QualificationView<'a> {
    pub(super) fn qualify_target(&self, device: &Device) -> Result<(), CatalogFailure> {
        let out_rows = semantic_i32(device, &[1], &[0], "target", "out_rows")?;
        let hidden = self.geometry.hidden;
        let hidden_values = vec![1.0_f32; hidden as usize];
        let hidden_residual = semantic_f32(
            device,
            &[1, hidden],
            &hidden_values,
            "target",
            "hidden residual",
        )?;

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
                    scale: 1.0,
                    normalize: 0,
                    epsilon: 0.0,
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
            qualify_attention(device, kernel, binding, &format!("{binding:?}"))?;
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
            let rate = semantic_zeros(
                device,
                Element::f32(),
                &[value_heads],
                "target_recurrent",
                &label,
            )?;
            let time_bias = semantic_zeros(
                device,
                Element::f32(),
                &[value_heads],
                "target_recurrent",
                &label,
            )?;
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
            let mut banks = seismic::SlabTensor::new(
                device,
                2,
                2,
                vec![
                    seismic::SlabRegion {
                        element: binding.activation,
                        row_shape: vec![taps, channels],
                    },
                    seismic::SlabRegion {
                        element: Element::f32(),
                        row_shape: vec![value_heads, w, w],
                    },
                    seismic::SlabRegion {
                        element: Element::f32(),
                        row_shape: vec![1, (value_heads + key_heads) * w + value_heads],
                    },
                ],
            )
            .map_err(|error| qualification_dynamic("target_recurrent", &label, error))?;
            banks
                .add_slab()
                .map_err(|error| qualification_dynamic("target_recurrent", &label, error))?;
            let mut window = banks
                .logical_region(0)
                .map_err(|error| qualification_dynamic("target_recurrent", &label, error))?;
            let mut delta = banks
                .logical_region(1)
                .map_err(|error| qualification_dynamic("target_recurrent", &label, error))?;
            let mut tape = banks
                .logical_region(2)
                .map_err(|error| qualification_dynamic("target_recurrent", &label, error))?;
            let projection = kernels
                .project
                .call(gated_delta_project::Args {
                    hidden: &hidden_residual,
                    input_norm: &norm,
                    qkv_weight: &qkv,
                    gate_weight: &gate,
                    alpha_weight: &alpha,
                    beta_weight: &beta,
                    epsilon: 1.0e-5,
                })
                .map_err(|error| qualification_dynamic("gated_delta_project", &label, error))?
                .value;
            let mixed = kernels
                .step
                .call(gated_delta_step::Args {
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
                    slab_banks: 2,
                })
                .map_err(|error| qualification_dynamic("gated_delta_step", &label, error))?
                .value;
            kernels
                .chunk
                .call(gated_delta_chunk::Args {
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
                    slab_banks: 2,
                })
                .map_err(|error| qualification_dynamic("gated_delta_chunk", &label, error))?;
            let result = kernels
                .output
                .call(gated_delta_output::Args {
                    hidden: &hidden_residual,
                    mixed: &mixed,
                    projection: &projection,
                    recurrent_norm: &recurrent_norm,
                    output_weight: &output,
                    epsilon: 1.0e-5,
                })
                .map_err(|error| qualification_dynamic("gated_delta_output", &label, error))?
                .value;
            require_f32_values(&result, &hidden_values, "gated_delta_output", &label)?;
        }

        let mut qualified = std::collections::HashSet::new();
        for (slot, attested) in self
            .plan
            .target()
            .blocks()
            .iter()
            .zip(&self.programs.target.blocks)
        {
            let (MixerProgramSlot::StateSpace(binding), AttestedMixer::StateSpace(kernels)) =
                (slot.mixer(), &attested.mixer)
            else {
                continue;
            };
            if qualified.insert(binding) {
                qualify_state_space(
                    device,
                    kernels,
                    binding,
                    &hidden_residual,
                    &hidden_values,
                )?;
            }
        }

        let mut qualified = std::collections::HashSet::new();
        for (slot, attested) in self
            .plan
            .target()
            .blocks()
            .iter()
            .zip(&self.programs.target.blocks)
        {
            let (MixerProgramSlot::ShortConv(binding), AttestedMixer::ShortConv(kernels)) =
                (slot.mixer(), &attested.mixer)
            else {
                continue;
            };
            if qualified.insert(binding) {
                qualify_short_conv(device, kernels, binding, &hidden_residual, &hidden_values)?;
            }
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
                Some(FeedForwardProgramSlot::Dense(binding)),
                Some(AttestedFeedForward::Dense(kernels)),
                Ok(crate::operators::PairedBlock {
                    feed_forward:
                        Some(crate::operators::FeedForwardSublayer {
                            op: crate::operators::FeedForward::Dense(dense),
                            ..
                        }),
                    ..
                }),
            ) = (
                slot.feed_forward(),
                &attested.feed_forward,
                crate::operators::paired_block(block),
            )
            else {
                continue;
            };
            let intermediate = &dense.intermediate;
            if !qualified.insert((binding, *intermediate)) {
                continue;
            }
            let label = format!("{binding:?}");
            let f = *intermediate;
            let norm = semantic_zeros(device, binding.norm, &[hidden], "target_dense", &label)?;
            let gate = semantic_zeros(device, binding.gate, &[f, hidden], "target_dense", &label)?;
            let up = semantic_zeros(device, binding.up, &[f, hidden], "target_dense", &label)?;
            let down = semantic_zeros(device, binding.down, &[hidden, f], "target_dense", &label)?;
            let scale =
                |extent| semantic_ones(device, Element::f32(), &[extent], "target_dense", &label);
            let product = kernels
                .expand
                .call(dense_expand::Args {
                    residual: &hidden_residual,
                    norm: &norm,
                    gate_weight: &gate,
                    up_weight: &up,
                    out_rows: &out_rows,
                    eps: 1.0e-5,
                    activation: 0,
                    gate_scale: &scale(binding.scales.gate)?,
                    up_scale: &scale(binding.scales.up)?,
                })
                .map_err(|error| qualification_dynamic("dense_expand", &label, error))?
                .value;
            let result = match (&kernels.output, binding.tail) {
                (SublayerOutput::Residual(output), SublayerTail::Residual) => output
                    .call(dense_output::Args {
                        residual: &hidden_residual,
                        product: &product,
                        down_weight: &down,
                        out_rows: &out_rows,
                        down_scale: &scale(binding.scales.down)?,
                    })
                    .map_err(|error| qualification_dynamic("dense_output", &label, error))?
                    .value,
                (SublayerOutput::PostNorm(tail), SublayerTail::PostNorm { norm, .. }) => {
                    qualify_post_norm(
                        device,
                        tail,
                        &hidden_residual,
                        &product,
                        &down,
                        binding.scales.down,
                        norm,
                        &label,
                    )?
                }
                _ => {
                    return Err(qualification_dynamic(
                        "dense_output",
                        &label,
                        "dense output entries disagree with the binding's tail",
                    ))
                }
            };
            require_f32_values(&result, &hidden_values, "dense_output", &label)?;
        }

        self.qualify_per_layer(device, &hidden_residual, &hidden_values)?;

        let mut qualified = std::collections::HashSet::new();
        for (slot, attested) in self
            .plan
            .target()
            .blocks()
            .iter()
            .zip(&self.programs.target.blocks)
        {
            let (
                Some(FeedForwardProgramSlot::Routed(binding)),
                Some(AttestedFeedForward::Routed(kernels)),
            ) = (slot.feed_forward(), &attested.feed_forward)
            else {
                continue;
            };
            if qualified.insert(binding) {
                qualify_routed(device, binding, kernels)?;
            }
        }

        let mut qualified = std::collections::HashSet::new();
        for (slot, attested) in self
            .plan
            .target()
            .blocks()
            .iter()
            .zip(&self.programs.target.blocks)
        {
            let (
                Some(FeedForwardProgramSlot::GeneralRouted(binding)),
                Some(AttestedFeedForward::GeneralRouted(kernels)),
            ) = (slot.feed_forward(), &attested.feed_forward)
            else {
                continue;
            };
            if qualified.insert(binding) {
                qualify_general_routed(device, binding, kernels)?;
            }
        }

        let mut qualified = std::collections::HashSet::new();
        for (slot, attested) in self
            .plan
            .target()
            .blocks()
            .iter()
            .zip(&self.programs.target.blocks)
        {
            let (
                Some(FeedForwardProgramSlot::Parallel(binding)),
                Some(AttestedFeedForward::Parallel(kernels)),
            ) = (slot.feed_forward(), &attested.feed_forward)
            else {
                continue;
            };
            if qualified.insert(binding) {
                qualify_parallel(device, binding, kernels, &hidden_residual, &hidden_values)?;
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
                    softcap: 0.0,
                    weight_scale: &semantic_ones(
                        device,
                        Element::f32(),
                        &[binding.weight_scale],
                        "target_readout",
                        &label,
                    )?,
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
                    softcap: 0.0,
                })
                .map_err(|error| qualification_dynamic("readout_selected_rows", &label, error))?
                .value;
            require_zero_result(&selected_result, "readout_selected_rows", &label)?;
        }

        if let (Some(binding), Some(kernel)) = (
            self.plan.target().features(),
            self.programs.target.features.as_ref(),
        ) {
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
pub(super) fn qualify_routed(
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
            .call(routed_route::Args {
                residual: &residual,
                norm: &norm,
                router: &router,
                shared_router: &shared_router,
                routes: &mut routes,
                scores: &mut scores,
                eps: 1.0e-5,
                normalize: 1,
            })
            .map_err(|error| qualification_dynamic("routed_route", &label, error))?;
        let first = zeros(Element::i32(), &[rows, k])?;
        let result = if rows <= DECODE_ROWS {
            let expanded = kernels
                .expand
                .call(routed_expand::Args {
                    normalized: &routed.r0,
                    routes: &first,
                    expert_gate: &expert_gate,
                    expert_up: &expert_up,
                    shared_gate: &shared_gate,
                    shared_up: &shared_up,
                })
                .map_err(|error| qualification_dynamic("routed_expand", &label, error))?;
            kernels
                .output
                .call(routed_output::Args {
                    residual: &residual,
                    expert_product: &expanded.r0,
                    shared_product: &expanded.r1,
                    routes: &first,
                    scores: &scores,
                    coefficient: &routed.r1,
                    expert_down: &expert_down,
                    shared_down: &shared_down,
                })
                .map_err(|error| qualification_dynamic("routed_output", &label, error))?
                .value
        } else {
            let blocks = grouped_blocks(rows, e, k)
                .map_err(|error| qualification_dynamic("routed_group", &label, error))?;
            let mut counts = zeros(Element::i32(), &[e])?;
            let mut order = zeros(Element::i32(), &[blocks, TILE_ROWS])?;
            let mut inverse = zeros(Element::i32(), &[rows, k])?;
            let mut block_experts = zeros(Element::i32(), &[blocks])?;
            kernels
                .group
                .call(routed_group::Args {
                    routes: &first,
                    counts: &mut counts,
                    order: &mut order,
                    inverse: &mut inverse,
                    blocks: &mut block_experts,
                })
                .map_err(|error| qualification_dynamic("routed_group", &label, error))?;
            let experts = kernels
                .experts
                .call(routed_experts::Args {
                    normalized: &routed.r0,
                    order: &order,
                    blocks: &block_experts,
                    expert_gate: &expert_gate,
                    expert_up: &expert_up,
                    expert_down: &expert_down,
                    activation: 0,
                })
                .map_err(|error| qualification_dynamic("routed_experts", &label, error))?
                .value;
            kernels
                .combine
                .call(routed_combine::Args {
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
                .map_err(|error| qualification_dynamic("routed_combine", &label, error))?
                .value
        };
        require_f32_values(&result, &values, "qwen_routed", &label)?;
    }
    Ok(())
}

/// A general routed sublayer at the model's geometry with zero weights, at
/// one decode row and one grouped row class: every choice routes to one
/// zero expert (expert tensors of one expert), the shared expert and latent
/// projections are zero, so the residual comes back unchanged.
/// A parallel sublayer: its routed branch as a general routed sublayer, then
/// the dense branch over zero weights and `moe_tail` over zero branch
/// outputs, which leaves the residual unchanged.
fn qualify_parallel(
    device: &Device,
    binding: crate::ParallelBinding,
    kernels: &ParallelKernels,
    hidden_residual: &Tensor,
    hidden_values: &[f32],
) -> Result<(), CatalogFailure> {
    qualify_general_routed(device, binding.routed, &kernels.routed)?;
    let label = format!("{binding:?}");
    let absent_scale = semantic_zeros(device, Element::f32(), &[0], "parallel", "scale")?;
    let hidden = binding.hidden;
    let dense = binding.dense;
    let zeros = |element: Element, extents: &[u64], entry: &'static str| {
        semantic_zeros(device, element, extents, entry, &label)
    };
    let features = dense.features;
    let out_rows = semantic_i32(device, &[1], &[0], "dense_expand", &label)?;
    let product = kernels
        .expand
        .call(dense_expand::Args {
            residual: hidden_residual,
            norm: &zeros(dense.norm, &[hidden], "dense_expand")?,
            gate_weight: &zeros(dense.gate, &[features, hidden], "dense_expand")?,
            up_weight: &zeros(dense.up, &[features, hidden], "dense_expand")?,
            out_rows: &out_rows,
            eps: 1.0e-5,
            activation: 0,
            gate_scale: &absent_scale,
            up_scale: &absent_scale,
        })
        .map_err(|error| qualification_dynamic("dense_expand", &label, error))?
        .value;
    let projected = kernels
        .down
        .call(project_rows::Args {
            source: &product,
            weight: &zeros(dense.down, &[hidden, features], "project_rows")?,
            weight_scale: &absent_scale,
        })
        .map_err(|error| qualification_dynamic("project_rows", &label, error))?
        .value;
    let branch = zeros(Element::f32(), &[1, hidden], "moe_tail")?;
    let norm = zeros(binding.norm, &[hidden], "moe_tail")?;
    let result = kernels
        .tail
        .call(moe_tail::Args {
            residual: hidden_residual,
            dense: &projected,
            routed: &branch,
            dense_norm: &norm,
            routed_norm: &norm,
            norm: &norm,
            epsilon: 1.0e-6,
            scale: 1.0,
        })
        .map_err(|error| qualification_dynamic("moe_tail", &label, error))?
        .value;
    require_f32_values(&result, hidden_values, "moe_tail", &label)
}

fn qualify_general_routed(
    device: &Device,
    binding: crate::GeneralRoutedBinding,
    kernels: &GeneralRoutedKernels,
) -> Result<(), CatalogFailure> {
    let label = format!("{binding:?}");
    let entry = "target_general_routed";
    // Each dense entry's scale ports at their bound extents.
    let scale = |extent| semantic_ones(device, Element::f32(), &[extent], entry, &label);
    let scales = binding.scales;
    let shape = binding.shape;
    let (h, e, k, f, x) = (
        shape.hidden,
        shape.experts,
        shape.selected,
        shape.intermediate,
        shape.expert_hidden,
    );
    let zeros = |element: Element, extents: &[u64]| semantic_zeros(device, element, extents, entry, &label);
    macro_rules! failed {
        ($name:literal) => {
            |error| qualification_dynamic($name, &label, error)
        };
    }
    let norm = zeros(binding.norm, &[h])?;
    let router_norm = zeros(binding.router_norm, &[h])?;
    let router = zeros(binding.router, &[e, h])?;
    let bias = zeros(Element::f32(), &[e])?;
    let expert_scale = zeros(Element::f32(), &[e])?;
    // One qualified expert: the up scales follow the expert weights.
    let up_scale = zeros(Element::f32(), &[1])?;
    let expert_gate = binding
        .expert_gate
        .map(|gate| zeros(gate, &[1, f, x]))
        .transpose()?;
    let expert_up = zeros(binding.expert_up, &[1, f, x])?;
    let expert_down = zeros(binding.expert_down, &[1, x, f])?;
    for rows in [1, DECODE_ROWS * 2] {
        let values = vec![1.0_f32; (rows * h) as usize];
        let residual = semantic_f32(device, &[rows, h], &values, entry, &label)?;
        let out_rows = semantic_i32(
            device,
            &[rows],
            &(0..rows as i32).collect::<Vec<_>>(),
            entry,
            &label,
        )?;
        let base = match (&kernels.shared, shape.shared, binding.shared) {
            (Some((expansion, output)), Some((s, expansion_shape)), Some((gate, up, down))) => {
                let up = zeros(up, &[s, h])?;
                let product = match (expansion, gate) {
                    (DenseExpansionKernel::Gated(kernel), Some(gate)) => kernel
                        .call(dense_expand::Args {
                            residual: &residual,
                            norm: &norm,
                            gate_weight: &zeros(gate, &[s, h])?,
                            up_weight: &up,
                            out_rows: &out_rows,
                            eps: 1.0e-5,
                            activation: expansion_shape.activation,
                            gate_scale: &scale(scales.shared.gate)?,
                            up_scale: &scale(scales.shared.up)?,
                        })
                        .map_err(failed!("dense_expand"))?
                        .value,
                    (DenseExpansionKernel::Plain(kernel), None) => kernel
                        .call(dense_up::Args {
                            residual: &residual,
                            norm: &norm,
                            up_weight: &up,
                            out_rows: &out_rows,
                            eps: 1.0e-5,
                            activation: expansion_shape.activation,
                            up_scale: &scale(scales.shared.up)?,
                        })
                        .map_err(failed!("dense_up"))?
                        .value,
                    _ => {
                        return Err(qualification_dynamic(
                            "dense_output",
                            &label,
                            "shared expert entries disagree with the binding",
                        ))
                    }
                };
                output
                    .call(dense_output::Args {
                        residual: &residual,
                        product: &product,
                        down_weight: &zeros(down, &[h, s])?,
                        out_rows: &out_rows,
                        down_scale: &scale(scales.shared.down)?,
                    })
                    .map_err(failed!("dense_output"))?
                    .value
            }
            (None, None, None) => residual.clone(),
            _ => {
                return Err(qualification_dynamic(
                    "dense_output",
                    &label,
                    "shared expert entries disagree with the binding",
                ))
            }
        };
        let mut routes = zeros(Element::i32(), &[rows, k])?;
        let mut weights = zeros(Element::f32(), &[rows, k])?;
        let normalized = kernels
            .select
            .call(routed_select::Args {
                residual: &residual,
                norm: &norm,
                router_norm: &router_norm,
                router: &router,
                bias: &bias,
                expert_scale: &expert_scale,
                routes: &mut routes,
                weights: &mut weights,
                epsilon: 1.0e-5,
                score: shape.score,
                normalization: shape.normalization,
                normalization_epsilon: f32::from_bits(shape.normalization_epsilon),
                scale: f32::from_bits(shape.scale),
            })
            .map_err(failed!("routed_select"))?
            .value;
        let (input, sum_base) = match (&kernels.latent, binding.latent) {
            (Some((project, _)), Some((down, _))) => (
                project
                    .call(project_rows::Args {
                        source: &normalized,
                        weight: &zeros(down, &[x, h])?,
                        weight_scale: &scale(scales.latent.0)?,
                    })
                    .map_err(failed!("project_rows"))?
                    .value,
                zeros(Element::f32(), &[rows, x])?,
            ),
            _ => (normalized, base.clone()),
        };
        // Every choice routes to the one zero expert.
        let first = zeros(Element::i32(), &[rows, k])?;
        let routed = if rows <= DECODE_ROWS {
            let product = match &kernels.experts {
                ExpertKernels::Gated { decode, .. } => decode
                    .call(routed_gate_up::Args {
                        normalized: &input,
                        routes: &first,
                        expert_gate: expert_gate.as_ref().unwrap_or(&expert_up),
                        expert_up: &expert_up,
                        activation: shape.experts_expansion.activation,
                    })
                    .map_err(failed!("routed_gate_up"))?
                    .value,
                ExpertKernels::Plain { decode, .. } => decode
                    .call(routed_up::Args {
                        normalized: &input,
                        routes: &first,
                        expert_up: &expert_up,
                        up_scale: &up_scale,
                        activation: shape.experts_expansion.activation,
                    })
                    .map_err(failed!("routed_up"))?
                    .value,
            };
            kernels
                .down
                .call(routed_down::Args {
                    base: &sum_base,
                    product: &product,
                    routes: &first,
                    weights: &weights,
                    expert_down: &expert_down,
                })
                .map_err(failed!("routed_down"))?
                .value
        } else {
            let blocks = grouped_blocks(rows, e, k).map_err(failed!("routed_group"))?;
            let mut counts = zeros(Element::i32(), &[e])?;
            let mut order = zeros(Element::i32(), &[blocks, TILE_ROWS])?;
            let mut inverse = zeros(Element::i32(), &[rows, k])?;
            let mut block_experts = zeros(Element::i32(), &[blocks])?;
            kernels
                .group
                .call(routed_group::Args {
                    routes: &first,
                    counts: &mut counts,
                    order: &mut order,
                    inverse: &mut inverse,
                    blocks: &mut block_experts,
                })
                .map_err(failed!("routed_group"))?;
            let expert_output = match &kernels.experts {
                ExpertKernels::Gated { grouped, .. } => grouped
                    .call(routed_experts::Args {
                        normalized: &input,
                        order: &order,
                        blocks: &block_experts,
                        expert_gate: expert_gate.as_ref().unwrap_or(&expert_up),
                        expert_up: &expert_up,
                        expert_down: &expert_down,
                        activation: shape.experts_expansion.activation,
                    })
                    .map_err(failed!("routed_experts"))?
                    .value,
                ExpertKernels::Plain { grouped, .. } => grouped
                    .call(routed_experts_up::Args {
                        normalized: &input,
                        order: &order,
                        blocks: &block_experts,
                        expert_up: &expert_up,
                        expert_down: &expert_down,
                        up_scale: &up_scale,
                        activation: shape.experts_expansion.activation,
                    })
                    .map_err(failed!("routed_experts_up"))?
                    .value,
            };
            kernels
                .scatter
                .call(routed_scatter::Args {
                    base: &sum_base,
                    expert_output: &expert_output,
                    inverse: &inverse,
                    weights: &weights,
                })
                .map_err(failed!("routed_scatter"))?
                .value
        };
        let result = match (&kernels.latent, binding.latent) {
            (Some((_, up_kernel)), Some((_, up))) => up_kernel
                .call(dense_output::Args {
                    residual: &base,
                    product: &routed,
                    down_weight: &zeros(up, &[h, x])?,
                    out_rows: &out_rows,
                    down_scale: &scale(scales.latent.1)?,
                })
                .map_err(failed!("dense_output"))?
                .value,
            _ => routed,
        };
        require_f32_values(&result, &values, "routed_down", &label)?;
    }
    Ok(())
}

/// One row of a short-convolution layer at the model's geometry over a
/// two-bank window arena: bank 0 is the zero seed the row reads, bank 1 its
/// successor. Zero weights leave the residual unchanged.
fn qualify_short_conv(
    device: &Device,
    kernels: &ShortConvKernels,
    binding: crate::ShortConvBinding,
    hidden_residual: &Tensor,
    hidden_values: &[f32],
) -> Result<(), CatalogFailure> {
    let label = format!("{binding:?}");
    let entry = "target_short_conv";
    let absent_scale = semantic_zeros(device, Element::f32(), &[0], entry, &label)?;
    let shape = binding.shape;
    let zeros = |element, extents: &[u64]| semantic_zeros(device, element, extents, entry, &label);
    let norm = zeros(binding.norm, &[shape.hidden])?;
    let input_gate = zeros(binding.input_gate, &[shape.channels, shape.hidden])?;
    let output_gate = zeros(binding.output_gate, &[shape.channels, shape.hidden])?;
    let value = zeros(binding.value, &[shape.channels, shape.hidden])?;
    let convolution = zeros(Element::f32(), &[shape.channels, shape.convolution_width])?;
    let output = zeros(binding.output, &[shape.hidden, shape.channels])?;
    let segments = semantic_i32(device, &[2, 2], &[0, 1, 1, 1], entry, &label)?;
    let stop = semantic_i32(device, &[1], &[1], entry, &label)?;
    let previous_bank = semantic_i32(device, &[1], &[0], entry, &label)?;
    let previous_tape = semantic_i32(device, &[1], &[0], entry, &label)?;
    let following_bank = semantic_i32(device, &[1], &[1], entry, &label)?;
    // One tape row, the least a bank holds.
    let mut banks = seismic::SlabTensor::new(
        device,
        2,
        2,
        vec![seismic::SlabRegion {
            element: Element::f32(),
            row_shape: vec![shape.convolution_width, shape.channels],
        }],
    )
    .map_err(|error| qualification_dynamic(entry, &label, error))?;
    banks
        .add_slab()
        .map_err(|error| qualification_dynamic(entry, &label, error))?;
    let mut window = banks
        .logical_region(0)
        .map_err(|error| qualification_dynamic(entry, &label, error))?;
    let projection = kernels
        .project
        .call(short_conv_project::Args {
            residual: hidden_residual,
            norm: &norm,
            b_weight: &input_gate,
            c_weight: &output_gate,
            x_weight: &value,
            eps: 1.0e-5,
            b_scale: &absent_scale,
            c_scale: &absent_scale,
            x_scale: &absent_scale,
        })
        .map_err(|error| qualification_dynamic("short_conv_project", &label, error))?
        .value;
    let gated = kernels
        .rows
        .call(short_conv_rows::Args {
            projection: &projection,
            convolution: &convolution,
            segments: &segments,
            stop: &stop,
            previous_bank: &previous_bank,
            previous_tape: &previous_tape,
            following_bank: &following_bank,
            window: &mut window,
            slab_banks: 2,
        })
        .map_err(|error| qualification_dynamic("short_conv_rows", &label, error))?
        .value
        .reshape(&[1, 1, shape.channels])
        .map_err(|error| qualification_dynamic("short_conv_rows", &label, error))?;
    let result = kernels
        .output
        .call(attention_output::Args {
            hidden: hidden_residual,
            gated: &gated,
            output_weight: &output,
        })
        .map_err(|error| qualification_dynamic("attention_output", &label, error))?
        .value;
    require_f32_values(&result, hidden_values, "attention_output", &label)
}

/// One row of a state-space layer at the model's geometry (the entries fix
/// it statically) over a two-bank arena: bank 0 is the zero seed the advance
/// reads, bank 1 its successor. Zero weights leave the residual unchanged.
fn qualify_state_space(
    device: &Device,
    kernels: &StateSpaceKernels,
    binding: crate::StateSpaceBinding,
    hidden_residual: &Tensor,
    hidden_values: &[f32],
) -> Result<(), CatalogFailure> {
    let label = format!("{binding:?}");
    let entry = "target_state_space";
    let shape = binding.shape;
    let zeros = |element, extents: &[u64]| semantic_zeros(device, element, extents, entry, &label);
    let f32 = Element::f32();
    let norm = zeros(binding.norm, &[shape.hidden])?;
    let projection_weight = zeros(binding.projection, &[shape.projection_width(), shape.hidden])?;
    let empty = projection_weight
        .slice_leading(0, 0)
        .map_err(|error| qualification_dynamic(entry, &label, error))?;
    let convolution = zeros(f32, &[shape.channels(), shape.convolution_width])?;
    let convolution_bias = zeros(f32, &[shape.channels()])?;
    let rate = zeros(f32, &[shape.heads])?;
    let time_bias = zeros(f32, &[shape.heads])?;
    let skip = zeros(f32, &[shape.heads])?;
    let state_norm = zeros(f32, &[shape.norm_groups(), shape.norm_heads, shape.head_width])?;
    let output = zeros(binding.output, &[shape.hidden, shape.inner()])?;
    let segments = semantic_i32(device, &[2, 2], &[0, 1, 1, 1], entry, &label)?;
    let stop = semantic_i32(device, &[1], &[1], entry, &label)?;
    let previous_bank = semantic_i32(device, &[1], &[0], entry, &label)?;
    let previous_tape = semantic_i32(device, &[1], &[0], entry, &label)?;
    let following_bank = semantic_i32(device, &[1], &[1], entry, &label)?;
    // One tape row, the least a bank holds.
    let mut banks = seismic::SlabTensor::new(
        device,
        2,
        2,
        vec![
            seismic::SlabRegion {
                element: binding.activation,
                row_shape: vec![shape.convolution_width, shape.channels()],
            },
            seismic::SlabRegion {
                element: f32,
                row_shape: vec![shape.heads, shape.head_width, shape.state],
            },
            seismic::SlabRegion {
                element: f32,
                row_shape: vec![1, shape.tape_width()],
            },
        ],
    )
    .map_err(|error| qualification_dynamic(entry, &label, error))?;
    banks
        .add_slab()
        .map_err(|error| qualification_dynamic(entry, &label, error))?;
    let region = |index| {
        banks
            .logical_region(index)
            .map_err(|error| qualification_dynamic(entry, &label, error))
    };
    let (mut window, mut state, mut tape) = (region(0)?, region(1)?, region(2)?);
    let projection = kernels
        .project
        .call(attention_project::Args {
            hidden: hidden_residual,
            input_norm: &norm,
            query_weight: &projection_weight,
            gate_weight: &empty,
            key_weight: &empty,
            value_weight: &empty,
            epsilon: 1.0e-5,
        })
        .map_err(|error| qualification_dynamic("attention_project", &label, error))?
        .r0;
    macro_rules! advance {
        ($module:ident, $kernel:expr) => {
            $kernel
                .call($module::Args {
                    projection: &projection,
                    convolution: &convolution,
                    convolution_bias: &convolution_bias,
                    rate: &rate,
                    time_bias: &time_bias,
                    skip: &skip,
                    segments: &segments,
                    stop: &stop,
                    previous_bank: &previous_bank,
                    previous_tape: &previous_tape,
                    following_bank: &following_bank,
                    window: &mut window,
                    state: &mut state,
                    tape: &mut tape,
                    slab_banks: 2,
                })
                .map_err(|error| qualification_dynamic(stringify!($module), &label, error))?
                .value
        };
    }
    let mixed = advance!(state_space_step, kernels.step);
    advance!(state_space_chunk, kernels.chunk);
    let gated = kernels
        .gate
        .call(state_space_gate::Args {
            mixed: &mixed,
            projection: &projection,
            state_norm: &state_norm,
            epsilon: 1.0e-5,
        })
        .map_err(|error| qualification_dynamic("state_space_gate", &label, error))?
        .value;
    let result = kernels
        .output
        .call(attention_output::Args {
            hidden: hidden_residual,
            gated: &gated,
            output_weight: &output,
        })
        .map_err(|error| qualification_dynamic("attention_output", &label, error))?
        .value;
    require_f32_values(&result, hidden_values, "attention_output", &label)
}

impl QualificationView<'_> {
    /// The per-layer entry over zero weights and table rows writes zero
    /// per-layer rows; a per-layer input sublayer over zero weights leaves
    /// the residual unchanged.
    fn qualify_per_layer(
        &self,
        device: &Device,
        hidden_residual: &Tensor,
        hidden_values: &[f32],
    ) -> Result<(), CatalogFailure> {
        let hidden = self.geometry.hidden;
        let absent_scale = semantic_zeros(device, Element::f32(), &[0], "per_layer", "scale")?;
        if let (Some(binding), Some(kernels), Some(entry)) = (
            self.plan.target().per_layer(),
            &self.programs.target.per_layer,
            &self.geometry.entry.per_layer,
        ) {
            let label = format!("{binding:?}");
            let channels = binding.layers * binding.width;
            let embedded = semantic_f32(
                device,
                &[1, 1, hidden],
                hidden_values,
                "per_layer_entry",
                &label,
            )?;
            let rounded = kernels
                .round
                .call(import_dense::Args { source: &embedded })
                .map_err(|error| qualification_dynamic("import_dense", &label, error))?
                .value;
            let projection = semantic_zeros(
                device,
                binding.projection,
                &[channels, hidden],
                "per_layer_entry",
                &label,
            )?;
            let projected = kernels
                .project
                .call(project_rows::Args {
                    source: &rounded.reshape(&[1, hidden]).map_err(|error| {
                        qualification_dynamic("project_rows", &label, error)
                    })?,
                    weight: &projection,
                    weight_scale: &absent_scale,
                })
                .map_err(|error| qualification_dynamic("project_rows", &label, error))?
                .value;
            let source = semantic_zeros(
                device,
                binding.table_source,
                &[1, 1, channels],
                "per_layer_entry",
                &label,
            )?;
            let table = match &kernels.table {
                TableConversion::Dense(kernel) => kernel
                    .call(import_dense::Args { source: &source })
                    .map_err(|error| qualification_dynamic("import_dense", &label, error))?
                    .value,
                TableConversion::Repack(kernel) => kernel
                    .call(repack_weight::Args { source: &source })
                    .map_err(|error| qualification_dynamic("repack_weight", &label, error))?
                    .value,
            };
            let norm = semantic_zeros(
                device,
                binding.norm,
                &[binding.width],
                "per_layer_inputs",
                &label,
            )?;
            let combined = kernels
                .inputs
                .call(per_layer_inputs::Args {
                    gathered: &table.reshape(&[1, channels]).map_err(|error| {
                        qualification_dynamic("per_layer_inputs", &label, error)
                    })?,
                    projected: &projected,
                    norm: &norm,
                    epsilon: entry.projection_norm.epsilon as f32,
                    gathered_scale: entry.table_scale as f32,
                    projected_scale: entry.projection_scale as f32,
                    scale: entry.combine_scale as f32,
                })
                .map_err(|error| qualification_dynamic("per_layer_inputs", &label, error))?
                .value;
            require_f32_values(
                &combined,
                &vec![0.0; channels as usize],
                "per_layer_inputs",
                &label,
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
            let (Some(binding), Some(kernels)) = (slot.per_layer(), &attested.per_layer) else {
                continue;
            };
            if !qualified.insert(binding) {
                continue;
            }
            let label = format!("{binding:?}");
            let SublayerTail::PostNorm { norm, .. } = binding.tail else {
                return Err(qualification_dynamic(
                    "per_layer_gate",
                    &label,
                    "a per-layer input sublayer ends in a post-norm tail",
                ));
            };
            let (layers, width) = (binding.layers, binding.width);
            let gate =
                semantic_zeros(device, binding.gate, &[width, hidden], "per_layer_gate", &label)?;
            let inputs = semantic_zeros(
                device,
                Element::f32(),
                &[1, layers, width],
                "per_layer_gate",
                &label,
            )?;
            let gated = kernels
                .gate
                .call(per_layer_gate::Args {
                    hidden: hidden_residual,
                    gate_weight: &gate,
                    inputs: &inputs,
                    layer: 0,
                    activation: 1,
                    gate_scale: &absent_scale,
                })
                .map_err(|error| qualification_dynamic("per_layer_gate", &label, error))?
                .value;
            let projection = semantic_zeros(
                device,
                binding.projection,
                &[hidden, width],
                "per_layer_gate",
                &label,
            )?;
            let result = qualify_post_norm(
                device,
                &kernels.output,
                hidden_residual,
                &gated,
                &projection,
                0,
                norm,
                &label,
            )?;
            require_f32_values(&result, hidden_values, "per_layer_gate", &label)?;
        }
        Ok(())
    }
}
