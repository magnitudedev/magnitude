use super::super::*;
use super::*;

impl<'a> QualificationView<'a> {
    pub(super) fn qualify_target(&self, device: &Device) -> Result<(), CatalogError> {
        let residual_values = [1.0_f32, -2.0, 3.0, -4.0];
        let residual = semantic_f32(device, &[1, 4], &residual_values, "target", "residual")?;
        let out_rows = semantic_i32(device, &[1], &[0], "target", "out_rows")?;
        const WIDE: u64 = 4;
        let wide_values = vec![1.0_f32; WIDE as usize];
        let wide_residual =
            semantic_f32(device, &[1, WIDE], &wide_values, "target", "wide residual")?;

        for binding in [self.plan.target().embedding()] {
            let label = format!("{binding:?}");
            let table = semantic_pattern(
                device,
                binding.table,
                &[1, 4],
                "qwen_embedding_rows",
                &label,
            )?;
            let tokens = semantic_i32(device, &[1], &[0], "qwen_embedding_rows", &label)?;
            let result = self
                .programs
                .target
                .embedding
                .call(qwen_embedding_rows::Args {
                    table: &table,
                    tokens: &tokens,
                })
                .map_err(|error| qualification_dynamic("qwen_embedding_rows", &label, error))?;
            require_finite_nonzero_f32(&result.r1, "qwen_embedding_rows", &label)?;
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
            let label = format!("{binding:?}");
            let input_norm =
                semantic_zeros(device, binding.norm, &[4], "target_attention", &label)?;
            let query_gate = semantic_zeros(
                device,
                binding.query_gate,
                &[8, 4],
                "target_attention",
                &label,
            )?;
            let key = semantic_zeros(device, binding.key, &[4, 4], "target_attention", &label)?;
            let value = semantic_zeros(device, binding.value, &[4, 4], "target_attention", &label)?;
            let output =
                semantic_zeros(device, binding.output, &[4, 4], "target_attention", &label)?;
            let query_norm =
                semantic_zeros(device, Element::f32(), &[4], "target_attention", &label)?;
            let key_norm =
                semantic_zeros(device, Element::f32(), &[4], "target_attention", &label)?;
            let coordinates =
                semantic_zeros(device, Element::i32(), &[1, 4], "target_attention", &label)?;
            let rotary = semantic_zeros(device, Element::i32(), &[1], "target_attention", &label)?;
            let visible = semantic_zeros(
                device,
                Element::i32(),
                &[1, 1, 2],
                "target_attention",
                &label,
            )?;
            let fresh = semantic_i32(device, &[1, 2], &[0, 1], "target_attention", &label)?;
            let destinations =
                semantic_zeros(device, Element::i32(), &[1], "target_attention", &label)?;
            let mut history_key = semantic_zeros(
                device,
                binding.activation,
                &[1, 1, 4],
                "target_attention",
                &label,
            )?;
            let mut history_value = semantic_zeros(
                device,
                binding.activation,
                &[1, 1, 4],
                "target_attention",
                &label,
            )?;
            let result = qualify_attention_stages(
                device,
                kernel,
                &residual,
                &input_norm,
                &query_gate,
                &key,
                &value,
                &query_norm,
                &key_norm,
                &output,
                &coordinates,
                &rotary,
                &visible,
                &fresh,
                &destinations,
                &mut history_key,
                &mut history_value,
                binding.activation,
                &label,
            )?;
            require_f32_values(&result, &residual_values, "qwen_attention_output", &label)?;
        }

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
            let label = format!("{binding:?}");
            const H: u64 = 4;
            const W: u64 = 2;
            const QKV: u64 = 3 * W;
            let recurrent_values = vec![1.0_f32; H as usize];
            let recurrent_hidden = semantic_f32(
                device,
                &[1, H],
                &recurrent_values,
                "target_recurrent",
                &label,
            )?;
            let norm = semantic_zeros(device, binding.norm, &[H], "target_recurrent", &label)?;
            let qkv = semantic_zeros(device, binding.qkv, &[QKV, H], "target_recurrent", &label)?;
            let gate = semantic_zeros(device, binding.gate, &[W, H], "target_recurrent", &label)?;
            let alpha = semantic_zeros(device, binding.alpha, &[1, H], "target_recurrent", &label)?;
            let beta = semantic_zeros(device, binding.beta, &[1, H], "target_recurrent", &label)?;
            let convolution = semantic_zeros(
                device,
                Element::f32(),
                &[QKV, 2],
                "target_recurrent",
                &label,
            )?;
            let rate = semantic_zeros(device, Element::f32(), &[1], "target_recurrent", &label)?;
            let time_bias =
                semantic_zeros(device, Element::f32(), &[1], "target_recurrent", &label)?;
            let recurrent_norm = semantic_zeros(
                device,
                binding.recurrent_norm,
                &[W],
                "target_recurrent",
                &label,
            )?;
            let output =
                semantic_zeros(device, binding.output, &[H, W], "target_recurrent", &label)?;
            let segments =
                semantic_i32(device, &[2, 2], &[0, 1, 1, 1], "target_recurrent", &label)?;
            let window = semantic_zeros(
                device,
                binding.activation,
                &[1, 1, QKV],
                "target_recurrent",
                &label,
            )?;
            let delta = semantic_zeros(
                device,
                Element::f32(),
                &[1, 1, W, W],
                "target_recurrent",
                &label,
            )?;
            let normalized = kernels
                .normalize
                .call(qwen_recurrent_normalize::Args {
                    hidden: &recurrent_hidden,
                    input_norm: &norm,
                    epsilon: 1.0e-5,
                })
                .map_err(|error| qualification_dynamic("qwen_recurrent_normalize", &label, error))?
                .value;
            let projection = kernels
                .project
                .call(qwen_recurrent_project::Args {
                    normalized: &normalized,
                    qkv_weight: &qkv,
                    gate_weight: &gate,
                    alpha_weight: &alpha,
                    beta_weight: &beta,
                })
                .map_err(|error| qualification_dynamic("qwen_recurrent_project", &label, error))?
                .value;
            let prepared = kernels
                .prepare
                .call(qwen_recurrent_prepare::Args {
                    projection: &projection,
                    convolution: &convolution,
                    rate: &rate,
                    time_bias: &time_bias,
                    recurrent_norm: &recurrent_norm,
                    segments: &segments,
                    window: &window,
                    preparation_epsilon: 1.0e-5,
                })
                .map_err(|error| qualification_dynamic("qwen_recurrent_prepare", &label, error))?;
            let scanned = kernels
                .scan
                .call(qwen_recurrent_scan::Args {
                    prepared: &prepared.r1,
                    decay: &prepared.r2,
                    segments: &segments,
                    delta: &delta,
                    grouped: true,
                })
                .map_err(|error| qualification_dynamic("qwen_recurrent_scan", &label, error))?;
            let gated = kernels
                .mix
                .call(qwen_recurrent_mix::Args {
                    projection: &projection,
                    mixed: &scanned.r1,
                    recurrent_norm: &recurrent_norm,
                    epsilon: 1.0e-5,
                })
                .map_err(|error| qualification_dynamic("qwen_recurrent_mix", &label, error))?
                .value;
            let result = kernels
                .output
                .call(qwen_recurrent_output::Args {
                    hidden: &recurrent_hidden,
                    gated: &gated,
                    output_weight: &output,
                })
                .map_err(|error| qualification_dynamic("qwen_recurrent_output", &label, error))?
                .value;
            require_f32_values(&result, &recurrent_values, "qwen_recurrent_output", &label)?;
        }

        for (slot, attested) in self
            .plan
            .target()
            .blocks()
            .iter()
            .zip(&self.programs.target.blocks)
        {
            let (FeedForwardProgramSlot::Dense(binding), AttestedFeedForward::Dense(kernels)) =
                (slot.feed_forward(), &attested.feed_forward)
            else {
                continue;
            };
            let label = format!("{binding:?}");
            let norm = semantic_zeros(device, binding.norm, &[WIDE], "target_dense", &label)?;
            let gate = semantic_zeros(device, binding.gate, &[WIDE, WIDE], "target_dense", &label)?;
            let up = semantic_zeros(device, binding.up, &[WIDE, WIDE], "target_dense", &label)?;
            let down = semantic_zeros(device, binding.down, &[WIDE, WIDE], "target_dense", &label)?;
            let product = kernels
                .expand
                .call(qwen_dense_expand::Args {
                    residual: &wide_residual,
                    norm: &norm,
                    gate_weight: &gate,
                    up_weight: &up,
                    eps: 1.0e-5,
                })
                .map_err(|error| qualification_dynamic("qwen_dense_expand", &label, error))?
                .value;
            let result = kernels
                .output
                .call(qwen_dense_output::Args {
                    residual: &wide_residual,
                    product: &product,
                    down_weight: &down,
                })
                .map_err(|error| qualification_dynamic("qwen_dense_output", &label, error))?
                .value;
            require_f32_values(&result, &wide_values, "qwen_dense_output", &label)?;
            let demanded_product = kernels
                .expand_demanded
                .call(qwen_dense_expand_demanded::Args {
                    residual: &wide_residual,
                    norm: &norm,
                    gate_weight: &gate,
                    up_weight: &up,
                    out_rows: &out_rows,
                    eps: 1.0e-5,
                })
                .map_err(|error| {
                    qualification_dynamic("qwen_dense_expand_demanded", &label, error)
                })?
                .value;
            let demanded = kernels
                .output_demanded
                .call(qwen_dense_output_demanded::Args {
                    residual: &wide_residual,
                    product: &demanded_product,
                    down_weight: &down,
                    out_rows: &out_rows,
                })
                .map_err(|error| {
                    qualification_dynamic("qwen_dense_output_demanded", &label, error)
                })?
                .value;
            require_f32_values(
                &demanded,
                &wide_values,
                "qwen_dense_output_demanded",
                &label,
            )?;
        }

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
            let label = format!("{binding:?}");
            let norm = semantic_zeros(device, binding.norm, &[WIDE], "target_routed", &label)?;
            let router =
                semantic_zeros(device, binding.router, &[1, WIDE], "target_routed", &label)?;
            let shared_router =
                semantic_zeros(device, Element::f32(), &[WIDE], "target_routed", &label)?;
            let expert_gate = semantic_zeros(
                device,
                binding.expert_gate,
                &[1, WIDE, WIDE],
                "target_routed",
                &label,
            )?;
            let expert_up = semantic_zeros(
                device,
                binding.expert_up,
                &[1, WIDE, WIDE],
                "target_routed",
                &label,
            )?;
            let expert_down = semantic_zeros(
                device,
                binding.expert_down,
                &[1, WIDE, WIDE],
                "target_routed",
                &label,
            )?;
            let shared_gate = semantic_zeros(
                device,
                binding.shared_gate,
                &[WIDE, WIDE],
                "target_routed",
                &label,
            )?;
            let shared_up = semantic_zeros(
                device,
                binding.shared_up,
                &[WIDE, WIDE],
                "target_routed",
                &label,
            )?;
            let shared_down = semantic_zeros(
                device,
                binding.shared_down,
                &[WIDE, WIDE],
                "target_routed",
                &label,
            )?;
            let source_rows =
                semantic_zeros(device, Element::i32(), &[1], "target_routed", &label)?;
            let normalized = kernels
                .normalize
                .call(qwen_routed_normalize::Args {
                    residual: &wide_residual,
                    norm: &norm,
                    source_rows: &source_rows,
                    eps: 1.0e-5,
                })
                .map_err(|e| qualification_dynamic("qwen_routed_normalize", &label, e))?
                .value;
            let logits = kernels
                .logits
                .call(qwen_routed_logits::Args {
                    normalized: &normalized,
                    router_weight: &router,
                })
                .map_err(|e| qualification_dynamic("qwen_routed_logits", &label, e))?
                .value;
            let mut routes =
                semantic_zeros(device, Element::i32(), &[1, 1], "target_routed", &label)?;
            let mut scores =
                semantic_zeros(device, Element::f32(), &[1, 1], "target_routed", &label)?;
            kernels
                .select
                .call(qwen_routed_select::Args {
                    logits: &logits,
                    selected: 1,
                    routes: &mut routes,
                    scores: &mut scores,
                })
                .map_err(|e| qualification_dynamic("qwen_routed_select", &label, e))?;
            let expanded = kernels
                .expand
                .call(qwen_routed_expand::Args {
                    normalized: &normalized,
                    expert_gate: &expert_gate,
                    expert_up: &expert_up,
                    shared_gate: &shared_gate,
                    shared_up: &shared_up,
                    shared_control: &shared_router,
                    routes: &routes,
                })
                .map_err(|e| qualification_dynamic("qwen_routed_expand", &label, e))?;
            let result = kernels
                .output
                .call(qwen_routed_output::Args {
                    residual: &wide_residual,
                    source_rows: &source_rows,
                    expert_product: &expanded.r0,
                    shared_product: &expanded.r1,
                    shared_coefficient: &expanded.r2,
                    expert_down: &expert_down,
                    shared_down: &shared_down,
                    routes: &routes,
                    scores: &scores,
                })
                .map_err(|e| qualification_dynamic("qwen_routed_output", &label, e))?
                .value;
            require_f32_values(&result, &wide_values, "qwen_routed_output", &label)?;
        }

        for binding in [self.plan.target().readout()] {
            let label = format!("{binding:?}");
            let norm = semantic_ones(device, binding.norm, &[WIDE], "target_readout", &label)?;
            let weight =
                semantic_zeros(device, binding.weight, &[1, WIDE], "target_readout", &label)?;
            let features = self
                .programs
                .target
                .readout
                .features
                .call(qwen_features_rows::Args {
                    hidden: &wide_residual,
                    norm: &norm,
                    out_rows: &out_rows,
                    epsilon: 1.0e-5,
                })
                .map_err(|error| qualification_dynamic("qwen_features_rows", &label, error))?
                .value;
            require_finite_nonzero(&features, "qwen_features_rows", &label)?;
            let logits = self
                .programs
                .target
                .readout
                .logits
                .call(head_logits_rows::Args {
                    features: &features,
                    weight: &weight,
                })
                .map_err(|error| qualification_dynamic("head_logits_rows", &label, error))?
                .value;
            require_zero_result(&logits, "head_logits_rows", &label)?;
            let selected = semantic_i32(device, &[1], &[0], "qwen_selected_rows", &label)?;
            let selected_result = self
                .programs
                .target
                .selected
                .call(qwen_selected_rows::Args {
                    hidden: &wide_residual,
                    norm: &norm,
                    weight: &weight,
                    out_rows: &out_rows,
                    selected: &selected,
                    epsilon: 1.0e-5,
                })
                .map_err(|error| qualification_dynamic("qwen_selected_rows", &label, error))?;
            require_finite_nonzero(&selected_result.r0, "qwen_selected_rows", &label)?;
            require_zero_result(&selected_result.r1, "qwen_selected_rows", &label)?;
        }

        for (binding, kernel) in self
            .plan
            .target()
            .features()
            .zip(self.programs.target.features.as_ref())
        {
            let label = format!("{binding:?}");
            let norm = semantic_ones(device, binding.norm, &[WIDE], "target_features", &label)?;
            let result = kernel
                .call(qwen_features_rows::Args {
                    hidden: &wide_residual,
                    norm: &norm,
                    out_rows: &out_rows,
                    epsilon: 1.0e-5,
                })
                .map_err(|error| qualification_dynamic("qwen_features_rows", &label, error))?
                .value;
            require_finite_nonzero(&result, "qwen_features_rows", &label)?;
        }
        Ok(())
    }
}
