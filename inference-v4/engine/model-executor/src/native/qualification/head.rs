use super::super::*;
use super::*;

impl<'a> QualificationView<'a> {
    pub(super) fn qualify_head(&self, device: &Device) -> Result<(), CatalogError> {
        let (Some(head_plan), Some(head)) = (self.plan.head(), self.programs.head.as_ref()) else {
            return Ok(());
        };
        for (&binding, block) in head_plan.blocks().iter().zip(&head.blocks) {
            let label = format!("{binding:?}");
            let tokens = semantic_zeros(device, Element::i32(), &[1], "head", &label)?;
            let table = semantic_pattern(device, binding.embedding_table, &[1, 4], "head", &label)?;
            let conditioning = semantic_dense_values(
                device,
                binding.activation,
                &[1, 4],
                &[1.0, -2.0, 3.0, -4.0],
                "head",
                &label,
            )?;
            let embedding_norm =
                semantic_ones(device, binding.embedding_norm, &[4], "head", &label)?;
            let hidden_norm = semantic_ones(device, binding.hidden_norm, &[4], "head", &label)?;
            let combine = semantic_pattern(device, binding.combine, &[4, 8], "head", &label)?;
            let features = block
                .input
                .call(qwen_head_rows::Args {
                    tokens: &tokens,
                    table: &table,
                    conditioning: &conditioning,
                    embedding_norm: &embedding_norm,
                    hidden_norm: &hidden_norm,
                    combine: &combine,
                    epsilon: 1.0e-5,
                })
                .map_err(|error| qualification_dynamic("qwen_head_rows", &label, error))?
                .value;
            let input_norm = semantic_zeros(device, binding.input_norm, &[4], "head", &label)?;
            let query_gate_weight =
                semantic_zeros(device, binding.query_gate, &[8, 4], "head", &label)?;
            let key_weight = semantic_zeros(device, binding.key, &[4, 4], "head", &label)?;
            let value_weight = semantic_zeros(device, binding.value, &[4, 4], "head", &label)?;
            let query_norm = semantic_zeros(device, Element::f32(), &[4], "head", &label)?;
            let key_norm = semantic_zeros(device, Element::f32(), &[4], "head", &label)?;
            let attention_output =
                semantic_zeros(device, binding.attention_output, &[4, 4], "head", &label)?;
            let coordinates = semantic_zeros(device, Element::i32(), &[1, 4], "head", &label)?;
            let rotary_components = semantic_zeros(device, Element::i32(), &[1], "head", &label)?;
            let visible = semantic_zeros(device, Element::i32(), &[1, 1, 2], "head", &label)?;
            let fresh = semantic_i32(device, &[1, 2], &[0, 1], "head", &label)?;
            let destinations = semantic_zeros(device, Element::i32(), &[1], "head", &label)?;
            let mut history_key =
                semantic_zeros(device, binding.activation, &[1, 1, 4], "head", &label)?;
            let mut history_value =
                semantic_zeros(device, binding.activation, &[1, 1, 4], "head", &label)?;
            let attended = qualify_attention_stages(
                device,
                &block.attention,
                &features,
                &input_norm,
                &query_gate_weight,
                &key_weight,
                &value_weight,
                &query_norm,
                &key_norm,
                &attention_output,
                &coordinates,
                &rotary_components,
                &visible,
                &fresh,
                &destinations,
                &mut history_key,
                &mut history_value,
                binding.activation,
                &label,
            )?;
            let feedforward_norm =
                semantic_zeros(device, binding.feedforward_norm, &[4], "head", &label)?;
            let gate = semantic_zeros(device, binding.gate, &[4, 4], "head", &label)?;
            let up = semantic_zeros(device, binding.up, &[4, 4], "head", &label)?;
            let down = semantic_zeros(device, binding.down, &[4, 4], "head", &label)?;
            let product = block
                .dense
                .expand
                .call(qwen_dense_expand::Args {
                    residual: &attended,
                    norm: &feedforward_norm,
                    gate_weight: &gate,
                    up_weight: &up,
                    eps: 1.0e-5,
                })
                .map_err(|error| qualification_dynamic("qwen_dense_expand", &label, error))?
                .value;
            let dense = block
                .dense
                .output
                .call(qwen_dense_output::Args {
                    residual: &attended,
                    product: &product,
                    down_weight: &down,
                })
                .map_err(|error| qualification_dynamic("qwen_dense_output", &label, error))?
                .value;
            let out_rows = semantic_zeros(device, Element::i32(), &[1], "head", &label)?;
            let demanded_product = block
                .dense
                .expand_demanded
                .call(qwen_dense_expand_demanded::Args {
                    residual: &attended,
                    norm: &feedforward_norm,
                    gate_weight: &gate,
                    up_weight: &up,
                    out_rows: &out_rows,
                    eps: 1.0e-5,
                })
                .map_err(|error| {
                    qualification_dynamic("qwen_dense_expand_demanded", &label, error)
                })?
                .value;
            let demanded = block
                .dense
                .output_demanded
                .call(qwen_dense_output_demanded::Args {
                    residual: &attended,
                    product: &demanded_product,
                    down_weight: &down,
                    out_rows: &out_rows,
                })
                .map_err(|error| {
                    qualification_dynamic("qwen_dense_output_demanded", &label, error)
                })?
                .value;
            require_finite_nonzero_f32(&demanded, "qwen_dense_output_demanded", &label)?;
            let output_norm = semantic_ones(device, binding.output_norm, &[4], "head", &label)?;
            let projected_features = block
                .features
                .call(qwen_features_rows::Args {
                    hidden: &dense,
                    norm: &output_norm,
                    out_rows: &out_rows,
                    epsilon: 1.0e-5,
                })
                .map_err(|error| qualification_dynamic("qwen_features_rows", &label, error))?
                .value;
            let projection = semantic_pattern(device, binding.projection, &[1, 4], "head", &label)?;
            let logits = block
                .logits
                .call(head_logits_rows::Args {
                    features: &projected_features,
                    weight: &projection,
                })
                .map_err(|error| qualification_dynamic("head_logits_rows", &label, error))?
                .value;
            require_finite_nonzero_f32(&logits, "head_logits_rows", &label)?;
        }
        Ok(())
    }
}
