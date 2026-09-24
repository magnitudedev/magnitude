use super::super::*;
use super::*;

impl<'a> QualificationView<'a> {
    pub(super) fn qualify_vision(&self, device: &Device) -> Result<(), CatalogFailure> {
        let (Some(vision_plan), Some(vision)) = (self.plan.vision(), self.programs.vision.as_ref())
        else {
            return Ok(());
        };
        for binding in [vision_plan.patch()] {
            let label = format!("{binding:?}");
            let pixels = semantic_f32(device, &[1, 1, 2, 1, 1], &[2.0, 3.0], "vision", &label)?;
            let temporal_weight_0 = semantic_dense_values(
                device,
                binding.temporal_weight_0,
                &[1, 1, 1, 2],
                &[5.0, 7.0],
                "vision",
                &label,
            )?;
            let temporal_weight_1 = semantic_dense_values(
                device,
                binding.temporal_weight_1,
                &[1, 1, 1, 2],
                &[11.0, 13.0],
                "vision",
                &label,
            )?;
            let bias =
                semantic_dense_values(device, binding.bias, &[2], &[17.0, 19.0], "vision", &label)?;
            let table = semantic_dense_values(
                device,
                binding.position,
                &[2, 3],
                &[23.0, 29.0, 31.0, 37.0, 41.0, 43.0],
                "vision",
                &label,
            )?;
            let indices = semantic_i32(device, &[1, 4], &[0, 2, 1, 2], "vision", &label)?;
            let coefficients =
                semantic_f32(device, &[1, 4], &[1.0, 0.0, 0.0, 0.0], "vision", &label)?;
            let result = vision
                .stem
                .call(qwen_vision_stem::Args {
                    pixels: &pixels,
                    temporal_weight_0: &temporal_weight_0,
                    temporal_weight_1: &temporal_weight_1,
                    bias: &bias,
                    table: &table,
                    indices: &indices,
                    coefficients: &coefficients,
                })
                .map_err(|error| qualification_dynamic("qwen_vision_stem", &label, error))?
                .value;
            require_dense_values(&result, &[83.0, 109.0], "qwen_vision_stem", &label)?;
        }
        for (&binding, kernel) in vision_plan.blocks().iter().zip(&vision.blocks) {
            let label = format!("{binding:?}");
            let hidden = semantic_dense_values(
                device,
                binding.activation,
                &[1, 1, 4, 1],
                &[1.0, -2.0, 3.0, -4.0],
                "vision",
                &label,
            )?;
            let coordinates = semantic_zeros(device, Element::i32(), &[1, 2], "vision", &label)?;
            let norm1_weight =
                semantic_zeros(device, binding.input_norm_weight, &[4], "vision", &label)?;
            let norm1_bias =
                semantic_zeros(device, binding.input_norm_bias, &[4], "vision", &label)?;
            let qkv_weight =
                semantic_zeros(device, binding.qkv_weight, &[4, 12], "vision", &label)?;
            let qkv_bias = semantic_zeros(device, binding.qkv_bias, &[12], "vision", &label)?;
            let projection_weight =
                semantic_zeros(device, binding.attention_output, &[4, 4], "vision", &label)?;
            let projection_bias = semantic_zeros(
                device,
                binding.attention_output_bias,
                &[4],
                "vision",
                &label,
            )?;
            let norm2_weight = semantic_ones(
                device,
                binding.feedforward_norm_weight,
                &[4],
                "vision",
                &label,
            )?;
            let norm2_bias = semantic_zeros(
                device,
                binding.feedforward_norm_bias,
                &[4],
                "vision",
                &label,
            )?;
            let up_weight = semantic_dense_values(
                device,
                binding.up,
                &[4, 3],
                &[
                    0.1, 0.2, 0.3, -0.4, 0.5, 0.6, 0.7, -0.8, 0.9, 1.0, -1.1, 1.2,
                ],
                "vision",
                &label,
            )?;
            let up_bias = semantic_zeros(device, binding.up_bias, &[3], "vision", &label)?;
            let down_weight = semantic_dense_values(
                device,
                binding.down,
                &[3, 4],
                &[
                    0.2, -0.3, 0.4, -0.5, 0.6, 0.7, -0.8, 0.9, -1.0, 1.1, 1.2, -1.3,
                ],
                "vision",
                &label,
            )?;
            let down_bias = semantic_zeros(device, binding.down_bias, &[4], "vision", &label)?;
            let result = kernel
                .call(qwen_vision_block::Args {
                    hidden: &hidden,
                    coordinates: &coordinates,
                    norm1_weight: &norm1_weight,
                    norm1_bias: &norm1_bias,
                    qkv_weight: &qkv_weight,
                    qkv_bias: &qkv_bias,
                    projection_weight: &projection_weight,
                    projection_bias: &projection_bias,
                    norm2_weight: &norm2_weight,
                    norm2_bias: &norm2_bias,
                    up_weight: &up_weight,
                    up_bias: &up_bias,
                    down_weight: &down_weight,
                    down_bias: &down_bias,
                    epsilon: 1.0e-5,
                })
                .map_err(|error| qualification_dynamic("qwen_vision_block", &label, error))?
                .value;
            require_finite_nonzero(&result, "qwen_vision_block", &label)?;
            require_not_dense_values(
                &result,
                &[1.0, -2.0, 3.0, -4.0],
                "qwen_vision_block",
                &label,
            )?;
        }
        for binding in [vision_plan.merger()] {
            let label = format!("{binding:?}");
            let hidden = semantic_dense_values(
                device,
                binding.activation,
                &[1, 4],
                &[1.0, -2.0, 3.0, -4.0],
                "vision",
                &label,
            )?;
            let norm_weight =
                semantic_ones(device, binding.output_norm_weight, &[4], "vision", &label)?;
            let norm_bias =
                semantic_zeros(device, binding.output_norm_bias, &[4], "vision", &label)?;
            let up_weight = semantic_dense_values(
                device,
                binding.hidden,
                &[4, 4],
                &[
                    0.1, 0.2, 0.3, 0.4, -0.5, 0.6, 0.7, 0.8, 0.9, -1.0, 1.1, 1.2, 1.3, 1.4, -1.5,
                    1.6,
                ],
                "vision",
                &label,
            )?;
            let up_bias = semantic_zeros(device, binding.hidden_bias, &[4], "vision", &label)?;
            let down_weight = semantic_dense_values(
                device,
                binding.output,
                &[4, 2],
                &[0.2, -0.3, 0.4, 0.5, -0.6, 0.7, 0.8, -0.9],
                "vision",
                &label,
            )?;
            let down_bias = semantic_zeros(device, binding.output_bias, &[2], "vision", &label)?;
            let merged = vision
                .merger
                .call(qwen_vision_merger::Args {
                    hidden: &hidden,
                    norm_weight: &norm_weight,
                    norm_bias: &norm_bias,
                    up_weight: &up_weight,
                    up_bias: &up_bias,
                    down_weight: &down_weight,
                    down_bias: &down_bias,
                    epsilon: 1.0e-5,
                })
                .map_err(|error| qualification_dynamic("qwen_vision_merger", &label, error))?
                .value;
            let output = vision
                .output
                .call(qwen_vision_feature_output::Args { source: &merged })
                .map_err(|error| {
                    qualification_dynamic("qwen_vision_feature_output", &label, error)
                })?
                .value;
            require_finite_nonzero_f32(&output, "qwen_vision_feature_output", &label)?;
        }
        Ok(())
    }
}
