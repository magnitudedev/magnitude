use super::super::*;
use super::*;
use magnitude_model_contracts::{WeightKind, WeightScope};

/// Small exact values (integers of magnitude at most `span / 2`), exact in
/// every dense element.
fn pattern(count: usize, span: usize, shift: usize) -> Vec<f32> {
    (0..count)
        .map(|index| ((index + shift) % span) as f32 - (span / 2) as f32)
        .collect()
}

impl<'a> QualificationView<'a> {
    /// The vision entries at the projector's geometry, on fixtures whose
    /// matrices are zero, so every result is an exact sum of biases, table
    /// rows and residual inputs: the plumbing of every launch (norms,
    /// projections, rotation, attention, residuals, activations) must hold.
    /// The arithmetic itself is covered by the kernel tests.
    pub(super) fn qualify_vision(&self, device: &Device) -> Result<(), CatalogFailure> {
        let (Some(vision_plan), Some(vision), Some(geometry)) =
            (self.plan.vision(), self.programs.vision.as_ref(), self.vision)
        else {
            return Ok(());
        };
        let as_usize = |value: u64| usize::try_from(value).expect("vision geometry fits usize");
        let merge = geometry.merge * geometry.merge;
        let hidden = geometry.hidden;
        let (m, h) = (as_usize(merge), as_usize(hidden));

        let binding = vision_plan.patch();
        let label = format!("{binding:?}");
        let entry = "qwen_vision_stem";
        let table_shape = self.weight_shape(WeightScope::Vision, WeightKind::PositionEmbedding, entry)?;
        let (channels, patch) = (geometry.channels, geometry.patch);
        let pixels = semantic_zeros(device, Element::f32(), &[merge, channels, 2, patch, patch], entry, &label)?;
        let weight_0 = semantic_zeros(device, binding.temporal_weight_0, &[hidden, channels, patch, patch], entry, &label)?;
        let weight_1 = semantic_zeros(device, binding.temporal_weight_1, &[hidden, channels, patch, patch], entry, &label)?;
        let bias_values = pattern(h, 7, 0);
        let bias = semantic_dense_values(device, binding.bias, &[hidden], &bias_values, entry, &label)?;
        let mut table_values = vec![0.0; as_usize(table_shape[0]) * h];
        table_values[..h].copy_from_slice(&pattern(h, 5, 1));
        let table = semantic_dense_values(device, binding.position, table_shape, &table_values, entry, &label)?;
        let indices = semantic_i32(device, &[merge, 4], &[0, 1, 2, 3].repeat(m), entry, &label)?;
        let coefficients = semantic_f32(device, &[merge, 4], &[1.0, 0.0, 0.0, 0.0].repeat(m), entry, &label)?;
        let stem = vision
            .stem
            .call(qwen_vision_stem::Args {
                pixels: &pixels,
                temporal_weight_0: &weight_0,
                temporal_weight_1: &weight_1,
                bias: &bias,
                table: &table,
                indices: &indices,
                coefficients: &coefficients,
            })
            .map_err(|error| qualification_dynamic(entry, &label, error))?
            .value;
        let expected = bias_values
            .iter()
            .zip(&table_values[..h])
            .map(|(bias, table)| bias + table)
            .collect::<Vec<_>>()
            .repeat(m);
        require_f32_values(&stem, &expected, entry, &label)?;

        let entry = "qwen_vision_block";
        let (heads, quarter) = (geometry.heads, hidden / geometry.heads / 4);
        for (&binding, kernel) in vision_plan.blocks().iter().zip(&vision.blocks) {
            let label = format!("{binding:?}");
            let scope = WeightScope::VisionBlock(0);
            let intermediate = self.weight_shape(scope, WeightKind::FeedForwardUpBias, entry)?[0];
            let input = pattern(m * h, 9, 0);
            let hidden_rows = semantic_f32(device, &[merge, heads, 4, quarter], &input, entry, &label)?;
            let coordinates = semantic_zeros(device, Element::i32(), &[merge, 2], entry, &label)?;
            let ones = |element| semantic_ones(device, element, &[hidden], entry, &label);
            let zeros = |element, extents: &[u64]| semantic_zeros(device, element, extents, entry, &label);
            let projection_values = pattern(h, 3, 0);
            let down_values = pattern(h, 5, 2);
            let result = kernel
                .call(qwen_vision_block::Args {
                    hidden: &hidden_rows,
                    coordinates: &coordinates,
                    norm1_weight: &ones(binding.input_norm_weight)?,
                    norm1_bias: &zeros(binding.input_norm_bias, &[hidden])?,
                    qkv_weight: &zeros(binding.qkv_weight, &[3 * hidden, hidden])?,
                    qkv_bias: &zeros(binding.qkv_bias, &[3 * hidden])?,
                    projection_weight: &zeros(binding.attention_output, &[hidden, hidden])?,
                    projection_bias: &semantic_dense_values(device, binding.attention_output_bias, &[hidden], &projection_values, entry, &label)?,
                    norm2_weight: &ones(binding.feedforward_norm_weight)?,
                    norm2_bias: &zeros(binding.feedforward_norm_bias, &[hidden])?,
                    up_weight: &zeros(binding.up, &[intermediate, hidden])?,
                    up_bias: &zeros(binding.up_bias, &[intermediate])?,
                    down_weight: &zeros(binding.down, &[hidden, intermediate])?,
                    down_bias: &semantic_dense_values(device, binding.down_bias, &[hidden], &down_values, entry, &label)?,
                    epsilon: geometry.epsilon as f32,
                })
                .map_err(|error| qualification_dynamic(entry, &label, error))?
                .value;
            let expected = input
                .chunks_exact(h)
                .flat_map(|row| {
                    row.iter()
                        .zip(&projection_values)
                        .zip(&down_values)
                        .map(|((input, projection), down)| input + projection + down)
                })
                .collect::<Vec<_>>();
            require_f32_values(&result, &expected, entry, &label)?;
        }

        let binding = vision_plan.merger();
        let label = format!("{binding:?}");
        let entry = "qwen_vision_merger";
        let output = self.weight_shape(WeightScope::Vision, WeightKind::MergerOutputBias, entry)?[0];
        let width = merge * hidden;
        let input = pattern(m * h, 9, 4);
        let rows = semantic_f32(device, &[merge, hidden], &input, entry, &label)?;
        let down_values = pattern(as_usize(output), 5, 3);
        let zeros = |element, extents: &[u64]| semantic_zeros(device, element, extents, entry, &label);
        let merged = vision
            .merger
            .call(qwen_vision_merger::Args {
                hidden: &rows,
                norm_weight: &semantic_ones(device, binding.output_norm_weight, &[hidden], entry, &label)?,
                norm_bias: &zeros(binding.output_norm_bias, &[hidden])?,
                up_weight: &zeros(binding.hidden, &[width, width])?,
                up_bias: &zeros(binding.hidden_bias, &[width])?,
                down_weight: &zeros(binding.output, &[output, width])?,
                down_bias: &semantic_dense_values(device, binding.output_bias, &[output], &down_values, entry, &label)?,
                epsilon: geometry.epsilon as f32,
            })
            .map_err(|error| qualification_dynamic(entry, &label, error))?
            .value;
        require_f32_values(&merged, &down_values, entry, &label)
    }
}
