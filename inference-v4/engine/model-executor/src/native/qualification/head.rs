use super::super::*;
use super::*;

impl<'a> QualificationView<'a> {
    pub(super) fn qualify_head(&self, device: &Device) -> Result<(), CatalogFailure> {
        let (Some(head_plan), Some(head)) = (self.plan.head(), self.programs.head.as_ref()) else {
            return Ok(());
        };
        let (hidden, vocabulary) = (
            self.geometry.hidden,
            crate::native::draft_vocabulary(self.geometry.vocabulary),
        );
        for (index, (&binding, block)) in head_plan.blocks().iter().zip(&head.blocks).enumerate() {
            let label = format!("{binding:?}");
            let scope = magnitude_model_contracts::WeightScope::HeadBlock(
                u32::try_from(index).map_err(|error| qualification("head", "fixture", error))?,
            );
            let intermediate = self.weight_shape(
                scope,
                magnitude_model_contracts::WeightKind::DenseGate,
                "dense_expand",
            )?[0];
            // One (token, status) selection row.
            let tokens = semantic_zeros(device, Element::i32(), &[1, 2], "head", &label)?;
            let table =
                semantic_pattern(device, binding.embedding_table, &[1, hidden], "head", &label)?;
            let pattern = (0..hidden)
                .map(|column| [1.0_f32, -2.0, 3.0, -4.0][column as usize % 4])
                .collect::<Vec<_>>();
            let conditioning = semantic_dense_values(
                device,
                binding.activation,
                &[1, hidden],
                &pattern,
                "head",
                &label,
            )?;
            let embedding_norm =
                semantic_ones(device, binding.embedding_norm, &[hidden], "head", &label)?;
            let hidden_norm = semantic_ones(device, binding.hidden_norm, &[hidden], "head", &label)?;
            let combine =
                semantic_pattern(device, binding.combine, &[hidden, 2 * hidden], "head", &label)?;
            let features = block
                .input
                .call(draft_rows::Args {
                    tokens: &tokens,
                    table: &table,
                    conditioning: &conditioning,
                    embedding_norm: &embedding_norm,
                    hidden_norm: &hidden_norm,
                    combine: &combine,
                    epsilon: 1.0e-5,
                })
                .map_err(|error| qualification_dynamic("draft_rows", &label, error))?
                .value;
            qualify_attention(
                device,
                &block.attention,
                binding.attention_shape,
                &AttentionElements {
                    input_norm: binding.input_norm,
                    query_gate: binding.query_gate,
                    key: binding.key,
                    value: binding.value,
                    output: binding.attention_output,
                    activation: binding.activation,
                },
                &label,
            )?;
            // The attention block keeps its residual (zero weights), so the
            // dense stage continues from the head input.
            let attended = features;
            let feedforward_norm =
                semantic_zeros(device, binding.feedforward_norm, &[hidden], "head", &label)?;
            let gate =
                semantic_zeros(device, binding.gate, &[intermediate, hidden], "head", &label)?;
            let up = semantic_zeros(device, binding.up, &[intermediate, hidden], "head", &label)?;
            let down =
                semantic_zeros(device, binding.down, &[hidden, intermediate], "head", &label)?;
            let out_rows = semantic_zeros(device, Element::i32(), &[1], "head", &label)?;
            let product = block
                .dense
                .expand
                .call(dense_expand::Args {
                    residual: &attended,
                    norm: &feedforward_norm,
                    gate_weight: &gate,
                    up_weight: &up,
                    out_rows: &out_rows,
                    eps: 1.0e-5,
                })
                .map_err(|error| qualification_dynamic("dense_expand", &label, error))?
                .value;
            let dense = block
                .dense
                .output
                .call(dense_output::Args {
                    residual: &attended,
                    product: &product,
                    down_weight: &down,
                    out_rows: &out_rows,
                })
                .map_err(|error| qualification_dynamic("dense_output", &label, error))?
                .value;
            require_finite_nonzero_f32(&dense, "dense_output", &label)?;
            let output_norm = semantic_ones(device, binding.output_norm, &[hidden], "head", &label)?;
            let projected_features = block
                .features
                .call(readout_features_rows::Args {
                    hidden: &dense,
                    norm: &output_norm,
                    out_rows: &out_rows,
                    epsilon: 1.0e-5,
                })
                .map_err(|error| qualification_dynamic("readout_features_rows", &label, error))?
                .value;
            let projection = semantic_pattern(device, binding.projection, &[vocabulary, hidden], "head", &label)?;
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
